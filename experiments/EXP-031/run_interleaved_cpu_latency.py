"""
EXP-031 Amendment A1 Section 7: Interleaved Round-Robin CPU Latency Benchmark
File: experiments/EXP-031/run_interleaved_cpu_latency.py

Benchmarks p1_100k, W100 (retrained), full_v5, P190, and Official SLAMSS-IFS at L=1200,
batch=1 in both FP32 and Dynamic INT8 modes at 1 and 4 threads.
Must be run AFTER all W100 training finishes on an otherwise idle machine.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import resource
import sys
import time
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.ao.quantization as quantization
import torch.nn as nn

EXP031_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP031_DIR, "..", ".."))
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
EXP021_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-021")
for _p in (REPO_ROOT, EXP031_DIR, EXP030_DIR, EXP029_DIR, EXP021_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp029_models import build_exp029_model  # noqa: E402
from exp030_models import build_exp030_model  # noqa: E402
from exp031_models import build_exp031_model  # noqa: E402
from official_slamss_ifs_adapter import OfficialSLAMSSIFSWrapper  # noqa: E402


MODEL_ORDER = ["p1_100k", "W100", "full_v5", "P190", "Official_SLAMSS_IFS"]


def get_cpu_info() -> Dict[str, Any]:
    cpu_model = "unknown"
    core_count = os.cpu_count() or 0
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    governor = "unknown"
    gov_path = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    if os.path.exists(gov_path):
        with open(gov_path, "r", encoding="utf-8") as f:
            governor = f.read().strip()
    return {
        "cpu_model": cpu_model,
        "logical_cores": core_count,
        "governor": governor,
        "torch_version": torch.__version__,
        "os_version": platform.platform(),
        "python_version": platform.python_version(),
    }


def compute_mmacs_unified(model: nn.Module, forward_fn: Callable[[nn.Module], Any]) -> float:
    macs = 0.0
    hooks = []

    def conv_hook(mod: nn.Conv1d, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal macs
        b, c_out, l_out = out.shape
        k = mod.kernel_size[0]
        macs += float(b * c_out * (mod.in_channels // mod.groups) * k * l_out)

    def linear_hook(mod: nn.Linear, inp: Tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        nonlocal macs
        elems = int(out.numel() // mod.out_features)
        macs += float(elems * mod.in_features * mod.out_features)

    def rnn_hook(mod: nn.Module, inp: Tuple[torch.Tensor, ...], out: Any, gates: int) -> None:
        nonlocal macs
        x_in = inp[0]
        if isinstance(x_in, nn.utils.rnn.PackedSequence):
            total_steps = int(x_in.data.shape[0])
        else:
            total_steps = int(x_in.shape[0] * x_in.shape[1]) if x_in.dim() == 3 else int(x_in.shape[0])
        num_dir = 2 if mod.bidirectional else 1
        h = mod.hidden_size
        for layer in range(mod.num_layers):
            in_sz = mod.input_size if layer == 0 else h * num_dir
            macs += float(num_dir * gates * h * (in_sz + h) * total_steps)

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.GRU):
            hooks.append(m.register_forward_hook(lambda mod, inp, out: rnn_hook(mod, inp, out, 3)))
        elif isinstance(m, nn.LSTM):
            hooks.append(m.register_forward_hook(lambda mod, inp, out: rnn_hook(mod, inp, out, 4)))

    try:
        model_cpu = model.cpu().eval()
        with torch.no_grad():
            forward_fn(model_cpu)
    finally:
        for h in hooks:
            h.remove()

    return float(macs / 1e6)


def build_models_and_inputs(seq_len: int = 1200) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, torch.Tensor]]:
    torch.manual_seed(42)
    x_ihr = torch.randn(1, seq_len, 30)
    x_acc = torch.randn(1, seq_len, 30)
    x_meta = torch.randn(1, seq_len, 5)
    seq_lens = torch.tensor([seq_len], dtype=torch.long)
    x_sig = torch.randn(1, seq_len * 30, 2)

    inputs = {
        "x_ihr": x_ihr,
        "x_acc": x_acc,
        "x_meta": x_meta,
        "seq_lens": seq_lens,
        "x_sig": x_sig,
    }

    models: Dict[str, Dict[str, Any]] = {}

    m_p1 = build_exp030_model("p1_100k").cpu().eval()
    m_w100 = build_exp031_model("A", loss_type="rccloss").cpu().eval()
    w100_ckpt = os.path.join(EXP031_DIR, "runs", "W100", "checkpoints", "W100_L1200_fold1_seed42.pt")
    if os.path.exists(w100_ckpt):
        m_w100.load_state_dict(torch.load(w100_ckpt, map_location="cpu", weights_only=True))
    m_v5 = build_exp029_model("full_v5").cpu().eval()
    m_p190 = build_exp031_model("P190", loss_type="rccloss").cpu().eval()
    p190_ckpt = os.path.join(EXP031_DIR, "checkpoints", "P190", "P190_L1200_fold1_seed42.pt")
    if os.path.exists(p190_ckpt):
        m_p190.load_state_dict(torch.load(p190_ckpt, map_location="cpu", weights_only=True))
    m_slamss = OfficialSLAMSSIFSWrapper(device=torch.device("cpu")).cpu().eval()

    raw_specs = [
        ("p1_100k", m_p1, lambda m: m(x_ihr=x_ihr, x_accel=x_acc, x_meta=x_meta, seq_lens=seq_lens)),
        ("W100", m_w100, lambda m: m(x_ihr=x_ihr, x_accel=x_acc, x_meta=x_meta, seq_lens=seq_lens)),
        ("full_v5", m_v5, lambda m: m(x_ihr=x_ihr, x_accel=x_acc, x_meta=x_meta, seq_lens=seq_lens)),
        ("P190", m_p190, lambda m: m(x_ihr=x_ihr, x_accel=x_acc, x_meta=x_meta, seq_lens=seq_lens)),
        ("Official_SLAMSS_IFS", m_slamss, lambda m: m(x_sig, x_meta)),
    ]

    for name, fp32_m, fwd in raw_specs:
        mmacs = compute_mmacs_unified(fp32_m, fwd)
        int8_m = quantization.quantize_dynamic(
            copy.deepcopy(fp32_m), {nn.Linear, nn.GRU, nn.LSTM}, dtype=torch.qint8
        )
        int8_m.eval()
        models[name] = {
            "n_params": int(sum(p.numel() for p in fp32_m.parameters())),
            "mmacs": mmacs,
            "fp32": fp32_m,
            "int8": int8_m,
            "fwd": fwd,
        }

    return models, inputs


def summarize_times(times_ms: List[float]) -> Dict[str, float]:
    arr = np.asarray(times_ms, dtype=np.float64)
    med = float(np.median(arr))
    q25, q75 = np.percentile(arr, [25, 75])
    iqr = float(q75 - q25)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    cv_pct = float((std / mean) * 100.0) if mean > 0 else 0.0
    return {
        "median_ms": med,
        "q25_ms": float(q25),
        "q75_ms": float(q75),
        "iqr_ms": iqr,
        "mean_ms": mean,
        "std_ms": std,
        "cv_pct": cv_pct,
        "n_passes": int(len(arr)),
    }


def run_single_thread_setting(
    threads: int,
    warmup: int = 30,
    passes: int = 200,
    max_cv_pct: float = 5.0,
) -> Dict[str, Any]:
    torch.set_num_threads(threads)
    models, _ = build_models_and_inputs(seq_len=1200)

    slots: List[Tuple[str, str]] = []
    for name in MODEL_ORDER:
        slots.append((name, "fp32"))
        slots.append((name, "int8"))

    # Warm-up interleaved
    with torch.no_grad():
        for _ in range(warmup):
            for name, mode in slots:
                models[name]["fwd"](models[name][mode])

    # Timed round-robin
    timings: Dict[Tuple[str, str], List[float]] = {slot: [] for slot in slots}
    with torch.no_grad():
        for _ in range(passes):
            for name, mode in slots:
                m = models[name][mode]
                fwd = models[name]["fwd"]
                t0 = time.perf_counter()
                fwd(m)
                timings[(name, mode)].append((time.perf_counter() - t0) * 1000.0)

    results: Dict[str, Dict[str, Any]] = {name: {} for name in MODEL_ORDER}
    for name, mode in slots:
        stats_dict = summarize_times(timings[(name, mode)])
        repeats = 0
        while stats_dict["cv_pct"] > max_cv_pct and repeats < 2:
            repeats += 1
            retry_times = []
            with torch.no_grad():
                for _ in range(warmup):
                    models[name]["fwd"](models[name][mode])
                for _ in range(passes):
                    t0 = time.perf_counter()
                    models[name]["fwd"](models[name][mode])
                    retry_times.append((time.perf_counter() - t0) * 1000.0)
            stats_dict = summarize_times(retry_times)
        stats_dict["repeats_for_cv"] = repeats
        results[name][mode] = stats_dict
        results[name]["n_params"] = models[name]["n_params"]
        results[name]["mmacs"] = models[name]["mmacs"]

    peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "threads": threads,
        "peak_rss_mib": float(peak_rss_kb / 1024.0),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, required=True, choices=[1, 4])
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--passes", type=int, default=200)
    parser.add_argument("--out-json", type=str, required=True)
    args = parser.parse_args()

    payload = {
        "system": get_cpu_info(),
        **run_single_thread_setting(args.threads, warmup=args.warmup, passes=args.passes),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[DONE] Saved {args.threads}T latency benchmark to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
