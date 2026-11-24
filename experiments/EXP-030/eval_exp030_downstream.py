"""
EXP-030 Downstream Evaluation Suite (Restricted to CNN-BiGRU-Star)
File: experiments/EXP-030/eval_exp030_downstream.py

Reads star_tag from experiments/EXP-030/elbow_selection.json and runs:
  Pillar 1: BIDSleep N=47 Holm-Bonferroni paired stats vs. Model A-164k & SLAMSS-IFS
  Pillar 2: W0-B Dynamic INT8 & MMAC profiling (curve across p1..p6 + OOF parity on star_tag)
  Pillar 3: Strictly zero-shot SleepAccel (n=25 QC-clean & N=31 all, em_off only)
            with like-for-like paired F1 and Bland-Altman tests
  Pillar 4: EXP-020 FT-2 & nested per-fold GATE_tau personalization (N=46)
            via lightweight ScaledConventionalCNNBiGRU state-dict cloning
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.ao.quantization as quantization

EXP030_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP030_DIR, "..", ".."))
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
EXP028_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-028")
EXP027_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-027")
EXP021_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-021")
EXP019_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-019")
STAGE2_DIR = os.path.join(REPO_ROOT, "experiments", "STAGE2_FINAL_MODEL")

for _p in (REPO_ROOT, EXP030_DIR, EXP029_DIR, EXP028_DIR, EXP027_DIR, EXP021_DIR, EXP019_DIR, STAGE2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_tsk08_paired_benchmark import participant_bootstrap_ci
from eval_exp029_downstream import holm_bonferroni
from evaluator import ClinicalHypnogramEvaluator, SleepEvaluator
from exp019_adaptation_core import prepare_calibration_batch
from exp029_calibration import LogitStitchingSlidingWindowPredictor
from exp030_models import EXP030_GRID, EXP030_GRID_ORDER, build_exp030_model
from run_exp019_stage_d import evaluate_records_macro_f1
from run_exp027_sleepaccel_inference import extract_8_metrics, load_preprocessed_sleepaccel
from run_exp028_sleepaccel_bland_altman import SUMMARY_METRICS, compute_bland_altman_stats
from run_exp029_cv import (
    AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
    CACHE_DIR_DEFAULT,
    MANIFESTS_DIR_DEFAULT,
    compute_epoch_metrics,
    compute_stage_loss_fp32,
    load_all_records_by_subject,
    set_deterministic_seed,
    sha256_file,
    verify_manifests_and_splits,
)

FOLDS = [1, 2, 3, 4, 5]
SEEDS = [42, 123, 2024]
TAU_GRID = [0.01, 0.02, 0.03, 0.05]


def load_elbow_selection(exp030_dir: str = EXP030_DIR) -> Dict[str, Any]:
    path = os.path.join(exp030_dir, "elbow_selection.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing elbow_selection.json at {path}. Run select_exp030_elbow.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_star_checkpoint_map(exp030_dir: str, star_tag: str) -> Dict[Tuple[int, int], str]:
    """
    Returns (fold, seed) -> checkpoint_path for star_tag, falling back to
    STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json when star_tag == 'p2_164k'.
    """
    ckpt_map: Dict[Tuple[int, int], str] = {}
    if star_tag == "p2_164k":
        manifest_path = os.path.join(STAGE2_DIR, "frozen_checkpoint_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Frozen manifest missing: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data["checkpoints"]:
            fold, seed = int(entry["fold"]), int(entry["seed"])
            abs_p = os.path.join(REPO_ROOT, entry["checkpoint_relpath"])
            if not os.path.exists(abs_p):
                raise FileNotFoundError(f"Frozen checkpoint missing: {abs_p}")
            assert sha256_file(abs_p) == entry["sha256"], f"SHA-256 mismatch: {abs_p}"
            ckpt_map[(fold, seed)] = abs_p
        return ckpt_map

    for fold in FOLDS:
        for seed in SEEDS:
            p = os.path.join(exp030_dir, "checkpoints", star_tag, f"{star_tag}_L1200_fold{fold}_seed{seed}.pt")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Checkpoint missing for {star_tag}: {p}")
            ckpt_map[(fold, seed)] = p
    return ckpt_map


def _paired_summary(diff: np.ndarray, prim_vals: np.ndarray, ref_vals: np.ndarray, scale: float = 100.0) -> Dict[str, Any]:
    scaled_diff = diff * scale
    ci_low, ci_high = participant_bootstrap_ci(scaled_diff, n_boot=10000, alpha=0.05, seed=42)
    _, p_wil = stats.wilcoxon(scaled_diff) if np.any(scaled_diff != 0) else (0.0, 1.0)
    sd = float(np.std(scaled_diff, ddof=1))
    dz = float(np.mean(scaled_diff) / sd) if sd > 1e-12 else 0.0
    return {
        "primary_mean": float(np.mean(prim_vals) * scale),
        "ref_mean": float(np.mean(ref_vals) * scale),
        "delta": float(np.mean(scaled_diff)),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "wilcoxon_p": float(p_wil),
        "cohens_dz": dz,
        "wins": int(np.sum(scaled_diff > 0)),
        "losses": int(np.sum(scaled_diff < 0)),
        "ties": int(np.sum(scaled_diff == 0)),
    }


def evaluate_bidsleep_paired(exp030_dir: str, star_tag: str) -> Dict[str, Any]:
    """
    Pillar 1: BIDSleep N=47 paired statistics vs. Model A-164k and SLAMSS-IFS
    with 10,000-resample bootstrap CIs, two-sided Wilcoxon, and Holm-Bonferroni.
    """
    prim_csv = os.path.join(exp030_dir, f"{star_tag}_subject_metrics.csv")
    if not os.path.exists(prim_csv):
        raise FileNotFoundError(f"Missing subject metrics CSV: {prim_csv}")
    df_prim = pd.read_csv(prim_csv).sort_values("subject_id").reset_index(drop=True)
    assert len(df_prim) == 47, f"Expected 47 subjects, got {len(df_prim)}"

    # Reference 1: p2_164k (prefer logit-stitched p2_164k CSV, fallback to DEC-037 CSV)
    p2_csv = os.path.join(exp030_dir, "p2_164k_subject_metrics.csv")
    dec037_csv = os.path.join(STAGE2_DIR, "final_model_oof_subject_metrics.csv")
    if os.path.exists(p2_csv):
        df_a = pd.read_csv(p2_csv)
    elif os.path.exists(dec037_csv):
        df_a_raw = pd.read_csv(dec037_csv)
        metric_cols = ["macro_f1", "accuracy", "kappa", "f1_wake", "f1_light", "f1_deep", "f1_rem"]
        df_a = df_a_raw.groupby("subject_id", as_index=False)[metric_cols].mean()
    else:
        raise FileNotFoundError("Neither p2_164k_subject_metrics.csv nor DEC-037 CSV found.")

    # Reference 2: Official SLAMSS-IFS
    slamss_csv = os.path.join(EXP021_DIR, "slamss_ifs_matched_subject_metrics.csv")
    if not os.path.exists(slamss_csv):
        raise FileNotFoundError(f"Missing SLAMSS-IFS CSV: {slamss_csv}")
    df_f_raw = pd.read_csv(slamss_csv)
    metric_cols = ["macro_f1", "accuracy", "kappa", "f1_wake", "f1_light", "f1_deep", "f1_rem"]
    if len(df_f_raw) > 47:
        df_f = df_f_raw.groupby("subject_id", as_index=False)[metric_cols].mean()
    else:
        df_f = df_f_raw

    refs = [("modelA_164k", df_a), ("slamss_ifs", df_f)]
    comparisons: List[Dict[str, Any]] = []
    non_self_indices: List[int] = []
    non_self_ps: List[float] = []

    for ref_name, df_ref in refs:
        if star_tag == "p2_164k" and ref_name == "modelA_164k":
            mean_f1 = float(df_prim["macro_f1"].mean() * 100.0)
            comparisons.append({
                "reference": ref_name,
                "self_comparison_guarded": True,
                "primary_mean_pct": mean_f1,
                "ref_mean_pct": mean_f1,
                "delta_pp": 0.0,
                "ci95_low_pp": 0.0,
                "ci95_high_pp": 0.0,
                "wilcoxon_p": 1.0,
                "wilcoxon_p_holm": 1.0,
                "cohens_dz": 0.0,
                "wins": 0,
                "losses": 0,
                "ties": 47,
                "secondary_metrics": {},
            })
            continue

        merged = pd.merge(df_prim, df_ref, on="subject_id", suffixes=("_prim", "_ref"))
        assert len(merged) == 47, f"Subject alignment mismatch for {ref_name}: {len(merged)}"
        res_main = _paired_summary(
            merged["macro_f1_prim"].values - merged["macro_f1_ref"].values,
            merged["macro_f1_prim"].values,
            merged["macro_f1_ref"].values,
            scale=100.0,
        )
        sec: Dict[str, Any] = {}
        for col, scale in [("accuracy", 100.0), ("kappa", 1.0), ("f1_wake", 100.0), ("f1_light", 100.0), ("f1_deep", 100.0), ("f1_rem", 100.0)]:
            if f"{col}_prim" in merged.columns and f"{col}_ref" in merged.columns:
                sec[col] = _paired_summary(
                    merged[f"{col}_prim"].values - merged[f"{col}_ref"].values,
                    merged[f"{col}_prim"].values,
                    merged[f"{col}_ref"].values,
                    scale=scale,
                )

        non_self_indices.append(len(comparisons))
        non_self_ps.append(res_main["wilcoxon_p"])
        comparisons.append({
            "reference": ref_name,
            "self_comparison_guarded": False,
            "primary_mean_pct": res_main["primary_mean"],
            "ref_mean_pct": res_main["ref_mean"],
            "delta_pp": res_main["delta"],
            "ci95_low_pp": res_main["ci95_low"],
            "ci95_high_pp": res_main["ci95"],
            "wilcoxon_p": res_main["wilcoxon_p"],
            "cohens_dz": res_main["cohens_dz"],
            "wins": res_main["wins"],
            "losses": res_main["losses"],
            "ties": res_main["ties"],
            "secondary_metrics": sec,
        } if "ci95" in res_main else {
            "reference": ref_name,
            "self_comparison_guarded": False,
            "primary_mean_pct": res_main["primary_mean"],
            "ref_mean_pct": res_main["ref_mean"],
            "delta_pp": res_main["delta"],
            "ci95_low_pp": res_main["ci95_low"],
            "ci95_high_pp": res_main["ci95_high"],
            "wilcoxon_p": res_main["wilcoxon_p"],
            "cohens_dz": res_main["cohens_dz"],
            "wins": res_main["wins"],
            "losses": res_main["losses"],
            "ties": res_main["ties"],
            "secondary_metrics": sec,
        })

    if non_self_ps:
        adj = holm_bonferroni(non_self_ps)
        for idx, p_adj in zip(non_self_indices, adj):
            comparisons[idx]["wilcoxon_p_holm"] = float(p_adj)

    out = {"star_tag": star_tag, "n_subjects": 47, "comparisons": comparisons}
    out_path = os.path.join(exp030_dir, f"{star_tag}_bidsleep_paired_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def compute_model_mmacs(model: nn.Module, seq_len: int = 1200, epoch_len: int = 30) -> float:
    """Computes exact forward MACs (in millions, MMACs) for input (1, seq_len, epoch_len)."""
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

    def gru_hook(mod: nn.GRU, inp: Tuple[torch.Tensor, ...], out: Any) -> None:
        nonlocal macs
        x_in = inp[0]
        if isinstance(x_in, nn.utils.rnn.PackedSequence):
            total_steps = int(x_in.data.shape[0])
        else:
            total_steps = int(x_in.shape[0] * x_in.shape[1])
        num_dir = 2 if mod.bidirectional else 1
        h = mod.hidden_size
        for layer in range(mod.num_layers):
            in_sz = mod.input_size if layer == 0 else h * num_dir
            macs += float(num_dir * 3 * h * (in_sz + h) * total_steps)

    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))
        elif isinstance(m, nn.GRU):
            hooks.append(m.register_forward_hook(gru_hook))

    try:
        model_cpu = model.cpu().eval()
        with torch.no_grad():
            model_cpu(
                torch.zeros(1, seq_len, epoch_len),
                torch.zeros(1, seq_len, epoch_len),
                torch.zeros(1, seq_len, 5),
            )
    finally:
        for h in hooks:
            h.remove()
    return float(macs / 1e6)


def compute_exp030_int8_profile(tag: str, benchmark_latency: bool = True) -> Dict[str, Any]:
    """Computes Dynamic INT8 accounting, MMACs, and optional 1T/4T CPU latency at (1, 1200, 30)."""
    model_fp32 = build_exp030_model(tag).cpu().eval()
    n_params = sum(p.numel() for p in model_fp32.parameters())

    quantizable_params = 0
    for mod in model_fp32.modules():
        if isinstance(mod, (nn.Linear, nn.GRU)):
            quantizable_params += sum(p.numel() for p in mod.parameters())
    fp32_only_params = n_params - quantizable_params
    in_mem_kib = (quantizable_params * 1 + fp32_only_params * 4) / 1024.0
    mmacs = compute_model_mmacs(model_fp32, seq_len=1200, epoch_len=30)

    model_int8 = quantization.quantize_dynamic(
        copy.deepcopy(model_fp32), {nn.Linear, nn.GRU}, dtype=torch.qint8
    ).eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_p = os.path.join(tmpdir, "fp32.pt")
        int8_p = os.path.join(tmpdir, "int8.pt")
        torch.save(model_fp32.state_dict(), fp32_p)
        torch.save(model_int8.state_dict(), int8_p)
        fp32_disk_kib = os.path.getsize(fp32_p) / 1024.0
        int8_disk_kib = os.path.getsize(int8_p) / 1024.0

    result: Dict[str, Any] = {
        "grid_tag": tag,
        "total_params": n_params,
        "quantizable_params": quantizable_params,
        "fp32_only_params": fp32_only_params,
        "quantized_ratio": float(quantizable_params / n_params),
        "int8_in_memory_kib": float(in_mem_kib),
        "fp32_serialized_kib": float(fp32_disk_kib),
        "int8_serialized_kib": float(int8_disk_kib),
        "mmacs_L1200": float(mmacs),
    }

    if benchmark_latency:
        x_ihr = torch.randn(1, 1200, 30)
        x_acc = torch.randn(1, 1200, 30)
        x_meta = torch.randn(1, 1200, 5)

        def bench_latency(m: nn.Module, n_threads: int, n_runs: int = 10) -> float:
            prev_threads = torch.get_num_threads()
            torch.set_num_threads(n_threads)
            try:
                with torch.no_grad():
                    for _ in range(3):
                        m(x_ihr, x_acc, x_meta)
                    t0 = time.perf_counter()
                    for _ in range(n_runs):
                        m(x_ihr, x_acc, x_meta)
                    t1 = time.perf_counter()
                return float(((t1 - t0) / n_runs) * 1000.0)
            finally:
                torch.set_num_threads(prev_threads)

        result["cpu_latency_1t_fp32_ms"] = bench_latency(model_fp32, 1)
        result["cpu_latency_1t_int8_ms"] = bench_latency(model_int8, 1)
        result["cpu_latency_4t_int8_ms"] = bench_latency(model_int8, 4)

    return result


def evaluate_w0b_int8_oof(
    exp030_dir: str,
    star_tag: str,
    cache_dir: str = CACHE_DIR_DEFAULT,
    manifests_dir: str = MANIFESTS_DIR_DEFAULT,
    authoritative_csv: str = AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
) -> Dict[str, Any]:
    """
    Pillar 2: Edge memory/MMAC curve across p1..p6 + full 15-checkpoint OOF
    FP32-vs-INT8 parity (N=47) on star_tag.
    """
    edge_curve = [compute_exp030_int8_profile(t, benchmark_latency=True) for t in EXP030_GRID_ORDER]
    with open(os.path.join(exp030_dir, "exp030_edge_curve.json"), "w", encoding="utf-8") as f:
        json.dump({"edge_curve": edge_curve}, f, indent=2)

    profile = next(p for p in edge_curve if p["grid_tag"] == star_tag)
    ckpt_map = load_star_checkpoint_map(exp030_dir, star_tag)
    manifests = verify_manifests_and_splits(manifests_dir, authoritative_csv)
    records_by_sub = load_all_records_by_subject(cache_dir)

    fp32_sub_scores: Dict[str, List[float]] = {}
    int8_sub_scores: Dict[str, List[float]] = {}

    for fold in FOLDS:
        test_subs = manifests[fold]["outer_test_subjects"]
        test_records = [r for s in test_subs for r in records_by_sub[s]]
        for seed in SEEDS:
            ckpt = ckpt_map[(fold, seed)]
            m_fp32 = build_exp030_model(star_tag).cpu().eval()
            m_fp32.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            m_int8 = quantization.quantize_dynamic(
                copy.deepcopy(m_fp32), {nn.Linear, nn.GRU}, dtype=torch.qint8
            ).eval()

            p_fp32 = LogitStitchingSlidingWindowPredictor(m_fp32, context_len=1200, device=torch.device("cpu"))
            p_int8 = LogitStitchingSlidingWindowPredictor(m_int8, context_len=1200, device=torch.device("cpu"))

            for pred, store in [(p_fp32, fp32_sub_scores), (p_int8, int8_sub_scores)]:
                by_sub_t: Dict[str, List[np.ndarray]] = {}
                by_sub_p: Dict[str, List[np.ndarray]] = {}
                for rec in test_records:
                    out = pred.predict_night(rec)
                    by_sub_t.setdefault(rec["subject"], []).append(out["y_true"])
                    by_sub_p.setdefault(rec["subject"], []).append(out["y_pred"])
                for sub in test_subs:
                    m = compute_epoch_metrics(np.concatenate(by_sub_t[sub]), np.concatenate(by_sub_p[sub]))
                    store.setdefault(sub, []).append(m["macro_f1"])

    subs = sorted(fp32_sub_scores.keys())
    fp32_means = np.array([np.mean(fp32_sub_scores[s]) for s in subs]) * 100.0
    int8_means = np.array([np.mean(int8_sub_scores[s]) for s in subs]) * 100.0
    diff = int8_means - fp32_means
    ci_low, ci_high = participant_bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=42)
    _, p_wil = stats.wilcoxon(diff) if np.any(diff != 0) else (0.0, 1.0)

    result = {
        **profile,
        "n_subjects": len(subs),
        "fp32_macro_f1_pct": float(np.mean(fp32_means)),
        "int8_macro_f1_pct": float(np.mean(int8_means)),
        "delta_pp": float(np.mean(diff)),
        "ci95_low_pp": float(ci_low),
        "ci95_high_pp": float(ci_high),
        "wilcoxon_p": float(p_wil),
    }
    out_path = os.path.join(exp030_dir, f"{star_tag}_w0b_int8_profile.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


@torch.no_grad()
def evaluate_sleepaccel_isolated(
    exp030_dir: str,
    star_tag: str,
    device_str: str = "cuda:0",
) -> Dict[str, Any]:
    """
    Pillar 3: Strictly zero-shot SleepAccel (n=25 QC-clean & N=31 all, em_off only)
    with like-for-like paired F1 and Bland-Altman comparisons vs. ModelA & SLAMSS_IFS.
    """
    prep_dir = os.path.join(EXP027_DIR, "cache", "preprocessed")
    pred_dir = os.path.join(exp030_dir, "sleepaccel_cache", "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    all_records = load_preprocessed_sleepaccel(prep_dir, qc_only=False)
    evaluator = SleepEvaluator()
    ckpt_map = load_star_checkpoint_map(exp030_dir, star_tag)

    probs_raw: Dict[str, List[np.ndarray]] = {r["subject"]: [] for r in all_records}
    for fold in FOLDS:
        for seed in SEEDS:
            model = build_exp030_model(star_tag).to(device).eval()
            model.load_state_dict(torch.load(ckpt_map[(fold, seed)], map_location=device, weights_only=True))
            pred_off = LogitStitchingSlidingWindowPredictor(model=model, context_len=1200, device=device)
            for rec in all_records:
                out = pred_off.predict_night(rec)
                probs_raw[rec["subject"]].append(out["probs"].astype(np.float32))

    summary_rows: List[Dict[str, Any]] = []
    ba_rows: List[Dict[str, Any]] = []
    for rec in all_records:
        sub = rec["subject"]
        y_true = rec["y"]
        ckpt_probs = np.stack(probs_raw[sub], axis=0)
        ckpt_preds = np.argmax(ckpt_probs, axis=-1)
        ens_probs = np.mean(ckpt_probs, axis=0)
        ens_pred = np.argmax(ens_probs, axis=-1)

        np.savez_compressed(
            os.path.join(pred_dir, f"{sub}__{star_tag}_em_off.npz"),
            y_true=y_true,
            ckpt_preds=ckpt_preds,
            ens_probs=ens_probs,
            qc_pass=rec["qc_pass"],
        )

        ckpt_m = [extract_8_metrics(evaluator, y_true, ckpt_preds[c]) for c in range(15)]
        mean_m = {k: float(np.mean([m[k] for m in ckpt_m])) for k in ckpt_m[0]}
        ens_m = extract_8_metrics(evaluator, y_true, ens_pred)
        summary_rows.append({"subject_id": str(sub), "qc_pass": bool(rec["qc_pass"]), "em_mode": "em_off", "eval_mode": "mean_single_model", **mean_m})
        summary_rows.append({"subject_id": str(sub), "qc_pass": bool(rec["qc_pass"]), "em_mode": "em_off", "eval_mode": "prob_ensemble", **ens_m})

        valid = (y_true >= 0) & (y_true < 4)
        yt_v = y_true[valid]
        ref_ba = ClinicalHypnogramEvaluator.compute_night_metrics(yt_v)
        ckpt_ba = [ClinicalHypnogramEvaluator.compute_night_metrics(ckpt_preds[c][valid]) for c in range(15)]
        ens_ba = ClinicalHypnogramEvaluator.compute_night_metrics(ens_pred[valid])
        for k in SUMMARY_METRICS:
            ba_rows.append({
                "subject_id": str(sub), "qc_pass": bool(rec["qc_pass"]), "em_mode": "em_off",
                "eval_mode": "mean_single_model", "metric": k,
                "ref_val": float(ref_ba[k]), "pred_val": float(np.mean([m[k] for m in ckpt_ba])),
            })
            ba_rows.append({
                "subject_id": str(sub), "qc_pass": bool(rec["qc_pass"]), "em_mode": "em_off",
                "eval_mode": "prob_ensemble", "metric": k,
                "ref_val": float(ref_ba[k]), "pred_val": float(ens_ba[k]),
            })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(exp030_dir, f"{star_tag}_sleepaccel_metrics.csv"), index=False)

    df_ba = pd.DataFrame(ba_rows)
    df_ba["diff"] = df_ba["pred_val"] - df_ba["ref_val"]
    ba_summary: List[Dict[str, Any]] = []
    for cohort_name, qc_filter in [("qc_clean", True), ("all", None)]:
        for eval_mode in ["mean_single_model", "prob_ensemble"]:
            for k in SUMMARY_METRICS:
                sub_df = df_ba[(df_ba["eval_mode"] == eval_mode) & (df_ba["metric"] == k)]
                if qc_filter is not None:
                    sub_df = sub_df[sub_df["qc_pass"]]
                ba_stats = compute_bland_altman_stats(sub_df["ref_val"].values, sub_df["pred_val"].values)
                ba_summary.append({"cohort": cohort_name, "em_mode": "em_off", "eval_mode": eval_mode, "metric": k, **ba_stats})
    pd.DataFrame(ba_summary).to_csv(os.path.join(exp030_dir, f"{star_tag}_sleepaccel_bland_altman.csv"), index=False)

    # Like-for-like paired comparisons vs EXP-027/028 (ModelA & SLAMSS_IFS)
    df_ref_f1 = pd.read_csv(os.path.join(EXP027_DIR, "exp027_per_subject_metrics.csv"))
    df_ref_f1["subject_id"] = df_ref_f1["subject_id"].astype(str)
    df_ref_ba = pd.read_csv(os.path.join(EXP028_DIR, "exp028_sleepaccel_raw_paired_tuples.csv"))
    df_ref_ba["subject_id"] = df_ref_ba["subject_id"].astype(str)

    paired_out: Dict[str, Any] = {"star_tag": star_tag, "f1_comparisons": [], "bland_altman_comparisons": []}
    for cohort_name, qc_filter in [("qc_clean", True), ("all", None)]:
        for eval_mode in ["mean_single_model", "prob_ensemble"]:
            df_p = df_sum[df_sum["eval_mode"] == eval_mode]
            if qc_filter is not None:
                df_p = df_p[df_p["qc_pass"]]
            for ref_arm in ["ModelA", "SLAMSS_IFS"]:
                if star_tag == "p2_164k" and ref_arm == "ModelA":
                    paired_out["f1_comparisons"].append({
                        "cohort": cohort_name, "eval_mode": eval_mode, "reference": ref_arm,
                        "self_comparison_guarded": True, "delta_pp": 0.0, "wilcoxon_p": 1.0,
                    })
                    continue
                df_r = df_ref_f1[(df_ref_f1["arm"] == ref_arm) & (df_ref_f1["eval_mode"] == eval_mode)]
                if qc_filter is not None:
                    df_r = df_r[df_r["qc_pass"]]
                m = pd.merge(df_p, df_r, on="subject_id", suffixes=("_prim", "_ref"))
                f1_res = _paired_summary(
                    m["macro_f1_prim"].values - m["macro_f1_ref"].values,
                    m["macro_f1_prim"].values, m["macro_f1_ref"].values, scale=100.0
                )
                per_stage = {}
                for col, sc in [("kappa", 1.0), ("wake_f1", 100.0), ("light_f1", 100.0), ("deep_f1", 100.0), ("rem_f1", 100.0)]:
                    per_stage[col] = _paired_summary(
                        m[f"{col}_prim"].values - m[f"{col}_ref"].values,
                        m[f"{col}_prim"].values, m[f"{col}_ref"].values, scale=sc
                    )
                paired_out["f1_comparisons"].append({
                    "cohort": cohort_name, "eval_mode": eval_mode, "reference": ref_arm,
                    "n_subjects": int(len(m)), "self_comparison_guarded": False,
                    **f1_res, "secondary": per_stage,
                })

            # Bland-Altman paired comparisons (bias & absolute error)
            for k in SUMMARY_METRICS:
                df_bp = df_ba[(df_ba["eval_mode"] == eval_mode) & (df_ba["metric"] == k)]
                if qc_filter is not None:
                    df_bp = df_bp[df_bp["qc_pass"]]
                for ref_arm in ["ModelA", "SLAMSS_IFS"]:
                    if star_tag == "p2_164k" and ref_arm == "ModelA":
                        continue
                    df_br = df_ref_ba[(df_ref_ba["arm"] == ref_arm) & (df_ref_ba["eval_mode"] == eval_mode) & (df_ref_ba["metric"] == k)]
                    if qc_filter is not None:
                        df_br = df_br[df_br["qc_pass"]]
                    mb = pd.merge(df_bp, df_br, on="subject_id", suffixes=("_prim", "_ref"))
                    abs_err_prim = np.abs(mb["diff_prim"].values)
                    abs_err_ref = np.abs(mb["diff_ref"].values)
                    bias_cmp = _paired_summary(mb["diff_prim"].values - mb["diff_ref"].values, mb["diff_prim"].values, mb["diff_ref"].values, scale=1.0)
                    ae_cmp = _paired_summary(abs_err_prim - abs_err_ref, abs_err_prim, abs_err_ref, scale=1.0)
                    paired_out["bland_altman_comparisons"].append({
                        "cohort": cohort_name, "eval_mode": eval_mode, "metric": k, "reference": ref_arm,
                        "n_subjects": int(len(mb)),
                        "bias_comparison": bias_cmp,
                        "abs_error_comparison": ae_cmp,
                    })

    with open(os.path.join(exp030_dir, f"{star_tag}_sleepaccel_paired_comparisons.json"), "w", encoding="utf-8") as f:
        json.dump(paired_out, f, indent=2)

    agg = {}
    for cohort_name, qc_filter in [("qc_clean", True), ("all", None)]:
        for eval_mode in ["mean_single_model", "prob_ensemble"]:
            sub_df = df_sum[df_sum["eval_mode"] == eval_mode]
            if qc_filter is not None:
                sub_df = sub_df[sub_df["qc_pass"]]
            agg[f"em_off__{eval_mode}__{cohort_name}_macro_f1"] = float(sub_df["macro_f1"].mean() * 100.0)
    with open(os.path.join(exp030_dir, f"{star_tag}_sleepaccel_summary.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    return agg


def clone_scaled_model_with_state(
    star_tag: str,
    base_state: Dict[str, torch.Tensor],
    device: torch.device,
    scope_regex: str = r"(classifier_head|temp_norm|temporal_gru)",
) -> Tuple[nn.Module, List[nn.Parameter]]:
    """
    Lightweight ScaledConventionalCNNBiGRU state-dict clone helper (decoupled from
    FinalSleepStagingSystem / exp019_stage_d_selected_configs.json).
    """
    model = build_exp030_model(star_tag).to(device)
    model.load_state_dict(copy.deepcopy(base_state))
    pattern = re.compile(scope_regex)
    for name, p in model.named_parameters():
        p.requires_grad = bool(pattern.search(name))
    trainable = [p for p in model.parameters() if p.requires_grad]
    return model, trainable


def select_tau_star_nested(
    inner_subs: List[str],
    inner_zs_mean: Dict[str, float],
    inner_ft2_mean: Dict[str, float],
    inner_min_frac: Dict[str, float],
    tau_grid: List[float] = TAU_GRID,
) -> Tuple[float, bool]:
    """
    Exact EXP-020 nested inner-validation tau* selection rule:
    maximizes 4-decimal-rounded mean Macro-F1, breaking ties by smallest tau.
    """
    candidates = []
    for tau in tau_grid:
        gated = [
            inner_zs_mean[s] if inner_min_frac[s] < tau else inner_ft2_mean[s]
            for s in inner_subs
        ]
        mean_f1 = float(np.mean(gated))
        candidates.append((round(mean_f1, 4), float(tau), mean_f1))
    candidates.sort(key=lambda x: (-x[0], x[1]))
    best_tau = float(candidates[0][1])
    tie_used = len([c for c in candidates if c[0] == candidates[0][0]]) > 1
    return best_tau, tie_used


def _compute_min_stage_frac(cal_recs: List[Dict[str, np.ndarray]]) -> float:
    all_y = np.concatenate([r["y"] for r in cal_recs])
    valid_y = all_y[(all_y >= 0) & (all_y < 4)]
    return min(float(np.sum(valid_y == c)) / max(len(valid_y), 1) for c in range(4))


def _run_ft2_single_subject(
    star_tag: str,
    base_state: Dict[str, torch.Tensor],
    cal_recs: List[Dict[str, np.ndarray]],
    eval_recs: List[Dict[str, np.ndarray]],
    seed: int,
    device: torch.device,
    lr: float = 1e-4,
    epochs: int = 20,
    scope_regex: str = r"(classifier_head|temp_norm|temporal_gru)",
) -> Tuple[float, float]:
    # ZS
    m_zs = build_exp030_model(star_tag).to(device).eval()
    m_zs.load_state_dict(copy.deepcopy(base_state))
    p_zs = LogitStitchingSlidingWindowPredictor(m_zs, context_len=1200, device=device)
    f1_zs = evaluate_records_macro_f1(p_zs, eval_recs)

    # FT-2
    set_deterministic_seed(seed)
    m_ft, trainable = clone_scaled_model_with_state(star_tag, base_state, device, scope_regex)
    opt = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    batch = prepare_calibration_batch(cal_recs, context_len=1200, device=str(device))
    m_ft.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = m_ft(batch["x_ihr"], batch["x_acc"], batch["x_meta"], seq_lens=batch.get("seq_lens", batch.get("seq_len")))
        loss = compute_stage_loss_fp32(m_ft, out["logits"], batch["y"], batch["mask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        opt.step()
    m_ft.eval()
    p_ft = LogitStitchingSlidingWindowPredictor(m_ft, context_len=1200, device=device)
    f1_ft = evaluate_records_macro_f1(p_ft, eval_recs)
    return f1_zs, f1_ft


def evaluate_exp020_gate_tau(
    exp030_dir: str,
    star_tag: str,
    device_str: str = "cuda:0",
    lr: float = 1e-4,
    epochs: int = 20,
    cache_dir: str = CACHE_DIR_DEFAULT,
    manifests_dir: str = MANIFESTS_DIR_DEFAULT,
    authoritative_csv: str = AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
) -> Dict[str, Any]:
    """
    Pillar 4: EXP-020 FT-2 & nested per-fold GATE_tau personalization on N=46
    multi-night subjects (>= 3 usable nights).
    """
    manifests = verify_manifests_and_splits(manifests_dir, authoritative_csv)
    records_by_sub = load_all_records_by_subject(cache_dir)
    ckpt_map = load_star_checkpoint_map(exp030_dir, star_tag)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    scope_regex = r"(classifier_head|temp_norm|temporal_gru)"

    rows: List[Dict[str, Any]] = []
    per_fold_selection: Dict[str, Any] = {}

    for fold in FOLDS:
        inner_subs = [
            s for s in manifests[fold]["inner_val_subjects"]
            if len(records_by_sub[s]) >= 3
        ]
        outer_subs = [
            s for s in manifests[fold]["outer_test_subjects"]
            if len(records_by_sub[s]) >= 3
        ]

        # 1. Inner-val sweep for tau_f*
        inner_zs: Dict[str, List[float]] = {s: [] for s in inner_subs}
        inner_ft2: Dict[str, List[float]] = {s: [] for s in inner_subs}
        inner_min_frac: Dict[str, float] = {}
        for s in inner_subs:
            recs = sorted(records_by_sub[s], key=lambda r: int(r["night"]))
            inner_min_frac[s] = _compute_min_stage_frac(recs[:2])

        for seed in SEEDS:
            base_state = torch.load(ckpt_map[(fold, seed)], map_location=device, weights_only=True)
            for s in inner_subs:
                recs = sorted(records_by_sub[s], key=lambda r: int(r["night"]))
                f1_zs, f1_ft = _run_ft2_single_subject(
                    star_tag, base_state, recs[:2], recs[2:], seed, device, lr, epochs, scope_regex
                )
                inner_zs[s].append(f1_zs)
                inner_ft2[s].append(f1_ft)

        inner_zs_mean = {s: float(np.mean(inner_zs[s])) for s in inner_subs}
        inner_ft2_mean = {s: float(np.mean(inner_ft2[s])) for s in inner_subs}
        tau_star, tie_used = select_tau_star_nested(inner_subs, inner_zs_mean, inner_ft2_mean, inner_min_frac, TAU_GRID)

        # 2. Outer-test evaluation using fold-specific tau_star
        outer_zs: Dict[str, List[float]] = {s: [] for s in outer_subs}
        outer_ft2: Dict[str, List[float]] = {s: [] for s in outer_subs}
        outer_min_frac: Dict[str, float] = {}
        for s in outer_subs:
            recs = sorted(records_by_sub[s], key=lambda r: int(r["night"]))
            outer_min_frac[s] = _compute_min_stage_frac(recs[:2])

        for seed in SEEDS:
            base_state = torch.load(ckpt_map[(fold, seed)], map_location=device, weights_only=True)
            for s in outer_subs:
                recs = sorted(records_by_sub[s], key=lambda r: int(r["night"]))
                f1_zs, f1_ft = _run_ft2_single_subject(
                    star_tag, base_state, recs[:2], recs[2:], seed, device, lr, epochs, scope_regex
                )
                outer_zs[s].append(f1_zs)
                outer_ft2[s].append(f1_ft)

        n_abstained_fold = 0
        for s in outer_subs:
            abstained = outer_min_frac[s] < tau_star
            if abstained:
                n_abstained_fold += 1
            zs_m = float(np.mean(outer_zs[s]))
            ft_m = float(np.mean(outer_ft2[s]))
            gate_m = zs_m if abstained else ft_m
            rows.append({
                "subject_id": s,
                "fold": fold,
                "tau_star_fold": tau_star,
                "min_stage_frac": outer_min_frac[s],
                "abstained": abstained,
                "zs_macro_f1": zs_m,
                "ft2_macro_f1": ft_m,
                "gate_tau_macro_f1": gate_m,
            })

        per_fold_selection[str(fold)] = {
            "tau_star": tau_star,
            "tie_breaker_used": tie_used,
            "n_inner_subs": len(inner_subs),
            "n_outer_subs": len(outer_subs),
            "n_outer_abstained": n_abstained_fold,
        }

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp030_dir, f"{star_tag}_exp020_gate_tau_metrics.csv"), index=False)

    ft2_vs_zs = _paired_summary(df["ft2_macro_f1"].values - df["zs_macro_f1"].values, df["ft2_macro_f1"].values, df["zs_macro_f1"].values, scale=100.0)
    gate_vs_zs = _paired_summary(df["gate_tau_macro_f1"].values - df["zs_macro_f1"].values, df["gate_tau_macro_f1"].values, df["zs_macro_f1"].values, scale=100.0)
    gate_vs_ft2 = _paired_summary(df["gate_tau_macro_f1"].values - df["ft2_macro_f1"].values, df["gate_tau_macro_f1"].values, df["ft2_macro_f1"].values, scale=100.0)

    summary = {
        "star_tag": star_tag,
        "n_subjects": int(len(df)),
        "per_fold_tau_selection": per_fold_selection,
        "n_abstained_total": int(df["abstained"].sum()),
        "zs_mean_pct": float(df["zs_macro_f1"].mean() * 100.0),
        "ft2_mean_pct": float(df["ft2_macro_f1"].mean() * 100.0),
        "gate_tau_mean_pct": float(df["gate_tau_macro_f1"].mean() * 100.0),
        "ft2_vs_zs": ft2_vs_zs,
        "gate_vs_zs": gate_vs_zs,
        "gate_vs_ft2": gate_vs_ft2,
    }
    with open(os.path.join(exp030_dir, f"{star_tag}_exp020_gate_tau_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-030 Downstream Evaluation Suite (CNN-BiGRU-Star Only)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["bidsleep", "int8", "sleepaccel", "gate_tau"],
        choices=["bidsleep", "int8", "sleepaccel", "gate_tau"],
    )
    parser.add_argument("--exp030-dir", type=str, default=EXP030_DIR)
    args = parser.parse_args()

    selection = load_elbow_selection(args.exp030_dir)
    star_tag = selection["star_tag"]
    print(f"[EXP-030 DOWNSTREAM] Locked CNN-BiGRU-Star = {star_tag} ({selection['star_params']:,} params)")

    if "bidsleep" in args.tasks:
        print(json.dumps(evaluate_bidsleep_paired(args.exp030_dir, star_tag), indent=2))
    if "int8" in args.tasks:
        print(json.dumps(evaluate_w0b_int8_oof(args.exp030_dir, star_tag), indent=2))
    if "sleepaccel" in args.tasks:
        print(json.dumps(evaluate_sleepaccel_isolated(args.exp030_dir, star_tag, args.device), indent=2))
    if "gate_tau" in args.tasks:
        print(json.dumps(evaluate_exp020_gate_tau(args.exp030_dir, star_tag, args.device), indent=2))


if __name__ == "__main__":
    main()
