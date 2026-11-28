"""
EXP-034 500-Epoch Convergence Audit on Compact CNN-BiGRU Widths
File: experiments/EXP-034/run_exp034_convergence.py

Preregistration: docs/designs/PREREG_EXP-034_500EPOCH_CONVERGENCE.md
Lock commit: c7d2d19e34563056a2d2fe07e6c53db879ab050e

Strict isolation: all EXP-018 through EXP-033 artifacts remain read-only;
all outputs are written exclusively under experiments/EXP-034/.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

EXP034_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP034_DIR, "..", ".."))
EXP031_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-031")
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
EXP018_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-018")
for _p in (REPO_ROOT, EXP034_DIR, EXP031_DIR, EXP030_DIR, EXP029_DIR, EXP018_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp018_dataset import RandomCropSleepDataset  # noqa: E402
from exp029_calibration import (  # noqa: E402
    LogitStitchingSlidingWindowPredictor,
    estimate_pi_src_from_records,
)
from exp030_models import build_exp030_model  # noqa: E402
from exp031_models import build_exp031_model  # noqa: E402
from run_exp029_cv import (  # noqa: E402
    AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
    CACHE_DIR_DEFAULT,
    FOLDS_DEFAULT,
    MANIFESTS_DIR_DEFAULT,
    SEEDS_DEFAULT,
    compute_stage_loss_fp32,
    evaluate_records,
    load_all_records_by_subject,
    parse_fold_seed_pairs,
    set_deterministic_seed,
    sha256_file,
    verify_manifests_and_splits,
)

OUTPUT_DIR_DEFAULT = Path("experiments/EXP-034")
EXPECTED_FOLD_MANIFEST_SHA256 = "36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf"

EXP034_ARMS_ORDER = ["W050_500ep", "W075_500ep", "p1_100k_500ep", "W100_500ep"]

EXP034_ARMS: Dict[str, Dict[str, Any]] = {
    "W050_500ep": {
        "factory": "exp031",
        "key": "W050",
        "expected_params": 43108,
        "baseline_csv": os.path.join(REPO_ROOT, "experiments", "EXP-031", "W050_subject_metrics.csv"),
    },
    "W075_500ep": {
        "factory": "exp031",
        "key": "W075",
        "expected_params": 93796,
        "baseline_csv": os.path.join(REPO_ROOT, "experiments", "EXP-031", "W075_subject_metrics.csv"),
    },
    "p1_100k_500ep": {
        "factory": "exp030",
        "key": "p1_100k",
        "expected_params": 98940,
        "baseline_csv": os.path.join(REPO_ROOT, "experiments", "EXP-030", "p1_100k_subject_metrics.csv"),
    },
    "W100_500ep": {
        "factory": "exp031",
        "key": "A",
        "expected_params": 163940,
        "baseline_csv": os.path.join(REPO_ROOT, "experiments", "EXP-031", "runs", "W100", "W100_subject_metrics.csv"),
    },
}


def build_exp034_model(arm_id: str) -> torch.nn.Module:
    spec = EXP034_ARMS[arm_id]
    if spec["factory"] == "exp031":
        model = build_exp031_model(arm=spec["key"], loss_type="rccloss")
    elif spec["factory"] == "exp030":
        model = build_exp030_model(tag=spec["key"], loss_type="rccloss")
    else:
        raise ValueError(f"Unknown factory: {spec['factory']}")
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == spec["expected_params"], (
        f"{arm_id}: expected {spec['expected_params']} params, got {n_params}"
    )
    return model


def verify_exact_manifest_sha256(csv_path: str) -> None:
    actual = sha256_file(csv_path)
    if actual != EXPECTED_FOLD_MANIFEST_SHA256:
        raise AssertionError(
            f"Manifest SHA-256 mismatch for {csv_path}: {actual} != {EXPECTED_FOLD_MANIFEST_SHA256}"
        )


def run_single_fold_seed(
    arm_id: str,
    fold: int,
    seed: int,
    manifest: Dict[str, Any],
    records_by_sub: Dict[str, List[Dict[str, np.ndarray]]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out_root = os.path.abspath(str(args.output_dir))
    base_out = os.path.join(out_root, "smoke_out") if args.smoke else out_root
    runs_dir = os.path.join(base_out, "run_results", arm_id)
    ckpt_dir = os.path.join(base_out, "checkpoints", arm_id)
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    run_json_path = os.path.join(runs_dir, f"{arm_id}_fold{fold}_seed{seed}.json")
    ckpt_path = os.path.join(ckpt_dir, f"{arm_id}_L1200_fold{fold}_seed{seed}.pt")

    if (not args.smoke) and (not args.force) and os.path.exists(run_json_path):
        print(f"[SKIP] Completed run already exists: {run_json_path}")
        with open(run_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    set_deterministic_seed(seed)

    train_subs = manifest["inner_train_subjects"]
    val_subs = manifest["inner_val_subjects"]
    test_subs = manifest["outer_test_subjects"]
    train_records = [r for s in train_subs for r in records_by_sub[s]]
    val_records = [r for s in val_subs for r in records_by_sub[s]]
    test_records = [r for s in test_subs for r in records_by_sub[s]]

    subject_to_idx = {s: i for i, s in enumerate(sorted(train_subs))}
    train_dataset = RandomCropSleepDataset(
        records=train_records,
        subject_to_idx=subject_to_idx,
        context_len=args.context_len,
        neutralize_circadian=True,
        seed=seed,
    )
    dl_gen = torch.Generator()
    dl_gen.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        generator=dl_gen,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = build_exp034_model(arm_id).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    max_epochs = 2 if args.smoke else args.max_epochs
    min_epochs = 1 if args.smoke else args.min_epochs
    patience = args.patience

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []
    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x_ihr = batch["ihr"].to(device, non_blocking=True)
            x_acc = batch["accel"].to(device, non_blocking=True)
            x_meta = batch["meta"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            seq_len = batch["seq_len"]

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x_ihr=x_ihr, x_accel=x_acc, x_meta=x_meta, seq_lens=seq_len)

            loss = compute_stage_loss_fp32(model, out["logits"], y, mask)
            assert torch.isfinite(loss), f"Non-finite loss at epoch {epoch}: {loss}"

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            n_batches += 1

        val_predictor = LogitStitchingSlidingWindowPredictor(
            model=model,
            context_len=args.context_len,
            neutralize_circadian=True,
            device=device,
        )
        val_pooled, _ = evaluate_records(val_predictor, val_records)
        val_f1 = val_pooled["macro_f1"]
        train_loss = running_loss / max(n_batches, 1)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_f1,
            "val_accuracy": val_pooled["accuracy"],
            "elapsed_sec": time.time() - ep_t0,
        })

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            tmp_ckpt = ckpt_path + ".tmp"
            torch.save(model.state_dict(), tmp_ckpt)
            os.replace(tmp_ckpt, ckpt_path)
        else:
            epochs_no_improve += 1

        if epoch % 25 == 0 or epoch == 1 or args.smoke:
            print(
                f"[{arm_id} F{fold} S{seed}] Ep {epoch:03d}/{max_epochs} | "
                f"loss={train_loss:.4f} | val_f1={val_f1*100:.2f}% (best={best_val_f1*100:.2f}% @ {best_epoch})",
                flush=True,
            )

        if epoch >= min_epochs and epochs_no_improve >= patience:
            print(
                f"[{arm_id} F{fold} S{seed}] Early stopping at epoch {epoch} (best={best_epoch}).",
                flush=True,
            )
            break

    wall_sec = time.time() - t_start
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda"
        else 0.0
    )

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    pi_src = estimate_pi_src_from_records(
        model=model,
        val_records=val_records,
        context_len=args.context_len,
        neutralize_circadian=True,
        device=device,
    )
    test_predictor = LogitStitchingSlidingWindowPredictor(
        model=model,
        context_len=args.context_len,
        neutralize_circadian=True,
        device=device,
    )
    test_pooled, test_per_sub = evaluate_records(test_predictor, test_records)

    completed_epochs = len(history)
    early_stopped = bool(completed_epochs < max_epochs)

    result = {
        "arm": arm_id,
        "grid_tag": arm_id,
        "fold": fold,
        "seed": seed,
        "n_params": n_params,
        "loss_type": "rccloss",
        "max_epochs": max_epochs,
        "min_epochs": min_epochs,
        "patience": patience,
        "completed_epochs": completed_epochs,
        "early_stopped": early_stopped,
        "smoke": bool(args.smoke),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "pi_src_inner_val": pi_src.tolist(),
        "wall_sec": wall_sec,
        "peak_vram_mb": peak_vram_mb,
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": sha256_file(ckpt_path),
        "outer_test_pooled": test_pooled,
        "outer_test_per_subject": test_per_sub,
        "history": history,
    }
    tmp_json = run_json_path + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_json, run_json_path)

    print(
        f"[DONE {arm_id} F{fold} S{seed}] Ep={completed_epochs}/{max_epochs} (best={best_epoch}, "
        f"ES={early_stopped}) | Test F1={test_pooled['macro_f1']*100:.2f}% | "
        f"VRAM={peak_vram_mb:.1f} MB | Time={wall_sec:.1f}s",
        flush=True,
    )
    return result


def aggregate_arm(
    arm_id: str,
    folds: List[int],
    seeds: List[int],
    output_dir: str,
) -> Dict[str, Any]:
    runs_dir = os.path.join(output_dir, "run_results", arm_id)
    rows: List[Dict[str, Any]] = []
    n_params = None
    wall_secs, peak_vrams, comp_eps, best_eps, es_flags = [], [], [], [], []

    for fold in folds:
        for seed in seeds:
            jpath = os.path.join(runs_dir, f"{arm_id}_fold{fold}_seed{seed}.json")
            if not os.path.exists(jpath):
                raise FileNotFoundError(f"Missing run JSON for aggregation: {jpath}")
            with open(jpath, "r", encoding="utf-8") as f:
                res = json.load(f)
            n_params = res.get("n_params", n_params)
            wall_secs.append(float(res["wall_sec"]))
            peak_vrams.append(float(res["peak_vram_mb"]))
            comp_eps.append(int(res["completed_epochs"]))
            best_eps.append(int(res["best_epoch"]))
            es_flags.append(bool(res["early_stopped"]))
            for sub, m in res["outer_test_per_subject"].items():
                rows.append({
                    "arm": arm_id,
                    "grid_tag": arm_id,
                    "fold": fold,
                    "seed": seed,
                    "subject_id": sub,
                    **m,
                })

    df_all = pd.DataFrame(rows)
    metric_cols = [
        "macro_f1", "weighted_f1", "accuracy", "kappa",
        "f1_wake", "f1_light", "f1_deep", "f1_rem",
    ]
    df_sub = df_all.groupby("subject_id", as_index=False)[metric_cols].mean()
    df_sub["arm"] = arm_id
    df_sub["grid_tag"] = arm_id

    sub_csv = os.path.join(output_dir, f"{arm_id}_subject_metrics.csv")
    df_sub.to_csv(sub_csv, index=False)

    summary = {
        "arm": arm_id,
        "n_params": n_params,
        "n_subjects": int(len(df_sub)),
        "mean_macro_f1_pct": float(df_sub["macro_f1"].mean() * 100.0),
        "std_macro_f1_pct": float(df_sub["macro_f1"].std(ddof=1) * 100.0),
        "completed_epochs_mean": float(np.mean(comp_eps)),
        "completed_epochs_range": [int(min(comp_eps)), int(max(comp_eps))],
        "best_epoch_mean": float(np.mean(best_eps)),
        "best_epoch_range": [int(min(best_eps)), int(max(best_eps))],
        "n_early_stopped": int(sum(es_flags)),
        "n_hit_500": int(sum(not x for x in es_flags)),
        "mean_wall_sec": float(np.mean(wall_secs)),
        "max_peak_vram_mb": float(np.max(peak_vrams)),
        "subject_metrics_csv": sub_csv,
    }
    return summary


def holm_bonferroni(pvals: List[float]) -> List[float]:
    m = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed):
        val = min(1.0, p * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def compute_pages_l_exact_spec(df_matrix: np.ndarray) -> Dict[str, float]:
    """
    Page's L test across N subjects x k ordered conditions (increasing order).
    Standard normal approximation per PREREG_EXP-034 Section 4.3.
    """
    n, k = df_matrix.shape
    ranks = np.apply_along_axis(stats.rankdata, 1, df_matrix)
    col_rank_sums = np.sum(ranks, axis=0)
    L = float(np.sum(np.arange(1, k + 1) * col_rank_sums))
    expected_L = n * k * ((k + 1) ** 2) / 4.0
    var_L = n * (k ** 2) * (k + 1) * ((k ** 2) - 1) / 144.0
    z_L = (L - expected_L) / math.sqrt(var_L)
    phi_z = float(stats.norm.cdf(z_L))
    p_one = float(1.0 - phi_z)
    p_two = float(2.0 * min(phi_z, 1.0 - phi_z))
    return {"N": int(n), "k": int(k), "L": L, "z_L": float(z_L), "p_one_sided": p_one, "p_two_sided": p_two}


def run_full_analysis(output_dir: str, folds: List[int], seeds: List[int]) -> None:
    arm_summaries = {}
    paired_results = []
    raw_pvals = []

    for arm_id in EXP034_ARMS_ORDER:
        s = aggregate_arm(arm_id, folds, seeds, output_dir)
        arm_summaries[arm_id] = s

        df_500 = pd.read_csv(s["subject_metrics_csv"]).sort_values("subject_id")
        df_100 = pd.read_csv(EXP034_ARMS[arm_id]["baseline_csv"]).sort_values("subject_id")
        merged = pd.merge(
            df_500[["subject_id", "macro_f1"]],
            df_100[["subject_id", "macro_f1"]],
            on="subject_id",
            suffixes=("_500", "_100"),
        )
        assert len(merged) == 47, f"{arm_id}: expected 47 merged subjects, got {len(merged)}"

        diff = (merged["macro_f1_500"].values - merged["macro_f1_100"].values) * 100.0
        mean_d = float(np.mean(diff))
        med_d = float(np.median(diff))
        sd_d = float(np.std(diff, ddof=1))

        rng = np.random.default_rng(42)
        idx = rng.integers(0, len(diff), size=(10000, len(diff)))
        boot_means = diff[idx].mean(axis=1)
        ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])

        _, p_wil = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")
        wins = int(np.sum(diff > 0))
        losses = int(np.sum(diff < 0))
        ties = int(np.sum(diff == 0))

        paired_results.append({
            "arm": arm_id,
            "n_params": s["n_params"],
            "f1_100ep_pct": float(merged["macro_f1_100"].mean() * 100.0),
            "f1_500ep_pct": float(merged["macro_f1_500"].mean() * 100.0),
            "delta_mean_pp": mean_d,
            "delta_median_pp": med_d,
            "sd_pp": sd_d,
            "ci95_low_pp": float(ci_lo),
            "ci95_high_pp": float(ci_hi),
            "p_raw_two_sided": float(p_wil),
            "wins": wins,
            "losses": losses,
            "ties": ties,
        })
        raw_pvals.append(float(p_wil))

    holm_pvals = holm_bonferroni(raw_pvals)
    any_compact_gain = False
    for i, r in enumerate(paired_results):
        p_h = float(holm_pvals[i])
        r["p_holm_two_sided"] = p_h
        if r["delta_mean_pp"] >= 0.75 and r["ci95_low_pp"] > 0.0 and p_h < 0.05:
            r["verdict"] = "SUPPORTED"
            any_compact_gain = True
        elif p_h < 0.05 and r["ci95_low_pp"] > 0.0:
            r["verdict"] = "STATISTICALLY DETECTABLE BUT BELOW PRACTICAL THRESHOLD"
            any_compact_gain = True
        else:
            r["verdict"] = "NOT SUPPORTED"

    # Section 4.3 Diagnostic H-SCALE Sensitivity Check (9-width and 10-width)
    grid_9 = [
        ("W050_500ep", os.path.join(output_dir, "W050_500ep_subject_metrics.csv")),
        ("W075_500ep", os.path.join(output_dir, "W075_500ep_subject_metrics.csv")),
        ("W100_500ep", os.path.join(output_dir, "W100_500ep_subject_metrics.csv")),
        ("P190", os.path.join(REPO_ROOT, "experiments", "EXP-031", "P190_subject_metrics.csv")),
        ("B", os.path.join(REPO_ROOT, "experiments", "EXP-029", "cnn_bigru_200k_subject_metrics.csv")),
        ("W125", os.path.join(REPO_ROOT, "experiments", "EXP-031", "W125_subject_metrics.csv")),
        ("W150", os.path.join(REPO_ROOT, "experiments", "EXP-031", "W150_subject_metrics.csv")),
        ("W200", os.path.join(REPO_ROOT, "experiments", "EXP-031", "W200_subject_metrics.csv")),
        ("W250", os.path.join(REPO_ROOT, "experiments", "EXP-031", "W250_subject_metrics.csv")),
    ]
    grid_10 = [
        grid_9[0],
        grid_9[1],
        ("p1_100k_500ep", os.path.join(output_dir, "p1_100k_500ep_subject_metrics.csv")),
        *grid_9[2:],
    ]

    def build_matrix(grid: List[Tuple[str, str]]) -> np.ndarray:
        dfs = [pd.read_csv(p).sort_values("subject_id")[["subject_id", "macro_f1"]] for _, p in grid]
        subs = dfs[0]["subject_id"].values
        for d in dfs[1:]:
            assert np.array_equal(subs, d["subject_id"].values)
        return np.column_stack([d["macro_f1"].values for d in dfs])

    pages_9 = compute_pages_l_exact_spec(build_matrix(grid_9))
    pages_10 = compute_pages_l_exact_spec(build_matrix(grid_10))

    summary_out = {
        "experiment": "EXP-034",
        "prereg_commit": "c7d2d19e34563056a2d2fe07e6c53db879ab050e",
        "arm_summaries": arm_summaries,
        "paired_500_vs_100": paired_results,
        "diagnostic_h_scale_9width": pages_9,
        "diagnostic_h_scale_10width": pages_10,
        "gating_rule_triggered_train_larger_500ep": any_compact_gain,
    }
    sum_path = os.path.join(output_dir, "exp034_summary.json")
    with open(sum_path, "w", encoding="utf-8") as f:
        json.dump(summary_out, f, indent=2)

    lines = [
        "# EXP-034 Final Report: 500-Epoch Convergence Audit on Compact CNN-BiGRU Widths",
        "",
        "- **Preregistration:** `docs/designs/PREREG_EXP-034_500EPOCH_CONVERGENCE.md` (lock commit `c7d2d19e34563056a2d2fe07e6c53db879ab050e`)",
        f"- **Gating Rule (`P190`–`W250` 500-epoch extension required?):** **{'YES' if any_compact_gain else 'NO'}**",
        "",
        "## 1. Paired Convergence Results (`500ep - 100ep`, $N = 47$)",
        "",
        "| Arm | Params | 100-ep F1 (%) | 500-ep F1 (%) | Mean $\\Delta$ (pp) | Median $\\Delta$ (pp) | 95% CI | W/L/T | $p_{\\text{raw}}$ (2s) | $p_{\\text{Holm}}$ (2s) | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in paired_results:
        lines.append(
            f"| `{r['arm']}` | {r['n_params']:,} | {r['f1_100ep_pct']:.2f} | {r['f1_500ep_pct']:.2f} | "
            f"{r['delta_mean_pp']:+.2f} | {r['delta_median_pp']:+.2f} | "
            f"[{r['ci95_low_pp']:+.2f}, {r['ci95_high_pp']:+.2f}] | {r['wins']}/{r['losses']}/{r['ties']} | "
            f"{r['p_raw_two_sided']:.4e} | {r['p_holm_two_sided']:.4e} | **{r['verdict']}** |"
        )
    lines.extend([
        "",
        "## 2. Per-Arm Training Convergence Logs (15 runs/arm)",
        "",
        "| Arm | Completed Epochs Mean [Range] | Best Epoch Mean [Range] | Early Stopped | Hit 500 Ceiling | Mean Wall (s) | Max VRAM (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for arm_id in EXP034_ARMS_ORDER:
        s = arm_summaries[arm_id]
        lines.append(
            f"| `{arm_id}` | {s['completed_epochs_mean']:.2f} [{s['completed_epochs_range'][0]}, {s['completed_epochs_range'][1]}] | "
            f"{s['best_epoch_mean']:.2f} [{s['best_epoch_range'][0]}, {s['best_epoch_range'][1]}] | "
            f"{s['n_early_stopped']}/15 | {s['n_hit_500']}/15 | {s['mean_wall_sec']:.1f} | {s['max_peak_vram_mb']:.1f} |"
        )
    lines.extend([
        "",
        "## 3. Diagnostic Mixed-Budget `H-SCALE` Sensitivity Check (Page's $L$, $N = 47$)",
        "",
        f"- **9-Width (`W050_500ep`, `W075_500ep`, `W100_500ep` + 100-ep `P190`–`W250`):** $L = {pages_9['L']:.1f}$, $z_L = {pages_9['z_L']:+.3f}$, $p_{{\\text{{one}}}} = {pages_9['p_one_sided']:.4e}$, $p_{{\\text{{two}}}} = {pages_9['p_two_sided']:.4e}$",
        f"- **10-Width (inserting `p1_100k_500ep`):** $L = {pages_10['L']:.1f}$, $z_L = {pages_10['z_L']:+.3f}$, $p_{{\\text{{one}}}} = {pages_10['p_one_sided']:.4e}$, $p_{{\\text{{two}}}} = {pages_10['p_two_sided']:.4e}$",
    ])
    rep_path = os.path.join(output_dir, "EXP034_FINAL_REPORT.md")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[REPORT WRITTEN] {sum_path} & {rep_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-034 500-Epoch Convergence Audit Driver")
    parser.add_argument("--arms", type=str, nargs="+", default=EXP034_ARMS_ORDER, choices=EXP034_ARMS_ORDER)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--folds", type=int, nargs="+", default=FOLDS_DEFAULT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    parser.add_argument("--pairs", type=str, nargs="*", default=None)
    parser.add_argument("--context-len", type=int, default=1200)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--analyze", action="store_true", help="Aggregate all 4 arms and write summary/report")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR_DEFAULT)
    parser.add_argument("--manifests-dir", type=str, default=MANIFESTS_DIR_DEFAULT)
    parser.add_argument("--authoritative-csv", type=str, default=AUTHORITATIVE_MANIFEST_CSV_DEFAULT)
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR_DEFAULT))
    args = parser.parse_args()

    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.analyze:
        run_full_analysis(out_dir, args.folds, args.seeds)
        return

    verify_exact_manifest_sha256(args.authoritative_csv)
    manifests = verify_manifests_and_splits(args.manifests_dir, args.authoritative_csv)
    records_by_sub = load_all_records_by_subject(args.cache_dir)

    pairs = [(1, 42)] if args.smoke and not args.pairs else parse_fold_seed_pairs(args)

    for arm_id in args.arms:
        for fold, seed in pairs:
            run_single_fold_seed(
                arm_id=arm_id,
                fold=fold,
                seed=seed,
                manifest=manifests[fold],
                records_by_sub=records_by_sub,
                args=args,
            )


if __name__ == "__main__":
    main()
