"""
EXP-029 Downstream Evaluation Suite
File: experiments/EXP-029/eval_exp029_downstream.py

Automated evaluation across 4 pillars:
  1. BIDSleep 5-fold CV (N=47): 10,000-resample paired bootstrap CIs, two-sided
     Wilcoxon signed-rank tests, and Holm-Bonferroni FWER vs. Arms (a), (b), (c), (d), (f).
  2. W0-B Dynamic INT8 edge quantization + 1T/4T CPU latency & payload profiling.
  3. PhysioNet SleepAccel (EXP-027/028 zero-shot + post-hoc DampedEMCalibrator
     sensitivity) writing strictly to experiments/EXP-029/sleepaccel_cache/predictions/.
  4. EXP-020 FT-2 / GATE_tau (tau*=0.01) adaptation safety check on N=46.
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

EXP029_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP029_DIR, "..", ".."))
EXP019_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-019")
EXP021_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-021")
EXP027_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-027")
STAGE2_DIR = os.path.join(REPO_ROOT, "experiments", "STAGE2_FINAL_MODEL")

for _p in (REPO_ROOT, EXP029_DIR, EXP019_DIR, EXP021_DIR, EXP027_DIR, STAGE2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_tsk08_paired_benchmark import participant_bootstrap_ci
from evaluator import ClinicalHypnogramEvaluator, SleepEvaluator
from exp019_adaptation_core import prepare_calibration_batch
from exp029_calibration import DampedEMCalibrator, LogitStitchingSlidingWindowPredictor
from exp029_models import ARM_ALIASES, build_exp029_model
from run_exp027_sleepaccel_inference import extract_8_metrics, load_preprocessed_sleepaccel
from run_exp028_sleepaccel_bland_altman import SUMMARY_METRICS, compute_bland_altman_stats
from run_exp029_cv import (
    CACHE_DIR_DEFAULT,
    MANIFESTS_DIR_DEFAULT,
    AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
    compute_epoch_metrics,
    compute_stage_loss_fp32,
    load_all_records_by_subject,
    set_deterministic_seed,
    verify_manifests_and_splits,
)

FOLDS = [1, 2, 3, 4, 5]
SEEDS = [42, 123, 2024]


def holm_bonferroni(p_values: List[float]) -> List[float]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        val = min(1.0, p * (m - rank))
        running_max = max(running_max, val)
        adj[orig_idx] = running_max
    return adj


def evaluate_bidsleep_paired(exp029_dir: str, primary_arm: str = "full_v5") -> Dict[str, Any]:
    """
    Compares primary_arm (N=47 3-seed participant means) against available
    EXP-029 arms and locked baselines (a) Model A and (f) SLAMSS-IFS.
    """
    primary_csv = os.path.join(exp029_dir, f"{primary_arm}_subject_metrics.csv")
    if not os.path.exists(primary_csv):
        raise FileNotFoundError(f"Primary arm CSV missing: {primary_csv}")
    df_prim = pd.read_csv(primary_csv).sort_values("subject_id").reset_index(drop=True)
    assert len(df_prim) == 47, f"Expected 47 subjects, got {len(df_prim)}"

    candidates: Dict[str, pd.DataFrame] = {}

    # Arm (a): Locked Model A (141 subject-seed rows -> 47 participant means)
    arm_a_path = os.path.join(STAGE2_DIR, "final_model_oof_subject_metrics.csv")
    if os.path.exists(arm_a_path):
        df_a = pd.read_csv(arm_a_path)
        sub_col = "subject_id" if "subject_id" in df_a.columns else "subject"
        df_a = df_a.groupby(sub_col, as_index=False)["macro_f1"].mean()
        df_a = df_a.rename(columns={sub_col: "subject_id"})
        candidates["arm_a_modelA_164k"] = df_a

    # Arm (f): Locked SLAMSS-IFS
    arm_f_path = os.path.join(EXP021_DIR, "slamss_ifs_matched_subject_metrics.csv")
    if os.path.exists(arm_f_path):
        df_f = pd.read_csv(arm_f_path)
        sub_col = "subject_id" if "subject_id" in df_f.columns else "subject"
        if len(df_f) > 47:
            df_f = df_f.groupby(sub_col, as_index=False)["macro_f1"].mean()
        df_f = df_f.rename(columns={sub_col: "subject_id"})
        candidates["arm_f_slamss_ifs"] = df_f

    for ref_arm in ["cnn_bigru_200k", "film_only", "pyramid_only", "full_v5_ce"]:
        p = os.path.join(exp029_dir, f"{ref_arm}_subject_metrics.csv")
        if os.path.exists(p):
            candidates[ref_arm] = pd.read_csv(p)

    comparisons: List[Dict[str, Any]] = []
    raw_ps: List[float] = []

    for name, df_ref in candidates.items():
        merged = pd.merge(
            df_prim[["subject_id", "macro_f1"]],
            df_ref[["subject_id", "macro_f1"]],
            on="subject_id",
            suffixes=("_prim", "_ref"),
        )
        assert len(merged) == 47, f"Subject alignment mismatch for {name}: {len(merged)}"
        diff = (merged["macro_f1_prim"].values - merged["macro_f1_ref"].values) * 100.0
        ci_low, ci_high = participant_bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=42)
        _, p_wil = stats.wilcoxon(diff) if np.any(diff != 0) else (0.0, 1.0)
        sd = float(np.std(diff, ddof=1))
        dz = float(np.mean(diff) / sd) if sd > 1e-12 else 0.0
        raw_ps.append(float(p_wil))
        comparisons.append({
            "reference": name,
            "primary_mean_pct": float(merged["macro_f1_prim"].mean() * 100.0),
            "ref_mean_pct": float(merged["macro_f1_ref"].mean() * 100.0),
            "delta_pp": float(np.mean(diff)),
            "ci95_low_pp": float(ci_low),
            "ci95_high_pp": float(ci_high),
            "wilcoxon_p": float(p_wil),
            "cohens_dz": dz,
            "wins": int(np.sum(diff > 0)),
            "losses": int(np.sum(diff < 0)),
            "ties": int(np.sum(diff == 0)),
        })

    if comparisons:
        adj_ps = holm_bonferroni(raw_ps)
        for comp, p_adj in zip(comparisons, adj_ps):
            comp["wilcoxon_p_holm"] = float(p_adj)

    out = {"primary_arm": primary_arm, "n_subjects": 47, "comparisons": comparisons}
    out_path = os.path.join(exp029_dir, f"{primary_arm}_bidsleep_paired_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def compute_int8_profile(arm: str = "full_v5") -> Dict[str, Any]:
    """Computes exact Dynamic INT8 parameter accounting and 1T/4T CPU latency."""
    model_fp32 = build_exp029_model(arm).cpu().eval()
    n_params = sum(p.numel() for p in model_fp32.parameters())

    quantizable_params = 0
    for mod in model_fp32.modules():
        if isinstance(mod, (nn.Linear, nn.GRU)):
            quantizable_params += sum(p.numel() for p in mod.parameters())
    fp32_only_params = n_params - quantizable_params
    in_mem_kib = (quantizable_params * 1 + fp32_only_params * 4) / 1024.0

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

    lat_fp32_1t = bench_latency(model_fp32, 1)
    lat_int8_1t = bench_latency(model_int8, 1)
    lat_int8_4t = bench_latency(model_int8, 4)

    return {
        "arm": arm,
        "total_params": n_params,
        "quantizable_params": quantizable_params,
        "fp32_only_params": fp32_only_params,
        "quantized_ratio": float(quantizable_params / n_params),
        "int8_in_memory_kib": float(in_mem_kib),
        "fp32_serialized_kib": float(fp32_disk_kib),
        "int8_serialized_kib": float(int8_disk_kib),
        "cpu_latency_1t_fp32_ms": lat_fp32_1t,
        "cpu_latency_1t_int8_ms": lat_int8_1t,
        "cpu_latency_4t_int8_ms": lat_int8_4t,
    }


def evaluate_w0b_int8_oof(
    exp029_dir: str,
    arm: str = "full_v5",
    cache_dir: str = CACHE_DIR_DEFAULT,
    manifests_dir: str = MANIFESTS_DIR_DEFAULT,
    authoritative_csv: str = AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
) -> Dict[str, Any]:
    """Evaluates FP32 vs Dynamic INT8 OOF Macro-F1 parity across all 47 subjects."""
    profile = compute_int8_profile(arm)
    manifests = verify_manifests_and_splits(manifests_dir, authoritative_csv)
    records_by_sub = load_all_records_by_subject(cache_dir)

    fp32_sub_scores: Dict[str, List[float]] = {}
    int8_sub_scores: Dict[str, List[float]] = {}

    for fold in FOLDS:
        test_subs = manifests[fold]["outer_test_subjects"]
        test_records = [r for s in test_subs for r in records_by_sub[s]]
        for seed in SEEDS:
            ckpt = os.path.join(exp029_dir, "checkpoints", arm, f"{arm}_L1200_fold{fold}_seed{seed}.pt")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Checkpoint missing for W0-B eval: {ckpt}")
            m_fp32 = build_exp029_model(arm).cpu().eval()
            m_fp32.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            m_int8 = quantization.quantize_dynamic(
                copy.deepcopy(m_fp32), {nn.Linear, nn.GRU}, dtype=torch.qint8
            ).eval()

            p_fp32 = LogitStitchingSlidingWindowPredictor(m_fp32, context_len=1200, device=torch.device("cpu"))
            p_int8 = LogitStitchingSlidingWindowPredictor(m_int8, context_len=1200, device=torch.device("cpu"))

            for mode_name, pred, store in [("fp32", p_fp32, fp32_sub_scores), ("int8", p_int8, int8_sub_scores)]:
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
    out_path = os.path.join(exp029_dir, f"{arm}_w0b_int8_profile.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


@torch.no_grad()
def evaluate_sleepaccel_isolated(
    exp029_dir: str,
    arm: str = "full_v5",
    device_str: str = "cuda:0",
) -> Dict[str, Any]:
    """
    Runs zero-shot SleepAccel evaluation (EXP-027/028) with DampedEMCalibrator
    off vs. on, writing isolated bundles to experiments/EXP-029/sleepaccel_cache/predictions/.
    """
    prep_dir = os.path.join(EXP027_DIR, "cache", "preprocessed")
    pred_dir = os.path.join(exp029_dir, "sleepaccel_cache", "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    all_records = load_preprocessed_sleepaccel(prep_dir, qc_only=False)
    evaluator = SleepEvaluator()

    # Load per-fold pi_src from run_results JSONs
    probs_raw: Dict[str, List[np.ndarray]] = {r["subject"]: [] for r in all_records}
    probs_em: Dict[str, List[np.ndarray]] = {r["subject"]: [] for r in all_records}
    pi_src_list: List[np.ndarray] = []

    for fold in FOLDS:
        for seed in SEEDS:
            ckpt = os.path.join(exp029_dir, "checkpoints", arm, f"{arm}_L1200_fold{fold}_seed{seed}.pt")
            rjson = os.path.join(exp029_dir, "run_results", arm, f"{arm}_fold{fold}_seed{seed}.json")
            with open(rjson, "r", encoding="utf-8") as f:
                pi_src = np.array(json.load(f)["pi_src_inner_val"], dtype=np.float64)
            pi_src_list.append(pi_src)

            model = build_exp029_model(arm).to(device).eval()
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))

            pred_off = LogitStitchingSlidingWindowPredictor(model=model, context_len=1200, device=device)
            cal = DampedEMCalibrator(pi_src=pi_src, alpha=0.5, n_iter=3, temperature=1.0)

            for rec in all_records:
                out = pred_off.predict_night(rec)
                p_off = out["probs"].astype(np.float32)
                p_on = cal.calibrate(p_off).astype(np.float32)
                probs_raw[rec["subject"]].append(p_off)
                probs_em[rec["subject"]].append(p_on)

    mean_pi_src = np.mean(np.stack(pi_src_list, axis=0), axis=0)
    ens_cal = DampedEMCalibrator(pi_src=mean_pi_src, alpha=0.5, n_iter=3, temperature=1.0)

    summary_rows: List[Dict[str, Any]] = []
    ba_rows: List[Dict[str, Any]] = []

    for em_mode, store in [("em_off", probs_raw), ("em_on", probs_em)]:
        for rec in all_records:
            sub = rec["subject"]
            y_true = rec["y"]
            ckpt_probs = np.stack(store[sub], axis=0)  # (15, N, 4)
            ckpt_preds = np.argmax(ckpt_probs, axis=-1)
            ens_probs = np.mean(
                np.stack(probs_raw[sub], axis=0), axis=0
            )
            if em_mode == "em_on":
                ens_probs = ens_cal.calibrate(ens_probs).astype(np.float32)
            ens_pred = np.argmax(ens_probs, axis=-1)

            np.savez_compressed(
                os.path.join(pred_dir, f"{sub}__{arm}_{em_mode}.npz"),
                y_true=y_true,
                ckpt_preds=ckpt_preds,
                ens_probs=ens_probs,
                qc_pass=rec["qc_pass"],
            )

            ckpt_m = [extract_8_metrics(evaluator, y_true, ckpt_preds[c]) for c in range(15)]
            mean_m = {k: float(np.mean([m[k] for m in ckpt_m])) for k in ckpt_m[0]}
            ens_m = extract_8_metrics(evaluator, y_true, ens_pred)
            summary_rows.append({"subject": sub, "qc_pass": rec["qc_pass"], "em_mode": em_mode, "eval_mode": "mean_single_model", **mean_m})
            summary_rows.append({"subject": sub, "qc_pass": rec["qc_pass"], "em_mode": em_mode, "eval_mode": "prob_ensemble", **ens_m})

            valid = (y_true >= 0) & (y_true < 4)
            yt_v = y_true[valid]
            ref_ba = ClinicalHypnogramEvaluator.compute_night_metrics(yt_v)
            ckpt_ba = [ClinicalHypnogramEvaluator.compute_night_metrics(ckpt_preds[c][valid]) for c in range(15)]
            ens_ba = ClinicalHypnogramEvaluator.compute_night_metrics(ens_pred[valid])
            for k in SUMMARY_METRICS:
                ba_rows.append({
                    "subject": sub, "qc_pass": rec["qc_pass"], "em_mode": em_mode,
                    "eval_mode": "mean_single_model", "metric": k,
                    "ref": float(ref_ba[k]), "pred": float(np.mean([m[k] for m in ckpt_ba])),
                })
                ba_rows.append({
                    "subject": sub, "qc_pass": rec["qc_pass"], "em_mode": em_mode,
                    "eval_mode": "prob_ensemble", "metric": k,
                    "ref": float(ref_ba[k]), "pred": float(ens_ba[k]),
                })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(exp029_dir, f"{arm}_sleepaccel_metrics.csv"), index=False)

    df_ba = pd.DataFrame(ba_rows)
    ba_summary: List[Dict[str, Any]] = []
    for em_mode in ["em_off", "em_on"]:
        for eval_mode in ["mean_single_model", "prob_ensemble"]:
            for k in SUMMARY_METRICS:
                sub_df = df_ba[
                    (df_ba["qc_pass"]) & (df_ba["em_mode"] == em_mode)
                    & (df_ba["eval_mode"] == eval_mode) & (df_ba["metric"] == k)
                ]
                ba_stats = compute_bland_altman_stats(sub_df["ref"].values, sub_df["pred"].values)
                ba_summary.append({"em_mode": em_mode, "eval_mode": eval_mode, "metric": k, **ba_stats})

    pd.DataFrame(ba_summary).to_csv(os.path.join(exp029_dir, f"{arm}_sleepaccel_bland_altman.csv"), index=False)

    agg = {}
    for em_mode in ["em_off", "em_on"]:
        for eval_mode in ["mean_single_model", "prob_ensemble"]:
            sub_df = df_sum[(df_sum["qc_pass"]) & (df_sum["em_mode"] == em_mode) & (df_sum["eval_mode"] == eval_mode)]
            agg[f"{em_mode}__{eval_mode}__qc_clean_macro_f1"] = float(sub_df["macro_f1"].mean() * 100.0)
    with open(os.path.join(exp029_dir, f"{arm}_sleepaccel_summary.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    return agg


def evaluate_exp020_gate_tau(
    exp029_dir: str,
    arm: str = "full_v5",
    device_str: str = "cuda:0",
    tau_star: float = 0.01,
    lr: float = 1e-4,
    epochs: int = 20,
    cache_dir: str = CACHE_DIR_DEFAULT,
    manifests_dir: str = MANIFESTS_DIR_DEFAULT,
    authoritative_csv: str = AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
) -> Dict[str, Any]:
    """
    Runs EXP-020 FT-2 / GATE_tau (tau*=0.01) safety check on N=46 multi-night
    subjects (adapting classifier_head + temp_norm + coarse_gru/temporal_gru).
    """
    manifests = verify_manifests_and_splits(manifests_dir, authoritative_csv)
    records_by_sub = load_all_records_by_subject(cache_dir)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    scope_regex = r"(classifier_head|temp_norm|coarse_gru|temporal_gru)"
    pattern = re.compile(scope_regex)

    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        test_subs = manifests[fold]["outer_test_subjects"]
        for sub in test_subs:
            recs = sorted(records_by_sub[sub], key=lambda r: int(r["night"]))
            if len(recs) < 3:
                continue  # N=46 multi-night cohort requires >= 2 cal + >= 1 eval night
            cal_recs = recs[:2]
            eval_recs = recs[2:]

            all_y = np.concatenate([r["y"] for r in cal_recs])
            valid_y = all_y[(all_y >= 0) & (all_y < 4)]
            min_frac = min(float(np.sum(valid_y == c)) / max(len(valid_y), 1) for c in range(4))
            abstained = min_frac < tau_star

            zs_scores, ft2_scores, gate_scores = [], [], []
            for seed in SEEDS:
                ckpt = os.path.join(exp029_dir, "checkpoints", arm, f"{arm}_L1200_fold{fold}_seed{seed}.pt")
                base_state = torch.load(ckpt, map_location=device, weights_only=True)

                # ZS
                m_zs = build_exp029_model(arm).to(device).eval()
                m_zs.load_state_dict(base_state)
                p_zs = LogitStitchingSlidingWindowPredictor(m_zs, context_len=1200, device=device)
                yt = np.concatenate([r["y"] for r in eval_recs])
                yp_zs = np.concatenate([p_zs.predict_night(r)["y_pred"] for r in eval_recs])
                f1_zs = compute_epoch_metrics(yt, yp_zs)["macro_f1"]
                zs_scores.append(f1_zs)

                # FT-2
                set_deterministic_seed(seed)
                m_ft = build_exp029_model(arm).to(device)
                m_ft.load_state_dict(base_state)
                for name, p in m_ft.named_parameters():
                    p.requires_grad = bool(pattern.search(name))
                trainable = [p for p in m_ft.parameters() if p.requires_grad]
                opt = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.999))
                batch = prepare_calibration_batch(cal_recs, context_len=1200, device=str(device))
                m_ft.train()
                for _ in range(epochs):
                    opt.zero_grad()
                    out = m_ft(batch["x_ihr"], batch["x_acc"], batch["x_meta"], seq_lens=batch["seq_len"])
                    loss = compute_stage_loss_fp32(m_ft, out["logits"], batch["y"], batch["mask"])
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                    opt.step()
                m_ft.eval()
                p_ft = LogitStitchingSlidingWindowPredictor(m_ft, context_len=1200, device=device)
                yp_ft = np.concatenate([p_ft.predict_night(r)["y_pred"] for r in eval_recs])
                f1_ft = compute_epoch_metrics(yt, yp_ft)["macro_f1"]
                ft2_scores.append(f1_ft)
                gate_scores.append(f1_zs if abstained else f1_ft)

            rows.append({
                "subject_id": sub,
                "fold": fold,
                "min_stage_frac": min_frac,
                "abstained": abstained,
                "zs_macro_f1": float(np.mean(zs_scores)),
                "ft2_macro_f1": float(np.mean(ft2_scores)),
                "gate_tau_macro_f1": float(np.mean(gate_scores)),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp029_dir, f"{arm}_exp020_gate_tau_metrics.csv"), index=False)
    summary = {
        "arm": arm,
        "n_subjects": int(len(df)),
        "tau_star": tau_star,
        "n_abstained": int(df["abstained"].sum()),
        "zs_mean_pct": float(df["zs_macro_f1"].mean() * 100.0),
        "ft2_mean_pct": float(df["ft2_macro_f1"].mean() * 100.0),
        "gate_tau_mean_pct": float(df["gate_tau_macro_f1"].mean() * 100.0),
        "gate_vs_zs_pp": float((df["gate_tau_macro_f1"] - df["zs_macro_f1"]).mean() * 100.0),
    }
    with open(os.path.join(exp029_dir, f"{arm}_exp020_gate_tau_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-029 Downstream Evaluation Suite")
    parser.add_argument("--arm", type=str, default="full_v5", choices=sorted(ARM_ALIASES.keys()))
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["bidsleep", "int8", "sleepaccel", "gate_tau"],
        choices=["bidsleep", "int8", "sleepaccel", "gate_tau"],
    )
    parser.add_argument("--exp029-dir", type=str, default=EXP029_DIR)
    args = parser.parse_args()
    arm = ARM_ALIASES[args.arm]

    if "bidsleep" in args.tasks:
        print(json.dumps(evaluate_bidsleep_paired(args.exp029_dir, arm), indent=2))
    if "int8" in args.tasks:
        print(json.dumps(evaluate_w0b_int8_oof(args.exp029_dir, arm), indent=2))
    if "sleepaccel" in args.tasks:
        print(json.dumps(evaluate_sleepaccel_isolated(args.exp029_dir, arm, args.device), indent=2))
    if "gate_tau" in args.tasks:
        print(json.dumps(evaluate_exp020_gate_tau(args.exp029_dir, arm, args.device), indent=2))


if __name__ == "__main__":
    main()
