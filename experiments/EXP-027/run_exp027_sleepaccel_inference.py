#!/usr/bin/env python3
"""
run_exp027_sleepaccel_inference.py
==================================
Strictly Zero-Shot External Validation on PhysioNet SleepAccel v1.0.0 (cuda:1).
Runs all 15 frozen Model A checkpoints AND all 15 EXP-021 Official SLAMSS-IFS
checkpoints under torch.no_grad(), reporting 15-checkpoint mean (primary) and
15-checkpoint probability ensemble (secondary), and saving handoff bundles for EXP-028.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [
    REPO_ROOT,
    os.path.join(REPO_ROOT, "experiments", "EXP-018"),
    os.path.join(REPO_ROOT, "experiments", "EXP-021"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluator import SleepEvaluator
from exp013_models import ConventionalCNNBiGRU
from exp018_inference import SlidingWindowPredictor
from analyze_tsk08_paired_benchmark import participant_bootstrap_ci
from official_slamss_ifs_adapter import (
    OfficialSLAMSSIFSWrapper,
    SLAMSSSlidingWindowPredictor,
)

FOLDS = [1, 2, 3, 4, 5]
SEEDS = [42, 123, 2024]
DEFAULT_EXP027_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-027")


def load_preprocessed_sleepaccel(prep_dir: str, qc_only: bool = True) -> List[Dict[str, Any]]:
    files = sorted(glob.glob(os.path.join(prep_dir, "*.npz")))
    records = []
    for fp in files:
        sub = os.path.basename(fp)[:-4]
        d = np.load(fp)
        qc_pass = bool(d["qc_pass"])
        if qc_only and not qc_pass:
            continue
        records.append({
            "subject": sub,
            "night": "1",
            "ihr": d["ihr"],
            "accel": d["accel"],
            "freq": d["freq"],
            "time": d["time"],
            "y": d["y"],
            "qc_pass": qc_pass,
        })
    return records


def extract_8_metrics(evaluator: SleepEvaluator, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    res = evaluator.evaluate_predictions(y_true=y_true, y_pred=y_pred)
    return {
        "macro_f1": res["primary"]["macro_f1"],
        "accuracy": res["secondary"]["accuracy"],
        "weighted_f1": res["secondary"]["weighted_f1"],
        "kappa": res["secondary"]["kappa"],
        "wake_f1": res["secondary"]["wake_f1"],
        "light_f1": res["secondary"]["light_f1"],
        "deep_f1": res["secondary"]["deep_f1"],
        "rem_f1": res["secondary"]["rem_f1"],
    }


def compute_paired_stats(a_vals: np.ndarray, s_vals: np.ndarray, scale: float, seed: int = 20261122) -> Dict[str, Any]:
    diff = (a_vals - s_vals) * scale
    ci_low, ci_high = participant_bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=seed)
    sd = float(np.std(diff, ddof=1))
    dz = float(np.mean(diff) / sd) if sd > 1e-12 else 0.0
    _, p_wil = stats.wilcoxon(diff) if np.any(diff != 0) else (0.0, 1.0)
    return {
        "model_a_mean": float(np.mean(a_vals) * scale),
        "slamss_mean": float(np.mean(s_vals) * scale),
        "delta_mean": float(np.mean(diff)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "wilcoxon_p": float(p_wil),
        "cohens_dz": dz,
        "wins": int(np.sum(diff > 0)),
        "losses": int(np.sum(diff < 0)),
        "ties": int(np.sum(diff == 0)),
    }


@torch.no_grad()
def run_exp027(
    exp027_dir: str = DEFAULT_EXP027_DIR,
    device_str: str = "cuda:1",
    seed_boot: int = 20261122,
) -> Dict[str, Any]:
    gate_path = os.path.join(exp027_dir, "preflight_distribution_gate.json")
    if not os.path.exists(gate_path):
        raise FileNotFoundError(f"Pre-flight gate JSON missing: {gate_path}. Run prepare_sleepaccel.py first.")
    with open(gate_path, "r", encoding="utf-8") as f:
        gate = json.load(f)
    if not gate.get("passed", False):
        raise RuntimeError("Pre-flight distribution gate did not pass.")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    prep_dir = os.path.join(exp027_dir, "cache", "preprocessed")
    pred_dir = os.path.join(exp027_dir, "cache", "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    all_records = load_preprocessed_sleepaccel(prep_dir, qc_only=False)
    evaluator = SleepEvaluator()

    # Storage: arm -> subject -> list of 15 probs (N_i, 4)
    probs_store: Dict[str, Dict[str, List[np.ndarray]]] = {
        "ModelA": {r["subject"]: [] for r in all_records},
        "SLAMSS_IFS": {r["subject"]: [] for r in all_records},
    }

    # 1. Model A (15 checkpoints)
    for fold in FOLDS:
        for seed in SEEDS:
            ckpt = os.path.join(REPO_ROOT, "experiments", "EXP-018", "checkpoints", f"modelA_L1200_fold{fold}_seed{seed}.pt")
            model = ConventionalCNNBiGRU(d_model=64, d_temp=96, n_classes=4, n_meta=5, dropout=0.2, use_rcc_loss=True)
            model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            predictor = SlidingWindowPredictor(model=model, context_len=1200, neutralize_circadian=True, device=device)
            for rec in all_records:
                out = predictor.predict_night(rec)
                probs_store["ModelA"][rec["subject"]].append(out["probs"].astype(np.float32))

    # 2. Official SLAMSS-IFS (15 checkpoints)
    for fold in FOLDS:
        for seed in SEEDS:
            ckpt = os.path.join(REPO_ROOT, "experiments", "EXP-021", "checkpoints", f"slamss_ifs_official_L1200_fold{fold}_seed{seed}.pt")
            model = OfficialSLAMSSIFSWrapper(device=device)
            state = torch.load(ckpt, map_location=device, weights_only=True)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            model.load_state_dict(state)
            predictor = SLAMSSSlidingWindowPredictor(model=model, context_len=1200, neutralize_circadian=True, device=device)
            outs = predictor.predict_nights_batched(all_records, batch_size=16)
            for rec, out in zip(all_records, outs):
                probs_store["SLAMSS_IFS"][rec["subject"]].append(out["probs"].astype(np.float32))

    # 3. Persist handoff bundles & compute per-subject metrics
    subj_rows: List[Dict[str, Any]] = []
    metric_names = ["macro_f1", "accuracy", "weighted_f1", "kappa", "wake_f1", "light_f1", "deep_f1", "rem_f1"]

    for rec in all_records:
        sub = rec["subject"]
        y_true = rec["y"]
        for arm in ["ModelA", "SLAMSS_IFS"]:
            p_stack = np.stack(probs_store[arm][sub], axis=0)  # (15, N_i, 4)
            ckpt_preds = np.argmax(p_stack, axis=-1).astype(np.int8)
            ens_probs = np.mean(p_stack, axis=0).astype(np.float32)
            ens_pred = np.argmax(ens_probs, axis=-1).astype(np.int8)

            np.savez_compressed(
                os.path.join(pred_dir, f"{sub}__{arm}.npz"),
                y_true=y_true.astype(np.int8),
                ckpt_preds=ckpt_preds,
                ens_probs=ens_probs,
                qc_pass=np.array(rec["qc_pass"]),
            )

            # Mean single-model (primary)
            m_list = [extract_8_metrics(evaluator, y_true, ckpt_preds[c]) for c in range(15)]
            mean_m = {k: float(np.mean([m[k] for m in m_list])) for k in metric_names}
            ens_m = extract_8_metrics(evaluator, y_true, ens_pred)

            subj_rows.append({"subject_id": sub, "qc_pass": rec["qc_pass"], "arm": arm, "eval_mode": "mean_single_model", **mean_m})
            subj_rows.append({"subject_id": sub, "qc_pass": rec["qc_pass"], "arm": arm, "eval_mode": "prob_ensemble", **ens_m})

    subj_df = pd.DataFrame(subj_rows)
    subj_csv = os.path.join(exp027_dir, "exp027_per_subject_metrics.csv")
    subj_df.to_csv(subj_csv, index=False)

    # 4. Paired comparisons (QC-clean primary + all-N=31 sensitivity)
    summary: Dict[str, Any] = {"device": str(device), "cohorts": {}}
    for cohort_name, df_sub in [("qc_clean", subj_df[subj_df["qc_pass"]]), ("all_n31_sensitivity", subj_df)]:
        cohort_res: Dict[str, Any] = {}
        for eval_mode in ["mean_single_model", "prob_ensemble"]:
            sub_m = df_sub[df_sub["eval_mode"] == eval_mode]
            a_df = sub_m[sub_m["arm"] == "ModelA"].sort_values("subject_id")
            s_df = sub_m[sub_m["arm"] == "SLAMSS_IFS"].sort_values("subject_id")
            mode_res = {"n_obs": len(a_df), "metrics": {}}
            for m in metric_names:
                scale = 1.0 if m == "kappa" else 100.0
                mode_res["metrics"][m] = compute_paired_stats(a_df[m].values, s_df[m].values, scale=scale, seed=seed_boot)
            cohort_res[eval_mode] = mode_res
        summary["cohorts"][cohort_name] = cohort_res

    sum_json = os.path.join(exp027_dir, "exp027_summary.json")
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="EXP-027 Zero-Shot SleepAccel Inference")
    parser.add_argument("--exp027-dir", default=DEFAULT_EXP027_DIR)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    res = run_exp027(exp027_dir=args.exp027_dir, device_str=args.device)
    print(f"EXP-027 complete. QC-clean n_obs={res['cohorts']['qc_clean']['mean_single_model']['n_obs']}")


if __name__ == "__main__":
    main()
