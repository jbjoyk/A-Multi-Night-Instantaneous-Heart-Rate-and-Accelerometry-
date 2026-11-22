#!/usr/bin/env python3
"""
run_exp028_sleepaccel_bland_altman.py
=====================================
EXP-028 External PSG Bland-Altman Agreement Analysis on PhysioNet SleepAccel v1.0.0.
Reuses evaluator.py::ClinicalHypnogramEvaluator.compute_night_metrics(), persists raw
subject-night paired tuples (remediating FIND-B-02), and computes Bland-Altman (1999)
SE(LoA), 95% CIs, proportional bias slope, and Shapiro-Wilk normality fallback.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluator import ClinicalHypnogramEvaluator

SUMMARY_METRICS = ["tst_min", "sol_min", "waso_min", "sleep_efficiency_pct", "deep_pct_tst", "rem_pct_tst"]
DEFAULT_EXP027_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-027")
DEFAULT_EXP028_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-028")


def compute_bland_altman_stats(ref_vals: np.ndarray, pred_vals: np.ndarray) -> Dict[str, Any]:
    """
    Computes Bland-Altman (1999) agreement statistics dynamically for n_obs = len(ref_vals),
    df_denom = n_obs - 1, matching research_engine/bidsleep_within_cohort_agreement_analysis.py:L64-81,
    plus OLS proportional bias slope and Shapiro-Wilk normality fallback.
    """
    ref = np.asarray(ref_vals, dtype=np.float64)
    pred = np.asarray(pred_vals, dtype=np.float64)
    n_obs = len(ref)
    assert n_obs >= 3, f"Need n_obs >= 3 for Bland-Altman, got {n_obs}"
    df_denom = n_obs - 1
    t_crit = float(stats.t.ppf(0.975, df=df_denom))

    diff = pred - ref
    mean_pair = 0.5 * (ref + pred)
    bias = float(np.mean(diff))
    sd_diff = float(np.std(diff, ddof=1))
    se_bias = float(sd_diff / np.sqrt(n_obs))
    ci_bias_low = float(bias - t_crit * se_bias)
    ci_bias_high = float(bias + t_crit * se_bias)

    loa_low = float(bias - 1.96 * sd_diff)
    loa_high = float(bias + 1.96 * sd_diff)
    var_loa = (sd_diff ** 2) * (1.0 / n_obs + (1.96 ** 2) / (2.0 * df_denom))
    se_loa = float(np.sqrt(var_loa))
    loa_low_ci_l = float(loa_low - t_crit * se_loa)
    loa_low_ci_h = float(loa_low + t_crit * se_loa)
    loa_high_ci_l = float(loa_high - t_crit * se_loa)
    loa_high_ci_h = float(loa_high + t_crit * se_loa)

    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    pearson_r = float(stats.pearsonr(ref, pred)[0]) if (np.std(ref) > 0 and np.std(pred) > 0) else float("nan")
    reg = stats.linregress(mean_pair, diff) if np.std(mean_pair) > 0 else None
    prop_slope = float(reg.slope) if reg is not None else float("nan")
    prop_p = float(reg.pvalue) if reg is not None else float("nan")

    sw_stat, sw_p = stats.shapiro(diff) if np.std(diff) > 0 else (1.0, 1.0)
    nonparam_median = float(np.median(diff))
    nonparam_loa_low, nonparam_loa_high = [float(x) for x in np.percentile(diff, [2.5, 97.5])]

    return {
        "n_obs": n_obs,
        "df_denom": df_denom,
        "ref_mean": float(np.mean(ref)),
        "pred_mean": float(np.mean(pred)),
        "bias": bias,
        "ci_bias_low": ci_bias_low,
        "ci_bias_high": ci_bias_high,
        "sd_diff": sd_diff,
        "loa_low": loa_low,
        "loa_high": loa_high,
        "se_loa": se_loa,
        "loa_low_ci_low": loa_low_ci_l,
        "loa_low_ci_high": loa_low_ci_h,
        "loa_high_ci_low": loa_high_ci_l,
        "loa_high_ci_high": loa_high_ci_h,
        "mae": mae,
        "rmse": rmse,
        "pearson_r": pearson_r,
        "prop_bias_slope": prop_slope,
        "prop_bias_p": prop_p,
        "shapiro_w": float(sw_stat),
        "shapiro_p": float(sw_p),
        "non_normal_flag": bool(sw_p < 0.05),
        "nonparam_median_diff": nonparam_median,
        "nonparam_loa_low": nonparam_loa_low,
        "nonparam_loa_high": nonparam_loa_high,
    }


def run_exp028(
    exp027_dir: str = DEFAULT_EXP027_DIR,
    exp028_dir: str = DEFAULT_EXP028_DIR,
) -> Dict[str, Any]:
    pred_dir = os.path.join(exp027_dir, "cache", "predictions")
    files = sorted(glob.glob(os.path.join(pred_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No EXP-027 prediction bundles found in {pred_dir}")

    os.makedirs(exp028_dir, exist_ok=True)
    raw_rows: List[Dict[str, Any]] = []

    for fp in files:
        stem = os.path.basename(fp)[:-4]
        sub, arm = stem.split("__")
        d = np.load(fp)
        y_true_raw = d["y_true"]
        ckpt_preds_raw = d["ckpt_preds"]  # (15, N)
        ens_probs_raw = d["ens_probs"]    # (N, 4)
        qc_pass = bool(d["qc_pass"])

        # Mask unscored PSG epochs (-100) consistently before ClinicalHypnogramEvaluator
        valid = (y_true_raw >= 0) & (y_true_raw < 4)
        yt = y_true_raw[valid]
        ref_m = ClinicalHypnogramEvaluator.compute_night_metrics(yt)

        # (a) mean_single_model: average 15 per-checkpoint summary metrics
        ckpt_m_list = [ClinicalHypnogramEvaluator.compute_night_metrics(ckpt_preds_raw[c][valid]) for c in range(15)]
        for k in SUMMARY_METRICS:
            pred_mean_val = float(np.mean([m[k] for m in ckpt_m_list]))
            ref_val = float(ref_m[k])
            raw_rows.append({
                "subject_id": sub,
                "qc_pass": qc_pass,
                "arm": arm,
                "eval_mode": "mean_single_model",
                "metric": k,
                "ref_val": ref_val,
                "pred_val": pred_mean_val,
                "diff": pred_mean_val - ref_val,
                "mean": 0.5 * (ref_val + pred_mean_val),
            })

        # (b) prob_ensemble
        ens_pred = np.argmax(ens_probs_raw, axis=-1)[valid]
        ens_m = ClinicalHypnogramEvaluator.compute_night_metrics(ens_pred)
        for k in SUMMARY_METRICS:
            pred_val = float(ens_m[k])
            ref_val = float(ref_m[k])
            raw_rows.append({
                "subject_id": sub,
                "qc_pass": qc_pass,
                "arm": arm,
                "eval_mode": "prob_ensemble",
                "metric": k,
                "ref_val": ref_val,
                "pred_val": pred_val,
                "diff": pred_val - ref_val,
                "mean": 0.5 * (ref_val + pred_val),
            })

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = os.path.join(exp028_dir, "exp028_sleepaccel_raw_paired_tuples.csv")
    raw_df.to_csv(raw_csv, index=False)

    ba_rows: List[Dict[str, Any]] = []
    for cohort_name, df_sub in [("qc_clean", raw_df[raw_df["qc_pass"]]), ("all_n31_sensitivity", raw_df)]:
        if len(df_sub) == 0:
            continue
        for (arm, eval_mode, metric), grp in df_sub.groupby(["arm", "eval_mode", "metric"], sort=True):
            stats_dict = compute_bland_altman_stats(grp["ref_val"].values, grp["pred_val"].values)
            ba_rows.append({"cohort": cohort_name, "arm": arm, "eval_mode": eval_mode, "metric": metric, **stats_dict})

    ba_df = pd.DataFrame(ba_rows)
    ba_csv = os.path.join(exp028_dir, "exp028_sleepaccel_bland_altman_summary.csv")
    ba_df.to_csv(ba_csv, index=False)
    return {"raw_tuples_csv": raw_csv, "summary_csv": ba_csv, "n_raw_rows": len(raw_df)}


def main():
    parser = argparse.ArgumentParser(description="EXP-028 External PSG Bland-Altman Agreement")
    parser.add_argument("--exp027-dir", default=DEFAULT_EXP027_DIR)
    parser.add_argument("--exp028-dir", default=DEFAULT_EXP028_DIR)
    args = parser.parse_args()
    out = run_exp028(exp027_dir=args.exp027_dir, exp028_dir=args.exp028_dir)
    print(f"EXP-028 complete: {out['raw_tuples_csv']} ({out['n_raw_rows']} raw paired tuples)")


if __name__ == "__main__":
    main()
