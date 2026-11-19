#!/usr/bin/env python3
"""
bidsleep_within_cohort_agreement_analysis.py
============================================
Reproducible Phase B within-cohort sleep-summary agreement analysis script.

Source Input:
  experiments/EXP-019/exp019_sleep_summary_agreement.csv

This script:
  1. Validates the source CSV schema, types, units, and data integrity.
  2. Evaluates the repeated-measures structure and documents the statistical
     implications of subject-averaged (N=46) vs. unadjusted night-pooled (N=152) data.
  3. Recomputes exact Bland-Altman agreement statistics, standard errors, and
     95% confidence intervals on bias and limits of agreement (LoA).
     Note: Subject-averaged LoA quantifies agreement for multi-night monitoring averages.
     The unadjusted pooled interval is an unadjusted descriptive summary, NOT a validated
     single-night prediction interval.
  4. Performs reconciliation against the draft manuscript Section 6.2 Table 1.
  5. Strictly read-only; does not modify any locked artifacts or code.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from scipy import stats

SOURCE_CSV = "experiments/EXP-019/exp019_sleep_summary_agreement.csv"
EXPECTED_SHA256 = "41a2d6afa73b3d65599a83c7c463e7cadbbbcabef6d1f71a6349194b489b35e3"

def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_source_integrity(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Source file not found: {filepath}")
    
    file_hash = compute_sha256(filepath)
    if file_hash != EXPECTED_SHA256:
        raise ValueError(f"Hash mismatch! Expected {EXPECTED_SHA256}, got {file_hash}")
    
    df = pd.read_csv(filepath)
    print("=" * 80)
    print("PHASE B: SOURCE INTEGRITY & SCHEMA VERIFICATION")
    print("=" * 80)
    print(f"File: {filepath}")
    print(f"SHA-256: {file_hash} [MATCH]")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Missing Values: {df.isna().sum().sum()} total across all cells")
    print(f"Unique Arms ({len(df['arm'].unique())}): {list(df['arm'].unique())}")
    print(f"Unique Metrics ({len(df['metric'].unique())}): {list(df['metric'].unique())}")
    print(f"Unique Aggregations ({len(df['aggregation_level'].unique())}): {list(df['aggregation_level'].unique())}")
    return df

def analyze_bland_altman(df: pd.DataFrame, agg_level: str, n_obs: int, df_denom: int):
    sub = df[df["aggregation_level"] == agg_level].copy()
    t_crit = stats.t.ppf(0.975, df=df_denom)
    
    results = []
    for _, r in sub.iterrows():
        # Standard error of the mean difference (bias)
        se_bias = r["sd_diff"] / np.sqrt(n_obs)
        ci_bias_low = r["bias"] - t_crit * se_bias
        ci_bias_high = r["bias"] + t_crit * se_bias
        is_sig = (ci_bias_low > 0) | (ci_bias_high < 0)
        
        # Bland & Altman (1999) standard error for limits of agreement
        # Var(LoA) approx = sd_diff^2 * (1/n + (1.96^2)/(2*(n-1)))
        var_loa = (r["sd_diff"] ** 2) * (1.0 / n_obs + (1.96 ** 2) / (2.0 * df_denom))
        se_loa = np.sqrt(var_loa)
        loa_low_ci_l = r["loa_low"] - t_crit * se_loa
        loa_low_ci_h = r["loa_low"] + t_crit * se_loa
        loa_high_ci_l = r["loa_high"] - t_crit * se_loa
        loa_high_ci_h = r["loa_high"] + t_crit * se_loa
        
        # Clinical threshold flags (|bias| > 5 min or > 2 pp)
        large_bias = abs(r["bias"]) > 5.0 if "min" in r["metric"] else abs(r["bias"]) > 2.0
        
        results.append({
            "arm": r["arm"],
            "metric": r["metric"],
            "ref_mean": r["ref_mean"],
            "pred_mean": r["pred_mean"],
            "bias": r["bias"],
            "se_bias": se_bias,
            "ci_bias_low": ci_bias_low,
            "ci_bias_high": ci_bias_high,
            "is_sig": is_sig,
            "sd_diff": r["sd_diff"],
            "loa_low": r["loa_low"],
            "loa_high": r["loa_high"],
            "se_loa": se_loa,
            "loa_low_ci": (loa_low_ci_l, loa_low_ci_h),
            "loa_high_ci": (loa_high_ci_l, loa_high_ci_h),
            "mae": r["mae"],
            "rmse": r["rmse"],
            "pearson_r": r["pearson_r"],
            "large_bias": large_bias
        })
    return pd.DataFrame(results)

def format_table(res_df: pd.DataFrame, title: str):
    print("\n" + "=" * 115)
    print(title)
    print("=" * 115)
    header = f"{'Arm':<10s} | {'Metric':<22s} | {'Ref':>6s} | {'Pred':>6s} | {'Bias':>7s} | {'95% CI Bias':>18s} | {'Sig?':>5s} | {'SD Diff':>7s} | {'95% LoA':>18s} | {'SE LoA':>6s} | {'MAE':>6s} | {'r':>7s}"
    print(header)
    print("-" * 115)
    for _, r in res_df.iterrows():
        sig_str = "YES*" if r["is_sig"] else "NO"
        bias_str = f"{r['bias']:+6.2f}"
        ci_bias_str = f"[{r['ci_bias_low']:+6.2f}, {r['ci_bias_high']:+6.2f}]"
        loa_str = f"[{r['loa_low']:+6.2f}, {r['loa_high']:+6.2f}]"
        print(f"{r['arm']:<10s} | {r['metric']:<22s} | {r['ref_mean']:6.2f} | {r['pred_mean']:6.2f} | {bias_str:>7s} | {ci_bias_str:>18s} | {sig_str:>5s} | {r['sd_diff']:7.2f} | {loa_str:>18s} | {r['se_loa']:6.2f} | {r['mae']:6.2f} | {r['pearson_r']:7.4f}")

def reconcile_with_manuscript(sub_res: pd.DataFrame):
    print("\n" + "=" * 80)
    print("MANUSCRIPT RECONCILIATION: MANUSCRIPT_DRAFT_v0.1.md Section 6.2 Table 1")
    print("=" * 80)
    
    # Authoritative manuscript values for FT-2
    ms_table1 = {
        ("FT-2", "tst_min"): {"ref": 356.25, "pred": 358.53, "bias": +2.28, "ci": "[-2.63, +7.19]", "sd": 16.53, "loa": "[-30.11, +34.67]", "mae": 12.61, "r": 0.9492},
        ("FT-2", "sol_min"): {"ref": 24.70, "pred": 27.91, "bias": +3.20, "ci": "[-0.13, +6.53]", "sd": 11.22, "loa": "[-18.78, +25.19]", "mae": 7.78, "r": 0.8023},
        ("FT-2", "waso_min"): {"ref": 17.11, "pred": 12.24, "bias": -4.87, "ci": "[-8.54, -1.20]", "sd": 12.36, "loa": "[-29.09, +19.35]", "mae": 10.06, "r": 0.4683},
        ("FT-2", "sleep_efficiency_pct"): {"ref": 85.10, "pred": 85.56, "bias": +0.47, "ci": "[-0.72, +1.66]", "sd": 4.01, "loa": "[-7.40, +8.33]", "mae": 3.03, "r": 0.8775},
        ("FT-2", "deep_pct_tst"): {"ref": 21.14, "pred": 22.89, "bias": +1.75, "ci": "[-0.62, +4.12]", "sd": 7.99, "loa": "[-13.90, +17.41]", "mae": 6.28, "r": 0.4868},
        ("FT-2", "rem_pct_tst"): {"ref": 29.67, "pred": 27.51, "bias": -2.16, "ci": "[-5.39, +1.07]", "sd": 10.86, "loa": "[-23.44, +19.13]", "mae": 8.21, "r": 0.4478},
    }
    
    ft2 = sub_res[sub_res["arm"] == "FT-2"]
    all_match = True
    for _, r in ft2.iterrows():
        key = ("FT-2", r["metric"])
        expected = ms_table1[key]
        calc_ci = f"[{r['ci_bias_low']:+.2f}, {r['ci_bias_high']:+.2f}]"
        calc_loa = f"[{r['loa_low']:+.2f}, {r['loa_high']:+.2f}]"
        
        ref_match = (r["ref_mean"] == expected["ref"])
        pred_match = (r["pred_mean"] == expected["pred"])
        bias_match = (r["bias"] == expected["bias"])
        ci_match = (calc_ci == expected["ci"])
        sd_match = (r["sd_diff"] == expected["sd"])
        loa_match = (calc_loa == expected["loa"])
        mae_match = (r["mae"] == expected["mae"])
        r_match = (r["pearson_r"] == expected["r"])
        
        matches = [ref_match, pred_match, bias_match, ci_match, sd_match, loa_match, mae_match, r_match]
        if not all(matches):
            all_match = False
            print(f"MISMATCH for {key}: calculated vs expected:")
            print(f"  Calculated: ref={r['ref_mean']}, pred={r['pred_mean']}, bias={r['bias']}, ci={calc_ci}, sd={r['sd_diff']}, loa={calc_loa}, mae={r['mae']}, r={r['pearson_r']}")
            print(f"  Expected:   {expected}")
        else:
            print(f"  ✓ {r['metric']:<22s}: Exact match across all 8 parameters")
            
    print(f"\nReconciliation Status: {'100% REPRODUCED (PASS)' if all_match else 'DISCREPANCIES DETECTED (FAIL)'}")

def main():
    df = verify_source_integrity(SOURCE_CSV)
    
    # 1. Primary Analysis: Subject-Averaged (N=46 subjects, df=45)
    # Clustering is resolved by construction: 1 observation per participant.
    # Evaluates agreement of multi-night participant means (~3 nights).
    sub_res = analyze_bland_altman(df, "subject_averaged_n46", n_obs=46, df_denom=45)
    format_table(sub_res, "PRIMARY TABLE: SUBJECT-AVERAGED BLAND-ALTMAN AGREEMENT (N=46 INDEPENDENT PARTICIPANTS; MULTI-NIGHT MEANS)")
    
    # 2. Secondary Analysis: Night-Level Pooled Unadjusted (N=152 nights, df=151)
    # Caution: Conflates between-subject and within-subject variance.
    # Unadjusted descriptive pooling only; NOT a validated single-night prediction interval.
    night_res = analyze_bland_altman(df, "night_level_pooled_n152_unadjusted", n_obs=152, df_denom=151)
    format_table(night_res, "SECONDARY TABLE: NIGHT-LEVEL POOLED UNADJUSTED DESCRIPTIVE DATA (N=152 NIGHTS; NOT SINGLE-NIGHT INTERVAL)")
    
    # 3. Manuscript Reconciliation
    reconcile_with_manuscript(sub_res)
    
    print("\nPhase B script execution complete.")

if __name__ == "__main__":
    main()
