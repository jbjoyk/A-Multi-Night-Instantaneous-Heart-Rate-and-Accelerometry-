#!/usr/bin/env python3
"""
bidsleep_sleep_summary_and_specificity_analysis.py
=================================================
Reproducible analysis script for Phase 2D:
  1. Part 1: Subject-averaged sleep-summary Bland-Altman analysis across adaptation arms (exp019_sleep_summary_agreement.csv)
  2. Part 2: Specificity control hypotheses testing (S1, S2, S3) and outlier sensitivity (exp019_subject_metrics.csv)

Strictly read-only; does not modify any locked artifacts.
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

def run_part1_sleep_summary(agreement_csv_path: str):
    print("=" * 80)
    print("PART 1: SLEEP-SUMMARY BLAND-ALTMAN ANALYSIS (SUBJECT-AVERAGED, N=46)")
    print("=" * 80)
    
    if not os.path.exists(agreement_csv_path):
        raise FileNotFoundError(f"Missing input artifact: {agreement_csv_path}")
        
    df = pd.read_csv(agreement_csv_path)
    sub_df = df[df["aggregation_level"] == "subject_averaged_n46"].copy()
    
    n_subs = 46
    df_t = n_subs - 1
    t_crit = stats.t.ppf(0.975, df=df_t)
    
    sub_df["se_bias"] = sub_df["sd_diff"] / np.sqrt(n_subs)
    sub_df["ci_low"] = sub_df["bias"] - t_crit * sub_df["se_bias"]
    sub_df["ci_high"] = sub_df["bias"] + t_crit * sub_df["se_bias"]
    sub_df["is_sig"] = (sub_df["ci_low"] > 0) | (sub_df["ci_high"] < 0)
    
    def is_large_bias(row):
        if "min" in row["metric"]:
            return abs(row["bias"]) > 5.0
        return abs(row["bias"]) > 2.0
        
    sub_df["large_bias"] = sub_df.apply(is_large_bias, axis=1)
    
    # 1a. Primary FT-2 Table
    print("\n[1a] Primary Arm: FT-2 (Supervised 2-Night Personalization)")
    print(f"{'Metric':<22s} | {'Bias':>7s} | {'95% CI':>18s} | {'Sig?':>6s} | {'SD Diff':>7s} | {'95% LoA':>18s} | {'MAE':>6s} | {'Pearson r':>9s}")
    print("-" * 110)
    ft2 = sub_df[sub_df["arm"] == "FT-2"]
    for _, r in ft2.iterrows():
        sig_str = "YES (*)" if r["is_sig"] else "NO"
        ci_str = f"[{r['ci_low']:+6.2f}, {r['ci_high']:+6.2f}]"
        loa_str = f"[{r['loa_low']:+6.2f}, {r['loa_high']:+6.2f}]"
        unit = "min" if "min" in r["metric"] else "%"
        print(f"{r['metric']:<22s} | {r['bias']:+6.2f} {unit} | {ci_str:>18s} | {sig_str:>6s} | {r['sd_diff']:6.2f}  | {loa_str:>18s} | {r['mae']:5.2f}  | {r['pearson_r']:9.4f}")

    # 1b. Cross-Arm Comparison
    print("\n[1b] Cross-Arm Comparison (ZS, FT-2, FT-other-2, PS-2)")
    print(f"{'Arm':<10s} | {'Metric':<22s} | {'Bias':>7s} | {'95% CI':>18s} | {'Sig?':>6s} | {'95% LoA':>18s} | {'MAE':>6s} | {'Pearson r':>9s} | {'Flag'}")
    print("-" * 125)
    for arm in ["ZS", "FT-2", "FT-other-2", "PS-2"]:
        arm_df = sub_df[sub_df["arm"] == arm]
        for _, r in arm_df.iterrows():
            sig_str = "*" if r["is_sig"] else " "
            ci_str = f"[{r['ci_low']:+6.2f}, {r['ci_high']:+6.2f}]"
            loa_str = f"[{r['loa_low']:+6.2f}, {r['loa_high']:+6.2f}]"
            large_str = "⚠ large bias" if r["large_bias"] else ""
            unit = "min" if "min" in r["metric"] else "%"
            print(f"{arm:<10s} | {r['metric']:<22s} | {r['bias']:+6.2f} {sig_str:<1s} | {ci_str:>18s} | {sig_str:>6s} | {loa_str:>18s} | {r['mae']:5.2f}  | {r['pearson_r']:9.4f} | {large_str}")

    return sub_df


def run_part2_specificity(subject_metrics_csv_path: str):
    print("\n" + "=" * 80)
    print("PART 2: SPECIFICITY CONTROLS & DATA-INTEGRITY ANALYSIS")
    print("=" * 80)
    
    if not os.path.exists(subject_metrics_csv_path):
        raise FileNotFoundError(f"Missing input artifact: {subject_metrics_csv_path}")
        
    df = pd.read_csv(subject_metrics_csv_path)
    n_subs = len(df)
    t_crit = stats.t.ppf(0.975, df=n_subs - 1)
    
    print("\n[2a] Paired Tests (Two-Sided) vs Null (0.0 pp)")
    print(f"{'Control':<10s} | {'N':>3s} | {'Mean':>7s} | {'SD':>7s} | {'Median':>7s} | {'IQR':>7s} | {'t-stat':>7s} | {'t p-val':>12s} | {'t 95% CI':>18s} | {'Wilcoxon W':>10s} | {'Wilc p-val':>12s}")
    print("-" * 125)
    
    for c_name in ["delta_S1", "delta_S2", "delta_S3"]:
        vals = df[c_name].values
        n = len(vals)
        mean_val = np.mean(vals)
        sd_val = np.std(vals, ddof=1)
        med_val = np.median(vals)
        q25, q75 = np.percentile(vals, [25, 75])
        iqr_val = q75 - q25
        
        if np.all(vals == 0.0):
            t_stat, p_t = np.nan, np.nan
            ci_str = "[ +0.00,  +0.00]"
            w_stat, p_w = 0.0, 1.0
        else:
            t_stat, p_t = stats.ttest_1samp(vals, 0.0)
            se = sd_val / np.sqrt(n)
            ci_l, ci_h = mean_val - t_crit * se, mean_val + t_crit * se
            ci_str = f"[{ci_l:+6.2f}, {ci_h:+6.2f}]"
            w_res = stats.wilcoxon(vals)
            w_stat, p_w = w_res.statistic, w_res.pvalue
            
        print(f"{c_name:<10s} | {n:3d} | {mean_val:+6.2f}  | {sd_val:6.2f}  | {med_val:+6.2f}  | {iqr_val:6.2f}  | {t_stat:+7.2f} | {p_t:12.4e} | {ci_str:>18s} | {w_stat:10.1f} | {p_w:12.4e}")

    # 2b. Outlier Sensitivity for S1
    print("\n[2b] Outlier Sensitivity for S1 (FT-2 vs FT-other-2)")
    sub68_val = df[df["subject_id"] == "Bidslab68"]["delta_S1"].values[0]
    sub34_val = df[df["subject_id"] == "Bidslab34"]["delta_S1"].values[0]
    print(f"Identified Outliers: Bidslab68 = {sub68_val:+.2f} pp, Bidslab34 = {sub34_val:+.2f} pp")
    
    df_sens = df[~df["subject_id"].isin(["Bidslab68", "Bidslab34"])]
    vals_sens = df_sens["delta_S1"].values
    n_sens = len(vals_sens)
    t_crit_sens = stats.t.ppf(0.975, df=n_sens - 1)
    
    t_s, p_ts = stats.ttest_1samp(vals_sens, 0.0)
    se_s = np.std(vals_sens, ddof=1) / np.sqrt(n_sens)
    w_s_res = stats.wilcoxon(vals_sens)
    
    print(f"  All Cohort (n=46):        Mean = {np.mean(df['delta_S1']):+.2f} pp (SD: {np.std(df['delta_S1'], ddof=1):.2f}) | t = {stats.ttest_1samp(df['delta_S1'], 0.0)[0]:.2f} (p = {stats.ttest_1samp(df['delta_S1'], 0.0)[1]:.4e}) | W = {stats.wilcoxon(df['delta_S1']).statistic:.1f} (p = {stats.wilcoxon(df['delta_S1']).pvalue:.4e})")
    print(f"  Outliers Excluded (n=44): Mean = {np.mean(vals_sens):+.2f} pp (SD: {np.std(vals_sens, ddof=1):.2f}) | t = {t_s:.2f} (p = {p_ts:.4e}) | W = {w_s_res.statistic:.1f} (p = {w_s_res.pvalue:.4e})")

    # 2c. S3 Data Integrity
    print("\n[2c] S3 Data-Integrity Verification")
    is_exact = np.array_equal(df["ZS"].values, df["UA-2"].values)
    is_all_zero = np.all(df["delta_S3"].values == 0.0)
    print(f"  Bitwise Identity Check (UA-2 == ZS across all 46 rows): {is_exact}")
    print(f"  Exact Zero Check (delta_S3 == 0.0 across all 46 rows):  {is_all_zero}")
    if is_exact and is_all_zero:
        print("  FLAG [CRITICAL DATA INTEGRITY]: The S3 (UA-2) self-supervision control is completely DEGENERATE.")
        print("  UA-2 performance values in exp019_subject_metrics.csv are identical copies of ZS values.")


if __name__ == "__main__":
    agree_path = "experiments/EXP-019/exp019_sleep_summary_agreement.csv"
    subject_path = "experiments/EXP-019/exp019_subject_metrics.csv"
    run_part1_sleep_summary(agree_path)
    run_part2_specificity(subject_path)
