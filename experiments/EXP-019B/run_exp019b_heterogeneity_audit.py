#!/usr/bin/env python3
"""
run_exp019b_heterogeneity_audit.py
==================================
CPU-only Exploratory Closure for EXP-019B (Subject-Level Heterogeneity Audit).
Strictly adheres to docs/designs/exp019b_exp020_closure_spec.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [
    REPO_ROOT,
    os.path.join(REPO_ROOT, "experiments", "EXP-019"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from slamss_ifs_train import load_night_features
from exp019_adaptation_core import compute_wearable_quality_mask
from audit_heterogeneity_mechanisms import compute_bh_fdr

DEFAULT_CACHE_DIR = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
DEFAULT_RAW_BIDSLEEP_DIR = os.path.join(
    REPO_ROOT, "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
)
DEFAULT_NIGHT_INVENTORY = os.path.join(REPO_ROOT, "review_package", "night_inventory.csv")
DEFAULT_EXP019_METRICS = os.path.join(REPO_ROOT, "experiments", "EXP-019", "exp019_subject_metrics.csv")
DEFAULT_EXP021_DELTAS = os.path.join(REPO_ROOT, "experiments", "EXP-021", "slamss_ifs_paired_subject_deltas.csv")
DEFAULT_DIAG_TABLE = os.path.join(REPO_ROOT, "research_knowledge", "updates", "bidsleep_subject_diagnostic_table.csv")

PRIMARY_FEATURES = [
    "min_stage_frac",
    "stage_entropy",
    "eff_ihr_hz",
    "missingness_rate",
    "baseline_hr_mean",
    "baseline_hr_iqr",
]


def load_usable_nights_by_subject(night_inventory_csv: str) -> Dict[str, List[str]]:
    """Returns chronological usable night IDs per subject from review_package/night_inventory.csv."""
    df = pd.read_csv(night_inventory_csv)
    df = df[df["usable"].astype(bool)].copy()
    df["night_int"] = df["night"].astype(int)
    df = df.sort_values(["subject_id", "night_int"])
    out: Dict[str, List[str]] = {}
    for sub, grp in df.groupby("subject_id", sort=True):
        out[str(sub)] = [str(n) for n in grp["night_int"].tolist()]
    return out


def compute_subject_window_features(
    subject_id: str,
    night_ids: List[str],
    cache_dir: str = DEFAULT_CACHE_DIR,
    raw_bidsleep_dir: str = DEFAULT_RAW_BIDSLEEP_DIR,
) -> Dict[str, float]:
    """
    Computes the 6 primary EXP-019B predictors deterministically over `night_ids`.
    Never pulls raw BPM from z-scored `rec["ihr"]`; reads raw `hr.csv` directly.
    """
    if not night_ids:
        raise ValueError(f"Empty night_ids for subject {subject_id}")

    stage_counts = np.zeros(4, dtype=np.int64)
    total_epochs = 0
    missing_epochs = 0
    total_hr_rows = 0
    total_hr_span_s = 0.0
    raw_bpm_samples: List[np.ndarray] = []

    for nid in night_ids:
        rec = load_night_features(cache_dir, subject_id, str(nid))
        if rec is None:
            raise FileNotFoundError(f"Missing cache .npz for {subject_id}__{nid} in {cache_dir}")

        y = rec["y"]
        ihr = rec["ihr"]
        accel = rec["accel"]
        n_ep = len(y)
        total_epochs += n_ep

        valid_label = (y >= 0) & (y < 4)
        for c in range(4):
            stage_counts[c] += int(np.sum(y[valid_label] == c))

        wearable_ok = compute_wearable_quality_mask(ihr, accel)
        is_missing = (~wearable_ok) | (~valid_label)
        missing_epochs += int(np.sum(is_missing))

        json_path = os.path.join(cache_dir, f"{subject_id}__{nid}.json")
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing cache metadata JSON: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            meta_json = json.load(f)

        total_hr_rows += int(meta_json["hr_file"]["n_rows_valid"])
        span_s = float(meta_json["hr_last_offset_s"]) - float(meta_json["hr_first_offset_s"])
        if span_s <= 0:
            raise ValueError(f"Non-positive HR span for {subject_id}__{nid}: {span_s}")
        total_hr_span_s += span_s
        rec_start_unix = float(meta_json["rec_start_unix"])

        hr_csv_path = os.path.join(raw_bidsleep_dir, subject_id, str(nid), "hr.csv")
        if not os.path.exists(hr_csv_path):
            raise FileNotFoundError(f"Missing raw hr.csv: {hr_csv_path}")
        hr_arr = np.loadtxt(hr_csv_path, delimiter=",", dtype=np.float64)
        if hr_arr.ndim == 1:
            hr_arr = hr_arr.reshape(1, -1)
        t_hr = hr_arr[:, 0]
        bpm_hr = hr_arr[:, 1]
        finite_mask = np.isfinite(t_hr) & np.isfinite(bpm_hr)
        t_hr = t_hr[finite_mask]
        bpm_hr = bpm_hr[finite_mask]

        # Valid unmasked 30s epochs: [rec_start_unix + 30*e, rec_start_unix + 30*(e+1))
        unmasked_ep = np.where(~is_missing)[0]
        if len(unmasked_ep) > 0:
            rel_ep = np.floor((t_hr - rec_start_unix) / 30.0).astype(np.int64)
            in_valid_ep = np.isin(rel_ep, unmasked_ep)
            if np.any(in_valid_ep):
                raw_bpm_samples.append(bpm_hr[in_valid_ep])

    n_valid_stages = int(stage_counts.sum())
    if n_valid_stages == 0:
        raise ValueError(f"Zero valid stage epochs for {subject_id} across {night_ids}")

    probs = stage_counts / float(n_valid_stages)
    min_stage_frac = float(np.min(probs))
    pos_probs = probs[probs > 0]
    stage_entropy = float(-np.sum(pos_probs * np.log2(pos_probs)))

    eff_ihr_hz = float(total_hr_rows / total_hr_span_s)
    missingness_rate = float(missing_epochs / float(total_epochs))

    if not raw_bpm_samples:
        raise ValueError(f"Zero raw BPM samples in valid unmasked epochs for {subject_id}")
    all_bpm = np.concatenate(raw_bpm_samples)
    baseline_hr_mean = float(np.mean(all_bpm))
    q75, q25 = np.percentile(all_bpm, [75.0, 25.0])
    baseline_hr_iqr = float(q75 - q25)

    return {
        "min_stage_frac": min_stage_frac,
        "stage_entropy": stage_entropy,
        "eff_ihr_hz": eff_ihr_hz,
        "missingness_rate": missingness_rate,
        "baseline_hr_mean": baseline_hr_mean,
        "baseline_hr_iqr": baseline_hr_iqr,
    }


def spearman_with_bootstrap(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes two-sided Spearman rho, p-value, and percentile bootstrap 95% CI."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    assert len(x_arr) == len(y_arr) and len(x_arr) >= 3

    rho, p_val = stats.spearmanr(x_arr, y_arr)
    rng = np.random.default_rng(seed)
    n = len(x_arr)
    boot_rhos = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb, yb = x_arr[idx], y_arr[idx]
        if np.std(xb) == 0 or np.std(yb) == 0:
            boot_rhos[b] = np.nan
        else:
            r_b, _ = stats.spearmanr(xb, yb)
            boot_rhos[b] = r_b

    n_nan = int(np.sum(np.isnan(boot_rhos)))
    ci_low, ci_high = np.nanpercentile(boot_rhos, [2.5, 97.5])
    return {
        "spearman_rho": float(rho),
        "p_value": float(p_val),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "n_nan_resamples": n_nan,
    }


def run_exp019b_audit(
    cache_dir: str = DEFAULT_CACHE_DIR,
    raw_bidsleep_dir: str = DEFAULT_RAW_BIDSLEEP_DIR,
    night_inventory_csv: str = DEFAULT_NIGHT_INVENTORY,
    exp019_metrics_csv: str = DEFAULT_EXP019_METRICS,
    exp021_deltas_csv: str = DEFAULT_EXP021_DELTAS,
    output_dir: str = os.path.join(REPO_ROOT, "experiments", "EXP-019B"),
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    usable_nights = load_usable_nights_by_subject(night_inventory_csv)
    exp019_df = pd.read_csv(exp019_metrics_csv)
    exp021_df = pd.read_csv(exp021_deltas_csv)

    exp019_subs = sorted(exp019_df["subject_id"].astype(str).tolist())
    exp021_subs = sorted(exp021_df["subject_id"].astype(str).tolist())
    assert len(exp019_subs) == 46 and "Bidslab49" not in exp019_subs
    assert len(exp021_subs) == 47 and "Bidslab49" in exp021_subs

    # Build 47-row subject feature table
    feat_rows: List[Dict[str, Any]] = []
    for sub in exp021_subs:
        nights = usable_nights[sub]
        all_f = compute_subject_window_features(sub, nights, cache_dir, raw_bidsleep_dir)
        row: Dict[str, Any] = {"subject_id": sub, "n_usable_nights": len(nights)}
        for k, v in all_f.items():
            row[f"all_nights_{k}"] = v
        if sub in exp019_subs:
            cal_f = compute_subject_window_features(sub, nights[:2], cache_dir, raw_bidsleep_dir)
            for k, v in cal_f.items():
                row[f"cal_{k}"] = v
        else:
            for k in PRIMARY_FEATURES:
                row[f"cal_{k}"] = np.nan
        feat_rows.append(row)

    feat_df = pd.DataFrame(feat_rows)
    feat_csv = os.path.join(output_dir, "exp019b_subject_features.csv")
    feat_df.to_csv(feat_csv, index=False)

    # Merge outcomes
    f1_df = exp019_df.merge(feat_df, on="subject_id", how="inner")
    f1_df["outcome_delta"] = (f1_df["FT-2"] - f1_df["ZS"]) * 100.0
    f2_df = exp021_df.merge(feat_df, on="subject_id", how="inner")
    f2_df["outcome_delta"] = f2_df["delta_macro_f1_pp"]

    corr_rows: List[Dict[str, Any]] = []
    # Primary Family 1 (N=46, cal_[:2]) & Primary Family 2 (N=47, all_nights)
    for fam_name, df_fam, prefix in [
        ("EXP-019_FT2_minus_ZS", f1_df, "cal_"),
        ("EXP-021_ModelA_minus_SLAMSS", f2_df, "all_nights_"),
    ]:
        p_vals = []
        temp_rows = []
        for feat in PRIMARY_FEATURES:
            res = spearman_with_bootstrap(df_fam[f"{prefix}{feat}"].values, df_fam["outcome_delta"].values, n_boot, seed)
            df_loo = df_fam[df_fam["subject_id"] != "Bidslab34"]
            rho_loo, _ = stats.spearmanr(df_loo[f"{prefix}{feat}"].values, df_loo["outcome_delta"].values)
            p_vals.append(res["p_value"])
            temp_rows.append({
                "outcome": fam_name,
                "analysis_type": "primary",
                "feature": feat,
                "n": len(df_fam),
                **res,
                "fdr_q_family": np.nan,
                "fdr_q_pooled": np.nan,
                "loo_bidslab34_rho": float(rho_loo),
            })
        q_fam = compute_bh_fdr(p_vals)
        for r, q in zip(temp_rows, q_fam):
            r["fdr_q_family"] = q
        corr_rows.extend(temp_rows)

    # Pooled BH-FDR across m=12 primary rows
    pooled_q = compute_bh_fdr([r["p_value"] for r in corr_rows])
    for r, q in zip(corr_rows, pooled_q):
        r["fdr_q_pooled"] = q

    # Sensitivity checks for Family 2 (excluded from BH-FDR)
    f2_n46 = f2_df[f2_df["subject_id"].isin(exp019_subs)]
    for sens_name, prefix in [
        ("sensitivity_n46_all_nights", "all_nights_"),
        ("sensitivity_n46_cal_only", "cal_"),
    ]:
        for feat in PRIMARY_FEATURES:
            res = spearman_with_bootstrap(f2_n46[f"{prefix}{feat}"].values, f2_n46["outcome_delta"].values, n_boot, seed)
            corr_rows.append({
                "outcome": "EXP-021_ModelA_minus_SLAMSS",
                "analysis_type": sens_name,
                "feature": feat,
                "n": len(f2_n46),
                **res,
                "fdr_q_family": np.nan,
                "fdr_q_pooled": np.nan,
                "loo_bidslab34_rho": np.nan,
            })

    corr_df = pd.DataFrame(corr_rows)
    corr_csv = os.path.join(output_dir, "exp019b_correlation_summary.csv")
    corr_df.to_csv(corr_csv, index=False)

    # Mandatory Diagnostics
    oldham_mean = 0.5 * (f1_df["ZS"].values + f1_df["FT-2"].values) * 100.0
    oldham_rho, oldham_p = stats.spearmanr(oldham_mean, f1_df["outcome_delta"].values)

    resp_mask = f1_df["outcome_delta"].values > 0.0
    subgroup_diag = {}
    for feat in PRIMARY_FEATURES:
        r_vals = f1_df.loc[resp_mask, f"cal_{feat}"].values
        nr_vals = f1_df.loc[~resp_mask, f"cal_{feat}"].values
        u_stat, u_p = stats.mannwhitneyu(r_vals, nr_vals, alternative="two-sided")
        subgroup_diag[feat] = {
            "responder_median": float(np.median(r_vals)),
            "non_responder_median": float(np.median(nr_vals)),
            "mann_whitney_u": float(u_stat),
            "mann_whitney_p": float(u_p),
        }

    diag = {
        "classification": "DESIGN_INFORMED_EXPLORATORY_CLOSURE",
        "n_family1": len(f1_df),
        "n_family2": len(f2_df),
        "n_responders_gt0": int(np.sum(resp_mask)),
        "n_non_responders_le0": int(np.sum(~resp_mask)),
        "oldham_headroom": {"spearman_rho": float(oldham_rho), "p_value": float(oldham_p)},
        "subgroup_mann_whitney": subgroup_diag,
    }
    diag_json = os.path.join(output_dir, "exp019b_diagnostics.json")
    with open(diag_json, "w", encoding="utf-8") as f:
        json.dump(diag, f, indent=2)

    # Concise exploratory markdown report
    report_path = os.path.join(output_dir, "EXP019B_CLOSURE_REPORT.md")
    lines = [
        "# EXP-019B Subject-Level Heterogeneity Closure Report (Exploratory)",
        "",
        "> **Prior-Access Notice:** Design-informed exploratory analysis on BIDSleep; no confirmatory claims.",
        "",
        f"- **Family 1 (`EXP-019` FT-2 - ZS):** N={len(f1_df)} (Responders Δ>0: {diag['n_responders_gt0']}, Non-responders: {diag['n_non_responders_le0']})",
        f"- **Family 2 (`EXP-021` Model A - SLAMSS-IFS):** N={len(f2_df)}",
        f"- **Oldham Headroom Check (Family 1):** rho={oldham_rho:.4f}, p={oldham_p:.4f}",
        "",
        "## Primary Correlation Summary",
        "",
        "| Outcome | Feature | N | Spearman rho | 95% CI | p-value | FDR q (family) | FDR q (pooled) | LOO Bidslab34 rho |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for _, r in corr_df[corr_df["analysis_type"] == "primary"].iterrows():
        lines.append(
            f"| {r['outcome']} | `{r['feature']}` | {int(r['n'])} | {r['spearman_rho']:+.4f} | "
            f"[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] | {r['p_value']:.4f} | "
            f"{r['fdr_q_family']:.4f} | {r['fdr_q_pooled']:.4f} | {r['loo_bidslab34_rho']:+.4f} |"
        )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {"features_csv": feat_csv, "correlation_csv": corr_csv, "diagnostics_json": diag_json, "report": report_path}


def main():
    parser = argparse.ArgumentParser(description="EXP-019B CPU-only Subject-Level Heterogeneity Closure")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_BIDSLEEP_DIR)
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "experiments", "EXP-019B"))
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    out = run_exp019b_audit(
        cache_dir=args.cache_dir,
        raw_bidsleep_dir=args.raw_dir,
        output_dir=args.output_dir,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(f"EXP-019B complete: {out['report']}")


if __name__ == "__main__":
    main()
