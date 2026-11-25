"""
EXP-031 Amendment A1: W100 Retrain Statistical Analysis & Decision Rule Evaluation
File: experiments/EXP-031/analyze_w100_retrain.py

Computes:
1. Primary comparison: Delta_R = W100 - frozen Model A (N=47, 10,000-resample participant
   percentile bootstrap 95% CI seed=42, two-sided Wilcoxon, Cohen's d_z).
2. Secondary comparisons (Holm-adjusted together):
   - W100 - P190 (Delta_P)
   - W100 - p1_100k
   - W100 - Official SLAMSS-IFS
3. Sensitivity H-SCALE-R: Page's L across the 9 ordered widths with W100 replacing frozen Model A.
4. Locked Section 6 decision rule (R1, R2, R3).
5. Writes experiments/EXP-031/W100_retrained_summary.json.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

EXP031_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP031_DIR, "..", ".."))
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
EXP021_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-021")
for _p in (REPO_ROOT, EXP031_DIR, EXP030_DIR, EXP029_DIR, EXP021_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_exp031_confirmatory import compute_pages_l  # noqa: E402
from analyze_tsk08_paired_benchmark import participant_bootstrap_ci  # noqa: E402
from eval_exp029_downstream import holm_bonferroni  # noqa: E402
from exp031_models import EXP031_ARMS, EXP031_ARMS_ORDER  # noqa: E402


def load_subject_metrics(path: str, groupby_subject: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path)
    if groupby_subject:
        metric_cols = [
            c for c in [
                "macro_f1", "weighted_f1", "accuracy", "kappa",
                "f1_wake", "f1_light", "f1_deep", "f1_rem",
            ] if c in df.columns
        ]
        df = df.groupby("subject_id", as_index=False)[metric_cols].mean()
    df = df.sort_values("subject_id").reset_index(drop=True)
    assert len(df) == 47, f"Expected 47 subjects in {path}, got {len(df)}"
    return df


def paired_comparison(
    a_scores: np.ndarray,
    b_scores: np.ndarray,
    label: str,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    a_arr = np.asarray(a_scores, dtype=np.float64)
    b_arr = np.asarray(b_scores, dtype=np.float64)
    diff_pp = (a_arr - b_arr) * 100.0
    delta_pp = float(np.mean(diff_pp))
    ci_low, ci_high = participant_bootstrap_ci(diff_pp, n_boot=n_boot, alpha=0.05, seed=seed)
    _, wilcoxon_p = stats.wilcoxon(diff_pp) if np.any(diff_pp != 0) else (0.0, 1.0)
    sd = float(np.std(diff_pp, ddof=1)) if len(diff_pp) > 1 else 0.0
    cohens_dz = float(delta_pp / sd) if sd > 1e-12 else 0.0
    return {
        "comparison": label,
        "n_subjects": int(len(diff_pp)),
        "a_mean_pct": float(np.mean(a_arr) * 100.0),
        "b_mean_pct": float(np.mean(b_arr) * 100.0),
        "delta_pp": delta_pp,
        "ci95_low_pp": float(ci_low),
        "ci95_high_pp": float(ci_high),
        "wilcoxon_p_raw": float(wilcoxon_p),
        "cohens_dz": cohens_dz,
        "wins_a": int(np.sum(diff_pp > 0)),
        "losses_a": int(np.sum(diff_pp < 0)),
        "ties": int(np.sum(diff_pp == 0)),
    }


def main() -> None:
    w100_csv = os.path.join(EXP031_DIR, "runs", "W100", "W100_subject_metrics.csv")
    w100_sum_json = os.path.join(EXP031_DIR, "runs", "W100", "W100_summary.json")
    frozen_a_csv = os.path.join(REPO_ROOT, "experiments", "STAGE2_FINAL_MODEL", "final_model_oof_subject_metrics.csv")
    p190_csv = os.path.join(EXP031_DIR, "P190_subject_metrics.csv")
    p1_100k_csv = os.path.join(EXP030_DIR, "p1_100k_subject_metrics.csv")
    slamss_csv = os.path.join(EXP021_DIR, "slamss_ifs_matched_subject_metrics.csv")
    full_v5_csv = os.path.join(EXP029_DIR, "full_v5_subject_metrics.csv")

    df_w100 = load_subject_metrics(w100_csv, groupby_subject=False)
    df_frozen_a = load_subject_metrics(frozen_a_csv, groupby_subject=True)
    df_p190 = load_subject_metrics(p190_csv, groupby_subject=False)
    df_p1_100k = load_subject_metrics(p1_100k_csv, groupby_subject=False)
    df_slamss = load_subject_metrics(slamss_csv, groupby_subject=True)
    df_v5 = load_subject_metrics(full_v5_csv, groupby_subject=False)

    subs = df_w100["subject_id"].tolist()
    for name, df in [
        ("frozen_A", df_frozen_a),
        ("P190", df_p190),
        ("p1_100k", df_p1_100k),
        ("SLAMSS-IFS", df_slamss),
        ("full_v5", df_v5),
    ]:
        assert df["subject_id"].tolist() == subs, f"Subject order mismatch in {name}"

    with open(w100_sum_json, "r", encoding="utf-8") as f:
        w100_agg = json.load(f)

    # Primary comparison (unadjusted)
    primary = paired_comparison(
        df_w100["macro_f1"].values,
        df_frozen_a["macro_f1"].values,
        "W100 - frozen Model A (Delta_R)",
    )

    # Secondary comparisons (Holm-adjusted together)
    sec_p190 = paired_comparison(df_w100["macro_f1"].values, df_p190["macro_f1"].values, "W100 - P190 (Delta_P)")
    sec_p1 = paired_comparison(df_w100["macro_f1"].values, df_p1_100k["macro_f1"].values, "W100 - p1_100k")
    sec_slamss = paired_comparison(df_w100["macro_f1"].values, df_slamss["macro_f1"].values, "W100 - Official SLAMSS-IFS")

    raw_ps = [sec_p190["wilcoxon_p_raw"], sec_p1["wilcoxon_p_raw"], sec_slamss["wilcoxon_p_raw"]]
    holm_ps = holm_bonferroni(raw_ps)
    for sec, hp in zip([sec_p190, sec_p1, sec_slamss], holm_ps):
        sec["wilcoxon_p_holm"] = float(hp)

    # Sensitivity H-SCALE-R: Page's L across 9 ordered widths with W100 replacing A
    width_cols = []
    width_means = {}
    for arm in EXP031_ARMS_ORDER:
        if arm == "A":
            df_arm = df_w100
            label = "W100"
        elif EXP031_ARMS[arm]["is_new"]:
            df_arm = load_subject_metrics(os.path.join(EXP031_DIR, f"{arm}_subject_metrics.csv"), groupby_subject=False)
            label = arm
        else:
            df_arm = load_subject_metrics(EXP031_ARMS[arm]["reused_oof_csv"], groupby_subject=False)
            label = arm
        assert df_arm["subject_id"].tolist() == subs
        width_cols.append(df_arm["macro_f1"].values)
        width_means[label] = {
            "n_params": int(EXP031_ARMS[arm]["expected_params"]),
            "macro_f1_mean_pct": float(df_arm["macro_f1"].mean() * 100.0),
            "macro_f1_std_pct": float(df_arm["macro_f1"].std(ddof=1) * 100.0),
        }
    mat_r = np.column_stack(width_cols)
    h_scale_r = compute_pages_l(mat_r)
    h_scale_r["ordered_arms"] = ["W100" if a == "A" else a for a in EXP031_ARMS_ORDER]
    h_scale_r["width_means"] = width_means

    # Locked Section 6 decision rule
    delta_r = primary["delta_pp"]
    ci_r_low = primary["ci95_low_pp"]
    ci_r_high = primary["ci95_high_pp"]
    ci_p_low = sec_p190["ci95_low_pp"]
    ci_p_high = sec_p190["ci95_high_pp"]

    if delta_r >= 1.0 and ci_r_low > 0.0 and ci_p_low > -1.0:
        verdict = "R1"
        verdict_label = "R1: recipe effect"
    elif ci_r_high < 1.0 and ci_p_high < 0.0:
        verdict = "R2"
        verdict_label = "R2: unexplained gap"
    else:
        verdict = "R3"
        verdict_label = "R3: inconclusive"

    out = {
        "amendment": "AMENDMENT_EXP-031_A1_W100_RETRAIN",
        "w100_aggregate": w100_agg,
        "reference_means_pct": {
            "W100": float(df_w100["macro_f1"].mean() * 100.0),
            "frozen_Model_A": float(df_frozen_a["macro_f1"].mean() * 100.0),
            "P190": float(df_p190["macro_f1"].mean() * 100.0),
            "p1_100k": float(df_p1_100k["macro_f1"].mean() * 100.0),
            "full_v5": float(df_v5["macro_f1"].mean() * 100.0),
            "Official_SLAMSS_IFS": float(df_slamss["macro_f1"].mean() * 100.0),
        },
        "primary_comparison": primary,
        "secondary_comparisons": {
            "W100_minus_P190": sec_p190,
            "W100_minus_p1_100k": sec_p1,
            "W100_minus_Official_SLAMSS_IFS": sec_slamss,
        },
        "h_scale_r_sensitivity": h_scale_r,
        "original_exp031_h_scale": {
            "L": 10938.0,
            "z": 2.493,
            "p_holm_two_sided": 0.0254,
            "verdict": "SUPPORTED",
        },
        "decision": {
            "verdict": verdict,
            "verdict_label": verdict_label,
            "delta_r_pp": delta_r,
            "ci95_delta_r_pp": [ci_r_low, ci_r_high],
            "delta_p_pp": sec_p190["delta_pp"],
            "ci95_delta_p_pp": [ci_p_low, ci_p_high],
        },
    }

    out_path = os.path.join(EXP031_DIR, "W100_retrained_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"W100 Macro-F1: {out['reference_means_pct']['W100']:.2f}% (vs frozen Model A {out['reference_means_pct']['frozen_Model_A']:.2f}%)")
    print(f"Delta_R: {delta_r:+.2f} pp [{ci_r_low:+.2f}, {ci_r_high:+.2f}], p={primary['wilcoxon_p_raw']:.3e}")
    print(f"Delta_P (W100 - P190): {sec_p190['delta_pp']:+.2f} pp [{ci_p_low:+.2f}, {ci_p_high:+.2f}], p_Holm={sec_p190['wilcoxon_p_holm']:.4f}")
    print(f"H-SCALE-R: L={h_scale_r['L']:.1f}, z={h_scale_r['z']:+.3f}, p_two_sided={h_scale_r['p_two_sided']:.4f}")
    print(f"VERDICT: {verdict_label}")


if __name__ == "__main__":
    main()
