"""
analyze_tsk08_paired_benchmark.py: Stage 3 Paired Comparative Statistical Inference Engine (EXP-021 / TSK-08).

Strictly implements PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md Sections 4 & 5 and DEC-033:
  - Primary Estimand: Cohort mean participant-level paired difference (N=47 independent subjects):
        d_i = mean_seeds(Macro-F1_{i, Model A}) - mean_seeds(Macro-F1_{i, SLAMSS-IFS})
        d_bar = (1 / 47) * sum_{i=1}^{47} d_i
  - Completeness Invariants: Enforces exact 47 subjects x 3 seeds (141 rows, zero duplicates,
    every participant has exact seed set {42, 123, 2024}, matching folds and valid epoch counts).
  - Bootstrap Inference: 10,000-resample participant-level percentile bootstrap 95% CI (seed=42).
  - Non-Parametric Test: Two-sided Wilcoxon signed-rank test (H0: Median(d) = 0).
  - Effect Size: Standardized paired Cohen's d_z = d_bar / s_d (ddof=1).
  - Secondary Paired Metrics (with 95% bootstrap CIs & Wilcoxon p):
        Overall Accuracy, Weighted-F1, Cohen's Kappa, Wake F1, Light F1, Deep F1, REM F1.
  - Strictly NO arbitrary non-inferiority margins (retracted under DEC-033).
  - 3-Tier Preregistered Interpretation Framework (with explicit bootstrap-vs-Wilcoxon discordance note):
        1. MODEL A ADVANTAGE (STATISTICALLY SIGNIFICANT): CI_low > 0 and p < 0.05
        2. SLAMSS-IFS ADVANTAGE (STATISTICALLY SIGNIFICANT): CI_high < 0 and p < 0.05
        3. PERFORMANCE INDISTINGUISHABLE (STATISTICAL PARITY): CI spans 0 or p >= 0.05

Outputs:
  - experiments/EXP-021/slamss_ifs_matched_subject_metrics.csv
  - experiments/EXP-021/slamss_ifs_paired_subject_deltas.csv
  - experiments/EXP-021/slamss_ifs_paired_comparison_summary.json
  - experiments/EXP-021/slamss_ifs_paired_comparative_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.stats as stats

EXP021_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP021_DIR, "..", ".."))
for _p in (REPO_ROOT, EXP021_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_tsk08_slamss_matched import aggregate_completed_runs_to_csv

STAGE2_OOF_CSV_DEFAULT = os.path.join(
    REPO_ROOT, "experiments", "STAGE2_FINAL_MODEL", "final_model_oof_subject_metrics.csv"
)
EXP018_PHASE2_DIR_DEFAULT = os.path.join(REPO_ROOT, "experiments", "EXP-018", "phase2_results")

MODEL_A_PARAMS = 163940
SLAMSS_PAPER_PARAMS = 666060
SLAMSS_REPO_PARAMS = 5465768
REQUIRED_SEEDS = {42, 123, 2024}


def participant_bootstrap_ci(
    deltas: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float]:
    """10,000-resample participant-level percentile bootstrap confidence interval."""
    arr = np.asarray(deltas, dtype=np.float64)
    n = len(arr)
    assert n > 0, "Empty delta array for bootstrap"
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    ci_low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return ci_low, ci_high


def paired_metric_inference(
    model_a_vals: np.ndarray,
    slamss_vals: np.ndarray,
    scale: float = 100.0,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Computes participant-level paired comparative statistics for a metric.
    scale=100.0 reports deltas in percentage points (pp); scale=1.0 for Kappa.
    """
    a = np.asarray(model_a_vals, dtype=np.float64)
    s = np.asarray(slamss_vals, dtype=np.float64)
    assert len(a) == len(s) and len(a) > 1, "Paired arrays must share length > 1"

    d_raw = a - s
    d_scaled = d_raw * scale
    mean_d = float(np.mean(d_scaled))
    sd_d = float(np.std(d_scaled, ddof=1))
    median_d = float(np.median(d_scaled))

    if sd_d <= 1e-12:
        cohens_dz = 0.0 if abs(mean_d) <= 1e-12 else float(np.sign(mean_d) * np.inf)
    else:
        cohens_dz = float(mean_d / sd_d)

    ci_low, ci_high = participant_bootstrap_ci(d_scaled, n_boot=n_boot, alpha=0.05, seed=seed)

    if np.allclose(d_raw, 0.0):
        w_stat, p_val = 0.0, 1.0
    else:
        res = stats.wilcoxon(d_raw, alternative="two-sided", zero_method="wilcox")
        w_stat, p_val = float(res.statistic), float(res.pvalue)

    wins_a = int(np.sum(d_raw > 0.0))
    wins_s = int(np.sum(d_raw < 0.0))
    ties = int(np.sum(d_raw == 0.0))

    return {
        "n_participants": int(len(a)),
        "model_a_mean": float(np.mean(a) * scale),
        "model_a_sd": float(np.std(a, ddof=1) * scale),
        "slamss_mean": float(np.mean(s) * scale),
        "slamss_sd": float(np.std(s, ddof=1) * scale),
        "mean_diff": mean_d,
        "sd_diff": sd_d,
        "median_diff": median_d,
        "ci_95_lower": ci_low,
        "ci_95_upper": ci_high,
        "cohens_dz": cohens_dz,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_two_sided": p_val,
        "model_a_wins": wins_a,
        "slamss_wins": wins_s,
        "ties": ties,
    }


def classify_preregistered_outcome(
    ci_lower: float,
    ci_upper: float,
    wilcoxon_p: float,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Applies the exact 3-tier Interpretation Framework from
    PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md Section 5 (DEC-033), plus an explicit
    discordance flag when bootstrap 95% CI and Wilcoxon two-sided p disagree.
    """
    ci_excludes_zero = (ci_lower > 0.0) or (ci_upper < 0.0)
    wilcoxon_sig = wilcoxon_p < alpha
    discordant = ci_excludes_zero != wilcoxon_sig

    if ci_lower > 0.0 and wilcoxon_sig:
        return {
            "category": "MODEL A ADVANTAGE (STATISTICALLY SIGNIFICANT)",
            "discordant_tests": False,
            "discordance_note": None,
            "interpretation": (
                "Model A statistically outperforms SLAMSS-IFS on matched 5-fold CV despite "
                "operating at 4.06x (paper text) to 33.34x (official repo) parameter compression."
            ),
        }
    if ci_upper < 0.0 and wilcoxon_sig:
        return {
            "category": "SLAMSS-IFS ADVANTAGE (STATISTICALLY SIGNIFICANT)",
            "discordant_tests": False,
            "discordance_note": None,
            "interpretation": (
                "SLAMSS-IFS statistically outperforms Model A on matched 5-fold CV. "
                "The empirical performance gap and its 95% confidence interval quantify the "
                "exact accuracy-vs-efficiency trade-off against 4.06x-33.34x parameter compression."
            ),
        }

    discordance_note: Optional[str] = None
    if discordant:
        if ci_excludes_zero and not wilcoxon_sig:
            discordance_note = (
                "Bootstrap 95% CI excludes 0.0 while two-sided Wilcoxon p >= 0.05 "
                "(mean-vs-median skewness); classified conservatively under joint preregistered criterion."
            )
        else:
            discordance_note = (
                "Two-sided Wilcoxon p < 0.05 while bootstrap 95% CI spans 0.0 "
                "(outlier leverage on participant mean); classified conservatively under joint preregistered criterion."
            )

    return {
        "category": "PERFORMANCE INDISTINGUISHABLE (STATISTICAL PARITY)",
        "discordant_tests": discordant,
        "discordance_note": discordance_note,
        "interpretation": (
            "Model A and SLAMSS-IFS do not meet the joint significance criterion (95% CI excluding 0 "
            "and Wilcoxon p < 0.05) under matched 5-fold CV. The 95% confidence interval bounds the "
            "maximum plausible performance disparity while Model A achieves 4.06x-33.34x compression."
        ),
    }


def validate_subject_seed_completeness(
    df: pd.DataFrame,
    name: str,
    expected_n_subjects: int = 47,
) -> pd.DataFrame:
    """
    Enforces strict per-subject 3-seed completeness, uniqueness, and absence of partial runs.
    """
    df_filt = df[df["seed"].isin(REQUIRED_SEEDS)].copy()
    expected_rows = expected_n_subjects * len(REQUIRED_SEEDS)

    dups = df_filt.duplicated(subset=["subject_id", "seed"])
    assert not dups.any(), (
        f"{name}: detected {int(dups.sum())} duplicate (subject_id, seed) rows!"
    )

    n_subs = df_filt["subject_id"].nunique()
    assert n_subs == expected_n_subjects, (
        f"{name}: expected {expected_n_subjects} unique subjects, got {n_subs}"
    )

    assert len(df_filt) == expected_rows, (
        f"{name}: expected exactly {expected_rows} subject-seed rows "
        f"({expected_n_subjects} x {len(REQUIRED_SEEDS)}), got {len(df_filt)} (partial run detected!)"
    )

    seeds_per_sub = df_filt.groupby("subject_id")["seed"].apply(set)
    incomplete = seeds_per_sub[seeds_per_sub != REQUIRED_SEEDS]
    assert len(incomplete) == 0, (
        f"{name}: incomplete seed sets for subjects: {incomplete.to_dict()}"
    )
    return df_filt


def load_model_a_oof_dataframe(
    stage2_csv: str,
    exp018_phase2_dir: str,
    allow_partial_fallback: bool = False,
) -> pd.DataFrame:
    """
    Loads Model A (Frozen Final Model) 47-subject x 3-seed OOF metrics.
    Requires Stage 2's verified final_model_oof_subject_metrics.csv by default so all 8
    preregistered primary + secondary metrics are present; partial EXP-018 JSON fallback
    is permitted only under explicit --allow-partial-fallback.
    """
    if os.path.exists(stage2_csv):
        return pd.read_csv(stage2_csv)

    if not allow_partial_fallback:
        raise FileNotFoundError(
            f"Stage 2 frozen OOF reference CSV not found at {stage2_csv}. "
            "Run `experiments/STAGE2_FINAL_MODEL/freeze_final_model.py` first so all 8 "
            "preregistered primary and secondary metrics are available "
            "(or pass --allow-partial-fallback to use macro_f1/accuracy/kappa only)."
        )

    print(
        "WARNING: Using partial EXP-018 JSON fallback; Weighted-F1 and per-stage F1 columns "
        "will be absent from the comparative report.",
        file=sys.stderr,
    )
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        for seed in sorted(REQUIRED_SEEDS):
            jpath = os.path.join(exp018_phase2_dir, f"run_L1200_fold{fold}_seed{seed}.json")
            if not os.path.exists(jpath):
                raise FileNotFoundError(f"EXP-018 fallback JSON missing: {jpath}")
            with open(jpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            tm = data["test_metrics"]
            for sub in sorted(tm["subject_macro_f1"].keys()):
                rows.append(
                    {
                        "subject_id": sub,
                        "fold": fold,
                        "seed": seed,
                        "macro_f1": float(tm["subject_macro_f1"][sub]),
                        "accuracy": float(tm["subject_accuracy"][sub]),
                        "kappa": float(tm["subject_kappa"][sub]),
                    }
                )
    return pd.DataFrame(rows)


def run_paired_comparative_analysis(
    df_model_a: pd.DataFrame,
    df_slamss: pd.DataFrame,
    n_boot: int = 10000,
    seed: int = 42,
    expected_n_subjects: int = 47,
    require_all_secondary: bool = True,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Executes complete paired participant-level comparison between Model A and SLAMSS-IFS."""
    df_a = validate_subject_seed_completeness(df_model_a, "Model A", expected_n_subjects)
    df_s = validate_subject_seed_completeness(df_slamss, "SLAMSS-IFS", expected_n_subjects)

    a_sub = df_a.groupby("subject_id").mean(numeric_only=True)
    s_sub = df_s.groupby("subject_id").mean(numeric_only=True)

    common_subs = sorted(set(a_sub.index) & set(s_sub.index))
    assert len(common_subs) == expected_n_subjects, (
        f"Expected {expected_n_subjects} aligned participants, got {len(common_subs)}"
    )

    a_sub = a_sub.loc[common_subs]
    s_sub = s_sub.loc[common_subs]

    # Assert exact fold alignment and (when present) valid epoch count alignment per participant
    a_folds = a_sub["fold"].round().astype(int)
    s_folds = s_sub["fold"].round().astype(int)
    assert (a_folds == s_folds).all(), "Fold assignment mismatch between Model A and SLAMSS-IFS!"

    if "n_valid_epochs" in a_sub.columns and "n_valid_epochs" in s_sub.columns:
        a_ep = a_sub["n_valid_epochs"].round().astype(int)
        s_ep = s_sub["n_valid_epochs"].round().astype(int)
        assert (a_ep == s_ep).all(), (
            f"Valid epoch count mismatch between Model A and SLAMSS-IFS for subjects: "
            f"{a_ep[a_ep != s_ep].index.tolist()}"
        )

    metric_specs = [
        ("macro_f1", "Macro-F1", 100.0, "pp", "delta_macro_f1_pp"),
        ("accuracy", "Overall Accuracy", 100.0, "pp", "delta_accuracy_pp"),
        ("weighted_f1", "Weighted-F1", 100.0, "pp", "delta_weighted_f1_pp"),
        ("kappa", "Cohen's Kappa", 1.0, "kappa", "delta_kappa"),
        ("f1_wake", "Wake F1", 100.0, "pp", "delta_f1_wake_pp"),
        ("f1_light", "Light F1", 100.0, "pp", "delta_f1_light_pp"),
        ("f1_deep", "Deep F1", 100.0, "pp", "delta_f1_deep_pp"),
        ("f1_rem", "REM F1", 100.0, "pp", "delta_f1_rem_pp"),
    ]

    metrics_summary: Dict[str, Any] = {}
    delta_cols: Dict[str, Any] = {
        "subject_id": common_subs,
        "fold": a_folds.tolist(),
    }

    for col, label, scale, unit, delta_col_name in metric_specs:
        if col not in a_sub.columns or col not in s_sub.columns:
            if require_all_secondary:
                raise KeyError(
                    f"Required preregistered metric column '{col}' missing from input DataFrames."
                )
            continue
        res = paired_metric_inference(
            model_a_vals=a_sub[col].to_numpy(),
            slamss_vals=s_sub[col].to_numpy(),
            scale=scale,
            n_boot=n_boot,
            seed=seed,
        )
        res["label"] = label
        res["unit"] = unit
        metrics_summary[col] = res

        delta_cols[f"modelA_{col}"] = a_sub[col].to_numpy()
        delta_cols[f"slamss_{col}"] = s_sub[col].to_numpy()
        delta_cols[delta_col_name] = (a_sub[col].to_numpy() - s_sub[col].to_numpy()) * scale

    primary = metrics_summary["macro_f1"]
    verdict = classify_preregistered_outcome(
        ci_lower=primary["ci_95_lower"],
        ci_upper=primary["ci_95_upper"],
        wilcoxon_p=primary["wilcoxon_p_two_sided"],
    )

    summary = {
        "experiment_id": "EXP-021",
        "task_id": "TSK-08",
        "preregistration": "PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md",
        "n_participants": len(common_subs),
        "n_bootstrap_resamples": n_boot,
        "bootstrap_seed": seed,
        "parameter_context": {
            "model_a_params": MODEL_A_PARAMS,
            "model_a_int8_kib_weights_only": round(MODEL_A_PARAMS / 1024.0, 2),
            "slamss_paper_text_params": SLAMSS_PAPER_PARAMS,
            "slamss_official_repo_params": SLAMSS_REPO_PARAMS,
            "compression_vs_paper_text": round(SLAMSS_PAPER_PARAMS / MODEL_A_PARAMS, 2),
            "compression_vs_official_repo": round(SLAMSS_REPO_PARAMS / MODEL_A_PARAMS, 2),
        },
        "primary_endpoint": "macro_f1",
        "verdict": verdict,
        "metrics": metrics_summary,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    return summary, pd.DataFrame(delta_cols)


def write_comparative_markdown_report(
    report_path: str,
    summary: Dict[str, Any],
) -> None:
    """Renders the publication-ready EXP-021 paired comparative benchmark report."""
    p_res = summary["metrics"]["macro_f1"]
    pc = summary["parameter_context"]
    verdict = summary["verdict"]
    n_p = summary["n_participants"]

    lines = [
        "# EXP-021 (TSK-08): SLAMSS-IFS Matched 5-Fold Cross-Validation Paired Comparative Report",
        "",
        f"- **Preregistration:** `{summary['preregistration']}` (DEC-011, DEC-033, DEC-034)",
        f"- **Generated (UTC):** `{summary['timestamp_utc']}`",
        f"- **Independent Participants:** `N = {n_p}` (3-seed participant means across seeds `42, 123, 2024`)",
        f"- **Bootstrap Inference:** `{summary['n_bootstrap_resamples']:,}` participant-level resamples (seed `{summary['bootstrap_seed']}`)",
        f"- **Primary Classification:** **`{verdict['category']}`**",
        "",
        "---",
        "",
        "## 1. Primary Estimand & Hardware Efficiency Trade-Off",
        "",
        f"- **Model A (`ConventionalCNNBiGRU`, {pc['model_a_params']:,} params, ~{pc['model_a_int8_kib_weights_only']} KiB INT8 weights):** "
        f"**{p_res['model_a_mean']:.2f}%** (SD {p_res['model_a_sd']:.2f} pp)",
        f"- **Official SLAMSS-IFS ({pc['slamss_official_repo_params']:,} repo params / {pc['slamss_paper_text_params']:,} paper text):** "
        f"**{p_res['slamss_mean']:.2f}%** (SD {p_res['slamss_sd']:.2f} pp)",
        f"- **Paired Difference ($\\bar{{d}} = \\text{{Model A}} - \\text{{SLAMSS-IFS}}$):** "
        f"**{p_res['mean_diff']:+.2f} pp** [95% CI: **{p_res['ci_95_lower']:+.2f}, {p_res['ci_95_upper']:+.2f} pp**]",
        f"- **Wilcoxon Signed-Rank Test (two-sided):** $W = {p_res['wilcoxon_stat']:.1f}$, $p = {p_res['wilcoxon_p_two_sided']:.4e}$",
        f"- **Paired Effect Size (Cohen's $d_z$):** `{p_res['cohens_dz']:+.3f}`",
        f"- **Participant Win / Loss / Tie:** Model A wins `{p_res['model_a_wins']}/{n_p}` "
        f"({100.0 * p_res['model_a_wins'] / n_p:.1f}%), "
        f"SLAMSS-IFS wins `{p_res['slamss_wins']}/{n_p}`, ties `{p_res['ties']}`",
        f"- **Parameter Compression:** **{pc['compression_vs_paper_text']}x** vs. paper text | **{pc['compression_vs_official_repo']}x** vs. official repository",
        "",
        f"> **Interpretation:** {verdict['interpretation']}",
    ]

    if verdict.get("discordance_note"):
        lines.append(f"> **Statistical Discordance Note:** {verdict['discordance_note']}")

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 2. Complete Primary & Secondary Paired Metrics (N = {n_p})",
            "",
            "| Metric | Model A (Mean ± SD) | SLAMSS-IFS (Mean ± SD) | Paired Diff ($\\bar{d}$) | 95% Bootstrap CI | Cohen's $d_z$ | Wilcoxon $p$ | Model A Wins |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )

    for _, m in summary["metrics"].items():
        u = m["unit"]
        if u == "pp":
            a_str = f"{m['model_a_mean']:.2f}% ± {m['model_a_sd']:.2f}"
            s_str = f"{m['slamss_mean']:.2f}% ± {m['slamss_sd']:.2f}"
            d_str = f"{m['mean_diff']:+.2f} pp"
            ci_str = f"[{m['ci_95_lower']:+.2f}, {m['ci_95_upper']:+.2f}] pp"
        else:
            a_str = f"{m['model_a_mean']:.4f} ± {m['model_a_sd']:.4f}"
            s_str = f"{m['slamss_mean']:.4f} ± {m['slamss_sd']:.4f}"
            d_str = f"{m['mean_diff']:+.4f}"
            ci_str = f"[{m['ci_95_lower']:+.4f}, {m['ci_95_upper']:+.4f}]"
        lines.append(
            f"| **{m['label']}** | {a_str} | {s_str} | {d_str} | {ci_str} "
            f"| {m['cohens_dz']:+.3f} | `{m['wilcoxon_p_two_sided']:.4e}` "
            f"| {m['model_a_wins']}/{m['n_participants']} |"
        )

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TSK-08 / EXP-021 Paired Comparative Statistical Analysis"
    )
    parser.add_argument("--output-dir", default=EXP021_DIR)
    parser.add_argument("--stage2-oof-csv", default=STAGE2_OOF_CSV_DEFAULT)
    parser.add_argument("--exp018-phase2-dir", default=EXP018_PHASE2_DIR_DEFAULT)
    parser.add_argument(
        "--slamss-csv",
        default=None,
        help="Optional explicit SLAMSS-IFS CSV path (if omitted, aggregates from <output-dir>/run_results/)",
    )
    parser.add_argument("--deltas-csv", default=None)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--report-md", default=None)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-partial-fallback",
        action="store_true",
        help="Allow EXP-018 JSON fallback (omits Weighted-F1 and per-stage F1 secondary metrics)",
    )
    args = parser.parse_args()

    deltas_csv = args.deltas_csv or os.path.join(args.output_dir, "slamss_ifs_paired_subject_deltas.csv")
    summary_json = args.summary_json or os.path.join(args.output_dir, "slamss_ifs_paired_comparison_summary.json")
    report_md = args.report_md or os.path.join(args.output_dir, "slamss_ifs_paired_comparative_report.md")

    if args.slamss_csv is None:
        slamss_csv = os.path.join(args.output_dir, "slamss_ifs_matched_subject_metrics.csv")
        aggregate_completed_runs_to_csv(args.output_dir, out_csv_path=slamss_csv)
    else:
        slamss_csv = args.slamss_csv

    if not os.path.exists(slamss_csv):
        raise FileNotFoundError(
            f"SLAMSS-IFS subject metrics CSV not found at {slamss_csv}. "
            "Complete all 15 TSK-08 fold-seed runs first."
        )

    df_model_a = load_model_a_oof_dataframe(
        args.stage2_oof_csv,
        args.exp018_phase2_dir,
        allow_partial_fallback=args.allow_partial_fallback,
    )
    df_slamss = pd.read_csv(slamss_csv)

    summary, df_deltas = run_paired_comparative_analysis(
        df_model_a=df_model_a,
        df_slamss=df_slamss,
        n_boot=args.n_boot,
        seed=args.seed,
        expected_n_subjects=47,
        require_all_secondary=(not args.allow_partial_fallback),
    )

    os.makedirs(os.path.dirname(os.path.abspath(deltas_csv)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(summary_json)), exist_ok=True)
    df_deltas.to_csv(deltas_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_comparative_markdown_report(report_md, summary)

    p_res = summary["metrics"]["macro_f1"]
    print("=" * 80)
    print("EXP-021 (TSK-08) PAIRED COMPARATIVE ANALYSIS COMPLETE")
    print("=" * 80)
    print(
        f"Primary Macro-F1 Diff (Model A - SLAMSS-IFS): "
        f"{p_res['mean_diff']:+.2f} pp [95% CI: {p_res['ci_95_lower']:+.2f}, {p_res['ci_95_upper']:+.2f} pp]"
    )
    print(
        f"Wilcoxon p (two-sided): {p_res['wilcoxon_p_two_sided']:.4e} | "
        f"Cohen's d_z: {p_res['cohens_dz']:+.3f}"
    )
    print(f"Official Classification: {summary['verdict']['category']}")
    print(f"Report saved to: {report_md}")
    print("=" * 80)


if __name__ == "__main__":
    main()
