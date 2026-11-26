"""
EXP-033 Statistical Analysis, Forest Plot, Depth Curve, and Final Report Generator.
File: experiments/EXP-033/analyze_exp033.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

EXP033_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP033_DIR, "..", ".."))
EXP031_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-031")
EXP019_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-019")

for _p in (REPO_ROOT, EXP033_DIR, EXP031_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_exp031_confirmatory import compute_pages_l  # noqa: E402
from analyze_tsk08_paired_benchmark import participant_bootstrap_ci  # noqa: E402
from exp033_core import BACKBONE_ORDER  # noqa: E402

TAU_GRID = [0.01, 0.02, 0.03, 0.05]


def paired_stats(a: np.ndarray, b: np.ndarray, seed: int = 42, scale: float = 100.0) -> Dict[str, Any]:
    """Compute paired difference (a - b) * scale with 10,000-resample bootstrap CI, Wilcoxon, Cohen's d_z."""
    diff = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) * scale
    n = len(diff)
    mean_d = float(np.mean(diff))
    sd_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    dz = float(mean_d / sd_d) if sd_d > 0 else 0.0
    ci_lo, ci_hi = participant_bootstrap_ci(diff, n_boot=10000, seed=seed)
    try:
        _, p_wil = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        p_wil = float(p_wil)
    except Exception:
        p_wil = 1.0
    return {
        "n_subjects": int(n),
        "mean_a_pct": float(np.mean(a) * scale),
        "mean_b_pct": float(np.mean(b) * scale),
        "delta_pp": mean_d,
        "sd_pp": sd_d,
        "ci95_low_pp": float(ci_lo),
        "ci95_high_pp": float(ci_hi),
        "wilcoxon_p_raw_two_sided": p_wil,
        "cohens_dz": dz,
        "n_improved": int(np.sum(diff > 0)),
        "pct_improved": float(100.0 * np.mean(diff > 0)),
    }


def locked_verdict(res: Dict[str, Any]) -> str:
    """Section 7 locked verdict rule (identical to H-A)."""
    if res["delta_pp"] >= 1.5 and res["ci95_low_pp"] > 0.0:
        return "SUPPORTED"
    if res["ci95_high_pp"] < 1.5:
        return "NOT SUPPORTED (practically)"
    return "INCONCLUSIVE"


def analyze_backbone(backbone: str) -> Dict[str, Any]:
    csv_path = os.path.join(EXP033_DIR, f"{backbone}_subject_seed_metrics.csv")
    df = pd.read_csv(csv_path)

    # Subject-level averages across 3 seeds
    ft2 = df[df["arm"] == "FT-2"].groupby("subject_id", as_index=False).mean(numeric_only=True)
    ft_other = df[df["arm"] == "FT-other-2"].groupby("subject_id", as_index=False).mean(numeric_only=True)

    m_ha2 = pd.merge(ft2, ft_other, on="subject_id", suffixes=("_ft2", "_other"))
    ha2_res = paired_stats(m_ha2["ft_macro_f1_ft2"].values, m_ha2["zs_macro_f1_ft2"].values)
    sspec_res = paired_stats(m_ha2["ft_macro_f1_ft2"].values, m_ha2["ft_macro_f1_other"].values)
    donor_harm_res = paired_stats(m_ha2["ft_macro_f1_other"].values, m_ha2["zs_macro_f1_ft2"].values)

    # Depth curve on available cohorts E_k and fixed common cohort E4 (N=33)
    ek_rows: List[Dict[str, Any]] = []
    e4_rows: List[Dict[str, Any]] = []
    e4_subs = sorted(df[(df["arm"] == "FT-4")]["subject_id"].unique())
    e4_matrix_cols: List[np.ndarray] = []

    for k in (1, 2, 3, 4):
        df_k = df[df["arm"] == f"FT-{k}"].groupby("subject_id", as_index=False).mean(numeric_only=True)
        res_k = paired_stats(df_k["ft_macro_f1"].values, df_k["zs_macro_f1"].values)
        ek_rows.append({"backbone": backbone, "cohort": f"E_{k}", "k": k, **res_k})

        df_e4_k = df_k[df_k["subject_id"].isin(e4_subs)].sort_values("subject_id")
        res_e4_k = paired_stats(df_e4_k["e4_ft_macro_f1"].values, df_e4_k["e4_zs_macro_f1"].values)
        e4_rows.append({"backbone": backbone, "cohort": "E4", "k": k, **res_e4_k})
        e4_matrix_cols.append((df_e4_k["e4_ft_macro_f1"].values - df_e4_k["e4_zs_macro_f1"].values) * 100.0)

    e4_mat = np.column_stack(e4_matrix_cols)
    pages_l = compute_pages_l(e4_mat)

    # GATE-tau sweep on FT-2
    gate_rows: List[Dict[str, Any]] = []
    for tau in TAU_GRID:
        abstained = ft2["min_cal_stage_frac"].values < tau
        gate_f1 = np.where(abstained, ft2["zs_macro_f1"].values, ft2["ft_macro_f1"].values)
        delta_pp = (gate_f1 - ft2["zs_macro_f1"].values) * 100.0
        gate_rows.append({
            "backbone": backbone,
            "tau_pct": tau * 100.0,
            "n_abstained": int(np.sum(abstained)),
            "mean_delta_pp": float(np.mean(delta_pp)),
            "min_delta_pp": float(np.min(delta_pp)),
            "p10_delta_pp": float(np.percentile(delta_pp, 10)),
            "n_harm_le_minus5pp": int(np.sum(delta_pp <= -5.0)),
        })

    # Per-stage F1 (FT-2 vs ZS)
    per_stage = {}
    for stg in ("wake", "light", "deep", "rem"):
        per_stage[stg] = paired_stats(ft2[f"ft_f1_{stg}"].values, ft2[f"zs_f1_{stg}"].values)

    # Night-index breakdown (FT-2 vs ZS)
    night_df = pd.read_csv(os.path.join(EXP033_DIR, f"{backbone}_night_index_metrics.csv"))
    night_sub = night_df.groupby(["subject_id", "eval_night_pos"], as_index=False).mean(numeric_only=True)
    night_summary = []
    for pos, grp in night_sub.groupby("eval_night_pos"):
        if len(grp) >= 3:
            r = paired_stats(grp["ft_macro_f1"].values, grp["zs_macro_f1"].values)
            night_summary.append({"eval_night_pos": int(pos), **r})

    # Selected configs
    with open(os.path.join(EXP033_DIR, f"{backbone}_selected_configs.json"), "r", encoding="utf-8") as f:
        sel_cfgs = json.load(f)

    return {
        "backbone": backbone,
        "selected_configs": {str(k): v["selected"] for k, v in sel_cfgs.items()},
        "H_A2": ha2_res,
        "S_SPEC": sspec_res,
        "donor_harm_FT_other_2_minus_ZS": donor_harm_res,
        "depth_curve_Ek": ek_rows,
        "depth_curve_E4": e4_rows,
        "pages_L_E4": pages_l,
        "gate_sweep": gate_rows,
        "per_stage_FT2_vs_ZS": per_stage,
        "night_index_FT2_vs_ZS": night_summary,
    }


def make_forest_plot(backbone_results: Dict[str, Any], legacy_exp019: Dict[str, Any], out_path: str) -> None:
    labels = []
    deltas = []
    lo_errs = []
    hi_errs = []
    colors = []

    order = [
        ("W100 (primary)", backbone_results["W100"]["H_A2"], "#1D4ED8"),
        ("p1_100k (descriptive)", backbone_results["p1_100k"]["H_A2"], "#4B5563"),
        ("P190 (descriptive)", backbone_results["P190"]["H_A2"], "#4B5563"),
        ("full_v5 (descriptive)", backbone_results["full_v5"]["H_A2"], "#4B5563"),
        ("Model A (older recipe, EXP-019)", legacy_exp019, "#9CA3AF"),
    ]

    for lbl, res, col in reversed(order):
        labels.append(lbl)
        d = res["delta_pp"]
        deltas.append(d)
        lo_errs.append(d - res["ci95_low_pp"])
        hi_errs.append(res["ci95_high_pp"] - d)
        colors.append(col)

    fig, ax = plt.subplots(figsize=(8.5, 4.2), dpi=180)
    y = np.arange(len(labels))
    ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0, label="Null (0 pp)")
    ax.axvline(1.5, color="#DC2626", linestyle=":", linewidth=1.2, label="H-A2 threshold (+1.5 pp)")

    for i in range(len(labels)):
        ax.errorbar(
            deltas[i], y[i],
            xerr=[[lo_errs[i]], [hi_errs[i]]],
            fmt="o", color=colors[i], ecolor=colors[i], elinewidth=2.0, capsize=4, markersize=6,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("FT-2 minus ZS Macro-F1 (percentage points, 95% bootstrap CI)", fontsize=9.5)
    ax.set_title("EXP-033: Two-Night Personalization Effect (FT-2 - ZS) Across Backbones (N=46)", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def make_depth_curve_plot(backbone_results: Dict[str, Any], out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=180)
    styles = {
        "W100": ("#1D4ED8", "-o", 2.2),
        "p1_100k": ("#059669", "--s", 1.5),
        "P190": ("#D97706", "-.^", 1.5),
        "full_v5": ("#7C3AED", ":d", 1.5),
    }
    for b in BACKBONE_ORDER:
        rows = backbone_results[b]["depth_curve_E4"]
        ks = [r["k"] for r in rows]
        ds = [r["delta_pp"] for r in rows]
        col, fmt, lw = styles[b]
        ax.plot(ks, ds, fmt, color=col, linewidth=lw, markersize=5, label=f"{b} (E4, N=33)")
    ax.axhline(0.0, color="#111827", linestyle="--", linewidth=0.9)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Calibration Nights (k)", fontsize=9.5)
    ax.set_ylabel("FT-k minus ZS Macro-F1 on E4 (pp)", fontsize=9.5)
    ax.set_title("EXP-033: Calibration-Depth Curve on Fixed Common Cohort E4 (N=33)", fontsize=10.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    backbone_results = {b: analyze_backbone(b) for b in BACKBONE_ORDER}

    # Confirmatory verdicts strictly on W100
    w100 = backbone_results["W100"]
    verdict_ha2 = locked_verdict(w100["H_A2"])
    verdict_sspec = locked_verdict(w100["S_SPEC"])
    depth_claimed = (
        w100["pages_L_E4"]["p_two_sided"] < 0.05
        and w100["depth_curve_E4"][-1]["ci95_low_pp"] > 0.0
    )

    # Legacy EXP-019 reference ("older recipe")
    exp019_sub = pd.read_csv(os.path.join(EXP019_DIR, "exp019_subject_metrics.csv"))
    legacy_exp019 = paired_stats(exp019_sub["FT-2"].values, exp019_sub["ZS"].values)

    # SleepAccel Part B
    with open(os.path.join(EXP033_DIR, "W100_sleepaccel_summary.json"), "r", encoding="utf-8") as f:
        sa_summary = json.load(f)
    with open(os.path.join(EXP033_DIR, "W100_sleepaccel_paired_comparisons.json"), "r", encoding="utf-8") as f:
        sa_paired = json.load(f)
    sa_ba = pd.read_csv(os.path.join(EXP033_DIR, "W100_sleepaccel_bland_altman.csv"))

    # Save tabular CSVs
    bb_rows = []
    for b in BACKBONE_ORDER:
        r = backbone_results[b]
        bb_rows.append({
            "backbone": b,
            "role": "primary" if b == "W100" else "descriptive",
            "ft2_zs_delta_pp": r["H_A2"]["delta_pp"],
            "ft2_zs_ci_lo": r["H_A2"]["ci95_low_pp"],
            "ft2_zs_ci_hi": r["H_A2"]["ci95_high_pp"],
            "ft2_zs_p_raw_2s": r["H_A2"]["wilcoxon_p_raw_two_sided"],
            "ft2_zs_dz": r["H_A2"]["cohens_dz"],
            "ft2_other_delta_pp": r["S_SPEC"]["delta_pp"],
            "ft2_other_ci_lo": r["S_SPEC"]["ci95_low_pp"],
            "ft2_other_ci_hi": r["S_SPEC"]["ci95_high_pp"],
            "ft2_other_p_raw_2s": r["S_SPEC"]["wilcoxon_p_raw_two_sided"],
            "other_zs_delta_pp": r["donor_harm_FT_other_2_minus_ZS"]["delta_pp"],
            "other_zs_p_raw_2s": r["donor_harm_FT_other_2_minus_ZS"]["wilcoxon_p_raw_two_sided"],
            "pages_L_E4_p_raw_2s": r["pages_L_E4"]["p_two_sided"],
            "pages_L_E4_p_raw_1s": r["pages_L_E4"]["p_one_sided"],
        })
    pd.DataFrame(bb_rows).to_csv(os.path.join(EXP033_DIR, "exp033_backbone_summary.csv"), index=False)

    depth_rows = []
    for b in BACKBONE_ORDER:
        depth_rows.extend(backbone_results[b]["depth_curve_E4"])
        depth_rows.extend(backbone_results[b]["depth_curve_Ek"])
    pd.DataFrame(depth_rows).to_csv(os.path.join(EXP033_DIR, "exp033_depth_curve_summary.csv"), index=False)

    gate_rows = []
    for b in BACKBONE_ORDER:
        gate_rows.extend(backbone_results[b]["gate_sweep"])
    pd.DataFrame(gate_rows).to_csv(os.path.join(EXP033_DIR, "exp033_gate_sweep.csv"), index=False)

    make_forest_plot(backbone_results, legacy_exp019, os.path.join(EXP033_DIR, "exp033_backbone_forest_plot.png"))
    make_depth_curve_plot(backbone_results, os.path.join(EXP033_DIR, "exp033_depth_curve.png"))

    summary = {
        "experiment_id": "EXP-033",
        "prereg_lock_commit": "c1b784305812d3c7208ed73c24a4b08a85c1ef3c",
        "verdicts": {
            "H_A2": verdict_ha2,
            "S_SPEC": verdict_sspec,
            "depth_effect_claimable": depth_claimed,
        },
        "backbones": backbone_results,
        "legacy_exp019_older_recipe": legacy_exp019,
        "sleepaccel_part_b": {
            "summary": sa_summary,
            "paired_comparisons": sa_paired["f1_comparisons"],
        },
    }
    with open(os.path.join(EXP033_DIR, "exp033_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Generate EXP033_FINAL_REPORT.md
    lines: List[str] = []
    lines.append("# EXP-033 Final Report: Personalization and External Evaluation on Recipe-Consistent W100")
    lines.append("")
    lines.append("- **Pre-registration:** `docs/designs/PREREG_EXP-033_W100_PERSONALIZATION_EXTERNAL.md` (lock commit `c1b784305812d3c7208ed73c24a4b08a85c1ef3c`)")
    lines.append(f"- **H-A2 Verdict (`W100` `FT-2 - ZS`, N=46):** **{verdict_ha2}** (`Δ = {w100['H_A2']['delta_pp']:+.2f} pp`, 95% CI `[{w100['H_A2']['ci95_low_pp']:+.2f}, {w100['H_A2']['ci95_high_pp']:+.2f}]`, Wilcoxon raw two-sided `p = {w100['H_A2']['wilcoxon_p_raw_two_sided']:.4e}`, `d_z = {w100['H_A2']['cohens_dz']:.3f}`)")
    lines.append(f"- **S-SPEC Verdict (`W100` `FT-2 - FT-other-2`, N=46):** **{verdict_sspec}** (`Δ = {w100['S_SPEC']['delta_pp']:+.2f} pp`, 95% CI `[{w100['S_SPEC']['ci95_low_pp']:+.2f}, {w100['S_SPEC']['ci95_high_pp']:+.2f}]`, Wilcoxon raw two-sided `p = {w100['S_SPEC']['wilcoxon_p_raw_two_sided']:.4e}`, `d_z = {w100['S_SPEC']['cohens_dz']:.3f}`)")
    lines.append(f"- **Calibration-Depth Claim (`W100` E4 N=33):** **{'CLAIMABLE' if depth_claimed else 'NOT CLAIMABLE'}** (Page's `L = {w100['pages_L_E4']['L']:.1f}`, `z = {w100['pages_L_E4']['z']:+.3f}`, raw two-sided `p = {w100['pages_L_E4']['p_two_sided']:.4e}`, raw one-sided `p = {w100['pages_L_E4']['p_one_sided']:.4e}`; E4 `k=4` `Δ = {w100['depth_curve_E4'][-1]['delta_pp']:+.2f} pp` `[{w100['depth_curve_E4'][-1]['ci95_low_pp']:+.2f}, {w100['depth_curve_E4'][-1]['ci95_high_pp']:+.2f}]`)")
    lines.append("")
    lines.append("## 1. Backbone Summary Table (FT-2 vs ZS & FT-other-2, N=46)")
    lines.append("")
    lines.append("| Backbone | Role | FT-2 - ZS (pp) | 95% CI | p (raw, 2-sided) | d_z | FT-2 - FT-other-2 (pp) | 95% CI | FT-other-2 - ZS (pp) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for b in BACKBONE_ORDER:
        r = backbone_results[b]
        role = "Primary" if b == "W100" else "Descriptive"
        h = r["H_A2"]
        s = r["S_SPEC"]
        dh = r["donor_harm_FT_other_2_minus_ZS"]
        lines.append(
            f"| `{b}` | {role} | `{h['delta_pp']:+.2f}` | `[{h['ci95_low_pp']:+.2f}, {h['ci95_high_pp']:+.2f}]` | "
            f"`{h['wilcoxon_p_raw_two_sided']:.4e}` | `{h['cohens_dz']:.3f}` | "
            f"`{s['delta_pp']:+.2f}` | `[{s['ci95_low_pp']:+.2f}, {s['ci95_high_pp']:+.2f}]` | `{dh['delta_pp']:+.2f}` |"
        )
    lines.append(
        f"| `Model A` (EXP-019) | Older recipe | `{legacy_exp019['delta_pp']:+.2f}` | "
        f"`[{legacy_exp019['ci95_low_pp']:+.2f}, {legacy_exp019['ci95_high_pp']:+.2f}]` | "
        f"`{legacy_exp019['wilcoxon_p_raw_two_sided']:.4e}` | `{legacy_exp019['cohens_dz']:.3f}` | — | — | — |"
    )
    lines.append("")
    lines.append("## 2. W100 Calibration-Depth Curve (Descriptive)")
    lines.append("")
    lines.append("| Cohort | k | N | ZS (%) | FT-k (%) | Δ (pp) | 95% CI | p (raw, 2-sided) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in w100["depth_curve_E4"] + w100["depth_curve_Ek"]:
        lines.append(
            f"| `{row['cohort']}` | {row['k']} | {row['n_subjects']} | `{row['mean_b_pct']:.2f}` | `{row['mean_a_pct']:.2f}` | "
            f"`{row['delta_pp']:+.2f}` | `[{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}]` | `{row['wilcoxon_p_raw_two_sided']:.4e}` |"
        )
    lines.append("")
    lines.append("## 3. GATE-τ Sweep (Descriptive, N=46)")
    lines.append("")
    lines.append("| Backbone | τ (%) | Abstained | Mean Δ vs ZS (pp) | Min Δ (pp) | 10th-pct Δ (pp) | N(Δ ≤ -5 pp) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in gate_rows:
        lines.append(
            f"| `{row['backbone']}` | {row['tau_pct']:.0f}% | {row['n_abstained']} | `{row['mean_delta_pp']:+.2f}` | "
            f"`{row['min_delta_pp']:+.2f}` | `{row['p10_delta_pp']:+.2f}` | {row['n_harm_le_minus5pp']} |"
        )
    lines.append("")
    lines.append("## 4. Part B: W100 Zero-Shot on SleepAccel (em_off, Descriptive)")
    lines.append("")
    lines.append("| Cohort | Eval Mode | Reference | N | W100 (%) | Ref (%) | Δ (pp) | 95% CI | p (raw, 2-sided) |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for c in sa_paired["f1_comparisons"]:
        lines.append(
            f"| `{c['cohort']}` | `{c['eval_mode']}` | `{c['reference']}` | {c['n_subjects']} | "
            f"`{c['primary_mean']:.2f}` | `{c['ref_mean']:.2f}` | `{c['delta']:+.2f}` | "
            f"`[{c['ci95_low']:+.2f}, {c['ci95_high']:+.2f}]` | `{c['wilcoxon_p']:.4e}` |"
        )
    lines.append("")
    lines.append("### W100 SleepAccel Bland-Altman (qc_clean n=25, mean_single_model; margins: none; descriptive only)")
    lines.append("")
    lines.append("| Metric | Bias | SD | LoA Lower | LoA Upper |")
    lines.append("|---|---:|---:|---:|---:|")
    ba_sub = sa_ba[(sa_ba["cohort"] == "qc_clean") & (sa_ba["eval_mode"] == "mean_single_model")]
    for _, row in ba_sub.iterrows():
        lines.append(
            f"| `{row['metric']}` | `{row['bias']:+.2f}` | `{row['sd_diff']:.2f}` | `{row['loa_low']:+.2f}` | `{row['loa_high']:+.2f}` |"
        )

    report_path = os.path.join(EXP033_DIR, "EXP033_FINAL_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[EXP-033 Analysis] Saved {report_path} and exp033_summary.json.", flush=True)


if __name__ == "__main__":
    main()
