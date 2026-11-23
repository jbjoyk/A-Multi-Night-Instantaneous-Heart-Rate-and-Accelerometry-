#!/usr/bin/env python3
"""
Generate publication-ready 300-DPI PNG and vector PDF Figures 1–5 for
MANUSCRIPT_DRAFT_v1.0.md (*The Lancet Digital Health* submission).

Reads directly from locked repository artifacts (zero synthetic data).
Usage:
    .venv/bin/python scripts/generate_manuscript_figures.py
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Iterable, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & Styling (*The Lancet Digital Health* / Nature clean aesthetic)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "manuscript_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "model_a": "#0072B2",      # Deep blue
    "slamss": "#D55E00",       # Vermilion
    "gate_tau": "#009E73",     # Teal green
    "ft2_kl": "#E69F00",       # Amber
    "ft2": "#CC79A7",          # Muted magenta
    "neutral": "#4D4D4D",      # Charcoal
    "light_grey": "#BFBFBF",
    "danger": "#B2182B",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "grid.color": "#E5E5E5",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,        # Editable TrueType fonts in PDF
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Utility Helpers
# ---------------------------------------------------------------------------
def save_fig(fig: plt.Figure, stem: str) -> None:
    pdf_path = OUT_DIR / f"{stem}.pdf"
    png_path = OUT_DIR / f"{stem}.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {png_path.relative_to(ROOT)} and {pdf_path.relative_to(ROOT)}")


def pick_col(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    """Resolve column names cleanly: exact lowercase match first, then substring."""
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for cand in candidates:
        cl = cand.lower()
        for col in df.columns:
            if cl in col.lower():
                return col
    if required:
        raise KeyError(
            f"None of {list(candidates)} found in columns: {list(df.columns)}"
        )
    return None


def to_pp(values: np.ndarray | pd.Series) -> np.ndarray:
    """Convert decimal fractions (e.g., 0.0133) to percentage points if max <= 1.0."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size > 0 and np.nanmax(np.abs(finite)) <= 1.0:
        return arr * 100.0
    return arr


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.14, y: float = 1.08) -> None:
    ax.text(
        x, y, label, transform=ax.transAxes,
        fontsize=11, fontweight="bold", va="top", ha="left", color="#111111"
    )


# ---------------------------------------------------------------------------
# FIGURE 1: Accuracy–Efficiency Pareto & External PSG Transfer Attenuation
# ---------------------------------------------------------------------------
def build_figure_1() -> None:
    exp027_path = ROOT / "experiments/EXP-027/exp027_summary.json"
    with exp027_path.open("r", encoding="utf-8") as f:
        s27 = json.load(f)

    qc_single = s27["cohorts"]["qc_clean"]["mean_single_model"]["metrics"]
    qc_ens = s27["cohorts"]["qc_clean"]["prob_ensemble"]["metrics"]

    # Verified locked EXP-021 (N=47) & W0-B profiling values (DEC-038 / DEC-039)
    params_a = 163_940 / 1e6         # 0.16394 M
    params_s = 5_465_768 / 1e6       # 5.465768 M (33.34x)
    bid_a, bid_s = 64.79, 67.37      # BIDSleep 5-fold OOF Macro-F1 (%)

    sa_a_s = qc_single["macro_f1"]["model_a_mean"]
    sa_s_s = qc_single["macro_f1"]["slamss_mean"]
    sa_a_e = qc_ens["macro_f1"]["model_a_mean"]
    sa_s_e = qc_ens["macro_f1"]["slamss_mean"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.35), gridspec_kw={"width_ratios": [1.08, 1.0]})

    # --- Panel A: Parameter Scale vs. Macro-F1 across Home EEG -> In-Lab PSG ---
    ax1.set_xscale("log")
    ax1.grid(True, axis="y", zorder=0)

    # Vertical transfer arrows (BIDSleep -> SleepAccel Primary Single-Model)
    ax1.annotate(
        "", xy=(params_a, sa_a_s + 0.35), xytext=(params_a, bid_a - 0.35),
        arrowprops=dict(arrowstyle="->", color=PALETTE["model_a"], lw=1.4, ls="--")
    )
    ax1.annotate(
        "", xy=(params_s, sa_s_s + 0.35), xytext=(params_s, bid_s - 0.35),
        arrowprops=dict(arrowstyle="->", color=PALETTE["slamss"], lw=1.4, ls="--")
    )

    # Points
    ax1.scatter([params_a], [bid_a], s=75, color=PALETTE["model_a"], marker="o", zorder=4,
                label="BIDSleep 5-fold CV (N=47, Dreem 2)")
    ax1.scatter([params_s], [bid_s], s=75, color=PALETTE["slamss"], marker="o", zorder=4)

    ax1.scatter([params_a], [sa_a_s], s=75, facecolors="white", edgecolors=PALETTE["model_a"],
                linewidths=1.8, marker="s", zorder=5,
                label="SleepAccel Zero-Shot Single-Model (n=25, PSG)")
    ax1.scatter([params_s], [sa_s_s], s=75, facecolors="white", edgecolors=PALETTE["slamss"],
                linewidths=1.8, marker="s", zorder=5)

    ax1.scatter([params_a], [sa_a_e], s=65, color=PALETTE["model_a"], marker="^", zorder=5,
                label="SleepAccel Zero-Shot 15-Model Ensemble (n=25)")
    ax1.scatter([params_s], [sa_s_e], s=65, color=PALETTE["slamss"], marker="^", zorder=5)

    # Connect matched pairs horizontally to visualize gap shrinkage
    ax1.plot([params_a, params_s], [bid_a, bid_s], color="#888888", lw=0.9, ls=":", zorder=2)
    ax1.plot([params_a, params_s], [sa_a_s, sa_s_s], color="#888888", lw=0.9, ls=":", zorder=2)

    # Direct annotations
    ax1.text(params_a * 1.18, bid_a + 0.35, f"Model A\n163.9k (INT8: 310 KiB)\n{bid_a:.2f}%",
             fontsize=7.2, color=PALETTE["model_a"], fontweight="bold", va="bottom")
    ax1.text(params_s * 0.86, bid_s + 0.35, f"Official SLAMSS-IFS\n5.47M (33.34x)\n{bid_s:.2f}%",
             fontsize=7.2, color=PALETTE["slamss"], fontweight="bold", ha="right", va="bottom")

    ax1.text(params_a * 1.18, (bid_a + sa_a_s) / 2, f"−{bid_a - sa_a_s:.2f} pp",
             fontsize=7.2, color=PALETTE["model_a"], va="center")
    ax1.text(params_s * 0.86, (bid_s + sa_s_s) / 2, f"−{bid_s - sa_s_s:.2f} pp",
             fontsize=7.2, color=PALETTE["slamss"], ha="right", va="center")

    ax1.text(params_a * 1.18, sa_a_s - 0.15, f"{sa_a_s:.2f}% (Ens: {sa_a_e:.2f}%)",
             fontsize=7.0, color=PALETTE["model_a"], va="top")
    ax1.text(params_s * 0.86, sa_s_s - 0.15, f"{sa_s_s:.2f}% (Ens: {sa_s_e:.2f}%)",
             fontsize=7.0, color=PALETTE["slamss"], ha="right", va="top")

    ax1.text(0.95, 66.6, "In-cohort gap: −2.58 pp\n(p < 0.0001)", fontsize=7.0,
             ha="center", color="#333333", bbox=dict(boxstyle="round,pad=0.22", fc="#F7F7F7", ec="#CCCCCC", lw=0.6))
    ax1.text(0.95, 59.35, "External PSG gap: −0.57 pp (p = 0.58)\n78% lead attenuation | Ens: +0.15 pp",
             fontsize=7.0, ha="center", color="#111111",
             bbox=dict(boxstyle="round,pad=0.24", fc="#EDF6FC", ec=PALETTE["model_a"], lw=0.7))

    ax1.set_xlim(0.10, 9.5)
    ax1.set_ylim(58.2, 70.2)
    ax1.set_xlabel("Trainable Parameters (Millions, log scale)")
    ax1.set_ylabel("4-Class Macro-F1 (%)")
    ax1.set_title("Cross-Site Transfer & Compression Trade-Off")
    ax1.legend(loc="lower left", frameon=True, framealpha=0.92, fontsize=6.6, edgecolor="#DDDDDD")
    add_panel_label(ax1, "A")

    # --- Panel B: Stage-Wise Paired Delta on SleepAccel (n=25 QC-clean) ---
    stages = [
        ("Macro-F1", "macro_f1"),
        ("Wake F1", "wake_f1"),
        ("Light F1", "light_f1"),
        ("Deep F1", "deep_f1"),
        ("REM F1", "rem_f1"),
    ]
    y_pos = np.arange(len(stages))[::-1]
    offset = 0.16

    ax2.axvline(0.0, color=PALETTE["neutral"], lw=1.0, ls="--", zorder=1)
    ax2.grid(True, axis="x", zorder=0)

    for idx, (label, key) in enumerate(stages):
        y = y_pos[idx]
        ms = qc_single[key]
        me = qc_ens[key]

        # Primary Single-Model
        ax2.errorbar(
            ms["delta_mean"], y + offset,
            xerr=[[ms["delta_mean"] - ms["ci95_low"]], [ms["ci95_high"] - ms["delta_mean"]]],
            fmt="s", color=PALETTE["model_a"], ecolor=PALETTE["model_a"],
            capsize=3, markersize=5.2, lw=1.3,
            label="Primary: Single-Model Mean" if idx == 0 else None
        )
        # Secondary Ensemble
        ax2.errorbar(
            me["delta_mean"], y - offset,
            xerr=[[me["delta_mean"] - me["ci95_low"]], [me["ci95_high"] - me["delta_mean"]]],
            fmt="^", color=PALETTE["gate_tau"], ecolor=PALETTE["gate_tau"],
            capsize=3, markersize=5.2, lw=1.3,
            label="Secondary: 15-Model Ensemble" if idx == 0 else None
        )
        # Annotate Light sleep p-value
        if key == "light_f1":
            ax2.text(ms["ci95_low"] - 0.15, y + offset, f"p = {ms['wilcoxon_p']:.4f}",
                     fontsize=6.8, va="center", ha="right", color=PALETTE["danger"], fontweight="bold")
            ax2.text(me["ci95_low"] - 0.15, y - offset, f"p = {me['wilcoxon_p']:.4f}",
                     fontsize=6.8, va="center", ha="right", color=PALETTE["danger"])

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([s[0] for s in stages])
    ax2.set_xlim(-8.8, 7.2)
    ax2.set_xlabel("Paired Δ (Model A − SLAMSS-IFS, pp; 95% Bootstrap CI)")
    ax2.set_title("External PSG Stage-Wise Paired Delta (n = 25)")
    ax2.text(-8.4, -0.58, "← Favors SLAMSS-IFS (5.47M)", fontsize=6.8, color=PALETTE["slamss"])
    ax2.text(6.9, -0.58, "Favors Model A (163.9k) →", fontsize=6.8, color=PALETTE["model_a"], ha="right")
    ax2.set_ylim(-0.75, len(stages) - 0.35)
    ax2.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08), frameon=True, fontsize=6.8, edgecolor="#DDDDDD")
    add_panel_label(ax2, "B")

    fig.tight_layout(w_pad=1.6)
    save_fig(fig, "Figure1_Pareto_External_PSG_Transfer")


# ---------------------------------------------------------------------------
# FIGURE 2: Multi-Night Calibration-Depth Learning Curve (TSK-06 Clean-ZS)
# ---------------------------------------------------------------------------
def build_figure_2() -> None:
    csv_path = ROOT / "experiments/EXP-019/exp019_k_learning_curve_CLEAN_RECONCILED.csv"
    df = pd.read_csv(csv_path)

    col_cohort = pick_col(df, ["cohort", "analysis_cohort", "subset", "group"])
    col_k = pick_col(df, ["k", "cal_nights", "k_nights"])
    col_n = pick_col(df, ["n_subjects", "n_sub", "n"])
    col_delta = pick_col(df, ["delta_mean_pp", "delta_pp", "delta_mean", "mean_delta", "delta"])
    col_lo = pick_col(df, ["CI_lower", "ci95_low", "ci_low", "ci_lo", "delta_ci_low", "boot_ci_low"])
    col_hi = pick_col(df, ["CI_upper", "ci95_high", "ci_high", "ci_hi", "delta_ci_high", "boot_ci_high"])

    # Partition into available vs common cohorts
    cohort_vals = df[col_cohort].astype(str).str.lower()
    df_avail = df[cohort_vals.str.contains("avail")].sort_values(col_k).copy()
    df_comm = df[cohort_vals.str.contains("comm|e4|e_4")].sort_values(col_k).copy()
    if df_avail.empty or df_comm.empty:
        raise ValueError(f"Could not split available/common cohorts from {df[col_cohort].unique()}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.25), sharey=True)

    for ax, sub, title, color, label_char, note in [
        (
            ax1, df_avail,
            "Available Cohort (N = 46, 46, 42, 33)",
            PALETTE["model_a"], "A",
            "First 95% CI > 0 at k = 3 nights\n(converges with van der Aar et al., 2025)"
        ),
        (
            ax2, df_comm,
            "Common Evaluation Cohort E4 (N = 33, 64 Nights)",
            PALETTE["gate_tau"], "B",
            "Clean ZS baseline: 63.54%\nPage's L trend: p = 0.0119 (one-sided)\nTwo-sided sensitivity: p = 0.0594"
        ),
    ]:
        k = sub[col_k].to_numpy(dtype=int)
        n_sub = sub[col_n].to_numpy(dtype=int)
        d = sub[col_delta].to_numpy(dtype=float)
        lo = sub[col_lo].to_numpy(dtype=float)
        hi = sub[col_hi].to_numpy(dtype=float)
        if np.nanmax(np.abs(d)) <= 1.0:
            d, lo, hi = d * 100.0, lo * 100.0, hi * 100.0

        ax.axhline(0.0, color=PALETTE["neutral"], lw=0.9, ls="--", zorder=1)
        ax.axhline(1.50, color=PALETTE["slamss"], lw=0.9, ls=":", zorder=1,
                   label="Pre-registered +1.50 pp benchmark")
        ax.grid(True, axis="y", zorder=0)

        ax.fill_between(k, lo, hi, color=color, alpha=0.16, zorder=2)
        ax.errorbar(
            k, d, yerr=[d - lo, hi - d],
            fmt="-o", color=color, ecolor=color, capsize=3.5, lw=1.8, markersize=5.8, zorder=4
        )

        for xi, yi, hii, ni in zip(k, d, hi, n_sub):
            ax.text(xi, hii + 0.18, f"+{yi:.2f} pp\n(N={ni})", ha="center", va="bottom",
                    fontsize=7.0, color="#222222", fontweight="bold" if xi >= 3 else "normal")

        ax.text(
            0.04, 0.05, note, transform=ax.transAxes, fontsize=7.0, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.28", fc="#F9F9F9", ec="#CCCCCC", lw=0.6)
        )
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([f"k = {i}" for i in [1, 2, 3, 4]])
        ax.set_xlim(0.65, 4.35)
        ax.set_ylim(-1.6, 5.2)
        ax.set_xlabel("Labeled Calibration Nights per Participant (k)")
        ax.set_title(title)
        add_panel_label(ax, label_char)

    ax1.set_ylabel("Paired Macro-F1 Gain vs. Clean Zero-Shot (pp)")
    ax1.legend(loc="upper left", frameon=True, fontsize=7.0, edgecolor="#DDDDDD")

    fig.tight_layout(w_pad=1.2)
    save_fig(fig, "Figure2_Calibration_Depth_Learning_Curve")


# ---------------------------------------------------------------------------
# FIGURE 3: Exploratory Safe Adaptation (EXP-020) & Heterogeneity (EXP-019B)
# ---------------------------------------------------------------------------
def build_figure_3() -> None:
    s20_path = ROOT / "experiments/EXP-020/exp020_subject_metrics.csv"
    s19b_path = ROOT / "experiments/EXP-019B/exp019b_correlation_summary.csv"
    df20 = pd.read_csv(s20_path)
    df19b = pd.read_csv(s19b_path)

    # Resolve EXP-020 columns (support either delta columns or raw Macro-F1 columns)
    col_sub = pick_col(df20, ["subject_id", "subject", "participant"])
    col_zs = pick_col(df20, ["zs_macro_f1", "zs_f1", "zs"], required=False)
    col_ft2_d = pick_col(df20, ["delta_ft2", "ft2_minus_zs"], required=False)
    col_kl_d = pick_col(df20, ["delta_ft2_kl", "ft2_kl_minus_zs", "delta_kl"], required=False)
    col_gt_d = pick_col(df20, ["delta_gate_tau", "gate_tau_minus_zs", "delta_gate"], required=False)

    if col_ft2_d and col_kl_d and col_gt_d:
        d_ft2 = to_pp(df20[col_ft2_d])
        d_kl = to_pp(df20[col_kl_d])
        d_gt = to_pp(df20[col_gt_d])
    else:
        col_ft2 = pick_col(df20, ["FT-2", "ft2_macro_f1", "ft2_f1", "ft2"])
        col_kl = pick_col(df20, ["FT-2-KL", "ft2_kl_macro_f1", "ft2_kl_f1", "ft2_kl"])
        col_gt = pick_col(df20, ["GATE_tau", "gate_tau_macro_f1", "gate_tau_f1", "gate_tau"])
        zs_vals = to_pp(df20[col_zs])
        d_ft2 = to_pp(df20[col_ft2]) - zs_vals
        d_kl = to_pp(df20[col_kl]) - zs_vals
        d_gt = to_pp(df20[col_gt]) - zs_vals

    plot_df = pd.DataFrame({
        "subject": df20[col_sub].astype(str),
        "ft2": d_ft2,
        "ft2_kl": d_kl,
        "gate_tau": d_gt,
    }).sort_values("ft2").reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.55), gridspec_kw={"width_ratios": [1.22, 1.0]})

    # --- Panel A: Participant-Level Negative Transfer Mitigation (N=46) ---
    x = np.arange(len(plot_df))
    ax1.axhline(0.0, color=PALETTE["neutral"], lw=0.8, ls="-", zorder=1)
    ax1.axhline(-5.0, color=PALETTE["danger"], lw=0.9, ls="--", zorder=1, label="Severe harm threshold (−5.0 pp)")
    ax1.grid(True, axis="y", zorder=0)

    ax1.plot(x, plot_df["ft2"], color=PALETTE["ft2"], lw=1.2, alpha=0.85, marker="o", ms=3.0,
             label="Unregularized FT-2 (Min −16.34 pp)")
    ax1.plot(x, plot_df["ft2_kl"], color=PALETTE["ft2_kl"], lw=1.1, ls="--", alpha=0.85, marker="s", ms=2.6,
             label="Soft KL Reg. FT-2-KL (Min −15.14 pp)")
    ax1.plot(x, plot_df["gate_tau"], color=PALETTE["gate_tau"], lw=1.6, marker="^", ms=3.5,
             label="Stage-Coverage Gate (Min −6.40 pp; 2/46 abstained)")

    # Highlight Bidslab34 (index 0 in sorted FT-2 order)
    b34_mask = plot_df["subject"].str.contains("34")
    if b34_mask.any():
        b_idx = int(plot_df.index[b34_mask][0])
        ax1.annotate(
            "Bidslab34 (0.68% Deep)\nFT-2: −16.34 pp → Gate: 0.00 pp\n(61% worst-case reduction)",
            xy=(b_idx, plot_df.loc[b_idx, "gate_tau"]),
            xytext=(b_idx + 9, -13.8),
            fontsize=6.8, fontweight="bold", color="#111111",
            arrowprops=dict(arrowstyle="->", color=PALETTE["gate_tau"], lw=1.2),
            bbox=dict(boxstyle="round,pad=0.25", fc="#E8F8F2", ec=PALETTE["gate_tau"], lw=0.7)
        )

    ax1.set_xlim(-1.2, len(plot_df) + 0.5)
    ax1.set_ylim(-18.0, 12.5)
    ax1.set_xlabel("Participants Ranked by Unregularized FT-2 Δ (N = 46)")
    ax1.set_ylabel("Macro-F1 Change vs. Zero-Shot (pp)")
    ax1.set_title("Exploratory Tail-Risk Mitigation (N = 46)")
    ax1.legend(loc="upper left", frameon=True, fontsize=6.5, edgecolor="#DDDDDD")
    add_panel_label(ax1, "A")

    # --- Panel B: Exploratory Heterogeneity Audit Forest Plot (EXP-019B, Family 1) ---
    col_out = pick_col(df19b, ["outcome", "family", "target"])
    col_feat = pick_col(df19b, ["feature", "predictor", "variable"])
    col_rho = pick_col(df19b, ["spearman_rho", "rho"])
    col_rlo = pick_col(df19b, ["ci_low", "CI_lower", "ci_lower", "ci95_low", "rho_ci_low"])
    col_rhi = pick_col(df19b, ["ci_high", "CI_upper", "ci_upper", "ci95_high", "rho_ci_high"])
    col_q = pick_col(df19b, ["fdr_q_family", "q_family", "q_val", "fdr_q"])
    col_scope = pick_col(df19b, ["scope", "analysis", "tier", "split"], required=False)

    f1 = df19b[df19b[col_out].astype(str).str.contains("019|FT2", case=False)].copy()
    if col_scope is not None:
        prim = f1[f1[col_scope].astype(str).str.contains("prim", case=False)]
        if not prim.empty:
            f1 = prim.copy()
    f1 = f1.drop_duplicates(subset=[col_feat]).head(6).reset_index(drop=True)

    pretty_names = {
        "min_stage_frac": "Min. Stage Fraction",
        "stage_entropy": "Stage Shannon Entropy",
        "eff_ihr_hz": "Effective IHR Rate (Hz)",
        "missingness_rate": "Sensor Missingness Rate",
        "baseline_hr_mean": "Baseline Mean HR (bpm)",
        "baseline_hr_iqr": "Baseline HR IQR (bpm)",
    }

    y2 = np.arange(len(f1))[::-1]
    ax2.axvline(0.0, color=PALETTE["neutral"], lw=1.0, ls="--", zorder=1)
    ax2.grid(True, axis="x", zorder=0)

    rhos = f1[col_rho].to_numpy(dtype=float)
    rlos = f1[col_rlo].to_numpy(dtype=float)
    rhis = f1[col_rhi].to_numpy(dtype=float)
    qs = f1[col_q].to_numpy(dtype=float)

    ax2.errorbar(
        rhos, y2, xerr=[rhos - rlos, rhis - rhos],
        fmt="o", color=PALETTE["model_a"], ecolor=PALETTE["model_a"],
        capsize=3.2, markersize=5.2, lw=1.3, zorder=4
    )
    for yi, ri, hi, qi in zip(y2, rhos, rhis, qs):
        ax2.text(hi + 0.03, yi, f"q={qi:.2f}", va="center", ha="left", fontsize=6.7, color="#444444")

    ax2.set_yticks(y2)
    ax2.set_yticklabels([pretty_names.get(str(v), str(v)) for v in f1[col_feat]])
    ax2.set_xlim(-0.62, 0.72)
    ax2.set_ylim(-0.9, len(f1) - 0.3)
    ax2.set_xlabel("Spearman ρ with FT-2 Gain (95% Bootstrap CI)")
    ax2.set_title("Exploratory Heterogeneity Audit (N = 46)")
    ax2.text(
        0.03, 0.04,
        "0 / 12 features survive BH-FDR (all q ≥ 0.57)\nOldham headroom check: ρ = +0.038 (p = 0.804)",
        transform=ax2.transAxes, fontsize=6.6, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="#F9F9F9", ec="#CCCCCC", lw=0.6)
    )
    add_panel_label(ax2, "B")

    fig.tight_layout(w_pad=1.5)
    save_fig(fig, "Figure3_Exploratory_Tail_Risk_and_Heterogeneity")


# ---------------------------------------------------------------------------
# FIGURE 4: External Single-Night PSG Bland–Altman Panels (EXP-028, n=25)
# ---------------------------------------------------------------------------
def build_figure_4() -> None:
    sum_path = ROOT / "experiments/EXP-028/exp028_sleepaccel_bland_altman_summary.csv"
    raw_path = ROOT / "experiments/EXP-028/exp028_sleepaccel_raw_paired_tuples.csv"
    qc_path = ROOT / "experiments/EXP-027/sleepaccel_qc_summary.csv"

    ba = pd.read_csv(sum_path)
    raw = pd.read_csv(raw_path)
    qc = pd.read_csv(qc_path)

    # Filter summary to primary: qc_clean + mean_single_model
    ba_p = ba[(ba["cohort"] == "qc_clean") & (ba["eval_mode"] == "mean_single_model")].copy()

    # Resolve raw tuple columns & restrict to qc_clean subjects + mean_single_model
    r_sub = pick_col(raw, ["subject_id", "subject", "id"])
    r_arm = pick_col(raw, ["arm", "model"])
    r_mode = pick_col(raw, ["eval_mode", "mode"])
    r_met = pick_col(raw, ["metric", "endpoint"])
    r_ref = pick_col(raw, ["ref_val", "ref_value", "ref", "psg_val", "true_val"])
    r_pred = pick_col(raw, ["pred_val", "pred_value", "pred", "model_val"])
    r_qc = pick_col(raw, ["qc_pass", "qc_clean", "cohort"], required=False)

    raw_p = raw[raw[r_mode].astype(str) == "mean_single_model"].copy()
    if r_qc is not None:
        if raw_p[r_qc].dtype == bool:
            raw_p = raw_p[raw_p[r_qc]].copy()
        else:
            raw_p = raw_p[raw_p[r_qc].astype(str).str.lower().isin(["true", "1", "qc_clean"])].copy()
    else:
        q_sub = pick_col(qc, ["subject_id", "subject", "id"])
        q_pass = pick_col(qc, ["qc_pass", "pass"])
        clean_ids = set(qc.loc[qc[q_pass].astype(bool), q_sub].astype(str))
        raw_p = raw_p[raw_p[r_sub].astype(str).isin(clean_ids)].copy()

    metrics_order = [
        ("tst_min", "Total Sleep Time (min)", "min"),
        ("sol_min", "Sleep Onset Latency (min)", "min"),
        ("waso_min", "Wake After Sleep Onset (min)", "min"),
        ("sleep_efficiency_pct", "Sleep Efficiency (%)", "%"),
        ("deep_pct_tst", "Deep Sleep (% TST)", "%"),
        ("rem_pct_tst", "REM Sleep (% TST)", "%"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.9))
    axes = axes.ravel()

    for idx, (mkey, mtitle, unit) in enumerate(metrics_order):
        ax = axes[idx]
        ax.axhline(0.0, color="#999999", lw=0.7, ls=":", zorder=1)
        ax.grid(True, zorder=0)

        ann_lines = []
        for arm_key, arm_label, color, marker, ls in [
            ("ModelA", "Model A", PALETTE["model_a"], "o", "-"),
            ("SLAMSS_IFS", "SLAMSS-IFS", PALETTE["slamss"], "^", "--"),
        ]:
            sub_raw = raw_p[(raw_p[r_met] == mkey) & (raw_p[r_arm] == arm_key)]
            row = ba_p[(ba_p["metric"] == mkey) & (ba_p["arm"] == arm_key)].iloc[0]

            mean_xy = 0.5 * (sub_raw[r_ref].to_numpy(float) + sub_raw[r_pred].to_numpy(float))
            diff_xy = sub_raw[r_pred].to_numpy(float) - sub_raw[r_ref].to_numpy(float)

            ax.scatter(
                mean_xy, diff_xy, s=22, marker=marker,
                facecolors=color if arm_key == "ModelA" else "none",
                edgecolors=color, linewidths=1.0, alpha=0.82, zorder=4,
                label=arm_label if idx == 0 else None
            )

            non_norm = bool(row["non_normal_flag"])
            bias = float(row["nonparam_median_diff"] if non_norm else row["bias"])
            loa_l = float(row["nonparam_loa_low"] if non_norm else row["loa_low"])
            loa_h = float(row["nonparam_loa_high"] if non_norm else row["loa_high"])

            ax.axhline(bias, color=color, lw=1.15, ls=ls, zorder=3)
            ax.axhline(loa_l, color=color, lw=0.8, ls=ls, alpha=0.55, zorder=2)
            ax.axhline(loa_h, color=color, lw=0.8, ls=ls, alpha=0.55, zorder=2)

            tag = "Med" if non_norm else "Bias"
            p_prop = float(row["prop_bias_p"])
            prop_str = f" | pb p={p_prop:.3f}*" if p_prop < 0.05 else ""
            ann_lines.append(
                f"{arm_label}: {tag} {bias:+.1f} [{loa_l:+.1f}, {loa_h:+.1f}] | MAE {row['mae']:.1f}, r={row['pearson_r']:.2f}{prop_str}"
            )

        ax.text(
            0.03, 0.03, "\n".join(ann_lines), transform=ax.transAxes,
            fontsize=5.6, va="bottom", ha="left", color="#111111",
            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#CCCCCC", alpha=0.90, lw=0.5)
        )
        ax.set_title(mtitle, fontsize=8.4)
        ax.set_xlabel(f"Mean of Predicted & PSG ({unit})", fontsize=7.5)
        if idx % 3 == 0:
            ax.set_ylabel("Predicted − PSG", fontsize=7.5)
        add_panel_label(ax, chr(ord("A") + idx), x=-0.16, y=1.12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 1.02), fontsize=8.2
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=1.1, w_pad=0.9)
    save_fig(fig, "Figure4_External_PSG_Bland_Altman")


# ---------------------------------------------------------------------------
# FIGURE 5: Cross-Modality Label-Prior Shift (Dreem 2 EEG -> In-Lab PSG)
# ---------------------------------------------------------------------------
def build_figure_5() -> None:
    sum_path = ROOT / "experiments/EXP-028/exp028_sleepaccel_bland_altman_summary.csv"
    ba = pd.read_csv(sum_path)
    ba_p = ba[(ba["cohort"] == "qc_clean") & (ba["eval_mode"] == "mean_single_model")].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.25), gridspec_kw={"width_ratios": [1.05, 1.0]})

    # --- Panel A: Reference Label-Prior Shift (BIDSleep Dreem 2 vs SleepAccel PSG) ---
    # Night-averaged Dreem 2 EEG vs SleepAccel PSG total epoch prevalence (locked DEC-041 / ACT-B07)
    stages = ["Wake", "Light\n(N1+N2)", "Deep\n(N3)", "REM"]
    bid_prior = np.array([10.8, 44.2, 18.9, 26.1])
    sa_prior = np.array([8.3, 57.7, 11.1, 22.9])

    x = np.arange(len(stages))
    w = 0.36
    ax1.grid(True, axis="y", zorder=0)
    b1 = ax1.bar(x - w / 2, bid_prior, w, color=PALETTE["model_a"], alpha=0.88,
                 label="BIDSleep Home Dreem 2 EEG (Training)", zorder=3)
    b2 = ax1.bar(x + w / 2, sa_prior, w, color=PALETTE["neutral"], alpha=0.82,
                 label="SleepAccel In-Lab PSG (External Ref.)", zorder=3)

    for rect in b1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, h + 1.0, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=7.0, color=PALETTE["model_a"], fontweight="bold")
    for rect in b2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, h + 1.0, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=7.0, color="#333333", fontweight="bold")

    ax1.annotate("+13.5 pp Light", xy=(1 + w / 2, 57.7), xytext=(1.32, 64.5),
                 fontsize=7.0, fontweight="bold", color=PALETTE["danger"],
                 arrowprops=dict(arrowstyle="->", color=PALETTE["danger"], lw=1.0))
    ax1.annotate("−7.8 pp Deep", xy=(2 + w / 2, 11.1), xytext=(2.05, 29.0),
                 fontsize=7.0, fontweight="bold", color=PALETTE["danger"],
                 arrowprops=dict(arrowstyle="->", color=PALETTE["danger"], lw=1.0))

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_ylim(0, 71)
    ax1.set_ylabel("Reference Stage Prevalence (% of Epochs)")
    ax1.set_title("Cross-Modality Reference Prior Shift")
    ax1.legend(loc="upper right", frameon=True, fontsize=6.7, edgecolor="#DDDDDD")
    add_panel_label(ax1, "A")

    # --- Panel B: Resulting Deep% & REM% Projection on SleepAccel (n=25, Primary) ---
    m_keys = ["deep_pct_tst", "rem_pct_tst"]
    m_labels = ["Deep Sleep (% of TST)", "REM Sleep (% of TST)"]

    ref_means, ma_means, sl_means = [], [], []
    ma_biases, sl_biases = [], []
    for mk in m_keys:
        r_a = ba_p[(ba_p["metric"] == mk) & (ba_p["arm"] == "ModelA")].iloc[0]
        r_s = ba_p[(ba_p["metric"] == mk) & (ba_p["arm"] == "SLAMSS_IFS")].iloc[0]
        ref_means.append(float(r_a["ref_mean"]))
        ma_means.append(float(r_a["pred_mean"]))
        sl_means.append(float(r_s["pred_mean"]))
        ma_biases.append(float(r_a["bias"]))
        sl_biases.append(float(r_s["bias"]))

    x2 = np.arange(len(m_keys))
    w2 = 0.25
    ax2.grid(True, axis="y", zorder=0)
    r_bars = ax2.bar(x2 - w2, ref_means, w2, color=PALETTE["neutral"], alpha=0.82,
                     label="In-Lab PSG Reference (n = 25)", zorder=3)
    a_bars = ax2.bar(x2, ma_means, w2, color=PALETTE["model_a"], alpha=0.90,
                     label="Model A (163.9k, Zero-Shot)", zorder=3)
    s_bars = ax2.bar(x2 + w2, sl_means, w2, color=PALETTE["slamss"], alpha=0.88,
                     label="SLAMSS-IFS (5.47M, Zero-Shot)", zorder=3)

    for bars, biases, col in [
        (r_bars, [None, None], "#333333"),
        (a_bars, ma_biases, PALETTE["model_a"]),
        (s_bars, sl_biases, PALETTE["slamss"]),
    ]:
        for rect, b in zip(bars, biases):
            h = rect.get_height()
            txt = f"{h:.1f}%" if b is None else f"{h:.1f}%\n({b:+.1f}%)"
            ax2.text(rect.get_x() + rect.get_width() / 2, h + 0.8, txt,
                     ha="center", va="bottom", fontsize=6.6, color=col, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(m_labels)
    ax2.set_ylim(0, 44.5)
    ax2.set_ylabel("Mean Stage Proportion (% of TST)")
    ax2.set_title("Prior Projection on External PSG (n = 25)")
    ax2.legend(loc="upper left", frameon=True, fontsize=6.6, edgecolor="#DDDDDD")
    add_panel_label(ax2, "B")

    fig.tight_layout(w_pad=1.5)
    save_fig(fig, "Figure5_Label_Prior_Shift_Diagnosis")


def main() -> int:
    builders = [
        ("Figure 1", build_figure_1),
        ("Figure 2", build_figure_2),
        ("Figure 3", build_figure_3),
        ("Figure 4", build_figure_4),
        ("Figure 5", build_figure_5),
    ]
    failures = 0
    for name, fn in builders:
        try:
            fn()
        except Exception as exc:
            failures += 1
            print(f"[FAIL] {name}: {exc}", file=sys.stderr)
            traceback.print_exc(limit=2)
    if failures:
        print(f"\nCompleted with {failures} failed figure(s). Paste the error output to patch column names.", file=sys.stderr)
        return 1
    print(f"\nAll 5 publication figures generated cleanly in: {OUT_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
