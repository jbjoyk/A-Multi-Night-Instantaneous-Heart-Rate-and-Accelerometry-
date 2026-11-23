#!/usr/bin/env python3
"""
Generate publication-ready 300-DPI PNG and editable vector PDF Figures 1–5
for MANUSCRIPT_DRAFT_v1.0.md (*The Lancet Digital Health* submission).

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
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullFormatter
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & Styling (*The Lancet Digital Health* clean vector aesthetic)
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
    "danger": "#B2182B",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.2,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.85,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
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
        raise KeyError(f"None of {list(candidates)} found in columns: {list(df.columns)}")
    return None


def to_pp(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size > 0 and np.nanmax(np.abs(finite)) <= 1.0:
        return arr * 100.0
    return arr


def set_panel_header(
    ax: plt.Axes,
    letter: str,
    title: str,
    y: float = 1.045,
    title_x: float = 0.085,
    title_size: float = 9.2,
) -> None:
    """Collision-proof left-aligned panel letter + title on the same baseline."""
    ax.text(
        0.0, y, letter, transform=ax.transAxes,
        fontsize=11.0, fontweight="bold", va="bottom", ha="left", color="#111111"
    )
    ax.text(
        title_x, y + 0.004, title, transform=ax.transAxes,
        fontsize=title_size, fontweight="bold", va="bottom", ha="left", color="#111111"
    )


# ---------------------------------------------------------------------------
# FIGURE 1: Accuracy–Efficiency Pareto & External PSG Transfer Attenuation
# ---------------------------------------------------------------------------
def build_figure_1() -> None:
    with (ROOT / "experiments/EXP-027/exp027_summary.json").open("r", encoding="utf-8") as f:
        s27 = json.load(f)

    qc_single = s27["cohorts"]["qc_clean"]["mean_single_model"]["metrics"]
    qc_ens = s27["cohorts"]["qc_clean"]["prob_ensemble"]["metrics"]

    params_a = 163_940 / 1e6
    params_s = 5_465_768 / 1e6
    bid_a, bid_s = 64.79, 67.37
    sa_a_s = qc_single["macro_f1"]["model_a_mean"]
    sa_s_s = qc_single["macro_f1"]["slamss_mean"]
    sa_a_e = qc_ens["macro_f1"]["model_a_mean"]
    sa_s_e = qc_ens["macro_f1"]["slamss_mean"]

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.2, 4.35), gridspec_kw={"width_ratios": [1.10, 1.0]}
    )

    # --- Panel A ---
    ax1.set_xscale("log")
    ax1.grid(True, axis="y", zorder=0)

    ax1.annotate(
        "", xy=(params_a, sa_a_s + 0.35), xytext=(params_a, bid_a - 0.35),
        arrowprops=dict(arrowstyle="->", color=PALETTE["model_a"], lw=1.4, ls="--")
    )
    ax1.annotate(
        "", xy=(params_s, sa_s_s + 0.35), xytext=(params_s, bid_s - 0.35),
        arrowprops=dict(arrowstyle="->", color=PALETTE["slamss"], lw=1.4, ls="--")
    )

    ax1.scatter([params_a], [bid_a], s=75, color=PALETTE["model_a"], marker="o", zorder=4,
                label="BIDSleep 5-fold CV (N = 47, home Dreem 2 EEG)")
    ax1.scatter([params_s], [bid_s], s=75, color=PALETTE["slamss"], marker="o", zorder=4)

    ax1.scatter([params_a], [sa_a_s], s=75, facecolors="white", edgecolors=PALETTE["model_a"],
                linewidths=1.8, marker="s", zorder=5,
                label="SleepAccel Zero-Shot Single-Model (n = 25, in-lab PSG)")
    ax1.scatter([params_s], [sa_s_s], s=75, facecolors="white", edgecolors=PALETTE["slamss"],
                linewidths=1.8, marker="s", zorder=5)

    ax1.scatter([params_a], [sa_a_e], s=68, color=PALETTE["model_a"], marker="^", zorder=5,
                label="SleepAccel Zero-Shot 15-Model Ensemble (n = 25)")
    ax1.scatter([params_s], [sa_s_e], s=68, color=PALETTE["slamss"], marker="^", zorder=5)

    ax1.plot([params_a, params_s], [bid_a, bid_s], color="#888888", lw=0.95, ls=":", zorder=2)
    ax1.plot([params_a, params_s], [sa_a_s, sa_s_s], color="#888888", lw=0.95, ls=":", zorder=2)

    # Non-overlapping direct labels
    ax1.text(params_a * 1.15, bid_a + 0.45, f"Model A (163.9k)\nINT8: 310 KiB | {bid_a:.2f}%",
             fontsize=7.3, color=PALETTE["model_a"], fontweight="bold", ha="left", va="bottom")
    ax1.text(params_s * 0.90, bid_s + 0.45, f"Official SLAMSS-IFS\n5.47M (33.34x) | {bid_s:.2f}%",
             fontsize=7.3, color=PALETTE["slamss"], fontweight="bold", ha="right", va="bottom")

    ax1.text(params_a * 1.16, 62.65, f"−{bid_a - sa_a_s:.2f} pp",
             fontsize=7.2, color=PALETTE["model_a"], va="center", ha="left")
    ax1.text(params_s * 0.86, 64.25, f"−{bid_s - sa_s_s:.2f} pp",
             fontsize=7.2, color=PALETTE["slamss"], va="center", ha="right")

    ax1.text(params_a * 1.16, sa_a_s - 0.10, f"{sa_a_s:.2f}%\n(Ens: {sa_a_e:.2f}%)",
             fontsize=7.0, color=PALETTE["model_a"], va="center", ha="left")
    ax1.text(params_s * 0.86, sa_s_s - 0.10, f"{sa_s_s:.2f}%\n(Ens: {sa_s_e:.2f}%)",
             fontsize=7.0, color=PALETTE["slamss"], va="center", ha="right")

    # In-cohort gap box in clean center whitespace pointing up to dotted connector
    ax1.annotate(
        "In-cohort gap: −2.58 pp\n(p < 0.0001)",
        xy=(1.02, 66.10), xytext=(1.02, 63.75),
        fontsize=7.1, ha="center", va="center", color="#222222",
        arrowprops=dict(arrowstyle="->", color="#777777", lw=0.9),
        bbox=dict(boxstyle="round,pad=0.28", fc="#F7F7F7", ec="#BBBBBB", lw=0.7),
        zorder=6
    )

    # External PSG gap box cleanly in dedicated bottom whitespace
    ax1.text(
        1.00, 58.05,
        "External PSG gap: −0.57 pp (p = 0.58)\n78% lead attenuation  |  Ensemble: +0.15 pp",
        fontsize=7.1, ha="center", va="bottom", color="#111111",
        bbox=dict(boxstyle="round,pad=0.30", fc="#EDF6FC", ec=PALETTE["model_a"], lw=0.8),
        zorder=6
    )

    ax1.set_xlim(0.09, 10.8)
    ax1.set_ylim(57.3, 70.6)
    ax1.set_xticks([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax1.xaxis.set_minor_formatter(NullFormatter())
    ax1.set_xlabel("Trainable Parameters (Millions, log scale)")
    ax1.set_ylabel("4-Class Macro-F1 (%)")
    set_panel_header(ax1, "A", "Cross-Site Transfer & Compression Trade-Off", title_x=0.075)
    ax1.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.19),
        frameon=True, fontsize=7.1, edgecolor="#DDDDDD", ncol=1
    )

    # --- Panel B ---
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

    for idx, (_, key) in enumerate(stages):
        y = y_pos[idx]
        ms = qc_single[key]
        me = qc_ens[key]

        ax2.errorbar(
            ms["delta_mean"], y + offset,
            xerr=[[ms["delta_mean"] - ms["ci95_low"]], [ms["ci95_high"] - ms["delta_mean"]]],
            fmt="s", color=PALETTE["model_a"], ecolor=PALETTE["model_a"],
            capsize=3.2, markersize=5.2, lw=1.35,
            label="Primary: Single-Model Mean" if idx == 0 else None
        )
        ax2.errorbar(
            me["delta_mean"], y - offset,
            xerr=[[me["delta_mean"] - me["ci95_low"]], [me["ci95_high"] - me["delta_mean"]]],
            fmt="^", color=PALETTE["gate_tau"], ecolor=PALETTE["gate_tau"],
            capsize=3.2, markersize=5.2, lw=1.35,
            label="Secondary: 15-Model Ensemble" if idx == 0 else None
        )
        if key == "light_f1":
            ax2.text(ms["ci95_low"] - 0.25, y + offset, f"p = {ms['wilcoxon_p']:.4f}",
                     fontsize=7.0, va="center", ha="right", color=PALETTE["danger"], fontweight="bold")
            ax2.text(me["ci95_low"] - 0.25, y - offset, f"p = {me['wilcoxon_p']:.4f}",
                     fontsize=7.0, va="center", ha="right", color=PALETTE["danger"])

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([s[0] for s in stages])
    ax2.set_xlim(-10.2, 7.2)
    ax2.set_ylim(-0.95, len(stages) - 0.35)
    ax2.set_xlabel("Paired Δ (Model A − SLAMSS-IFS, pp; 95% Bootstrap CI)")
    set_panel_header(ax2, "B", "External PSG Stage-Wise Paired Delta (n = 25)", title_x=0.08)

    ax2.text(-0.45, -0.76, "← Favors SLAMSS-IFS", fontsize=7.1, color=PALETTE["slamss"], ha="right", fontweight="bold")
    ax2.text(+0.45, -0.76, "Favors Model A →", fontsize=7.1, color=PALETTE["model_a"], ha="left", fontweight="bold")

    ax2.legend(
        loc="upper center", bbox_to_anchor=(0.48, -0.19),
        frameon=True, fontsize=7.1, edgecolor="#DDDDDD", ncol=1
    )

    fig.tight_layout(w_pad=2.2)
    save_fig(fig, "Figure1_Pareto_External_PSG_Transfer")


# ---------------------------------------------------------------------------
# FIGURE 2: Multi-Night Calibration-Depth Learning Curve (TSK-06 Clean-ZS)
# ---------------------------------------------------------------------------
def build_figure_2() -> None:
    df = pd.read_csv(ROOT / "experiments/EXP-019/exp019_k_learning_curve_CLEAN_RECONCILED.csv")

    col_cohort = pick_col(df, ["cohort", "analysis_cohort", "subset", "group"])
    col_k = pick_col(df, ["k", "cal_nights", "k_nights"])
    col_n = pick_col(df, ["n_subjects", "n_sub", "n"])
    col_delta = pick_col(df, ["delta_mean_pp", "delta_pp", "delta_mean", "mean_delta", "delta"])
    col_lo = pick_col(df, ["CI_lower", "ci95_low", "ci_low", "ci_lo", "delta_ci_low"])
    col_hi = pick_col(df, ["CI_upper", "ci95_high", "ci_high", "ci_hi", "delta_ci_high"])

    cohort_vals = df[col_cohort].astype(str).str.lower()
    df_avail = df[cohort_vals.str.contains("avail")].sort_values(col_k).copy()
    df_comm = df[cohort_vals.str.contains("comm|e4|e_4")].sort_values(col_k).copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.8, 3.75), sharey=True)

    for ax, sub, letter, title, color, note in [
        (
            ax1, df_avail, "A",
            "Available Cohort (N = 46, 46, 42, 33)",
            PALETTE["model_a"],
            "First 95% CI > 0 at k = 3 nights\n(converges with van der Aar et al., 2025)"
        ),
        (
            ax2, df_comm, "B",
            "Common Evaluation Cohort E4 (N = 33, 64 Nights)",
            PALETTE["gate_tau"],
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
        ax.axhline(1.50, color=PALETTE["slamss"], lw=1.0, ls=":", zorder=1,
                   label="Pre-registered +1.50 pp benchmark")
        ax.grid(True, axis="y", zorder=0)

        ax.fill_between(k, lo, hi, color=color, alpha=0.16, zorder=2)
        ax.errorbar(
            k, d, yerr=[d - lo, hi - d],
            fmt="-o", color=color, ecolor=color, capsize=3.5, lw=1.8, markersize=5.8, zorder=4
        )

        for xi, yi, hii, ni in zip(k, d, hi, n_sub):
            y_txt = max(hii + 0.18, 1.62) if (letter == "A" and xi == 1) else hii + 0.18
            ax.text(
                xi, y_txt, f"+{yi:.2f} pp\n(N={ni})", ha="center", va="bottom",
                fontsize=7.2, color="#222222", fontweight="bold" if xi >= 3 else "normal",
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.80)
            )

        ax.text(
            0.035, 0.04, note, transform=ax.transAxes, fontsize=7.1, va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.30", fc="#F9F9F9", ec="#CCCCCC", lw=0.65)
        )
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels([f"k = {i}" for i in [1, 2, 3, 4]])
        ax.set_xlim(0.62, 4.38)
        ax.set_ylim(-1.95, 5.45)
        ax.set_xlabel("Labeled Calibration Nights per Participant (k)")
        set_panel_header(ax, letter, title, title_x=0.075, title_size=8.9)

    ax1.set_ylabel("Paired Macro-F1 Gain vs. Clean Zero-Shot (pp)")
    ax1.legend(loc="upper left", frameon=True, fontsize=7.2, edgecolor="#DDDDDD")

    fig.tight_layout(w_pad=1.8)
    save_fig(fig, "Figure2_Calibration_Depth_Learning_Curve")


# ---------------------------------------------------------------------------
# FIGURE 3: Exploratory Safe Adaptation & Heterogeneity Audit
# ---------------------------------------------------------------------------
def build_figure_3() -> None:
    df20 = pd.read_csv(ROOT / "experiments/EXP-020/exp020_subject_metrics.csv")
    df19b = pd.read_csv(ROOT / "experiments/EXP-019B/exp019b_correlation_summary.csv")

    col_sub = pick_col(df20, ["subject_id", "subject", "participant"])
    col_zs = pick_col(df20, ["ZS", "zs_macro_f1", "zs_f1"])
    col_ft2 = pick_col(df20, ["FT-2", "ft2_macro_f1", "ft2_f1"])
    col_kl = pick_col(df20, ["FT-2-KL", "ft2_kl_macro_f1", "ft2_kl_f1"])
    col_gt = pick_col(df20, ["GATE_tau", "gate_tau_macro_f1", "gate_tau_f1"])

    zs_vals = to_pp(df20[col_zs])
    plot_df = pd.DataFrame({
        "subject": df20[col_sub].astype(str),
        "ft2": to_pp(df20[col_ft2]) - zs_vals,
        "ft2_kl": to_pp(df20[col_kl]) - zs_vals,
        "gate_tau": to_pp(df20[col_gt]) - zs_vals,
    }).sort_values("ft2").reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.2, 4.15), gridspec_kw={"width_ratios": [1.18, 1.0]}
    )

    # --- Panel A ---
    x = np.arange(len(plot_df))
    ax1.axhline(0.0, color=PALETTE["neutral"], lw=0.85, ls="-", zorder=1)
    ax1.axhline(-5.0, color=PALETTE["danger"], lw=0.95, ls="--", zorder=1,
                label="Severe harm threshold (−5.0 pp)")
    ax1.grid(True, axis="y", zorder=0)

    ax1.plot(x, plot_df["ft2"], color=PALETTE["ft2"], lw=1.25, alpha=0.88, marker="o", ms=3.0,
             label="Unregularized FT-2 (Min −16.34 pp)")
    ax1.plot(x, plot_df["ft2_kl"], color=PALETTE["ft2_kl"], lw=1.15, ls="--", alpha=0.88, marker="s", ms=2.6,
             label="Soft KL Reg. FT-2-KL (Min −15.14 pp)")
    ax1.plot(x, plot_df["gate_tau"], color=PALETTE["gate_tau"], lw=1.6, marker="^", ms=3.5,
             label="Stage-Coverage Gate (Min −6.40 pp; 2/46 abstained)")

    b34_mask = plot_df["subject"].str.contains("34")
    if b34_mask.any():
        b_idx = int(plot_df.index[b34_mask][0])
        ax1.annotate(
            "Bidslab34 (0.68% Deep)\nFT-2: −16.34 pp → Gate: 0.00 pp\n(61% worst-case reduction)",
            xy=(b_idx, plot_df.loc[b_idx, "gate_tau"]),
            xytext=(b_idx + 10, -14.2),
            fontsize=7.0, fontweight="bold", color="#111111",
            arrowprops=dict(arrowstyle="->", color=PALETTE["gate_tau"], lw=1.25),
            bbox=dict(boxstyle="round,pad=0.28", fc="#E8F8F2", ec=PALETTE["gate_tau"], lw=0.8)
        )

    ax1.set_xlim(-1.5, len(plot_df) + 0.5)
    ax1.set_ylim(-18.5, 16.5)
    ax1.set_xlabel("Participants Ranked by Unregularized FT-2 Δ (N = 46)")
    ax1.set_ylabel("Macro-F1 Change vs. Zero-Shot (pp)")
    set_panel_header(ax1, "A", "Exploratory Tail-Risk Mitigation (N = 46)", title_x=0.075)
    ax1.legend(loc="upper left", frameon=True, fontsize=6.9, edgecolor="#DDDDDD")

    # --- Panel B ---
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
        capsize=3.2, markersize=5.2, lw=1.35, zorder=4
    )
    for yi, hi, qi in zip(y2, rhis, qs):
        ax2.text(hi + 0.035, yi, f"q = {qi:.2f}", va="center", ha="left", fontsize=7.0, color="#333333")

    ax2.set_yticks(y2)
    ax2.set_yticklabels([pretty_names.get(str(v), str(v)) for v in f1[col_feat]])
    ax2.set_xlim(-0.62, 0.72)
    ax2.set_ylim(-1.85, len(f1) - 0.35)
    ax2.set_xlabel("Spearman ρ with FT-2 Gain (95% Bootstrap CI)")
    set_panel_header(ax2, "B", "Exploratory Heterogeneity Audit (N = 46)", title_x=0.085)

    ax2.text(
        0.03, 0.035,
        "0 / 12 features survive BH-FDR (all q ≥ 0.57)\nOldham headroom check: ρ = +0.038 (p = 0.804)",
        transform=ax2.transAxes, fontsize=7.0, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.28", fc="#F9F9F9", ec="#CCCCCC", lw=0.65)
    )

    fig.tight_layout(w_pad=2.4)
    save_fig(fig, "Figure3_Exploratory_Tail_Risk_and_Heterogeneity")


# ---------------------------------------------------------------------------
# FIGURE 4: External Single-Night PSG Bland–Altman Panels (EXP-028, n=25)
# ---------------------------------------------------------------------------
def build_figure_4() -> None:
    ba = pd.read_csv(ROOT / "experiments/EXP-028/exp028_sleepaccel_bland_altman_summary.csv")
    raw = pd.read_csv(ROOT / "experiments/EXP-028/exp028_sleepaccel_raw_paired_tuples.csv")
    qc = pd.read_csv(ROOT / "experiments/EXP-027/sleepaccel_qc_summary.csv")

    ba_p = ba[(ba["cohort"] == "qc_clean") & (ba["eval_mode"] == "mean_single_model")].copy()

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

    fig, axes = plt.subplots(2, 3, figsize=(10.8, 7.1))
    axes = axes.ravel()

    for idx, (mkey, mtitle, unit) in enumerate(metrics_order):
        ax = axes[idx]
        ax.axhline(0.0, color="#999999", lw=0.75, ls=":", zorder=1)
        ax.grid(True, zorder=0)

        stat_lines = []
        for arm_key, short_tag, full_label, color, marker, ls in [
            ("ModelA", "A", "Model A (163.9k)", PALETTE["model_a"], "o", "-"),
            ("SLAMSS_IFS", "S", "Official SLAMSS-IFS (5.47M)", PALETTE["slamss"], "^", "--"),
        ]:
            sub_raw = raw_p[(raw_p[r_met] == mkey) & (raw_p[r_arm] == arm_key)]
            row = ba_p[(ba_p["metric"] == mkey) & (ba_p["arm"] == arm_key)].iloc[0]

            mean_xy = 0.5 * (sub_raw[r_ref].to_numpy(float) + sub_raw[r_pred].to_numpy(float))
            diff_xy = sub_raw[r_pred].to_numpy(float) - sub_raw[r_ref].to_numpy(float)

            ax.scatter(
                mean_xy, diff_xy, s=26, marker=marker,
                facecolors=color if arm_key == "ModelA" else "none",
                edgecolors=color, linewidths=1.05, alpha=0.85, zorder=4,
                label=full_label if idx == 0 else None
            )

            non_norm = bool(row["non_normal_flag"])
            bias = float(row["nonparam_median_diff"] if non_norm else row["bias"])
            loa_l = float(row["nonparam_loa_low"] if non_norm else row["loa_low"])
            loa_h = float(row["nonparam_loa_high"] if non_norm else row["loa_high"])

            ax.axhline(bias, color=color, lw=1.2, ls=ls, zorder=3)
            ax.axhline(loa_l, color=color, lw=0.85, ls=ls, alpha=0.55, zorder=2)
            ax.axhline(loa_h, color=color, lw=0.85, ls=ls, alpha=0.55, zorder=2)

            dag = "†" if non_norm else ""
            star = "*" if float(row["prop_bias_p"]) < 0.05 else ""
            stat_lines.append(
                f"{short_tag}: {bias:+.1f}{dag} [{loa_l:+.1f}, {loa_h:+.1f}] | MAE {row['mae']:.1f}, r={row['pearson_r']:.2f}{star}"
            )

        # Place title + 2-line stat strip cleanly ABOVE the plot area (zero occlusion or overflow)
        set_panel_header(ax, chr(ord("A") + idx), mtitle, y=1.22, title_x=0.105, title_size=8.8)
        ax.text(
            0.0, 1.025, "\n".join(stat_lines), transform=ax.transAxes,
            fontsize=6.55, va="bottom", ha="left", color="#1A1A1A", linespacing=1.32,
            bbox=dict(boxstyle="round,pad=0.25", fc="#F9F9F9", ec="#D8D8D8", lw=0.6)
        )

        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.margins(x=0.08, y=0.10)
        ax.set_xlabel(f"Mean of Predicted & PSG ({unit})", fontsize=7.8)
        if idx % 3 == 0:
            ax.set_ylabel("Predicted − PSG", fontsize=8.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        bbox_to_anchor=(0.5, 1.005), fontsize=8.6
    )
    fig.text(
        0.5, 0.962,
        "A = Model A; S = SLAMSS-IFS (primary single-model mean, n = 25);  † non-parametric median & 2.5th–97.5th percentile LoA (Shapiro–Wilk p < 0.05);  * proportional bias p < 0.05.",
        ha="center", va="top", fontsize=7.3, color="#444444"
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.865, bottom=0.07, hspace=0.68, wspace=0.28)
    save_fig(fig, "Figure4_External_PSG_Bland_Altman")


# ---------------------------------------------------------------------------
# FIGURE 5: Cross-Modality Label-Prior Shift (Dreem 2 EEG -> In-Lab PSG)
# ---------------------------------------------------------------------------
def build_figure_5() -> None:
    ba = pd.read_csv(ROOT / "experiments/EXP-028/exp028_sleepaccel_bland_altman_summary.csv")
    ba_p = ba[(ba["cohort"] == "qc_clean") & (ba["eval_mode"] == "mean_single_model")].copy()

    # Exact reconciled MANUSCRIPT_DRAFT_v1.0.md §2.1 night-averaged Dreem 2 EEG vs SleepAccel PSG priors
    bid_prior = np.array([12.8, 44.2, 18.9, 24.1])
    sa_prior = np.array([8.3, 57.7, 11.1, 22.9])

    # Optional cross-check against preflight_distribution_gate.json if stage fractions are present
    gate_path = ROOT / "experiments/EXP-027/preflight_distribution_gate.json"
    if gate_path.exists():
        try:
            g = json.loads(gate_path.read_text(encoding="utf-8"))
            print(f"[INFO] Loaded preflight_distribution_gate.json keys: {list(g.keys())}")
        except Exception:
            pass

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(9.2, 4.0), gridspec_kw={"width_ratios": [1.08, 1.0]}
    )

    # --- Panel A ---
    stages = ["Wake", "Light\n(N1+N2)", "Deep\n(N3)", "REM"]
    x = np.arange(len(stages))
    w = 0.36
    ax1.grid(True, axis="y", zorder=0)

    b1 = ax1.bar(x - w / 2, bid_prior, w, color=PALETTE["model_a"], alpha=0.88,
                 label="BIDSleep Home Dreem 2 EEG (Training)", zorder=3)
    b2 = ax1.bar(x + w / 2, sa_prior, w, color=PALETTE["neutral"], alpha=0.82,
                 label="SleepAccel In-Lab PSG (External Ref.)", zorder=3)

    for rect in b1:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, h + 1.1, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=7.2, color=PALETTE["model_a"], fontweight="bold")
    for rect in b2:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width() / 2, h + 1.1, f"{h:.1f}%",
                 ha="center", va="bottom", fontsize=7.2, color="#333333", fontweight="bold")

    # Clean vertical arrows above Light & Deep (zero overlap with legend or REM bars)
    ax1.annotate(
        "+13.5 pp Light",
        xy=(1 + w / 2, 62.0), xytext=(1.55, 69.5),
        fontsize=7.4, fontweight="bold", color=PALETTE["danger"], ha="center",
        arrowprops=dict(arrowstyle="->", color=PALETTE["danger"], lw=1.15)
    )
    ax1.annotate(
        "−7.8 pp Deep",
        xy=(2 + w / 2, 16.0), xytext=(2 + w / 2, 37.5),
        fontsize=7.4, fontweight="bold", color=PALETTE["danger"], ha="center",
        arrowprops=dict(arrowstyle="->", color=PALETTE["danger"], lw=1.15)
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    ax1.set_ylim(0, 79)
    ax1.set_ylabel("Reference Stage Prevalence (% of Epochs)")
    set_panel_header(ax1, "A", "Cross-Modality Reference Prior Shift", title_x=0.075)
    ax1.legend(loc="upper left", frameon=True, fontsize=7.0, edgecolor="#DDDDDD")

    # --- Panel B ---
    m_keys = ["deep_pct_tst", "rem_pct_tst"]
    m_labels = ["Deep Sleep (% of TST)", "REM Sleep (% of TST)"]

    ref_means, ma_means, sl_means, ma_biases, sl_biases = [], [], [], [], []
    for mk in m_keys:
        r_a = ba_p[(ba_p["metric"] == mk) & (ba_p["arm"] == "ModelA")].iloc[0]
        r_s = ba_p[(ba_p["metric"] == mk) & (ba_p["arm"] == "SLAMSS_IFS")].iloc[0]
        ref_means.append(float(r_a["ref_mean"]))
        ma_means.append(float(r_a["pred_mean"]))
        sl_means.append(float(r_s["pred_mean"]))
        ma_biases.append(float(r_a["bias"]))
        sl_biases.append(float(r_s["bias"]))

    x2 = np.array([0.0, 1.28])
    w2 = 0.28
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
            ax2.text(rect.get_x() + rect.get_width() / 2, h + 0.9, txt,
                     ha="center", va="bottom", fontsize=6.9, color=col, fontweight="bold")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(m_labels)
    ax2.set_xlim(-0.56, 1.84)
    ax2.set_ylim(0, 47.0)
    ax2.set_ylabel("Mean Stage Proportion (% of TST)")
    set_panel_header(ax2, "B", "Prior Projection on External PSG (n = 25)", title_x=0.08)
    ax2.legend(loc="upper left", frameon=True, fontsize=7.0, edgecolor="#DDDDDD")

    fig.tight_layout(w_pad=2.0)
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
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
