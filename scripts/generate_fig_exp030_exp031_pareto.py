#!/usr/bin/env python3
"""
Generate publication-grade 2-panel vector PDF + 300-DPI PNG figure:
  figures/manuscript_v2/fig_exp030_exp031_pareto.{pdf,png}

Reads directly from locked EXP-030 and EXP-031 repository artifacts
(zero hardcoded numerical values).

Usage:
    .venv/bin/python scripts/generate_fig_exp030_exp031_pareto.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "manuscript_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "plain": "#0072B2",        # Deep blue (EXP-031 plain width sweep)
    "pareto_star": "#009E73",  # Teal green (p1_100k / P190 < 400 KiB selections)
    "anchor": "#4D4D4D",       # Charcoal (Model A-164k)
    "slamss": "#D55E00",       # Vermilion (Official SLAMSS-IFS 5.47M)
    "full_v5": "#CC79A7",      # Muted magenta (full_v5 negative ablation)
    "zone": "#E6F4EA",         # Soft mint shading for < 400 KiB INT8 zone
    "zone_edge": "#009E73",
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
    "legend.fontsize": 7.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.85,
    "grid.color": "#E6E6E6",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.10,
})


def load_data():
    # EXP-031 width sweep table (9 ordered plain widths: W050..W250)
    df_31 = pd.read_csv(ROOT / "experiments" / "EXP-031" / "exp031_width_table.csv")
    df_31 = df_31.sort_values("n_params").reset_index(drop=True)

    # Derive SLAMSS-IFS reference Macro-F1 directly from table
    slamss_f1 = float((df_31["macro_f1_mean_pct"] - df_31["delta_vs_SLAMSS_pp"]).iloc[0])

    # EXP-030 Pareto table (if present) or locked JSONs
    pareto_csv = ROOT / "experiments" / "EXP-030" / "exp030_pareto_table.csv"
    if pareto_csv.exists():
        df_30 = pd.read_csv(pareto_csv)
        p1_row = df_30.loc[df_30["grid_tag"] == "p1_100k"].iloc[0].to_dict()
    else:
        with open(ROOT / "experiments" / "EXP-030" / "elbow_selection.json") as f:
            elbow = json.load(f)
        with open(ROOT / "experiments" / "EXP-030" / "exp030_edge_curve.json") as f:
            edge = json.load(f)
        p1_grid = next(x for x in elbow["grid_summary"] if x["tag"] == "p1_100k")
        p1_edge = next(x for x in edge["edge_curve"] if x["grid_tag"] == "p1_100k")
        p1_row = {
            "grid_tag": "p1_100k",
            "n_params": p1_grid["n_params"],
            "macro_f1_mean_pct": p1_grid["macro_f1_pct"],
            "int8_in_memory_kib": p1_edge["int8_in_memory_kib"],
        }

    # EXP-031 confirmatory summary (H-ARCH)
    with open(ROOT / "experiments" / "EXP-031" / "exp031_confirmatory_summary.json") as f:
        conf_31 = json.load(f)

    # SleepAccel external zero-shot comparisons (qc_clean, n=25, prob_ensemble, em_off)
    with open(ROOT / "experiments" / "EXP-031" / "P190_sleepaccel_paired_comparisons.json") as f:
        sa_comp = json.load(f)["f1_comparisons"]

    sa_ens = {
        row["reference"]: row["ref_mean"]
        for row in sa_comp
        if row["cohort"] == "qc_clean" and row["eval_mode"] == "prob_ensemble"
    }
    sa_ens["P190"] = next(
        row["primary_mean"]
        for row in sa_comp
        if row["cohort"] == "qc_clean" and row["eval_mode"] == "prob_ensemble"
    )

    return df_31, p1_row, slamss_f1, conf_31, sa_ens


def make_figure():
    df_31, p1_row, slamss_f1, conf_31, sa_ens = load_data()

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11.2, 4.35), gridspec_kw={"width_ratios": [1.32, 1.0], "wspace": 0.28}
    )

    # ------------------------------------------------------------------
    # Panel A: Capacity-Accuracy-Edge Pareto Curve (BIDSleep N=47)
    # ------------------------------------------------------------------
    eligible = df_31[df_31["eligible_int8_400kib"] == True]
    max_eligible_params = float(eligible["n_params"].max())
    min_x = 32_000
    max_x = 1_350_000

    # Shaded < 400 KiB Dynamic INT8 eligible zone
    ax_a.axvspan(
        min_x, max_eligible_params * 1.07,
        color=PALETTE["zone"], alpha=0.85, zorder=0,
        label="< 400 KiB Dynamic INT8 zone"
    )
    ax_a.axvline(
        max_eligible_params * 1.07,
        color=PALETTE["zone_edge"], linestyle=":", linewidth=1.1, zorder=1
    )

    # SLAMSS-IFS horizontal reference (5.47M params)
    ax_a.axhline(
        slamss_f1, color=PALETTE["slamss"], linestyle="--", linewidth=1.4, zorder=2,
        label=f"Official SLAMSS-IFS (5.47M): {slamss_f1:.2f}%"
    )

    # Plain width sweep curve (EXP-031 W050..W250)
    ax_a.plot(
        df_31["n_params"], df_31["macro_f1_mean_pct"],
        color=PALETTE["plain"], linewidth=1.6, zorder=3, alpha=0.9
    )

    # Plot individual EXP-031 points
    for _, r in df_31.iterrows():
        arm = r["arm"]
        x, y = r["n_params"], r["macro_f1_mean_pct"]
        if arm == "P190":
            ax_a.scatter(
                x, y, s=95, marker="D", color=PALETTE["pareto_star"],
                edgecolor="black", linewidth=0.9, zorder=6,
                label=f"P190 ({x/1e3:.1f}k, {r['int8_in_memory_kib']:.1f} KiB)"
            )
        elif arm == "A":
            ax_a.scatter(
                x, y, s=70, marker="s", color=PALETTE["anchor"],
                edgecolor="black", linewidth=0.8, zorder=5,
                label=f"Model A-164k ({r['int8_in_memory_kib']:.1f} KiB)"
            )
        else:
            ax_a.scatter(
                x, y, s=46, marker="o",
                color=PALETTE["plain"] if r["eligible_int8_400kib"] else "#7A99B8",
                edgecolor="white", linewidth=0.6, zorder=4
            )

    # Highlight EXP-030 p1_100k
    ax_a.scatter(
        p1_row["n_params"], p1_row["macro_f1_mean_pct"],
        s=135, marker="*", color="#E69F00", edgecolor="black", linewidth=0.8, zorder=7,
        label=f"p1_100k ({p1_row['n_params']/1e3:.1f}k, {p1_row['int8_in_memory_kib']:.1f} KiB)"
    )

    # Clean non-overlapping annotations on Panel A
    offsets = {
        "W050": (0, -13),
        "W075": (-20, -12),
        "A": (-50, -3),
        "P190": (-84, -18),
        "B": (6, -12),
        "W125": (6, 7),
        "W150": (6, -10),
        "W200": (0, -13),
        "W250": (-8, 8),
    }
    for _, r in df_31.iterrows():
        arm = r["arm"]
        dx, dy = offsets.get(arm, (4, 6))
        label_txt = arm if arm != "A" else "Model A"
        if arm == "P190":
            label_txt = f"P190 ({r['int8_in_memory_kib']:.1f} KiB)"
        arrow_kw = dict(arrowstyle="-", color="#333333", lw=0.6) if arm == "P190" else None
        ax_a.annotate(
            label_txt,
            (r["n_params"], r["macro_f1_mean_pct"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=7.0,
            fontweight="bold" if arm in {"P190", "A"} else "normal",
            color="#111111" if arm in {"P190", "A"} else "#333333",
            arrowprops=arrow_kw,
        )

    ax_a.annotate(
        f"p1_100k\n({p1_row['macro_f1_mean_pct']:.2f}%, {p1_row['int8_in_memory_kib']:.1f} KiB)",
        (p1_row["n_params"], p1_row["macro_f1_mean_pct"]),
        xytext=(-54, 14), textcoords="offset points",
        fontsize=7.2, fontweight="bold", color="#7A4F00",
        arrowprops=dict(arrowstyle="-", color="#7A4F00", lw=0.7),
    )

    ax_a.set_xscale("log")
    ax_a.set_xlim(min_x, max_x)
    ax_a.set_ylim(63.75, 67.95)
    ax_a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1e3:.0f}k" if v < 1e6 else f"{v/1e6:.1f}M"))
    ax_a.set_xlabel("Trainable Parameters (log scale)")
    ax_a.set_ylabel("BIDSleep 5-Fold OOF Macro-F1 (%, N = 47)")
    ax_a.set_title("A. Capacity–Accuracy–Edge Pareto Curve (EXP-030 & EXP-031)", loc="left", pad=8)
    ax_a.grid(True, which="major", axis="both", linestyle="-", alpha=0.6)
    ax_a.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#CCCCCC")

    # ------------------------------------------------------------------
    # Panel B: Matched-Parameter H-ARCH Ablation & SleepAccel Transfer
    # ------------------------------------------------------------------
    h_arch = conf_31["confirmatory"]["h_arch"]
    v5_f1 = h_arch["full_v5_mean_pct"]
    p190_f1 = h_arch["p190_mean_pct"]
    delta_pp = h_arch["delta_pp"]
    ci_lo = h_arch["ci95_low_pp"]
    ci_hi = h_arch["ci95_high_pp"]
    p_holm = h_arch["holm_p"]

    categories = [
        "BIDSleep\n(N = 47)\nfull_v5\n(190.8k)",
        "BIDSleep\n(N = 47)\nP190\n(190.9k)",
        "SleepAccel\n(n = 25)\nP190\n(190.9k)",
        "SleepAccel\n(n = 25)\nModel A\n(163.9k)",
        "SleepAccel\n(n = 25)\nSLAMSS-IFS\n(5.47M)",
    ]
    values = [
        v5_f1,
        p190_f1,
        sa_ens["P190"],
        sa_ens["ModelA"],
        sa_ens["SLAMSS_IFS"],
    ]
    colors = [
        PALETTE["full_v5"],
        PALETTE["pareto_star"],
        PALETTE["pareto_star"],
        PALETTE["anchor"],
        PALETTE["slamss"],
    ]

    x_pos = np.array([0.0, 0.9, 2.25, 3.15, 4.05])
    bars = ax_b.bar(x_pos, values, width=0.68, color=colors, edgecolor="black", linewidth=0.75, zorder=3)

    for bar, val in zip(bars, values):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.14,
            f"{val:.2f}%",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#111111"
        )

    # Separator between BIDSleep H-ARCH ablation and SleepAccel external transfer
    ax_b.axvline(1.575, color="#888888", linestyle="--", linewidth=0.9, zorder=2)

    # Bracket annotating H-ARCH NOT SUPPORTED
    bracket_y = max(v5_f1, p190_f1) + 0.72
    ax_b.plot([0.0, 0.0, 0.9, 0.9], [bracket_y - 0.12, bracket_y, bracket_y, bracket_y - 0.12],
              color="#222222", lw=0.95)
    ax_b.text(
        0.45, bracket_y + 0.10,
        f"H-ARCH: Δ = +{delta_pp:.2f} pp\n[{ci_lo:+.2f}, {ci_hi:+.2f}], p={p_holm:.2f}\n(NOT SUPPORTED)",
        ha="center", va="bottom", fontsize=6.9, color="#222222",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#F8F9FA", edgecolor="#CCCCCC", lw=0.6)
    )

    ax_b.text(
        3.15, bracket_y + 0.35,
        "External Zero-Shot PSG\n(qc_clean n = 25, em_off,\n15-Model Ensemble)",
        ha="center", va="bottom", fontsize=7.2, fontweight="bold", color="#333333"
    )

    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(categories, fontsize=7.0)
    ax_b.set_ylim(59.5, 69.3)
    ax_b.set_ylabel("4-Class Macro-F1 (%)")
    ax_b.set_title("B. Matched-Param H-ARCH & External PSG Transfer", loc="left", pad=8)
    ax_b.grid(True, which="major", axis="y", linestyle="-", alpha=0.6)

    pdf_path = OUT_DIR / "fig_exp030_exp031_pareto.pdf"
    png_path = OUT_DIR / "fig_exp030_exp031_pareto.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved {pdf_path.relative_to(ROOT)} and {png_path.relative_to(ROOT)}")


if __name__ == "__main__":
    make_figure()
