#!/usr/bin/env python3
"""
Generate publication-grade figures (300 DPI PNG + vector PDF) for author_report_bundle/:
  1. fig_architectures_baseline_vs_proposed.{pdf,png}
  2. fig_dataset_splits.{pdf,png}
  3. fig_confusion_matrices.{pdf,png}  (reads author_report_bundle/data/confusion_matrices.json)
  4. fig_latency_memory_comparison.{pdf,png}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "author_report_bundle"
DATA_DIR = OUT_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "legend.fontsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.85,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.pdf", format="pdf", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.png", format="png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}.pdf and {stem}.png")


def draw_box(
    ax, x: float, y: float, w: float, h: float, title: str, subtitle: str = "",
    fc: str = "#F8FAFC", ec: str = "#334155", lw: float = 1.1,
    tc: str = "#0F172A", sc: str = "#475569", fs: float = 7.8, sub_fs: float = 6.7,
    zorder: int = 3
) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=zorder + 1)
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                fontsize=sub_fs, color=sc, zorder=zorder + 1)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=zorder + 1)


def draw_arrow(ax, x1: float, y1: float, x2: float, y2: float, color: str = "#475569", lw: float = 1.1) -> None:
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.35", color=color, lw=lw, shrinkA=0, shrinkB=0),
        zorder=2,
    )


# ---------------------------------------------------------------------------
# Figure 1: Side-by-Side Architecture Comparison
# ---------------------------------------------------------------------------
def build_fig1_architectures() -> None:
    fig, ax = plt.subplots(figsize=(14.5, 8.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.50, 0.968, "Architectural Comparison: Baselines vs. Proposed Lightweight & Width-Scaled Models",
            ha="center", va="center", fontsize=12.5, fontweight="bold", color="#0F172A")
    ax.text(0.50, 0.938, "Full-night context L = 1,200 epochs (30s/epoch, ~0.2 Hz IHR + 3-axis ACC indexed to 1 Hz [30 samples/epoch] + 5-dim epoch metadata) -> 4-stage probabilities",
            ha="center", va="center", fontsize=8.5, color="#475569")

    cols = [
        {
            "name": "Baseline 1: Model A",
            "sub": "ConventionalCNNBiGRU\n163,940 params | 309.54 KiB INT8",
            "fc_hdr": "#F1F5F9", "ec_hdr": "#475569",
            "input": "IHR (B, L, 30)\nACC (B, L, 30)\nMeta (B, L, 5)",
            "enc": "HREncoder & ACCEncoder\nConv1d(1->32->64->64)\n+ Meta Linear(5->16)",
            "fuse": "Direct Concatenation\n[h_HR ; h_ACC ; h_meta]\n(B, L, 144)",
            "temp": "2-Layer BiGRU\nhidden = 48/dir (d=96)\n+ LayerNorm(96)",
            "head": "MLP Head (96->64->4)\nSoftmax -> 4 Stages\n(Wake, Light, Deep, REM)",
            "perf": "BIDSleep: 64.79% F1\nSleepAccel: 60.51% / 62.94%",
        },
        {
            "name": "Baseline 2: Official SLAMSS-IFS",
            "sub": "CNN + Multi-Scale + Seq2Seq\n5,465,768 params | 19.49 MiB INT8",
            "fc_hdr": "#FFF7ED", "ec_hdr": "#D55E00",
            "input": "Signal (B, L*30, 2)\n[IHR, ACC] @ 1 Hz\nMeta (B, L, 5)",
            "enc": "CNN(128) + 3L BiGRU(64)\n+ CNN_ACT(5->64)\n-> CNN3(k=11,15,21 -> 512)",
            "fuse": "Concat + Meta (517-d)\n+ Encode_micro\n2L BiGRU(32/dir -> 128)",
            "temp": "Seq2Seq Uni-GRU (96)\nEncoder + Bahdanau Attn\n+ Decoder (TF = 0.0)",
            "head": "Linear(841 -> 4)\nSoftmax -> 4 Stages\n(Wake, Light, Deep, REM)",
            "perf": "BIDSleep: 67.37% F1\nSleepAccel: 61.08% / 62.79%",
        },
        {
            "name": "EXP-029: full_v5",
            "sub": "Ultradian-Pyramidal-Coupler\n190,788 params | 360.50 KiB INT8",
            "fc_hdr": "#FDF2F8", "ec_hdr": "#CC79A7",
            "input": "IHR (B, L, 30)\nACC (B, L, 30)\nMeta (B, L, 5)",
            "enc": "PrePoolEncoder (d=64)\nSub-epoch (B*L, 64, 7)\n~0.233 Hz feature maps",
            "fuse": "IntraEpochFiLM (ACC->HR)\ngamma, beta residual mod.\n-> Pool + Concat (B, L, 144)",
            "temp": "2-Scale Pyramidal BiGRU\nFine 1L (48/dir) + 4x Pool\n-> Coarse 1L (48/dir) -> 96",
            "head": "MLP Head (96->64->4)\nSoftmax -> 4 Stages\n(Wake, Light, Deep, REM)",
            "perf": "BIDSleep: 66.74% F1\nSleepAccel: 61.78% / 63.51%",
        },
        {
            "name": "EXP-030: p1_100k (Pareto*)",
            "sub": "ScaledConventionalCNNBiGRU\n98,940 params | 182.00 KiB INT8",
            "fc_hdr": "#ECFDF5", "ec_hdr": "#009E73",
            "input": "IHR (B, L, 30)\nACC (B, L, 30)\nMeta (B, L, 5)",
            "enc": "ScaledEncoders (d_enc=48)\nConv1d(1->24->48->48)\n+ Meta Linear(5->16)",
            "fuse": "Direct Concatenation\n[h_HR ; h_ACC ; h_meta]\n(B, L, 112)",
            "temp": "2-Layer BiGRU\nhidden = 38/dir (d=76)\n+ LayerNorm(76)",
            "head": "MLP Head (76->48->4)\nSoftmax -> 4 Stages\n(Wake, Light, Deep, REM)",
            "perf": "BIDSleep: 66.67% F1\nSleepAccel: 60.88% / 62.51%",
        },
        {
            "name": "EXP-031: P190 (w=1.08)",
            "sub": "ScaledConventionalCNNBiGRU\n190,940 params | 375.16 KiB INT8",
            "fc_hdr": "#EFF6FF", "ec_hdr": "#0072B2",
            "input": "IHR (B, L, 30)\nACC (B, L, 30)\nMeta (B, L, 5)",
            "enc": "ScaledEncoders (d_enc=72)\nConv1d(1->36->72->72)\n+ Meta Linear(5->16)",
            "fuse": "Direct Concatenation\n[h_HR ; h_ACC ; h_meta]\n(B, L, 160)",
            "temp": "2-Layer BiGRU\nhidden = 50/dir (d=100)\n+ LayerNorm(100)",
            "head": "MLP Head (100->64->4)\nSoftmax -> 4 Stages\n(Wake, Light, Deep, REM)",
            "perf": "BIDSleep: 66.60% F1\nSleepAccel: 61.52% / 63.42%",
        },
    ]

    col_w = 0.176
    gap = 0.020
    x0 = 0.020
    y_rows = [0.80, 0.67, 0.52, 0.37, 0.22, 0.085, 0.020]
    heights = [0.095, 0.095, 0.110, 0.110, 0.110, 0.095, 0.048]

    for idx, c in enumerate(cols):
        cx = x0 + idx * (col_w + gap)
        # Background column card
        bg = FancyBboxPatch(
            (cx - 0.005, 0.012), col_w + 0.010, 0.890,
            boxstyle="round,pad=0.003,rounding_size=0.012",
            facecolor="#FAFBFC", edgecolor="#CBD5E1", linewidth=0.9, zorder=1,
        )
        ax.add_patch(bg)

        draw_box(ax, cx, y_rows[0], col_w, heights[0], c["name"], c["sub"],
                 fc=c["fc_hdr"], ec=c["ec_hdr"], lw=1.6, fs=8.3, sub_fs=6.8)
        draw_box(ax, cx, y_rows[1], col_w, heights[1], "Input Tensors", c["input"],
                 fc="#FFFFFF", ec="#94A3B8", fs=7.6, sub_fs=6.5)
        draw_box(ax, cx, y_rows[2], col_w, heights[2], "Modality Encoders", c["enc"],
                 fc="#EFF6FF", ec="#3B82F6", fs=7.6, sub_fs=6.5)
        draw_box(ax, cx, y_rows[3], col_w, heights[3], "Fusion / Coupling", c["fuse"],
                 fc="#F5F3FF", ec="#7C3AED", fs=7.6, sub_fs=6.5)
        draw_box(ax, cx, y_rows[4], col_w, heights[4], "Temporal Sequence", c["temp"],
                 fc="#ECFDF5", ec="#10B981", fs=7.6, sub_fs=6.5)
        draw_box(ax, cx, y_rows[5], col_w, heights[5], "Output Classifier", c["head"],
                 fc="#FEF3C7", ec="#D97706", fs=7.6, sub_fs=6.5)
        draw_box(ax, cx, y_rows[6], col_w, heights[6], c["perf"], "",
                 fc="#F8FAFC", ec=c["ec_hdr"], lw=1.1, fs=6.7)

        mx = cx + col_w / 2
        for r in range(1, 5):
            draw_arrow(ax, mx, y_rows[r], mx, y_rows[r + 1] + heights[r + 1], color="#64748B", lw=1.1)

    save_fig(fig, "fig_architectures_baseline_vs_proposed")


# ---------------------------------------------------------------------------
# Figure 2: Dataset Splits & Evaluation Pipeline
# ---------------------------------------------------------------------------
def build_fig2_dataset_splits() -> None:
    fig, ax = plt.subplots(figsize=(13.8, 7.2), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.50, 0.962, "Cohort Partitioning, Zero-Shot External Validation, and Causal Personalization Pipeline",
            ha="center", va="center", fontsize=12.2, fontweight="bold", color="#0F172A")
    ax.text(0.50, 0.928, "Strict subject-disjoint cross-validation, zero-retraining external PSG transfer, and chronological intra-subject adaptation",
            ha="center", va="center", fontsize=8.4, color="#475569")

    # Track 1: BIDSleep 5-fold CV
    t1 = FancyBboxPatch((0.02, 0.56), 0.96, 0.33, boxstyle="round,pad=0.004,rounding_size=0.012",
                        facecolor="#F8FAFC", edgecolor="#0072B2", linewidth=1.2, zorder=1)
    ax.add_patch(t1)
    ax.text(0.035, 0.865, "Track 1: In-Cohort BIDSleep 5-Fold Subject-Disjoint Cross-Validation (5 Folds x 3 Seeds: 42, 123, 2024)",
            ha="left", va="center", fontsize=9.2, fontweight="bold", color="#0072B2")

    draw_box(ax, 0.035, 0.61, 0.19, 0.21,
             "BIDSleep Source Cohort",
             "47 healthy adults | 253 nights\nQC (>=60% valid): 246 nights\n211,016 total (203,338 valid) epochs\nDreem 2 EEG + Apple Watch",
             fc="#EFF6FF", ec="#2563EB", fs=8.2, sub_fs=6.9)
    draw_box(ax, 0.265, 0.61, 0.23, 0.21,
             "Outer 5-Fold Subject Split",
             "Strictly subject-disjoint (N=47)\nFolds 1-2: 10 test | 37 train+val\nFolds 3-5: 9 test | 38 train+val\nZero subject or night overlap",
             fc="#FFFFFF", ec="#0072B2", fs=8.2, sub_fs=6.9)
    draw_box(ax, 0.535, 0.61, 0.20, 0.21,
             "Inner Train / Val Split",
             "Train: 32-33 subjects (fixed)\nInner Val: 5 subjects per fold\nCheckpoint & tau* selection\nstrictly inside inner val",
             fc="#FFFFFF", ec="#0072B2", fs=8.2, sub_fs=6.9)
    draw_box(ax, 0.775, 0.61, 0.19, 0.21,
             "Out-of-Fold Evaluation",
             "Full-night L=1200 stitching\nCircadian-neutralized x_meta\nN=47 participant means\n(10,000-resample bootstrap)",
             fc="#ECFDF5", ec="#009E73", fs=8.2, sub_fs=6.9)

    draw_arrow(ax, 0.225, 0.715, 0.265, 0.715)
    draw_arrow(ax, 0.495, 0.715, 0.535, 0.715)
    draw_arrow(ax, 0.735, 0.715, 0.775, 0.715)

    # Track 2: External Zero-Shot SleepAccel
    t2 = FancyBboxPatch((0.02, 0.295), 0.96, 0.235, boxstyle="round,pad=0.004,rounding_size=0.012",
                        facecolor="#FFFBF5", edgecolor="#D55E00", linewidth=1.2, zorder=1)
    ax.add_patch(t2)
    ax.text(0.035, 0.505, "Track 2: External Zero-Shot Validation on PhysioNet SleepAccel v1.0.0 (Zero Retraining / Zero Fine-Tuning)",
            ha="left", va="center", fontsize=9.2, fontweight="bold", color="#D55E00")

    draw_box(ax, 0.035, 0.325, 0.21, 0.15,
             "SleepAccel Cohort (N=31)",
             "31 single-night recordings\nApple Watch + In-Lab PSG\n(Sensitivity cohort: all N=31)",
             fc="#FFF7ED", ec="#D55E00", fs=8.2, sub_fs=6.9)
    draw_box(ax, 0.285, 0.325, 0.23, 0.15,
             "Pre-Execution QC Gate",
             "6 excluded: 5 with <600 epochs\n+ 1 ACC dropout (cov_mo=0.1365)\nPrimary qc_clean: n = 25 nights",
             fc="#FFFFFF", ec="#D55E00", fs=8.2, sub_fs=6.9)
    draw_box(ax, 0.555, 0.325, 0.19, 0.15,
             "15-Checkpoint Inference",
             "All 15 frozen BIDSleep models\nPrimary: mean_single_model\nSecondary: 15-model ensemble",
             fc="#FFFFFF", ec="#D55E00", fs=8.2, sub_fs=6.9)
    draw_box(ax, 0.785, 0.325, 0.18, 0.15,
             "EXP-027 & EXP-028",
             "4-class Macro-F1 & Kappa\nBland-Altman (1999) PSG\nTST, SOL, WASO, SE, Deep%, REM%",
             fc="#FEF3C7", ec="#D97706", fs=8.2, sub_fs=6.9)

    draw_arrow(ax, 0.245, 0.400, 0.285, 0.400)
    draw_arrow(ax, 0.515, 0.400, 0.555, 0.400)
    draw_arrow(ax, 0.745, 0.400, 0.785, 0.400)

    # Track 3: Causal Multi-Night Personalization
    t3 = FancyBboxPatch((0.02, 0.03), 0.96, 0.235, boxstyle="round,pad=0.004,rounding_size=0.012",
                        facecolor="#F5F3FF", edgecolor="#7C3AED", linewidth=1.2, zorder=1)
    ax.add_patch(t3)
    ax.text(0.035, 0.240, "Track 3: Causal Multi-Night Personalization & Safety Gating (BIDSleep N=46, Bidslab49 Excluded [2 Nights Only])",
            ha="left", va="center", fontsize=9.2, fontweight="bold", color="#7C3AED")

    draw_box(ax, 0.035, 0.055, 0.22, 0.155,
             "Chronological Split (C_k vs E_k)",
             "Earlier nights 1..k -> Adaptation\nFuture nights k+1..M -> Test\nStrict causal order (C_k ^ E_k = empty)",
             fc="#FFFFFF", ec="#7C3AED", fs=8.1, sub_fs=6.8)
    draw_box(ax, 0.295, 0.055, 0.23, 0.155,
             "k=1..4 Learning Curve (EXP-019-K)",
             "Available cohort E_k: N=46, 46, 42, 33\nCommon cohort E_4: N=33 (64 nights)\nPage's L-trend + Holm-Bonferroni",
             fc="#FFFFFF", ec="#7C3AED", fs=8.1, sub_fs=6.8)
    draw_box(ax, 0.565, 0.055, 0.20, 0.155,
             "Controls & Unsupervised (k=2)",
             "FT-other-2: donor harm control\nPS-2: scratch baseline\nUA-2: label-free InfoMax",
             fc="#FFFFFF", ec="#7C3AED", fs=8.1, sub_fs=6.8)
    draw_box(ax, 0.805, 0.055, 0.16, 0.155,
             "EXP-020 GATE_tau",
             "Nested inner-val tau*\nAbstains if min_stage < tau*\n(guards Bidslab34)",
             fc="#ECFDF5", ec="#009E73", fs=8.1, sub_fs=6.8)

    draw_arrow(ax, 0.255, 0.132, 0.295, 0.132)
    draw_arrow(ax, 0.525, 0.132, 0.565, 0.132)
    draw_arrow(ax, 0.765, 0.132, 0.805, 0.132)

    save_fig(fig, "fig_dataset_splits")


# ---------------------------------------------------------------------------
# Figure 4: Latency, Memory, Parameter & FP32 Size Comparison
# ---------------------------------------------------------------------------
def build_fig4_latency_memory() -> None:
    models = ["p1_100k\n(EXP-030)", "Model A\n(Baseline 1)", "full_v5\n(EXP-029)", "P190\n(EXP-031)", "SLAMSS-IFS\n(Baseline 2)"]
    params_k = np.array([98.940, 163.940, 190.788, 190.940, 5465.768])
    fp32_kib = np.array([386.48, 640.39, 745.27, 745.86, 21350.66])
    int8_kib = np.array([182.00, 309.54, 360.50, 375.16, 19960.51])
    lat_4t_ms = np.array([46.35, 47.73, 44.02, 53.07, 3373.83])
    colors = ["#009E73", "#4D4D4D", "#CC79A7", "#0072B2", "#D55E00"]

    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), dpi=300)
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0, 0], params_k, "A. Parameter Count (Thousands, log scale)", "Parameters (k)", True, [f"{v:,.1f}k" if v < 1000 else f"{v/1000:.2f}M" for v in params_k]),
        (axes[0, 1], fp32_kib, "B. FP32 Weight Payload (KiB, log scale)", "FP32 Weights (KiB)", True, [f"{v:,.0f} KiB" if v < 2000 else f"{v/1024:.1f} MiB" for v in fp32_kib]),
        (axes[1, 0], int8_kib, "C. Dynamic INT8 In-Memory Payload (KiB, log scale)", "Dynamic INT8 Payload (KiB)", True, [f"{v:,.1f} KiB" if v < 2000 else f"{v/1024:.1f} MiB" for v in int8_kib]),
        (axes[1, 1], lat_4t_ms, "D. 4-Thread CPU Inference Time (ms / night, log scale)", "CPU Latency (ms / night, L=1200)", True, [f"{v:.1f} ms" if v < 1000 else f"{v/1000:.2f} s" for v in lat_4t_ms]),
    ]

    x = np.arange(len(models))
    for ax, vals, title, ylabel, use_log, labels in panels:
        bars = ax.bar(x, vals, width=0.58, color=colors, edgecolor="#1E293B", linewidth=0.7, zorder=3)
        if use_log:
            ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=7.6)
        ax.set_ylabel(ylabel, fontsize=8.3)
        ax.set_title(title, loc="left", fontsize=9.4, pad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.55, zorder=0)
        ymax = max(vals) * (2.6 if use_log else 1.2)
        ymin = min(vals) * 0.45 if use_log else 0
        ax.set_ylim(ymin, ymax)
        for bar, lbl in zip(bars, labels):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h * 1.16, lbl,
                    ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#0F172A")

    # Add 400 KiB budget line on Panel C
    axes[1, 0].axhline(400.0, color="#009E73", linestyle="--", linewidth=1.2, zorder=4, label="400 KiB Edge Budget")
    axes[1, 0].legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=7.5)

    fig.suptitle("Computational & Memory Footprint: 5-Model Comparison (Full-Night L = 1,200)",
                 fontsize=11.5, fontweight="bold", y=0.99, color="#0F172A")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "fig_latency_memory_comparison")


# ---------------------------------------------------------------------------
# Figure 3: Row-Normalized 4x4 Confusion Matrices (BIDSleep & SleepAccel)
# ---------------------------------------------------------------------------
def build_fig3_confusion_matrices() -> None:
    cm_path = DATA_DIR / "confusion_matrices.json"
    if not cm_path.exists():
        print("Skipping fig_confusion_matrices: confusion_matrices.json not found yet.")
        return

    with open(cm_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models = ["Model A", "Official SLAMSS-IFS", "p1_100k", "P190"]
    stages = ["Wake", "Light", "Deep", "REM"]

    fig, axes = plt.subplots(2, 4, figsize=(14.4, 7.2), dpi=300, sharex=True, sharey=True)
    fig.patch.set_facecolor("white")

    cmap = plt.cm.Blues

    for row_idx, ds_key in enumerate(["BIDSleep", "SleepAccel"]):
        for col_idx, m_name in enumerate(models):
            ax = axes[row_idx, col_idx]
            d = data[ds_key][m_name]
            mat = np.array(
                d["row_normalized"] if ds_key == "BIDSleep" else d["row_normalized_pooled_15ckpt"],
                dtype=float,
            ) * 100.0

            im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=85, aspect="equal")
            for i in range(4):
                for j in range(4):
                    val = mat[i, j]
                    color = "white" if val > 48.0 else "#0F172A"
                    weight = "bold" if i == j else "normal"
                    ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                            color=color, fontsize=7.4, fontweight=weight)

            f1_str = (
                f"{d['participant_mean_macro_f1_pct']:.2f}%"
                if ds_key == "BIDSleep"
                else f"{d['mean_single_model_macro_f1_pct']:.2f}%"
            )
            ds_label = "BIDSleep (N=47)" if ds_key == "BIDSleep" else "SleepAccel (n=25)"
            ax.set_title(f"{m_name}\n{ds_label} | F1={f1_str}", fontsize=8.4, fontweight="bold", pad=6)
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(stages, fontsize=7.6)
            ax.set_yticklabels(stages, fontsize=7.6)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            if row_idx == 1:
                ax.set_xlabel("Predicted Stage", fontsize=8.0)
            if col_idx == 0:
                ax.set_ylabel("True Stage", fontsize=8.0)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.018, pad=0.02)
    cbar.set_label("Row-Normalized Recall (%)", fontsize=8.2)
    fig.suptitle("Row-Normalized 4x4 Confusion Matrices: BIDSleep 5-Fold OOF (Top) & Zero-Shot SleepAccel (Bottom)",
                 fontsize=11.5, fontweight="bold", y=0.985, color="#0F172A")
    save_fig(fig, "fig_confusion_matrices")


def main() -> None:
    build_fig1_architectures()
    build_fig2_dataset_splits()
    build_fig4_latency_memory()
    build_fig3_confusion_matrices()


if __name__ == "__main__":
    main()
