"""
generate_report_charts.py
Generates publication-quality charts for the EXP-012 Benchmark Report.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set overall style
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8

os.makedirs("report_assets", exist_ok=True)

# -----------------------------------------------------------------------------
# 1. ARCHITECTURE BLOCK DIAGRAM: SleepAdaptiveNet (B0_GRL) vs SLAMSS-IFS
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8.5)
ax.axis("off")

# Colors
c_input = "#E3F2FD"      # Light Blue
c_encoder = "#E8F5E9"    # Light Green
c_fusion = "#FFF3E0"     # Light Orange
c_temporal = "#F3E5F5"   # Light Purple
c_head = "#FFEBEE"       # Light Red
c_grl = "#EDE7F6"        # Deep Purple tint
c_trans = "#FFF8E1"      # Amber
c_border = "#37474F"     # Dark slate

# Title
ax.text(7, 8.1, "Architectural Comparison: Proposed SleepAdaptiveNet (B0_GRL) vs. Baseline SLAMSS-IFS",
        ha="center", va="center", fontsize=15, fontweight="bold", color="#1A237E")

# ----------------- PANEL A: SleepAdaptiveNet (Left/Top) -----------------
# Container Box
rect_a = patches.FancyBboxPatch((0.4, 0.4), 8.8, 7.3, boxstyle="round,pad=0.2",
                                facecolor="#FAFAFA", edgecolor="#283593", linewidth=1.8)
ax.add_patch(rect_a)
ax.text(0.7, 7.4, "A. Proposed SleepAdaptiveNet (B0_GRL)  [162,325 Parameters — 33.7x Compression]",
        fontsize=12, fontweight="bold", color="#283593")

# Inputs
rect_in1 = patches.FancyBboxPatch((0.8, 6.1), 2.2, 0.9, boxstyle="round,pad=0.1",
                                  facecolor=c_input, edgecolor=c_border, linewidth=1)
ax.add_patch(rect_in1)
ax.text(1.9, 6.7, "Instantaneous HR", ha="center", fontsize=9, fontweight="bold")
ax.text(1.9, 6.35, "x_ihr (1 Hz, 30s)", ha="center", fontsize=8, color="#555")

rect_in2 = patches.FancyBboxPatch((0.8, 4.8), 2.2, 0.9, boxstyle="round,pad=0.1",
                                  facecolor=c_input, edgecolor=c_border, linewidth=1)
ax.add_patch(rect_in2)
ax.text(1.9, 5.4, "Accelerometer Mag", ha="center", fontsize=9, fontweight="bold")
ax.text(1.9, 5.05, "x_accel (1 Hz, 30s)", ha="center", fontsize=8, color="#555")

rect_in3 = patches.FancyBboxPatch((0.8, 3.5), 2.2, 0.9, boxstyle="round,pad=0.1",
                                  facecolor=c_input, edgecolor=c_border, linewidth=1)
ax.add_patch(rect_in3)
ax.text(1.9, 4.1, "Metadata Features", ha="center", fontsize=9, fontweight="bold")
ax.text(1.9, 3.75, "x_meta (5 dims)", ha="center", fontsize=8, color="#555")

# Encoders
rect_enc1 = patches.FancyBboxPatch((3.5, 6.1), 2.5, 0.9, boxstyle="round,pad=0.1",
                                   facecolor=c_encoder, edgecolor="#2E7D32", linewidth=1.2)
ax.add_patch(rect_enc1)
ax.text(4.75, 6.7, "HREncoder (1D Res-CNN)", ha="center", fontsize=9, fontweight="bold")
ax.text(4.75, 6.35, "Conv1D + GN + Res + LN -> 64d", ha="center", fontsize=7.5, color="#2E7D32")

rect_enc2 = patches.FancyBboxPatch((3.5, 4.8), 2.5, 0.9, boxstyle="round,pad=0.1",
                                   facecolor=c_encoder, edgecolor="#2E7D32", linewidth=1.2)
ax.add_patch(rect_enc2)
ax.text(4.75, 5.4, "ACCEncoder (1D Res-CNN)", ha="center", fontsize=9, fontweight="bold")
ax.text(4.75, 5.05, "Conv1D + GN + Res + LN -> 64d", ha="center", fontsize=7.5, color="#2E7D32")

rect_meta = patches.FancyBboxPatch((3.5, 3.5), 2.5, 0.9, boxstyle="round,pad=0.1",
                                   facecolor=c_encoder, edgecolor="#2E7D32", linewidth=1.2)
ax.add_patch(rect_meta)
ax.text(4.75, 4.1, "Meta Projection", ha="center", fontsize=9, fontweight="bold")
ax.text(4.75, 3.75, "Linear(5 -> 16) + GELU", ha="center", fontsize=7.5, color="#2E7D32")

# Arrows to Encoders
ax.annotate("", xy=(3.5, 6.55), xytext=(3.0, 6.55), arrowprops=dict(arrowstyle="->", lw=1.2, color="#37474F"))
ax.annotate("", xy=(3.5, 5.25), xytext=(3.0, 5.25), arrowprops=dict(arrowstyle="->", lw=1.2, color="#37474F"))
ax.annotate("", xy=(3.5, 3.95), xytext=(3.0, 3.95), arrowprops=dict(arrowstyle="->", lw=1.2, color="#37474F"))

# Fusion Gate
rect_gate = patches.FancyBboxPatch((6.5, 5.2), 2.3, 1.5, boxstyle="round,pad=0.1",
                                   facecolor=c_fusion, edgecolor="#E65100", linewidth=1.4)
ax.add_patch(rect_gate)
ax.text(7.65, 6.4, "Adaptive Reliability Gate", ha="center", fontsize=9, fontweight="bold", color="#BF360C")
ax.text(7.65, 6.0, r"$\alpha_t = \sigma(W_2 \cdot \text{GELU}(W_1 [h_{\text{HR}}; h_{\text{ACC}}]))$", ha="center", fontsize=7.5)
ax.text(7.65, 5.5, r"$z_t = \alpha_t h_{\text{HR}} + (1 - \alpha_t) h_{\text{ACC}}$", ha="center", fontsize=8, fontweight="bold", color="#BF360C")

# Arrows into gate
ax.annotate("", xy=(6.5, 6.2), xytext=(6.0, 6.55), arrowprops=dict(arrowstyle="->", lw=1.2, color="#37474F"))
ax.annotate("", xy=(6.5, 5.7), xytext=(6.0, 5.25), arrowprops=dict(arrowstyle="->", lw=1.2, color="#37474F"))

# BiGRU Temporal Context Box
rect_bigru = patches.FancyBboxPatch((6.5, 3.2), 2.3, 1.4, boxstyle="round,pad=0.1",
                                    facecolor=c_temporal, edgecolor="#6A1B9A", linewidth=1.4)
ax.add_patch(rect_bigru)
ax.text(7.65, 4.25, "Bidirectional GRU", ha="center", fontsize=9, fontweight="bold", color="#4A148C")
ax.text(7.65, 3.9, "2 Layers, H=48*2 = 96d", ha="center", fontsize=8, color="#555")
ax.text(7.65, 3.55, "Sequence T = 600 epochs (5h)", ha="center", fontsize=7.5, color="#555")

# Arrow from gate & meta into BiGRU
ax.annotate("", xy=(7.65, 4.6), xytext=(7.65, 5.2), arrowprops=dict(arrowstyle="->", lw=1.2, color="#37474F"))
ax.annotate("", xy=(6.5, 3.8), xytext=(6.0, 3.95), arrowprops=dict(arrowstyle="->", lw=1.2, color="#37474F"))

# Transition Module & Head
rect_head = patches.FancyBboxPatch((4.2, 1.8), 2.4, 1.0, boxstyle="round,pad=0.1",
                                   facecolor=c_head, edgecolor="#C62828", linewidth=1.3)
ax.add_patch(rect_head)
ax.text(5.4, 2.5, "Stage Classifier Head", ha="center", fontsize=8.5, fontweight="bold", color="#B71C1C")
ax.text(5.4, 2.1, "MLP (96 -> 64 -> 4 classes)", ha="center", fontsize=7.5)

rect_trans = patches.FancyBboxPatch((7.0, 1.8), 2.0, 1.0, boxstyle="round,pad=0.1",
                                    facecolor=c_trans, edgecolor="#F57F17", linewidth=1.3)
ax.add_patch(rect_trans)
ax.text(8.0, 2.5, "Transition Module", ha="center", fontsize=8.5, fontweight="bold", color="#F57F17")
ax.text(8.0, 2.1, r"$\mathcal{M}_{4 \times 4} + \lambda_{\text{trans}} \mathcal{L}_{\text{trans}}$", ha="center", fontsize=7.5)

# GRL Subject Discriminator
rect_grl = patches.FancyBboxPatch((0.8, 1.8), 3.0, 1.0, boxstyle="round,pad=0.1",
                                  facecolor=c_grl, edgecolor="#4527A0", linewidth=1.3)
ax.add_patch(rect_grl)
ax.text(2.3, 2.5, "Subject Discriminator (GRL)", ha="center", fontsize=8.5, fontweight="bold", color="#311B92")
ax.text(2.3, 2.1, r"$\text{GRL}(\alpha=0.1) \to 31 \text{ train subjs } (\mathcal{L}_{\text{adv}})$", ha="center", fontsize=7.5)

# Arrows from BiGRU to branches
ax.annotate("", xy=(5.4, 2.8), xytext=(7.0, 3.2), arrowprops=dict(arrowstyle="->", lw=1.2, color="#37474F"))
ax.annotate("", xy=(8.0, 2.8), xytext=(7.8, 3.2), arrowprops=dict(arrowstyle="->", lw=1.2, color="#37474F"))
ax.annotate("", xy=(2.3, 2.8), xytext=(6.5, 3.4), arrowprops=dict(arrowstyle="->", lw=1.2, color="#37474F", connectionstyle="arc3,rad=-0.2"))

# Output Staging Banner
rect_out = patches.FancyBboxPatch((4.0, 0.6), 4.8, 0.8, boxstyle="round,pad=0.1",
                                  facecolor="#E8EAF6", edgecolor="#1A237E", linewidth=1.4)
ax.add_patch(rect_out)
ax.text(6.4, 1.15, "Final 4-Class Sleep Stage Predictions & Composite Objective", ha="center", fontsize=8.5, fontweight="bold", color="#1A237E")
ax.text(6.4, 0.85, r"$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + 0.01 \mathcal{L}_{\text{RW}} + 0.10 \mathcal{L}_{\text{trans}} + 0.10 \mathcal{L}_{\text{adv}}$", ha="center", fontsize=8)

# ----------------- PANEL B: SLAMSS-IFS Baseline (Right) -----------------
rect_b = patches.FancyBboxPatch((9.6, 0.4), 4.0, 7.3, boxstyle="round,pad=0.2",
                                facecolor="#FFFDE7", edgecolor="#F57F17", linewidth=1.8)
ax.add_patch(rect_b)
ax.text(9.9, 7.4, "B. Baseline SLAMSS-IFS  [5,465,768 Params]",
        fontsize=11.5, fontweight="bold", color="#B78103")

# Sub-blocks in SLAMSS-IFS
slamss_items = [
    ("Intra-Epoch CNN", "5 Conv1D layers (128 ch, K=5)\nMaxPool(6), MaxPool(5) -> 128d", "#FFF9C4"),
    ("Intra-Epoch BiGRU / LSTM", "2-layer BiGRU/BiLSTM (H=32)\nSample t=29 extracted -> 128d", "#FFF9C4"),
    ("Multi-Scale CNN3 Block", "Massive Conv11 + Conv15 + Conv21\n512 channels [4,705,792 parameters!]", "#FFECB3"),
    ("CNN_ACT Subnetwork", "5-channel time, freq & HR stats\nTemporal convolution block", "#FFF9C4"),
    ("Seq2Seq Inter-Epoch Model", "Encoder GRU (H=96) + Decoder GRU (H=96)\nBahdanau Additive Attention", "#FFE082"),
    ("Loss & Output Head", "Skip connection (258d) bypassed\nLinear(450/841 -> 4) + RCCLoss", "#FFD54F")
]

y_pos = 6.4
for title, desc, col in slamss_items:
    r = patches.FancyBboxPatch((9.9, y_pos - 0.7), 3.4, 0.75, boxstyle="round,pad=0.08",
                               facecolor=col, edgecolor="#E65100", linewidth=0.9)
    ax.add_patch(r)
    ax.text(11.6, y_pos - 0.15, title, ha="center", fontsize=8.5, fontweight="bold", color="#BF360C")
    ax.text(11.6, y_pos - 0.48, desc, ha="center", fontsize=7.0, color="#3E2723")
    y_pos -= 0.95

# Highlight Comparison Callout Box
rect_callout = patches.FancyBboxPatch((9.8, 0.6), 3.6, 0.8, boxstyle="round,pad=0.1",
                                      facecolor="#E0F2F1", edgecolor="#00796B", linewidth=1.3)
ax.add_patch(rect_callout)
ax.text(11.6, 1.15, "Efficiency & Design Contrast", ha="center", fontsize=8.5, fontweight="bold", color="#004D40")
ax.text(11.6, 0.85, "SleepAdaptiveNet is 33.7x smaller (162k vs 5.47M),\navoiding bloated CNN3 while adding GRL & Gating.", ha="center", fontsize=7.2, color="#004D40")

plt.tight_layout()
plt.savefig("report_assets/fig1_architecture_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig1_architecture_comparison.png")

# -----------------------------------------------------------------------------
# 2. BENCHMARK COMPARISON BAR CHART (Parameters, Accuracy, Weighted-F1)
# -----------------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 4.2), dpi=300)

models = ["SLAMSS-IFS\n(Baseline Repo)", "SleepAdaptiveNet\n(EXP-012 B0_GRL)"]
colors = ["#FFA000", "#1976D2"]

# Panel 1: Parameter Count (Millions)
params = [5.465768, 0.162325]
bars1 = ax1.bar(models, params, color=colors, width=0.5, edgecolor="#333", linewidth=1.2)
ax1.set_ylabel("Parameters (Millions)", fontsize=11, fontweight="bold")
ax1.set_title("Model Capacity Comparison\n(33.7x Compression)", fontsize=11, fontweight="bold", pad=10)
ax1.set_ylim(0, 6.2)
for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"{h:.3f}M\n({int(h*1e6):,} params)",
             ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax1.grid(axis="y", linestyle="--", alpha=0.5)

# Panel 2: Accuracy (%)
accs = [71.04, 68.44]
acc_errs = [0.0, 1.35]
bars2 = ax2.bar(models, accs, yerr=acc_errs, capsize=6, color=colors, width=0.5, edgecolor="#333", linewidth=1.2)
ax2.set_ylabel("Test Accuracy (%)", fontsize=11, fontweight="bold")
ax2.set_title("Held-Out Test Accuracy\n(Difference: -2.60 pp)", fontsize=11, fontweight="bold", pad=10)
ax2.set_ylim(60, 78)
for i, bar in enumerate(bars2):
    h = bar.get_height()
    lbl = f"{h:.2f}%" if i == 0 else f"{h:.2f}% ± {acc_errs[i]:.2f}%"
    ax2.text(bar.get_x() + bar.get_width()/2, h + 1.6, lbl,
             ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax2.grid(axis="y", linestyle="--", alpha=0.5)

# Panel 3: Weighted-F1 Score
wf1s = [0.7079, 0.6835]
wf1_errs = [0.0, 0.0141]
bars3 = ax3.bar(models, wf1s, yerr=wf1_errs, capsize=6, color=colors, width=0.5, edgecolor="#333", linewidth=1.2)
ax3.set_ylabel("Weighted-F1 Score", fontsize=11, fontweight="bold")
ax3.set_title("Matched Weighted-F1\n(Difference: -0.0244)", fontsize=11, fontweight="bold", pad=10)
ax3.set_ylim(0.60, 0.78)
for i, bar in enumerate(bars3):
    h = bar.get_height()
    lbl = f"{h:.4f}" if i == 0 else f"{h:.4f} ± {wf1_errs[i]:.4f}"
    ax3.text(bar.get_x() + bar.get_width()/2, h + 0.015, lbl,
             ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax3.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("report_assets/fig2_benchmark_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig2_benchmark_comparison.png")

# -----------------------------------------------------------------------------
# 3. PER-CLASS METRICS BREAKDOWN (Precision, Recall, F1)
# -----------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)

classes = ["Wake (0)", "Light (1)", "Deep (2)", "REM (3)"]
x = np.arange(len(classes))
width = 0.25

precision_mean = [64.01, 65.68, 73.89, 71.10]
precision_sd   = [4.10, 0.94, 3.46, 1.47]

recall_mean    = [75.95, 70.02, 72.66, 58.15]
recall_sd      = [1.20, 1.72, 1.86, 2.94]

f1_mean        = [69.40, 67.78, 73.23, 63.95]
f1_sd          = [1.92, 1.19, 1.74, 2.07]

rects1 = ax.bar(x - width, precision_mean, width, yerr=precision_sd, capsize=4,
                label="Precision (%)", color="#42A5F5", edgecolor="#333", linewidth=1.0)
rects2 = ax.bar(x, recall_mean, width, yerr=recall_sd, capsize=4,
                label="Recall (%)", color="#66BB6A", edgecolor="#333", linewidth=1.0)
rects3 = ax.bar(x + width, f1_mean, width, yerr=f1_sd, capsize=4,
                label="F1-Score (%)", color="#FFA726", edgecolor="#333", linewidth=1.0)

ax.set_ylabel("Score (%)", fontsize=11, fontweight="bold")
ax.set_title("SleepAdaptiveNet Test Performance by Sleep Stage (Mean ± SD across 3 Seeds)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=10, fontweight="bold")
ax.set_ylim(40, 90)
ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9.5)
ax.grid(axis="y", linestyle="--", alpha=0.5)

# Add values above bars
for rects in [rects1, rects2, rects3]:
    for rect in rects:
        h = rect.get_height()
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h + 2.5),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                    fontsize=7.8, fontweight="bold")

plt.tight_layout()
plt.savefig("report_assets/fig3_per_class_breakdown.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig3_per_class_breakdown.png")

# -----------------------------------------------------------------------------
# 4. TRANSITION BOUNDARY VS STABLE NON-BOUNDARY ERROR DYNAMICS
# -----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)

# Panel 1: Error Rates
regimes = ["Transition Boundaries\n(yt ≠ yt-1)\n1,379 Windows", "Stable Epochs\n(yt = yt-1)\n32,862 Windows"]
err_means = [59.08, 30.41]
err_sds = [0.47, 1.39]
colors_err = ["#EF5350", "#26A69A"]

bars = ax1.bar(regimes, err_means, yerr=err_sds, capsize=6, color=colors_err, width=0.45, edgecolor="#333", linewidth=1.2)
ax1.set_ylabel("Classification Error Rate (%)", fontsize=11, fontweight="bold")
ax1.set_title("Error Rate: Boundary vs. Stable Epochs\n(1.94x Relative Error Ratio)", fontsize=11, fontweight="bold", pad=10)
ax1.set_ylim(0, 75)
ax1.grid(axis="y", linestyle="--", alpha=0.5)

for bar in bars:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 2.5, f"{h:.2f}%\n(±{err_sds[0 if h>50 else 1]:.2f}%)",
             ha="center", va="bottom", fontsize=10, fontweight="bold")

# Panel 2: Total Test Window Breakdown Pie Chart
counts = [1379, 32862]
labels = ["Transition Boundaries\n(1,379 windows, 4.03%)", "Stable Epochs\n(32,862 windows, 95.97%)"]
colors_pie = ["#EF5350", "#80CBC4"]
explode = (0.1, 0)

ax2.pie(counts, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors_pie,
        explode=explode, textprops=dict(fontsize=9.5, fontweight="bold"),
        wedgeprops=dict(edgecolor="#333", linewidth=1.2))
ax2.set_title("Distribution of 34,241 Real Test Windows", fontsize=11, fontweight="bold", pad=10)

plt.tight_layout()
plt.savefig("report_assets/fig4_boundary_error_analysis.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig4_boundary_error_analysis.png")

# -----------------------------------------------------------------------------
# 5. PER-SUBJECT PERFORMANCE BREAKDOWN ACROSS 11 TEST SUBJECTS
# -----------------------------------------------------------------------------
sub_df = pd.read_csv("EXP012_AUDIT/results/exp012_subject_metrics_recomputed.csv")
sub_summary = sub_df.groupby("subject_id").agg({
    "accuracy": "mean",
    "macro_f1": "mean",
    "kappa": "mean",
    "real_windows": "first"
}).reset_index()

# Sort by accuracy
sub_summary = sub_summary.sort_values(by="accuracy", ascending=True)

fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)

y_pos = np.arange(len(sub_summary))
height = 0.38

bars_acc = ax.barh(y_pos + height/2, sub_summary["accuracy"] * 100, height,
                   label="Accuracy (%)", color="#1E88E5", edgecolor="#333", linewidth=1.0)
bars_f1 = ax.barh(y_pos - height/2, sub_summary["macro_f1"] * 100, height,
                  label="Macro-F1 (%)", color="#FB8C00", edgecolor="#333", linewidth=1.0)

ax.set_yticks(y_pos)
ax.set_yticklabels([f"{row['subject_id']} ({row['real_windows']:,} w)" for _, row in sub_summary.iterrows()],
                   fontsize=9.5, fontweight="bold")
ax.set_xlabel("Score (%)", fontsize=11, fontweight="bold")
ax.set_title("Test Set Generalization Across All 11 Independent Test Subjects (Mean of 3 Seeds)",
             fontsize=12, fontweight="bold", pad=12)
ax.set_xlim(45, 88)
ax.axvline(68.44, color="#D32F2F", linestyle="--", linewidth=1.5, label="Pooled Test Accuracy (68.44%)")
ax.axvline(68.29, color="#7B1FA2", linestyle=":", linewidth=1.5, label="Subject-Mean Accuracy (68.29%)")

ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9, loc="lower right")
ax.grid(axis="x", linestyle="--", alpha=0.5)

# Value labels on bars
for bar in bars_acc:
    w = bar.get_width()
    ax.text(w + 0.6, bar.get_y() + bar.get_height()/2, f"{w:.1f}%",
            va="center", ha="left", fontsize=7.8, fontweight="bold", color="#1565C0")

plt.tight_layout()
plt.savefig("report_assets/fig5_per_subject_performance.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved fig5_per_subject_performance.png")

print("All charts successfully generated in report_assets/!")
