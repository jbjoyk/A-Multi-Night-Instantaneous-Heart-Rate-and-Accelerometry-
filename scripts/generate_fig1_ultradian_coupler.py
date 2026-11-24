#!/usr/bin/env python3
"""
Code-Verified Publication Figure 1 Generator for EXP-029:
Ultradian-Pyramidal-Coupler (v5, 190,788 parameters).

Programmatically verifies exact module parameter counts from
build_exp029_model("full_v5") before rendering vector PDF and 300-DPI PNG
on a clean white background.
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
for _p in (REPO_ROOT, EXP029_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp029_models import build_exp029_model


def verify_module_params() -> dict[str, int]:
    model = build_exp029_model("full_v5")

    def count(m):
        return sum(p.numel() for p in m.parameters())

    p_hr = count(model.hr_encoder)
    p_acc = count(model.acc_encoder)
    p_meta = count(model.meta_proj)
    p_enc_meta = p_hr + p_acc + p_meta

    p_film = count(model.intra_film)
    p_fine = count(model.fine_gru)
    p_coarse = count(model.coarse_gru)
    p_fusion_norm = count(model.fusion_proj) + count(model.temp_norm)
    p_temporal = p_fine + p_coarse + p_fusion_norm
    p_head = count(model.classifier_head)
    p_total = count(model)

    assert p_hr == 29_568 and p_acc == 29_568 and p_meta == 96
    assert p_enc_meta == 59_232, f"Expected 59,232, got {p_enc_meta}"
    assert p_film == 8_320, f"Expected 8,320, got {p_film}"
    assert p_temporal == 116_640, f"Expected 116,640, got {p_temporal}"
    assert p_head == 6_596, f"Expected 6,596, got {p_head}"
    assert p_total == 190_788, f"Expected 190,788, got {p_total}"
    assert p_enc_meta + p_film + p_temporal + p_head == p_total

    return {
        "hr": p_hr,
        "acc": p_acc,
        "meta": p_meta,
        "enc_meta": p_enc_meta,
        "film": p_film,
        "fine": p_fine,
        "coarse": p_coarse,
        "fusion_norm": p_fusion_norm,
        "temporal": p_temporal,
        "head": p_head,
        "total": p_total,
    }


def draw_box(ax, x, y, w, h, title, subtitle="", fc="#F8FAFC", ec="#334155", lw=1.4, title_color="#0F172A", sub_color="#475569", fontsize=9.5):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(box)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=title_color, zorder=4)
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=fontsize - 1.5, color=sub_color, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=title_color, zorder=4)


def draw_arrow(ax, x1, y1, x2, y2, color="#475569", lw=1.4, style="-|>", ls="-"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle=style, color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0),
        zorder=2,
    )


def generate_figure():
    counts = verify_module_params()

    fig, ax = plt.subplots(figsize=(13.5, 6.4), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title
    ax.text(
        0.50, 0.955,
        "EXP-029 Ultradian-Pyramidal-Coupler (v5) Architecture",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#0F172A",
    )
    ax.text(
        0.50, 0.915,
        f"Code-Verified Parameter Accounting: {counts['enc_meta']:,} (Encoders+Meta) + {counts['film']:,} (FiLM) + {counts['temporal']:,} (2-Scale BiGRU+Norm) + {counts['head']:,} (Head) = {counts['total']:,} Params",
        ha="center", va="center", fontsize=9.5, color="#475569",
    )

    # Column 1: Inputs
    draw_box(ax, 0.02, 0.68, 0.12, 0.14, "IHR Input", "(B, T, 30) ~0.2 Hz", fc="#EFF6FF", ec="#2563EB")
    draw_box(ax, 0.02, 0.42, 0.12, 0.14, "ACC Input", "(B, T, 30) 3-axis mag", fc="#F0FDF4", ec="#16A34A")
    draw_box(ax, 0.02, 0.16, 0.12, 0.14, "Meta Input", "(B, T, 5) [freq, HR stats]", fc="#F8FAFC", ec="#64748B")

    # Column 2: PrePool Encoders & Meta Proj
    draw_box(ax, 0.18, 0.68, 0.15, 0.14, "HR PrePool CNN", f"(B*T, 64, 7) | {counts['hr']:,} p", fc="#DBEAFE", ec="#1D4ED8")
    draw_box(ax, 0.18, 0.42, 0.15, 0.14, "ACC PrePool CNN", f"(B*T, 64, 7) | {counts['acc']:,} p", fc="#DCFCE7", ec="#15803D")
    draw_box(ax, 0.18, 0.16, 0.15, 0.14, "Meta Linear+GELU", f"(B, T, 16) | {counts['meta']:,} p", fc="#F1F5F9", ec="#475569")

    draw_arrow(ax, 0.14, 0.75, 0.18, 0.75, color="#2563EB")
    draw_arrow(ax, 0.14, 0.49, 0.18, 0.49, color="#16A34A")
    draw_arrow(ax, 0.14, 0.23, 0.18, 0.23, color="#64748B")

    # Column 3: Sub-Epoch FiLM + Pool/Proj + Concat
    draw_box(ax, 0.37, 0.67, 0.15, 0.16, "Sub-Epoch FiLM", f"ACC->HR Residual\nConv1d(64,128,1) | {counts['film']:,} p", fc="#FEF3C7", ec="#D97706", fontsize=9.0)
    draw_box(ax, 0.37, 0.42, 0.15, 0.14, "Pool + Proj + LN", "h_hr, h_acc: (B, T, 64)", fc="#F8FAFC", ec="#475569")
    draw_box(ax, 0.37, 0.22, 0.15, 0.12, "Concat Features", "(B, T, 144) [hr; acc; meta]", fc="#EEF2FF", ec="#4F46E5")

    draw_arrow(ax, 0.33, 0.75, 0.37, 0.75, color="#2563EB")
    draw_arrow(ax, 0.33, 0.51, 0.37, 0.71, color="#D97706", ls="--")  # ACC -> FiLM
    draw_arrow(ax, 0.33, 0.47, 0.37, 0.47, color="#16A34A")             # ACC skip
    draw_arrow(ax, 0.445, 0.67, 0.445, 0.56, color="#D97706")           # FiLM -> Pool
    draw_arrow(ax, 0.445, 0.42, 0.445, 0.34, color="#475569")           # Pool -> Concat
    draw_arrow(ax, 0.33, 0.23, 0.37, 0.26, color="#64748B")             # Meta -> Concat

    # Column 4: 2-Scale Pyramidal BiGRU
    draw_box(ax, 0.56, 0.65, 0.18, 0.16, "Fine 1L BiGRU (30s)", f"144 -> 48/dir (96)\npack_padded | {counts['fine']:,} p", fc="#E0E7FF", ec="#4338CA", fontsize=9.0)
    draw_box(ax, 0.56, 0.43, 0.18, 0.12, "MaskWeightedPool 4x", "(B, ceil(T/4), 96) | 0 p", fc="#F1F5F9", ec="#64748B", fontsize=8.8)
    draw_box(ax, 0.56, 0.21, 0.18, 0.16, "Coarse 1L BiGRU (2m)", f"96 -> 48/dir (96)\nrepeat_interleave(4) | {counts['coarse']:,} p", fc="#E0E7FF", ec="#4338CA", fontsize=9.0)

    draw_arrow(ax, 0.52, 0.28, 0.56, 0.73, color="#4F46E5")
    draw_arrow(ax, 0.65, 0.65, 0.65, 0.55, color="#4338CA")
    draw_arrow(ax, 0.65, 0.43, 0.65, 0.37, color="#4338CA")

    # Column 5: Fusion + Classifier + Output
    draw_box(ax, 0.78, 0.54, 0.19, 0.16, "Fusion + LayerNorm", f"Linear(192->96) + LN(96)\n{counts['fusion_norm']:,} p", fc="#F3E8FF", ec="#7E22CE", fontsize=9.0)
    draw_box(ax, 0.78, 0.30, 0.19, 0.15, "Classifier Head", f"96 -> 64 -> 4\n{counts['head']:,} p", fc="#FCE7F3", ec="#BE185D", fontsize=9.0)
    draw_box(ax, 0.78, 0.10, 0.19, 0.12, "4-Class Sleep Stages", "Logit-Stitched L=1200\nWake / Light / Deep / REM", fc="#ECFDF5", ec="#059669", fontsize=8.8)

    draw_arrow(ax, 0.74, 0.73, 0.78, 0.65, color="#4338CA")
    draw_arrow(ax, 0.74, 0.29, 0.78, 0.59, color="#4338CA")
    draw_arrow(ax, 0.875, 0.54, 0.875, 0.45, color="#7E22CE")
    draw_arrow(ax, 0.875, 0.30, 0.875, 0.22, color="#BE185D")

    # Bottom Summary Strip
    ax.text(
        0.50, 0.04,
        "DEC-042 Verified Performance: BIDSleep 5-Fold CV (N=47) 66.74% (RCCLoss) / 66.95% (CE) Macro-F1 (+1.95 pp vs. Model A, p_Holm=1.43e-6; parity vs. 5.47M SLAMSS-IFS at 28.65x compression) | INT8: 360.50 KiB, 44.02 ms/night (4T)",
        ha="center", va="center", fontsize=8.4, color="#334155",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.0),
    )

    out_dir = os.path.join(REPO_ROOT, "figures")
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "fig1_ultradian_pyramidal_coupler.pdf")
    png_path = os.path.join(out_dir, "fig1_ultradian_pyramidal_coupler.png")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Verified module params: {counts}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    generate_figure()
