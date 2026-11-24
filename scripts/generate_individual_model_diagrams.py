#!/usr/bin/env python3
"""
Generate 5 separate, code-verified, publication-grade model architecture diagrams
(300 DPI PNG + vector PDF) in author_report_bundle/:
  1. fig_arch_model_a.{png,pdf}
  2. fig_arch_official_slamss_ifs.{png,pdf}
  3. fig_arch_full_v5.{png,pdf}
  4. fig_arch_p1_100k.{png,pdf}
  5. fig_arch_p190.{png,pdf}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "author_report_bundle"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for p in [
    REPO_ROOT,
    REPO_ROOT / "experiments" / "EXP-021",
    REPO_ROOT / "experiments" / "EXP-029",
    REPO_ROOT / "experiments" / "EXP-030",
    REPO_ROOT / "experiments" / "EXP-031",
]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp013_models import ConventionalCNNBiGRU  # noqa: E402
from exp029_models import build_exp029_model  # noqa: E402
from exp030_models import build_exp030_model  # noqa: E402
from exp031_models import build_exp031_model  # noqa: E402
from official_slamss_ifs_adapter import OfficialSLAMSSIFSWrapper  # noqa: E402

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def save_fig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT_DIR / f"{stem}.pdf", format="pdf", facecolor="white")
    fig.savefig(OUT_DIR / f"{stem}.png", format="png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}.png & {stem}.pdf")


def draw_box(
    ax, x: float, y: float, w: float, h: float, title: str, subtitle: str = "",
    fc: str = "#F8FAFC", ec: str = "#334155", lw: float = 1.3,
    tc: str = "#0F172A", sc: str = "#475569", fs: float = 8.6, sub_fs: float = 7.3,
    zorder: int = 3
) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.005,rounding_size=0.012",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=zorder + 1)
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
                fontsize=sub_fs, color=sc, zorder=zorder + 1)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=zorder + 1)


def draw_arrow(ax, x1: float, y1: float, x2: float, y2: float, color: str = "#475569", lw: float = 1.3, ls: str = "-") -> None:
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.40", color=color, lw=lw, linestyle=ls, shrinkA=0, shrinkB=0),
        zorder=2,
    )


def build_scaled_cnn_bigru_diagram(
    stem: str,
    title: str,
    model: torch.nn.Module,
    d_enc: int,
    d_temp: int,
    head_hidden: int,
    int8_kib: float,
    lat_4t_ms: float,
    footer_perf: str,
    accent: str,
) -> None:
    c1 = d_enc // 2
    c2 = d_enc
    p_hr = n_params(model.hr_encoder)
    p_acc = n_params(model.acc_encoder)
    p_meta = n_params(model.meta_proj)
    p_gru = n_params(model.temporal_gru) + n_params(model.temp_norm)
    p_head = n_params(model.classifier_head)
    p_tot = n_params(model)
    assert p_hr + p_acc + p_meta + p_gru + p_head == p_tot

    in_temp = 2 * d_enc + 16
    h_dir = d_temp // 2

    fig, ax = plt.subplots(figsize=(14.2, 6.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.50, 0.955, title, ha="center", va="center", fontsize=13.0, fontweight="bold", color="#0F172A")
    ax.text(
        0.50, 0.912,
        f"Code-Verified Params: {p_hr:,} (HR Enc) + {p_acc:,} (ACC Enc) + {p_meta:,} (Meta) + {p_gru:,} (2L BiGRU+LN) + {p_head:,} (Head) = {p_tot:,} | INT8: {int8_kib:.2f} KiB | 4T CPU: {lat_4t_ms:.2f} ms",
        ha="center", va="center", fontsize=8.6, color="#475569",
    )

    # Column 1: Inputs
    draw_box(ax, 0.02, 0.67, 0.13, 0.16, "IHR Input", "(B, L, 30)\n~0.2 Hz -> 1 Hz epoch", fc="#EFF6FF", ec="#2563EB")
    draw_box(ax, 0.02, 0.42, 0.13, 0.16, "ACC Input", "(B, L, 30)\n3-axis mag @ 1 Hz", fc="#F0FDF4", ec="#16A34A")
    draw_box(ax, 0.02, 0.17, 0.13, 0.16, "Meta Input", "(B, L, 5)\n[freq, 0, 0, mu, sd]", fc="#F8FAFC", ec="#64748B")

    # Column 2: Detailed 1D CNN Encoders + Meta
    hr_sub = (
        f"Conv1d(1->{c1}, k=5) -> GN(4) -> GELU -> Pool(2) [30->15]\n"
        f"ResBlock({c1}->{c2}, k=5) -> Pool(2) [15->7]\n"
        f"ResBlock({c2}->{c2}, k=3) -> AvgPool(1) -> Linear({c2}->{d_enc})+LN\n"
        f"Output: (B, L, {d_enc}) | {p_hr:,} params"
    )
    acc_sub = (
        f"Conv1d(1->{c1}, k=5) -> GN(4) -> GELU -> Pool(2) [30->15]\n"
        f"ResBlock({c1}->{c2}, k=5) -> Pool(2) [15->7]\n"
        f"ResBlock({c2}->{c2}, k=3) -> AvgPool(1) -> Linear({c2}->{d_enc})+LN\n"
        f"Output: (B, L, {d_enc}) | {p_acc:,} params"
    )
    draw_box(ax, 0.19, 0.64, 0.27, 0.22, f"HREncoder (d_enc={d_enc})", hr_sub, fc="#DBEAFE", ec="#1D4ED8", sub_fs=6.8)
    draw_box(ax, 0.19, 0.39, 0.27, 0.22, f"ACCEncoder (d_enc={d_enc})", acc_sub, fc="#DCFCE7", ec="#15803D", sub_fs=6.8)
    draw_box(ax, 0.19, 0.17, 0.27, 0.16, "Meta Projection", f"Linear(5 -> 16) + GELU\nOutput: (B, L, 16) | {p_meta:,} params", fc="#F1F5F9", ec="#475569")

    draw_arrow(ax, 0.15, 0.75, 0.19, 0.75, color="#2563EB")
    draw_arrow(ax, 0.15, 0.50, 0.19, 0.50, color="#16A34A")
    draw_arrow(ax, 0.15, 0.25, 0.19, 0.25, color="#64748B")

    # Column 3: Concat + 2L BiGRU
    draw_box(ax, 0.50, 0.41, 0.13, 0.18, "Concatenation", f"[h_HR ; h_ACC ; h_meta]\n(B, L, {in_temp})\n0 params", fc="#EEF2FF", ec="#4F46E5")
    draw_arrow(ax, 0.46, 0.75, 0.50, 0.54, color="#1D4ED8")
    draw_arrow(ax, 0.46, 0.50, 0.50, 0.50, color="#15803D")
    draw_arrow(ax, 0.46, 0.25, 0.50, 0.46, color="#475569")

    draw_box(
        ax, 0.66, 0.37, 0.16, 0.26,
        "2-Layer BiGRU + LN",
        f"in={in_temp}, hidden={h_dir}/dir\nd_temp={d_temp}, drop=0.2\npack_padded_sequence\n+ LayerNorm({d_temp})\n{p_gru:,} params",
        fc="#E0E7FF", ec="#4338CA", sub_fs=7.0,
    )
    draw_arrow(ax, 0.63, 0.50, 0.66, 0.50, color="#4F46E5")

    # Column 4: Classifier + Output
    draw_box(
        ax, 0.85, 0.54, 0.13, 0.22,
        "Classifier Head",
        f"Linear({d_temp}->{head_hidden})\nLN({head_hidden}) + GELU\nDrop(0.2) + Linear({head_hidden}->4)\n{p_head:,} params",
        fc="#FCE7F3", ec="#BE185D", sub_fs=6.8,
    )
    draw_box(
        ax, 0.85, 0.25, 0.13, 0.18,
        "4-Stage Output",
        "Logit-Stitched L=1200\nSoftmax -> (B, L, 4)\nWake / Light / Deep / REM",
        fc="#ECFDF5", ec=accent, sub_fs=6.9,
    )
    draw_arrow(ax, 0.82, 0.54, 0.85, 0.65, color="#4338CA")
    draw_arrow(ax, 0.915, 0.54, 0.915, 0.43, color="#BE185D")

    ax.text(
        0.50, 0.055, footer_perf,
        ha="center", va="center", fontsize=8.3, color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F8FAFC", edgecolor=accent, lw=1.1),
    )

    save_fig(fig, stem)


def build_official_slamss_diagram() -> None:
    model = OfficialSLAMSSIFSWrapper(device=torch.device("cpu"))
    p_cnn_conv = n_params(model.cnn.cnn)
    p_cnn_rnn = n_params(model.cnn.rnn)
    p_cnn_act = n_params(model.cnn.cnn_ACT)
    p_cnn3 = n_params(model.cnn.cnn3)
    p_cnn_tot = n_params(model.cnn)

    p_micro = n_params(model.s2s.rnn)
    p_enc = n_params(model.s2s.encoder)
    p_attn = n_params(model.s2s.decoder.attention)
    p_dec_rnn = n_params(model.s2s.decoder.rnn)
    p_dec_out = n_params(model.s2s.decoder.out)
    p_s2s_tot = n_params(model.s2s)
    p_tot = n_params(model)
    assert p_cnn_tot + p_s2s_tot == p_tot == 5_465_768

    fig, ax = plt.subplots(figsize=(15.0, 7.6), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.50, 0.96, "Baseline 2: Official SLAMSS-IFS Architecture (5,465,768 Parameters)",
            ha="center", va="center", fontsize=13.0, fontweight="bold", color="#0F172A")
    ax.text(
        0.50, 0.92,
        f"Code-Verified Params: CNN={p_cnn_tot:,} (cnn3={p_cnn3:,}) + Seq2Seq={p_s2s_tot:,} (Encode_micro={p_micro:,}, Enc={p_enc:,}, Dec={p_attn+p_dec_rnn+p_dec_out:,}) = {p_tot:,} | INT8: 19.49 MiB | 4T CPU: 3,373.8 ms",
        ha="center", va="center", fontsize=8.4, color="#475569",
    )

    # Column 1: Inputs
    draw_box(ax, 0.015, 0.58, 0.125, 0.20, "Signal Input", "(B, L*30, 2)\n[IHR, ACC] @ 1 Hz\nL=1200 -> 36,000 steps", fc="#EFF6FF", ec="#2563EB", sub_fs=6.9)
    draw_box(ax, 0.015, 0.22, 0.125, 0.18, "Meta Input", "(B, L, 5)\n[freq, 0, 0, mu, sd]\ncircadian-neutralized", fc="#F8FAFC", ec="#64748B", sub_fs=6.9)

    # Column 2: 3 Parallel Feature Extractors + Encode_micro
    draw_box(ax, 0.175, 0.69, 0.20, 0.17, "CNN.cnn (5-Layer 1D CNN)", f"5x Conv1d(128, k=5) + LeakyReLU\nMaxPool(6) + MaxPool(5) [30->1]\n(B, 128, L) | {p_cnn_conv:,} p", fc="#DBEAFE", ec="#1D4ED8", sub_fs=6.7)
    draw_box(ax, 0.175, 0.49, 0.20, 0.16, "CNN.rnn (3-Layer BiGRU)", f"BiGRU(2->64/dir=128, drop=0.5)\nSubsampled [::30]\n(B, 128, L) | {p_cnn_rnn:,} p", fc="#DBEAFE", ec="#1D4ED8", sub_fs=6.7)
    draw_box(ax, 0.175, 0.29, 0.20, 0.16, "CNN.cnn_ACT (Meta CNN)", f"Conv1d(5->16->32->64, k=9)\nLeakyReLU + Drop(0.5)\n(B, 64, L) | {p_cnn_act:,} p", fc="#F1F5F9", ec="#475569", sub_fs=6.7)
    draw_box(ax, 0.175, 0.11, 0.20, 0.14, "Encode_micro (Per-Epoch)", f"2L BiGRU(2->32/dir=64) + Linear(64->128)\nRuns on (B, 30, 2) per step t\nmicro: (B, 128) | {p_micro:,} p", fc="#FEF3C7", ec="#D97706", sub_fs=6.6)

    draw_arrow(ax, 0.14, 0.71, 0.175, 0.77, color="#2563EB")
    draw_arrow(ax, 0.14, 0.66, 0.175, 0.57, color="#2563EB")
    draw_arrow(ax, 0.14, 0.61, 0.175, 0.18, color="#D97706")
    draw_arrow(ax, 0.14, 0.31, 0.175, 0.37, color="#64748B")

    # Column 3: Multi-Scale CNN3 Fusion
    draw_box(ax, 0.41, 0.47, 0.18, 0.27, "Multi-Scale CNN3", f"Concat [128; 128; 64] = 320-d\nConv1d(320->256, k=11)\nConv1d(256->256, k=15)\nConv1d(256->512, k=21)\n+ Meta skip -> cnnf (B, L, 517)\n{p_cnn3:,} params (86.0% of model)", fc="#FFEDD5", ec="#D55E00", sub_fs=6.8)
    draw_arrow(ax, 0.375, 0.77, 0.41, 0.66, color="#1D4ED8")
    draw_arrow(ax, 0.375, 0.57, 0.41, 0.60, color="#1D4ED8")
    draw_arrow(ax, 0.375, 0.37, 0.41, 0.54, color="#475569")

    # Column 4: Seq2Seq Encoder + Attention
    draw_box(ax, 0.625, 0.57, 0.15, 0.21, "Seq2Seq Encoder", f"1-Layer Uni-GRU(517->96)\n+ Linear(96->96)\nenc_out: (B, L, 96)\n{p_enc:,} params", fc="#E0E7FF", ec="#4338CA", sub_fs=6.8)
    draw_box(ax, 0.625, 0.31, 0.15, 0.19, "Bahdanau Attention", f"Linear(192->96) + tanh\nSoftmax weights over L\nweighted: (B, 96)\n{p_attn:,} params", fc="#F3E8FF", ec="#7E22CE", sub_fs=6.8)
    draw_arrow(ax, 0.59, 0.63, 0.625, 0.67, color="#D55E00")
    draw_arrow(ax, 0.70, 0.57, 0.70, 0.50, color="#4338CA")

    # Column 5: Autoregressive Decoder + Output
    draw_box(ax, 0.81, 0.42, 0.175, 0.32, "Seq2Seq Decoder\n(t = 0 .. L-1)", f"Zero-Leakage TF=0.0 (trg_zeros)\nUni-GRU(96+4 -> 96) [{p_dec_rnn:,} p]\nConcat [out(96); attn(96); in(4);\n  cnnf(517); micro(128)] = 841-d\nLinear(841 -> 4) [{p_dec_out:,} p]", fc="#FCE7F3", ec="#BE185D", sub_fs=6.7)
    draw_box(ax, 0.81, 0.18, 0.175, 0.16, "4-Stage Output", "Logit-Stitched L=1200\nSoftmax -> (B, L, 4)\nWake / Light / Deep / REM", fc="#ECFDF5", ec="#059669", sub_fs=6.9)

    draw_arrow(ax, 0.775, 0.67, 0.81, 0.62, color="#4338CA")
    draw_arrow(ax, 0.775, 0.40, 0.81, 0.50, color="#7E22CE")
    draw_arrow(ax, 0.59, 0.54, 0.81, 0.54, color="#D55E00", ls="--")
    draw_arrow(ax, 0.375, 0.18, 0.81, 0.46, color="#D97706", ls="--")
    draw_arrow(ax, 0.8975, 0.42, 0.8975, 0.34, color="#BE185D")

    ax.text(
        0.50, 0.045,
        "DEC-038 & DEC-041 Verified: BIDSleep 5-Fold CV (N=47) 67.37% Macro-F1 (Acc 68.75%, kappa 0.5378) | SleepAccel (n=25) 61.08% single / 62.79% ensemble | 5.12 GiB Training VRAM",
        ha="center", va="center", fontsize=8.3, color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF7ED", edgecolor="#D55E00", lw=1.1),
    )

    save_fig(fig, "fig_arch_official_slamss_ifs")


def build_full_v5_diagram() -> None:
    model = build_exp029_model("full_v5")
    p_hr = n_params(model.hr_encoder)
    p_acc = n_params(model.acc_encoder)
    p_meta = n_params(model.meta_proj)
    p_film = n_params(model.intra_film)
    p_fine = n_params(model.fine_gru)
    p_coarse = n_params(model.coarse_gru)
    p_fn = n_params(model.fusion_proj) + n_params(model.temp_norm)
    p_head = n_params(model.classifier_head)
    p_tot = n_params(model)
    assert p_tot == 190_788

    fig, ax = plt.subplots(figsize=(14.5, 6.8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.50, 0.955, "EXP-029: full_v5 (Ultradian-Pyramidal-Coupler, 190,788 Parameters)",
            ha="center", va="center", fontsize=13.0, fontweight="bold", color="#0F172A")
    ax.text(
        0.50, 0.912,
        f"Code-Verified Params: {p_hr+p_acc+p_meta:,} (Encoders+Meta) + {p_film:,} (FiLM) + {p_fine+p_coarse+p_fn:,} (2-Scale BiGRU+Norm) + {p_head:,} (Head) = {p_tot:,} | INT8: 360.50 KiB | 4T CPU: 44.02 ms",
        ha="center", va="center", fontsize=8.5, color="#475569",
    )

    draw_box(ax, 0.02, 0.68, 0.115, 0.14, "IHR Input", "(B, L, 30)\n~0.2 Hz", fc="#EFF6FF", ec="#2563EB")
    draw_box(ax, 0.02, 0.42, 0.115, 0.14, "ACC Input", "(B, L, 30)\n3-axis mag", fc="#F0FDF4", ec="#16A34A")
    draw_box(ax, 0.02, 0.16, 0.115, 0.14, "Meta Input", "(B, L, 5)\n[freq, HR stats]", fc="#F8FAFC", ec="#64748B")

    draw_box(ax, 0.165, 0.66, 0.18, 0.18, "HR PrePoolEncoder", f"Conv1d(1->32->64->64)\nSub-epoch (B*L, 64, 7)\n~0.233 Hz | {p_hr:,} p", fc="#DBEAFE", ec="#1D4ED8", sub_fs=6.9)
    draw_box(ax, 0.165, 0.40, 0.18, 0.18, "ACC PrePoolEncoder", f"Conv1d(1->32->64->64)\nSub-epoch (B*L, 64, 7)\n~0.233 Hz | {p_acc:,} p", fc="#DCFCE7", ec="#15803D", sub_fs=6.9)
    draw_box(ax, 0.165, 0.16, 0.18, 0.14, "Meta Projection", f"Linear(5->16) + GELU\n(B, L, 16) | {p_meta:,} p", fc="#F1F5F9", ec="#475569", sub_fs=6.9)

    draw_arrow(ax, 0.135, 0.75, 0.165, 0.75, color="#2563EB")
    draw_arrow(ax, 0.135, 0.49, 0.165, 0.49, color="#16A34A")
    draw_arrow(ax, 0.135, 0.23, 0.165, 0.23, color="#64748B")

    draw_box(ax, 0.375, 0.66, 0.165, 0.18, "IntraEpochFiLM", f"ACC->HR Residual Mod.\nConv1d(64->128, k=1) zero-init\ngamma=1+tanh(dg) | {p_film:,} p", fc="#FEF3C7", ec="#D97706", sub_fs=6.7)
    draw_box(ax, 0.375, 0.41, 0.165, 0.15, "AvgPool(1) + Proj + LN", "h_hr, h_acc: (B, L, 64)\nDrop(0.2) + Linear(64->64)", fc="#F8FAFC", ec="#475569", sub_fs=6.8)
    draw_box(ax, 0.375, 0.20, 0.165, 0.13, "Concat Features", "(B, L, 144)\n[h_HR ; h_ACC ; h_meta]", fc="#EEF2FF", ec="#4F46E5", sub_fs=6.9)

    draw_arrow(ax, 0.345, 0.75, 0.375, 0.75, color="#1D4ED8")
    draw_arrow(ax, 0.345, 0.51, 0.375, 0.71, color="#D97706", ls="--")
    draw_arrow(ax, 0.345, 0.46, 0.375, 0.46, color="#15803D")
    draw_arrow(ax, 0.4575, 0.66, 0.4575, 0.56, color="#D97706")
    draw_arrow(ax, 0.4575, 0.41, 0.4575, 0.33, color="#475569")
    draw_arrow(ax, 0.345, 0.23, 0.375, 0.26, color="#475569")

    draw_box(ax, 0.57, 0.65, 0.185, 0.17, "Fine 1L BiGRU (30s)", f"in=144, hidden=48/dir (96)\npack_padded | {p_fine:,} p", fc="#E0E7FF", ec="#4338CA", sub_fs=6.9)
    draw_box(ax, 0.57, 0.43, 0.185, 0.13, "MaskWeightedPool1d 4x", "(B, ceil(L/4), 96) [2-min]\n0 params", fc="#F1F5F9", ec="#64748B", sub_fs=6.9)
    draw_box(ax, 0.57, 0.20, 0.185, 0.17, "Coarse 1L BiGRU (2m)", f"in=96, hidden=48/dir (96)\nrepeat_interleave(4) | {p_coarse:,} p", fc="#E0E7FF", ec="#4338CA", sub_fs=6.9)

    draw_arrow(ax, 0.54, 0.265, 0.57, 0.735, color="#4F46E5")
    draw_arrow(ax, 0.6625, 0.65, 0.6625, 0.56, color="#4338CA")
    draw_arrow(ax, 0.6625, 0.43, 0.6625, 0.37, color="#4338CA")

    draw_box(ax, 0.79, 0.56, 0.19, 0.17, "Pyramidal Fusion + LN", f"Concat [Fine; Coarse] (192)\nLinear(192->96) + LN(96)\n{p_fn:,} params", fc="#F3E8FF", ec="#7E22CE", sub_fs=6.9)
    draw_box(ax, 0.79, 0.33, 0.19, 0.16, "Classifier Head", f"Linear(96->64)+LN+GELU\nDrop(0.2)+Linear(64->4)\n{p_head:,} params", fc="#FCE7F3", ec="#BE185D", sub_fs=6.8)
    draw_box(ax, 0.79, 0.13, 0.19, 0.13, "4-Stage Output", "Logit-Stitched L=1200\nWake / Light / Deep / REM", fc="#ECFDF5", ec="#059669", sub_fs=6.9)

    draw_arrow(ax, 0.755, 0.73, 0.79, 0.67, color="#4338CA")
    draw_arrow(ax, 0.755, 0.28, 0.79, 0.61, color="#4338CA")
    draw_arrow(ax, 0.885, 0.56, 0.885, 0.49, color="#7E22CE")
    draw_arrow(ax, 0.885, 0.33, 0.885, 0.26, color="#BE185D")

    ax.text(
        0.50, 0.045,
        "DEC-042 & DEC-044: BIDSleep 66.74% (RCCLoss) / 66.95% (CE) | SleepAccel 61.78% / 63.51% | EXP-031 H-ARCH vs. P190: +0.14 pp (p_Holm=0.4037, NS)",
        ha="center", va="center", fontsize=8.3, color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FDF2F8", edgecolor="#CC79A7", lw=1.1),
    )

    save_fig(fig, "fig_arch_full_v5")


def main() -> None:
    # 1. Model A
    build_scaled_cnn_bigru_diagram(
        stem="fig_arch_model_a",
        title="Baseline 1: Model A (ConventionalCNNBiGRU, 163,940 Parameters)",
        model=ConventionalCNNBiGRU(),
        d_enc=64, d_temp=96, head_hidden=64,
        int8_kib=309.54, lat_4t_ms=47.73,
        footer_perf="DEC-037 & DEC-038: BIDSleep 5-Fold CV (N=47) 64.79% Macro-F1 (Acc 66.18%, kappa 0.5078) | SleepAccel (n=25) 60.51% single / 62.94% ensemble",
        accent="#4D4D4D",
    )
    # 2. Official SLAMSS-IFS
    build_official_slamss_diagram()
    # 3. full_v5
    build_full_v5_diagram()
    # 4. p1_100k
    build_scaled_cnn_bigru_diagram(
        stem="fig_arch_p1_100k",
        title="EXP-030 Pareto Selection: p1_100k (ScaledConventionalCNNBiGRU, 98,940 Parameters)",
        model=build_exp030_model("p1_100k"),
        d_enc=48, d_temp=76, head_hidden=48,
        int8_kib=182.00, lat_4t_ms=46.35,
        footer_perf="DEC-043: BIDSleep (N=47) 66.67% Macro-F1 (+1.88 pp vs. Model A, p_Holm=1.19e-6; parity vs. SLAMSS-IFS at 55.24x compression) | SleepAccel 60.88% / 62.51%",
        accent="#009E73",
    )
    # 5. P190
    build_scaled_cnn_bigru_diagram(
        stem="fig_arch_p190",
        title="EXP-031 Section 9 Selection: P190 (ScaledConventionalCNNBiGRU, w=1.08, 190,940 Parameters)",
        model=build_exp031_model("P190"),
        d_enc=72, d_temp=100, head_hidden=64,
        int8_kib=375.16, lat_4t_ms=53.07,
        footer_perf="DEC-044: BIDSleep (N=47) 66.60% Macro-F1 (+1.80 pp vs. Model A, p=2.96e-6; -0.77 pp vs. SLAMSS-IFS, p=0.0727) | SleepAccel 61.52% / 63.42%",
        accent="#0072B2",
    )


if __name__ == "__main__":
    main()
