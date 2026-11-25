#!/usr/bin/env python3
"""
Deliverable 2: Code-Extracted Architecture Diagrams & Spec JSON.
Instantiates all 5 models from the repository:
  - Baseline 1: Model A (ConventionalCNNBiGRU, 163,940 params)
  - Baseline 2: Official SLAMSS-IFS (OfficialSLAMSSIFSWrapper, 5,465,768 params)
  - Last 3 Trained Models:
      1. full_v5 (UltradianPyramidalCoupler, 190,788 params)
      2. p1_100k (ScaledConventionalCNNBiGRU, 98,940 params)
      3. P190 (ScaledConventionalCNNBiGRU, 190,940 params)

Asserts exact parameter totals, captures real intermediate tensor shapes via
PyTorch forward hooks at full-night context L=1200, exports
author_report_bundle/data/architecture_specs.json, and renders:
  - author_report_bundle/fig1_baseline_vs_proposed_architectures.pdf
  - author_report_bundle/fig1_baseline_vs_proposed_architectures.png (300 DPI)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
for p in [
    ROOT,
    ROOT / "experiments" / "EXP-021",
    ROOT / "experiments" / "EXP-029",
    ROOT / "experiments" / "EXP-030",
    ROOT / "experiments" / "EXP-031",
]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp013_models import ConventionalCNNBiGRU  # noqa: E402
from official_slamss_ifs_adapter import OfficialSLAMSSIFSWrapper  # noqa: E402
from exp029_models import UltradianPyramidalCoupler  # noqa: E402
from exp030_models import build_exp030_model  # noqa: E402
from exp031_models import build_exp031_model  # noqa: E402

OUT_DIR = ROOT / "author_report_bundle"
DATA_DIR = OUT_DIR / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans"],
    "font.size": 8.0,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.12,
})


def count_params(mod: nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)


def shape_str(obj: Any) -> str:
    if isinstance(obj, torch.Tensor):
        return "(" + ", ".join(str(x) for x in obj.shape) + ")"
    if isinstance(obj, nn.utils.rnn.PackedSequence):
        return f"Packed({tuple(obj.data.shape)})"
    if isinstance(obj, (tuple, list)):
        parts = [shape_str(x) for x in obj[:2]]
        return parts[0] if len(parts) == 1 else "; ".join(parts)
    return str(type(obj).__name__)


def trace_model(
    name: str,
    model: nn.Module,
    expected_params: int,
    is_slamss: bool = False,
) -> Dict[str, Any]:
    model.eval()
    total_params = count_params(model)
    assert total_params == expected_params, (
        f"{name}: expected {expected_params:,} params, got {total_params:,}"
    )

    hook_records: Dict[str, Dict[str, Any]] = {}
    handles = []

    def make_hook(mname: str):
        def hook(_mod, inp, out):
            if mname not in hook_records:
                hook_records[mname] = {
                    "in_shape": shape_str(inp[0]) if inp else "",
                    "out_shape": shape_str(out),
                    "params": count_params(_mod),
                    "type": _mod.__class__.__name__,
                }
        return hook

    for mname, mod in model.named_modules():
        if mname:
            handles.append(mod.register_forward_hook(make_hook(mname)))

    with torch.no_grad():
        if is_slamss:
            x_sig = torch.zeros(1, 1200 * 30, 2)
            x_meta = torch.zeros(1, 1200, 5)
            out = model(x_sig, x_meta)
            out_shape = shape_str(out)
        else:
            x_ihr = torch.zeros(1, 1200, 30)
            x_acc = torch.zeros(1, 1200, 30)
            x_meta = torch.zeros(1, 1200, 5)
            seq_lens = torch.tensor([1200], dtype=torch.int64)
            out = model(x_ihr, x_acc, x_meta, seq_lens=seq_lens)
            out_shape = shape_str(out["logits"])

    for h in handles:
        h.remove()

    children_summary = []
    for cname, child in model.named_children():
        cp = count_params(child)
        if cp == 0 and cname in {"ce_loss", "rcc_loss"}:
            continue
        rec = hook_records.get(cname, {})
        in_s = rec.get("in_shape", "")
        out_s = rec.get("out_shape", "")
        # UltradianPyramidalCoupler calls hr_encoder.forward_prepool / pool_and_project
        if not in_s and f"{cname}.prepool" in hook_records:
            in_s = "(1, 1200, 30)"
            out_s = hook_records[f"{cname}.prepool"]["out_shape"]
        children_summary.append({
            "name": cname,
            "type": child.__class__.__name__,
            "params": cp,
            "pct_of_total": round(100.0 * cp / total_params, 2),
            "in_shape": in_s,
            "out_shape": out_s,
        })

    return {
        "name": name,
        "class": model.__class__.__name__,
        "total_params": total_params,
        "output_shape": out_shape,
        "children": children_summary,
        "modules": hook_records,
    }


def draw_box(
    ax, x: float, y: float, w: float, h: float,
    title: str, subtitle: str = "",
    fc: str = "#F8FAFC", ec: str = "#334155", lw: float = 1.0,
    tc: str = "#0F172A", sc: str = "#334155",
    tfs: float = 7.4, sfs: float = 6.3, zorder: int = 3
) -> None:
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.65, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=tc, zorder=zorder + 1)
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
                fontsize=sfs, color=sc, zorder=zorder + 1)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold", color=tc, zorder=zorder + 1)


def arrow(ax, x1: float, y1: float, x2: float, y2: float, color: str = "#475569") -> None:
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.32",
                        color=color, lw=1.0, shrinkA=0, shrinkB=0),
        zorder=2,
    )


def build_specs() -> Dict[str, Any]:
    specs = {}
    specs["Model A"] = trace_model("Model A", ConventionalCNNBiGRU(), 163940, is_slamss=False)
    specs["Official SLAMSS-IFS"] = trace_model("Official SLAMSS-IFS", OfficialSLAMSSIFSWrapper(), 5465768, is_slamss=True)
    specs["full_v5"] = trace_model("full_v5", UltradianPyramidalCoupler("full_v5"), 190788, is_slamss=False)
    specs["p1_100k"] = trace_model("p1_100k", build_exp030_model("p1_100k"), 98940, is_slamss=False)
    specs["P190"] = trace_model("P190", build_exp031_model("P190"), 190940, is_slamss=False)

    with open(DATA_DIR / "architecture_specs.json", "w") as f:
        json.dump(specs, f, indent=2)
    print(f"Saved {DATA_DIR / 'architecture_specs.json'}")
    return specs


def render_cnn_bigru_card(ax, spec: Dict[str, Any], badge: str, accent: str, bg_tint: str) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    mods = spec["modules"]
    children = {c["name"]: c for c in spec["children"]}
    tot = spec["total_params"]

    # Outer card
    card = FancyBboxPatch((0.02, 0.02), 0.96, 0.95,
                          boxstyle="round,pad=0.01,rounding_size=0.025",
                          facecolor="#FFFFFF", edgecolor=accent, linewidth=1.4, zorder=1)
    ax.add_patch(card)

    # Header
    ax.text(0.50, 0.93, f"{spec['name']}  ({tot:,} params)",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color="#0F172A")
    ax.text(0.50, 0.885, badge, ha="center", va="center", fontsize=7.0, color=accent, fontweight="bold")

    # Input encoders (3 columns)
    hr_p = children["hr_encoder"]["params"]
    acc_p = children["acc_encoder"]["params"]
    meta_p = children["meta_proj"]["params"]
    hr_out = children["hr_encoder"]["out_shape"]
    acc_out = children["acc_encoder"]["out_shape"]
    meta_out = children["meta_proj"]["out_shape"]

    draw_box(ax, 0.05, 0.69, 0.28, 0.14,
             f"HREncoder ({hr_p:,}p)",
             f"(1,1200,30) -> {hr_out}",
             fc=bg_tint, ec=accent, tfs=6.8, sfs=5.8)
    draw_box(ax, 0.36, 0.69, 0.28, 0.14,
             f"ACCEncoder ({acc_p:,}p)",
             f"(1,1200,30) -> {acc_out}",
             fc=bg_tint, ec=accent, tfs=6.8, sfs=5.8)
    draw_box(ax, 0.67, 0.69, 0.28, 0.14,
             f"MetaProj ({meta_p:,}p)",
             f"(1,1200,5) -> {meta_out}",
             fc="#F1F5F9", ec="#64748B", tfs=6.8, sfs=5.8)

    # Arrows down to concat
    arrow(ax, 0.19, 0.69, 0.35, 0.59)
    arrow(ax, 0.50, 0.69, 0.50, 0.59)
    arrow(ax, 0.81, 0.69, 0.65, 0.59)

    gru_in = mods["temporal_gru"]["in_shape"]
    gru_out = children["temporal_gru"]["out_shape"]
    gru_p = children["temporal_gru"]["params"]
    norm_p = children["temp_norm"]["params"]
    norm_out = mods["temp_norm"]["out_shape"]
    head_p = children["classifier_head"]["params"]
    head_out = children["classifier_head"]["out_shape"]

    # Extract in_dim and d_temp from shapes (d_temp from LayerNorm output, 2*hidden_size)
    in_dim = gru_in.split(",")[-1].strip(") ")
    d_temp = norm_out.split(",")[-1].strip(") ")
    d_enc = hr_out.split(",")[-1].strip(") ")
    # Head hidden from classifier_head.0
    head_hid = mods["classifier_head.0"]["out_shape"].split(",")[-1].strip(") ")

    draw_box(ax, 0.15, 0.51, 0.70, 0.08,
             f"Concatenation: [h_HR; h_ACC; h_meta]  (d_enc={d_enc})",
             f"Tensor: {gru_in}",
             fc="#F8FAFC", ec="#475569", tfs=6.9, sfs=6.0)
    arrow(ax, 0.50, 0.51, 0.50, 0.44)

    draw_box(ax, 0.10, 0.28, 0.80, 0.16,
             f"2-Layer BiGRU + LayerNorm ({gru_p + norm_p:,}p)",
             f"in={in_dim}, hidden={int(d_temp)//2}/dir (d_temp={d_temp}) -> {norm_out}",
             fc=bg_tint, ec=accent, tfs=7.0, sfs=6.0)
    arrow(ax, 0.50, 0.28, 0.50, 0.21)

    draw_box(ax, 0.12, 0.06, 0.76, 0.15,
             f"Classifier Head ({head_p:,}p)",
             f"Linear({d_temp}->{head_hid}) -> LN -> GELU -> Linear({head_hid}->4) | {head_out}",
             fc="#F1F5F9", ec="#334155", tfs=6.9, sfs=5.9)


def render_full_v5_card(ax, spec: Dict[str, Any]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    mods = spec["modules"]
    children = {c["name"]: c for c in spec["children"]}
    tot = spec["total_params"]
    accent = "#0F766E"
    bg_tint = "#F0FDFA"

    card = FancyBboxPatch((0.02, 0.02), 0.96, 0.95,
                          boxstyle="round,pad=0.01,rounding_size=0.025",
                          facecolor="#FFFFFF", edgecolor=accent, linewidth=1.4, zorder=1)
    ax.add_patch(card)

    ax.text(0.50, 0.93, f"full_v5  ({tot:,} params)",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color="#0F172A")
    ax.text(0.50, 0.885, "EXP-029 UltradianPyramidalCoupler (FiLM + 2-Scale Pyramid)",
            ha="center", va="center", fontsize=6.8, color=accent, fontweight="bold")

    hr_p = children["hr_encoder"]["params"]
    acc_p = children["acc_encoder"]["params"]
    meta_p = children["meta_proj"]["params"]
    film_p = children["intra_film"]["params"]
    prepool_shape = mods["hr_encoder.prepool"]["out_shape"]
    film_out = children["intra_film"]["out_shape"]

    draw_box(ax, 0.05, 0.72, 0.28, 0.12,
             f"HR PrePool ({hr_p:,}p)",
             f"prepool: {prepool_shape}",
             fc=bg_tint, ec=accent, tfs=6.7, sfs=5.7)
    draw_box(ax, 0.36, 0.72, 0.28, 0.12,
             f"ACC PrePool ({acc_p:,}p)",
             f"prepool: {prepool_shape}",
             fc=bg_tint, ec=accent, tfs=6.7, sfs=5.7)
    draw_box(ax, 0.67, 0.72, 0.28, 0.12,
             f"MetaProj ({meta_p:,}p)",
             f"{children['meta_proj']['out_shape']}",
             fc="#F1F5F9", ec="#64748B", tfs=6.7, sfs=5.7)

    arrow(ax, 0.36, 0.78, 0.33, 0.78, color=accent)
    draw_box(ax, 0.08, 0.56, 0.56, 0.10,
             f"IntraEpochFiLM ({film_p:,}p, zero-init)",
             f"ACC conditions HR @7-step -> {film_out} -> pool (1,1200,64)",
             fc="#CCFBF1", ec=accent, tfs=6.7, sfs=5.7)
    arrow(ax, 0.19, 0.72, 0.25, 0.66)
    arrow(ax, 0.50, 0.72, 0.45, 0.66)

    fine_p = children["fine_gru"]["params"]
    fine_out = children["fine_gru"]["out_shape"]
    pool_out = children["mask_pool"]["out_shape"]
    coarse_p = children["coarse_gru"]["params"]
    coarse_out = children["coarse_gru"]["out_shape"]
    fus_p = children["fusion_proj"]["params"]
    fus_out = children["fusion_proj"]["out_shape"]
    head_p = children["classifier_head"]["params"]
    head_out = children["classifier_head"]["out_shape"]

    arrow(ax, 0.36, 0.56, 0.30, 0.49)
    arrow(ax, 0.81, 0.72, 0.70, 0.49)

    draw_box(ax, 0.06, 0.38, 0.42, 0.11,
             f"Fine 1L-BiGRU ({fine_p:,}p)",
             f"30s scale: {fine_out}",
             fc=bg_tint, ec=accent, tfs=6.6, sfs=5.7)
    draw_box(ax, 0.52, 0.38, 0.42, 0.11,
             f"Pool4x + Coarse BiGRU ({coarse_p:,}p)",
             f"{pool_out} -> {coarse_out}",
             fc=bg_tint, ec=accent, tfs=6.5, sfs=5.6)
    arrow(ax, 0.48, 0.435, 0.52, 0.435, color=accent)

    arrow(ax, 0.27, 0.38, 0.42, 0.29)
    arrow(ax, 0.73, 0.38, 0.58, 0.29)

    draw_box(ax, 0.10, 0.20, 0.80, 0.09,
             f"Pyramid Fusion + LayerNorm ({fus_p + children['temp_norm']['params']:,}p)",
             f"Concat[Fine, Upsample(Coarse)] -> Linear(192->96): {fus_out}",
             fc="#F8FAFC", ec="#475569", tfs=6.7, sfs=5.7)
    arrow(ax, 0.50, 0.20, 0.50, 0.15)

    draw_box(ax, 0.12, 0.05, 0.76, 0.10,
             f"Classifier Head ({head_p:,}p)",
             f"Linear(96->64) -> LN -> GELU -> Linear(64->4) | {head_out}",
             fc="#F1F5F9", ec="#334155", tfs=6.7, sfs=5.7)


def render_slamss_card(ax, spec: Dict[str, Any]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    mods = spec["modules"]
    children = {c["name"]: c for c in spec["children"]}
    tot = spec["total_params"]
    accent = "#B91C1C"
    bg_tint = "#FEF2F2"

    card = FancyBboxPatch((0.02, 0.02), 0.96, 0.95,
                          boxstyle="round,pad=0.01,rounding_size=0.025",
                          facecolor="#FFFFFF", edgecolor=accent, linewidth=1.4, zorder=1)
    ax.add_patch(card)

    ax.text(0.50, 0.93, f"Baseline 2: Official SLAMSS-IFS  ({tot:,} params | 33.34x Model A)",
            ha="center", va="center", fontsize=8.8, fontweight="bold", color="#0F172A")
    ax.text(0.50, 0.885, "EXP-021 Official Repo Wrapper (teacher_forcing_ratio=0.0, 5.12 GiB VRAM @ B=4)",
            ha="center", va="center", fontsize=6.9, color=accent, fontweight="bold")

    cnn_p = children["cnn"]["params"]
    s2s_p = children["s2s"]["params"]
    c_cnn_p = mods["cnn.cnn"]["params"]
    c_rnn_p = mods["cnn.rnn"]["params"]
    c_act_p = mods["cnn.cnn_ACT"]["params"]
    c_cnn3_p = mods["cnn.cnn3"]["params"]

    c_cnn_out = mods["cnn.cnn"]["out_shape"]
    c_rnn_out = mods["cnn.rnn"]["out_shape"]
    c_act_out = mods["cnn.cnn_ACT"]["out_shape"]
    c_cnn3_out = mods["cnn.cnn3"]["out_shape"]
    cnn_out = children["cnn"]["out_shape"]

    # 3 parallel branches inside CNN
    draw_box(ax, 0.05, 0.70, 0.28, 0.13,
             f"cnn.cnn ({c_cnn_p:,}p)",
             f"5x Conv1d(128,k=5) -> {c_cnn_out}",
             fc=bg_tint, ec=accent, tfs=6.7, sfs=5.7)
    draw_box(ax, 0.36, 0.70, 0.28, 0.13,
             f"cnn.rnn ({c_rnn_p:,}p)",
             f"3L-BiGRU(64) -> {c_rnn_out}",
             fc=bg_tint, ec=accent, tfs=6.7, sfs=5.7)
    draw_box(ax, 0.67, 0.70, 0.28, 0.13,
             f"cnn.cnn_ACT ({c_act_p:,}p)",
             f"3x Conv1d(k=9) -> {c_act_out}",
             fc="#F1F5F9", ec="#64748B", tfs=6.7, sfs=5.7)

    arrow(ax, 0.19, 0.70, 0.35, 0.60)
    arrow(ax, 0.50, 0.70, 0.50, 0.60)
    arrow(ax, 0.81, 0.70, 0.65, 0.60)

    draw_box(ax, 0.08, 0.45, 0.84, 0.15,
             f"Multi-Scale Fusion cnn.cnn3 ({c_cnn3_p:,}p | 86.0% of model) | CNN total: {cnn_p:,}p",
             f"Conv1d(320->256,k=11) -> (256->256,k=15) -> (256->512,k=21) -> {c_cnn3_out}; +meta -> {cnn_out}",
             fc="#FEE2E2", ec=accent, tfs=6.8, sfs=5.8)

    arrow(ax, 0.50, 0.45, 0.50, 0.37)

    enc_p = mods["s2s.encoder"]["params"]
    dec_p = mods["s2s.decoder"]["params"]
    enc_out = mods["s2s.encoder"]["out_shape"]

    draw_box(ax, 0.08, 0.22, 0.40, 0.15,
             f"s2s.encoder ({enc_p:,}p)",
             f"GRU(517->96) + Linear -> {enc_out}",
             fc="#F8FAFC", ec="#475569", tfs=6.7, sfs=5.7)
    draw_box(ax, 0.52, 0.22, 0.40, 0.15,
             f"s2s.decoder + Attn ({dec_p:,}p)",
             f"Bahdanau Attn + GRU(100->96) + Linear(841->4)",
             fc="#F8FAFC", ec="#475569", tfs=6.7, sfs=5.7)
    arrow(ax, 0.48, 0.295, 0.52, 0.295)

    arrow(ax, 0.72, 0.22, 0.50, 0.14)
    draw_box(ax, 0.15, 0.05, 0.70, 0.09,
             f"Seq2Seq Output ({s2s_p:,}p total)",
             f"Logits: {spec['output_shape']} | FP32 weights: 20.85 MiB",
             fc="#F1F5F9", ec="#334155", tfs=6.8, sfs=5.8)


def save_individual_diagrams(specs: Dict[str, Any]) -> None:
    configs = [
        ("fig1a_model_a", "Model A", lambda ax, sp: render_cnn_bigru_card(
            ax, sp,
            badge="Baseline 1 (EXP-013 / Stage 2 Frozen) | 309.54 KiB INT8",
            accent="#1D4ED8", bg_tint="#EFF6FF",
        ), (7.5, 5.6)),
        ("fig1b_official_slamss_ifs", "Official SLAMSS-IFS", lambda ax, sp: render_slamss_card(ax, sp), (8.6, 5.8)),
        ("fig1c_full_v5", "full_v5", lambda ax, sp: render_full_v5_card(ax, sp), (7.5, 5.8)),
        ("fig1d_p1_100k", "p1_100k", lambda ax, sp: render_cnn_bigru_card(
            ax, sp,
            badge="EXP-030 Pareto Elbow (55.24x vs SLAMSS) | 182.00 KiB INT8",
            accent="#7C3AED", bg_tint="#F5F3FF",
        ), (7.5, 5.6)),
        ("fig1e_p190", "P190", lambda ax, sp: render_cnn_bigru_card(
            ax, sp,
            badge="EXP-031 Matched-Param Plain Control | 375.16 KiB INT8",
            accent="#B45309", bg_tint="#FFFBEB",
        ), (7.5, 5.6)),
    ]
    for stem, key, render_fn, figsize in configs:
        fig, ax = plt.subplots(figsize=figsize)
        render_fn(ax, specs[key])
        out_pdf = OUT_DIR / f"{stem}.pdf"
        out_png = OUT_DIR / f"{stem}.png"
        fig.savefig(out_pdf, format="pdf", facecolor="white")
        fig.savefig(out_png, format="png", dpi=300, facecolor="white")
        plt.close(fig)
        print(f"Saved {out_pdf} and {out_png}")


def main() -> None:
    specs = build_specs()
    save_individual_diagrams(specs)

    fig = plt.figure(figsize=(15.5, 10.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.08], hspace=0.18)

    # Panel A: 2 columns (Model A vs Official SLAMSS-IFS)
    gs_a = gs[0].subgridspec(1, 2, wspace=0.06, width_ratios=[1.0, 1.15])
    ax_a1 = fig.add_subplot(gs_a[0, 0])
    ax_a2 = fig.add_subplot(gs_a[0, 1])

    render_cnn_bigru_card(
        ax_a1, specs["Model A"],
        badge="Baseline 1 (EXP-013 / Stage 2 Frozen) | 309.54 KiB INT8",
        accent="#1D4ED8", bg_tint="#EFF6FF",
    )
    render_slamss_card(ax_a2, specs["Official SLAMSS-IFS"])

    # Panel B: 3 columns (full_v5, p1_100k, P190)
    gs_b = gs[1].subgridspec(1, 3, wspace=0.05)
    ax_b1 = fig.add_subplot(gs_b[0, 0])
    ax_b2 = fig.add_subplot(gs_b[0, 1])
    ax_b3 = fig.add_subplot(gs_b[0, 2])

    render_full_v5_card(ax_b1, specs["full_v5"])
    render_cnn_bigru_card(
        ax_b2, specs["p1_100k"],
        badge="EXP-030 Pareto Elbow (55.24x vs SLAMSS) | 182.00 KiB INT8",
        accent="#7C3AED", bg_tint="#F5F3FF",
    )
    render_cnn_bigru_card(
        ax_b3, specs["P190"],
        badge="EXP-031 Matched-Param Plain Control | 375.16 KiB INT8",
        accent="#B45309", bg_tint="#FFFBEB",
    )

    # Panel headers
    fig.text(0.015, 0.975, "A. Reference Baselines: Lightweight Model A (163,940) vs. Official SLAMSS-IFS (5,465,768)",
             fontsize=10.5, fontweight="bold", color="#0F172A")
    fig.text(0.015, 0.485, "B. Last 3 Trained Models: Ultradian-Pyramidal-Coupler (full_v5) vs. Width-Scaled Plain CNN-BiGRUs (p1_100k, P190)",
             fontsize=10.5, fontweight="bold", color="#0F172A")

    out_pdf = OUT_DIR / "fig1_baseline_vs_proposed_architectures.pdf"
    out_png = OUT_DIR / "fig1_baseline_vs_proposed_architectures.png"
    fig.savefig(out_pdf, format="pdf", facecolor="white")
    fig.savefig(out_png, format="png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()
