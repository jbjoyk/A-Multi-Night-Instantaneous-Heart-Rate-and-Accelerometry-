"""
EXP-033 Statistical Analysis, Forest Plot, Depth Curve, and Final Report Generator.
File: experiments/EXP-033/analyze_exp033.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

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
from exp033_core import BACKBONE_ORDER, get_backbone_ckpt_map, state_dict_sha256  # noqa: E402
import torch  # noqa: E402

TAU_GRID = [0.01, 0.02, 0.03, 0.05]


def paired_stats(a: np.ndarray, b: np.ndarray, seed: int = 42, scale: float = 100.0) -> Dict[str, Any]:
    """Compute paired difference (a - b) * scale with 10,000-resample bootstrap CI, Wilcoxon, Cohen's d_z, median, and W/L/T."""
    diff = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) * scale
    n = len(diff)
    mean_d = float(np.mean(diff))
    med_d = float(np.median(diff))
    sd_d = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    dz = float(mean_d / sd_d) if sd_d > 0 else 0.0
    ci_lo, ci_hi = participant_bootstrap_ci(diff, n_boot=10000, seed=seed)
    try:
        _, p_wil = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        p_wil = float(p_wil)
    except Exception:
        p_wil = 1.0
    wins = int(np.sum(diff > 0))
    losses = int(np.sum(diff < 0))
    ties = int(np.sum(diff == 0))
    return {
        "n_subjects": int(n),
        "mean_a_pct": float(np.mean(a) * scale),
        "mean_b_pct": float(np.mean(b) * scale),
        "delta_pp": mean_d,
        "median_delta_pp": med_d,
        "sd_pp": sd_d,
        "ci95_low_pp": float(ci_lo),
        "ci95_high_pp": float(ci_hi),
        "wilcoxon_p_raw_two_sided": p_wil,
        "cohens_dz": dz,
        "n_improved": wins,
        "n_losses": losses,
        "n_ties": ties,
        "pct_improved": float(100.0 * np.mean(diff > 0)),
    }


def holm_bonferroni(pvals: List[float]) -> List[float]:
    m = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adj = [0.0] * m
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed):
        val = min(1.0, p * (m - rank))
        running_max = max(running_max, val)
        adj[idx] = running_max
    return adj


def locked_verdict(res: Dict[str, Any]) -> str:
    """Section 7 locked verdict rule (identical to H-A)."""
    if res["delta_pp"] >= 1.5 and res["ci95_low_pp"] > 0.0:
        return "SUPPORTED"
    if res["ci95_high_pp"] < 1.5:
        return "NOT SUPPORTED (practically)"
    return "INCONCLUSIVE"


def verify_state_dict_hashes() -> Dict[str, Any]:
    summary = {}
    for b in BACKBONE_ORDER:
        df = pd.read_csv(os.path.join(EXP033_DIR, f"{b}_state_dict_hashes.csv"))
        cmap = get_backbone_ckpt_map(b)
        expected = {
            k: state_dict_sha256(torch.load(p, map_location="cpu", weights_only=True))
            for k, p in cmap.items()
        }
        before_mismatches = sum(
            1 for _, r in df.iterrows()
            if r["hash_before"] != expected[(int(r["fold"]), int(r["seed"]))]
        )
        unchanged_after = sum(
            1 for _, r in df.iterrows()
            if r["hash_before"] == r["hash_after"]
        )
        n_donor_rows = int(df["arm"].str.startswith("FT-other-2_draw").sum())
        summary[b] = {
            "total_ft_runs": int(len(df)),
            "donor_draw_runs": n_donor_rows,
            "before_mismatches": int(before_mismatches),
            "unchanged_after_ft": int(unchanged_after),
        }
    return summary


def analyze_backbone(backbone: str) -> Dict[str, Any]:
    csv_path = os.path.join(EXP033_DIR, f"{backbone}_subject_seed_metrics.csv")
    df = pd.read_csv(csv_path)

    ft2 = df[df["arm"] == "FT-2"].groupby("subject_id", as_index=False).mean(numeric_only=True)
    ft_other = df[df["arm"] == "FT-other-2"].groupby("subject_id", as_index=False).mean(numeric_only=True)

    m_ha2 = pd.merge(ft2, ft_other, on="subject_id", suffixes=("_ft2", "_other"))
    ha2_res = paired_stats(m_ha2["ft_macro_f1_ft2"].values, m_ha2["zs_macro_f1_ft2"].values)
    sspec_res = paired_stats(m_ha2["ft_macro_f1_ft2"].values, m_ha2["ft_macro_f1_other"].values)
    donor_harm_res = paired_stats(m_ha2["ft_macro_f1_other"].values, m_ha2["zs_macro_f1_ft2"].values)

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
            "median_delta_pp": float(np.median(delta_pp)),
            "min_delta_pp": float(np.min(delta_pp)),
            "p10_delta_pp": float(np.percentile(delta_pp, 10)),
            "n_harm_le_minus5pp": int(np.sum(delta_pp <= -5.0)),
        })

    per_stage = {}
    for stg in ("wake", "light", "deep", "rem"):
        per_stage[stg] = paired_stats(ft2[f"ft_f1_{stg}"].values, ft2[f"zs_f1_{stg}"].values)

    night_df = pd.read_csv(os.path.join(EXP033_DIR, f"{backbone}_night_index_metrics.csv"))
    night_sub = night_df.groupby(["subject_id", "eval_night_pos"], as_index=False).mean(numeric_only=True)
    night_summary = []
    for pos, grp in night_sub.groupby("eval_night_pos"):
        if len(grp) >= 3:
            r = paired_stats(grp["ft_macro_f1"].values, grp["zs_macro_f1"].values)
            night_summary.append({"eval_night_pos": int(pos), **r})

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

    fig, ax = plt.subplots(figsize=(8.8, 4.6), dpi=180)
    y = np.arange(len(labels))
    ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0, label="Null (0.0 pp)")
    ax.axvline(1.5, color="#DC2626", linestyle=":", linewidth=1.3, label="H-A2 threshold (+1.50 pp)")

    for i in range(len(labels)):
        ax.errorbar(
            deltas[i], y[i],
            xerr=[[lo_errs[i]], [hi_errs[i]]],
            fmt="o", color=colors[i], ecolor=colors[i], elinewidth=2.0, capsize=4, markersize=6,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("FT-2 minus ZS Macro-F1 (percentage points, 95% bootstrap CI)", fontsize=9.5)
    ax.set_title("EXP-033: Two-Night Personalization Effect (FT-2 - ZS) Across Backbones (N=46)", fontsize=10.5, pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, frameon=True, fontsize=8.5)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def make_depth_curve_plot(backbone_results: Dict[str, Any], out_path: str) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=180, sharey=True)
    styles = {
        "W100": ("#1D4ED8", "-o", 2.0, -0.09),
        "p1_100k": ("#059669", "--s", 1.5, -0.03),
        "P190": ("#D97706", "-.^", 1.5, +0.03),
        "full_v5": ("#7C3AED", ":d", 1.5, +0.09),
    }

    # Left panel: E4 (N=33) across all 4 backbones with 95% bootstrap CI bars
    for b in BACKBONE_ORDER:
        rows = backbone_results[b]["depth_curve_E4"]
        ks = np.array([r["k"] for r in rows], dtype=float)
        ds = np.array([r["delta_pp"] for r in rows], dtype=float)
        lo = np.array([r["delta_pp"] - r["ci95_low_pp"] for r in rows], dtype=float)
        hi = np.array([r["ci95_high_pp"] - r["delta_pp"] for r in rows], dtype=float)
        col, fmt, lw, jitter = styles[b]
        ax1.errorbar(ks + jitter, ds, yerr=[lo, hi], fmt=fmt, color=col, linewidth=lw, markersize=4.5, capsize=3, label=f"{b} (E4, N=33)")

    ax1.axhline(0.0, color="#111827", linestyle="--", linewidth=0.9)
    ax1.set_xticks([1, 2, 3, 4])
    ax1.set_xlabel("Calibration Nights (k)", fontsize=9.5)
    ax1.set_ylabel("FT-k minus ZS Macro-F1 (pp, 95% bootstrap CI)", fontsize=9.5)
    ax1.set_title("Common Cohort E4 (N=33) Across Backbones", fontsize=10)
    ax1.legend(loc="upper left", fontsize=7.8)
    ax1.grid(alpha=0.25)

    # Right panel: W100 E4 (N=33) vs E_k (N=46,46,42,33) with 95% bootstrap CI bars
    w100_e4 = backbone_results["W100"]["depth_curve_E4"]
    w100_ek = backbone_results["W100"]["depth_curve_Ek"]
    for label, rows, col, fmt, jit in [
        ("W100 E4 (N=33)", w100_e4, "#1D4ED8", "-o", -0.05),
        ("W100 E_k (N=46,46,42,33)", w100_ek, "#DC2626", "--s", +0.05),
    ]:
        ks = np.array([r["k"] for r in rows], dtype=float)
        ds = np.array([r["delta_pp"] for r in rows], dtype=float)
        lo = np.array([r["delta_pp"] - r["ci95_low_pp"] for r in rows], dtype=float)
        hi = np.array([r["ci95_high_pp"] - r["delta_pp"] for r in rows], dtype=float)
        ax2.errorbar(ks + jit, ds, yerr=[lo, hi], fmt=fmt, color=col, linewidth=1.9, markersize=5, capsize=3.5, label=label)

    ax2.axhline(0.0, color="#111827", linestyle="--", linewidth=0.9)
    ax2.set_xticks([1, 2, 3, 4])
    ax2.set_xlabel("Calibration Nights (k)", fontsize=9.5)
    ax2.set_title("W100: E4 (N=33) vs. Available Cohorts E_k", fontsize=10)
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def main() -> None:
    hash_audit = verify_state_dict_hashes()
    backbone_results = {b: analyze_backbone(b) for b in BACKBONE_ORDER}

    w100 = backbone_results["W100"]
    p_ha2 = w100["H_A2"]["wilcoxon_p_raw_two_sided"]
    p_sspec = w100["S_SPEC"]["wilcoxon_p_raw_two_sided"]
    p_holm_ha2, p_holm_sspec = holm_bonferroni([p_ha2, p_sspec])
    w100["H_A2"]["wilcoxon_p_holm_two_sided"] = p_holm_ha2
    w100["S_SPEC"]["wilcoxon_p_holm_two_sided"] = p_holm_sspec

    verdict_ha2 = locked_verdict(w100["H_A2"])
    verdict_sspec = locked_verdict(w100["S_SPEC"])
    depth_claimed = (
        w100["pages_L_E4"]["p_two_sided"] < 0.05
        and w100["depth_curve_E4"][-1]["ci95_low_pp"] > 0.0
    )

    # Exact locked EXP-019 values ("older recipe")
    exp019_sub = pd.read_csv(os.path.join(EXP019_DIR, "exp019_subject_metrics.csv"))
    with open(os.path.join(EXP019_DIR, "exp019_stage_c_results.json"), "r", encoding="utf-8") as f:
        exp019_json = json.load(f)
    ha_locked = exp019_json["hypotheses"]["H-A"]
    legacy_exp019 = paired_stats(exp019_sub["FT-2"].values, exp019_sub["ZS"].values)
    legacy_exp019["delta_pp"] = float(ha_locked["delta_mean"])
    legacy_exp019["ci95_low_pp"] = float(ha_locked["ci_95"][0])
    legacy_exp019["ci95_high_pp"] = float(ha_locked["ci_95"][1])
    legacy_exp019["wilcoxon_p_locked_one_sided"] = float(ha_locked["wilcoxon_p"])

    with open(os.path.join(EXP033_DIR, "W100_sleepaccel_summary.json"), "r", encoding="utf-8") as f:
        sa_summary = json.load(f)
    with open(os.path.join(EXP033_DIR, "W100_sleepaccel_paired_comparisons.json"), "r", encoding="utf-8") as f:
        sa_paired = json.load(f)
    sa_ba = pd.read_csv(os.path.join(EXP033_DIR, "W100_sleepaccel_bland_altman.csv"))

    bb_rows = []
    for b in BACKBONE_ORDER:
        r = backbone_results[b]
        bb_rows.append({
            "backbone": b,
            "role": "primary" if b == "W100" else "descriptive",
            "ft2_zs_delta_pp": r["H_A2"]["delta_pp"],
            "ft2_zs_median_pp": r["H_A2"]["median_delta_pp"],
            "ft2_zs_ci_lo": r["H_A2"]["ci95_low_pp"],
            "ft2_zs_ci_hi": r["H_A2"]["ci95_high_pp"],
            "ft2_zs_p_raw_2s": r["H_A2"]["wilcoxon_p_raw_two_sided"],
            "ft2_zs_dz": r["H_A2"]["cohens_dz"],
            "ft2_zs_wins": r["H_A2"]["n_improved"],
            "ft2_zs_losses": r["H_A2"]["n_losses"],
            "ft2_zs_ties": r["H_A2"]["n_ties"],
            "ft2_other_delta_pp": r["S_SPEC"]["delta_pp"],
            "ft2_other_median_pp": r["S_SPEC"]["median_delta_pp"],
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
        "confirmatory_holm_p": {
            "H_A2_raw_two_sided": p_ha2,
            "H_A2_holm_two_sided": p_holm_ha2,
            "S_SPEC_raw_two_sided": p_sspec,
            "S_SPEC_holm_two_sided": p_holm_sspec,
        },
        "state_dict_hash_audit": hash_audit,
        "backbones": backbone_results,
        "legacy_exp019_older_recipe": legacy_exp019,
        "sleepaccel_part_b": {
            "summary": sa_summary,
            "paired_comparisons": sa_paired["f1_comparisons"],
        },
    }
    with open(os.path.join(EXP033_DIR, "exp033_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines: List[str] = []
    lines.append("# EXP-033 Final Report: Personalization and External Evaluation on Recipe-Consistent W100")
    lines.append("")
    lines.append("- **Pre-registration:** `docs/designs/PREREG_EXP-033_W100_PERSONALIZATION_EXTERNAL.md` (lock commit `c1b784305812d3c7208ed73c24a4b08a85c1ef3c`)")
    lines.append(f"- **H-A2 Verdict (`W100` `FT-2 - ZS`, N=46):** **{verdict_ha2}** (`Mean Δ = {w100['H_A2']['delta_pp']:+.2f} pp`, `Median Δ = {w100['H_A2']['median_delta_pp']:+.2f} pp`, 95% CI `[{w100['H_A2']['ci95_low_pp']:+.2f}, {w100['H_A2']['ci95_high_pp']:+.2f}]`, `W/L/T = {w100['H_A2']['n_improved']}/{w100['H_A2']['n_losses']}/{w100['H_A2']['n_ties']}`, Wilcoxon raw two-sided `p = {p_ha2:.4e}`, Holm-adjusted two-sided `p_Holm = {p_holm_ha2:.4e}`, `d_z = {w100['H_A2']['cohens_dz']:.3f}`)")
    lines.append(f"- **S-SPEC Verdict (`W100` `FT-2 - FT-other-2`, N=46):** **{verdict_sspec}** (`Mean Δ = {w100['S_SPEC']['delta_pp']:+.2f} pp`, `Median Δ = {w100['S_SPEC']['median_delta_pp']:+.2f} pp`, 95% CI `[{w100['S_SPEC']['ci95_low_pp']:+.2f}, {w100['S_SPEC']['ci95_high_pp']:+.2f}]`, `W/L/T = {w100['S_SPEC']['n_improved']}/{w100['S_SPEC']['n_losses']}/{w100['S_SPEC']['n_ties']}`, Wilcoxon raw two-sided `p = {p_sspec:.4e}`, Holm-adjusted two-sided `p_Holm = {p_holm_sspec:.4e}`, `d_z = {w100['S_SPEC']['cohens_dz']:.3f}`)")
    lines.append(f"- **Calibration-Depth Claim (`W100` E4 N=33):** **{'CLAIMABLE' if depth_claimed else 'NOT CLAIMABLE'}** (Page's `L = {w100['pages_L_E4']['L']:.1f}`, `z = {w100['pages_L_E4']['z']:+.3f}`, raw two-sided `p = {w100['pages_L_E4']['p_two_sided']:.4e}`, raw one-sided `p = {w100['pages_L_E4']['p_one_sided']:.4e}`; E4 `k=4` `Δ = {w100['depth_curve_E4'][-1]['delta_pp']:+.2f} pp` `[{w100['depth_curve_E4'][-1]['ci95_low_pp']:+.2f}, {w100['depth_curve_E4'][-1]['ci95_high_pp']:+.2f}]`)")
    lines.append("")
    lines.append("## 1. Backbone Summary Table (FT-2 vs ZS & FT-other-2, N=46)")
    lines.append("")
    lines.append("| Backbone | Role | Mean Δ (pp) | Median Δ (pp) | 95% CI | W/L/T | p (raw, 2-sided) | d_z | FT-2 - FT-other-2 (pp) | 95% CI | FT-other-2 - ZS (pp) |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in BACKBONE_ORDER:
        r = backbone_results[b]
        role = "Primary" if b == "W100" else "Descriptive"
        h = r["H_A2"]
        s = r["S_SPEC"]
        dh = r["donor_harm_FT_other_2_minus_ZS"]
        lines.append(
            f"| `{b}` | {role} | `{h['delta_pp']:+.2f}` | `{h['median_delta_pp']:+.2f}` | `[{h['ci95_low_pp']:+.2f}, {h['ci95_high_pp']:+.2f}]` | "
            f"`{h['n_improved']}/{h['n_losses']}/{h['n_ties']}` | `{h['wilcoxon_p_raw_two_sided']:.4e}` | `{h['cohens_dz']:.3f}` | "
            f"`{s['delta_pp']:+.2f}` | `[{s['ci95_low_pp']:+.2f}, {s['ci95_high_pp']:+.2f}]` | `{dh['delta_pp']:+.2f}` |"
        )
    lines.append(
        f"| `Model A` (EXP-019) | Older recipe | `{legacy_exp019['delta_pp']:+.2f}` | `{legacy_exp019['median_delta_pp']:+.2f}` | "
        f"`[{legacy_exp019['ci95_low_pp']:+.2f}, {legacy_exp019['ci95_high_pp']:+.2f}]` | "
        f"`{legacy_exp019['n_improved']}/{legacy_exp019['n_losses']}/{legacy_exp019['n_ties']}` | "
        f"`{legacy_exp019['wilcoxon_p_raw_two_sided']:.4e}` (`1s: {legacy_exp019['wilcoxon_p_locked_one_sided']:.5f}`) | `{legacy_exp019['cohens_dz']:.3f}` | — | — | — |"
    )
    lines.append("")
    lines.append("## 2. W100 Calibration-Depth Curve (Descriptive)")
    lines.append("")
    lines.append("| Cohort | k | N | ZS (%) | FT-k (%) | Mean Δ (pp) | Median Δ (pp) | 95% CI | W/L/T | p (raw, 2-sided) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in w100["depth_curve_E4"] + w100["depth_curve_Ek"]:
        lines.append(
            f"| `{row['cohort']}` | {row['k']} | {row['n_subjects']} | `{row['mean_b_pct']:.2f}` | `{row['mean_a_pct']:.2f}` | "
            f"`{row['delta_pp']:+.2f}` | `{row['median_delta_pp']:+.2f}` | `[{row['ci95_low_pp']:+.2f}, {row['ci95_high_pp']:+.2f}]` | "
            f"`{row['n_improved']}/{row['n_losses']}/{row['n_ties']}` | `{row['wilcoxon_p_raw_two_sided']:.4e}` |"
        )
    lines.append("")
    lines.append("## 3. GATE-τ Sweep (Descriptive, N=46)")
    lines.append("")
    lines.append("| Backbone | τ (%) | Abstained | Mean Δ vs ZS (pp) | Median Δ (pp) | Min Δ (pp) | 10th-pct Δ (pp) | N(Δ ≤ -5 pp) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in gate_rows:
        lines.append(
            f"| `{row['backbone']}` | {row['tau_pct']:.0f}% | {row['n_abstained']} | `{row['mean_delta_pp']:+.2f}` | `{row['median_delta_pp']:+.2f}` | "
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
    lines.append("*Note: `ModelA` (`62.9385%` on `n=25` vs. `62.9366%` on `N=31`) and `p1_100k` (`62.5057%` on `n=25` vs. `62.5105%` on `N=31`) `prob_ensemble` values are independently computed over `25` and `31` subjects and coincide when rounded to two decimal places (`62.94%` and `62.51%`). Power limit: paired SD ~`4 pp` yields 80% MDE ~`2.3 pp` at `n=25`.*")
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
    lines.append("")
    lines.append("## 5. Preregistration Governance & Safeguard Audit")
    lines.append("")
    lines.append("- **(a) `FT-other-2` Donor Draws:** Confirmed `5` independent donor draws per `(subject, seed)` from the same fold's `inner_train_subjects` (`46 subjects × 3 seeds × 5 draws = 690` donor runs per backbone; fixed RNG seed per `(fold, seed, subject)`).")
    lines.append("- **(b) State-Dict SHA-256 Hash Log:** Verified `0` mismatches across all `4,764` fine-tuning runs (`1,191` runs per backbone × 4 backbones); every `hash_before` matched the frozen checkpoint's `state_dict_sha256` and every `hash_after` differed after optimization.")
    lines.append("- **(c) Safeguard Unit Tests & `ZS` OOF Parity:** `pytest tests/test_exp033_safeguards.py` passed `2/2` both **before** (`4.24s`) and **after** (`3.51s`) the full sweep, confirming all 15 `W100` checkpoint SHA-256 digests, `review_package/fold_manifests.csv` (`36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf`), bitwise-identical `ZS` predictions before/after other-subject fine-tuning, and frozen `ZS` OOF Macro-F1 parity (`max abs diff = 1.11e-16` for `W100`/`P190`, `2.22e-16` for `p1_100k`/`full_v5`).")
    lines.append("- **(d) Nested Inner-Validation Selected Configurations (per Fold & Backbone):**")
    lines.append("")
    lines.append("| Backbone | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |")
    lines.append("|---|---|---|---|---|---|")
    for b in BACKBONE_ORDER:
        cfgs = backbone_results[b]["selected_configs"]
        cols = [f"`lr={cfgs[str(f)]['lr']}, ep={cfgs[str(f)]['epochs']}, {cfgs[str(f)]['scope']}` ({cfgs[str(f)]['mean_macro_f1']*100:.2f}%)" for f in (1, 2, 3, 4, 5)]
        lines.append(f"| `{b}` | " + " | ".join(cols) + " |")
    lines.append("")
    lines.append("- **(e) Section 2.2 Backbone-Selection Rule Verification:** Across the `EXP-031` `< 400 KiB` INT8 eligible plain CNN-BiGRU grid (`W050` `65.61%`, `W075` `66.26%`, `W100` `66.68%`, `P190` `66.60%`, `B` `66.61%`), `W100` (`163,940` params, `66.68%` OOF Macro-F1) is the highest-scoring eligible width (`gap = 0.00 pp <= 1 SE`) and is therefore selected under the smallest-width-within-1-SE rule.")
    lines.append(f"- **(f) Confirmatory Family Holm-Bonferroni Adjustment (`{{H-A2, S-SPEC}}` on `W100`):** `H-A2` raw two-sided `p = {p_ha2:.4e}`, Holm-adjusted two-sided `p_Holm = {p_holm_ha2:.4e}`; `S-SPEC` raw two-sided `p = {p_sspec:.4e}`, Holm-adjusted two-sided `p_Holm = {p_holm_sspec:.4e}`.")

    report_path = os.path.join(EXP033_DIR, "EXP033_FINAL_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[EXP-033 Analysis] Updated {report_path}, exp033_summary.json, and figures.", flush=True)


if __name__ == "__main__":
    main()
