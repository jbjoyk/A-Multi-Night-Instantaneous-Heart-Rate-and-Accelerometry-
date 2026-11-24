"""
EXP-031 Confirmatory Statistical Analysis, Section 9 Backbone Selection & CPU Profiling
File: experiments/EXP-031/analyze_exp031_confirmatory.py

Implements:
1. Confirmatory H-ARCH: full_v5 minus P190 on BIDSleep N=47 (10,000-resample participant
   bootstrap 95% CI with seed=42, two-sided Wilcoxon, Cohen's d_z, +0.75 pp threshold).
2. Confirmatory H-SCALE: Page's L across all 9 ordered plain widths (W050..W250),
   reporting one-sided & two-sided p-values, Holm-Bonferroni across {H-ARCH, H-SCALE},
   and directional verdict (SUPPORTED iff L > mu_L [z > 0] and two-sided Holm p < 0.05).
3. Section 9 Backbone Selection: strictly over the 5 < 400 KiB Dynamic INT8 eligible
   plain widths (W050, W075, A, P190, B; V5 excluded), 1-SE bootstrap rule (seed=42),
   writing experiments/EXP-031/selected_backbone.json.
4. Section 5 CPU Profiling: trainable params, FP32 KiB, Dynamic INT8 KiB, MMACs at L=1200,
   and 1T/4T CPU latency across all 9 plain widths.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.ao.quantization as quantization

EXP031_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP031_DIR, "..", ".."))
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
EXP021_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-021")
for _p in (REPO_ROOT, EXP031_DIR, EXP030_DIR, EXP029_DIR, EXP021_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_tsk08_paired_benchmark import participant_bootstrap_ci  # noqa: E402
from eval_exp029_downstream import holm_bonferroni  # noqa: E402
from eval_exp030_downstream import compute_model_mmacs  # noqa: E402
from exp031_models import (  # noqa: E402
    EXP031_ARMS,
    EXP031_ARMS_ORDER,
    EXP031_ELIGIBLE_INT8_ARMS,
    build_exp031_model,
)


def evaluate_h_arch(
    full_v5_scores: np.ndarray,
    p190_scores: np.ndarray,
    practical_threshold_pp: float = 0.75,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Confirmatory H-ARCH decision rule (Section 8):
      delta = (full_v5 - P190) * 100.0 (in percentage points, or if inputs are already
      in [0, 1], scaled by 100).
      - SUPPORTED: delta >= +0.75 pp and CI_low > 0
      - NOT SUPPORTED (practically): CI_high < +0.75 pp
      - INCONCLUSIVE: otherwise
    """
    full_v5_arr = np.asarray(full_v5_scores, dtype=np.float64)
    p190_arr = np.asarray(p190_scores, dtype=np.float64)
    assert full_v5_arr.shape == p190_arr.shape and len(full_v5_arr) > 0

    diff_pp = (full_v5_arr - p190_arr) * 100.0
    delta_pp = float(np.mean(diff_pp))
    ci_low, ci_high = participant_bootstrap_ci(diff_pp, n_boot=n_boot, alpha=0.05, seed=seed)
    _, wilcoxon_p = stats.wilcoxon(diff_pp) if np.any(diff_pp != 0) else (0.0, 1.0)
    sd = float(np.std(diff_pp, ddof=1)) if len(diff_pp) > 1 else 0.0
    cohens_dz = float(delta_pp / sd) if sd > 1e-12 else 0.0

    if delta_pp >= practical_threshold_pp and ci_low > 0.0:
        verdict = "SUPPORTED"
    elif ci_high < practical_threshold_pp:
        verdict = "NOT SUPPORTED (practically)"
    else:
        verdict = "INCONCLUSIVE"

    return {
        "n_subjects": int(len(diff_pp)),
        "full_v5_mean_pct": float(np.mean(full_v5_arr) * 100.0),
        "p190_mean_pct": float(np.mean(p190_arr) * 100.0),
        "delta_pp": delta_pp,
        "ci95_low_pp": float(ci_low),
        "ci95_high_pp": float(ci_high),
        "wilcoxon_p": float(wilcoxon_p),
        "cohens_dz": cohens_dz,
        "practical_threshold_pp": float(practical_threshold_pp),
        "wins_full_v5": int(np.sum(diff_pp > 0)),
        "losses_full_v5": int(np.sum(diff_pp < 0)),
        "ties": int(np.sum(diff_pp == 0)),
        "verdict": verdict,
    }


def compute_pages_l(scores_matrix: np.ndarray) -> Dict[str, Any]:
    """
    Page's L trend test across ordered columns (N subjects x k ordered widths).
    Ranks each subject's row from 1 (lowest) to k (highest), breaking ties by average.
    Returns L, mu_L, sigma_L, z (continuity-corrected), p_one_sided, p_two_sided.
    """
    mat = np.asarray(scores_matrix, dtype=np.float64)
    N, k = mat.shape
    ranks = np.apply_along_axis(lambda row: stats.rankdata(row, method="average"), axis=1, arr=mat)
    R = np.sum(ranks, axis=0)
    weights = np.arange(1, k + 1, dtype=np.float64)
    L = float(np.sum(weights * R))
    mu_L = float(N * k * ((k + 1) ** 2) / 4.0)
    var_L = float(N * (k ** 2) * (k + 1) * (k ** 2 - 1) / 144.0)
    sigma_L = float(np.sqrt(var_L))

    if L > mu_L:
        z = float((L - mu_L - 0.5) / sigma_L)
    elif L < mu_L:
        z = float((L - mu_L + 0.5) / sigma_L)
    else:
        z = 0.0

    p_one_sided = float(1.0 - stats.norm.cdf(z))
    p_two_sided = float(min(1.0, 2.0 * (1.0 - stats.norm.cdf(abs(z)))))

    return {
        "n_subjects": int(N),
        "k_widths": int(k),
        "L": L,
        "mu_L": mu_L,
        "sigma_L": sigma_L,
        "z": z,
        "direction_positive": bool(L > mu_L and z > 0.0),
        "p_one_sided": p_one_sided,
        "p_two_sided": p_two_sided,
        "rank_sums": [float(x) for x in R],
    }


def evaluate_confirmatory_tests(
    full_v5_scores: np.ndarray,
    width_scores_ordered: Dict[str, np.ndarray],
    ordered_arms: List[str] = EXP031_ARMS_ORDER,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Runs both confirmatory tests (H-ARCH and H-SCALE) and applies Holm-Bonferroni
    correction across the two confirmatory two-sided p-values.
    """
    h_arch = evaluate_h_arch(
        full_v5_scores=full_v5_scores,
        p190_scores=width_scores_ordered["P190"],
        practical_threshold_pp=0.75,
        n_boot=n_boot,
        seed=seed,
    )
    score_mat = np.column_stack([width_scores_ordered[arm] for arm in ordered_arms])
    h_scale = compute_pages_l(score_mat)

    raw_ps = [h_arch["wilcoxon_p"], h_scale["p_two_sided"]]
    adj_ps = holm_bonferroni(raw_ps)
    h_arch["holm_p"] = float(adj_ps[0])
    h_scale["holm_p_two_sided"] = float(adj_ps[1])
    h_scale["verdict"] = (
        "SUPPORTED"
        if (h_scale["L"] > h_scale["mu_L"] and h_scale["z"] > 0.0 and h_scale["holm_p_two_sided"] < 0.05)
        else "NOT SUPPORTED"
    )
    return {
        "h_arch": h_arch,
        "h_scale": h_scale,
    }


def select_backbone_section9(
    width_scores: Dict[str, np.ndarray],
    eligible_arms: List[str] = EXP031_ELIGIBLE_INT8_ARMS,
    arm_meta: Dict[str, Dict[str, Any]] = EXP031_ARMS,
    n_boot: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Section 9 Backbone Selection Rule (LOCKED):
    1. Filter strictly to the 5 < 400 KiB Dynamic INT8 eligible plain widths
       (W050, W075, A, P190, B). Non-plain V5 is excluded.
    2. Let m* be the eligible plain width with highest mean Macro-F1 across N=47
       subjects (ties broken toward fewer parameters).
    3. Select the smallest-parameter eligible plain width m satisfying:
       mean(F1_{m*}) - mean(F1_m) <= SE_boot(m),
       where SE_boot(m) = SD_boot(mean(F1_{m*} - F1_m)) over 10,000 participant-level
       bootstrap resamples (seed=42).
    """
    for arm in eligible_arms:
        if arm not in width_scores:
            raise KeyError(f"Missing scores for eligible arm '{arm}'")
        if not arm_meta[arm]["eligible_int8_400kib"]:
            raise ValueError(f"Arm '{arm}' is not < 400 KiB INT8 eligible")

    sorted_eligible = sorted(eligible_arms, key=lambda a: int(arm_meta[a]["expected_params"]))
    means_pct = {a: float(np.mean(width_scores[a]) * 100.0) for a in sorted_eligible}

    # m* maximizing mean Macro-F1, breaking ties toward fewer parameters
    m_star = min(
        sorted_eligible,
        key=lambda a: (-means_pct[a], int(arm_meta[a]["expected_params"])),
    )
    f1_star = np.asarray(width_scores[m_star], dtype=np.float64) * 100.0
    n = len(f1_star)

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_boot, n))

    candidates: List[Dict[str, Any]] = []
    selected_arm: Optional[str] = None

    for arm in sorted_eligible:
        f1_m = np.asarray(width_scores[arm], dtype=np.float64) * 100.0
        diff = f1_star - f1_m
        mean_diff = float(np.mean(diff))
        if arm == m_star:
            se_boot = 0.0
        else:
            boot_means = diff[boot_indices].mean(axis=1)
            se_boot = float(np.std(boot_means, ddof=1))
        qualifies = bool(mean_diff <= se_boot + 1e-12)
        if qualifies and selected_arm is None:
            selected_arm = arm
        candidates.append({
            "arm": arm,
            "n_params": int(arm_meta[arm]["expected_params"]),
            "int8_in_memory_kib": float(arm_meta[arm]["expected_int8_kib"]),
            "mean_macro_f1_pct": means_pct[arm],
            "gap_from_m_star_pp": mean_diff,
            "se_boot_pp": se_boot,
            "qualifies": qualifies,
        })

    assert selected_arm is not None
    return {
        "eligible_arms": sorted_eligible,
        "excluded_non_plain": ["V5"],
        "m_star": m_star,
        "m_star_mean_macro_f1_pct": means_pct[m_star],
        "selected_arm": selected_arm,
        "selected_params": int(arm_meta[selected_arm]["expected_params"]),
        "selected_mean_macro_f1_pct": means_pct[selected_arm],
        "n_boot": int(n_boot),
        "seed": int(seed),
        "candidates": candidates,
    }


def profile_single_arm_cpu(arm: str, benchmark_latency: bool = True) -> Dict[str, Any]:
    """
    Section 5 CPU profiling for a single plain arm:
    trainable params, FP32 KiB, Dynamic INT8 in-memory KiB, serialized KiBs,
    MMACs at L=1200, and 1-thread / 4-thread CPU latency.
    """
    model_fp32 = build_exp031_model(arm).cpu().eval()
    n_params = sum(p.numel() for p in model_fp32.parameters())
    quantizable_params = sum(
        sum(p.numel() for p in mod.parameters())
        for mod in model_fp32.modules()
        if isinstance(mod, (nn.Linear, nn.GRU))
    )
    fp32_only_params = n_params - quantizable_params
    fp32_weight_kib = (n_params * 4) / 1024.0
    int8_in_mem_kib = (quantizable_params * 1 + fp32_only_params * 4) / 1024.0
    mmacs = compute_model_mmacs(model_fp32, seq_len=1200, epoch_len=30)

    model_int8 = quantization.quantize_dynamic(
        copy.deepcopy(model_fp32), {nn.Linear, nn.GRU}, dtype=torch.qint8
    ).eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        fp32_p = os.path.join(tmpdir, "fp32.pt")
        int8_p = os.path.join(tmpdir, "int8.pt")
        torch.save(model_fp32.state_dict(), fp32_p)
        torch.save(model_int8.state_dict(), int8_p)
        fp32_disk_kib = os.path.getsize(fp32_p) / 1024.0
        int8_disk_kib = os.path.getsize(int8_p) / 1024.0

    res: Dict[str, Any] = {
        "arm": arm,
        "w": float(EXP031_ARMS[arm]["w"]),
        "d_enc": int(EXP031_ARMS[arm]["d_enc"]),
        "d_temp": int(EXP031_ARMS[arm]["d_temp"]),
        "head_hidden": int(EXP031_ARMS[arm]["head_hidden"]),
        "total_params": int(n_params),
        "quantizable_params": int(quantizable_params),
        "fp32_only_params": int(fp32_only_params),
        "fp32_weight_kib": float(fp32_weight_kib),
        "int8_in_memory_kib": float(int8_in_mem_kib),
        "fp32_serialized_kib": float(fp32_disk_kib),
        "int8_serialized_kib": float(int8_disk_kib),
        "mmacs_L1200": float(mmacs),
        "eligible_int8_400kib": bool(int8_in_mem_kib < 400.0),
    }

    if benchmark_latency:
        x_ihr = torch.randn(1, 1200, 30)
        x_acc = torch.randn(1, 1200, 30)
        x_meta = torch.randn(1, 1200, 5)

        def bench(m: nn.Module, n_threads: int, n_runs: int = 10) -> float:
            prev = torch.get_num_threads()
            torch.set_num_threads(n_threads)
            try:
                with torch.no_grad():
                    for _ in range(3):
                        m(x_ihr, x_acc, x_meta)
                    t0 = time.perf_counter()
                    for _ in range(n_runs):
                        m(x_ihr, x_acc, x_meta)
                    t1 = time.perf_counter()
                return float(((t1 - t0) / n_runs) * 1000.0)
            finally:
                torch.set_num_threads(prev)

        res["cpu_latency_1t_fp32_ms"] = bench(model_fp32, 1)
        res["cpu_latency_4t_fp32_ms"] = bench(model_fp32, 4)
        res["cpu_latency_1t_int8_ms"] = bench(model_int8, 1)
        res["cpu_latency_4t_int8_ms"] = bench(model_int8, 4)

    return res


def profile_all_9_widths(output_dir: str = EXP031_DIR, benchmark_latency: bool = True) -> List[Dict[str, Any]]:
    profiles = [profile_single_arm_cpu(arm, benchmark_latency=benchmark_latency) for arm in EXP031_ARMS_ORDER]
    out_path = os.path.join(output_dir, "exp031_cpu_profile.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"profiles": profiles}, f, indent=2)
    return profiles


def _load_arm_subject_df(arm: str, exp031_dir: str) -> pd.DataFrame:
    reused = EXP031_ARMS[arm]["reused_oof_csv"]
    path = reused if reused is not None else os.path.join(exp031_dir, f"{arm}_subject_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing subject metrics CSV for arm '{arm}': {path}")
    df = pd.read_csv(path).sort_values("subject_id").reset_index(drop=True)
    assert len(df) == 47, f"Expected 47 subjects for {arm}, got {len(df)}"
    return df


def run_full_confirmatory_analysis(exp031_dir: str = EXP031_DIR, profile_cpu: bool = True) -> Dict[str, Any]:
    # Load all 9 plain widths aligned by subject_id
    dfs: Dict[str, pd.DataFrame] = {arm: _load_arm_subject_df(arm, exp031_dir) for arm in EXP031_ARMS_ORDER}
    base_subs = dfs["A"]["subject_id"].tolist()
    for arm, df in dfs.items():
        assert df["subject_id"].tolist() == base_subs, f"Subject ID mismatch in {arm}"

    width_scores = {arm: dfs[arm]["macro_f1"].values for arm in EXP031_ARMS_ORDER}

    # Load full_v5 and SLAMSS-IFS
    v5_csv = os.path.join(EXP029_DIR, "full_v5_subject_metrics.csv")
    df_v5 = pd.read_csv(v5_csv).sort_values("subject_id").reset_index(drop=True)
    assert df_v5["subject_id"].tolist() == base_subs

    slamss_csv = os.path.join(EXP021_DIR, "slamss_ifs_matched_subject_metrics.csv")
    df_s_raw = pd.read_csv(slamss_csv)
    df_s = df_s_raw.groupby("subject_id", as_index=False)["macro_f1"].mean().sort_values("subject_id").reset_index(drop=True)
    assert df_s["subject_id"].tolist() == base_subs

    confirmatory = evaluate_confirmatory_tests(
        full_v5_scores=df_v5["macro_f1"].values,
        width_scores_ordered=width_scores,
        ordered_arms=EXP031_ARMS_ORDER,
        n_boot=10000,
        seed=42,
    )
    selection = select_backbone_section9(
        width_scores=width_scores,
        eligible_arms=EXP031_ELIGIBLE_INT8_ARMS,
        arm_meta=EXP031_ARMS,
        n_boot=10000,
        seed=42,
    )
    with open(os.path.join(exp031_dir, "selected_backbone.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    # Descriptive width table vs A and vs SLAMSS-IFS
    cpu_profiles = (
        profile_all_9_widths(exp031_dir, benchmark_latency=True)
        if profile_cpu
        else [profile_single_arm_cpu(arm, benchmark_latency=False) for arm in EXP031_ARMS_ORDER]
    )
    prof_by_arm = {p["arm"]: p for p in cpu_profiles}

    a_scores = width_scores["A"]
    s_scores = df_s["macro_f1"].values
    width_rows: List[Dict[str, Any]] = []
    for arm in EXP031_ARMS_ORDER:
        sc = width_scores[arm]
        d_a = (sc - a_scores) * 100.0
        ci_a_lo, ci_a_hi = participant_bootstrap_ci(d_a, n_boot=10000, alpha=0.05, seed=42)
        _, p_a = stats.wilcoxon(d_a) if np.any(d_a != 0) else (0.0, 1.0)

        d_s = (sc - s_scores) * 100.0
        ci_s_lo, ci_s_hi = participant_bootstrap_ci(d_s, n_boot=10000, alpha=0.05, seed=42)
        _, p_s = stats.wilcoxon(d_s) if np.any(d_s != 0) else (0.0, 1.0)

        pinfo = prof_by_arm[arm]
        width_rows.append({
            "arm": arm,
            "w": pinfo["w"],
            "n_params": pinfo["total_params"],
            "fp32_weight_kib": pinfo["fp32_weight_kib"],
            "int8_in_memory_kib": pinfo["int8_in_memory_kib"],
            "mmacs_L1200": pinfo["mmacs_L1200"],
            "cpu_latency_1t_int8_ms": pinfo.get("cpu_latency_1t_int8_ms"),
            "cpu_latency_4t_int8_ms": pinfo.get("cpu_latency_4t_int8_ms"),
            "eligible_int8_400kib": pinfo["eligible_int8_400kib"],
            "macro_f1_mean_pct": float(np.mean(sc) * 100.0),
            "macro_f1_std_pct": float(np.std(sc, ddof=1) * 100.0),
            "delta_vs_A_pp": float(np.mean(d_a)),
            "ci_vs_A_low_pp": float(ci_a_lo),
            "ci_vs_A_high_pp": float(ci_a_hi),
            "wilcoxon_p_vs_A": float(p_a),
            "delta_vs_SLAMSS_pp": float(np.mean(d_s)),
            "ci_vs_SLAMSS_low_pp": float(ci_s_lo),
            "ci_vs_SLAMSS_high_pp": float(ci_s_hi),
            "wilcoxon_p_vs_SLAMSS": float(p_s),
        })

    pd.DataFrame(width_rows).to_csv(os.path.join(exp031_dir, "exp031_width_table.csv"), index=False)

    out = {
        "confirmatory": confirmatory,
        "section9_selection": selection,
        "width_table": width_rows,
    }
    with open(os.path.join(exp031_dir, "exp031_confirmatory_summary.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-031 Confirmatory Analysis & Section 9 Selection")
    parser.add_argument("--exp031-dir", type=str, default=EXP031_DIR)
    parser.add_argument("--profile-only", action="store_true", help="Run Section 5 CPU profiling only (pre-training)")
    parser.add_argument("--skip-cpu-bench", action="store_true")
    args = parser.parse_args()

    if args.profile_only:
        profiles = profile_all_9_widths(args.exp031_dir, benchmark_latency=(not args.skip_cpu_bench))
        print(json.dumps({"profiles": profiles}, indent=2))
        return

    res = run_full_confirmatory_analysis(args.exp031_dir, profile_cpu=(not args.skip_cpu_bench))
    print(json.dumps(res["confirmatory"], indent=2))
    print(json.dumps(res["section9_selection"], indent=2))


if __name__ == "__main__":
    main()
