"""
reconcile_tsk06_clean_zs.py: CPU-only forensic reconciliation for DEC-039 (TSK-06).

Recomputes strictly uncontaminated Zero-Shot (ZS) Macro-F1 baselines on CPU from the
15 frozen Model A checkpoints (5 folds x 3 seeds, zero retraining, zero weight mutation),
joins them with the already-computed clean Fine-Tuned (FT) metrics in
`exp019_k_curve_subject_metrics.csv`, and computes:
  - 10,000-resample participant-level percentile bootstrap 95% CIs (seed=42)
  - Exact two-sided Wilcoxon signed-rank p-values (method='exact')
  - Page's L-trend test across k in {1, 2, 3, 4} on E4 (N=33)
  - Updated 5-test Holm-Bonferroni adjusted p-values

Saves clean reconciled summary to:
  experiments/EXP-019/exp019_k_learning_curve_CLEAN_RECONCILED.csv
Does NOT overwrite any historical executed CSVs.
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import scipy.stats as stats
import torch
from sklearn.metrics import f1_score, accuracy_score
from typing import Dict, List, Tuple

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
exp018_dir = os.path.abspath(os.path.join(repo_root, "experiments", "EXP-018"))
for p in [repo_root, exp018_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from exp013_models import ConventionalCNNBiGRU
from exp018_inference import SlidingWindowPredictor
from slamss_ifs_train import prepare_dataset_records

CACHE_DIR = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
MANIFESTS_DIR = os.path.join(repo_root, "experiments", "EXP-014", "EXP014_MANIFESTS")
CHECKPOINTS_DIR = os.path.join(repo_root, "experiments", "EXP-018", "checkpoints")
SUBJECT_METRICS_CSV = os.path.join(repo_root, "experiments", "EXP-019", "exp019_k_curve_subject_metrics.csv")
LOCKED_EXP019_CSV = os.path.join(repo_root, "experiments", "EXP-019", "exp019_subject_metrics.csv")
OUTPUT_RECONCILED_CSV = os.path.join(repo_root, "experiments", "EXP-019", "exp019_k_learning_curve_CLEAN_RECONCILED.csv")

SEEDS = [42, 123, 2024]
K_VALUES = [1, 2, 3, 4]


def pages_l_test(ranks_matrix: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Identical to run_tsk06_k_curve.pages_l_test."""
    N, k = ranks_matrix.shape
    R = np.sum(ranks_matrix, axis=0)
    weights = np.arange(1, k + 1)
    L = float(np.sum(weights * R))
    mean_L = N * k * ((k + 1) ** 2) / 4.0
    var_L = N * (k ** 2) * (k + 1) * (k ** 2 - 1) / 144.0
    std_L = np.sqrt(var_L)
    if L > mean_L:
        z = (L - mean_L - 0.5) / std_L
    else:
        z = (L - mean_L + 0.5) / std_L
    p_value = float(1.0 - stats.norm.cdf(z))
    return L, float(z), p_value, mean_L, std_L


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Identical to run_tsk06_k_curve.bootstrap_ci."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def holm_bonferroni_adjust(p_values: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    """
    Computes step-down Holm-Bonferroni adjusted p-values and rejection decisions.
    p_adj[(j)] = min(1.0, max_{i <= j} (m - i) * p[(i)]) for 0-indexed sorted order.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    adj_sorted = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank_idx, orig_idx in enumerate(order):
        val = (m - rank_idx) * p_values[orig_idx]
        running_max = max(running_max, val)
        adj_sorted[rank_idx] = min(1.0, running_max)
    adj = [0.0] * m
    for rank_idx, orig_idx in enumerate(order):
        adj[orig_idx] = float(adj_sorted[rank_idx])
    rejected = [p <= alpha for p in adj]
    return adj, rejected


def pooled_macro_f1(night_preds: List[Tuple[np.ndarray, np.ndarray]]) -> float:
    all_preds = np.concatenate([p for p, _ in night_preds])
    all_gts = np.concatenate([g for _, g in night_preds])
    valid = (all_gts >= 0) & (all_gts < 4)
    if not np.any(valid):
        return 0.0
    return float(f1_score(all_gts[valid], all_preds[valid], average="macro"))


def compute_clean_zs_cpu() -> pd.DataFrame:
    t0 = time.time()
    torch.set_num_threads(4)
    device = torch.device("cpu")

    # 1. Load fold manifests
    sub_to_fold: Dict[str, int] = {}
    for fold in range(1, 6):
        with open(os.path.join(MANIFESTS_DIR, f"fold_{fold}.json"), "r") as f:
            m = json.load(f)
        for s in m.get("test", m.get("outer_test_subjects", [])):
            sub_to_fold[s] = fold

    # 2. Load preprocessed records
    split_dict = {"train": [], "val": [], "test": list(sub_to_fold.keys())}
    records_by_split = prepare_dataset_records(CACHE_DIR, split_dict)
    records_by_sub: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for r in records_by_split["test"]:
        records_by_sub.setdefault(r["subject"], []).append(r)
    for s in records_by_sub:
        records_by_sub[s].sort(key=lambda rec: int(rec["night"]))

    eligible_subjects = sorted([s for s in records_by_sub if s != "Bidslab49" and len(records_by_sub[s]) >= 3])
    assert len(eligible_subjects) == 46, f"Expected 46 eligible subjects, got {len(eligible_subjects)}"

    rows = []
    n_inferences = 0

    with torch.no_grad():
        for fold in range(1, 6):
            fold_subjects = [s for s in eligible_subjects if sub_to_fold[s] == fold]
            for seed in SEEDS:
                ckpt_path = os.path.join(CHECKPOINTS_DIR, f"modelA_L1200_fold{fold}_seed{seed}.pt")
                model = ConventionalCNNBiGRU().to(device)
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
                model.load_state_dict(ckpt)
                model.eval()
                predictor = SlidingWindowPredictor(
                    model, context_len=1200, neutralize_circadian=True, device=device
                )

                for sub in fold_subjects:
                    nights = records_by_sub[sub]
                    n_nights = len(nights)
                    # Predict nights[1:] once (k>=1 never evaluates nights[0])
                    preds_by_idx: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
                    for idx in range(1, n_nights):
                        res = predictor.predict_night(nights[idx])
                        preds_by_idx[idx] = (res["y_pred"], res["y_true"])
                        n_inferences += 1

                    zs_e4_f1 = (
                        pooled_macro_f1([preds_by_idx[idx] for idx in range(4, n_nights)])
                        if n_nights >= 5
                        else np.nan
                    )

                    for k in K_VALUES:
                        if n_nights <= k:
                            continue
                        zs_ek_f1 = pooled_macro_f1([preds_by_idx[idx] for idx in range(k, n_nights)])
                        rows.append({
                            "subject": sub,
                            "fold": fold,
                            "seed": seed,
                            "k": k,
                            "is_common_cohort": (n_nights >= 5),
                            "Clean_ZS_E4_MacroF1": zs_e4_f1,
                            "Clean_ZS_Ek_MacroF1": zs_ek_f1,
                        })

    elapsed = time.time() - t0
    print(f"Computed clean ZS on CPU across {n_inferences} night inferences in {elapsed:.2f}s.")
    df_seed = pd.DataFrame(rows)
    df_clean = df_seed.groupby(["subject", "k", "is_common_cohort"]).agg({
        "Clean_ZS_E4_MacroF1": "mean",
        "Clean_ZS_Ek_MacroF1": "mean",
    }).reset_index()
    return df_clean


def main():
    df_clean_zs = compute_clean_zs_cpu()
    df_ft = pd.read_csv(SUBJECT_METRICS_CSV)
    df_locked = pd.read_csv(LOCKED_EXP019_CSV)

    # Merge clean ZS with already-computed clean FT predictions
    df = pd.merge(df_ft, df_clean_zs, on=["subject", "k", "is_common_cohort"], how="inner")
    assert len(df) == len(df_ft) == 167, f"Merge mismatch: {len(df)} vs {len(df_ft)}"

    df["Clean_Delta_E4_pp"] = (df["FT_E4_MacroF1"] - df["Clean_ZS_E4_MacroF1"]) * 100.0
    df["Clean_Delta_Ek_pp"] = (df["FT_Ek_MacroF1"] - df["Clean_ZS_Ek_MacroF1"]) * 100.0

    # --- Forensic Verification Checks ---
    print("\n" + "=" * 80)
    print("FORENSIC VERIFICATION OF CLEAN ZERO-SHOT BASELINE")
    print("=" * 80)
    # 1. Check k=2 Available Cohort vs locked EXP-019 ZS
    df_k2 = df[df["k"] == 2].merge(df_locked[["subject_id", "ZS"]], left_on="subject", right_on="subject_id")
    max_diff_locked = np.max(np.abs(df_k2["Clean_ZS_Ek_MacroF1"] - df_k2["ZS"]))
    print(f"k=2 Clean ZS mean (N=46):       {df_k2['Clean_ZS_Ek_MacroF1'].mean()*100:.4f}% (locked EXP-019: {df_k2['ZS'].mean()*100:.4f}%)")
    print(f"Max |Clean_ZS_k2 - Locked_ZS|:  {max_diff_locked:.6f} (locked CSV rounded to 4 decimals)")

    # 2. Check fold-first pristine subjects against recorded ZS in exp019_k_curve_subject_metrics.csv
    first_subs = ["Bidslab08", "Bidslab07", "Bidslab06", "Bidslab00", "Bidslab01"]
    df_first = df[df["subject"].isin(first_subs)]
    max_diff_first_ek = np.max(np.abs(df_first["Clean_ZS_Ek_MacroF1"] - df_first["ZS_Ek_MacroF1"]))
    max_diff_first_e4 = np.nanmax(np.abs(df_first["Clean_ZS_E4_MacroF1"] - df_first["ZS_E4_MacroF1"]))
    print(f"Max |Clean_ZS - Recorded_ZS| on 5 fold-first subjects: Ek={max_diff_first_ek:.2e}, E4={max_diff_first_e4:.2e}")

    # 3. Check Common Cohort (N=33, E4) Clean ZS mean
    df_common = df[df["is_common_cohort"]].copy()
    subjects_33 = sorted(df_common["subject"].unique())
    assert len(subjects_33) == 33
    clean_zs_e4_mean = df_common[df_common["k"] == 1]["Clean_ZS_E4_MacroF1"].mean()
    print(f"Common Cohort E4 Clean ZS mean (N=33): {clean_zs_e4_mean*100:.4f}% (TSK06 audit: 63.54%)")

    # --- Build Delta Matrix (33 x 4) on E4 ---
    delta_matrix = np.zeros((33, 4), dtype=np.float64)
    for i, s in enumerate(subjects_33):
        sub_df = df_common[df_common["subject"] == s].sort_values("k")
        delta_matrix[i, :] = sub_df["Clean_Delta_E4_pp"].values

    ranks_matrix = np.zeros_like(delta_matrix)
    for i in range(33):
        ranks_matrix[i, :] = stats.rankdata(delta_matrix[i, :])
    L_stat, z_L, p_trend, mean_L, std_L = pages_l_test(ranks_matrix)

    # Incremental steps on E4
    step_21 = delta_matrix[:, 1] - delta_matrix[:, 0]
    step_32 = delta_matrix[:, 2] - delta_matrix[:, 1]
    step_43 = delta_matrix[:, 3] - delta_matrix[:, 2]

    # Exact two-sided Wilcoxon signed-rank tests on E4 for k=1,2,3,4 vs Clean ZS
    e4_stats = {}
    for idx, k in enumerate(K_VALUES):
        d = delta_matrix[:, idx]
        ci_lo, ci_hi = bootstrap_ci(d, n_boot=10000, alpha=0.05, seed=42)
        w_two, p_two = stats.wilcoxon(d, alternative="two-sided", method="exact")
        w_gt, p_gt = stats.wilcoxon(d, alternative="greater", method="exact")
        wins = int(np.sum(d > 0))
        losses = int(np.sum(d < 0))
        ties = int(np.sum(d == 0))
        e4_stats[k] = {
            "deltas": d,
            "mean_zs": float(df_common[df_common["k"] == k]["Clean_ZS_E4_MacroF1"].mean()),
            "mean_ft": float(df_common[df_common["k"] == k]["FT_E4_MacroF1"].mean()),
            "delta_pp": float(np.mean(d)),
            "sd_pp": float(np.std(d, ddof=1)),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "W": float(w_two),
            "p_two_exact": float(p_two),
            "p_one_exact": float(p_gt),
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }

    # Incremental step Wilcoxon tests (two-sided exact & one-sided)
    _, p_step21_two = stats.wilcoxon(step_21, alternative="two-sided", method="exact")
    _, p_step32_two = stats.wilcoxon(step_32, alternative="two-sided", method="exact")
    _, p_step43_two = stats.wilcoxon(step_43, alternative="two-sided", method="exact")
    _, p_sat_lt050 = stats.wilcoxon(step_43 - 0.50, alternative="less", method="exact")
    ci_step21 = bootstrap_ci(step_21)
    ci_step32 = bootstrap_ci(step_32)
    ci_step43 = bootstrap_ci(step_43)

    # --- 5-Test Holm-Bonferroni Families ---
    # Family 1 (5-test E4 Clean ZS vs k=1,2,3,4 + Page's L): [Page's L, k=1, k=2, k=3, k=4]
    fam_allk_names = ["T_trend (Page's L)", "T_k1 (E4 vs Clean ZS)", "T_k2 (E4 vs Clean ZS)", "T_k3 (E4 vs Clean ZS)", "T_k4 (E4 vs Clean ZS)"]
    fam_allk_p = [p_trend, e4_stats[1]["p_two_exact"], e4_stats[2]["p_two_exact"], e4_stats[3]["p_two_exact"], e4_stats[4]["p_two_exact"]]
    fam_allk_adj, fam_allk_sig = holm_bonferroni_adjust(fam_allk_p)

    # Family 2 (Executed Driver 5-test family from run_tsk06_k_curve.py with two-sided exact Wilcoxon):
    # [Page's L, k=3 vs Clean ZS, k=1 vs Clean ZS, k=4 vs Clean ZS, T_sat (delta_43 < +0.50pp)]
    fam_driver_names = ["T_trend (Page's L)", "T_k3 (E4 vs Clean ZS)", "T_k1 (E4 vs Clean ZS)", "T_k4 (E4 vs Clean ZS)", "T_sat (delta_43 < +0.50 pp)"]
    fam_driver_p = [p_trend, e4_stats[3]["p_two_exact"], e4_stats[1]["p_two_exact"], e4_stats[4]["p_two_exact"], float(p_sat_lt050)]
    fam_driver_adj, fam_driver_sig = holm_bonferroni_adjust(fam_driver_p)

    # Family 3 (Preregistered PREREG_EXP-019_K_CURVE.md Sec 3.4 5-test family):
    # [Page's L, k=3 vs Clean ZS, step 2-1, step 3-2, step 4-3]
    fam_prereg_names = ["T1: Page's L trend", "T2: k=3 vs Clean ZS (E4)", "T3: step delta_{2-1}", "T4: step delta_{3-2}", "T5: step delta_{4-3}"]
    fam_prereg_p = [p_trend, e4_stats[3]["p_two_exact"], float(p_step21_two), float(p_step32_two), float(p_step43_two)]
    fam_prereg_adj, fam_prereg_sig = holm_bonferroni_adjust(fam_prereg_p)

    # --- Available Cohort (Ek: N=46, 46, 42, 33) ---
    ek_stats = {}
    for k in K_VALUES:
        df_k = df[df["k"] == k].sort_values("subject")
        d = df_k["Clean_Delta_Ek_pp"].dropna().values
        ci_lo, ci_hi = bootstrap_ci(d, n_boot=10000, alpha=0.05, seed=42)
        w_two, p_two = stats.wilcoxon(d, alternative="two-sided", method="exact")
        w_gt, p_gt = stats.wilcoxon(d, alternative="greater", method="exact")
        wins = int(np.sum(d > 0))
        losses = int(np.sum(d < 0))
        ties = int(np.sum(d == 0))
        ek_stats[k] = {
            "N": len(d),
            "mean_zs": float(df_k["Clean_ZS_Ek_MacroF1"].mean()),
            "mean_ft": float(df_k["FT_Ek_MacroF1"].mean()),
            "delta_pp": float(np.mean(d)),
            "sd_pp": float(np.std(d, ddof=1)),
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "W": float(w_two),
            "p_two_exact": float(p_two),
            "p_one_exact": float(p_gt),
            "wins": wins,
            "losses": losses,
            "ties": ties,
        }

    # --- Print Complete Results ---
    print("\n" + "=" * 80)
    print("1. COMMON COHORT (N=33, E4) vs CLEAN ZERO-SHOT (63.54%)")
    print("=" * 80)
    print(f"Page's L-Trend Test: L = {L_stat:.1f} (mu = {mean_L:.1f}, sigma = {std_L:.2f}), z = {z_L:.3f}, p = {p_trend:.6f}")
    print(f"{'k':>2} | {'N':>2} | {'Clean ZS':>8} | {'Mean FT':>8} | {'Delta pp':>9} | {'95% Bootstrap CI':>20} | {'W':>6} | {'p (2s exact)':>12} | {'Wins':>7}")
    print("-" * 95)
    for k in K_VALUES:
        st = e4_stats[k]
        print(
            f"{k:>2} | {33:>2} | {st['mean_zs']*100:>7.2f}% | {st['mean_ft']*100:>7.2f}% | "
            f"{st['delta_pp']:>+8.2f}  | [{st['ci_lo']:>+7.2f}, {st['ci_hi']:>+7.2f}] | "
            f"{st['W']:>6.1f} | {st['p_two_exact']:>12.6g} | {st['wins']:>2}/33"
        )

    print("\nIncremental Steps on E4 (N=33):")
    print(f"  delta_(2-1): {np.mean(step_21):+.4f} pp [95% CI: {ci_step21[0]:+.2f}, {ci_step21[1]:+.2f}], exact 2s p = {p_step21_two:.6g}")
    print(f"  delta_(3-2): {np.mean(step_32):+.4f} pp [95% CI: {ci_step32[0]:+.2f}, {ci_step32[1]:+.2f}], exact 2s p = {p_step32_two:.6g}")
    print(f"  delta_(4-3): {np.mean(step_43):+.4f} pp [95% CI: {ci_step43[0]:+.2f}, {ci_step43[1]:+.2f}], exact 2s p = {p_step43_two:.6g}, p_sat(<0.50pp) = {p_sat_lt050:.6g}")

    print("\n" + "=" * 80)
    print("2. AVAILABLE COHORT (Ek, N=46/46/42/33) vs CLEAN ZERO-SHOT")
    print("=" * 80)
    print(f"{'k':>2} | {'N':>2} | {'Clean ZS':>8} | {'Mean FT':>8} | {'Delta pp':>9} | {'95% Bootstrap CI':>20} | {'W':>6} | {'p (2s exact)':>12} | {'Wins':>7}")
    print("-" * 95)
    for k in K_VALUES:
        st = ek_stats[k]
        print(
            f"{k:>2} | {st['N']:>2} | {st['mean_zs']*100:>7.2f}% | {st['mean_ft']*100:>7.2f}% | "
            f"{st['delta_pp']:>+8.2f}  | [{st['ci_lo']:>+7.2f}, {st['ci_hi']:>+7.2f}] | "
            f"{st['W']:>6.1f} | {st['p_two_exact']:>12.6g} | {st['wins']:>2}/{st['N']}"
        )

    print("\n" + "=" * 80)
    print("3. 5-TEST HOLM-BONFERRONI ADJUSTED P-VALUES (THREE FAMILY CHECKS)")
    print("=" * 80)
    print("\n[A] Executed 5-Test Family (Page's L, k=1, k=2, k=3, k=4 vs Clean ZS on E4; 2-sided exact Wilcoxon):")
    for name, raw_p, adj_p, sig in zip(fam_allk_names, fam_allk_p, fam_allk_adj, fam_allk_sig):
        print(f"  {name:<30} raw p = {raw_p:.6g} | p_holm = {adj_p:.6g} | Sig(0.05) = {sig}")

    print("\n[B] Driver 5-Test Family (run_tsk06_k_curve.py: Page's L, k=3, k=1, k=4 vs Clean ZS, T_sat<0.50pp; 2-sided exact Wilcoxon):")
    for name, raw_p, adj_p, sig in zip(fam_driver_names, fam_driver_p, fam_driver_adj, fam_driver_sig):
        print(f"  {name:<30} raw p = {raw_p:.6g} | p_holm = {adj_p:.6g} | Sig(0.05) = {sig}")

    print("\n[C] Preregistered 5-Test Family (PREREG_EXP-019_K_CURVE.md Sec 3.4: Page's L, k=3 vs Clean ZS, steps 2-1, 3-2, 4-3; 2-sided exact Wilcoxon):")
    for name, raw_p, adj_p, sig in zip(fam_prereg_names, fam_prereg_p, fam_prereg_adj, fam_prereg_sig):
        print(f"  {name:<30} raw p = {raw_p:.6g} | p_holm = {adj_p:.6g} | Sig(0.05) = {sig}")

    # --- Save Clean Reconciled CSV ---
    # Include both Holm-Bonferroni adjusted p-values (Family A: Page's L + k=1..4; Family B: Driver family; Family C: Preregistered Sec 3.4)
    # Wait: let's design clean, transparent columns!
    out_rows = []
    # Map k -> Holm adj p for Family A (Page's L + k=1,2,3,4), Family B (Driver: Page's L, k=3, k=1, k=4, T_sat), Family C (Prereg: Page's L, k=3, steps)
    fam_a_map = {1: (fam_allk_adj[1], fam_allk_sig[1]), 2: (fam_allk_adj[2], fam_allk_sig[2]), 3: (fam_allk_adj[3], fam_allk_sig[3]), 4: (fam_allk_adj[4], fam_allk_sig[4])}
    fam_b_map = {1: (fam_driver_adj[2], fam_driver_sig[2]), 2: (np.nan, np.nan), 3: (fam_driver_adj[1], fam_driver_sig[1]), 4: (fam_driver_adj[3], fam_driver_sig[3])}
    fam_c_map = {1: (np.nan, np.nan), 2: (np.nan, np.nan), 3: (fam_prereg_adj[1], fam_prereg_sig[1]), 4: (np.nan, np.nan)}

    for k in K_VALUES:
        st = e4_stats[k]
        out_rows.append({
            "Cohort": "Common (N=33, E4)",
            "k": k,
            "N": 33,
            "Mean_ZS_Clean": st["mean_zs"],
            "Mean_FT": st["mean_ft"],
            "Delta_pp": st["delta_pp"],
            "CI_lower": st["ci_lo"],
            "CI_upper": st["ci_hi"],
            "Wilcoxon_W": st["W"],
            "p_wilcoxon_two_sided_exact": st["p_two_exact"],
            "p_holm_5test_E4_allk": fam_a_map[k][0],
            "Holm_Sig_5test_E4_allk": fam_a_map[k][1],
            "p_holm_5test_driver": fam_b_map[k][0],
            "Holm_Sig_5test_driver": fam_b_map[k][1],
            "p_holm_5test_prereg": fam_c_map[k][0],
            "Holm_Sig_5test_prereg": fam_c_map[k][1],
            "Wins": st["wins"],
            "Losses": st["losses"],
            "Ties": st["ties"],
        })

    for k in K_VALUES:
        st = ek_stats[k]
        out_rows.append({
            "Cohort": f"Available (Ek, k={k})",
            "k": k,
            "N": st["N"],
            "Mean_ZS_Clean": st["mean_zs"],
            "Mean_FT": st["mean_ft"],
            "Delta_pp": st["delta_pp"],
            "CI_lower": st["ci_lo"],
            "CI_upper": st["ci_hi"],
            "Wilcoxon_W": st["W"],
            "p_wilcoxon_two_sided_exact": st["p_two_exact"],
            "p_holm_5test_E4_allk": np.nan,
            "Holm_Sig_5test_E4_allk": np.nan,
            "p_holm_5test_driver": np.nan,
            "Holm_Sig_5test_driver": np.nan,
            "p_holm_5test_prereg": np.nan,
            "Holm_Sig_5test_prereg": np.nan,
            "Wins": st["wins"],
            "Losses": st["losses"],
            "Ties": st["ties"],
        })

    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(OUTPUT_RECONCILED_CSV, index=False)
    print(f"\nSaved clean reconciled table to: {OUTPUT_RECONCILED_CSV}")


if __name__ == "__main__":
    main()
