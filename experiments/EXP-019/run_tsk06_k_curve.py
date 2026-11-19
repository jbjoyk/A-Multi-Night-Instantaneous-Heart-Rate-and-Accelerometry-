"""
run_tsk06_k_curve.py: Calibration-Depth Learning Curve Execution Driver (TSK-06 / EXP-019-K).

Governed by: PREREG_EXP-019_K_CURVE.md
Authority: RESEARCH_DIRECTION_v4_FULL.md, DEC-033, DEC-034.

Dual-Cohort Design:
  1. Primary Analysis: Common-cohort of N=33 fixed participants (n_usable >= 5)
     evaluated on fixed late evaluation nights E_4 (64 nights total; Nights 5+).
  2. Sensitivity Analysis: Available cohort (N_k in {46, 46, 42, 33})
     evaluated on all chronological nights E_k (Nights k+1+).

Statistical Protocol:
  - Page's L-trend test for repeated measures across k in {1, 2, 3, 4}.
  - van der Aar replication test: Delta_k=3 vs ZS on E_4 (H0: Delta <= 0 vs H1: Delta > 0).
  - Saturation test: delta_{4-3} = Delta_k=4 - Delta_k=3 against +0.50 pp equivalence bound.
  - Multiplicity control: 5-test Holm-Bonferroni step-down family (alpha = 0.05).
  - Two-dimensional decision classification.

Historical Invariant:
  - Historical k=2 results on E_2 in exp019_subject_metrics.csv remain strictly locked.
  - Evaluation of k=2 on E_4 is performed on the identical Model A checkpoints without altering historical artifacts.
"""

import os
import sys
import copy
import json
import time
import argparse
import numpy as np
import scipy.stats as stats
import pandas as pd
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
exp018_dir = os.path.abspath(os.path.join(repo_root, "experiments", "EXP-018"))
for p in [repo_root, exp018_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from sklearn.metrics import f1_score, accuracy_score
from exp013_models import ConventionalCNNBiGRU
from exp018_inference import SlidingWindowPredictor
from slamss_ifs_train import prepare_dataset_records
from exp019_adaptation_core import (
    reset_model_weights,
    adapt_supervised_ft
)

CACHE_DIR_DEFAULT = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
MANIFESTS_DIR_DEFAULT = os.path.join(repo_root, "experiments", "EXP-014", "EXP014_MANIFESTS")
CHECKPOINTS_DIR_DEFAULT = os.path.join(repo_root, "experiments", "EXP-018", "checkpoints")
STAGE_D_CONFIGS_DEFAULT = os.path.join(repo_root, "experiments", "EXP-019", "exp019_stage_d_selected_configs.json")
OUTPUT_CSV_DEFAULT = os.path.join(repo_root, "experiments", "EXP-019", "exp019_k_learning_curve_executed.csv")
SUBJECT_CSV_DEFAULT = os.path.join(repo_root, "experiments", "EXP-019", "exp019_k_curve_subject_metrics.csv")

SEEDS_DEFAULT = [42, 123, 2024]
K_VALUES_DEFAULT = [1, 2, 3, 4]

SCOPE_REGEX_MAP = {
    "head_gru": r"(classifier_head|temporal_gru.*weight_.*l1|temporal_gru.*bias_.*l1|temp_norm)",
    "all": r".*"
}


def pages_l_test(ranks_matrix: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Computes Page's L-trend test for ordered alternatives in repeated measures.
    ranks_matrix: shape (N, k), where ranks r_{ik} in {1, ..., k} are computed within each row i.
    Returns: (L, z, p_value, mean_L, std_L)
    """
    N, k = ranks_matrix.shape
    R = np.sum(ranks_matrix, axis=0)  # Rank sum for each condition k
    weights = np.arange(1, k + 1)
    L = float(np.sum(weights * R))

    # Theoretical null distribution parameters
    mean_L = N * k * ((k + 1) ** 2) / 4.0
    var_L = N * (k ** 2) * (k + 1) * (k ** 2 - 1) / 144.0
    std_L = np.sqrt(var_L)

    # Continuity-corrected z-score towards the mean
    if L > mean_L:
        z = (L - mean_L - 0.5) / std_L
    else:
        z = (L - mean_L + 0.5) / std_L

    p_value = float(1.0 - stats.norm.cdf(z))  # Upper-tailed test for ordered increase
    return L, float(z), p_value, mean_L, std_L


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """Computes participant-level percentile bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    low = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    high = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return low, high


def evaluate_records(predictor: SlidingWindowPredictor, records: List[Dict[str, np.ndarray]]) -> Tuple[float, float]:
    """Evaluates a list of records and returns (Macro-F1, Accuracy) pooled at valid epoch level [0..3]."""
    preds, gts = [], []
    for r in records:
        pred_res = predictor.predict_night(r)
        preds.append(pred_res["y_pred"])
        gts.append(pred_res["y_true"])
    all_preds = np.concatenate(preds)
    all_gts = np.concatenate(gts)
    valid = (all_gts >= 0) & (all_gts < 4)
    if not np.any(valid):
        return 0.0, 0.0
    macro_f1 = float(f1_score(all_gts[valid], all_preds[valid], average="macro"))
    accuracy = float(accuracy_score(all_gts[valid], all_preds[valid]))
    return macro_f1, accuracy


def run_k_learning_curve(args):
    print("=" * 80)
    print("EXP-019-K: CALIBRATION-DEPTH LEARNING CURVE DRIVER (TSK-06)")
    print(f"Device: {args.device} | Seeds: {args.seeds} | Dry Run: {args.dry_run}")
    print(f"Target Cache: {args.cache_dir}")
    print("=" * 80)

    # 1. Load Split Manifests
    fold_manifests = {}
    sub_to_fold = {}
    for fold in range(1, 6):
        mpath = os.path.join(args.manifests_dir, f"fold_{fold}.json")
        with open(mpath, "r") as f:
            m = json.load(f)
            fold_manifests[fold] = m
            test_subs = m.get("test", m.get("outer_test_subjects", []))
            for s in test_subs:
                sub_to_fold[s] = fold

    # 2. Ingest Records Grouped by Subject and Chronologically Sorted
    print("Loading preprocessed dataset records...")
    # Load all records across all test subjects
    split_dict = {"train": [], "val": [], "test": list(sub_to_fold.keys())}
    records_by_split = prepare_dataset_records(args.cache_dir, split_dict)
    test_records = records_by_split["test"]

    records_by_sub: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for r in test_records:
        records_by_sub.setdefault(r["subject"], []).append(r)

    # Sort each subject's nights chronologically by night number
    for s in records_by_sub:
        records_by_sub[s].sort(key=lambda rec: int(rec["night"]))

    # Eligible cohort (46 subjects, excluding Bidslab49)
    eligible_subjects = sorted([s for s in records_by_sub if s != "Bidslab49" and len(records_by_sub[s]) >= 3])
    if args.subjects:
        eligible_subjects = [s for s in eligible_subjects if s in args.subjects]

    print(f"Total Eligible Subjects: {len(eligible_subjects)}")

    # Dual-Cohort Partitioning
    common_cohort_33 = [s for s in eligible_subjects if len(records_by_sub[s]) >= 5]
    print(f"Common-Cohort Subjects (n_usable >= 5): {len(common_cohort_33)}")
    print(f"Available Subjects: k=1: {len(eligible_subjects)}, k=2: {len(eligible_subjects)}, "
          f"k=3: {len([s for s in eligible_subjects if len(records_by_sub[s]) >= 4])}, "
          f"k=4: {len(common_cohort_33)}")

    # Load Stage D selected configurations
    with open(args.configs_json, "r") as f:
        stage_d_cfg = json.load(f)
    print("Loaded Stage D Configurations per fold.")

    # Container for all subject evaluations across seeds and k
    # subject_records: subject -> seed -> k -> {eval_set -> {Macro-F1, Acc}}
    results_store = []

    # Filter folds if specified
    active_folds = args.folds if args.folds else list(range(1, 6))
    print(f"Executing Outer Folds: {active_folds}")

    for fold in active_folds:
        fold_subjects = [s for s in eligible_subjects if sub_to_fold[s] == fold]
        if not fold_subjects:
            continue
        print(f"\n>>> PROCESSING OUTER FOLD {fold} ({len(fold_subjects)} subjects) <<<")

        # Fold winning hyperparameters
        cfg_k = stage_d_cfg.get("folds", {}).get(str(fold), {}).get("FT-2", {})
        lr = cfg_k.get("lr", 3e-5)
        epochs = cfg_k.get("epochs", 5)
        scope = cfg_k.get("scope", "all")
        scope_regex = SCOPE_REGEX_MAP.get(scope, scope)

        for seed in args.seeds:
            ckpt_path = os.path.join(args.checkpoints_dir, f"modelA_L1200_fold{fold}_seed{seed}.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            # Instantiate Model A backbone
            base_model = ConventionalCNNBiGRU().to(args.device)
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
            base_model.load_state_dict(ckpt)
            base_model.eval()
            backbone_state = copy.deepcopy(base_model.state_dict())

            predictor_zs = SlidingWindowPredictor(base_model, context_len=1200, neutralize_circadian=True, device=torch.device(args.device))

            for sub in fold_subjects:
                nights = records_by_sub[sub]
                n_nights = len(nights)

                # 1. Zero-Shot Evaluation
                # E_4 evaluation (Nights 5+)
                e4_records = nights[4:] if n_nights >= 5 else []
                zs_e4_f1, zs_e4_acc = (evaluate_records(predictor_zs, e4_records) if e4_records else (np.nan, np.nan))

                # E_k evaluations
                zs_ek_metrics = {}
                for k in K_VALUES_DEFAULT:
                    ek_recs = nights[k:] if n_nights > k else []
                    if ek_recs:
                        f1, acc = evaluate_records(predictor_zs, ek_recs)
                        zs_ek_metrics[k] = (f1, acc)
                    else:
                        zs_ek_metrics[k] = (np.nan, np.nan)

                # 2. Adaptation for each k in {1, 2, 3, 4}
                for k in K_VALUES_DEFAULT:
                    if n_nights <= k:
                        # Subject has no evaluation night at depth k
                        continue

                    # Calibration nights: strictly first k chronological nights
                    cal_records = nights[:k]
                    # Evaluation nights: E_k (all subsequent nights)
                    eval_records_ek = nights[k:]
                    # Evaluation nights: E_4 (Nights 5+, if eligible)
                    eval_records_e4 = nights[4:] if n_nights >= 5 else []

                    # Restore clean backbone state
                    reset_model_weights(base_model, backbone_state)

                    if not args.dry_run:
                        # Fine-tune model on calibration nights
                        adapted_model = adapt_supervised_ft(
                            model=base_model,
                            cal_records=cal_records,
                            lr=lr,
                            epochs=epochs,
                            scope_regex=scope_regex,
                            device=args.device
                        )
                    else:
                        # Dry-run: skip gradient steps for fast verification
                        adapted_model = base_model

                    predictor_ft = SlidingWindowPredictor(adapted_model, context_len=1200, neutralize_circadian=True, device=torch.device(args.device))

                    # Evaluate on E_k (Sensitivity cohort)
                    ft_ek_f1, ft_ek_acc = evaluate_records(predictor_ft, eval_records_ek)
                    zs_ek_f1, _ = zs_ek_metrics[k]
                    delta_ek = (ft_ek_f1 - zs_ek_f1) * 100.0

                    # Evaluate on E_4 (Common cohort, if eligible)
                    if eval_records_e4:
                        ft_e4_f1, ft_e4_acc = evaluate_records(predictor_ft, eval_records_e4)
                        delta_e4 = (ft_e4_f1 - zs_e4_f1) * 100.0
                    else:
                        ft_e4_f1, ft_e4_acc, delta_e4 = np.nan, np.nan, np.nan

                    row = {
                        "subject": sub,
                        "fold": fold,
                        "seed": seed,
                        "k": k,
                        "n_cal_nights": k,
                        "total_nights": n_nights,
                        "is_common_cohort": (n_nights >= 5),
                        "ZS_E4_MacroF1": zs_e4_f1,
                        "FT_E4_MacroF1": ft_e4_f1,
                        "Delta_E4_pp": delta_e4,
                        "ZS_Ek_MacroF1": zs_ek_f1,
                        "FT_Ek_MacroF1": ft_ek_f1,
                        "Delta_Ek_pp": delta_ek,
                        "n_eval_nights_E4": len(eval_records_e4),
                        "n_eval_nights_Ek": len(eval_records_ek)
                    }
                    results_store.append(row)

                if args.dry_run:
                    print(f"Dry-run check passed for subject {sub}")
                    break
            if args.dry_run:
                break
        if args.dry_run:
            break

    df_results = pd.DataFrame(results_store)
    print(f"\nCompleted execution across {len(df_results)} subject-seed-depth instances.")

    if args.dry_run:
        print("Dry run completed successfully. Zero production artifacts modified.")
        return df_results

    # 3. Statistical Analysis & Decision Rules
    # Average across seeds per subject and k
    df_subject_means = df_results.groupby(["subject", "k", "is_common_cohort"]).agg({
        "ZS_E4_MacroF1": "mean",
        "FT_E4_MacroF1": "mean",
        "Delta_E4_pp": "mean",
        "ZS_Ek_MacroF1": "mean",
        "FT_Ek_MacroF1": "mean",
        "Delta_Ek_pp": "mean"
    }).reset_index()

    # Save subject-level metrics
    os.makedirs(os.path.dirname(args.subject_csv), exist_ok=True)
    df_subject_means.to_csv(args.subject_csv, index=False)
    print(f"Saved subject metrics to {args.subject_csv}")

    # Primary Analysis: Common-Cohort N=33 on E4
    df_common = df_subject_means[df_subject_means["is_common_cohort"]].copy()
    subjects_33 = df_common["subject"].unique()
    assert len(subjects_33) == 33, f"Expected 33 common-cohort subjects, found {len(subjects_33)}"

    # Build delta matrix (33 subjects x 4 depths) for Page's L-test
    delta_matrix = np.zeros((33, 4), dtype=np.float64)
    for i, s in enumerate(sorted(subjects_33)):
        sub_df = df_common[df_common["subject"] == s].sort_values("k")
        delta_matrix[i, :] = sub_df["Delta_E4_pp"].values

    # Compute ranks per row (participant) for Page's test
    ranks_matrix = np.zeros_like(delta_matrix)
    for i in range(33):
        ranks_matrix[i, :] = stats.rankdata(delta_matrix[i, :])

    L_stat, z_L, p_trend, mean_L, std_L = pages_l_test(ranks_matrix)

    # Hypothesis tests on E4
    delta_k1_e4 = delta_matrix[:, 0]
    delta_k2_e4 = delta_matrix[:, 1]
    delta_k3_e4 = delta_matrix[:, 2]
    delta_k4_e4 = delta_matrix[:, 3]

    # Wilcoxon signed-rank tests vs ZS (H0: Delta <= 0)
    w_stat_k3, p_k3_two = stats.wilcoxon(delta_k3_e4, alternative='greater')
    w_stat_k1, p_k1_two = stats.wilcoxon(delta_k1_e4, alternative='greater')
    w_stat_k4, p_k4_two = stats.wilcoxon(delta_k4_e4, alternative='greater')

    # Saturation increment delta_{4-3}
    step_43 = delta_k4_e4 - delta_k3_e4
    w_stat_sat, p_sat = stats.wilcoxon(step_43 - 0.50, alternative='less')  # Testing if step < +0.50 pp

    # 5-test Holm-Bonferroni Family
    p_family = [p_trend, p_k3_two, p_k1_two, p_k4_two, p_sat]
    test_names = ["T_trend (Page L)", "T_k=3 (vs ZS)", "T_k=1 (vs ZS)", "T_k=4 (vs ZS)", "T_sat (delta_43 < 0.5pp)"]
    alpha_family = 0.05
    m = len(p_family)

    order = np.argsort(p_family)
    holm_rejected = [False] * m
    for rank_idx, test_idx in enumerate(order):
        adjusted_alpha = alpha_family / (m - rank_idx)
        if p_family[test_idx] <= adjusted_alpha:
            holm_rejected[test_idx] = True
        else:
            break

    # Bootstrap CIs for Common Cohort
    ci_k1 = bootstrap_ci(delta_k1_e4)
    ci_k2 = bootstrap_ci(delta_k2_e4)
    ci_k3 = bootstrap_ci(delta_k3_e4)
    ci_k4 = bootstrap_ci(delta_k4_e4)
    ci_sat = bootstrap_ci(step_43)

    # Compile Summary Table
    summary_rows = [
        {"Cohort": "Common (N=33, E4)", "k": 1, "N": 33, "Mean_ZS": float(np.mean(df_common[df_common['k']==1]['ZS_E4_MacroF1'])), "Mean_FT": float(np.mean(df_common[df_common['k']==1]['FT_E4_MacroF1'])), "Delta_pp": float(np.mean(delta_k1_e4)), "CI_lower": ci_k1[0], "CI_upper": ci_k1[1], "p_val": p_k1_two, "Holm_Sig": holm_rejected[2]},
        {"Cohort": "Common (N=33, E4)", "k": 2, "N": 33, "Mean_ZS": float(np.mean(df_common[df_common['k']==2]['ZS_E4_MacroF1'])), "Mean_FT": float(np.mean(df_common[df_common['k']==2]['FT_E4_MacroF1'])), "Delta_pp": float(np.mean(delta_k2_e4)), "CI_lower": ci_k2[0], "CI_upper": ci_k2[1], "p_val": np.nan, "Holm_Sig": np.nan},
        {"Cohort": "Common (N=33, E4)", "k": 3, "N": 33, "Mean_ZS": float(np.mean(df_common[df_common['k']==3]['ZS_E4_MacroF1'])), "Mean_FT": float(np.mean(df_common[df_common['k']==3]['FT_E4_MacroF1'])), "Delta_pp": float(np.mean(delta_k3_e4)), "CI_lower": ci_k3[0], "CI_upper": ci_k3[1], "p_val": p_k3_two, "Holm_Sig": holm_rejected[1]},
        {"Cohort": "Common (N=33, E4)", "k": 4, "N": 33, "Mean_ZS": float(np.mean(df_common[df_common['k']==4]['ZS_E4_MacroF1'])), "Mean_FT": float(np.mean(df_common[df_common['k']==4]['FT_E4_MacroF1'])), "Delta_pp": float(np.mean(delta_k4_e4)), "CI_lower": ci_k4[0], "CI_upper": ci_k4[1], "p_val": p_k4_two, "Holm_Sig": holm_rejected[3]},
    ]

    # Sensitivity Analysis on Available Cohort (Ek)
    for k in K_VALUES_DEFAULT:
        df_k = df_subject_means[df_subject_means["k"] == k]
        deltas_ek = df_k["Delta_Ek_pp"].dropna().values
        ci_ek = bootstrap_ci(deltas_ek)
        summary_rows.append({
            "Cohort": f"Available (Ek, k={k})",
            "k": k,
            "N": len(deltas_ek),
            "Mean_ZS": float(np.mean(df_k["ZS_Ek_MacroF1"])),
            "Mean_FT": float(np.mean(df_k["FT_Ek_MacroF1"])),
            "Delta_pp": float(np.mean(deltas_ek)),
            "CI_lower": ci_ek[0],
            "CI_upper": ci_ek[1],
            "p_val": np.nan,
            "Holm_Sig": np.nan
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(args.output_csv, index=False)
    print(f"\nSerialized executed learning curve summary to {args.output_csv}")

    # Decision Categorization
    print("\n" + "=" * 80)
    print("PRIMARY STATISTICAL INFERENCE & DECISION CATEGORIZATION")
    print("=" * 80)
    print(f"Page's L-Trend Test: L = {L_stat:.1f} (mu = {mean_L:.1f}, sig = {std_L:.2f}), z = {z_L:.3f}, p = {p_trend:.4e} (Holm Sig: {holm_rejected[0]})")
    print(f"k=3 Gain vs ZS: Delta = {np.mean(delta_k3_e4):+.2f} pp [95% CI: {ci_k3[0]:+.2f}, {ci_k3[1]:+.2f} pp], p = {p_k3_two:.4e} (Holm Sig: {holm_rejected[1]})")
    print(f"Marginal Step delta_{{4-3}}: {np.mean(step_43):+.2f} pp [95% CI: {ci_sat[0]:+.2f}, {ci_sat[1]:+.2f} pp], p_sat = {p_sat:.4e} (Holm Sig: {holm_rejected[4]})")

    # Classification
    if holm_rejected[1] and np.mean(delta_k3_e4) >= 1.50 and ci_k3[0] > 0.0:
        d1_verdict = "REPLICATION CONFIRMED (van der Aar Replicated)"
    elif holm_rejected[1] and ci_k3[0] > 0.0:
        d1_verdict = "SUB-BENCHMARK GAIN (Statistically Superior, Below +1.50 pp)"
    elif ci_k3[1] < 1.50 and not holm_rejected[1]:
        d1_verdict = "CLINICAL BENCHMARK RULED OUT"
    else:
        d1_verdict = "INCONCLUSIVE"

    if holm_rejected[0] and ci_sat[1] < 0.50:
        d2_verdict = "MONOTONIC SCALING WITH EARLY SATURATION (Plateau at k=3)"
    elif holm_rejected[0] and ci_sat[0] >= 0.50:
        d2_verdict = "MONOTONIC SCALING WITH CONTINUING RETURNS"
    elif not holm_rejected[0]:
        d2_verdict = "NO MONOTONIC SCALING (Trend Null Not Rejected)"
    else:
        d2_verdict = "INCONCLUSIVE SCALING"

    print(f"\nDimension 1 (van der Aar Replication): {d1_verdict}")
    print(f"Dimension 2 (Depth Scaling Trajectory):  {d2_verdict}")
    print("=" * 80)

    return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSK-06 Calibration-Depth Learning Curve Driver")
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--manifests-dir", default=MANIFESTS_DIR_DEFAULT)
    parser.add_argument("--checkpoints-dir", default=CHECKPOINTS_DIR_DEFAULT)
    parser.add_argument("--configs-json", default=STAGE_D_CONFIGS_DEFAULT)
    parser.add_argument("--output-csv", default=OUTPUT_CSV_DEFAULT)
    parser.add_argument("--subject-csv", default=SUBJECT_CSV_DEFAULT)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    parser.add_argument("--folds", nargs="+", type=int, default=None)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Runs single verification without model adaptation")
    args = parser.parse_args()
    run_k_learning_curve(args)
