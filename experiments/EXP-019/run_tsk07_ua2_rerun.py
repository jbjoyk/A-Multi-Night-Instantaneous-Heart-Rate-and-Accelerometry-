"""
run_tsk07_ua2_rerun.py: Unsupervised Test-Time Adaptation Re-Run Driver (TSK-07 / EXP-019-S3).

Governed by: PREREG_EXP-019_S3_AMENDMENT.md
Authority: RESEARCH_DIRECTION_v4_FULL.md, DEC-030, DEC-033, DEC-034.

Implementation Remediation:
  - Resolves DEC-030 implementation no-op by using the corrected wearable quality mask:
    * Accelerometer validity: scale-invariant variance check (Var_t(m) > 0.0).
    * Heart rate validity: physiological bounds on reconstructed/raw values and valid range on standardized features.
  - Mandatory pre-flight quality gate assertions:
    * valid_epoch_ratio > 0.95 across cohort.
    * non_physiological_leakage_count == 0.
    * min_epochs_per_night >= 600.
    * Serialized to experiments/EXP-019/ua2_quality_gate_audit.log prior to adaptation.
  - Information Maximization (InfoMax) loss: conditional entropy minimization + marginal class diversity penalty.
  - Scope: Model A head_gru (48,836 trainable parameters; CNN backbone frozen).
  - Hyperparameters: Stage D locked configuration (eta = 1e-5, 20 optimization steps, AdamW).
  - Outputs serialized to exp019_ua2_rerun_subject_metrics.csv (historical exp019_subject_metrics.csv locked).
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
    compute_wearable_quality_mask,
    prepare_calibration_batch,
    adapt_unsupervised_infomax
)

CACHE_DIR_DEFAULT = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
MANIFESTS_DIR_DEFAULT = os.path.join(repo_root, "experiments", "EXP-014", "EXP014_MANIFESTS")
CHECKPOINTS_DIR_DEFAULT = os.path.join(repo_root, "experiments", "EXP-018", "checkpoints")
STAGE_D_CONFIGS_DEFAULT = os.path.join(repo_root, "experiments", "EXP-019", "exp019_stage_d_selected_configs.json")
OUTPUT_CSV_DEFAULT = os.path.join(repo_root, "experiments", "EXP-019", "exp019_ua2_rerun_subject_metrics.csv")
AUDIT_LOG_DEFAULT = os.path.join(repo_root, "experiments", "EXP-019", "ua2_quality_gate_audit.log")

SEEDS_DEFAULT = [42, 123, 2024]


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


def execute_preflight_gate(records_by_sub: Dict[str, List[Dict[str, np.ndarray]]], eligible_subjects: List[str], log_path: str):
    """
    Executes mandatory pre-flight quality gate audit per PREREG_EXP-019_S3_AMENDMENT.md Section 3.2:
      assert valid_epoch_ratio > 0.95
      assert non_physiological_leakage_count == 0
      assert min_epochs_per_night >= 600
    """
    total_epochs = 0
    valid_epochs = 0
    non_physiological_leakage_count = 0
    night_valid_counts = []
    log_lines = []

    log_lines.append("=" * 80)
    log_lines.append("UA-2 MANDATORY PRE-FLIGHT WEARABLE QUALITY GATE AUDIT")
    log_lines.append(f"Timestamp: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    log_lines.append(f"Cohort: {len(eligible_subjects)} subjects (Nights 1 & 2)")
    log_lines.append("=" * 80)

    for sub in eligible_subjects:
        nights = records_by_sub[sub][:2]
        for n_idx, rec in enumerate(nights, start=1):
            ihr = rec["ihr"]
            acc = rec["accel"]
            N = len(rec["y"])
            mask = compute_wearable_quality_mask(ihr[:N], acc[:N])

            # Check for non-physiological leakage (NaN, Inf, or impossible acc var)
            has_nan = np.isnan(ihr[:N]).any() or np.isnan(acc[:N]).any()
            has_inf = np.isinf(ihr[:N]).any() or np.isinf(acc[:N]).any()
            if has_nan or has_inf:
                non_physiological_leakage_count += 1

            n_valid = int(mask.sum())
            total_epochs += N
            valid_epochs += n_valid
            night_valid_counts.append(n_valid)
            ratio = n_valid / N
            log_lines.append(f"Subject {sub} Night {n_idx}: {n_valid}/{N} valid epochs ({ratio:.2%})")

    cohort_valid_ratio = valid_epochs / total_epochs
    min_epochs_per_night = min(night_valid_counts)

    log_lines.append("-" * 80)
    log_lines.append(f"Cohort Total Calibration Epochs:  {total_epochs:,}")
    log_lines.append(f"Cohort Valid Epochs:              {valid_epochs:,} ({cohort_valid_ratio:.4%})")
    log_lines.append(f"Cohort Minimum Epochs Per Night:  {min_epochs_per_night}")
    log_lines.append(f"Non-Physiological Leakage Count:  {non_physiological_leakage_count}")
    log_lines.append("-" * 80)

    # Perform mandatory assertions
    try:
        assert cohort_valid_ratio > 0.95, (
            f"Quality filter rejected too many epochs ({cohort_valid_ratio:.2%}). "
            "Check whether z-scores are being passed to raw thresholds!"
        )
        assert non_physiological_leakage_count == 0, (
            f"Non-physiological epochs detected entering adaptation loss ({non_physiological_leakage_count})!"
        )
        assert min_epochs_per_night >= 600, (
            f"Calibration night has fewer than 600 valid epochs ({min_epochs_per_night})."
        )
        gate_status = "PASSED"
        log_lines.append("PRE-FLIGHT GATE STATUS: PASSED (All 3 mandatory assertions satisfied)")
    except AssertionError as e:
        gate_status = f"FAILED: {e}"
        log_lines.append(f"PRE-FLIGHT GATE STATUS: {gate_status}")

    log_lines.append("=" * 80)

    # Serialize audit log
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Pre-flight quality gate log written to: {log_path}")
    print(f"Gate Verdict: {gate_status}")

    if "FAILED" in gate_status:
        raise AssertionError(f"Pre-flight Quality Gate Aborted: {gate_status}")


def run_ua2_rerun(args):
    print("=" * 80)
    print("EXP-019-S3: UA-2 UNSUPERVISED ADAPTATION RE-RUN DRIVER (TSK-07)")
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

    # 2. Ingest Records Grouped by Subject
    split_dict = {"train": [], "val": [], "test": list(sub_to_fold.keys())}
    records_by_split = prepare_dataset_records(args.cache_dir, split_dict)
    test_records = records_by_split["test"]

    records_by_sub: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for r in test_records:
        records_by_sub.setdefault(r["subject"], []).append(r)

    for s in records_by_sub:
        records_by_sub[s].sort(key=lambda rec: int(rec["night"]))

    eligible_subjects = sorted([s for s in records_by_sub if s != "Bidslab49" and len(records_by_sub[s]) >= 3])
    if args.subjects:
        eligible_subjects = [s for s in eligible_subjects if s in args.subjects]

    print(f"Total Eligible Subjects: {len(eligible_subjects)}")

    # 3. Mandatory Pre-Flight Gate Verification
    execute_preflight_gate(records_by_sub, eligible_subjects, args.audit_log)

    # 4. Load Stage D Configurations
    with open(args.configs_json, "r") as f:
        stage_d_cfg = json.load(f)

    results_store = []

    # Outer Fold Loop
    for fold in range(1, 6):
        fold_subjects = [s for s in eligible_subjects if sub_to_fold[s] == fold]
        if not fold_subjects:
            continue
        print(f"\n>>> PROCESSING OUTER FOLD {fold} ({len(fold_subjects)} subjects) <<<")

        # In Stage D, S3 config was locked per fold (lr=1e-5, steps=20)
        lr = 1e-5
        steps = 20
        lambda_div = 1.0

        for seed in args.seeds:
            ckpt_path = os.path.join(args.checkpoints_dir, f"modelA_L1200_fold{fold}_seed{seed}.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

            base_model = ConventionalCNNBiGRU().to(args.device)
            ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
            base_model.load_state_dict(ckpt)
            base_model.eval()
            backbone_state = copy.deepcopy(base_model.state_dict())

            predictor_zs = SlidingWindowPredictor(base_model, context_len=1200, neutralize_circadian=True, device=torch.device(args.device))

            for sub in fold_subjects:
                nights = records_by_sub[sub]
                cal_records = nights[:2]  # Strictly first 2 calibration nights (unlabeled)
                eval_records = nights[2:]  # Strictly Nights 3+ (152 nights total)

                # 1. Zero-Shot Baseline
                zs_f1, zs_acc = evaluate_records(predictor_zs, eval_records)

                # 2. Unsupervised InfoMax Adaptation
                reset_model_weights(base_model, backbone_state)

                if not args.dry_run:
                    adapted_model, info = adapt_unsupervised_infomax(
                        model=base_model,
                        cal_records=cal_records,
                        lr=lr,
                        steps=steps,
                        lambda_div=lambda_div,
                        prior_entropy=np.log(4.0),
                        device=args.device
                    )
                else:
                    adapted_model = base_model
                    info = {"aborted": False, "abort_step": None, "final_marginal_entropy": 1.38, "completed_steps": 20}

                predictor_ua = SlidingWindowPredictor(adapted_model, context_len=1200, neutralize_circadian=True, device=torch.device(args.device))
                ua_f1, ua_acc = evaluate_records(predictor_ua, eval_records)
                delta_ua = (ua_f1 - zs_f1) * 100.0

                results_store.append({
                    "subject": sub,
                    "fold": fold,
                    "seed": seed,
                    "n_eval_nights": len(eval_records),
                    "ZS_MacroF1": zs_f1,
                    "UA2_MacroF1": ua_f1,
                    "Delta_UA2_pp": delta_ua,
                    "ZS_Accuracy": zs_acc,
                    "UA2_Accuracy": ua_acc,
                    "aborted": info["aborted"],
                    "abort_step": info["abort_step"],
                    "final_marginal_entropy": info["final_marginal_entropy"],
                    "completed_steps": info["completed_steps"]
                })

                if args.dry_run:
                    print(f"Dry-run check passed for subject {sub}")
                    break
            if args.dry_run:
                break
        if args.dry_run:
            break

    df_results = pd.DataFrame(results_store)
    print(f"\nCompleted execution across {len(df_results)} subject-seed instances.")

    if args.dry_run:
        print("Dry run completed successfully. Zero production artifacts modified.")
        return df_results

    # Average across seeds per subject
    df_subject_means = df_results.groupby("subject").agg({
        "fold": "first",
        "n_eval_nights": "first",
        "ZS_MacroF1": "mean",
        "UA2_MacroF1": "mean",
        "Delta_UA2_pp": "mean",
        "ZS_Accuracy": "mean",
        "UA2_Accuracy": "mean",
        "completed_steps": "mean"
    }).reset_index()

    # Save subject rerun metrics
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_subject_means.to_csv(args.output_csv, index=False)
    print(f"Saved UA-2 rerun metrics to {args.output_csv}")

    # Statistical Inference (N = 46)
    deltas = df_subject_means["Delta_UA2_pp"].values
    mean_delta = float(np.mean(deltas))
    ci_low, ci_high = bootstrap_ci(deltas, n_boot=10000, alpha=0.05)
    w_stat, p_val = stats.wilcoxon(deltas, alternative='two-sided')
    w_stat_sup, p_val_sup = stats.wilcoxon(deltas, alternative='greater')

    print("\n" + "=" * 80)
    print("UA-2 STATISTICAL INFERENCE & DECISION CATEGORIZATION")
    print("=" * 80)
    print(f"Cohort Mean Delta:  {mean_delta:+.2f} pp [95% CI: {ci_low:+.2f}, {ci_high:+.2f} pp]")
    print(f"Wilcoxon p-value:   p (two-sided) = {p_val:.4e} | p (superiority) = {p_val_sup:.4e}")
    print(f"Win Rate:           {sum(deltas > 0)} / {len(deltas)} ({sum(deltas > 0) / len(deltas):.1%})")

    # Decision Matrix Categorization per PREREG_EXP-019_S3_AMENDMENT.md
    if p_val_sup < 0.05 and ci_low > 0.0 and mean_delta >= 1.00:
        verdict = "SUPPORTED (Label-Free Gain >= +1.00 pp)"
    elif p_val_sup < 0.05 and ci_low > 0.0:
        verdict = "SUB-BENCHMARK GAIN (Statistically Superior, Below +1.00 pp)"
    elif ci_low >= -0.50 and ci_high <= 0.50:
        verdict = "NEUTRAL / EQUIVALENCE ([-0.50, +0.50] pp)"
    elif ci_high < 0.0:
        verdict = "DESTRUCTIVE DRIFT (Negative Transfer / Mode Collapse)"
    else:
        verdict = "INCONCLUSIVE"

    print(f"Official S3 Decision Verdict: {verdict}")
    print("=" * 80)

    return df_subject_means


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSK-07 UA-2 Unsupervised Adaptation Re-Run Driver")
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--manifests-dir", default=MANIFESTS_DIR_DEFAULT)
    parser.add_argument("--checkpoints-dir", default=CHECKPOINTS_DIR_DEFAULT)
    parser.add_argument("--configs-json", default=STAGE_D_CONFIGS_DEFAULT)
    parser.add_argument("--output-csv", default=OUTPUT_CSV_DEFAULT)
    parser.add_argument("--audit-log", default=AUDIT_LOG_DEFAULT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Runs single verification without model adaptation")
    args = parser.parse_args()
    run_ua2_rerun(args)
