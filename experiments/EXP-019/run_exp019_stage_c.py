"""
run_exp019_stage_c.py: Stage C Confirmatory Evaluation Driver for EXP-019.

Protocol Governance:
  - Evaluates 46 eligible outer test subjects (152 outer evaluation nights) across 5 folds x 3 seeds.
  - Exactly 5 arms: ZS, FT-2, FT-other-2, PS-2, UA-2.
  - Zero-shot backbone loaded from experiments/EXP-018/checkpoints/modelA_L1200_fold{fold}_seed{seed}.pt.
  - Parameters locked from Stage D output: experiments/EXP-019/exp019_stage_d_selected_configs.json.
  - Strict in-memory deep copy state neutralization: zero weight/optimizer leakage.
  - Statistical testing: 10,000 paired cluster bootstrap 95% CIs and Wilcoxon signed-rank tests.
  - Evaluates H-A (primary), S1 (specificity), S2 (scratch), S3 (unsupervised InfoMax).
  - Exports: experiments/EXP-019/exp019_stage_c_results.json.
"""

import os
import sys
import copy
import json
import time
import argparse
import subprocess
from typing import Dict, List, Tuple, Any
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score, accuracy_score, cohen_kappa_score, matthews_corrcoef, recall_score
)

# Ensure paths
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
exp018_dir = os.path.abspath(os.path.join(repo_root, "experiments", "EXP-018"))
for p in [repo_root, exp018_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from exp013_models import ConventionalCNNBiGRU
from exp018_inference import SlidingWindowPredictor
from slamss_ifs_train import prepare_dataset_records
from evaluator import ClinicalHypnogramEvaluator
from exp019_adaptation_core import (
    reset_model_weights,
    adapt_supervised_ft,
    adapt_unsupervised_infomax,
    train_per_subject_scratch
)

SCOPE_REGEX_MAP = {
    "head_gru": r"(classifier_head|temporal_gru.*weight_.*l1|temporal_gru.*bias_.*l1|temp_norm)",
    "all": r".*"
}


def evaluate_records_full_metrics(predictor: SlidingWindowPredictor, records: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    """Evaluates records and returns complete classification metrics."""
    preds = []
    for r in records:
        pred_res = predictor.predict_night(r)
        preds.append(pred_res)
    all_y_true = np.concatenate([p["y_true"] for p in preds])
    all_y_pred = np.concatenate([p["y_pred"] for p in preds])
    valid = (all_y_true >= 0) & (all_y_true < 4)
    if not np.any(valid):
        return {"macro_f1": 0.0, "accuracy": 0.0, "kappa": 0.0}

    y_t = all_y_true[valid]
    y_p = all_y_pred[valid]

    macro_f1 = float(f1_score(y_t, y_p, average="macro"))
    acc = float(accuracy_score(y_t, y_p))
    kappa = float(cohen_kappa_score(y_t, y_p))
    mcc = float(matthews_corrcoef(y_t, y_p))
    per_class_f1 = [float(v) for v in f1_score(y_t, y_p, average=None)]

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "kappa": kappa,
        "mcc": mcc,
        "per_class_f1": per_class_f1,
        "n_epochs": int(len(y_t))
    }


def compute_bootstrap_ci(deltas: np.ndarray, n_bootstraps: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    """Computes mean and 95% percentile bootstrap confidence interval of paired deltas."""
    rng = np.random.RandomState(seed)
    n = len(deltas)
    mean_val = float(np.mean(deltas))
    boot_means = np.empty(n_bootstraps, dtype=np.float64)
    for b in range(n_bootstraps):
        sample = rng.choice(deltas, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return mean_val, ci_low, ci_high


def select_donor_subject(
    train_subs: List[str],
    sub_records: Dict[str, List[Dict[str, np.ndarray]]],
    target_sub_idx: int
) -> str:
    """
    Selects a deterministic donor subject from training set for FT-other-2.
    Donor must have all 4 stages and >= 300 valid epochs across first 2 calibration nights.
    """
    eligible_donors = []
    for ts in sorted(train_subs):
        if len(sub_records[ts]) < 2:
            continue
        cal = sub_records[ts][:2]
        combined_y = np.concatenate([r["y"] for r in cal])
        valid_y = combined_y[(combined_y >= 0) & (combined_y < 4)]
        if len(valid_y) >= 300 and len(np.unique(valid_y)) == 4:
            eligible_donors.append(ts)

    assert len(eligible_donors) > 0, "No eligible donors found in training fold!"
    return eligible_donors[target_sub_idx % len(eligible_donors)]


def run_stage_c(
    cache_dir: str,
    manifest_dir: str,
    checkpoint_dir: str,
    selected_configs_json: str,
    output_path: str,
    device: str = "cuda:0",
    folds: List[int] = [1, 2, 3, 4, 5],
    seeds: List[int] = [42, 123, 2024],
    dry_run: bool = False
) -> Dict[str, Any]:
    print("=" * 70)
    print("STARTING EXP-019 STAGE C: CONFIRMATORY EVALUATION")
    print(f"Device: {device} | Folds: {folds} | Seeds: {seeds} | Dry Run: {dry_run}")
    print("=" * 70)

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_commit = "unknown"

    with open(selected_configs_json, "r") as f:
        selected_data = json.load(f)
    locked_threshold = float(selected_data.get("confirmed_threshold", 1.5))
    fold_configs = selected_data["folds"]

    # Preload all subjects
    all_needed_subs = set()
    for f in folds:
        mpath = os.path.join(manifest_dir, f"fold_{f}.json")
        with open(mpath, "r") as mfile:
            m = json.load(mfile)
            all_needed_subs.update(m["outer_test_subjects"])
            all_needed_subs.update(m["inner_train_subjects"])

    print(f"Loading records for {len(all_needed_subs)} subjects...")
    records_dict = prepare_dataset_records(cache_dir, {"train": list(all_needed_subs), "val": [], "test": []})
    sub_records = {s: [] for s in all_needed_subs}
    for r in records_dict["train"]:
        sub_records[r["subject"]].append(r)

    for s in sub_records:
        sub_records[s] = sorted(sub_records[s], key=lambda x: int(x["night"]))

    # Subject result aggregator
    per_subject_results = []
    arm_names = ["ZS", "FT-2", "FT-other-2", "PS-2", "UA-2"]
    total_eval_nights = 0

    for fold in folds:
        print(f"\n>>> PROCESSING OUTER TEST FOLD {fold}/5 <<<")
        mpath = os.path.join(manifest_dir, f"fold_{fold}.json")
        with open(mpath, "r") as mfile:
            manifest = json.load(mfile)

        outer_test_subs = sorted(manifest["outer_test_subjects"])
        inner_train_subs = sorted(manifest["inner_train_subjects"])
        cfgs = fold_configs[str(fold)]

        ft_cfg = cfgs["FT-2"]
        ps_cfg = cfgs["PS-2"]
        ua_cfg = cfgs["UA-2"]

        # Filter out ineligible subjects (n_nights < 3, e.g. Bidslab49 if ever in test)
        eligible_test_subs = [s for s in outer_test_subs if len(sub_records[s]) >= 3]
        print(f"Eligible outer test subjects ({len(eligible_test_subs)}): {eligible_test_subs}")

        # Check all checkpoints exist for this fold
        for s in seeds:
            ckpt = os.path.join(checkpoint_dir, f"modelA_L1200_fold{fold}_seed{s}.pt")
            assert os.path.exists(ckpt), f"Missing required checkpoint: {ckpt}"

        for sub_idx, sub in enumerate(eligible_test_subs):
            nights = sub_records[sub]
            cal_nights = nights[:2]
            eval_nights = nights[2:]
            total_eval_nights += len(eval_nights)

            # Select donor for FT-other-2
            donor_sub = select_donor_subject(inner_train_subs, sub_records, sub_idx)
            donor_cal_nights = sub_records[donor_sub][:2]

            sub_arm_metrics = {arm: [] for arm in arm_names}

            for seed in seeds:
                ckpt_path = os.path.join(checkpoint_dir, f"modelA_L1200_fold{fold}_seed{seed}.pt")
                backbone_state = torch.load(ckpt_path, map_location=device)

                base_model = ConventionalCNNBiGRU(
                    d_model=64, d_temp=96, n_classes=4, n_meta=5, dropout=0.2, use_rcc_loss=True
                ).to(device)

                # 1. Arm 1: ZS
                reset_model_weights(base_model, backbone_state)
                base_model.eval()
                zs_pred = SlidingWindowPredictor(model=base_model, context_len=1200, neutralize_circadian=True, device=torch.device(device))
                sub_arm_metrics["ZS"].append(evaluate_records_full_metrics(zs_pred, eval_nights)["macro_f1"])

                # 2. Arm 2: FT-2
                reset_model_weights(base_model, backbone_state)
                adapted_ft = adapt_supervised_ft(
                    model=base_model,
                    cal_records=cal_nights,
                    lr=ft_cfg["lr"],
                    epochs=ft_cfg["epochs"],
                    scope_regex=SCOPE_REGEX_MAP[ft_cfg["scope"]],
                    device=device
                )
                pred_ft = SlidingWindowPredictor(model=adapted_ft, context_len=1200, neutralize_circadian=True, device=torch.device(device))
                sub_arm_metrics["FT-2"].append(evaluate_records_full_metrics(pred_ft, eval_nights)["macro_f1"])

                # 3. Arm 3: FT-other-2
                reset_model_weights(base_model, backbone_state)
                adapted_fto = adapt_supervised_ft(
                    model=base_model,
                    cal_records=donor_cal_nights,
                    lr=ft_cfg["lr"],
                    epochs=ft_cfg["epochs"],
                    scope_regex=SCOPE_REGEX_MAP[ft_cfg["scope"]],
                    device=device
                )
                pred_fto = SlidingWindowPredictor(model=adapted_fto, context_len=1200, neutralize_circadian=True, device=torch.device(device))
                sub_arm_metrics["FT-other-2"].append(evaluate_records_full_metrics(pred_fto, eval_nights)["macro_f1"])

                # 4. Arm 4: PS-2
                scratch_model = train_per_subject_scratch(
                    cal_records=cal_nights,
                    lr=ps_cfg["lr"],
                    steps=ps_cfg["steps"],
                    device=device
                )
                pred_ps = SlidingWindowPredictor(model=scratch_model, context_len=1200, neutralize_circadian=True, device=torch.device(device))
                sub_arm_metrics["PS-2"].append(evaluate_records_full_metrics(pred_ps, eval_nights)["macro_f1"])

                # 5. Arm 5: UA-2
                reset_model_weights(base_model, backbone_state)
                adapted_ua, _ = adapt_unsupervised_infomax(
                    model=base_model,
                    cal_records=cal_nights,
                    lr=ua_cfg["lr"],
                    steps=ua_cfg["steps"],
                    device=device
                )
                pred_ua = SlidingWindowPredictor(model=adapted_ua, context_len=1200, neutralize_circadian=True, device=torch.device(device))
                sub_arm_metrics["UA-2"].append(evaluate_records_full_metrics(pred_ua, eval_nights)["macro_f1"])

            # Compute seed-averaged Macro-F1 per arm
            sub_summary = {
                "subject_id": sub,
                "fold": fold,
                "n_eval_nights": len(eval_nights),
                "eval_nights": [int(r["night"]) for r in eval_nights],
                "donor_sub": donor_sub
            }
            for arm in arm_names:
                sub_summary[arm] = round(float(np.mean(sub_arm_metrics[arm])), 4)
            sub_summary["delta_HA"] = round((sub_summary["FT-2"] - sub_summary["ZS"]) * 100.0, 4)
            sub_summary["delta_S1"] = round((sub_summary["FT-2"] - sub_summary["FT-other-2"]) * 100.0, 4)
            sub_summary["delta_S2"] = round((sub_summary["FT-2"] - sub_summary["PS-2"]) * 100.0, 4)
            sub_summary["delta_S3"] = round((sub_summary["UA-2"] - sub_summary["ZS"]) * 100.0, 4)

            per_subject_results.append(sub_summary)
            print(f"  [{sub}] ZS: {sub_summary['ZS']:.4f} | FT-2: {sub_summary['FT-2']:.4f} | Delta H-A: {sub_summary['delta_HA']:+.2f} pp")

    # Cohort census verification
    assert len(per_subject_results) == 46, f"Expected exactly 46 subjects, got {len(per_subject_results)}"
    assert total_eval_nights == 152, f"Expected exactly 152 evaluation nights, got {total_eval_nights}"

    # Extract paired deltas across the 46 subjects
    deltas_ha = np.array([s["delta_HA"] for s in per_subject_results])
    deltas_s1 = np.array([s["delta_S1"] for s in per_subject_results])
    deltas_s2 = np.array([s["delta_S2"] for s in per_subject_results])
    deltas_s3 = np.array([s["delta_S3"] for s in per_subject_results])

    # 1. H-A Testing
    ha_mean, ha_ci_low, ha_ci_high = compute_bootstrap_ci(deltas_ha)
    ha_wilcoxon = float(stats.wilcoxon(deltas_ha, alternative="greater").pvalue)
    ha_supported = bool((ha_mean >= locked_threshold) and (ha_ci_low > 0.0))

    # 2. S1 Testing (Specificity: FT-2 > FT-other-2)
    s1_mean, s1_ci_low, s1_ci_high = compute_bootstrap_ci(deltas_s1)
    s1_wilcoxon = float(stats.wilcoxon(deltas_s1, alternative="greater").pvalue)
    s1_supported = bool((s1_ci_low > 0.0) and (s1_wilcoxon < 0.05))

    # 3. S2 Testing (Transfer Value: FT-2 > PS-2)
    s2_mean, s2_ci_low, s2_ci_high = compute_bootstrap_ci(deltas_s2)
    s2_wilcoxon = float(stats.wilcoxon(deltas_s2, alternative="greater").pvalue)
    s2_supported = bool((s2_ci_low > 0.0) and (s2_wilcoxon < 0.05))

    # 4. S3 Testing (Unsupervised InfoMax: UA-2 > ZS)
    s3_mean, s3_ci_low, s3_ci_high = compute_bootstrap_ci(deltas_s3)
    s3_wilcoxon = float(stats.wilcoxon(deltas_s3, alternative="greater").pvalue)
    s3_supported = bool((s3_ci_low > 0.0) and (s3_wilcoxon < 0.05))

    # Summary by arm
    summary_by_arm = {}
    for arm in arm_names:
        scores = [s[arm] for s in per_subject_results]
        summary_by_arm[arm] = {
            "macro_f1_mean": round(float(np.mean(scores)), 4),
            "macro_f1_std": round(float(np.std(scores, ddof=1)), 4)
        }

    output_results = {
        "metadata": {
            "git_commit": git_commit,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_subjects": 46,
            "n_eval_nights": total_eval_nights,
            "locked_threshold": locked_threshold,
            "device": str(device)
        },
        "hypotheses": {
            "H-A": {
                "delta_mean": round(ha_mean, 4),
                "ci_95": [round(ha_ci_low, 4), round(ha_ci_high, 4)],
                "wilcoxon_p": round(ha_wilcoxon, 6),
                "threshold": locked_threshold,
                "supported": ha_supported
            },
            "S1": {
                "delta_mean": round(s1_mean, 4),
                "ci_95": [round(s1_ci_low, 4), round(s1_ci_high, 4)],
                "wilcoxon_p": round(s1_wilcoxon, 6),
                "supported": s1_supported
            },
            "S2": {
                "delta_mean": round(s2_mean, 4),
                "ci_95": [round(s2_ci_low, 4), round(s2_ci_high, 4)],
                "wilcoxon_p": round(s2_wilcoxon, 6),
                "supported": s2_supported
            },
            "S3": {
                "delta_mean": round(s3_mean, 4),
                "ci_95": [round(s3_ci_low, 4), round(s3_ci_high, 4)],
                "wilcoxon_p": round(s3_wilcoxon, 6),
                "supported": s3_supported
            }
        },
        "summary_by_arm": summary_by_arm,
        "per_subject_results": per_subject_results
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_results, f, indent=2)
    print(f"\nStage C results saved to {output_path}")

    print("\n" + "=" * 70)
    print("EXP-019 STAGE C FINAL HYPOTHESIS VERDICTS")
    print("=" * 70)
    print(f"H-A (FT-2 vs ZS): Delta = {ha_mean:+.2f} pp [95% CI: {ha_ci_low:+.2f}, {ha_ci_high:+.2f}] | p = {ha_wilcoxon:.5f} | Threshold: {locked_threshold:.2f} pp -> {'SUPPORTED' if ha_supported else 'NOT SUPPORTED'}")
    print(f"S1  (FT-2 vs FTO): Delta = {s1_mean:+.2f} pp [95% CI: {s1_ci_low:+.2f}, {s1_ci_high:+.2f}] | p = {s1_wilcoxon:.5f} -> {'SUPPORTED' if s1_supported else 'NOT SUPPORTED'}")
    print(f"S2  (FT-2 vs PS-2): Delta = {s2_mean:+.2f} pp [95% CI: {s2_ci_low:+.2f}, {s2_ci_high:+.2f}] | p = {s2_wilcoxon:.5f} -> {'SUPPORTED' if s2_supported else 'NOT SUPPORTED'}")
    print(f"S3  (UA-2 vs ZS): Delta = {s3_mean:+.2f} pp [95% CI: {s3_ci_low:+.2f}, {s3_ci_high:+.2f}] | p = {s3_wilcoxon:.5f} -> {'SUPPORTED' if s3_supported else 'NOT SUPPORTED'}")

    return output_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP-019 Stage C Confirmatory Driver")
    parser.add_argument("--cache-dir", type=str,
                        default="/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b")
    parser.add_argument("--manifest-dir", type=str, default="experiments/EXP-014/EXP014_MANIFESTS")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/EXP-018/checkpoints")
    parser.add_argument("--selected-configs", type=str, default="experiments/EXP-019/exp019_stage_d_selected_configs.json")
    parser.add_argument("--output-json", type=str, default="experiments/EXP-019/exp019_stage_c_results.json")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Folds to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024], help="Random seeds")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true", help="Run quick dry run")
    args = parser.parse_args()

    run_stage_c(
        cache_dir=args.cache_dir,
        manifest_dir=args.manifest_dir,
        checkpoint_dir=args.checkpoint_dir,
        selected_configs_json=args.selected_configs,
        output_path=args.output_json,
        device=args.device,
        folds=args.folds,
        seeds=args.seeds,
        dry_run=args.dry_run
    )
