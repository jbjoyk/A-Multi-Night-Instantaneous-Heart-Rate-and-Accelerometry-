"""
run_exp019_stage_d.py: Nested Inner-Validation Hyperparameter Selection for EXP-019.

Protocol Governance:
  - Restricted STRICTLY to the 5 inner-validation subjects per fold (25 subjects across 5 folds).
  - ZERO access to outer test subjects.
  - Evaluates pre-declared candidate grids:
      FT-2 (12 configs): LR in {3e-5, 1e-4, 3e-4} x epochs in {5, 15} x scope in {"head_gru", "all"}
      PS-2 (4 configs): LR in {1e-4, 1e-3} x steps in {300, 500}
      UA-2 (4 configs): LR in {1e-5, 1e-4} x steps in {20, 100}
  - Macro-F1 evaluated on inner-val evaluation nights E (k >= 1) after calibration on C (k=2).
  - 4-decimal rounded tie-breaker:
      (1) higher Macro-F1 -> (2) lower LR -> (3) fewer epochs/steps -> (4) fewer parameters.
  - Computes empirical standard deviation (s_delta) of paired differences (FT-2 - ZS) across 25 subjects.
  - Confirmed threshold = max(1.5, 0.4131 * s_delta).
  - Exports: experiments/EXP-019/exp019_stage_d_selected_configs.json.
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
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

# Ensure paths
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
exp018_dir = os.path.abspath(os.path.join(repo_root, "experiments", "EXP-018"))
for p in [repo_root, exp018_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from exp013_models import ConventionalCNNBiGRU
from exp018_inference import SlidingWindowPredictor
from slamss_ifs_train import prepare_dataset_records
from exp019_adaptation_core import (
    reset_model_weights,
    adapt_supervised_ft,
    adapt_unsupervised_infomax,
    train_per_subject_scratch
)

# Pre-declared candidate grids
FT_GRID = [
    {"lr": 3e-5, "epochs": 5, "scope": "head_gru"},
    {"lr": 3e-5, "epochs": 5, "scope": "all"},
    {"lr": 3e-5, "epochs": 15, "scope": "head_gru"},
    {"lr": 3e-5, "epochs": 15, "scope": "all"},
    {"lr": 1e-4, "epochs": 5, "scope": "head_gru"},
    {"lr": 1e-4, "epochs": 5, "scope": "all"},
    {"lr": 1e-4, "epochs": 15, "scope": "head_gru"},
    {"lr": 1e-4, "epochs": 15, "scope": "all"},
    {"lr": 3e-4, "epochs": 5, "scope": "head_gru"},
    {"lr": 3e-4, "epochs": 5, "scope": "all"},
    {"lr": 3e-4, "epochs": 15, "scope": "head_gru"},
    {"lr": 3e-4, "epochs": 15, "scope": "all"},
]

PS_GRID = [
    {"lr": 1e-4, "steps": 300},
    {"lr": 1e-4, "steps": 500},
    {"lr": 1e-3, "steps": 300},
    {"lr": 1e-3, "steps": 500},
]

UA_GRID = [
    {"lr": 1e-5, "steps": 20},
    {"lr": 1e-5, "steps": 100},
    {"lr": 1e-4, "steps": 20},
    {"lr": 1e-4, "steps": 100},
]

SCOPE_REGEX_MAP = {
    "head_gru": r"(classifier_head|temporal_gru.*weight_.*l1|temporal_gru.*bias_.*l1|temp_norm)",
    "all": r".*"
}


def evaluate_records_macro_f1(predictor: SlidingWindowPredictor, records: List[Dict[str, np.ndarray]]) -> float:
    """Evaluates SlidingWindowPredictor across records and returns Macro-F1 on valid epochs [0..3]."""
    preds = []
    for r in records:
        pred_res = predictor.predict_night(r)
        preds.append(pred_res)
    all_y_true = np.concatenate([p["y_true"] for p in preds])
    all_y_pred = np.concatenate([p["y_pred"] for p in preds])
    valid = (all_y_true >= 0) & (all_y_true < 4)
    if not np.any(valid):
        return 0.0
    return float(f1_score(all_y_true[valid], all_y_pred[valid], average="macro"))


def select_best_config(results: List[Dict[str, Any]], arm_type: str) -> Dict[str, Any]:
    """
    Applies the strict 4-decimal rounded tie-breaking hierarchy:
      1. Higher rounded Macro-F1
      2. Lower learning rate
      3. Fewer epochs/steps
      4. Fewer parameters ('head_gru' before 'all')
    """
    def sort_key(item):
        cfg = item["config"]
        score = round(item["mean_macro_f1"], 4)
        lr = cfg["lr"]
        steps = cfg.get("epochs", cfg.get("steps", 0))
        scope_penalty = 1 if cfg.get("scope") == "all" else 0
        # For sorting: (-score, lr, steps, scope_penalty)
        return (-score, lr, steps, scope_penalty)

    sorted_results = sorted(results, key=sort_key)
    best = sorted_results[0]
    out = copy.deepcopy(best["config"])
    out["val_macro_f1"] = round(best["mean_macro_f1"], 4)
    return out


def run_stage_d(
    cache_dir: str,
    manifest_dir: str,
    checkpoint_dir: str,
    output_path: str,
    device: str = "cuda:0",
    folds: List[int] = [1, 2, 3, 4, 5],
    seeds: List[int] = [42, 123, 2024]
) -> Dict[str, Any]:
    print("=" * 70)
    print("STARTING EXP-019 STAGE D: NESTED INNER-VALIDATION SELECTION")
    print(f"Device: {device} | Folds: {folds} | Seeds: {seeds}")
    print("=" * 70)

    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        git_commit = "unknown"

    # Preload all records partitioned by subject
    all_manifest_subs = set()
    for f in folds:
        mpath = os.path.join(manifest_dir, f"fold_{f}.json")
        with open(mpath, "r") as mfile:
            m = json.load(mfile)
            all_manifest_subs.update(m["inner_val_subjects"])

    print(f"Loading records for {len(all_manifest_subs)} inner-validation subjects...")
    records_dict = prepare_dataset_records(cache_dir, {"train": list(all_manifest_subs), "val": [], "test": []})
    sub_records = {s: [] for s in all_manifest_subs}
    for r in records_dict["train"]:
        sub_records[r["subject"]].append(r)

    # Sort each subject's nights chronologically by night number
    for s in sub_records:
        sub_records[s] = sorted(sub_records[s], key=lambda x: int(x["night"]))

    fold_selected = {}
    all_inner_val_deltas = []  # Paired deltas (FT-2 - ZS) in percentage points across 25 subjects

    for fold in folds:
        print(f"\n>>> PROCESSING INNER-VALIDATION FOR FOLD {fold}/5 <<<")
        mpath = os.path.join(manifest_dir, f"fold_{fold}.json")
        with open(mpath, "r") as mfile:
            manifest = json.load(mfile)

        inner_val_subs = manifest["inner_val_subjects"]
        print(f"Inner-val subjects ({len(inner_val_subs)}): {inner_val_subs}")

        # Check all checkpoints exist for this fold
        for s in seeds:
            ckpt = os.path.join(checkpoint_dir, f"modelA_L1200_fold{fold}_seed{s}.pt")
            assert os.path.exists(ckpt), f"Missing required checkpoint: {ckpt}"

        # Evaluate candidate configs across the 5 inner-val subjects and 3 seeds
        ft_scores = {i: [] for i in range(len(FT_GRID))}
        ps_scores = {i: [] for i in range(len(PS_GRID))}
        ua_scores = {i: [] for i in range(len(UA_GRID))}
        zs_scores_by_sub = {sub: [] for sub in inner_val_subs}
        ft_scores_by_sub_and_cfg = {i: {sub: [] for sub in inner_val_subs} for i in range(len(FT_GRID))}

        for seed in seeds:
            ckpt_path = os.path.join(checkpoint_dir, f"modelA_L1200_fold{fold}_seed{seed}.pt")
            backbone_state = torch.load(ckpt_path, map_location=device)

            base_model = ConventionalCNNBiGRU(
                d_model=64, d_temp=96, n_classes=4, n_meta=5, dropout=0.2, use_rcc_loss=True
            ).to(device)

            for sub in inner_val_subs:
                nights = sub_records[sub]
                assert len(nights) >= 3, f"Inner-val subject {sub} has fewer than 3 nights: {len(nights)}"
                cal_nights = nights[:2]
                eval_nights = nights[2:]

                # 1. Zero-Shot Baseline (ZS)
                reset_model_weights(base_model, backbone_state)
                base_model.eval()
                zs_pred = SlidingWindowPredictor(model=base_model, context_len=1200, neutralize_circadian=True, device=torch.device(device))
                zs_f1 = evaluate_records_macro_f1(zs_pred, eval_nights)
                zs_scores_by_sub[sub].append(zs_f1)

                # 2. Evaluate FT-2 Grid
                for cfg_idx, cfg in enumerate(FT_GRID):
                    reset_model_weights(base_model, backbone_state)
                    adapted = adapt_supervised_ft(
                        model=base_model,
                        cal_records=cal_nights,
                        lr=cfg["lr"],
                        epochs=cfg["epochs"],
                        scope_regex=SCOPE_REGEX_MAP[cfg["scope"]],
                        device=device
                    )
                    pred = SlidingWindowPredictor(model=adapted, context_len=1200, neutralize_circadian=True, device=torch.device(device))
                    f1 = evaluate_records_macro_f1(pred, eval_nights)
                    ft_scores[cfg_idx].append(f1)
                    ft_scores_by_sub_and_cfg[cfg_idx][sub].append(f1)

                # 3. Evaluate PS-2 Grid
                for cfg_idx, cfg in enumerate(PS_GRID):
                    scratch_model = train_per_subject_scratch(
                        cal_records=cal_nights,
                        lr=cfg["lr"],
                        steps=cfg["steps"],
                        device=device
                    )
                    pred = SlidingWindowPredictor(model=scratch_model, context_len=1200, neutralize_circadian=True, device=torch.device(device))
                    f1 = evaluate_records_macro_f1(pred, eval_nights)
                    ps_scores[cfg_idx].append(f1)

                # 4. Evaluate UA-2 Grid
                for cfg_idx, cfg in enumerate(UA_GRID):
                    reset_model_weights(base_model, backbone_state)
                    adapted, _ = adapt_unsupervised_infomax(
                        model=base_model,
                        cal_records=cal_nights,
                        lr=cfg["lr"],
                        steps=cfg["steps"],
                        device=device
                    )
                    pred = SlidingWindowPredictor(model=adapted, context_len=1200, neutralize_circadian=True, device=torch.device(device))
                    f1 = evaluate_records_macro_f1(pred, eval_nights)
                    ua_scores[cfg_idx].append(f1)

        # Summarize & Select best configs for this fold
        ft_results = [{"config": FT_GRID[i], "mean_macro_f1": float(np.mean(ft_scores[i]))} for i in range(len(FT_GRID))]
        ps_results = [{"config": PS_GRID[i], "mean_macro_f1": float(np.mean(ps_scores[i]))} for i in range(len(PS_GRID))]
        ua_results = [{"config": UA_GRID[i], "mean_macro_f1": float(np.mean(ua_scores[i]))} for i in range(len(UA_GRID))]

        best_ft = select_best_config(ft_results, "FT-2")
        best_ps = select_best_config(ps_results, "PS-2")
        best_ua = select_best_config(ua_results, "UA-2")

        # Find best FT config index
        best_ft_idx = None
        for i, cfg in enumerate(FT_GRID):
            if cfg["lr"] == best_ft["lr"] and cfg["epochs"] == best_ft["epochs"] and cfg["scope"] == best_ft["scope"]:
                best_ft_idx = i
                break

        print(f"  Fold {fold} Selected FT-2: {best_ft}")
        print(f"  Fold {fold} Selected PS-2: {best_ps}")
        print(f"  Fold {fold} Selected UA-2: {best_ua}")

        fold_selected[str(fold)] = {
            "FT-2": best_ft,
            "PS-2": best_ps,
            "UA-2": best_ua
        }

        # Compute paired differences on the 5 inner-val subjects for this fold
        for sub in inner_val_subs:
            mean_zs = float(np.mean(zs_scores_by_sub[sub]))
            mean_ft = float(np.mean(ft_scores_by_sub_and_cfg[best_ft_idx][sub]))
            delta_pp = (mean_ft - mean_zs) * 100.0  # in percentage points
            all_inner_val_deltas.append(delta_pp)

    # Compute empirical standard deviation s_delta across all 25 inner-val subjects
    assert len(all_inner_val_deltas) == 25, f"Expected 25 inner-val deltas, got {len(all_inner_val_deltas)}"
    empirical_s_delta = float(np.std(all_inner_val_deltas, ddof=1))
    continuous_power_threshold = 0.4131 * empirical_s_delta
    confirmed_threshold = float(max(1.5, continuous_power_threshold))

    print("\n" + "=" * 70)
    print(f"STAGE D COMPLETE: Empirical s_delta = {empirical_s_delta:.4f} pp")
    print(f"Continuous Power Threshold (0.4131 * s_delta) = {continuous_power_threshold:.4f} pp")
    print(f"Confirmed Practical Threshold = max(1.5, {continuous_power_threshold:.4f}) = {confirmed_threshold:.4f} pp")
    print("=" * 70)

    output_data = {
        "metadata": {
            "git_commit": git_commit,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(device),
            "n_inner_val_subjects": 25,
            "folds": folds
        },
        "empirical_s_delta": round(empirical_s_delta, 4),
        "continuous_power_threshold": round(continuous_power_threshold, 4),
        "confirmed_threshold": round(confirmed_threshold, 4),
        "all_inner_val_deltas": [round(d, 4) for d in all_inner_val_deltas],
        "folds": fold_selected
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Selected configurations saved to {output_path}")

    return output_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EXP-019 Stage D Selection")
    parser.add_argument("--cache-dir", type=str,
                        default="/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b")
    parser.add_argument("--manifest-dir", type=str, default="experiments/EXP-014/EXP014_MANIFESTS")
    parser.add_argument("--checkpoint-dir", type=str, default="experiments/EXP-018/checkpoints")
    parser.add_argument("--output-json", type=str, default="experiments/EXP-019/exp019_stage_d_selected_configs.json")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Folds to run")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024], help="Random seeds")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_stage_d(
        cache_dir=args.cache_dir,
        manifest_dir=args.manifest_dir,
        checkpoint_dir=args.checkpoint_dir,
        output_path=args.output_json,
        device=args.device,
        folds=args.folds,
        seeds=args.seeds
    )
