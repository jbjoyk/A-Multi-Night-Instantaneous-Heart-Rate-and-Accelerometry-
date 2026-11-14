"""
EXP-009: Multi-Seed Stability Evaluation for SleepAdaptiveNet Candidates.

Candidates Evaluated:
  - Candidate 1 (B0_GRL): Base A4, Adaptive Fusion=True, Transition=True, Boundary=False (lambda=0.0), GRL=True
  - Candidate 2 (B2):     Base A4, Adaptive Fusion=True, Transition=True, Boundary=True (lambda=0.20), GRL=True

Seeds:
  - Seed 42:   Loaded directly from EXP-008 (exp008_boundary_sweep_results.pkl)
  - Seed 123:  Executed in EXP-009
  - Seed 2024: Executed in EXP-009

Strict Experimental Controls:
  - Dataset: 31 train subjects (160 usable nights), 5 validation subjects (28 usable nights)
  - Test set (11 subjects, 58 nights): STRICTLY NOT LOADED / NOT USED
  - Sequence length: 600 epochs (5 continuous hours, 0 padding artifacts)
  - Batch size: 4, Optimizer: Adam (lr = 1e-4, weight_decay = 0.0), Epochs: 10
  - Hardware: NVIDIA GeForce RTX 3060
"""

import os
import sys
import glob
import time
import pickle
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sleep_adaptive_net import build_ablation_model
from slamss_ifs_baseline import make_reproducible_split
from slamss_ifs_train import load_night_features
from run_controlled_experiment import ControlledSleepDataset
from run_exp008_boundary_sweep import evaluate_validation_sweep, train_sweep_model


def run_exp009():
    print("=" * 75)
    print("EXP-009: MULTI-SEED STABILITY EVALUATION (SEEDS 42, 123, 2024)")
    print("=" * 75)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(device)})")
    
    # 1. Load Seed 42 baseline results from EXP-008
    exp008_path = "exp008_boundary_sweep_results.pkl"
    assert os.path.exists(exp008_path), f"Missing {exp008_path}!"
    with open(exp008_path, "rb") as f:
        exp008_results = pickle.load(f)
    print(f"Successfully loaded Seed 42 results from {exp008_path}.")
    
    # 2. Setup Data Loading (Train + Val only; NEVER test)
    cache_dir = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
    data_dir = "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    
    mat_files = sorted(glob.glob(os.path.join(data_dir, "Bidslab*", "*", "labels.mat")))
    all_nights = [(f.split(os.sep)[-3], f.split(os.sep)[-2]) for f in mat_files]
    all_subjects = sorted(list(set(s for s, n in all_nights)))
    
    split = make_reproducible_split(all_subjects, split_type="paper", seed=0)
    train_subs = split["train"] # 31 subjects
    val_subs = split["val"]     # 5 subjects
    test_subs = split["test"]   # 11 subjects
    
    assert len(train_subs) == 31
    assert len(val_subs) == 5
    assert len(test_subs) == 11
    
    subject_to_idx = {s: i for i, s in enumerate(train_subs)}
    
    train_records = []
    val_records = []
    for s, n in all_nights:
        if s in train_subs:
            rec = load_night_features(cache_dir, s, n)
            if rec is not None:
                train_records.append(rec)
        elif s in val_subs:
            rec = load_night_features(cache_dir, s, n)
            if rec is not None:
                val_records.append(rec)
        elif s in test_subs:
            # STRICT CONTROL: DO NOT LOAD OR TOUCH TEST DATA
            pass
            
    assert len(train_records) == 160, f"Expected 160 train, got {len(train_records)}"
    assert len(val_records) == 28, f"Expected 28 val, got {len(val_records)}"
    print(f"Loaded 160 usable train nights and 28 usable val nights.")
    print(f"CONFIRMATION: Held-out test set (11 subjects, 58 nights) is STRICTLY EXCLUDED.")
    
    train_ds = ControlledSleepDataset(train_records, subject_to_idx=subject_to_idx, seq_len=600)
    val_ds = ControlledSleepDataset(val_records, subject_to_idx=subject_to_idx, seq_len=600)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    # 3. Setup Candidates
    candidates = [
        ("B0_GRL", "A4", 0.0, False),
        ("B2",     "A4", 0.20, True),
    ]
    
    seeds_to_run = [123, 2024]
    
    # Structure runs dict
    runs = {
        "B0_GRL": {42: exp008_results["B0_GRL"]},
        "B2":     {42: exp008_results["B2"]}
    }
    
    t_start = time.time()
    for name, base_abl, l_b, b_aware in candidates:
        for seed in seeds_to_run:
            print(f"\n>>> Running Candidate: {name} | Seed: {seed} <<<")
            res = train_sweep_model(
                model_name=f"{name}_s{seed}",
                base_ablation=base_abl,
                lambda_boundary=l_b,
                boundary_aware=b_aware,
                train_loader=train_loader,
                val_loader=val_loader,
                n_train_subjects=len(train_subs),
                epochs=10,
                lr=1e-4,
                device=device,
                seed=seed
            )
            runs[name][seed] = res
            
    total_time = time.time() - t_start
    print("\n" + "=" * 75)
    print(f"ALL EXP-009 RUNS COMPLETED IN {total_time:.1f} s ({total_time/60:.2f} min).")
    print("=" * 75)
    
    # 4. Aggregate metrics across seeds (42, 123, 2024)
    all_seeds = [42, 123, 2024]
    
    def extract_metrics(c_name: str) -> Dict[str, Dict[str, any]]:
        metrics_keys = [
            "accuracy", "macro_f1", "kappa",
            "wake_f1", "light_f1", "deep_f1", "rem_f1",
            "wake_recall", "light_recall", "deep_recall", "rem_recall",
            "boundary_error_rate", "non_boundary_error_rate",
            "ece", "mean_entropy"
        ]
        
        raw_by_seed = {s: {} for s in all_seeds}
        agg = {}
        
        for k in metrics_keys:
            vals = [runs[c_name][s]["val_metrics"][k] for s in all_seeds]
            # convert accuracy/recalls/errors to percentage for clarity where standard
            raw_by_seed_vals = {s: runs[c_name][s]["val_metrics"][k] for s in all_seeds}
            for s in all_seeds:
                raw_by_seed[s][k] = raw_by_seed_vals[s]
                
            agg[k] = {
                "values": vals,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals))
            }
            
        # Extract gate alpha stats
        alpha_means = [runs[c_name][s]["val_metrics"]["alpha_stats"]["mean"] for s in all_seeds]
        alpha_stds = [runs[c_name][s]["val_metrics"]["alpha_stats"]["std"] for s in all_seeds]
        alpha_wake = [runs[c_name][s]["val_metrics"]["alpha_by_stage"]["Wake"] for s in all_seeds]
        alpha_deep = [runs[c_name][s]["val_metrics"]["alpha_by_stage"]["Deep"] for s in all_seeds]
        alpha_rem = [runs[c_name][s]["val_metrics"]["alpha_by_stage"]["REM"] for s in all_seeds]
        alpha_light = [runs[c_name][s]["val_metrics"]["alpha_by_stage"]["Light"] for s in all_seeds]
        
        agg["alpha_mean"] = {"values": alpha_means, "mean": float(np.mean(alpha_means)), "std": float(np.std(alpha_means, ddof=1)), "min": float(np.min(alpha_means)), "max": float(np.max(alpha_means))}
        agg["alpha_std"] = {"values": alpha_stds, "mean": float(np.mean(alpha_stds)), "std": float(np.std(alpha_stds, ddof=1)), "min": float(np.min(alpha_stds)), "max": float(np.max(alpha_stds))}
        agg["alpha_wake"] = {"values": alpha_wake, "mean": float(np.mean(alpha_wake)), "std": float(np.std(alpha_wake, ddof=1)), "min": float(np.min(alpha_wake)), "max": float(np.max(alpha_wake))}
        agg["alpha_deep"] = {"values": alpha_deep, "mean": float(np.mean(alpha_deep)), "std": float(np.std(alpha_deep, ddof=1)), "min": float(np.min(alpha_deep)), "max": float(np.max(alpha_deep))}
        agg["alpha_rem"] = {"values": alpha_rem, "mean": float(np.mean(alpha_rem)), "std": float(np.std(alpha_rem, ddof=1)), "min": float(np.min(alpha_rem)), "max": float(np.max(alpha_rem))}
        agg["alpha_light"] = {"values": alpha_light, "mean": float(np.mean(alpha_light)), "std": float(np.std(alpha_light, ddof=1)), "min": float(np.min(alpha_light)), "max": float(np.max(alpha_light))}
        
        # Timing and memory
        vram = [runs[c_name][s]["peak_vram_mb"] for s in all_seeds]
        train_time = [runs[c_name][s]["train_time_s"] for s in all_seeds]
        agg["peak_vram_mb"] = {"values": vram, "mean": float(np.mean(vram)), "std": float(np.std(vram, ddof=1)), "min": float(np.min(vram)), "max": float(np.max(vram))}
        agg["train_time_s"] = {"values": train_time, "mean": float(np.mean(train_time)), "std": float(np.std(train_time, ddof=1)), "min": float(np.min(train_time)), "max": float(np.max(train_time))}
        
        return {"raw_by_seed": raw_by_seed, "aggregated": agg}

    b0_grl_stats = extract_metrics("B0_GRL")
    b2_stats = extract_metrics("B2")
    
    # 5. Paired differences: B2 - B0_GRL per seed
    deltas = {}
    key_comp = ["accuracy", "macro_f1", "kappa", "boundary_error_rate", "non_boundary_error_rate", "wake_f1", "rem_f1", "wake_recall", "rem_recall", "ece"]
    for k in key_comp:
        b0_vals = np.array(b0_grl_stats["aggregated"][k]["values"])
        b2_vals = np.array(b2_stats["aggregated"][k]["values"])
        d = b2_vals - b0_vals
        deltas[k] = {
            "by_seed": {s: float(d[i]) for i, s in enumerate(all_seeds)},
            "mean": float(np.mean(d)),
            "std": float(np.std(d, ddof=1)),
            "min": float(np.min(d)),
            "max": float(np.max(d))
        }
        
    summary_report = {
        "seeds": all_seeds,
        "B0_GRL": b0_grl_stats,
        "B2": b2_stats,
        "deltas_B2_minus_B0_GRL": deltas,
    }
    
    full_output = {
        "runs": runs,
        "summary": summary_report
    }
    
    output_path = "exp009_multiseed_results.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(full_output, f)
    print(f"\nSaved complete results to {output_path}")
    
    # 6. Print Formatted Synthesis Table
    print("\n" + "=" * 90)
    print("EXP-009 MULTI-SEED STABILITY SUMMARY TABLE (N=3 seeds: 42, 123, 2024)")
    print("=" * 90)
    print(f"{'Metric':<25} | {'B0_GRL (Mean +/- Std)':<22} | {'B2 (Mean +/- Std)':<22} | {'Delta (B2 - B0_GRL)':<18}")
    print("-" * 90)
    
    display_keys = [
        ("Accuracy (%)", "accuracy", 100.0, "%"),
        ("Macro-F1", "macro_f1", 1.0, ""),
        ("Cohen's Kappa (κ)", "kappa", 1.0, ""),
        ("Wake F1", "wake_f1", 1.0, ""),
        ("Light F1", "light_f1", 1.0, ""),
        ("Deep F1", "deep_f1", 1.0, ""),
        ("REM F1", "rem_f1", 1.0, ""),
        ("Wake Recall (%)", "wake_recall", 100.0, "%"),
        ("REM Recall (%)", "rem_recall", 100.0, "%"),
        ("Boundary Error (%)", "boundary_error_rate", 100.0, "%"),
        ("Non-Boundary Error (%)", "non_boundary_error_rate", 100.0, "%"),
        ("ECE", "ece", 1.0, ""),
        ("Gate Alpha Mean", "alpha_mean", 1.0, ""),
        ("Gate Alpha (Wake)", "alpha_wake", 1.0, ""),
        ("Gate Alpha (Deep)", "alpha_deep", 1.0, ""),
        ("Gate Alpha (REM)", "alpha_rem", 1.0, ""),
    ]
    
    for label, k, scale, unit in display_keys:
        b0_m = b0_grl_stats["aggregated"][k]["mean"] * scale
        b0_s = b0_grl_stats["aggregated"][k]["std"] * scale
        b2_m = b2_stats["aggregated"][k]["mean"] * scale
        b2_s = b2_stats["aggregated"][k]["std"] * scale
        if k in deltas:
            d_m = deltas[k]["mean"] * scale
            d_s = deltas[k]["std"] * scale
            d_str = f"{d_m:+.2f}{unit} +/- {d_s:.2f}"
        else:
            d_str = f"{b2_m - b0_m:+.3f}"
        print(f"{label:<25} | {b0_m:6.2f}{unit} +/- {b0_s:4.2f}{unit:<3} | {b2_m:6.2f}{unit} +/- {b2_s:4.2f}{unit:<3} | {d_str:<18}")
        
    print("=" * 90)
    return full_output


if __name__ == "__main__":
    run_exp009()
