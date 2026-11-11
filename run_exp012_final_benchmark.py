"""
EXP-012: FINAL FROZEN TEST BENCHMARK FOR SLEEPADAPTIVENET (B0_GRL).

Final Frozen Specification:
  - Model: SleepAdaptiveNet (B0_GRL, 162,325 parameters)
  - Components:
      - Adaptive Modality Fusion: ON
      - Transition Module: ON (lambda_trans = 0.10)
      - Boundary Loss: OFF (boundary_aware = False, lambda_boundary = 0.0)
      - Subject-Invariant GRL: ON (lambda_adv = 0.10, 31 train subjects)
      - Uncertainty Head: OFF
  - Loss: RCCLoss (CE + 0.01 * RW with asymmetric matrix)
  - Optimizer: Adam (lr = 1e-4, weight_decay = 0.0)
  - Training Budget: EXACTLY 80 EPOCHS PER SEED (Frozen from EXP-011 convergence window)
  - Seeds: 42, 123, 2024
  - Batch Size: 4, Sequence Length: T = 600 epochs (5 continuous hours, 0 padding)
  - Checkpoints: exp012_seed42_final.pt, exp012_seed123_final.pt, exp012_seed2024_final.pt

Strict Protocol:
  - Train Partition: 31 subjects, 160 usable nights
  - Val Partition:   5 subjects, 28 usable nights
  - Test Partition:  11 subjects, 58 usable nights (STRICTLY UNTOUCHED UNTIL EPOCH 80 COMPLETES)
"""

import os
import sys
import glob
import time
import json
import pickle
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, precision_recall_fscore_support
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sleep_adaptive_net import build_ablation_model
from slamss_ifs_baseline import make_reproducible_split
from slamss_ifs_train import load_night_features
from run_controlled_experiment import ControlledSleepDataset, compute_ece
from run_exp008_boundary_sweep import evaluate_validation_sweep


def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, any]:
    """Evaluates the frozen epoch-80 model on the complete official held-out test partition."""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_boundaries = []
    all_probs = []
    all_subjects = []
    all_nights = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_ihr = batch["ihr"].to(device)
            x_acc = batch["accel"].to(device)
            x_meta = batch["meta"].to(device)
            targets = batch["y"].to(device)
            mask = batch["mask"].to(device)
            # Test subjects have subject_idx = -100 (filtered from discriminator)
            subj_idx = batch["subject_idx"].to(device)
            
            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subj_idx)
            logits = out["logits"]
            probs = out["probabilities"]
            preds = torch.argmax(logits, dim=-1)
            
            # Boundary calculation (y_t != y_{t-1})
            y_prev = targets[:, :-1]
            y_curr = targets[:, 1:]
            is_bound = torch.zeros_like(targets, dtype=torch.bool)
            is_bound[:, 1:] = (y_prev != y_curr) & (y_prev >= 0) & (y_curr >= 0)
            
            valid = (targets >= 0) & (targets < 4) & mask
            
            # Batch metadata
            subjs = batch["subject"]
            nights = batch["night"]
            
            # Store per epoch
            for b_i in range(len(subjs)):
                s_name = subjs[b_i]
                n_name = nights[b_i]
                m_b = valid[b_i].cpu().numpy()
                
                p_b = preds[b_i][m_b].cpu().numpy()
                t_b = targets[b_i][m_b].cpu().numpy()
                bnd_b = is_bound[b_i][m_b].cpu().numpy()
                prb_b = probs[b_i][m_b].cpu().numpy()
                
                all_preds.extend(p_b)
                all_targets.extend(t_b)
                all_boundaries.extend(bnd_b)
                all_probs.append(prb_b)
                all_subjects.extend([s_name] * len(p_b))
                all_nights.extend([f"{s_name}_{n_name}"] * len(p_b))
                
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    is_bound = np.array(all_boundaries, dtype=bool)
    probs_concat = np.concatenate(all_probs, axis=0)
    subjects_arr = np.array(all_subjects)
    
    # 1. Overall Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    ece = compute_ece(probs_concat, y_true, n_bins=10)
    
    # 2. Per-Class Metrics: 0: Wake, 1: Light, 2: Deep, 3: REM
    prec, rec, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    # 3. Boundary vs Non-Boundary Error
    is_correct = (y_true == y_pred)
    err_boundary = float(1.0 - is_correct[is_bound].mean()) if is_bound.sum() > 0 else 0.0
    err_non_boundary = float(1.0 - is_correct[~is_bound].mean()) if (~is_bound).sum() > 0 else 0.0
    
    # 4. Subject-Level Analysis
    unique_subs = sorted(list(set(all_subjects)))
    subject_metrics = []
    subject_cms = {}
    
    for s_id in unique_subs:
        idx_s = (subjects_arr == s_id)
        yt_s = y_true[idx_s]
        yp_s = y_pred[idx_s]
        cm_s = confusion_matrix(yt_s, yp_s, labels=[0, 1, 2, 3])
        subject_cms[s_id] = cm_s
        
        acc_s = accuracy_score(yt_s, yp_s)
        mf1_s = f1_score(yt_s, yp_s, average="macro", zero_division=0)
        try:
            kap_s = cohen_kappa_score(yt_s, yp_s)
        except Exception:
            kap_s = np.nan
            
        subject_metrics.append({
            "subject_id": s_id,
            "n_windows": int(len(yt_s)),
            "accuracy": float(acc_s),
            "macro_f1": float(mf1_s),
            "kappa": float(kap_s)
        })
        
    # 5. Subject-Level Bootstrap 95% Confidence Intervals (B = 1000)
    rng = np.random.default_rng(0)
    B = 1000
    n_subs = len(unique_subs)
    
    boot_accs = []
    boot_macro_f1s = []
    boot_kappas = []
    
    for _ in range(B):
        resampled_subs = rng.choice(unique_subs, size=n_subs, replace=True)
        # Sum confusion matrices of resampled subjects
        pooled_cm = sum(subject_cms[s] for s in resampled_subs)
        total_n = pooled_cm.sum()
        
        # Accuracy from pooled CM
        b_acc = np.diag(pooled_cm).sum() / total_n
        
        # Macro-F1 from pooled CM
        f1_classes = []
        for c in range(4):
            tp = pooled_cm[c, c]
            fp = pooled_cm[:, c].sum() - tp
            fn = pooled_cm[c, :].sum() - tp
            denom = 2 * tp + fp + fn
            f1_classes.append((2 * tp / denom) if denom > 0 else 0.0)
        b_mf1 = np.mean(f1_classes)
        
        # Cohen's Kappa from pooled CM
        po = b_acc
        pe = (pooled_cm.sum(axis=1) * pooled_cm.sum(axis=0)).sum() / (total_n ** 2)
        b_kap = (po - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0
        
        boot_accs.append(b_acc)
        boot_macro_f1s.append(b_mf1)
        boot_kappas.append(b_kap)
        
    ci_acc = (float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5)))
    ci_mf1 = (float(np.percentile(boot_macro_f1s, 2.5)), float(np.percentile(boot_macro_f1s, 97.5)))
    ci_kap = (float(np.percentile(boot_kappas, 2.5)), float(np.percentile(boot_kappas, 97.5)))
    
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "kappa": float(kappa),
        "mcc": float(mcc),
        "ece": float(ece),
        "wake_f1": float(f1_per_class[0]),
        "light_f1": float(f1_per_class[1]),
        "deep_f1": float(f1_per_class[2]),
        "rem_f1": float(f1_per_class[3]),
        "wake_recall": float(rec[0]),
        "light_recall": float(rec[1]),
        "deep_recall": float(rec[2]),
        "rem_recall": float(rec[3]),
        "precision": [float(p) for p in prec],
        "recall": [float(r) for r in rec],
        "f1": [float(f) for f in f1_per_class],
        "support": [int(s) for s in support],
        "boundary_error_rate": err_boundary,
        "non_boundary_error_rate": err_non_boundary,
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "subject_metrics": subject_metrics,
        "bootstrap_ci": {
            "accuracy_95ci": ci_acc,
            "macro_f1_95ci": ci_mf1,
            "kappa_95ci": ci_kap
        },
        "y_true": y_true,
        "y_pred": y_pred,
        "subjects": subjects_arr
    }


def run_exp012():
    print("=" * 85)
    print("EXP-012: FINAL FROZEN TEST BENCHMARK FOR SLEEPADAPTIVENET (B0_GRL)")
    print("=" * 85)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device} ({torch.cuda.get_device_name(device)})")
    
    # 1. Preflight Protocol Audit
    cache_dir = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
    data_dir = "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    
    mat_files = sorted(glob.glob(os.path.join(data_dir, "Bidslab*", "*", "labels.mat")))
    all_nights = [(f.split(os.sep)[-3], f.split(os.sep)[-2]) for f in mat_files]
    all_subjects = sorted(list(set(s for s, n in all_nights)))
    
    split = make_reproducible_split(all_subjects, split_type="paper", seed=0)
    train_subs = split["train"] # 31
    val_subs = split["val"]     # 5
    test_subs = split["test"]   # 11
    
    assert len(train_subs) == 31
    assert len(val_subs) == 5
    assert len(test_subs) == 11
    assert len(set(train_subs).intersection(val_subs)) == 0
    assert len(set(train_subs).intersection(test_subs)) == 0
    assert len(set(val_subs).intersection(test_subs)) == 0
    print("Preflight Audit: Zero subject overlap verified across 31 train, 5 val, 11 test.")
    
    subject_to_idx = {s: i for i, s in enumerate(train_subs)}
    
    # Load Training and Validation Records ONLY (Test records remain strictly unloaded during training)
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
                
    assert len(train_records) == 160
    assert len(val_records) == 28
    print(f"Loaded 160 usable train nights and 28 usable val nights.")
    print("TEST ISOLATION CONFIRMATION: Held-out test partition is NOT loaded during training.")
    
    train_ds = ControlledSleepDataset(train_records, subject_to_idx=subject_to_idx, seq_len=600)
    val_ds = ControlledSleepDataset(val_records, subject_to_idx=subject_to_idx, seq_len=600)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    seeds = [42, 123, 2024]
    trained_models = {}
    
    # 2. Train Exactly 80 Epochs for Each Seed (Same execution flow as EXP-011)
    for seed in seeds:
        print(f"\n>>> Training B0_GRL (Seed {seed}) for EXACTLY 80 Epochs <<<")
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        model = build_ablation_model(
            "A4",
            n_train_subjects=len(train_subs),
            loss_type="rccloss",
            boundary_aware=False,
            lambda_boundary=0.0
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
        
        t0 = time.time()
        for ep in range(1, 81):
            model.train()
            batch_losses = []
            for batch in train_loader:
                x_ihr = batch["ihr"].to(device)
                x_acc = batch["accel"].to(device)
                x_meta = batch["meta"].to(device)
                targets = batch["y"].to(device)
                mask = batch["mask"].to(device)
                subj_idx = batch["subject_idx"].to(device)
                
                optimizer.zero_grad()
                out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subj_idx)
                loss = out["total_loss"]
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
                
            # Inter-epoch validation evaluation (identical execution flow as EXP-011)
            val_m = evaluate_validation_sweep(model, val_loader, device=device)
            
            if ep % 20 == 0 or ep == 80:
                print(f"  Seed {seed:4d} | Ep {ep:02d}/80 | TrainLoss: {np.mean(batch_losses):.4f} | "
                      f"ValLoss: {val_m['val_loss']:.4f} | ValAcc: {val_m['accuracy']*100:.2f}% | "
                      f"ValMacro-F1: {val_m['macro_f1']:.4f} | Elapsed: {time.time()-t0:.1f}s")
                      
        # Save Final Frozen Model Checkpoint
        ckpt_path = f"exp012_seed{seed}_final.pt"
        torch.save({
            "seed": seed,
            "epoch": 80,
            "model_state": model.state_dict(),
            "val_metrics": val_m,
            "n_params": sum(p.numel() for p in model.parameters())
        }, ckpt_path)
        print(f"  Saved frozen Epoch-80 checkpoint to {ckpt_path}")
        trained_models[seed] = model

    # 3. NOW AND ONLY NOW: Load Test Set for the Single Final Evaluation
    print("\n" + "=" * 85)
    print("LOADING OFFICIAL HELD-OUT TEST DATASET (11 SUBJECTS, 58 USABLE NIGHTS)...")
    print("=" * 85)
    
    test_records = []
    for s, n in all_nights:
        if s in test_subs:
            rec = load_night_features(cache_dir, s, n)
            if rec is not None:
                test_records.append(rec)
                
    assert len(test_records) == 58, f"Expected 58 usable test nights, got {len(test_records)}"
    total_test_windows = sum(len(r["y"][:600]) for r in test_records)
    print(f"Loaded {len(test_records)} usable test nights ({total_test_windows} total 30-second windows across 11 subjects).")
    
    test_ds = ControlledSleepDataset(test_records, subject_to_idx=subject_to_idx, seq_len=600)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    # 4. Evaluate Each Frozen Model on Test Set
    final_test_results = {}
    subject_rows_all_seeds = []
    
    for seed in seeds:
        print(f"\n--- Evaluating Frozen Epoch-80 Model (Seed {seed}) on Test Set ---")
        model = trained_models[seed]
        res = evaluate_test_set(model, test_loader, device=device)
        
        print(f"Seed {seed} Test Results:")
        print(f"  - Accuracy:     {res['accuracy']*100:.2f}% (95% CI: [{res['bootstrap_ci']['accuracy_95ci'][0]*100:.2f}%, {res['bootstrap_ci']['accuracy_95ci'][1]*100:.2f}%])")
        print(f"  - Macro-F1:     {res['macro_f1']:.4f} (95% CI: [{res['bootstrap_ci']['macro_f1_95ci'][0]:.4f}, {res['bootstrap_ci']['macro_f1_95ci'][1]:.4f}])")
        print(f"  - Weighted-F1:  {res['weighted_f1']:.4f}")
        print(f"  - Cohen's κ:    {res['kappa']:.4f} (95% CI: [{res['bootstrap_ci']['kappa_95ci'][0]:.4f}, {res['bootstrap_ci']['kappa_95ci'][1]:.4f}])")
        print(f"  - MCC:          {res['mcc']:.4f}")
        print(f"  - ECE:          {res['ece']:.4f}")
        print(f"  - Wake F1:      {res['wake_f1']:.4f} (Recall: {res['wake_recall']*100:.2f}%)")
        print(f"  - Light F1:     {res['light_f1']:.4f} (Recall: {res['light_recall']*100:.2f}%)")
        print(f"  - Deep F1:      {res['deep_f1']:.4f} (Recall: {res['deep_recall']*100:.2f}%)")
        print(f"  - REM F1:       {res['rem_f1']:.4f} (Recall: {res['rem_recall']*100:.2f}%)")
        print(f"  - Boundary Err: {res['boundary_error_rate']*100:.2f}% | Non-Bound Err: {res['non_boundary_error_rate']*100:.2f}%")
        
        for s_row in res["subject_metrics"]:
            subject_rows_all_seeds.append({
                "seed": seed,
                "subject_id": s_row["subject_id"],
                "n_windows": s_row["n_windows"],
                "accuracy": s_row["accuracy"],
                "macro_f1": s_row["macro_f1"],
                "kappa": s_row["kappa"]
            })
            
        final_test_results[seed] = res

    # 5. Export Per-Subject Metrics to CSV
    df_subjects = pd.DataFrame(subject_rows_all_seeds)
    df_subjects.to_csv("exp012_subject_metrics.csv", index=False)
    print("\nSaved per-subject metrics to exp012_subject_metrics.csv")
    
    # 6. Generate Confusion Matrix Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    stage_labels = ["Wake", "Light", "Deep", "REM"]
    
    for i, seed in enumerate(seeds):
        cm_norm = np.array(final_test_results[seed]["confusion_matrix_norm"]) * 100
        cm_raw = np.array(final_test_results[seed]["confusion_matrix_raw"])
        
        im = axes[i].imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=100)
        axes[i].set_title(f"Seed {seed} (Acc: {final_test_results[seed]['accuracy']*100:.1f}%, F1: {final_test_results[seed]['macro_f1']:.3f})")
        axes[i].set_xticks(range(4))
        axes[i].set_yticks(range(4))
        axes[i].set_xticklabels(stage_labels)
        axes[i].set_yticklabels(stage_labels)
        axes[i].set_xlabel("Predicted Stage")
        axes[i].set_ylabel("True Stage")
        
        for r in range(4):
            for c in range(4):
                val_pct = cm_norm[r, c]
                val_raw = cm_raw[r, c]
                txt_color = "white" if val_pct > 50 else "black"
                axes[i].text(c, r, f"{val_pct:.1f}%\n({val_raw:,})", ha="center", va="center", color=txt_color, fontsize=9)
                
    plt.tight_layout()
    plt.savefig("exp012_confusion_matrices.png", dpi=200)
    plt.close()
    print("Saved confusion matrix visualization to exp012_confusion_matrices.png")
    
    # 7. Aggregate Across 3 Seeds
    summary = {
        "seeds": seeds,
        "results_by_seed": {
            s: {k: v for k, v in final_test_results[s].items() if k not in ["y_true", "y_pred", "subjects"]}
            for s in seeds
        },
        "aggregate": {}
    }
    
    metrics_to_agg = [
        "accuracy", "macro_f1", "weighted_f1", "kappa", "mcc", "ece",
        "wake_f1", "light_f1", "deep_f1", "rem_f1",
        "wake_recall", "light_recall", "deep_recall", "rem_recall",
        "boundary_error_rate", "non_boundary_error_rate"
    ]
    
    for m in metrics_to_agg:
        vals = [final_test_results[s][m] for s in seeds]
        summary["aggregate"][m] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals))
        }
        
    with open("exp012_test_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open("exp012_test_results.pkl", "wb") as f:
        pickle.dump(final_test_results, f)
    print("Saved test results to exp012_test_results.json and exp012_test_results.pkl")
    
    # Print Synthesis Summary Table
    print("\n" + "=" * 95)
    print("EXP-012 FINAL FROZEN TEST BENCHMARK SUMMARY (N=3 SEEDS: 42, 123, 2024)")
    print("=" * 95)
    print(f"{'Metric':<25} | {'Seed 42':<12} | {'Seed 123':<12} | {'Seed 2024':<12} | {'Mean +/- Std':<20}")
    print("-" * 95)
    
    display_rows = [
        ("Accuracy (%)", "accuracy", 100.0, "%"),
        ("Macro-F1", "macro_f1", 1.0, ""),
        ("Weighted-F1", "weighted_f1", 1.0, ""),
        ("Cohen's Kappa (κ)", "kappa", 1.0, ""),
        ("MCC", "mcc", 1.0, ""),
        ("ECE", "ece", 1.0, ""),
        ("Wake F1", "wake_f1", 1.0, ""),
        ("Light F1", "light_f1", 1.0, ""),
        ("Deep F1", "deep_f1", 1.0, ""),
        ("REM F1", "rem_f1", 1.0, ""),
        ("Wake Recall (%)", "wake_recall", 100.0, "%"),
        ("REM Recall (%)", "rem_recall", 100.0, "%"),
        ("Boundary Error (%)", "boundary_error_rate", 100.0, "%"),
        ("Non-Boundary Error (%)", "non_boundary_error_rate", 100.0, "%"),
    ]
    
    for label, k, scale, unit in display_rows:
        v42 = final_test_results[42][k] * scale
        v123 = final_test_results[123][k] * scale
        v2024 = final_test_results[2024][k] * scale
        m = summary["aggregate"][k]["mean"] * scale
        s = summary["aggregate"][k]["std"] * scale
        print(f"{label:<25} | {v42:6.2f}{unit:<5} | {v123:6.2f}{unit:<5} | {v2024:6.2f}{unit:<5} | {m:6.2f}{unit} +/- {s:4.2f}{unit}")
        
    print("=" * 95)
    return summary


if __name__ == "__main__":
    run_exp012()
