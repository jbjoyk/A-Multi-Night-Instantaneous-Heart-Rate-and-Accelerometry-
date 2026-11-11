"""
EXP-014: 5-Fold Subject Cross-Validation Runner
Script: run_exp014_cv.py

Protocol:
  - 5 models:
      Model A: Matched Conventional CNN-BiGRU (163,940 params)
      Model B: SleepAdaptiveNet minus GRL (153,974 params)
      Model C: SleepAdaptiveNet minus Transition (162,325 params)
      Model D: SleepAdaptiveNet minus Dynamic Gate (153,876 params)
      Model E: Full SleepAdaptiveNet (162,325 params)
  - 5 outer folds (all 47 subjects, 246 usable nights, strictly disjoint)
  - 3 seeds per model per fold (42, 123, 2024)
  - Total runs = 5 * 5 * 3 = 75 training runs
  - 80 fixed optimization epochs per run
  - Batch size = 4 nights, sequence length T = 600 epochs (5.0 hours)
  - Optimizer: Adam (lr = 1e-4, weight_decay = 0.0)
  - Loss: RCCLoss (official fn/fp weights, lambda_rcc = 0.01)
  - Inter-epoch validation on inner-validation subjects (5 subjects per fold)
  - Outer test partition evaluated strictly ONCE at epoch 80

DO NOT RUN WITHOUT USER AUTHORIZATION.
"""

import os
import sys
import glob
import time
import json
import pickle
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, precision_recall_fscore_support
)

from exp014_models import build_exp014_model
from slamss_ifs_train import load_night_features
from run_controlled_experiment import ControlledSleepDataset, compute_ece
from run_exp013_attribution import evaluate_validation_sweep_exp013

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
DATA_DIR = os.path.join(BASE_DIR, "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0")
MANIFEST_DIR = os.path.join(BASE_DIR, "EXP014_MANIFESTS")
OUTPUT_DIR = os.path.join(BASE_DIR, "EXP014_RESULTS")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PRED_DIR = os.path.join(OUTPUT_DIR, "predictions")

MODELS = ["A", "B", "C", "D", "E"]
FOLDS = [1, 2, 3, 4, 5]
SEEDS = [42, 123, 2024]
EPOCHS = 80
BATCH_SIZE = 4
SEQ_LEN = 600
LR = 1e-4


def evaluate_outer_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates an epoch-80 model on the fold's outer test partition.
    Extracts predictions, targets, subject labels, boundary indicators, and probabilities.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_boundaries = []
    all_probs = []
    all_subjects = []
    all_nights = []
    all_night_ids = []
    all_alphas = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_ihr = batch["ihr"].to(device)
            x_acc = batch["accel"].to(device)
            x_meta = batch["meta"].to(device)
            targets = batch["y"].to(device)
            mask = batch["mask"].to(device)
            subj_idx = batch["subject_idx"].to(device)
            
            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subj_idx)
            logits = out["logits"]
            probs = out["probabilities"]
            preds = torch.argmax(logits, dim=-1)
            
            # Boundary detection: y_t != y_{t-1}
            y_prev = targets[:, :-1]
            y_curr = targets[:, 1:]
            is_bound = torch.zeros_like(targets, dtype=torch.bool)
            is_bound[:, 1:] = (y_prev != y_curr) & (y_prev >= 0) & (y_prev < 4) & (y_curr >= 0) & (y_curr < 4)
            
            valid = (targets >= 0) & (targets < 4) & mask
            subjs = batch["subject"]
            nights = batch["night"]
            alpha_tensor = out.get("alpha_gate")
            
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
                all_nights.extend([n_name] * len(p_b))
                all_night_ids.extend([f"{s_name}__{n_name}"] * len(p_b))
                
                if alpha_tensor is not None:
                    alp_b = alpha_tensor[b_i][m_b].cpu().squeeze(-1).numpy()
                    all_alphas.extend(alp_b)
                    
    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    is_bound = np.array(all_boundaries, dtype=bool)
    probs_concat = np.concatenate(all_probs, axis=0)
    subjects_arr = np.array(all_subjects)
    alphas_arr = np.array(all_alphas) if len(all_alphas) > 0 else None
    
    # Primary Metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    ece = float(compute_ece(probs_concat, y_true, n_bins=10))
    
    # Per-class metrics
    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    # Boundary error rates
    correct = (y_true == y_pred)
    err_b = float(1.0 - correct[is_bound].mean()) if is_bound.sum() > 0 else 0.0
    err_nb = float(1.0 - correct[~is_bound].mean()) if (~is_bound).sum() > 0 else 0.0
    
    # Subject-level metrics
    subject_metrics = []
    subject_predictions = {}
    for s_id in np.unique(subjects_arr):
        m_s = (subjects_arr == s_id)
        yt_s = y_true[m_s]
        yp_s = y_pred[m_s]
        acc_s = float(accuracy_score(yt_s, yp_s))
        f1_s = float(f1_score(yt_s, yp_s, average="macro", zero_division=0))
        kap_s = float(cohen_kappa_score(yt_s, yp_s))
        subject_metrics.append({
            "subject_id": s_id,
            "n_windows": int(np.sum(m_s)),
            "accuracy": acc_s,
            "macro_f1": f1_s,
            "kappa": kap_s
        })
        subject_predictions[s_id] = (yt_s, yp_s)
        
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "kappa": kappa,
        "mcc": mcc,
        "ece": ece,
        "wake_f1": float(f1_cls[0]),
        "light_f1": float(f1_cls[1]),
        "deep_f1": float(f1_cls[2]),
        "rem_f1": float(f1_cls[3]),
        "wake_recall": float(rec[0]),
        "light_recall": float(rec[1]),
        "deep_recall": float(rec[2]),
        "rem_recall": float(rec[3]),
        "precision": [float(p) for p in prec],
        "recall": [float(r) for r in rec],
        "f1": [float(f) for f in f1_cls],
        "support": [int(s) for s in supp],
        "boundary_error_rate": err_b,
        "non_boundary_error_rate": err_nb,
        "confusion_matrix_raw": cm.tolist(),
        "confusion_matrix_norm": cm_norm.tolist(),
        "subject_metrics": subject_metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "subjects": subjects_arr,
        "night_id": np.array(all_night_ids),
        "night_ids": np.array(all_night_ids),
        "nights": np.array(all_nights),
        "subject_predictions": subject_predictions
    }


def run_exp014_cv():
    """Main execution function for EXP-014 5-Fold Subject Cross-Validation."""
    print("=" * 85)
    print("EXP-014: 5-FOLD SUBJECT CROSS-VALIDATION (75 RUNS)")
    print("=" * 85)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device} ({torch.cuda.get_device_name(device)})")
    
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(PRED_DIR, exist_ok=True)
    
    # Load all dataset records
    all_mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "Bidslab*", "*", "labels.mat")))
    all_nights_dataset = [(f.split(os.sep)[-3], f.split(os.sep)[-2].replace("Night ", "")) for f in all_mat_files]
    
    records_by_sub = {}
    for s, n in all_nights_dataset:
        rec = load_night_features(CACHE_DIR, s, n)
        if rec is not None:
            if s not in records_by_sub:
                records_by_sub[s] = []
            records_by_sub[s].append(rec)
            
    total_loaded_nights = sum(len(recs) for recs in records_by_sub.values())
    assert total_loaded_nights == 246, f"Expected 246 usable nights, loaded {total_loaded_nights}"
    print(f"Dataset Loaded: 47 subjects, 246 usable nights [VERIFIED]")
    
    all_results = []
    
    # Outer Fold Loop (1..5)
    for fold in FOLDS:
        manifest_path = os.path.join(MANIFEST_DIR, f"fold_{fold}.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        inner_train_subs = manifest["inner_train_subjects"]
        inner_val_subs = manifest["inner_val_subjects"]
        outer_test_subs = manifest["outer_test_subjects"]
        
        train_records = [r for s in inner_train_subs for r in records_by_sub[s]]
        val_records = [r for s in inner_val_subs for r in records_by_sub[s]]
        test_records = [r for s in outer_test_subs for r in records_by_sub[s]]
        
        subject_to_idx = {s: i for i, s in enumerate(inner_train_subs)}
        
        val_ds = ControlledSleepDataset(val_records, subject_to_idx=subject_to_idx, seq_len=SEQ_LEN)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
        test_ds = ControlledSleepDataset(test_records, subject_to_idx=subject_to_idx, seq_len=SEQ_LEN)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
        
        print("\n" + "=" * 80)
        print(f"OUTER FOLD {fold}/5: {len(outer_test_subs)} Test Subs ({len(test_records)} nts) | "
              f"{len(inner_train_subs)} Train Subs ({len(train_records)} nts) | {len(inner_val_subs)} Val Subs ({len(val_records)} nts)")
        print("=" * 80)
        
        # Model Loop (A, B, C, D, E)
        for m_id in MODELS:
            print(f"\n--- Model {m_id} [Fold {fold}] ---")
            
            # Seed Loop (42, 123, 2024)
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                
                # Build fresh model instance for this fold/seed (Model E freshly created, NOT EXP-012)
                model = build_exp014_model(m_id, n_train_subjects=len(inner_train_subs)).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
                
                train_ds = ControlledSleepDataset(train_records, subject_to_idx=subject_to_idx, seq_len=SEQ_LEN)
                train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
                
                t0 = time.time()
                for ep in range(1, EPOCHS + 1):
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
                        
                    # Deterministic inter-epoch inner validation evaluation
                    val_m = evaluate_validation_sweep_exp013(model, val_loader, device=device)
                    
                    if ep % 40 == 0 or ep == EPOCHS:
                        print(f"  Fold {fold} | Model {m_id} | Seed {seed} | Ep {ep:02d}/{EPOCHS} | "
                              f"TrainLoss: {np.mean(batch_losses):.4f} | ValAcc: {val_m['accuracy']*100:.2f}% | "
                              f"ValMacro-F1: {val_m['macro_f1']:.4f} | Elapsed: {time.time()-t0:.1f}s")
                              
                # Save trained fold checkpoint
                ckpt_path = os.path.join(CKPT_DIR, f"model_{m_id}_fold{fold}_seed{seed}.pt")
                torch.save({
                    "model_id": m_id,
                    "fold": fold,
                    "seed": seed,
                    "epoch": EPOCHS,
                    "model_state": model.state_dict(),
                    "val_metrics": val_m
                }, ckpt_path)
                
                # Single outer-test evaluation at locked epoch 80
                test_m = evaluate_outer_test(model, test_loader, device=device)
                print(f"  >> Fold {fold} | Model {m_id} | Seed {seed} Outer Test: "
                      f"Acc: {test_m['accuracy']*100:.2f}% | Macro-F1: {test_m['macro_f1']:.4f} | "
                      f"Kappa: {test_m['kappa']:.4f} | ECE: {test_m['ece']:.4f}")
                      
                # Save predictions
                pred_path = os.path.join(PRED_DIR, f"pred_{m_id}_fold{fold}_seed{seed}.pkl")
                with open(pred_path, "wb") as f:
                    pickle.dump({
                        "model_id": m_id,
                        "fold": fold,
                        "seed": seed,
                        "y_true": test_m["y_true"],
                        "y_pred": test_m["y_pred"],
                        "subjects": test_m["subjects"],
                        "night_id": test_m["night_id"],
                        "night_ids": test_m["night_ids"],
                        "nights": test_m["nights"],
                        "confusion_matrix_raw": test_m["confusion_matrix_raw"],
                        "subject_metrics": test_m["subject_metrics"]
                    }, f)
                    
                all_results.append({
                    "model_id": m_id,
                    "fold": fold,
                    "seed": seed,
                    "accuracy": test_m["accuracy"],
                    "macro_f1": test_m["macro_f1"],
                    "weighted_f1": test_m["weighted_f1"],
                    "kappa": test_m["kappa"],
                    "mcc": test_m["mcc"],
                    "ece": test_m["ece"],
                    "wake_f1": test_m["wake_f1"],
                    "light_f1": test_m["light_f1"],
                    "deep_f1": test_m["deep_f1"],
                    "rem_f1": test_m["rem_f1"],
                    "wake_recall": test_m["wake_recall"],
                    "light_recall": test_m["light_recall"],
                    "deep_recall": test_m["deep_recall"],
                    "rem_recall": test_m["rem_recall"],
                    "boundary_error_rate": test_m["boundary_error_rate"],
                    "non_boundary_error_rate": test_m["non_boundary_error_rate"],
                    "train_time_s": time.time() - t0
                })
                
    # Save overall summary dataframe
    df_results = pd.DataFrame(all_results)
    df_results.to_csv(os.path.join(BASE_DIR, "EXP014_RESULTS.csv"), index=False)
    print("\nEXP-014 EXECUTION COMPLETE. All 75 runs saved to EXP014_RESULTS.csv.")


if __name__ == "__main__":
    run_exp014_cv()
