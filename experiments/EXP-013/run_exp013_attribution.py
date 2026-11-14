"""
EXP-013: Rigorous Component Attribution Experiment Runner
Script: run_exp013_attribution.py

Evaluates:
  - Model A: Matched-capacity conventional CNN-BiGRU (no gate, direct concat, no trans, no GRL)
  - Model B: Full adaptive fusion + transition, NO GRL
  - Model C: Adaptive fusion + GRL, NO transition
  - Model D: Fixed equal fusion (0.5/0.5) + transition + GRL
  - Model E: Frozen EXP-012 B0_GRL reference (read-only, never retrained)

Strict Experimental Controls:
  - 80 fixed optimization epochs per seed (seeds: 42, 123, 2024).
  - Adam (lr = 1e-4, weight_decay = 0.0, batch_size = 4, seq_len = 600).
  - Identical official split: 31 train (160 nights) / 5 val (28 nights) / 11 test (58 nights).
  - Deterministic inter-epoch validation evaluation after every epoch (matched to EXP-011/012 RNG flow).
  - Test set strictly isolated during training; evaluated once per seed at locked epoch 80.
  - Paired subject-level cluster bootstrap (B=1000) comparing Model E against A, B, C, D.

DO NOT LAUNCH UNTIL USER REVIEWS PRE-FLIGHT AUDIT AND AUTHORIZES EXECUTION.
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

from exp013_models import build_exp013_model
from slamss_ifs_baseline import make_reproducible_split
from slamss_ifs_train import load_night_features
from run_controlled_experiment import ControlledSleepDataset, compute_ece

def evaluate_validation_sweep_exp013(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Detailed validation evaluation matching run_exp008_boundary_sweep.evaluate_validation_sweep,
    with safe handling when alpha_gate is None (e.g. Model A).
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_boundaries = []
    all_confidences = []
    all_entropies = []
    all_alphas = []
    all_probs = []
    
    val_losses = []
    loss_components = {"stage": [], "trans": [], "bound": [], "adv": [], "conf": []}
    
    with torch.no_grad():
        for batch in val_loader:
            x_ihr = batch["ihr"].to(device)
            x_acc = batch["accel"].to(device)
            x_meta = batch["meta"].to(device)
            targets = batch["y"].to(device)
            mask = batch["mask"].to(device)
            subj_idx = batch["subject_idx"].to(device)
            
            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subj_idx)
            
            val_losses.append(out["total_loss"].item())
            loss_components["stage"].append(out["loss_stage"].item())
            loss_components["trans"].append(out["loss_trans"].item())
            loss_components["bound"].append(out["loss_boundary"].item())
            loss_components["adv"].append(out["loss_adv"].item())
            loss_components["conf"].append(out["loss_conf"].item())
            
            logits = out["logits"]
            probs = out["probabilities"]
            preds = torch.argmax(logits, dim=-1)
            conf = out["confidence"].squeeze(-1) if out.get("confidence") is not None else torch.zeros_like(preds, dtype=torch.float32)
            ent = out["entropy"].squeeze(-1) if out.get("entropy") is not None else torch.zeros_like(preds, dtype=torch.float32)
            
            alpha_tensor = out.get("alpha_gate")
            if alpha_tensor is not None:
                alpha = alpha_tensor.squeeze(-1)
            else:
                alpha = torch.full_like(preds, 0.5, dtype=torch.float32)
            
            # Boundary mask (y_t != y_{t-1})
            y_prev = targets[:, :-1]
            y_curr = targets[:, 1:]
            is_bound = torch.zeros_like(targets, dtype=torch.bool)
            is_bound[:, 1:] = (y_prev != y_curr) & (y_prev >= 0) & (y_curr >= 0)
            
            valid = (targets >= 0) & (targets < 4) & mask
            
            all_preds.extend(preds[valid].cpu().numpy())
            all_targets.extend(targets[valid].cpu().numpy())
            all_boundaries.extend(is_bound[valid].cpu().numpy())
            all_confidences.extend(conf[valid].cpu().numpy())
            all_entropies.extend(ent[valid].cpu().numpy())
            all_alphas.extend(alpha[valid].cpu().numpy())
            all_probs.append(probs[valid].cpu().numpy())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    is_bound = np.array(all_boundaries, dtype=bool)
    conf = np.array(all_confidences)
    ent = np.array(all_entropies)
    alpha = np.array(all_alphas)
    probs_concat = np.concatenate(all_probs, axis=0)
    
    # Primary Metrics
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    prec, rec, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    is_correct = (y_true == y_pred)
    err_boundary = 1.0 - is_correct[is_bound].mean() if is_bound.sum() > 0 else 0.0
    err_non_boundary = 1.0 - is_correct[~is_bound].mean() if (~is_bound).sum() > 0 else 0.0
    
    ece = compute_ece(probs_concat, y_true, n_bins=10)
    
    alpha_stats = {
        "mean": float(alpha.mean()),
        "std": float(alpha.std()),
        "min": float(alpha.min()),
        "max": float(alpha.max()),
        "correct_mean": float(alpha[is_correct].mean()) if is_correct.sum() > 0 else 0.5,
        "correct_std": float(alpha[is_correct].std()) if is_correct.sum() > 0 else 0.0,
        "incorrect_mean": float(alpha[~is_correct].mean()) if (~is_correct).sum() > 0 else 0.5,
        "incorrect_std": float(alpha[~is_correct].std()) if (~is_correct).sum() > 0 else 0.0,
    }
    alpha_by_stage = {}
    for idx, name in enumerate(["Wake", "Light", "Deep", "REM"]):
        m_stg = (y_true == idx)
        alpha_by_stage[name] = float(alpha[m_stg].mean()) if m_stg.sum() > 0 else 0.5
        
    return {
        "val_loss": float(np.mean(val_losses)),
        "loss_stage": float(np.mean(loss_components["stage"])),
        "loss_trans": float(np.mean(loss_components["trans"])),
        "loss_bound": float(np.mean(loss_components["bound"])),
        "loss_adv": float(np.mean(loss_components["adv"])),
        "loss_conf": float(np.mean(loss_components["conf"])),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "kappa": float(kappa),
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
        "boundary_error_rate": float(err_boundary),
        "non_boundary_error_rate": float(err_non_boundary),
        "ece": float(ece),
        "alpha_stats": alpha_stats,
        "alpha_by_stage": alpha_by_stage
    }

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
DATA_DIR = os.path.join(BASE_DIR, "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0")
EXP012_PKL = os.path.join(BASE_DIR, "exp012_test_results.pkl")
OUTPUT_DIR = os.path.join(BASE_DIR, "EXP013_OUTPUTS")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)


def evaluate_model_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    Evaluates a frozen epoch-80 model on the official held-out test set.
    Extracts predictions, targets, boundaries, subjects, and gate alphas.
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    all_boundaries = []
    all_probs = []
    all_subjects = []
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
            alpha_tensor = out.get("alpha_gate") # (B, T, 1) or None
            
            for b_i in range(len(subjs)):
                s_name = subjs[b_i]
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
                
                if alpha_tensor is not None:
                    alp_b = alpha_tensor[b_i][m_b].cpu().squeeze(-1).numpy()
                    all_alphas.extend(alp_b)
                    
    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    is_bound = np.array(all_boundaries, dtype=bool)
    probs_concat = np.concatenate(all_probs, axis=0)
    subjects_arr = np.array(all_subjects)
    alphas_arr = np.array(all_alphas) if len(all_alphas) > 0 else None
    
    # Assert exactly 34,241 valid test windows
    assert len(y_true) == 34241, f"Expected 34241 real test windows, got {len(y_true)}"
    
    # 1. Primary Metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    ece = float(compute_ece(probs_concat, y_true, n_bins=10))
    
    # 2. Per-Class Metrics
    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    # 3. Boundary Dynamics
    correct = (y_true == y_pred)
    err_b = float(1.0 - correct[is_bound].mean()) if is_bound.sum() > 0 else 0.0
    err_nb = float(1.0 - correct[~is_bound].mean()) if (~is_bound).sum() > 0 else 0.0
    
    # 4. Per-Subject Metrics
    unique_subs = sorted(list(set(all_subjects)))
    subject_metrics = []
    subject_predictions = {}
    
    for s_id in unique_subs:
        idx_s = (subjects_arr == s_id)
        yt_s = y_true[idx_s]
        yp_s = y_pred[idx_s]
        acc_s = float(accuracy_score(yt_s, yp_s))
        f1_s = float(f1_score(yt_s, yp_s, average="macro", zero_division=0))
        try:
            kap_s = float(cohen_kappa_score(yt_s, yp_s))
        except Exception:
            kap_s = np.nan
            
        subject_metrics.append({
            "subject_id": s_id,
            "n_windows": int(len(yt_s)),
            "accuracy": acc_s,
            "macro_f1": f1_s,
            "kappa": kap_s
        })
        subject_predictions[s_id] = (yt_s, yp_s)
        
    # 5. Alpha Gate Analysis (if available)
    alpha_stats = None
    if alphas_arr is not None:
        alpha_stats = {
            "mean": float(np.mean(alphas_arr)),
            "std": float(np.std(alphas_arr)),
            "per_stage": {}
        }
        for c in range(4):
            m_c = (y_true == c)
            if m_c.any():
                alpha_stats["per_stage"][c] = {
                    "mean": float(np.mean(alphas_arr[m_c])),
                    "std": float(np.std(alphas_arr[m_c]))
                }
                
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
        "alpha_stats": alpha_stats,
        "y_true": y_true,
        "y_pred": y_pred,
        "subjects": subjects_arr,
        "subject_predictions": subject_predictions
    }


def compute_paired_bootstrap(
    model_e_results: Dict[int, Any],
    comp_results: Dict[int, Any],
    B: int = 1000,
    seed: int = 0
) -> Dict[str, Any]:
    """
    Computes paired cluster bootstrap deltas (Model E - Comparator)
    across resampled subject clusters using IDENTICAL resampled subjects.
    """
    rng = np.random.default_rng(seed)
    seeds = sorted(list(model_e_results.keys()))
    
    paired_deltas = {
        "delta_accuracy": [],
        "delta_macro_f1": [],
        "delta_kappa": [],
        "delta_mcc": []
    }
    
    unique_subs = sorted(list(model_e_results[seeds[0]]["subject_predictions"].keys()))
    n_subs = len(unique_subs)
    
    for s in seeds:
        e_sub_preds = model_e_results[s]["subject_predictions"]
        c_sub_preds = comp_results[s]["subject_predictions"]
        
        # Pre-compute confusion matrices per subject
        e_cms = {sub: confusion_matrix(e_sub_preds[sub][0], e_sub_preds[sub][1], labels=[0, 1, 2, 3]) for sub in unique_subs}
        c_cms = {sub: confusion_matrix(c_sub_preds[sub][0], c_sub_preds[sub][1], labels=[0, 1, 2, 3]) for sub in unique_subs}
        
        for _ in range(B // len(seeds)):
            resampled = rng.choice(unique_subs, size=n_subs, replace=True)
            
            # Pooled CMs for resampled subject set
            e_cm_boot = sum(e_cms[sub] for sub in resampled)
            c_cm_boot = sum(c_cms[sub] for sub in resampled)
            
            # Accuracy
            e_acc = np.diag(e_cm_boot).sum() / e_cm_boot.sum()
            c_acc = np.diag(c_cm_boot).sum() / c_cm_boot.sum()
            paired_deltas["delta_accuracy"].append(float(e_acc - c_acc))
            
            # Macro-F1
            def calc_mf1(cm_mat):
                f1s = []
                for c in range(4):
                    tp = cm_mat[c, c]
                    fp = cm_mat[:, c].sum() - tp
                    fn = cm_mat[c, :].sum() - tp
                    denom = 2 * tp + fp + fn
                    f1s.append(2 * tp / denom if denom > 0 else 0.0)
                return np.mean(f1s)
                
            e_mf1 = calc_mf1(e_cm_boot)
            c_mf1 = calc_mf1(c_cm_boot)
            paired_deltas["delta_macro_f1"].append(float(e_mf1 - c_mf1))
            
            # Kappa
            def calc_kap(cm_mat):
                tot = cm_mat.sum()
                po = np.diag(cm_mat).sum() / tot
                pe = (cm_mat.sum(axis=1) * cm_mat.sum(axis=0)).sum() / (tot ** 2)
                return (po - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0
                
            e_kap = calc_kap(e_cm_boot)
            c_kap = calc_kap(c_cm_boot)
            paired_deltas["delta_kappa"].append(float(e_kap - c_kap))

            # MCC
            def calc_mcc(cm_mat):
                c_val = np.trace(cm_mat)
                s_val = cm_mat.sum()
                pk = cm_mat.sum(axis=0)
                tk = cm_mat.sum(axis=1)
                num = c_val * s_val - np.dot(pk, tk)
                denom = np.sqrt((s_val**2 - np.dot(pk, pk)) * (s_val**2 - np.dot(tk, tk)))
                return float(num / denom) if denom > 0 else 0.0

            e_mcc = calc_mcc(e_cm_boot)
            c_mcc = calc_mcc(c_cm_boot)
            paired_deltas["delta_mcc"].append(float(e_mcc - c_mcc))
            
    summary_deltas = {}
    for k, v in paired_deltas.items():
        arr = np.array(v)
        summary_deltas[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "ci_95": (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))),
            "excludes_zero": bool((np.percentile(arr, 2.5) > 0) or (np.percentile(arr, 97.5) < 0))
        }
        
    return summary_deltas


def run_exp013():
    """Main execution function for EXP-013."""
    print("=" * 85)
    print("EXP-013: COMPONENT ATTRIBUTION EXPERIMENT EXECUTION")
    print("=" * 85)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device} ({torch.cuda.get_device_name(device)})")
    
    # 1. Dataset Partitioning
    mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "Bidslab*", "*", "labels.mat")))
    all_nights = [(f.split(os.sep)[-3], f.split(os.sep)[-2]) for f in mat_files]
    all_subjects = sorted(list(set(s for s, n in all_nights)))
    
    split = make_reproducible_split(all_subjects, split_type="paper", seed=0)
    train_subs = split["train"] # 31
    val_subs = split["val"]     # 5
    test_subs = split["test"]   # 11
    
    subject_to_idx = {s: i for i, s in enumerate(train_subs)}
    
    # Load Training and Validation Records
    train_records, val_records, test_records = [], [], []
    for s, n in all_nights:
        rec = load_night_features(CACHE_DIR, s, n)
        if rec is not None:
            if s in train_subs:
                train_records.append(rec)
            elif s in val_subs:
                val_records.append(rec)
            elif s in test_subs:
                test_records.append(rec)
                
    assert len(train_records) == 160
    assert len(val_records) == 28
    assert len(test_records) == 58
    print(f"Partitions Loaded: 160 train nights / 28 val nights / 58 test nights [VERIFIED]")
    
    val_ds = ControlledSleepDataset(val_records, subject_to_idx=subject_to_idx, seq_len=600)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    # Load Frozen Model E Reference Artifacts
    assert os.path.exists(EXP012_PKL), f"Missing EXP-012 artifact: {EXP012_PKL}"
    with open(EXP012_PKL, "rb") as f:
        model_e_raw = pickle.load(f)
        
    # Re-structure Model E subject predictions
    model_e_results = {}
    for s in [42, 123, 2024]:
        sub_preds = {}
        yt_e = model_e_raw[s]["y_true"]
        yp_e = model_e_raw[s]["y_pred"]
        sub_e = model_e_raw[s]["subjects"]
        for sub_id in set(sub_e):
            m = (sub_e == sub_id)
            sub_preds[sub_id] = (yt_e[m], yp_e[m])
        model_e_results[s] = dict(model_e_raw[s])
        model_e_results[s]["subject_predictions"] = sub_preds
        
    print("Loaded Frozen Model E (EXP-012 B0_GRL) Reference Artifacts [VERIFIED]")
    
    models_to_train = ["A", "B", "C", "D"]
    seeds = [42, 123, 2024]
    epochs = 80
    
    all_experiment_results = {"E": model_e_results}
    
    # 2. Sequential Training Loop
    for m_id in models_to_train:
        print("\n" + "=" * 80)
        print(f"TRAINING MODEL {m_id} ACROSS 3 SEEDS ({seeds}) FOR {epochs} EPOCHS EACH")
        print("=" * 80)
        
        m_results = {}
        for seed in seeds:
            print(f"\n--- Model {m_id} | Seed {seed} ---")
            torch.manual_seed(seed)
            np.random.seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            # Build clean model instance
            model = build_exp013_model(m_id, n_train_subjects=len(train_subs)).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
            
            train_ds = ControlledSleepDataset(train_records, subject_to_idx=subject_to_idx, seq_len=600)
            train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
            
            t0 = time.time()
            for ep in range(1, epochs + 1):
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
                    
                # Exact identical validation evaluation schedule (RNG flow preservation)
                val_m = evaluate_validation_sweep_exp013(model, val_loader, device=device)
                
                if ep % 20 == 0 or ep == epochs:
                    print(f"  Model {m_id} | Seed {seed} | Ep {ep:02d}/{epochs} | "
                          f"TrainLoss: {np.mean(batch_losses):.4f} | ValLoss: {val_m['val_loss']:.4f} | "
                          f"ValAcc: {val_m['accuracy']*100:.2f}% | ValMacro-F1: {val_m['macro_f1']:.4f} | "
                          f"Elapsed: {time.time()-t0:.1f}s")
                          
            # Save frozen checkpoint
            ckpt_path = os.path.join(CKPT_DIR, f"model_{m_id}_seed{seed}_final.pt")
            torch.save({
                "model_id": m_id,
                "seed": seed,
                "epoch": epochs,
                "model_state": model.state_dict(),
                "n_params": sum(p.numel() for p in model.parameters())
            }, ckpt_path)
            
            m_results[seed] = {"model": model, "ckpt": ckpt_path}
            
        all_experiment_results[m_id] = m_results
        
    # 3. Test Evaluation (Single Evaluation after All Training is Frozen)
    print("\n" + "=" * 80)
    print("EVALUATING ALL FROZEN MODELS ON UNTOUCHED HELD-OUT TEST PARTITION")
    print("=" * 80)
    
    test_ds = ControlledSleepDataset(test_records, subject_to_idx=subject_to_idx, seq_len=600)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    test_eval_results = {"E": model_e_results}
    
    for m_id in models_to_train:
        test_eval_results[m_id] = {}
        for seed in seeds:
            model = all_experiment_results[m_id][seed]["model"]
            test_m = evaluate_model_on_test(model, test_loader, device=device)
            test_eval_results[m_id][seed] = test_m
            print(f"Model {m_id} (Seed {seed}) Test Acc: {test_m['accuracy']*100:.2f}% | Macro-F1: {test_m['macro_f1']:.4f} | Kappa: {test_m['kappa']:.4f}")
            
    # 4. Paired Subject-Level Bootstrap Comparison vs Model E
    print("\n" + "=" * 80)
    print("COMPUTING PAIRED SUBJECT-LEVEL BOOTSTRAP DELTAS (MODEL E - COMPARATOR)")
    print("=" * 80)
    
    paired_bootstrap_results = {}
    for m_id in models_to_train:
        paired_deltas = compute_paired_bootstrap(model_e_results, test_eval_results[m_id], B=1000, seed=0)
        paired_bootstrap_results[m_id] = paired_deltas
        print(f"Paired Bootstrap Delta (Model E - Model {m_id}):")
        print(f"  Δ Accuracy: {paired_deltas['delta_accuracy']['mean']*100:+.2f}% [95% CI: {paired_deltas['delta_accuracy']['ci_95'][0]*100:+.2f}%, {paired_deltas['delta_accuracy']['ci_95'][1]*100:+.2f}%]")
        print(f"  Δ Macro-F1: {paired_deltas['delta_macro_f1']['mean']:+.4f} [95% CI: {paired_deltas['delta_macro_f1']['ci_95'][0]:+.4f}, {paired_deltas['delta_macro_f1']['ci_95'][1]:+.4f}]")
        print(f"  Δ Kappa:    {paired_deltas['delta_kappa']['mean']:+.4f} [95% CI: {paired_deltas['delta_kappa']['ci_95'][0]:+.4f}, {paired_deltas['delta_kappa']['ci_95'][1]:+.4f}]")
        print(f"  Δ MCC:      {paired_deltas['delta_mcc']['mean']:+.4f} [95% CI: {paired_deltas['delta_mcc']['ci_95'][0]:+.4f}, {paired_deltas['delta_mcc']['ci_95'][1]:+.4f}]")
        
    # 5. Programmatically Generate Output CSVs and JSON
    print("\n" + "=" * 80)
    print("SERIALIZING METRIC TABLES TO CSV AND JSON")
    print("=" * 80)
    
    # Model Metrics Table (Per-seed and Mean +/- Std)
    model_rows = []
    class_rows = []
    subject_rows = []
    alpha_rows = []
    
    stage_names = ["Wake", "Light", "Deep", "REM"]
    all_models = ["A", "B", "C", "D", "E"]
    
    for m_id in all_models:
        for seed in seeds:
            res = test_eval_results[m_id][seed]
            model_rows.append({
                "model_id": m_id,
                "seed": seed,
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "weighted_f1": res["weighted_f1"],
                "kappa": res["kappa"],
                "mcc": res["mcc"],
                "ece": res["ece"],
                "boundary_error_rate": res["boundary_error_rate"],
                "non_boundary_error_rate": res["non_boundary_error_rate"],
                "wake_f1": res["wake_f1"],
                "light_f1": res["light_f1"],
                "deep_f1": res["deep_f1"],
                "rem_f1": res["rem_f1"],
                "wake_recall": res["wake_recall"],
                "light_recall": res["light_recall"],
                "deep_recall": res["deep_recall"],
                "rem_recall": res["rem_recall"],
            })
            
            for c_idx in range(4):
                class_rows.append({
                    "model_id": m_id,
                    "seed": seed,
                    "class_name": stage_names[c_idx],
                    "precision": res["precision"][c_idx],
                    "recall": res["recall"][c_idx],
                    "f1": res["f1"][c_idx],
                    "support": res["support"][c_idx]
                })
                
            for s_row in res["subject_metrics"]:
                subject_rows.append({
                    "model_id": m_id,
                    "seed": seed,
                    "subject_id": s_row["subject_id"],
                    "n_windows": s_row["n_windows"],
                    "accuracy": s_row["accuracy"],
                    "macro_f1": s_row["macro_f1"],
                    "kappa": s_row["kappa"]
                })
                
            if res.get("alpha_stats") is not None:
                a_st = res["alpha_stats"]
                alpha_rows.append({
                    "model_id": m_id,
                    "seed": seed,
                    "mean_alpha": a_st["mean"],
                    "std_alpha": a_st["std"],
                    "wake_alpha": a_st["per_stage"].get(0, {}).get("mean", np.nan),
                    "light_alpha": a_st["per_stage"].get(1, {}).get("mean", np.nan),
                    "deep_alpha": a_st["per_stage"].get(2, {}).get("mean", np.nan),
                    "rem_alpha": a_st["per_stage"].get(3, {}).get("mean", np.nan),
                })
                
    df_models = pd.DataFrame(model_rows)
    df_classes = pd.DataFrame(class_rows)
    df_subjects = pd.DataFrame(subject_rows)
    df_alphas = pd.DataFrame(alpha_rows)
    
    df_models.to_csv(os.path.join(OUTPUT_DIR, "exp013_model_metrics.csv"), index=False)
    df_classes.to_csv(os.path.join(OUTPUT_DIR, "exp013_per_class_metrics.csv"), index=False)
    df_subjects.to_csv(os.path.join(OUTPUT_DIR, "exp013_subject_metrics.csv"), index=False)
    if len(df_alphas) > 0:
        df_alphas.to_csv(os.path.join(OUTPUT_DIR, "exp013_alpha_analysis.csv"), index=False)
        
    # Paired Bootstrap Table
    paired_rows = []
    for m_id in models_to_train:
        b_res = paired_bootstrap_results[m_id]
        for metric_name in ["delta_accuracy", "delta_macro_f1", "delta_kappa", "delta_mcc"]:
            m_data = b_res[metric_name]
            paired_rows.append({
                "comparator": f"Model {m_id}",
                "metric": metric_name,
                "mean_delta": m_data["mean"],
                "std_delta": m_data["std"],
                "ci_95_lower": m_data["ci_95"][0],
                "ci_95_upper": m_data["ci_95"][1],
                "ci_excludes_zero": m_data["excludes_zero"]
            })
    df_paired = pd.DataFrame(paired_rows)
    df_paired.to_csv(os.path.join(OUTPUT_DIR, "exp013_paired_bootstrap.csv"), index=False)
    
    # Save artifacts
    save_path_pkl = os.path.join(OUTPUT_DIR, "exp013_results.pkl")
    with open(save_path_pkl, "wb") as f:
        pickle.dump({
            "test_results": test_eval_results,
            "paired_bootstrap": paired_bootstrap_results
        }, f)
        
    # JSON Summary
    summary_json = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "models": all_models,
        "seeds": seeds,
        "aggregate_metrics": {},
        "paired_bootstrap": paired_bootstrap_results
    }
    
    for m_id in all_models:
        df_m = df_models[df_models["model_id"] == m_id]
        summary_json["aggregate_metrics"][m_id] = {
            "accuracy": {"mean": float(df_m["accuracy"].mean()), "std": float(df_m["accuracy"].std())},
            "macro_f1": {"mean": float(df_m["macro_f1"].mean()), "std": float(df_m["macro_f1"].std())},
            "weighted_f1": {"mean": float(df_m["weighted_f1"].mean()), "std": float(df_m["weighted_f1"].std())},
            "kappa": {"mean": float(df_m["kappa"].mean()), "std": float(df_m["kappa"].std())},
            "mcc": {"mean": float(df_m["mcc"].mean()), "std": float(df_m["mcc"].std())},
            "ece": {"mean": float(df_m["ece"].mean()), "std": float(df_m["ece"].std())},
            "boundary_error_rate": {"mean": float(df_m["boundary_error_rate"].mean()), "std": float(df_m["boundary_error_rate"].std())},
            "non_boundary_error_rate": {"mean": float(df_m["non_boundary_error_rate"].mean()), "std": float(df_m["non_boundary_error_rate"].std())},
        }
        
    with open(os.path.join(OUTPUT_DIR, "exp013_results.json"), "w") as f:
        json.dump(summary_json, f, indent=2)
        
    print(f"Saved complete results to {save_path_pkl} and {os.path.join(OUTPUT_DIR, 'exp013_results.json')}")
    print("EXP-013 EXECUTION COMPLETE.")


if __name__ == "__main__":
    run_exp013()

