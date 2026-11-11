"""
EXP-008: Controlled Boundary Weight Sweep (lambda_boundary in {0.10, 0.20, 0.50}) & GRL Disentanglement.

Strict Scientific Controls:
  - Seed: 42 (deterministic initialization, shuffling, CUDA)
  - Train: Exactly 31 paper subjects (160 usable nights)
  - Val: Exactly 5 paper subjects (28 usable nights)
  - Test: 11 subjects (58 nights) -> STRICTLY NOT LOADED / NOT USED
  - Sequence length: 600 epochs (5 continuous hours, 0 padding artifacts)
  - Batch size: 4
  - Optimizer: Adam (lr = 1e-4, weight_decay = 0.0)
  - Epochs: 10 epochs per model
  - Loss: RCCLoss + auxiliary objectives

Models evaluated:
  - B0:     A2 reference (Adaptive Fusion=True, Transition=True, Boundary=False, GRL=False, lambda_bound=0.0)
  - B0_GRL: A4 reference with Boundary=False (Adaptive Fusion=True, Transition=True, Boundary=False, GRL=True, lambda_bound=0.0)
  - B1:     A4 with lambda_boundary = 0.10
  - B2:     A4 with lambda_boundary = 0.20
  - B3:     A4 with lambda_boundary = 0.50
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
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from sleep_adaptive_net import SleepAdaptiveNet, build_ablation_model
from slamss_ifs_baseline import make_reproducible_split
from slamss_ifs_train import load_night_features
from run_controlled_experiment import ControlledSleepDataset, compute_ece


def evaluate_validation_sweep(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, any]:
    """Detailed validation evaluation including alpha for correct vs incorrect epochs and stage recalls."""
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
            conf = out["confidence"].squeeze(-1)
            ent = out["entropy"].squeeze(-1)
            alpha = out["alpha_gate"].squeeze(-1)
            
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
    
    # Per-Class Metrics: 0: Wake, 1: Light, 2: Deep, 3: REM
    prec, rec, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    # Boundary Error Analysis
    is_correct = (y_true == y_pred)
    err_boundary = 1.0 - is_correct[is_bound].mean() if is_bound.sum() > 0 else 0.0
    err_non_boundary = 1.0 - is_correct[~is_bound].mean() if (~is_bound).sum() > 0 else 0.0
    
    # Uncertainty & Calibration
    ece = compute_ece(probs_concat, y_true, n_bins=10)
    conf_correct = float(conf[is_correct].mean()) if is_correct.sum() > 0 else 0.0
    conf_incorrect = float(conf[~is_correct].mean()) if (~is_correct).sum() > 0 else 0.0
    
    # Reliability Gate Alpha Distribution
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
        "confusion_matrix": cm,
        "boundary_error_rate": float(err_boundary),
        "non_boundary_error_rate": float(err_non_boundary),
        "ece": float(ece),
        "conf_correct": conf_correct,
        "conf_incorrect": conf_incorrect,
        "mean_entropy": float(ent.mean()),
        "alpha_stats": alpha_stats,
        "alpha_by_stage": alpha_by_stage
    }


def train_sweep_model(
    model_name: str,
    base_ablation: str,
    lambda_boundary: float,
    boundary_aware: bool,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_train_subjects: int,
    epochs: int = 10,
    lr: float = 1e-4,
    device: torch.device = torch.device("cuda:0"),
    seed: int = 42
) -> Dict[str, any]:
    """Trains a specific boundary sweep configuration."""
    print(f"\n=======================================================")
    print(f"RUNNING SWEEP [{model_name}]: Base={base_ablation}, boundary_aware={boundary_aware}, lambda={lambda_boundary}")
    print(f"=======================================================")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model = build_ablation_model(
        base_ablation,
        n_train_subjects=n_train_subjects,
        loss_type="rccloss",
        boundary_aware=boundary_aware,
        lambda_boundary=lambda_boundary
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    n_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    
    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for step, batch in enumerate(train_loader):
            x_ihr = batch["ihr"].to(device)
            x_acc = batch["accel"].to(device)
            x_meta = batch["meta"].to(device)
            targets = batch["y"].to(device)
            mask = batch["mask"].to(device)
            subj_idx = batch["subject_idx"].to(device)
            
            optimizer.zero_grad()
            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subj_idx)
            loss = out["total_loss"]
            
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"NaN/Inf loss encountered in {model_name} at epoch {epoch+1}")
                
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        mean_tr = np.mean(batch_losses)
        epoch_losses.append(mean_tr)
        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {mean_tr:.4f}")
        
    train_time = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    print(f"  Training finished in {train_time:.1f} s. Peak VRAM: {peak_vram_mb:.1f} MB.")
    print("  Evaluating on Validation set (28 usable nights)...")
    val_metrics = evaluate_validation_sweep(model, val_loader, device=device)
    
    print(f"  Validation Results [{model_name}]:")
    print(f"    - Accuracy:        {val_metrics['accuracy'] * 100:.2f}%")
    print(f"    - Macro-F1:        {val_metrics['macro_f1']:.4f}")
    print(f"    - Cohen's Kappa:   {val_metrics['kappa']:.4f}")
    print(f"    - Per-stage F1:    Wake={val_metrics['wake_f1']:.3f}, Light={val_metrics['light_f1']:.3f}, "
          f"Deep={val_metrics['deep_f1']:.3f}, REM={val_metrics['rem_f1']:.3f}")
    print(f"    - Wake Recall:     {val_metrics['wake_recall'] * 100:.2f}%, REM Recall: {val_metrics['rem_recall'] * 100:.2f}%")
    print(f"    - Boundary Error:  {val_metrics['boundary_error_rate'] * 100:.2f}% (Non-bound: {val_metrics['non_boundary_error_rate'] * 100:.2f}%)")
    print(f"    - ECE:             {val_metrics['ece']:.4f}")
    print(f"    - Gate alpha:      mean={val_metrics['alpha_stats']['mean']:.3f}, std={val_metrics['alpha_stats']['std']:.3f}")
    print(f"      Correct epochs:  {val_metrics['alpha_stats']['correct_mean']:.3f} +/- {val_metrics['alpha_stats']['correct_std']:.3f}")
    print(f"      Incorrect epochs:{val_metrics['alpha_stats']['incorrect_mean']:.3f} +/- {val_metrics['alpha_stats']['incorrect_std']:.3f}")
    print(f"    - Gate by stage:   {val_metrics['alpha_by_stage']}")
    
    return {
        "model_name": model_name,
        "base_ablation": base_ablation,
        "lambda_boundary": lambda_boundary,
        "boundary_aware": boundary_aware,
        "n_params": n_params,
        "trainable_params": trainable_params,
        "train_time_s": train_time,
        "peak_vram_mb": peak_vram_mb,
        "train_loss_history": epoch_losses,
        "final_train_loss": epoch_losses[-1],
        "val_metrics": val_metrics
    }


def run_exp008():
    print("=" * 75)
    print("EXP-008: CONTROLLED BOUNDARY WEIGHT SWEEP & GRL DISENTANGLEMENT (SEED 42)")
    print("=" * 75)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(device)})")
    
    cache_dir = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
    data_dir = "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    
    mat_files = sorted(glob.glob(os.path.join(data_dir, "Bidslab*", "*", "labels.mat")))
    all_nights = [(f.split(os.sep)[-3], f.split(os.sep)[-2]) for f in mat_files]
    all_subjects = sorted(list(set(s for s, n in all_nights)))
    
    split = make_reproducible_split(all_subjects, split_type="paper", seed=0)
    train_subs = split["train"] # 31 subjects
    val_subs = split["val"]     # 5 subjects
    
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
                
    assert len(train_records) == 160, f"Expected 160 train, got {len(train_records)}"
    assert len(val_records) == 28, f"Expected 28 val, got {len(val_records)}"
    print(f"Loaded 160 usable train nights and 28 usable val nights. Test set strictly excluded.")
    
    train_ds = ControlledSleepDataset(train_records, subject_to_idx=subject_to_idx, seq_len=600)
    val_ds = ControlledSleepDataset(val_records, subject_to_idx=subject_to_idx, seq_len=600)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    # Models to run:
    # B0: A2 reference (Boundary=False, lambda=0.0, GRL=False)
    # B0_GRL: A4 reference with Boundary=False (lambda=0.0, GRL=True)
    # B1: A4 with lambda_boundary = 0.10
    # B2: A4 with lambda_boundary = 0.20
    # B3: A4 with lambda_boundary = 0.50
    sweep_configs = [
        ("B0",     "A2", 0.0,  False),
        ("B0_GRL", "A4", 0.0,  False),
        ("B1",     "A4", 0.10, True),
        ("B2",     "A4", 0.20, True),
        ("B3",     "A4", 0.50, True),
    ]
    
    results = {}
    t_start = time.time()
    for name, base_abl, l_b, b_aware in sweep_configs:
        res = train_sweep_model(
            model_name=name,
            base_ablation=base_abl,
            lambda_boundary=l_b,
            boundary_aware=b_aware,
            train_loader=train_loader,
            val_loader=val_loader,
            n_train_subjects=len(train_subs),
            epochs=10,
            lr=1e-4,
            device=device,
            seed=42
        )
        results[name] = res
        
    total_time = time.time() - t_start
    print("\n" + "=" * 75)
    print(f"EXP-008 COMPLETED IN {total_time:.1f} s ({total_time/60:.2f} min).")
    print("=" * 75)
    
    with open("exp008_boundary_sweep_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved experiment results to exp008_boundary_sweep_results.pkl")
    return results

if __name__ == "__main__":
    run_exp008()
