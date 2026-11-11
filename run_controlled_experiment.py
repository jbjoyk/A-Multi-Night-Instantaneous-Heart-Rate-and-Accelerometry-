"""
EXP-007: First Controlled Scientific Experiment (Ablations A0 - A5 on Seed 42).

Strict Scientific Controls:
  - Seed: 42 (deterministic initialization, shuffling, CUDA)
  - Train: Exactly 31 paper subjects (160 usable nights)
  - Val: Exactly 5 paper subjects (28 usable nights)
  - Test: 11 subjects (58 nights) -> STRICTLY NOT LOADED / NOT USED
  - Sequence length: 600 epochs (5 continuous hours, 0 padding artifacts)
  - Batch size: 4
  - Optimizer: Adam (lr = 1e-4, weight_decay = 0.0)
  - Epochs: 10 epochs per ablation
  - Loss: RCCLoss + auxiliary ablation objectives
  - Identical data loader, optimizer, initialization policy across A0-A5
"""

import os
import sys
import glob
import time
import copy
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support

from sleep_adaptive_net import SleepAdaptiveNet, build_ablation_model
from slamss_ifs_baseline import make_reproducible_split
from slamss_ifs_train import load_night_features


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across 10 probability bins."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


class ControlledSleepDataset(Dataset):
    """
    Dataset for controlled ablation study.
    Fixed sequence length T = 600 epochs (5.0 hours).
    Every night in dataset has >= 610 epochs -> 0 zero-padding for any recording.
    """
    def __init__(
        self,
        records: List[Dict[str, np.ndarray]],
        subject_to_idx: Dict[str, int],
        seq_len: int = 600
    ):
        self.records = records
        self.subject_to_idx = subject_to_idx
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        subj = rec["subject"]
        subj_idx = self.subject_to_idx.get(subj, -100)
        
        T = self.seq_len
        ihr = rec["ihr"][:T]
        acc = rec["accel"][:T]
        freq = rec["freq"][:T]
        time_feat = rec["time"][:T]
        y = rec["y"][:T]
        
        hr_mean = ihr.mean(axis=1, keepdims=True)
        hr_std = ihr.std(axis=1, keepdims=True)
        clock = np.zeros_like(time_feat)
        meta = np.concatenate([freq, clock, time_feat, hr_mean, hr_std], axis=-1)
        mask = np.ones(T, dtype=bool)
        
        return {
            "ihr": torch.from_numpy(ihr.astype(np.float32)),
            "accel": torch.from_numpy(acc.astype(np.float32)),
            "meta": torch.from_numpy(meta.astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.int64)),
            "mask": torch.from_numpy(mask),
            "subject_idx": torch.tensor(subj_idx, dtype=torch.int64),
            "subject": subj,
            "night": rec["night"]
        }


def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Dict[str, any]:
    """Evaluates validation set only. Computes primary, per-class, boundary, uncertainty, and gate metrics."""
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
        "max": float(alpha.max())
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


def train_and_evaluate_ablation(
    ablation: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_train_subjects: int,
    epochs: int = 10,
    lr: float = 1e-4,
    device: torch.device = torch.device("cuda:0"),
    seed: int = 42
) -> Dict[str, any]:
    """Trains one ablation model under strictly controlled conditions and returns full log."""
    print(f"\n=======================================================")
    print(f"RUNNING ABLATION [{ablation}] (Seed {seed}, Epochs {epochs})")
    print(f"=======================================================")
    
    # Enforce strict determinism
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model = build_ablation_model(ablation, n_train_subjects=n_train_subjects, loss_type="rccloss").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    n_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    
    epoch_train_losses = []
    
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
            
            # Check for NaN / Inf
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"Abnormal loss encountered in {ablation} at epoch {epoch+1}, step {step+1}: {loss.item()}")
                
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        mean_tr_loss = np.mean(batch_losses)
        epoch_train_losses.append(mean_tr_loss)
        print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {mean_tr_loss:.4f}")
        
    train_time = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    print(f"  Training finished in {train_time:.1f} s. Peak VRAM: {peak_vram_mb:.1f} MB.")
    print("  Evaluating on Validation set (28 usable nights)...")
    val_metrics = evaluate_validation(model, val_loader, device=device)
    
    print(f"  Validation Results [{ablation}]:")
    print(f"    - Accuracy:        {val_metrics['accuracy'] * 100:.2f}%")
    print(f"    - Macro-F1:        {val_metrics['macro_f1']:.4f}")
    print(f"    - Cohen's Kappa:   {val_metrics['kappa']:.4f}")
    print(f"    - Per-stage F1:    Wake={val_metrics['wake_f1']:.3f}, Light={val_metrics['light_f1']:.3f}, "
          f"Deep={val_metrics['deep_f1']:.3f}, REM={val_metrics['rem_f1']:.3f}")
    print(f"    - Boundary Error:  {val_metrics['boundary_error_rate'] * 100:.2f}% (Non-bound: {val_metrics['non_boundary_error_rate'] * 100:.2f}%)")
    print(f"    - ECE:             {val_metrics['ece']:.4f}")
    if ablation != "A0":
        print(f"    - Gate alpha:      mean={val_metrics['alpha_stats']['mean']:.3f}, std={val_metrics['alpha_stats']['std']:.3f}")
        print(f"    - Gate by stage:   {val_metrics['alpha_by_stage']}")
        
    return {
        "ablation": ablation,
        "n_params": n_params,
        "trainable_params": trainable_params,
        "train_time_s": train_time,
        "peak_vram_mb": peak_vram_mb,
        "train_loss_history": epoch_train_losses,
        "final_train_loss": epoch_train_losses[-1],
        "val_metrics": val_metrics
    }


def run_all_ablations_seed42(epochs: int = 10, batch_size: int = 4):
    print("=" * 75)
    print("EXP-007: CONTROLLED SCIENTIFIC ABLATION EXPERIMENT (A0 - A5, SEED 42)")
    print("=" * 75)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(device)})")
    
    cache_dir = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
    data_dir = "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    
    # 1. Subject Discovery and Protocol Split
    mat_files = sorted(glob.glob(os.path.join(data_dir, "Bidslab*", "*", "labels.mat")))
    all_nights = [(f.split(os.sep)[-3], f.split(os.sep)[-2]) for f in mat_files]
    all_subjects = sorted(list(set(s for s, n in all_nights)))
    
    split = make_reproducible_split(all_subjects, split_type="paper", seed=0)
    train_subs = split["train"] # 31 subjects
    val_subs = split["val"]     # 5 subjects
    test_subs = split["test"]   # 11 subjects
    
    subject_to_idx = {s: i for i, s in enumerate(train_subs)}
    
    # 2. Load ALL eligible Train and Val nights (Test nights are NEVER loaded)
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
                
    print(f"\nDataset Partitions Loaded:")
    print(f"  - Training Usable Nights:    {len(train_records)} across {len(train_subs)} subjects")
    print(f"  - Validation Usable Nights:  {len(val_records)} across {len(val_subs)} subjects")
    print(f"  - Test Set (11 subjects, 58 nights): STRICTLY EXCLUDED (Not Loaded)")
    
    assert len(train_records) == 160, f"Expected 160 usable train nights, got {len(train_records)}"
    assert len(val_records) == 28, f"Expected 28 usable val nights, got {len(val_records)}"
    
    train_ds = ControlledSleepDataset(train_records, subject_to_idx=subject_to_idx, seq_len=600)
    val_ds = ControlledSleepDataset(val_records, subject_to_idx=subject_to_idx, seq_len=600)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    # 3. Train A0 through A5 in sequence
    results = {}
    ablations = ["A0", "A1", "A2", "A3", "A4", "A5"]
    
    t_suite_start = time.time()
    for abl in ablations:
        res = train_and_evaluate_ablation(
            ablation=abl,
            train_loader=train_loader,
            val_loader=val_loader,
            n_train_subjects=len(train_subs),
            epochs=epochs,
            lr=1e-4,
            device=device,
            seed=42
        )
        results[abl] = res
        
    suite_time = time.time() - t_suite_start
    print("\n" + "=" * 75)
    print(f"ALL 6 ABLATIONS COMPLETED IN {suite_time:.1f} s ({suite_time/60:.2f} min).")
    print("=" * 75)
    
    # Save results dictionary
    import pickle
    with open("exp007_controlled_ablation_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved experiment results to exp007_controlled_ablation_results.pkl")
    return results

if __name__ == "__main__":
    run_all_ablations_seed42(epochs=10, batch_size=4)
