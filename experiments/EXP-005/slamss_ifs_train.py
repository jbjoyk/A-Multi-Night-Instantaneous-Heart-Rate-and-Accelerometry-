"""
Training and Validation Harness for SLAMSS-IFS Baseline.
Supports:
  - Smoke test (1-2 batches, forward + backward)
  - Short verification run (3-10 epochs on Train + Val only)
  - Full training with validation-based checkpointing
  - Strict preservation of the held-out Test set
"""

import os
import time
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
import scipy.io as sio
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from slamss_ifs_baseline import (
    SLAMSS_IFS,
    RWLoss,
    make_reproducible_split
)


# ==============================================================================
# 1. DATA LOADER FOR SLAMSS-IFS
# ==============================================================================

class SLAMSSDataset(Dataset):
    """
    Dataset for SLAMSS-IFS.
    Reads preprocessed or cached per-night arrays:
      - x_ihr: (T, 30)
      - x_accel: (T, 30)
      - x_freq: (T, 1)
      - x_time: (T, 1)
      - y: (T,)
    Returns full-night padded to max_len (default 1200) or sliced segments.
    """
    def __init__(
        self,
        night_records: List[Dict[str, np.ndarray]],
        max_seq_len: int = 1200,
        pad: bool = True
    ):
        self.records = night_records
        self.max_len = max_seq_len
        self.pad = pad

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        T_orig = len(rec["y"])
        
        # If night is longer than max_len, truncate or slice
        if T_orig >= self.max_len:
            T = self.max_len
            ihr = rec["ihr"][:T]
            acc = rec["accel"][:T]
            freq = rec["freq"][:T]
            time_feat = rec["time"][:T]
            y = rec["y"][:T]
            mask = np.ones(T, dtype=bool)
        else:
            T = T_orig
            if self.pad:
                # Pad to max_len
                pad_len = self.max_len - T
                ihr = np.pad(rec["ihr"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
                acc = np.pad(rec["accel"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
                freq = np.pad(rec["freq"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
                time_feat = np.pad(rec["time"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
                y = np.pad(rec["y"], (0, pad_len), mode="constant", constant_values=-100)
                mask = np.zeros(self.max_len, dtype=bool)
                mask[:T] = True
            else:
                ihr = rec["ihr"]
                acc = rec["accel"]
                freq = rec["freq"]
                time_feat = rec["time"]
                y = rec["y"]
                mask = np.ones(T, dtype=bool)
                
        return {
            "ihr": torch.from_numpy(ihr.astype(np.float32)),
            "accel": torch.from_numpy(acc.astype(np.float32)),
            "freq": torch.from_numpy(freq.astype(np.float32)),
            "time": torch.from_numpy(time_feat.astype(np.float32)),
            "y": torch.from_numpy(y.astype(np.int64)),
            "mask": torch.from_numpy(mask),
            "orig_len": torch.tensor(T_orig, dtype=torch.int64)
        }


# ==============================================================================
# 2. EXTRACT SLAMSS FEATURES FROM PREPROCESSED CACHE
# ==============================================================================

def load_night_features(cache_dir: str, subject: str, night_id: str) -> Optional[Dict[str, np.ndarray]]:
    """Loads a night from cache, extracting IHR, Accel magnitude, Freq, Time, y."""
    fname = os.path.join(cache_dir, f"{subject}__{night_id}.npz")
    if not os.path.exists(fname):
        return None
    data = np.load(fname)
    X = data["X"]          # (T, 10 or 13, 30)
    y_raw = data["y"]      # (T,)
    meta = data["meta"]    # (T, 5)
    
    # 4-class stage mapping:
    # In cache: CANON_STAGES = (Wake=0, N1=1, N2=2, N3=3, REM=4)
    # 4-class grouping: Wake=0, Light=(1,2) -> 1, Deep=3 -> 2, REM=4 -> 3
    y_4class = np.full_like(y_raw, -100)
    y_4class[y_raw == 0] = 0
    y_4class[(y_raw == 1) | (y_raw == 2)] = 1
    y_4class[y_raw == 3] = 2
    y_4class[y_raw == 4] = 3
    
    # IHR is channel 0 (1 Hz instantaneous heart rate)
    ihr = X[:, 0, :]       # (T, 30)
    # Accel magnitude is channel 6
    accel = X[:, 6, :]     # (T, 30)
    # Freq is meta column 0 (sampling rate proxy)
    freq = meta[:, 0:1]    # (T, 1)
    # Time is meta column 2 (elapsed time proxy)
    time_feat = meta[:, 2:3]  # (T, 1)
    
    return {
        "subject": subject,
        "night": night_id,
        "ihr": ihr,
        "accel": accel,
        "freq": freq,
        "time": time_feat,
        "y": y_4class
    }


def prepare_dataset_records(cache_dir: str, split_dict: Dict[str, List[str]]) -> Dict[str, List[Dict[str, np.ndarray]]]:
    """Loads all records partitioned strictly by subject."""
    all_files = sorted(os.listdir(cache_dir))
    records_by_split = {"train": [], "val": [], "test": []}
    
    sub_to_split = {}
    for sp_name, subs in split_dict.items():
        for s in subs:
            sub_to_split[s] = sp_name
            
    for fn in all_files:
        if not fn.endswith(".npz"):
            continue
        stem = fn[:-4]
        sub, night = stem.split("__")
        if sub in sub_to_split:
            sp = sub_to_split[sub]
            rec = load_night_features(cache_dir, sub, night)
            if rec is not None and len(rec["y"]) >= 600:  # >= 5 hours
                records_by_split[sp].append(rec)
                
    return records_by_split


# ==============================================================================
# 3. EVALUATION FUNCTION (TRAIN & VALIDATION ONLY)
# ==============================================================================

def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates model without computing gradients."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            ihr = batch["ihr"].to(device)
            acc = batch["accel"].to(device)
            freq = batch["freq"].to(device)
            time_feat = batch["time"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(ihr, acc, freq, time_feat, mask=mask)
                loss = loss_fn(logits, y, mask=mask)
                
            total_loss += loss.item() * ihr.size(0)
            
            preds = torch.argmax(logits, dim=-1)
            
            # Mask out invalid / padded / unknown targets
            valid = mask & (y >= 0) & (y < 4)
            valid_p = preds[valid].cpu().numpy()
            valid_t = y[valid].cpu().numpy()
            
            all_preds.extend(valid_p)
            all_targets.extend(valid_t)
            
    n_samples = len(dataloader.dataset)
    mean_loss = total_loss / max(n_samples, 1)
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    if len(all_targets) == 0:
        return {"loss": mean_loss, "accuracy": 0.0, "macro_f1": 0.0, "kappa": 0.0}
        
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(all_targets, all_preds)
    
    per_class_f1 = f1_score(all_targets, all_preds, average=None, labels=[0, 1, 2, 3], zero_division=0)
    
    return {
        "loss": mean_loss,
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "kappa": float(kappa),
        "f1_wake": float(per_class_f1[0]),
        "f1_light": float(per_class_f1[1]),
        "f1_deep": float(per_class_f1[2]),
        "f1_rem": float(per_class_f1[3])
    }


# ==============================================================================
# 4. TRAINING ROUTINE
# ==============================================================================

def train_slamss_ifs(
    train_records: List[Dict[str, np.ndarray]],
    val_records: List[Dict[str, np.ndarray]],
    epochs: int = 5,
    effective_batch_size: int = 20,
    micro_batch_size: int = 4,
    learning_rate: float = 1.5e-4,
    save_path: str = "slamss_ifs_best.pt",
    device: Optional[torch.device] = None,
    seed: int = 42
) -> Dict[str, List[float]]:
    """
    Trains the SLAMSS-IFS model according to paper specifications:
      - Optimizer: Adam
      - Learning rate: 1.5e-4
      - Effective batch size: 20 (via micro_batch_size=4 and gradient accumulation steps=5)
      - Checkpoint saved on best Validation Macro-F1
    """
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining on Device: {device}")
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Calculate gradient accumulation steps
    accum_steps = max(1, effective_batch_size // micro_batch_size)
    print(f"Batch config: Micro-batch = {micro_batch_size}, Accumulation steps = {accum_steps} (Effective Batch = {micro_batch_size * accum_steps})")
    
    # Datasets and Loaders
    train_dataset = SLAMSSDataset(train_records, max_seq_len=1200, pad=True)
    val_dataset = SLAMSSDataset(val_records, max_seq_len=1200, pad=True)
    
    train_loader = DataLoader(train_dataset, batch_size=micro_batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=micro_batch_size, shuffle=False)
    
    # Initialize Model, Loss, Optimizer
    model = SLAMSS_IFS(n_classes=4).to(device)
    
    # Calculate inverse class frequency weights on training data for RW Loss
    all_tr_y = np.concatenate([r["y"][r["y"] >= 0] for r in train_records])
    counts = np.bincount(all_tr_y, minlength=4)
    if_weights = counts.sum() / (4.0 * np.maximum(counts, 1.0))
    if_weights = torch.tensor(if_weights / if_weights.mean(), dtype=torch.float32, device=device)
    print(f"Training Class Counts: {counts.tolist()} -> Inverse Frequency Weights: {[round(w.item(), 3) for w in if_weights]}")
    
    loss_fn = RWLoss(n_classes=4, fn_weights=if_weights, ignore_index=-100).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    
    best_val_f1 = -1.0
    history = {"train_loss": [], "val_loss": [], "val_macro_f1": [], "val_acc": [], "val_kappa": []}
    
    print("\n" + "-" * 75)
    print(f"{'Epoch':<6} | {'Train Loss':<11} | {'Val Loss':<10} | {'Val Acc':<9} | {'Val Macro-F1':<13} | {'Val Kappa':<10} | {'Status'}")
    print("-" * 75)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            ihr = batch["ihr"].to(device)
            acc = batch["accel"].to(device)
            freq = batch["freq"].to(device)
            time_feat = batch["time"].to(device)
            y = batch["y"].to(device)
            mask = batch["mask"].to(device)
            
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(ihr, acc, freq, time_feat, mask=mask)
                loss = loss_fn(logits, y, mask=mask)
                loss_scaled = loss / accum_steps
                
            scaler.scale(loss_scaled).backward()
            running_loss += loss.item() * ihr.size(0)
            
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
        epoch_tr_loss = running_loss / len(train_dataset)
        
        # Validation evaluation
        val_metrics = evaluate_model(model, val_loader, loss_fn, device)
        val_f1 = val_metrics["macro_f1"]
        
        history["train_loss"].append(epoch_tr_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_macro_f1"].append(val_f1)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_kappa"].append(val_metrics["kappa"])
        
        status = ""
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
                "if_weights": if_weights
            }, save_path)
            status = f"* BEST (Saved)"
            
        dur = time.time() - t0
        print(f"{epoch:<6d} | {epoch_tr_loss:<11.4f} | {val_metrics['loss']:<10.4f} | {val_metrics['accuracy']:<9.4f} | {val_f1:<13.4f} | {val_metrics['kappa']:<10.4f} | {status} ({dur:.1f}s)")
        
    print("-" * 75)
    print(f"Finished {epochs} epochs. Best Validation Macro-F1: {best_val_f1:.4f} (Saved to {save_path})")
    return history


# ==============================================================================
# 5. CLI RUNNER FOR SMOKE TEST AND SHORT RUN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLAMSS-IFS Training and Verification")
    parser.add_argument("--mode", choices=["smoke", "short", "full"], default="smoke",
                        help="Execution mode: 'smoke' (2 batches), 'short' (3-5 epochs), 'full' (500 epochs)")
    parser.add_argument("--split", choices=["paper", "current"], default="paper",
                        help="Subject split mode: 'paper' (31/5/11) or 'current' (33/7/7)")
    parser.add_argument("--epochs", type=int, default=3, help="Epoch count for short or full mode")
    parser.add_argument("--cache_dir", type=str,
                        default="/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b",
                        help="Path to preprocessed cache")
    args = parser.parse_args()
    
    print("=" * 65)
    print(f"SLAMSS-IFS RUNNER: MODE = {args.mode.upper()}, SPLIT = {args.split.upper()}")
    print("=" * 65)
    
    data_dir = "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith("Bidslab")])
    split_dict = make_reproducible_split(subjects, split_type=args.split, seed=0)
    print(f"Partitioned Subjects: Train={len(split_dict['train'])}, Val={len(split_dict['val'])}, Test={len(split_dict['test'])}")
    
    print("Loading night records from cache...")
    records = prepare_dataset_records(args.cache_dir, split_dict)
    print(f"Loaded Nights: Train={len(records['train'])}, Val={len(records['val'])}, Test={len(records['test'])}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    if args.mode == "smoke":
        print("\n--- RUNNING SMOKE TEST (2 batches forward & backward) ---")
        tiny_train = records["train"][:4]
        tiny_val = records["val"][:2]
        history = train_slamss_ifs(
            tiny_train,
            tiny_val,
            epochs=2,
            effective_batch_size=4,
            micro_batch_size=2,
            learning_rate=1.5e-4,
            save_path="slamss_smoke_test.pt",
            device=device,
            seed=42
        )
        print("Smoke test successfully executed!")
        
    elif args.mode == "short":
        print(f"\n--- RUNNING SHORT VERIFICATION RUN ({args.epochs} epochs on Train + Val only) ---")
        history = train_slamss_ifs(
            records["train"],
            records["val"],
            epochs=args.epochs,
            effective_batch_size=20,
            micro_batch_size=4,
            learning_rate=1.5e-4,
            save_path="slamss_ifs_short_run.pt",
            device=device,
            seed=42
        )
        print("Short verification run successfully executed!")
