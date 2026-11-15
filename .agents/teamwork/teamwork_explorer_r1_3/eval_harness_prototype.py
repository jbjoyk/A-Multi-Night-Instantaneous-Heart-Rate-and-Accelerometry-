"""
Prototype Standalone 1-Epoch Evaluation Harness for Fold 1
Location: .agents/teamwork/teamwork_explorer_r1_3/eval_harness_prototype.py
"""

import os
import sys
import glob
import json
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, precision_recall_fscore_support

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from slamss_ifs_train import load_night_features
from run_controlled_experiment import ControlledSleepDataset
from exp013_models import ConventionalCNNBiGRU

CACHE_DIR = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
DATA_DIR = os.path.join(WORKSPACE_ROOT, "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0")
MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014/EXP014_MANIFESTS/fold_1.json")


def load_fold1_data(batch_size: int = 4, seq_len: int = 600):
    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    inner_train_subs = manifest["inner_train_subjects"]
    inner_val_subs = manifest["inner_val_subjects"]
    outer_test_subs = manifest["outer_test_subjects"]

    all_mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "Bidslab*", "*", "labels.mat")))
    all_nights_dataset = [(f.split(os.sep)[-3], f.split(os.sep)[-2].replace("Night ", "")) for f in all_mat_files]

    records_by_sub = {}
    for s, n in all_nights_dataset:
        rec = load_night_features(CACHE_DIR, s, n)
        if rec is not None:
            if s not in records_by_sub:
                records_by_sub[s] = []
            records_by_sub[s].append(rec)

    train_records = [r for s in inner_train_subs for r in records_by_sub.get(s, [])]
    val_records = [r for s in inner_val_subs for r in records_by_sub.get(s, [])]
    test_records = [r for s in outer_test_subs for r in records_by_sub.get(s, [])]

    subject_to_idx = {s: i for i, s in enumerate(inner_train_subs)}

    train_ds = ControlledSleepDataset(train_records, subject_to_idx=subject_to_idx, seq_len=seq_len)
    test_ds = ControlledSleepDataset(test_records, subject_to_idx=subject_to_idx, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, test_loader, manifest


def evaluate_1epoch_fold1(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    lr: float = 1e-4,
    seed: int = 42
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    # 1 Epoch of Training
    model.train()
    batch_losses = []
    t_start = time.time()
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

    train_time = time.time() - t_start
    mean_train_loss = float(np.mean(batch_losses))

    # Evaluation on Outer Test Partition
    model.eval()
    all_preds = []
    all_targets = []
    t_eval_start = time.time()

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
            preds = torch.argmax(logits, dim=-1)

            valid = (targets >= 0) & (targets < 4) & mask
            all_preds.extend(preds[valid].cpu().numpy())
            all_targets.extend(targets[valid].cpu().numpy())

    eval_time = time.time() - t_eval_start
    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )

    stage_names = ["Wake", "Light", "Deep", "REM"]
    per_class = {
        f"{stage_names[i]}_f1": float(f1_cls[i]) for i in range(4)
    }

    results = {
        "mean_train_loss": mean_train_loss,
        "train_time_sec": train_time,
        "eval_time_sec": eval_time,
        "n_evaluated_windows": len(y_true),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "kappa": kappa,
        "per_class_f1": per_class,
        "support": [int(s) for s in supp]
    }
    return results


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_loader, test_loader, manifest = load_fold1_data(batch_size=4, seq_len=600)
    print(f"Fold 1: {len(train_loader)} train batches ({len(train_loader.dataset)} nights), {len(test_loader)} test batches ({len(test_loader.dataset)} nights)")

    model = ConventionalCNNBiGRU()
    print(f"Model A parameters: {sum(p.numel() for p in model.parameters()):,}")

    results = evaluate_1epoch_fold1(model, train_loader, test_loader, device=device)
    print("\n--- 1-Epoch Evaluation Results ---")
    print(f"Train Loss: {results['mean_train_loss']:.4f} (took {results['train_time_sec']:.2f}s)")
    print(f"Eval Time: {results['eval_time_sec']:.2f}s, Windows: {results['n_evaluated_windows']}")
    print(f"Overall Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Macro-F1: {results['macro_f1']:.4f}")
    print(f"Cohen's Kappa: {results['kappa']:.4f}")
    print(f"Per-class F1: {results['per_class_f1']}")
