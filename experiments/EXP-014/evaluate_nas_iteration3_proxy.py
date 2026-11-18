"""
NAS Iteration 3: Fast Proxy Evaluation on Fold 1 (50 Epochs, Seed 42, cuda:0)
File: experiments/EXP-014/evaluate_nas_iteration3_proxy.py

Execution Specifications:
  - Architecture: Candidate Alpha (CircadianCrossModalCNNBiGRU, 190,502 parameters)
  - Dataset Partition: Fold 1 (experiments/EXP-014/EXP014_MANIFESTS/fold_1.json)
      Inner Train: 32 subjects (168 nights, 42 batches/epoch)
      Inner Val: 5 subjects (24 nights, 6 batches)
      Outer Test: 10 subjects (54 nights, 14 batches, 32,098 valid windows)
  - Preprocessed Feature Cache:
      /home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b
  - Optimization:
      50 epochs
      Batch size = 4
      Adam(lr=1e-4, weight_decay=0.0)
      Grad clipping = 1.0 (torch.nn.utils.clip_grad_norm_)
      Sequence length = 600 epochs
      Loss = CrossEntropy + 0.01 * RCCLoss
      Seed = 42
      Device = cuda:0
  - Output Artifact:
      experiments/EXP-014/nas_iteration3_proxy_results.json
  - Gate Filter Decision:
      PASS if Proxy Accuracy >= 70.0% and Proxy Weighted-F1 >= 0.6900
      REVERT if Proxy Accuracy < 70.0% or Proxy Weighted-F1 < 0.6900
"""

import os
import sys
import glob
import json
import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support
)

# Workspace paths
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

EXP014_DIR = os.path.dirname(__file__)
if EXP014_DIR not in sys.path:
    sys.path.insert(0, EXP014_DIR)

from slamss_ifs_train import load_night_features
from run_controlled_experiment import ControlledSleepDataset
from nas_iteration3_model import CircadianCrossModalCNNBiGRU

CACHE_DIR = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
DATA_DIR = os.path.join(WORKSPACE_ROOT, "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0")
MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014/EXP014_MANIFESTS/fold_1.json")
RESULTS_OUTPUT_PATH = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014/nas_iteration3_proxy_results.json")

# Gating Thresholds per Requirement R1.3
PROXY_ACCURACY_THRESHOLD = 0.7000      # 70.0%
PROXY_WEIGHTED_F1_THRESHOLD = 0.6900   # 0.6900


def load_fold1_data(
    batch_size: int = 4,
    seq_len: int = 600
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """Loads Fold 1 inner-train, inner-val, and outer-test datasets."""
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
    val_ds = ControlledSleepDataset(val_records, subject_to_idx=subject_to_idx, seq_len=seq_len)
    test_ds = ControlledSleepDataset(test_records, subject_to_idx=subject_to_idx, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader, manifest


def evaluate_dataset(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """Runs evaluation on a partition (val or outer test)."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x_ihr = batch["ihr"].to(device)
            x_acc = batch["accel"].to(device)
            x_meta = batch["meta"].to(device)
            targets = batch["y"].to(device)
            mask = batch["mask"].to(device)
            subj_idx = batch["subject_idx"].to(device)
            seq_lens = batch.get("seq_len", None)
            if seq_lens is not None:
                seq_lens = seq_lens.to(device)

            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subj_idx, seq_lens=seq_lens)
            logits = out["logits"]
            preds = torch.argmax(logits, dim=-1)

            valid = (targets >= 0) & (targets < 4) & mask
            all_preds.extend(preds[valid].cpu().numpy())
            all_targets.extend(targets[valid].cpu().numpy())

    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()

    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )

    stage_names = ["Wake", "Light", "Deep", "REM"]
    per_class_f1 = {stage_names[i]: float(f1_cls[i]) for i in range(4)}
    per_class_recall = {stage_names[i]: float(rec[i]) for i in range(4)}
    per_class_precision = {stage_names[i]: float(prec[i]) for i in range(4)}

    return {
        "n_windows": len(y_true),
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "kappa": kappa,
        "confusion_matrix": cm,
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "support": [int(s) for s in supp]
    }


def run_proxy_evaluation(
    epochs: int = 50,
    seed: int = 42,
    lr: float = 1e-4,
    batch_size: int = 4,
    device_str: str = "cuda:0"
) -> Dict[str, Any]:
    """Executes the full 50-epoch proxy training on Fold 1 and evaluates the proxy gate."""
    print("=" * 80)
    print("NAS ITERATION 3: FAST PROXY EVALUATION ON FOLD 1")
    print("Candidate Alpha: CircadianCrossModalCNNBiGRU (190,502 parameters)")
    print(f"Schedule: {epochs} epochs | Seed: {seed} | LR: {lr} | Batch Size: {batch_size} | Device: {device_str}")
    print("=" * 80)

    # Determinism
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = CircadianCrossModalCNNBiGRU().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Instantiated CircadianCrossModalCNNBiGRU with {total_params:,} parameters.")
    assert total_params == 190502, f"Expected 190,502 parameters, got {total_params}"

    train_loader, val_loader, test_loader, manifest = load_fold1_data(batch_size=batch_size, seq_len=600)
    print(f"Data Loaded: Train {len(train_loader)} batches | Val {len(val_loader)} batches | Test {len(test_loader)} batches")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    t_start = time.time()
    epoch_history = []

    print("\nStarting 50-Epoch Training Loop...")
    for ep in range(1, epochs + 1):
        t_ep_start = time.time()
        model.train()
        batch_losses = []

        for batch in train_loader:
            x_ihr = batch["ihr"].to(device)
            x_acc = batch["accel"].to(device)
            x_meta = batch["meta"].to(device)
            targets = batch["y"].to(device)
            mask = batch["mask"].to(device)
            subj_idx = batch["subject_idx"].to(device)
            seq_lens = batch.get("seq_len", None)
            if seq_lens is not None:
                seq_lens = seq_lens.to(device)

            optimizer.zero_grad()
            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subj_idx, seq_lens=seq_lens)
            loss = out["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        ep_duration = time.time() - t_ep_start
        mean_loss = float(np.mean(batch_losses))
        epoch_history.append({"epoch": ep, "train_loss": mean_loss, "epoch_sec": ep_duration})

        if ep % 10 == 0 or ep == epochs:
            elapsed = time.time() - t_start
            print(f"Epoch {ep:02d}/{epochs} ({elapsed:.1f}s) | Train Loss: {mean_loss:.4f} | Ep Duration: {ep_duration:.2f}s")

    total_train_time = time.time() - t_start
    print(f"\nTraining completed in {total_train_time:.2f}s (~{total_train_time/60:.2f}m).")

    # Evaluate on Inner Val Partition
    print("\nEvaluating on Inner Validation Partition (5 subjects, 24 nights)...")
    val_res = evaluate_dataset(model, val_loader, device)
    print(f"  Val Accuracy:    {val_res['accuracy']*100:.2f}%")
    print(f"  Val Weighted-F1: {val_res['weighted_f1']:.4f}")
    print(f"  Val Macro-F1:    {val_res['macro_f1']:.4f}")

    # Evaluate on Outer Test Partition (Locked 10 subjects, 54 nights, 32,098 windows)
    print("\nEvaluating on Outer Test Partition (10 subjects, 54 nights, 32,098 windows)...")
    test_res = evaluate_dataset(model, test_loader, device)

    proxy_acc = test_res["accuracy"]
    proxy_weighted_f1 = test_res["weighted_f1"]
    proxy_macro_f1 = test_res["macro_f1"]
    proxy_kappa = test_res["kappa"]

    passed_acc = proxy_acc >= PROXY_ACCURACY_THRESHOLD
    passed_f1 = proxy_weighted_f1 >= PROXY_WEIGHTED_F1_THRESHOLD
    proxy_passed = passed_acc and passed_f1
    action = "PASS" if proxy_passed else "REVERT"

    print("\n" + "=" * 80)
    print("FAST PROXY EVALUATION RESULTS & GATE VERDICT")
    print("=" * 80)
    print(f"  Model Architecture: Candidate Alpha (CircadianCrossModalCNNBiGRU)")
    print(f"  Parameters:         {total_params:,}")
    print(f"  Evaluated Windows:  {test_res['n_windows']:,}")
    print(f"  Proxy Accuracy:     {proxy_acc*100:.2f}% (Threshold: >= {PROXY_ACCURACY_THRESHOLD*100:.1f}% -> {'PASSED' if passed_acc else 'FAILED'})")
    print(f"  Proxy Weighted-F1:  {proxy_weighted_f1:.4f} (Threshold: >= {PROXY_WEIGHTED_F1_THRESHOLD:.4f} -> {'PASSED' if passed_f1 else 'FAILED'})")
    print(f"  Proxy Macro-F1:     {proxy_macro_f1:.4f}")
    print(f"  Cohen's Kappa:      {proxy_kappa:.4f}")
    print("  Per-class F1 Scores:")
    for stage, f1 in test_res["per_class_f1"].items():
        rec = test_res["per_class_recall"][stage]
        prec = test_res["per_class_precision"][stage]
        print(f"    - {stage:5s}: F1 = {f1:.4f} | Recall = {rec*100:.2f}% | Precision = {prec*100:.2f}%")
    print(f"  Confusion Matrix (Wake, Light, Deep, REM):\n    {test_res['confusion_matrix']}")
    print(f"  PROXY GATE ACTION:  {action}")
    print("=" * 80)

    result_bundle = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iteration": 3,
        "candidate": "Candidate Alpha (CircadianCrossModalCNNBiGRU)",
        "parameters": total_params,
        "fold": 1,
        "epochs": epochs,
        "seed": seed,
        "device": str(device),
        "train_time_sec": total_train_time,
        "val_metrics": val_res,
        "outer_test_metrics": test_res,
        "gating": {
            "proxy_accuracy": proxy_acc,
            "accuracy_threshold": PROXY_ACCURACY_THRESHOLD,
            "passed_accuracy": passed_acc,
            "proxy_weighted_f1": proxy_weighted_f1,
            "weighted_f1_threshold": PROXY_WEIGHTED_F1_THRESHOLD,
            "passed_weighted_f1": passed_f1,
            "proxy_passed": proxy_passed,
            "action": action
        },
        "epoch_history": epoch_history
    }

    with open(RESULTS_OUTPUT_PATH, "w") as f:
        json.dump(result_bundle, f, indent=2)
    print(f"\nSaved proxy evaluation report to: {RESULTS_OUTPUT_PATH}")

    return result_bundle


if __name__ == "__main__":
    run_proxy_evaluation()
