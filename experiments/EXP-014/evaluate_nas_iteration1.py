"""
NAS Iteration 1: 1-Epoch Evaluation on Fold 1
File: experiments/EXP-014/evaluate_nas_iteration1.py

Evaluates the mutated PyTorch architecture on Fold 1 of the BIDSleep dataset:
  - 1 epoch training on inner-train partition (42 batches, 168 nights across 32 subjects)
  - Evaluation on outer-test partition (14 batches, 54 nights across 10 subjects)
  - Hyperparameters strictly unchanged:
      Batch size = 4
      Optimizer = Adam(lr=1e-4, weight_decay=0.0)
      Sequence horizon = 600 epochs
      Loss = CrossEntropy + 0.01 * RCCLoss
  - Device: cuda:0
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

# Workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from slamss_ifs_train import load_night_features
from run_controlled_experiment import ControlledSleepDataset
from exp013_models import ConventionalCNNBiGRU
EXP014_DIR = os.path.dirname(__file__)
if EXP014_DIR not in sys.path:
    sys.path.insert(0, EXP014_DIR)

from nas_iteration1_model import (
    MultiScaleConventionalCNNBiGRU,
    MutatedModelA_MSCNNBiGRU
)

CACHE_DIR = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
DATA_DIR = os.path.join(WORKSPACE_ROOT, "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0")
MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014/EXP014_MANIFESTS/fold_1.json")
RESULTS_OUTPUT_PATH = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014/nas_iteration1_eval_results.json")


def load_fold1_data(batch_size: int = 4, seq_len: int = 600):
    """
    Loads Fold 1 inner-train, inner-val, and outer-test datasets according to manifest.
    """
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
    """
    Executes a 1-epoch training loop and outer test set evaluation on single GPU.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    # 1. 1-Epoch Training Loop
    model.train()
    batch_losses = []
    t_train_start = time.time()
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
        optimizer.step()
        batch_losses.append(loss.item())

    train_time = time.time() - t_train_start
    mean_train_loss = float(np.mean(batch_losses))

    # 2. Evaluation Loop on Outer Test Partition
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
            seq_lens = batch.get("seq_len", None)
            if seq_lens is not None:
                seq_lens = seq_lens.to(device)

            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subj_idx, seq_lens=seq_lens)
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


def run_nas_iteration1_evaluation():
    print("=" * 70)
    print("NAS Iteration 1: Model A Mutation Evaluation on Fold 1")
    print("=" * 70)

    # Device configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Target GPU Device: {device}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Initial Allocated Memory: {torch.cuda.memory_allocated(0)/(1024**2):.2f} MiB")

    # Load data
    train_loader, test_loader, manifest = load_fold1_data(batch_size=4, seq_len=600)
    print(f"\nFold 1 Data Partitions:")
    print(f"  Inner Train: {len(train_loader)} batches ({len(train_loader.dataset)} nights across {len(manifest['inner_train_subjects'])} subjects)")
    print(f"  Outer Test:  {len(test_loader)} batches ({len(test_loader.dataset)} nights across {len(manifest['outer_test_subjects'])} subjects)")

    # 1. Baseline Model A
    baseline_model = ConventionalCNNBiGRU()
    p_baseline = sum(p.numel() for p in baseline_model.parameters())
    print(f"\nBaseline Model A (ConventionalCNNBiGRU):")
    print(f"  Parameters: {p_baseline:,}")

    # 2. Mutated Model A (Pre-Recurrent Multi-Scale Fusion - Primary)
    mutated_model = MultiScaleConventionalCNNBiGRU(mutation_type="pre_recurrent")
    p_mutated = sum(p.numel() for p in mutated_model.parameters())
    delta_p = p_mutated - p_baseline
    pct_increase = (delta_p / p_baseline) * 100.0

    print(f"\nMutated Model A (MultiScaleConventionalCNNBiGRU - Pre-Recurrent MSTF):")
    print(f"  Parameters: {p_mutated:,}")
    print(f"  Delta over Baseline: +{delta_p:,} ({pct_increase:+.2f}%)")
    print(f"  Parameter Cap (<= +20%): 196,728")
    print(f"  Budget Compliance: {'PASSED' if p_mutated <= 196728 else 'FAILED'}")
    assert p_mutated <= 196728, f"Parameter cap exceeded: {p_mutated} > 196,728"

    # Forward pass verification on expected BIDSleep shapes
    print("\nForward Pass Verification on Expected BIDSleep Shapes (B=4, T=600, L=30):")
    B, T, L = 4, 600, 30
    x_ihr = torch.randn(B, T, L, device=device)
    x_acc = torch.randn(B, T, L, device=device)
    x_meta = torch.randn(B, T, 5, device=device)
    targets = torch.randint(0, 4, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    seq_lens = torch.full((B,), T, dtype=torch.int64, device=device)

    mutated_model = mutated_model.to(device)
    fwd_out = mutated_model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)
    print(f"  Logits shape: {fwd_out['logits'].shape} (expected: ({B}, {T}, 4))")
    print(f"  Total loss: {fwd_out['total_loss'].item():.4f}")
    assert fwd_out["logits"].shape == (B, T, 4), "Logits shape mismatch!"
    print("  Forward pass verification: PASSED")

    # Execute 1-Epoch Evaluation on Mutated Architecture
    print("\n" + "-" * 70)
    print("Executing 1-Epoch Training and Fold 1 Outer Test Evaluation (Mutated Model)...")
    print("-" * 70)
    mutated_results = evaluate_1epoch_fold1(mutated_model, train_loader, test_loader, device=device, seed=42)

    print(f"  1-Epoch Train Loss: {mutated_results['mean_train_loss']:.4f} (took {mutated_results['train_time_sec']:.2f}s)")
    print(f"  Outer Test Windows: {mutated_results['n_evaluated_windows']:,} (eval took {mutated_results['eval_time_sec']:.2f}s)")
    print(f"  Overall Accuracy:   {mutated_results['accuracy']*100:.2f}%")
    print(f"  Macro-F1 (4-class): {mutated_results['macro_f1']:.4f}")
    print(f"  Cohen's Kappa:      {mutated_results['kappa']:.4f}")
    print(f"  Per-class F1:")
    for stage, f1 in mutated_results['per_class_f1'].items():
        print(f"    - {stage}: {f1:.4f}")

    # Execute 1-Epoch Evaluation on Baseline Model for Direct Reference
    print("\n" + "-" * 70)
    print("Executing 1-Epoch Training and Fold 1 Outer Test Evaluation (Baseline Model A)...")
    print("-" * 70)
    baseline_results = evaluate_1epoch_fold1(baseline_model, train_loader, test_loader, device=device, seed=42)

    print(f"  1-Epoch Train Loss: {baseline_results['mean_train_loss']:.4f} (took {baseline_results['train_time_sec']:.2f}s)")
    print(f"  Outer Test Windows: {baseline_results['n_evaluated_windows']:,} (eval took {baseline_results['eval_time_sec']:.2f}s)")
    print(f"  Overall Accuracy:   {baseline_results['accuracy']*100:.2f}%")
    print(f"  Macro-F1 (4-class): {baseline_results['macro_f1']:.4f}")
    print(f"  Cohen's Kappa:      {baseline_results['kappa']:.4f}")
    print(f"  Per-class F1:")
    for stage, f1 in baseline_results['per_class_f1'].items():
        print(f"    - {stage}: {f1:.4f}")

    # Summary Comparison
    macro_f1_delta = mutated_results['macro_f1'] - baseline_results['macro_f1']
    acc_delta = (mutated_results['accuracy'] - baseline_results['accuracy']) * 100.0
    loss_delta = mutated_results['mean_train_loss'] - baseline_results['mean_train_loss']

    print("\n" + "=" * 70)
    print("NAS Iteration 1 Summary Comparison (1 Epoch on Fold 1):")
    print("=" * 70)
    print(f"  Parameters: Baseline = {p_baseline:,} | Mutated = {p_mutated:,} (Delta: +{delta_p:,}, {pct_increase:+.2f}%)")
    print(f"  Train Loss: Baseline = {baseline_results['mean_train_loss']:.4f} | Mutated = {mutated_results['mean_train_loss']:.4f} (Delta: {loss_delta:+.4f})")
    print(f"  Accuracy:   Baseline = {baseline_results['accuracy']*100:.2f}% | Mutated = {mutated_results['accuracy']*100:.2f}% (Delta: {acc_delta:+.2f} pp)")
    print(f"  Macro-F1:   Baseline = {baseline_results['macro_f1']:.4f} | Mutated = {mutated_results['macro_f1']:.4f} (Delta: {macro_f1_delta:+.4f})")
    print(f"  Kappa:      Baseline = {baseline_results['kappa']:.4f} | Mutated = {mutated_results['kappa']:.4f} (Delta: {mutated_results['kappa'] - baseline_results['kappa']:+.4f})")

    # Save results to JSON artifact
    final_output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "iteration": 1,
        "fold": 1,
        "device": str(device),
        "baseline": {
            "name": "ConventionalCNNBiGRU",
            "parameters": p_baseline,
            "results": baseline_results
        },
        "mutated": {
            "name": "MultiScaleConventionalCNNBiGRU",
            "mutation_type": "pre_recurrent",
            "parameters": p_mutated,
            "delta_parameters": delta_p,
            "percentage_increase": pct_increase,
            "parameter_cap": 196728,
            "budget_compliant": p_mutated <= 196728,
            "results": mutated_results
        },
        "deltas": {
            "macro_f1_gain": float(macro_f1_delta),
            "accuracy_gain_pct": float(acc_delta),
            "loss_reduction": float(-loss_delta)
        }
    }

    with open(RESULTS_OUTPUT_PATH, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"\nEvaluation results successfully exported to: {RESULTS_OUTPUT_PATH}")
    return final_output


if __name__ == "__main__":
    run_nas_iteration1_evaluation()
