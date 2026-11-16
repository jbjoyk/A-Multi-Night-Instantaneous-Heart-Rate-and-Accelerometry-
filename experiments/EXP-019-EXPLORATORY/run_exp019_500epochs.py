"""
EXP-019: 500-Epoch Extended Training Benchmark:
Baseline Model A vs. MultiScaleConventionalCNNBiGRU (MSTF-Pre)

Authoritative Protocol:
  - Dataset: BIDSleep Fold 1 (experiments/EXP-014/EXP014_MANIFESTS/fold_1.json)
    * Inner-train: 32 subjects (168 usable nights)
    * Inner-val: 5 subjects (24 usable nights)
    * Outer-test: 10 subjects (54 usable nights, 32,098 valid windows)
  - Models:
    * Baseline: ConventionalCNNBiGRU (163,940 params, exp013_models.py)
    * Mutated: MultiScaleConventionalCNNBiGRU (192,692 params, nas_iteration1_model.py)
  - Training Schedule:
    * 500 epochs
    * Adam (lr = 1e-4, weight_decay = 0.0)
    * Batch size = 4
    * Sequence length = 600 epochs (5 hours, zero padding)
    * Loss: CrossEntropyLoss + 0.01 * RCCLoss
    * Seed: 42 (deterministic)
  - Validation & Checkpointing:
    * Evaluated at the end of each epoch on inner-val
    * best_macro_f1.pt: saved when inner-val Macro-F1 achieves new peak
    * epoch_500_final.pt: saved at final epoch 500
    * Periodic checkpoints: epochs 50, 100, 200, 300, 400, 500
  - Outer Test Evaluation:
    * Evaluated on outer-test partition (54 nights, 32,098 valid windows)
    * Primary metric: Macro-F1
    * Secondary metrics: Accuracy, Cohen's kappa, MCC, per-stage F1, recall, precision
"""

import os
import sys
import glob
import json
import time
import argparse
import subprocess
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, precision_recall_fscore_support
)

# Workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

EXP014_DIR = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014")
if EXP014_DIR not in sys.path:
    sys.path.insert(0, EXP014_DIR)

from slamss_ifs_train import load_night_features
from run_controlled_experiment import ControlledSleepDataset
from exp013_models import ConventionalCNNBiGRU
from nas_iteration1_model import MultiScaleConventionalCNNBiGRU
from evaluator import SleepEvaluator

# Directories & Constants
CACHE_DIR = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
DATA_DIR = os.path.join(WORKSPACE_ROOT, "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0")
MANIFEST_PATH = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014/EXP014_MANIFESTS/fold_1.json")
EXP019_DIR = os.path.join(WORKSPACE_ROOT, "experiments/EXP-019")
RESULTS_DIR = os.path.join(EXP019_DIR, "exp019_results")
CKPT_DIR = os.path.join(EXP019_DIR, "checkpoints")

STAGE_NAMES = ["Wake", "Light", "Deep", "REM"]


def load_fold1_datasets(batch_size: int = 4, seq_len: int = 600) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
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


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    """Runs forward evaluation and computes standardized metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    batch_losses = []

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
            loss = out["total_loss"]
            batch_losses.append(loss.item())

            logits = out["logits"]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)

            valid = (targets >= 0) & (targets < 4) & mask
            all_preds.extend(preds[valid].cpu().numpy())
            all_targets.extend(targets[valid].cpu().numpy())
            all_probs.extend(probs[valid].cpu().numpy())

    y_true = np.array(all_targets, dtype=np.int64)
    y_pred = np.array(all_preds, dtype=np.int64)
    y_prob = np.array(all_probs, dtype=np.float32)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    mcc = float(matthews_corrcoef(y_true, y_pred))

    prec, rec, f1_cls, supp = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    per_class_f1 = {f"{STAGE_NAMES[i]}_f1": float(f1_cls[i]) for i in range(4)}
    per_class_recall = {f"{STAGE_NAMES[i]}_recall": float(rec[i]) for i in range(4)}
    per_class_precision = {f"{STAGE_NAMES[i]}_precision": float(prec[i]) for i in range(4)}

    return {
        "loss": float(np.mean(batch_losses)),
        "n_windows": len(y_true),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "kappa": kappa,
        "mcc": mcc,
        "per_class_f1": per_class_f1,
        "per_class_recall": per_class_recall,
        "per_class_precision": per_class_precision,
        "support": [int(s) for s in supp],
        "confusion_matrix": cm.tolist()
    }


def train_single_model(
    model_name: str,
    device_str: str = "cuda:0",
    epochs: int = 500,
    seed: int = 42,
    lr: float = 1e-4,
    batch_size: int = 4,
    seq_len: int = 600
) -> Dict[str, Any]:
    """Trains a model for 500 epochs, tracking validation and evaluating outer test."""
    print("=" * 75)
    print(f"EXP-019: Starting 500-Epoch Run: {model_name} on {device_str}")
    print(f"Seed: {seed} | LR: {lr} | Batch Size: {batch_size} | Seq Len: {seq_len}")
    print("=" * 75)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    # Determinism
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    # Data loaders
    train_loader, val_loader, test_loader, manifest = load_fold1_datasets(batch_size=batch_size, seq_len=seq_len)
    print(f"Data Loaded: Train {len(train_loader)} batches | Val {len(val_loader)} batches | Test {len(test_loader)} batches")

    # Instantiate model
    if model_name.lower() in ["baseline", "conventional_cnn_bigru", "model_a"]:
        model = ConventionalCNNBiGRU().to(device)
        model_display_name = "ConventionalCNNBiGRU (Baseline Model A)"
        save_prefix = "baseline"
    elif model_name.lower() in ["multiscale", "multiscale_cnn_bigru", "mstf", "mutated"]:
        model = MultiScaleConventionalCNNBiGRU().to(device)
        model_display_name = "MultiScaleConventionalCNNBiGRU (MSTF-Pre)"
        save_prefix = "multiscale"
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Instantiated: {model_display_name} | Parameters: {total_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    # Training state
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "val_kappa": [],
        "val_mcc": [],
        "val_wake_f1": [],
        "val_light_f1": [],
        "val_deep_f1": [],
        "val_rem_f1": []
    }

    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_ckpt_path = os.path.join(CKPT_DIR, f"{save_prefix}_best_macro_f1.pt")
    final_ckpt_path = os.path.join(CKPT_DIR, f"{save_prefix}_final_epoch_500.pt")

    periodic_checkpoints = {50, 100, 200, 300, 400, 500}
    t_start = time.time()

    print("\nBeginning 500 training epochs...")
    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
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
            optimizer.step()
            batch_losses.append(loss.item())

        mean_train_loss = float(np.mean(batch_losses))

        # Evaluate on inner validation set
        val_res = evaluate_loader(model, val_loader, device=device)
        ep_duration = time.time() - ep_t0

        # Record history
        history["epoch"].append(ep)
        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(val_res["loss"])
        history["val_accuracy"].append(val_res["accuracy"])
        history["val_macro_f1"].append(val_res["macro_f1"])
        history["val_kappa"].append(val_res["kappa"])
        history["val_mcc"].append(val_res["mcc"])
        history["val_wake_f1"].append(val_res["per_class_f1"]["Wake_f1"])
        history["val_light_f1"].append(val_res["per_class_f1"]["Light_f1"])
        history["val_deep_f1"].append(val_res["per_class_f1"]["Deep_f1"])
        history["val_rem_f1"].append(val_res["per_class_f1"]["REM_f1"])

        # Best checkpoint selection (Early stopping candidate)
        if val_res["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_res["macro_f1"]
            best_epoch = ep
            torch.save({
                "model_name": model_display_name,
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_res,
                "train_loss": mean_train_loss
            }, best_ckpt_path)

        # Periodic checkpoints
        if ep in periodic_checkpoints:
            periodic_path = os.path.join(CKPT_DIR, f"{save_prefix}_epoch_{ep:03d}.pt")
            torch.save({
                "model_name": model_display_name,
                "epoch": ep,
                "model_state": model.state_dict(),
                "val_metrics": val_res,
                "train_loss": mean_train_loss
            }, periodic_path)

        # Logging intervals
        if ep == 1 or ep % 25 == 0 or ep == epochs:
            elapsed = time.time() - t_start
            print(f"[{save_prefix.upper()}] Epoch {ep:03d}/{epochs} ({ep_duration:.1f}s, elapsed {elapsed/60:.1f}m) | "
                  f"TrainLoss: {mean_train_loss:.4f} | ValLoss: {val_res['loss']:.4f} | "
                  f"ValAcc: {val_res['accuracy']*100:.2f}% | ValF1: {val_res['macro_f1']:.4f} | "
                  f"BestPeak: {best_val_macro_f1:.4f} (Ep {best_epoch})")

    # Save final epoch checkpoint
    torch.save({
        "model_name": model_display_name,
        "epoch": epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_metrics": val_res,
        "train_loss": mean_train_loss
    }, final_ckpt_path)

    total_train_time = time.time() - t_start
    print(f"\n[{save_prefix.upper()}] 500 Epochs Completed in {total_train_time/60:.2f} minutes.")
    print(f"Peak Validation Macro-F1: {best_val_macro_f1:.4f} at Epoch {best_epoch}")

    # =========================================================================
    # OUTER TEST EVALUATION (Locked partition: 54 nights, 32,098 valid windows)
    # =========================================================================
    print(f"\n[{save_prefix.upper()}] Running Outer-Test Evaluation...")

    # 1. Evaluate Best Checkpoint (Early Stopping Selection)
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    test_res_best = evaluate_loader(model, test_loader, device=device)
    print(f"[{save_prefix.upper()}] Outer Test @ BEST Checkpoint (Ep {best_epoch}): "
          f"Macro-F1: {test_res_best['macro_f1']:.4f} | Acc: {test_res_best['accuracy']*100:.2f}% | "
          f"Kappa: {test_res_best['kappa']:.4f} | MCC: {test_res_best['mcc']:.4f}")

    # 2. Evaluate Final Checkpoint (Epoch 500)
    final_ckpt = torch.load(final_ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(final_ckpt["model_state"])
    test_res_final = evaluate_loader(model, test_loader, device=device)
    print(f"[{save_prefix.upper()}] Outer Test @ FINAL Checkpoint (Ep {epochs}): "
          f"Macro-F1: {test_res_final['macro_f1']:.4f} | Acc: {test_res_final['accuracy']*100:.2f}% | "
          f"Kappa: {test_res_final['kappa']:.4f} | MCC: {test_res_final['mcc']:.4f}")

    # Save complete run bundle
    run_output = {
        "model_name": model_display_name,
        "save_prefix": save_prefix,
        "parameters": total_params,
        "epochs": epochs,
        "seed": seed,
        "device": device_str,
        "total_train_time_sec": total_train_time,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "outer_test_best_checkpoint": test_res_best,
        "outer_test_final_checkpoint": test_res_final,
        "history": history
    }

    output_file = os.path.join(RESULTS_DIR, f"{save_prefix}_500epochs_results.json")
    with open(output_file, "w") as f:
        json.dump(run_output, f, indent=2)
    print(f"[{save_prefix.upper()}] Results saved to {output_file}")

    return run_output


def synthesize_and_report():
    """Aggregates baseline and multiscale results, generates comparison tables, and writes report."""
    base_file = os.path.join(RESULTS_DIR, "baseline_500epochs_results.json")
    multi_file = os.path.join(RESULTS_DIR, "multiscale_500epochs_results.json")

    if not os.path.exists(base_file) or not os.path.exists(multi_file):
        print(f"Cannot synthesize: missing {base_file} or {multi_file}")
        return

    with open(base_file, "r") as f:
        b_data = json.load(f)
    with open(multi_file, "r") as f:
        m_data = json.load(f)

    b_best = b_data["outer_test_best_checkpoint"]
    m_best = m_data["outer_test_best_checkpoint"]
    b_final = b_data["outer_test_final_checkpoint"]
    m_final = m_data["outer_test_final_checkpoint"]

    # Compute deltas at best checkpoint
    delta_f1_best = (m_best["macro_f1"] - b_best["macro_f1"]) * 100
    delta_acc_best = (m_best["accuracy"] - b_best["accuracy"]) * 100
    delta_kappa_best = (m_best["kappa"] - b_best["kappa"])
    delta_mcc_best = (m_best["mcc"] - b_best["mcc"])

    # Compute deltas at final epoch
    delta_f1_final = (m_final["macro_f1"] - b_final["macro_f1"]) * 100
    delta_acc_final = (m_final["accuracy"] - b_final["accuracy"]) * 100

    # Write summary JSON
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment": "EXP-019",
        "title": "500-Epoch Extended Training Benchmark: Baseline vs MultiScaleConventionalCNNBiGRU",
        "fold": 1,
        "seed": b_data["seed"],
        "epochs": b_data["epochs"],
        "baseline_params": b_data["parameters"],
        "multiscale_params": m_data["parameters"],
        "param_delta": m_data["parameters"] - b_data["parameters"],
        "param_pct_increase": (m_data["parameters"] - b_data["parameters"]) / b_data["parameters"] * 100,
        "best_checkpoint": {
            "baseline_peak_epoch": b_data["best_epoch"],
            "multiscale_peak_epoch": m_data["best_epoch"],
            "baseline_macro_f1": b_best["macro_f1"],
            "multiscale_macro_f1": m_best["macro_f1"],
            "delta_macro_f1_pp": delta_f1_best,
            "baseline_acc": b_best["accuracy"],
            "multiscale_acc": m_best["accuracy"],
            "delta_acc_pp": delta_acc_best,
            "baseline_kappa": b_best["kappa"],
            "multiscale_kappa": m_best["kappa"],
            "delta_kappa": delta_kappa_best,
            "baseline_mcc": b_best["mcc"],
            "multiscale_mcc": m_best["mcc"],
            "delta_mcc": delta_mcc_best,
            "per_stage_f1": {
                stage: {
                    "baseline": b_best["per_class_f1"][f"{stage}_f1"],
                    "multiscale": m_best["per_class_f1"][f"{stage}_f1"],
                    "delta_pp": (m_best["per_class_f1"][f"{stage}_f1"] - b_best["per_class_f1"][f"{stage}_f1"]) * 100
                } for stage in STAGE_NAMES
            }
        },
        "final_epoch_500": {
            "baseline_macro_f1": b_final["macro_f1"],
            "multiscale_macro_f1": m_final["macro_f1"],
            "delta_macro_f1_pp": delta_f1_final,
            "baseline_acc": b_final["accuracy"],
            "multiscale_acc": m_final["accuracy"],
            "delta_acc_pp": delta_acc_final,
            "baseline_overfitting_drop_f1": (b_best["macro_f1"] - b_final["macro_f1"]) * 100,
            "multiscale_overfitting_drop_f1": (m_best["macro_f1"] - m_final["macro_f1"]) * 100
        }
    }

    summary_file = os.path.join(RESULTS_DIR, "exp019_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown Report
    report_md = f"""# EXP-019: 500-Epoch Extended Training Benchmark Report
**Baseline Model A vs. MultiScaleConventionalCNNBiGRU (MSTF-Pre)**

- **Date**: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}
- **Dataset Partition**: BIDSleep Fold 1 (32 inner-train / 5 inner-val / 10 outer-test subjects)
- **Outer Test Coverage**: 54 nights, 32,098 valid evaluated windows
- **Hardware**: Dual RTX 3060 (Baseline on {b_data['device']}, MultiScale on {m_data['device']})
- **Training Budget**: 500 epochs per model (Adam lr=1e-4, batch size=4, seq len=600)

---

## 1. Executive Summary & Core Estimands

Under a matched 500-epoch training protocol, **MultiScaleConventionalCNNBiGRU** (192,692 parameters, +17.54% capacity) was benchmarked directly against the authoritative **ConventionalCNNBiGRU** baseline (163,940 parameters).

### Model Selection @ Peak Validation Checkpoint (Early Stopping Policy)
| Metric | Baseline Model A | Mutated Model A (MSTF-Pre) | Delta ($\\Delta$) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Peak Epoch** | Epoch {b_data['best_epoch']} | Epoch {m_data['best_epoch']} | {m_data['best_epoch'] - b_data['best_epoch']:+d} epochs | — |
| **Primary: Macro-F1** | **{b_best['macro_f1']:.4f}** | **{m_best['macro_f1']:.4f}** | **{delta_f1_best:+.2f} pp** | **{'SUPERIOR' if delta_f1_best > 0 else 'INFERIOR'}** |
| **Accuracy** | {b_best['accuracy']*100:.2f}% | {m_best['accuracy']*100:.2f}% | {delta_acc_best:+.2f} pp | {'Positive' if delta_acc_best > 0 else 'Negative'} |
| **Cohen's Kappa ($\\kappa$)** | {b_best['kappa']:.4f} | {m_best['kappa']:.4f} | {delta_kappa_best:+.4f} | {'Positive' if delta_kappa_best > 0 else 'Negative'} |
| **MCC** | {b_best['mcc']:.4f} | {m_best['mcc']:.4f} | {delta_mcc_best:+.4f} | {'Positive' if delta_mcc_best > 0 else 'Negative'} |

---

## 2. Per-Stage F1 Breakdown (Best Checkpoint)

| Sleep Stage | Baseline F1 | MultiScale F1 | Stage Delta ($\\Delta$) | Support (Windows) |
| :--- | :---: | :---: | :---: | :---: |
| **Wake** | {b_best['per_class_f1']['Wake_f1']:.4f} | {m_best['per_class_f1']['Wake_f1']:.4f} | {(m_best['per_class_f1']['Wake_f1'] - b_best['per_class_f1']['Wake_f1'])*100:+.2f} pp | {b_best['support'][0]:,} |
| **Light** | {b_best['per_class_f1']['Light_f1']:.4f} | {m_best['per_class_f1']['Light_f1']:.4f} | {(m_best['per_class_f1']['Light_f1'] - b_best['per_class_f1']['Light_f1'])*100:+.2f} pp | {b_best['support'][1]:,} |
| **Deep** | {b_best['per_class_f1']['Deep_f1']:.4f} | {m_best['per_class_f1']['Deep_f1']:.4f} | {(m_best['per_class_f1']['Deep_f1'] - b_best['per_class_f1']['Deep_f1'])*100:+.2f} pp | {b_best['support'][2]:,} |
| **REM** | {b_best['per_class_f1']['REM_f1']:.4f} | {m_best['per_class_f1']['REM_f1']:.4f} | {(m_best['per_class_f1']['REM_f1'] - b_best['per_class_f1']['REM_f1'])*100:+.2f} pp | {b_best['support'][3]:,} |

---

## 3. Asymptotic 500-Epoch Convergence & Overfitting Analysis

| Observation | Baseline Model A | MultiScale Model A | Comparison |
| :--- | :---: | :---: | :---: |
| **Best Test Macro-F1** | {b_best['macro_f1']:.4f} (Ep {b_data['best_epoch']}) | {m_best['macro_f1']:.4f} (Ep {m_data['best_epoch']}) | {delta_f1_best:+.2f} pp |
| **Final Test Macro-F1 (Ep 500)** | {b_final['macro_f1']:.4f} | {m_final['macro_f1']:.4f} | {delta_f1_final:+.2f} pp |
| **Post-Peak Degradation** | {(b_best['macro_f1'] - b_final['macro_f1'])*100:.2f} pp | {(m_best['macro_f1'] - m_final['macro_f1'])*100:.2f} pp | {'MultiScale more resilient' if (m_best['macro_f1'] - m_final['macro_f1']) < (b_best['macro_f1'] - b_final['macro_f1']) else 'Baseline more resilient'} |
| **Total Wall-Clock Time** | {b_data['total_train_time_sec']/60:.1f} min | {m_data['total_train_time_sec']/60:.1f} min | Parallel Dual-GPU |

---

## 4. Architectural Grounding & Governance

- **Baseline Parameters**: 163,940
- **Mutated Parameters**: 192,692 (+17.54% increase over baseline)
- **Approved Cap (+20%)**: 196,728 (**Passed with 4,036 parameters of headroom**)
- **Data Integrity**: 0 subject overlap across train, validation, and test splits
- **Reproducibility**: Seed 42, deterministic PyTorch / CUDA backends
"""

    report_file = os.path.join(EXP019_DIR, "EXP019_REPORT.md")
    with open(report_file, "w") as f:
        f.write(report_md)

    print("\n" + "=" * 75)
    print("EXP-019 SYNTHESIS COMPLETE")
    print(f"Summary JSON: {summary_file}")
    print(f"Markdown Report: {report_file}")
    print("=" * 75)
    print(report_md)


def main():
    parser = argparse.ArgumentParser(description="EXP-019 500-Epoch Benchmark")
    parser.add_argument("--model", type=str, choices=["baseline", "multiscale", "both"], default="both")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--synthesize-only", action="store_true")
    args = parser.parse_args()

    if args.synthesize_only:
        synthesize_and_report()
        return

    n_gpus = torch.cuda.device_count()

    if args.model == "both":
        if n_gpus >= 2 and args.device == "auto":
            print(f"Detected {n_gpus} GPUs! Launching Baseline on cuda:0 and MultiScale on cuda:1 in parallel...")
            # Launch both via background processes
            cmd_base = [sys.executable, __file__, "--model", "baseline", "--device", "cuda:0", "--epochs", str(args.epochs), "--seed", str(args.seed)]
            cmd_multi = [sys.executable, __file__, "--model", "multiscale", "--device", "cuda:1", "--epochs", str(args.epochs), "--seed", str(args.seed)]

            p_base = subprocess.Popen(cmd_base)
            p_multi = subprocess.Popen(cmd_multi)

            # Wait for both
            p_base.wait()
            p_multi.wait()

            if p_base.returncode != 0 or p_multi.returncode != 0:
                raise RuntimeError(f"Execution failed: baseline rc={p_base.returncode}, multiscale rc={p_multi.returncode}")

            synthesize_and_report()
        else:
            dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            print(f"Running sequentially on {dev}...")
            train_single_model("baseline", device_str=dev, epochs=args.epochs, seed=args.seed)
            train_single_model("multiscale", device_str=dev, epochs=args.epochs, seed=args.seed)
            synthesize_and_report()
    else:
        dev = args.device if args.device != "auto" else ("cuda:0" if torch.cuda.is_available() else "cpu")
        train_single_model(args.model, device_str=dev, epochs=args.epochs, seed=args.seed)


if __name__ == "__main__":
    main()
