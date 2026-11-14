"""
EXP-010: Long-Training Convergence Evaluation for SleepAdaptiveNet B0_GRL.

Model Configuration (B0_GRL):
  - Base: A4
  - Adaptive Fusion: ON
  - Transition Module: ON (lambda_trans = 0.1)
  - Boundary Loss: OFF (boundary_aware = False, lambda_boundary = 0.0)
  - Subject-Invariant GRL: ON (lambda_adv = 0.1, 31 train subjects)
  - Uncertainty Head: OFF
  - Loss Type: RCCLoss (composite CE + 0.01 * RW with asymmetric cost matrix)
  - Epochs: 500
  - Seed: 42
  - Batch size: 4
  - Optimizer: Adam (lr = 1e-4, weight_decay = 0.0)
  - Sequence length: T = 600 epochs (5 continuous hours, 0 padding)

Strict Scientific Controls:
  - Train: 31 subjects (160 usable nights)
  - Val: 5 subjects (28 usable nights)
  - Test: 11 subjects (58 usable nights) -> STRICTLY UNTOUCHED / NOT LOADED
  - Preprocessing: prep_57738d534b
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
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sleep_adaptive_net import build_ablation_model
from slamss_ifs_baseline import make_reproducible_split
from slamss_ifs_train import load_night_features
from run_controlled_experiment import ControlledSleepDataset
from run_exp008_boundary_sweep import evaluate_validation_sweep


def run_exp010():
    print("=" * 80)
    print("EXP-010: LONG-TRAINING CONVERGENCE EVALUATION (B0_GRL, 500 EPOCHS, SEED 42)")
    print("=" * 80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(device)})")
    
    # Preflight Protocol Audit
    cache_dir = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
    data_dir = "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    
    mat_files = sorted(glob.glob(os.path.join(data_dir, "Bidslab*", "*", "labels.mat")))
    all_nights = [(f.split(os.sep)[-3], f.split(os.sep)[-2]) for f in mat_files]
    all_subjects = sorted(list(set(s for s, n in all_nights)))
    
    split = make_reproducible_split(all_subjects, split_type="paper", seed=0)
    train_subs = split["train"] # 31
    val_subs = split["val"]     # 5
    test_subs = split["test"]   # 11
    
    assert len(train_subs) == 31, f"Expected 31 train subjects, got {len(train_subs)}"
    assert len(val_subs) == 5, f"Expected 5 val subjects, got {len(val_subs)}"
    assert len(test_subs) == 11, f"Expected 11 test subjects, got {len(test_subs)}"
    assert len(set(train_subs).intersection(val_subs)) == 0, "Subject leakage between train and val!"
    assert len(set(train_subs).intersection(test_subs)) == 0, "Subject leakage between train and test!"
    assert len(set(val_subs).intersection(test_subs)) == 0, "Subject leakage between val and test!"
    print("Zero subject overlap: VERIFIED (31 train / 5 val / 11 test).")
    
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
        elif s in test_subs:
            # STRICT REQUIREMENT: HELD-OUT TEST DATA IS NEVER LOADED
            pass
            
    assert len(train_records) == 160, f"Expected 160 usable train nights, got {len(train_records)}"
    assert len(val_records) == 28, f"Expected 28 usable val nights, got {len(val_records)}"
    print(f"Data loading verified: 160 usable train nights, 28 usable val nights.")
    print("STRICT PROTOCOL: Held-out test set (11 subjects, 58 nights) is UNTOUCHED.")
    
    train_ds = ControlledSleepDataset(train_records, subject_to_idx=subject_to_idx, seq_len=600)
    val_ds = ControlledSleepDataset(val_records, subject_to_idx=subject_to_idx, seq_len=600)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    # Checkpoint Directory
    ckpt_dir = "checkpoints_exp010"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Deterministic Initialization
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Build B0_GRL Model
    model = build_ablation_model(
        "A4",
        n_train_subjects=len(train_subs),
        loss_type="rccloss",
        boundary_aware=False,
        lambda_boundary=0.0
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: B0_GRL (SleepAdaptiveNet) | Total params: {n_params:,} | Trainable: {trainable_params:,}")
    
    # Training Loop: 500 Epochs
    epochs = 500
    periodic_ckpt_epochs = {50, 100, 200, 300, 400, 500}
    
    best_macro_f1 = -1.0
    best_macro_f1_epoch = -1
    best_accuracy = -1.0
    best_accuracy_epoch = -1
    best_kappa = -1.0
    best_kappa_epoch = -1
    
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "loss_stage": [],
        "loss_trans": [],
        "loss_adv": [],
        "accuracy": [],
        "macro_f1": [],
        "kappa": [],
        "wake_f1": [],
        "light_f1": [],
        "deep_f1": [],
        "rem_f1": [],
        "wake_recall": [],
        "light_recall": [],
        "deep_recall": [],
        "rem_recall": [],
        "wake_precision": [],
        "light_precision": [],
        "deep_precision": [],
        "rem_precision": [],
        "boundary_error_rate": [],
        "non_boundary_error_rate": [],
        "ece": [],
        "alpha_mean": [],
        "alpha_std": [],
        "alpha_wake": [],
        "alpha_light": [],
        "alpha_deep": [],
        "alpha_rem": [],
        "alpha_correct": [],
        "alpha_incorrect": []
    }
    
    torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    
    print("\nStarting 500-epoch training run...")
    for ep in range(1, epochs + 1):
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
                raise ValueError(f"NaN/Inf loss encountered at epoch {ep}, step {step}")
                
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        tr_loss = float(np.mean(batch_losses))
        
        # Validation Evaluation
        val_m = evaluate_validation_sweep(model, val_loader, device=device)
        
        # Record metrics
        history["epoch"].append(ep)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_m["val_loss"])
        history["loss_stage"].append(val_m["loss_stage"])
        history["loss_trans"].append(val_m["loss_trans"])
        history["loss_adv"].append(val_m["loss_adv"])
        history["accuracy"].append(val_m["accuracy"])
        history["macro_f1"].append(val_m["macro_f1"])
        history["kappa"].append(val_m["kappa"])
        history["wake_f1"].append(val_m["wake_f1"])
        history["light_f1"].append(val_m["light_f1"])
        history["deep_f1"].append(val_m["deep_f1"])
        history["rem_f1"].append(val_m["rem_f1"])
        history["wake_recall"].append(val_m["wake_recall"])
        history["light_recall"].append(val_m["light_recall"])
        history["deep_recall"].append(val_m["deep_recall"])
        history["rem_recall"].append(val_m["rem_recall"])
        history["wake_precision"].append(val_m["precision"][0])
        history["light_precision"].append(val_m["precision"][1])
        history["deep_precision"].append(val_m["precision"][2])
        history["rem_precision"].append(val_m["precision"][3])
        history["boundary_error_rate"].append(val_m["boundary_error_rate"])
        history["non_boundary_error_rate"].append(val_m["non_boundary_error_rate"])
        history["ece"].append(val_m["ece"])
        history["alpha_mean"].append(val_m["alpha_stats"]["mean"])
        history["alpha_std"].append(val_m["alpha_stats"]["std"])
        history["alpha_wake"].append(val_m["alpha_by_stage"]["Wake"])
        history["alpha_light"].append(val_m["alpha_by_stage"]["Light"])
        history["alpha_deep"].append(val_m["alpha_by_stage"]["Deep"])
        history["alpha_rem"].append(val_m["alpha_by_stage"]["REM"])
        history["alpha_correct"].append(val_m["alpha_stats"]["correct_mean"])
        history["alpha_incorrect"].append(val_m["alpha_stats"]["incorrect_mean"])
        
        # Checkpointing
        # 1. Best Macro-F1
        if val_m["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_m["macro_f1"]
            best_macro_f1_epoch = ep
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_m,
                "train_loss": tr_loss
            }, os.path.join(ckpt_dir, "best_macro_f1.pt"))
            
        # 2. Best Accuracy
        if val_m["accuracy"] > best_accuracy:
            best_accuracy = val_m["accuracy"]
            best_accuracy_epoch = ep
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_m,
                "train_loss": tr_loss
            }, os.path.join(ckpt_dir, "best_accuracy.pt"))
            
        # 3. Best Kappa
        if val_m["kappa"] > best_kappa:
            best_kappa = val_m["kappa"]
            best_kappa_epoch = ep
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_m,
                "train_loss": tr_loss
            }, os.path.join(ckpt_dir, "best_kappa.pt"))
            
        # Periodic Checkpoints
        if ep in periodic_ckpt_epochs:
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_metrics": val_m,
                "train_loss": tr_loss
            }, os.path.join(ckpt_dir, f"checkpoint_epoch_{ep}.pt"))
            
        # Periodic Logging
        if ep % 10 == 0 or ep == 1 or ep == epochs:
            elapsed = time.time() - t0
            print(f"Epoch {ep:03d}/{epochs:03d} | Train: {tr_loss:.4f} | Val Loss: {val_m['val_loss']:.4f} | "
                  f"Acc: {val_m['accuracy']*100:.2f}% | Macro-F1: {val_m['macro_f1']:.4f} (Best: {best_macro_f1:.4f} @ Ep {best_macro_f1_epoch:03d}) | "
                  f"κ: {val_m['kappa']:.4f} | REM F1: {val_m['rem_f1']:.3f} | Wake F1: {val_m['wake_f1']:.3f} | "
                  f"Elapsed: {elapsed:.1f}s")
                  
    total_time = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print("\n" + "=" * 80)
    print(f"500-EPOCH TRAINING COMPLETE in {total_time:.1f} s ({total_time/60:.2f} min). Peak VRAM: {peak_vram_mb:.1f} MB.")
    print(f"Peak Performance Summary:")
    print(f"  - Best Macro-F1: {best_macro_f1:.4f} at Epoch {best_macro_f1_epoch}")
    print(f"  - Best Accuracy: {best_accuracy*100:.2f}% at Epoch {best_accuracy_epoch}")
    print(f"  - Best Kappa:    {best_kappa:.4f} at Epoch {best_kappa_epoch}")
    print("=" * 80)
    
    # Save Final Model Checkpoint
    torch.save({
        "epoch": epochs,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_metrics": val_m,
        "train_loss": tr_loss
    }, os.path.join(ckpt_dir, "final_epoch_500.pt"))
    
    # Packaging results
    summary_data = {
        "model": "B0_GRL",
        "epochs": epochs,
        "seed": seed,
        "total_time_s": total_time,
        "peak_vram_mb": peak_vram_mb,
        "n_params": n_params,
        "trainable_params": trainable_params,
        "best_macro_f1": best_macro_f1,
        "best_macro_f1_epoch": best_macro_f1_epoch,
        "best_accuracy": best_accuracy,
        "best_accuracy_epoch": best_accuracy_epoch,
        "best_kappa": best_kappa,
        "best_kappa_epoch": best_kappa_epoch,
        "final_epoch_metrics": val_m,
        "checkpoint_paths": {
            "best_macro_f1": os.path.join(ckpt_dir, "best_macro_f1.pt"),
            "best_accuracy": os.path.join(ckpt_dir, "best_accuracy.pt"),
            "best_kappa": os.path.join(ckpt_dir, "best_kappa.pt"),
            "final_epoch_500": os.path.join(ckpt_dir, "final_epoch_500.pt"),
        },
        "history": history
    }
    
    history_pkl_path = "exp010_training_history.pkl"
    with open(history_pkl_path, "wb") as f:
        pickle.dump(summary_data, f)
    print(f"Saved machine-readable history to {history_pkl_path}")
    
    # Generate Convergence Plots
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    eps = history["epoch"]
    
    # 1. Loss Curves
    axes[0, 0].plot(eps, history["train_loss"], label="Train Loss", color="blue", alpha=0.8)
    axes[0, 0].plot(eps, history["val_loss"], label="Val Loss", color="red", alpha=0.8)
    axes[0, 0].set_title("Training vs Validation Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    
    # 2. Macro-F1 & Accuracy
    axes[0, 1].plot(eps, [a * 100 for a in history["accuracy"]], label="Val Accuracy (%)", color="green")
    axes[0, 1].plot(eps, [f * 100 for f in history["macro_f1"]], label="Val Macro-F1 (x100)", color="darkorange")
    axes[0, 1].axvline(best_macro_f1_epoch, color="darkorange", linestyle=":", label=f"Peak F1 (Ep {best_macro_f1_epoch})")
    axes[0, 1].axvline(best_accuracy_epoch, color="green", linestyle="--", label=f"Peak Acc (Ep {best_accuracy_epoch})")
    axes[0, 1].set_title("Validation Accuracy & Macro-F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Score (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    
    # 3. Per-Stage F1
    axes[1, 0].plot(eps, history["wake_f1"], label="Wake F1", color="purple")
    axes[1, 0].plot(eps, history["light_f1"], label="Light F1", color="gold")
    axes[1, 0].plot(eps, history["deep_f1"], label="Deep F1", color="blue")
    axes[1, 0].plot(eps, history["rem_f1"], label="REM F1", color="crimson")
    axes[1, 0].set_title("Per-Class F1 Scores")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    
    # 4. Wake vs REM Recall
    axes[1, 1].plot(eps, [r * 100 for r in history["wake_recall"]], label="Wake Recall (%)", color="purple")
    axes[1, 1].plot(eps, [r * 100 for r in history["rem_recall"]], label="REM Recall (%)", color="crimson")
    axes[1, 1].set_title("Wake vs REM Recall Trajectory")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Recall (%)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    
    # 5. Boundary vs Non-Boundary Error
    axes[2, 0].plot(eps, [e * 100 for e in history["boundary_error_rate"]], label="Boundary Error (%)", color="darkred")
    axes[2, 0].plot(eps, [e * 100 for e in history["non_boundary_error_rate"]], label="Non-Boundary Error (%)", color="darkblue")
    axes[2, 0].set_title("Boundary vs Non-Boundary Error Rate")
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("Error Rate (%)")
    axes[2, 0].legend()
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)
    
    # 6. Adaptive Fusion Gate Alpha Trajectory
    axes[2, 1].plot(eps, history["alpha_wake"], label="Alpha Wake", color="purple", linestyle="--")
    axes[2, 1].plot(eps, history["alpha_deep"], label="Alpha Deep", color="blue")
    axes[2, 1].plot(eps, history["alpha_light"], label="Alpha Light", color="gold", linestyle=":")
    axes[2, 1].plot(eps, history["alpha_rem"], label="Alpha REM", color="crimson", linestyle="-.")
    axes[2, 1].set_title("Adaptive Modality Gate Alpha by Stage")
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("Alpha (Heart Rate Weight)")
    axes[2, 1].legend()
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_path = "exp010_convergence_curves.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved convergence curves to {plot_path}")
    
    return summary_data


if __name__ == "__main__":
    run_exp010()
