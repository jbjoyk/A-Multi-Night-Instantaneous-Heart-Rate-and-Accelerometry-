"""
EXP-011: Multi-Seed Long-Training Convergence & Reproducibility Confirmation.

Seeds Evaluated:
  - Seed 42:   Reference replication of EXP-010
  - Seed 123:  Independent convergence confirmation
  - Seed 2024: Independent convergence confirmation

Model Configuration (B0_GRL):
  - Architecture: SleepAdaptiveNet (162,325 parameters)
  - Adaptive Modality Fusion: ON
  - Transition Module: ON (lambda_trans = 0.10)
  - Boundary Loss: OFF (boundary_aware = False, lambda_boundary = 0.0)
  - Subject-Invariant GRL: ON (lambda_adv = 0.10, 31 train subjects)
  - Uncertainty Head: OFF
  - Loss Type: RCCLoss (CE + 0.01 * RW with asymmetric cost matrix)
  - Optimization: Adam (lr = 1e-4, weight_decay = 0.0)
  - Batch size: 4
  - Sequence length: T = 600 epochs (5 continuous hours, 0 padding)
  - Epochs: 500 per seed

Strict Protocol:
  - Train: 31 subjects (160 usable nights)
  - Val: 5 subjects (28 usable nights)
  - Test: 11 subjects (58 usable nights) -> STRICTLY UNTOUCHED & NOT LOADED
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


def train_seed_convergence(
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_train_subjects: int,
    epochs: int = 500,
    device: torch.device = torch.device("cuda:0"),
    base_ckpt_dir: str = "checkpoints_exp011"
) -> Dict[str, any]:
    print(f"\n" + "=" * 80)
    print(f"STARTING 500-EPOCH CONVERGENCE RUN | SEED {seed}")
    print(f"=" * 80)
    
    seed_ckpt_dir = os.path.join(base_ckpt_dir, f"seed_{seed}")
    os.makedirs(seed_ckpt_dir, exist_ok=True)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    model = build_ablation_model(
        "A4",
        n_train_subjects=n_train_subjects,
        loss_type="rccloss",
        boundary_aware=False,
        lambda_boundary=0.0
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    n_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
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
                raise ValueError(f"NaN/Inf loss encountered in seed {seed} at epoch {ep}, step {step}")
                
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        tr_loss = float(np.mean(batch_losses))
        
        # Validation Evaluation
        val_m = evaluate_validation_sweep(model, val_loader, device=device)
        
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
        
        # Best Metric Tracking & Checkpointing
        if val_m["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_m["macro_f1"]
            best_macro_f1_epoch = ep
            torch.save({
                "epoch": ep, "seed": seed, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "val_metrics": val_m, "train_loss": tr_loss
            }, os.path.join(seed_ckpt_dir, "best_macro_f1.pt"))
            
        if val_m["accuracy"] > best_accuracy:
            best_accuracy = val_m["accuracy"]
            best_accuracy_epoch = ep
            torch.save({
                "epoch": ep, "seed": seed, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "val_metrics": val_m, "train_loss": tr_loss
            }, os.path.join(seed_ckpt_dir, "best_accuracy.pt"))
            
        if val_m["kappa"] > best_kappa:
            best_kappa = val_m["kappa"]
            best_kappa_epoch = ep
            torch.save({
                "epoch": ep, "seed": seed, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "val_metrics": val_m, "train_loss": tr_loss
            }, os.path.join(seed_ckpt_dir, "best_kappa.pt"))
            
        if ep in periodic_ckpt_epochs:
            torch.save({
                "epoch": ep, "seed": seed, "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "val_metrics": val_m, "train_loss": tr_loss
            }, os.path.join(seed_ckpt_dir, f"checkpoint_epoch_{ep}.pt"))
            
        if ep % 20 == 0 or ep == 1 or ep == 10 or ep == epochs:
            elapsed = time.time() - t0
            print(f"Seed {seed:4d} | Ep {ep:03d}/500 | Train: {tr_loss:.4f} | ValLoss: {val_m['val_loss']:.4f} | "
                  f"Acc: {val_m['accuracy']*100:.2f}% | F1: {val_m['macro_f1']:.4f} (Best: {best_macro_f1:.4f} @ Ep {best_macro_f1_epoch:03d}) | "
                  f"κ: {val_m['kappa']:.4f} | REM F1: {val_m['rem_f1']:.3f} | Wake F1: {val_m['wake_f1']:.3f} | {elapsed:.1f}s")
                  
    total_time = time.time() - t0
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    
    torch.save({
        "epoch": epochs, "seed": seed, "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(), "val_metrics": val_m, "train_loss": tr_loss
    }, os.path.join(seed_ckpt_dir, "final_epoch_500.pt"))
    
    print(f"Seed {seed} complete in {total_time:.1f} s ({total_time/60:.2f} min). Best Macro-F1: {best_macro_f1:.4f} at Ep {best_macro_f1_epoch}.")
    
    return {
        "seed": seed,
        "epochs": epochs,
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
        "checkpoint_dir": seed_ckpt_dir,
        "history": history
    }


def run_exp011():
    print("=" * 80)
    print("EXP-011: MULTI-SEED CONVERGENCE & REPRODUCIBILITY CONFIRMATION")
    print("=" * 80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Hardware Device: {device} ({torch.cuda.get_device_name(device)})")
    
    cache_dir = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
    data_dir = "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    
    mat_files = sorted(glob.glob(os.path.join(data_dir, "Bidslab*", "*", "labels.mat")))
    all_nights = [(f.split(os.sep)[-3], f.split(os.sep)[-2]) for f in mat_files]
    all_subjects = sorted(list(set(s for s, n in all_nights)))
    
    split = make_reproducible_split(all_subjects, split_type="paper", seed=0)
    train_subs = split["train"] # 31
    val_subs = split["val"]     # 5
    test_subs = split["test"]   # 11
    
    assert len(train_subs) == 31
    assert len(val_subs) == 5
    assert len(test_subs) == 11
    assert len(set(train_subs).intersection(val_subs)) == 0
    assert len(set(train_subs).intersection(test_subs)) == 0
    assert len(set(val_subs).intersection(test_subs)) == 0
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
            
    assert len(train_records) == 160
    assert len(val_records) == 28
    print("160 usable train nights and 28 usable val nights loaded. Test set STRICTLY UNTOUCHED.")
    
    train_ds = ControlledSleepDataset(train_records, subject_to_idx=subject_to_idx, seq_len=600)
    val_ds = ControlledSleepDataset(val_records, subject_to_idx=subject_to_idx, seq_len=600)
    
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
    base_ckpt_dir = "checkpoints_exp011"
    os.makedirs(base_ckpt_dir, exist_ok=True)
    
    seeds = [42, 123, 2024]
    results_by_seed = {}
    
    t_global_start = time.time()
    for s in seeds:
        res = train_seed_convergence(
            seed=s,
            train_loader=train_loader,
            val_loader=val_loader,
            n_train_subjects=len(train_subs),
            epochs=500,
            device=device,
            base_ckpt_dir=base_ckpt_dir
        )
        results_by_seed[s] = res
        
    global_time = time.time() - t_global_start
    print("\n" + "=" * 80)
    print(f"ALL 3 SEEDS COMPLETED IN {global_time:.1f} s ({global_time/60:.2f} min).")
    print("=" * 80)
    
    # Save Full Multi-Seed History
    output_pkl = "exp011_multiseed_history.pkl"
    with open(output_pkl, "wb") as f:
        pickle.dump(results_by_seed, f)
    print(f"Saved complete multi-seed history to {output_pkl}")
    
    # Generate 6-Panel Multi-Seed Convergence Figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    colors = {42: "tab:blue", 123: "tab:green", 2024: "tab:orange"}
    
    # Panel 1: Validation Macro-F1 vs Epoch
    for s in seeds:
        h = results_by_seed[s]["history"]
        best_ep = results_by_seed[s]["best_macro_f1_epoch"]
        best_f1 = results_by_seed[s]["best_macro_f1"]
        axes[0, 0].plot(h["epoch"], h["macro_f1"], label=f"Seed {s} (Peak {best_f1:.4f} @ Ep {best_ep})", color=colors[s], alpha=0.85)
        axes[0, 0].scatter([best_ep], [best_f1], color=colors[s], s=50, zorder=5)
    axes[0, 0].set_title("Validation Macro-F1 vs Epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Macro-F1")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle="--", alpha=0.5)
    
    # Panel 2: Validation Accuracy vs Epoch
    for s in seeds:
        h = results_by_seed[s]["history"]
        best_ep = results_by_seed[s]["best_accuracy_epoch"]
        best_acc = results_by_seed[s]["best_accuracy"] * 100
        axes[0, 1].plot(h["epoch"], [a * 100 for a in h["accuracy"]], label=f"Seed {s} (Peak {best_acc:.2f}% @ Ep {best_ep})", color=colors[s], alpha=0.85)
        axes[0, 1].scatter([best_ep], [best_acc], color=colors[s], s=50, zorder=5)
    axes[0, 1].set_title("Validation Accuracy (%) vs Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle="--", alpha=0.5)
    
    # Panel 3: Validation Kappa vs Epoch
    for s in seeds:
        h = results_by_seed[s]["history"]
        best_ep = results_by_seed[s]["best_kappa_epoch"]
        best_kap = results_by_seed[s]["best_kappa"]
        axes[1, 0].plot(h["epoch"], h["kappa"], label=f"Seed {s} (Peak {best_kap:.4f} @ Ep {best_ep})", color=colors[s], alpha=0.85)
        axes[1, 0].scatter([best_ep], [best_kap], color=colors[s], s=50, zorder=5)
    axes[1, 0].set_title("Validation Cohen's Kappa (κ) vs Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Cohen's Kappa")
    axes[1, 0].legend()
    axes[1, 0].grid(True, linestyle="--", alpha=0.5)
    
    # Panel 4: Training Loss vs Validation Loss
    for s in seeds:
        h = results_by_seed[s]["history"]
        axes[1, 1].plot(h["epoch"], h["train_loss"], label=f"Train Loss (Seed {s})", color=colors[s], linestyle="--", alpha=0.6)
        axes[1, 1].plot(h["epoch"], h["val_loss"], label=f"Val Loss (Seed {s})", color=colors[s], alpha=0.85)
    axes[1, 1].set_title("Training Loss (Dashed) vs Validation Loss (Solid)")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].legend()
    axes[1, 1].grid(True, linestyle="--", alpha=0.5)
    
    # Panel 5: Expected Calibration Error (ECE) vs Epoch
    for s in seeds:
        h = results_by_seed[s]["history"]
        axes[2, 0].plot(h["epoch"], h["ece"], label=f"Seed {s}", color=colors[s], alpha=0.85)
    axes[2, 0].set_title("Expected Calibration Error (ECE) vs Epoch")
    axes[2, 0].set_xlabel("Epoch")
    axes[2, 0].set_ylabel("ECE")
    axes[2, 0].legend()
    axes[2, 0].grid(True, linestyle="--", alpha=0.5)
    
    # Panel 6: REM F1 vs Epoch
    for s in seeds:
        h = results_by_seed[s]["history"]
        axes[2, 1].plot(h["epoch"], h["rem_f1"], label=f"Seed {s}", color=colors[s], alpha=0.85)
    axes[2, 1].set_title("REM Stage F1 Score vs Epoch")
    axes[2, 1].set_xlabel("Epoch")
    axes[2, 1].set_ylabel("REM F1")
    axes[2, 1].legend()
    axes[2, 1].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plot_path = "exp011_convergence_curves.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Saved 6-panel convergence figure to {plot_path}")
    
    return results_by_seed


if __name__ == "__main__":
    run_exp011()
