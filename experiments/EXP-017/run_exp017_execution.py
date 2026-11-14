"""
run_exp017_execution.py: Governed Execution Driver for EXP-017.

Controlled Evaluation of Sequence Truncation in Wearable Sleep Staging.
Approved by researcher on 2026-11-14.

Primary Comparison:
  - Model A (ConventionalCNNBiGRU, 163,940 parameters)
  - Condition 1: T = 600 (first 600 chronological epoch positions)
  - Condition 2: T <= 1200 (10-hour-capped extended sequence condition with pack_padded_sequence)
  - Primary Evaluation: Both models evaluated on the identical first 600 chronological epoch positions, with invalid positions masked.
  - Secondary Evaluation: Non-causal descriptive reporting of T <= 1200 model across available coverage.

Controls:
  - Seeds: 42, 123, 2024 (N=3)
  - Epochs: 80 per run
  - Batch size: 4 nights
  - Optimizer: Adam (lr = 1e-4, weight_decay = 0.0)
  - Split: 31 train (160 nights) / 5 validation (28 nights) / 11 test (58 nights)
  - Test cohort: STRICTLY LOCKED (NEVER LOADED, NEVER EVALUATED)
  - GPU: Dedicated lease on cuda:0 via GPUResourceManager
"""

import os
import sys
import glob
import time
import json
import copy
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

from exp013_models import ConventionalCNNBiGRU
from slamss_ifs_baseline import make_reproducible_split
from slamss_ifs_train import load_night_features
from run_controlled_experiment import ControlledSleepDataset, compute_ece
from dataset_gate import DatasetGate
from research_engine.resource_manager import GPUResourceManager
from research_engine.memory_validator import ResearchMemoryValidator


def evaluate_model_on_epochs(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_eval_epochs: int = 600
) -> Dict[str, Any]:
    """
    Evaluates model on validation data up to max_eval_epochs positions.
    When max_eval_epochs=600: evaluates strictly on chronological positions t in [0, 600),
    with invalid positions masked.
    When max_eval_epochs=1200: evaluates on all available positions t in [0, 1200).
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_boundaries = []
    all_probs = []
    all_subjects = []
    
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
            seq_lens = batch.get("seq_len", None)
            
            # Forward through model with recurrent padding protection
            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subj_idx, seq_lens=seq_lens)
            
            val_losses.append(out["total_loss"].item())
            loss_components["stage"].append(out["loss_stage"].item())
            
            logits = out["logits"]
            probs = out["probabilities"]
            preds = torch.argmax(logits, dim=-1)
            
            # Restrict evaluation window to max_eval_epochs
            B, T_batch, _ = logits.shape
            eval_T = min(T_batch, max_eval_epochs)
            
            preds_sub = preds[:, :eval_T]
            targets_sub = targets[:, :eval_T]
            mask_sub = mask[:, :eval_T]
            probs_sub = probs[:, :eval_T]
            
            # Boundary mask (y_t != y_{t-1})
            y_prev = targets_sub[:, :-1]
            y_curr = targets_sub[:, 1:]
            is_bound = torch.zeros_like(targets_sub, dtype=torch.bool)
            is_bound[:, 1:] = (y_prev != y_curr) & (y_prev >= 0) & (y_curr >= 0)
            
            valid = (targets_sub >= 0) & (targets_sub < 4) & mask_sub
            
            for b in range(B):
                v_b = valid[b]
                subj_name = batch["subject"][b]
                if v_b.any():
                    p_valid = preds_sub[b, v_b].cpu().numpy()
                    t_valid = targets_sub[b, v_b].cpu().numpy()
                    b_valid = is_bound[b, v_b].cpu().numpy()
                    prob_valid = probs_sub[b, v_b].cpu().numpy()
                    
                    all_preds.extend(p_valid)
                    all_targets.extend(t_valid)
                    all_boundaries.extend(b_valid)
                    all_probs.append(prob_valid)
                    all_subjects.extend([subj_name] * len(p_valid))
                    
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    is_bound = np.array(all_boundaries, dtype=bool)
    probs_concat = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, 4))
    subjects_arr = np.array(all_subjects)
    
    # Primary Metrics
    acc = accuracy_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred) if len(y_true) > 0 else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if len(y_true) > 0 else 0.0
    
    prec, rec, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    
    # Boundary vs Stable Error Rates
    if is_bound.sum() > 0:
        err_boundary = 1.0 - accuracy_score(y_true[is_bound], y_pred[is_bound])
    else:
        err_boundary = 0.0
        
    if (~is_bound).sum() > 0:
        err_non_boundary = 1.0 - accuracy_score(y_true[~is_bound], y_pred[~is_bound])
    else:
        err_non_boundary = 0.0
        
    ece = compute_ece(probs_concat, y_true) if len(y_true) > 0 else 0.0
    
    # Per-subject breakdown
    subject_metrics = {}
    unique_subs = sorted(list(set(all_subjects)))
    for s in unique_subs:
        s_mask = (subjects_arr == s)
        s_true = y_true[s_mask]
        s_pred = y_pred[s_mask]
        s_prec, s_rec, s_f1, s_supp = precision_recall_fscore_support(
            s_true, s_pred, labels=[0, 1, 2, 3], zero_division=0
        )
        subject_metrics[s] = {
            "n_epochs": int(len(s_true)),
            "accuracy": float(accuracy_score(s_true, s_pred)),
            "macro_f1": float(f1_score(s_true, s_pred, average="macro", zero_division=0)),
            "kappa": float(cohen_kappa_score(s_true, s_pred)) if len(set(s_true)) > 1 else 0.0,
            "rem_recall": float(s_rec[3]),
            "wake_recall": float(s_rec[0]),
            "light_recall": float(s_rec[1]),
            "deep_recall": float(s_rec[2]),
            "rem_f1": float(s_f1[3]),
            "wake_f1": float(s_f1[0]),
            "light_f1": float(s_f1[1]),
            "deep_f1": float(s_f1[2]),
            "rem_support": int(s_supp[3])
        }
        
    return {
        "val_loss": float(np.mean(val_losses)) if val_losses else 0.0,
        "n_evaluated_epochs": int(len(y_true)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "kappa": float(kappa),
        "mcc": float(mcc),
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
        "confusion_matrix": cm.tolist(),
        "boundary_error_rate": float(err_boundary),
        "non_boundary_error_rate": float(err_non_boundary),
        "ece": float(ece),
        "subject_metrics": subject_metrics
    }


def train_single_run(
    condition: str,
    seq_len: int,
    seed: int,
    train_records: List[Dict[str, np.ndarray]],
    val_records: List[Dict[str, np.ndarray]],
    subject_to_idx: Dict[str, int],
    epochs: int = 80,
    batch_size: int = 4,
    lr: float = 1e-4,
    device: torch.device = torch.device("cuda:0"),
    checkpoint_dir: str = "research/experiments/EXP-017/checkpoints"
) -> Dict[str, Any]:
    """
    Trains Model A for exactly 80 epochs under frozen protocol controls.
    """
    print(f"\n=======================================================")
    print(f"STARTING RUN: Condition={condition} (T={seq_len}), Seed={seed}, Epochs={epochs}")
    print(f"=======================================================")
    
    # 1. Deterministic Seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.reset_peak_memory_stats(device)
        
    # 2. Build Datasets & Loaders
    train_ds = ControlledSleepDataset(train_records, subject_to_idx=subject_to_idx, seq_len=seq_len)
    val_ds = ControlledSleepDataset(val_records, subject_to_idx=subject_to_idx, seq_len=seq_len)
    
    # Worker 0 for strict determinism
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 3. Model & Optimizer
    model = ConventionalCNNBiGRU(
        d_model=64,
        d_temp=96,
        n_classes=4,
        n_meta=5,
        dropout=0.2,
        use_rcc_loss=True
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model A Instantiated: {n_params:,} parameters (Device: {device})")
    
    train_history = []
    t_train_start = time.time()
    
    for ep in range(epochs):
        model.train()
        batch_losses = []
        t_ep_start = time.time()
        
        for step, batch in enumerate(train_loader):
            x_ihr = batch["ihr"].to(device)
            x_acc = batch["accel"].to(device)
            x_meta = batch["meta"].to(device)
            targets = batch["y"].to(device)
            mask = batch["mask"].to(device)
            subj_idx = batch["subject_idx"].to(device)
            seq_lens = batch.get("seq_len", None)
            
            optimizer.zero_grad()
            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subj_idx, seq_lens=seq_lens)
            loss = out["total_loss"]
            
            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"CRITICAL ANOMALY: Abnormal loss {loss.item()} at epoch {ep+1}, step {step+1}")
                
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
            
        ep_loss = float(np.mean(batch_losses))
        ep_time = time.time() - t_ep_start
        
        # Inter-epoch validation evaluation (matched to EXP-010/011/013 RNG stepping)
        val_eval_600 = evaluate_model_on_epochs(model, val_loader, device, max_eval_epochs=600)
        
        train_history.append({
            "epoch": ep + 1,
            "train_loss": ep_loss,
            "val_loss": val_eval_600["val_loss"],
            "val_macro_f1_600": val_eval_600["macro_f1"],
            "val_rem_recall_600": val_eval_600["rem_recall"],
            "val_accuracy_600": val_eval_600["accuracy"],
            "epoch_time_s": ep_time
        })
        
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            print(f"  Epoch {ep+1:02d}/{epochs:02d} | Train Loss: {ep_loss:.4f} | "
                  f"Val Macro-F1 (600): {val_eval_600['macro_f1']:.4f} | "
                  f"Val REM Recall (600): {val_eval_600['rem_recall']:.4f} ({ep_time:.1f}s)")
            
    total_train_time = time.time() - t_train_start
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
    print(f"Training Complete in {total_train_time:.1f}s. Peak VRAM: {peak_vram_mb:.1f} MB")
    
    # Save checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, f"model_A_{condition}_seed{seed}_epoch80.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved Checkpoint: {ckpt_path}")
    
    # 4. Final Evaluations at Locked Epoch 80
    print("Running Primary Evaluation (Matched-600 positions)...")
    final_eval_600 = evaluate_model_on_epochs(model, val_loader, device, max_eval_epochs=600)
    
    final_eval_extended = None
    if condition == "T1200":
        print("Running Secondary Descriptive Evaluation (Extended coverage <= 1200)...")
        final_eval_extended = evaluate_model_on_epochs(model, val_loader, device, max_eval_epochs=1200)
        
    return {
        "condition": condition,
        "seq_len": seq_len,
        "seed": seed,
        "n_params": n_params,
        "train_time_s": total_train_time,
        "peak_vram_mb": peak_vram_mb,
        "checkpoint_path": ckpt_path,
        "training_history": train_history,
        "eval_matched600": final_eval_600,
        "eval_extended_descriptive": final_eval_extended
    }


def compute_paired_cluster_bootstrap(
    deltas_by_subject: Dict[str, float],
    n_iterations: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes paired subject-level cluster bootstrap over N=5 validation subjects.
    """
    np.random.seed(seed)
    subjects = sorted(list(deltas_by_subject.keys()))
    N = len(subjects)
    delta_vals = np.array([deltas_by_subject[s] for s in subjects])
    
    bootstrap_means = []
    for _ in range(n_iterations):
        indices = np.random.choice(N, size=N, replace=True)
        resampled_mean = float(np.mean(delta_vals[indices]))
        bootstrap_means.append(resampled_mean)
        
    bootstrap_means = np.array(bootstrap_means)
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))
    point_mean = float(np.mean(delta_vals))
    point_median = float(np.median(delta_vals))
    
    return {
        "point_mean": point_mean,
        "point_median": point_median,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "strictly_positive": bool(ci_lower > 0.0),
        "strictly_negative": bool(ci_upper < 0.0),
        "spans_zero": bool(ci_lower <= 0.0 <= ci_upper),
        "bootstrap_samples": bootstrap_means.tolist()
    }


def run_governed_exp017():
    print("=" * 80)
    print("EXP-017: CONTROLLED EVALUATION OF SEQUENCE TRUNCATION IN WEARABLE SLEEP STAGING")
    print("=" * 80)
    
    base_dir = "/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels"
    os.chdir(base_dir)
    
    # STEP 1: PREFLIGHT AUDIT & REPRODUCIBILITY VERIFICATION
    print("\n--- STEP 1: PREFLIGHT AUDIT & GOVERNANCE CHECKS ---")
    gate = DatasetGate(base_dir=base_dir)
    
    ds_fp = gate.compute_dataset_fingerprint()
    print(f"1. Dataset Preprocessing Fingerprint: {ds_fp}")
    expected_fp = "8ae7a916a4cd5709"
    if ds_fp != expected_fp:
        raise ValueError(f"FINGERPRINT MISMATCH: Got {ds_fp}, expected {expected_fp}")
    print("   [PASS] Preprocessing fingerprint verified.")
    
    # Subject Discovery and Split Verification
    all_mat_files = sorted(glob.glob(os.path.join(gate.dataset_dir, "Bidslab*", "*", "labels.mat")))
    dataset_nights = [(f.split(os.sep)[-3], f.split(os.sep)[-2].replace("Night ", "")) for f in all_mat_files]
    subjects = sorted(list(set(s for s, _ in dataset_nights)))
    
    splits = make_reproducible_split(subjects, split_type="paper", seed=0)
    train_subs = splits["train"]
    val_subs = splits["val"]
    test_subs = splits["test"]
    
    split_info = gate.verify_split(train_subs, val_subs, test_subs, strict_cohort_coverage=True)
    split_hash = split_info["split_hash"]
    print(f"2. Split Hash: {split_hash} (Train: {len(train_subs)}, Val: {len(val_subs)}, Test: {len(test_subs)})")
    expected_split = "bf33573623f47f3f"
    if split_hash != expected_split:
        raise ValueError(f"SPLIT HASH MISMATCH: Got {split_hash}, expected {expected_split}")
    print("   [PASS] 3-way partition bitwise verified.")
    
    # Test Set Isolation Assertion
    print("3. Test Cohort Isolation Gate:")
    gate.assert_no_test_leakage("train", train_subs, test_subs)
    gate.assert_no_test_leakage("validation", val_subs, test_subs)
    print("   [PASS] Test set leakage check passed: 11 held-out test subjects strictly protected.")
    
    # Load Train and Validation nights from audited cache
    print("4. Loading Cached Features:")
    train_records = []
    val_records = []
    for s, n in dataset_nights:
        if s in set(train_subs):
            rec = load_night_features(gate.cache_dir, s, n)
            if rec is not None:
                train_records.append(rec)
        elif s in set(val_subs):
            rec = load_night_features(gate.cache_dir, s, n)
            if rec is not None:
                val_records.append(rec)
                
    print(f"   - Training records loaded:   {len(train_records)} usable nights (31 subjects)")
    print(f"   - Validation records loaded: {len(val_records)} usable nights (5 subjects)")
    print(f"   - Test records:              0 loaded (STRICTLY LOCKED)")
    assert len(train_records) == 160, f"Expected 160 usable train nights, got {len(train_records)}"
    assert len(val_records) == 28, f"Expected 28 usable val nights, got {len(val_records)}"
    print("   [PASS] Usable night counts bitwise match official partition.")
    
    subject_to_idx = {s: i for i, s in enumerate(train_subs)}
    
    # STEP 2: RESOURCE ALLOCATION
    print("\n--- STEP 2: GPU RESOURCE LEASE ---")
    res_mgr = GPUResourceManager()
    
    out_dir = os.path.join(base_dir, "research", "experiments", "EXP-017")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    with res_mgr.lease("EXP-017", "STANDARD") as device_str:
        device = torch.device(device_str if torch.cuda.is_available() and device_str != "cpu" else "cpu")
        print(f"Leased Hardware: {device} ({torch.cuda.get_device_name(device)})")
        
        # STEP 3: EXECUTE ALL 6 RUNS (2 CONDITIONS X 3 SEEDS)
        seeds = [42, 123, 2024]
        conditions = [
            ("T600", 600),
            ("T1200", 1200)
        ]
        
        all_run_results = {}
        
        for cond_name, seq_len in conditions:
            for seed in seeds:
                run_id = f"ModelA_{cond_name}_seed{seed}"
                run_res = train_single_run(
                    condition=cond_name,
                    seq_len=seq_len,
                    seed=seed,
                    train_records=train_records,
                    val_records=val_records,
                    subject_to_idx=subject_to_idx,
                    epochs=80,
                    batch_size=4,
                    lr=1e-4,
                    device=device,
                    checkpoint_dir=ckpt_dir
                )
                all_run_results[run_id] = run_res
                
                # Immediately serialize individual run artifact
                run_artifact_path = os.path.join(out_dir, f"{run_id}_results.json")
                with open(run_artifact_path, "w") as f:
                    # Strip raw tensors / large matrices for compact JSON
                    compact_run = copy.deepcopy(run_res)
                    compact_run.pop("training_history", None)
                    json.dump(compact_run, f, indent=2)
                print(f"Persisted Run Artifact: {run_artifact_path}")
                
        # STEP 4: PRIMARY STATISTICAL ANALYSIS (MATCHED-600 POSITIONS)
        print("\n" + "=" * 80)
        print("STEP 4: COMPUTING PRIMARY ESTIMAND & STATISTICAL ANALYSIS")
        print("=" * 80)
        
        # Collect matched-600 metrics per seed and condition
        analysis_rows = []
        subject_delta_rows = []
        
        for seed in seeds:
            res_600 = all_run_results[f"ModelA_T600_seed{seed}"]["eval_matched600"]
            res_1200 = all_run_results[f"ModelA_T1200_seed{seed}"]["eval_matched600"]
            
            delta_rem = res_1200["rem_recall"] - res_600["rem_recall"]
            delta_acc = res_1200["accuracy"] - res_600["accuracy"]
            delta_macro_f1 = res_1200["macro_f1"] - res_600["macro_f1"]
            delta_kappa = res_1200["kappa"] - res_600["kappa"]
            delta_mcc = res_1200["mcc"] - res_600["mcc"]
            delta_wake_rec = res_1200["wake_recall"] - res_600["wake_recall"]
            delta_light_rec = res_1200["light_recall"] - res_600["light_recall"]
            delta_deep_rec = res_1200["deep_recall"] - res_600["deep_recall"]
            
            analysis_rows.append({
                "seed": seed,
                "T600_rem_recall": res_600["rem_recall"],
                "T1200_rem_recall": res_1200["rem_recall"],
                "delta_rem_recall": delta_rem,
                "T600_macro_f1": res_600["macro_f1"],
                "T1200_macro_f1": res_1200["macro_f1"],
                "delta_macro_f1": delta_macro_f1,
                "T600_accuracy": res_600["accuracy"],
                "T1200_accuracy": res_1200["accuracy"],
                "delta_accuracy": delta_acc,
                "T600_kappa": res_600["kappa"],
                "T1200_kappa": res_1200["kappa"],
                "delta_kappa": delta_kappa,
                "T600_mcc": res_600["mcc"],
                "T1200_mcc": res_1200["mcc"],
                "delta_mcc": delta_mcc,
                "delta_wake_recall": delta_wake_rec,
                "delta_light_recall": delta_light_rec,
                "delta_deep_recall": delta_deep_rec
            })
            
            # Subject-level paired deltas
            for s in val_subs:
                s_600 = res_600["subject_metrics"][s]
                s_1200 = res_1200["subject_metrics"][s]
                subject_delta_rows.append({
                    "seed": seed,
                    "subject": s,
                    "n_epochs": s_600["n_epochs"],
                    "T600_rem_recall": s_600["rem_recall"],
                    "T1200_rem_recall": s_1200["rem_recall"],
                    "delta_rem_recall": s_1200["rem_recall"] - s_600["rem_recall"],
                    "T600_macro_f1": s_600["macro_f1"],
                    "T1200_macro_f1": s_1200["macro_f1"],
                    "delta_macro_f1": s_1200["macro_f1"] - s_600["macro_f1"],
                    "T600_accuracy": s_600["accuracy"],
                    "T1200_accuracy": s_1200["accuracy"],
                    "delta_accuracy": s_1200["accuracy"] - s_600["accuracy"],
                    "delta_wake_recall": s_1200["wake_recall"] - s_600["wake_recall"],
                    "delta_light_recall": s_1200["light_recall"] - s_600["light_recall"],
                    "delta_deep_recall": s_1200["deep_recall"] - s_600["deep_recall"]
                })
                
        df_summary = pd.DataFrame(analysis_rows)
        df_subjects = pd.DataFrame(subject_delta_rows)
        
        # Aggregate subject-level deltas averaged across seeds for honest N=5 cluster bootstrap
        mean_delta_by_subject = {}
        wake_delta_by_subj = {}
        light_delta_by_subj = {}
        deep_delta_by_subj = {}
        macro_delta_by_subj = {}
        
        for s in val_subs:
            s_rows = df_subjects[df_subjects["subject"] == s]
            mean_delta_by_subject[s] = float(s_rows["delta_rem_recall"].mean())
            wake_delta_by_subj[s] = float(s_rows["delta_wake_recall"].mean())
            light_delta_by_subj[s] = float(s_rows["delta_light_recall"].mean())
            deep_delta_by_subj[s] = float(s_rows["delta_deep_recall"].mean())
            macro_delta_by_subj[s] = float(s_rows["delta_macro_f1"].mean())
            
        bootstrap_rem = compute_paired_cluster_bootstrap(mean_delta_by_subject, n_iterations=1000, seed=42)
        bootstrap_wake = compute_paired_cluster_bootstrap(wake_delta_by_subj, n_iterations=1000, seed=42)
        bootstrap_light = compute_paired_cluster_bootstrap(light_delta_by_subj, n_iterations=1000, seed=42)
        bootstrap_deep = compute_paired_cluster_bootstrap(deep_delta_by_subj, n_iterations=1000, seed=42)
        bootstrap_macro = compute_paired_cluster_bootstrap(macro_delta_by_subj, n_iterations=1000, seed=42)
        
        # Subject-level direction consistency
        pos_subjects = sum(1 for s in val_subs if mean_delta_by_subject[s] > 0.0)
        neg_subjects = sum(1 for s in val_subs if mean_delta_by_subject[s] < 0.0)
        tie_subjects = sum(1 for s in val_subs if mean_delta_by_subject[s] == 0.0)
        direction_consistency = f"{pos_subjects}/{len(val_subs)} positive, {neg_subjects}/{len(val_subs)} negative"
        
        # Check Decision Rule from design_draft.md Section K
        pt_est = bootstrap_rem["point_mean"]
        ci_low = bootstrap_rem["ci_lower"]
        ci_high = bootstrap_rem["ci_upper"]
        
        non_inferiority_passed = (
            bootstrap_wake["ci_lower"] > -0.030 and
            bootstrap_light["ci_lower"] > -0.030 and
            bootstrap_deep["ci_lower"] > -0.030 and
            bootstrap_macro["ci_lower"] > -0.010
        )
        
        if ci_low > 0.0 and pt_est >= 0.030 and non_inferiority_passed:
            decision = "SUPPORTED"
        elif ci_low > 0.0 and (pt_est < 0.030 or not non_inferiority_passed):
            decision = "WEAK_OR_UNCERTAIN_POSITIVE"
        elif ci_high <= 0.0 and pt_est <= 0.0:
            decision = "REJECTED"
        else:
            decision = "INCONCLUSIVE"
            
        print("\nDECISION RULE EVALUATION:")
        print(f"  Point Estimate:        {pt_est * 100:+.2f} pp")
        print(f"  95% Bootstrap CI:      [{ci_low * 100:+.2f} pp, {ci_high * 100:+.2f} pp]")
        print(f"  Direction Consistency: {direction_consistency}")
        print(f"  Non-Inferiority Gate:  {'PASSED' if non_inferiority_passed else 'FAILED'}")
        print(f"  -> FINAL VERDICT:      {decision}")
        
        # Save analysis tables
        df_summary.to_csv(os.path.join(out_dir, "EXP017_SUMMARY.csv"), index=False)
        df_subjects.to_csv(os.path.join(out_dir, "EXP017_SUBJECT_DELTAS.csv"), index=False)
        
        bootstrap_record = {
            "primary_rem_recall": bootstrap_rem,
            "wake_recall_check": bootstrap_wake,
            "light_recall_check": bootstrap_light,
            "deep_recall_check": bootstrap_deep,
            "macro_f1_check": bootstrap_macro,
            "direction_consistency": direction_consistency,
            "non_inferiority_passed": non_inferiority_passed,
            "decision": decision
        }
        with open(os.path.join(out_dir, "EXP017_BOOTSTRAP.json"), "w") as f:
            json.dump(bootstrap_record, f, indent=2)
            
        # STEP 5: SAVE MASTER EXPERIMENT RECORD IN RESEARCH MEMORY
        print("\n--- STEP 5: PERSISTING IN RESEARCH MEMORY ---")
        memory_path = os.path.join(base_dir, "research_memory.json")
        with open(memory_path, "r") as f:
            mem_data = json.load(f)
            
        exp017_record = {
            "experiment_id": "EXP-017",
            "name": "Controlled Sequence Truncation Horizon Evaluation",
            "parent_experiment": "EXP-014",
            "execution_mode": "REAL",
            "lifecycle_status": "COMPLETED",
            "real_training": True,
            "scientific_evidence": True,
            "eligible_for_model_selection": True,
            "eligible_for_research_conclusions": True,
            "eligible_for_leaderboard": False, # Development / validation comparison only
            "status": "COMPLETED",
            "decision": decision,
            "reason": f"EXP-017 completed 6 runs (seeds: {seeds}). Primary estimand point mean: {pt_est*100:+.2f} pp [95% CI: {ci_low*100:+.2f}, {ci_high*100:+.2f} pp]. Verdict: {decision}.",
            "hypothesis": "Training with the extended sequence condition (T <= 1200) improves validation REM recall relative to T=600 when both models are evaluated on the identical first 600 chronological epoch positions.",
            "research_question": "Does extending the sequence training horizon from T=600 to T <= 1200 materially affect sleep-staging performance when both models are evaluated on the identical first 600 chronological epoch positions, with invalid positions masked?",
            "change": "Sequence training horizon T=600 vs T<=1200 with recurrent pack_padded_sequence",
            "git_commit": "7194e17de69bf954fbd4a79a1263dbdb1d4ca6da",
            "dataset_fingerprint": ds_fp,
            "split_hash": split_hash,
            "seeds": seeds,
            "budget": "STANDARD",
            "gpu": device_str,
            "metrics": {
                "matched_600": {
                    "T600_mean_rem_recall": float(df_summary["T600_rem_recall"].mean()),
                    "T1200_mean_rem_recall": float(df_summary["T1200_rem_recall"].mean()),
                    "delta_rem_recall_mean": pt_est,
                    "delta_rem_recall_median": bootstrap_rem["point_median"],
                    "ci_95": [ci_low, ci_high],
                    "T600_mean_macro_f1": float(df_summary["T600_macro_f1"].mean()),
                    "T1200_mean_macro_f1": float(df_summary["T1200_macro_f1"].mean()),
                    "delta_macro_f1_mean": float(df_summary["delta_macro_f1"].mean()),
                    "T600_mean_accuracy": float(df_summary["T600_accuracy"].mean()),
                    "T1200_mean_accuracy": float(df_summary["T1200_accuracy"].mean()),
                    "delta_accuracy_mean": float(df_summary["delta_accuracy"].mean()),
                    "T600_mean_kappa": float(df_summary["T600_kappa"].mean()),
                    "T1200_mean_kappa": float(df_summary["T1200_kappa"].mean()),
                    "delta_kappa_mean": float(df_summary["delta_kappa"].mean())
                }
            },
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        
        # Check if EXP-017 already exists in memory and replace or append
        existing_idx = None
        for i, e in enumerate(mem_data.get("experiments", [])):
            if e.get("experiment_id") == "EXP-017":
                existing_idx = i
                break
        if existing_idx is not None:
            mem_data["experiments"][existing_idx] = exp017_record
        else:
            mem_data["experiments"].append(exp017_record)
            
        with open(memory_path, "w") as f:
            json.dump(mem_data, f, indent=2)
            
        ResearchMemoryValidator.validate_memory_file(memory_path)
        print(f"Updated and validated {memory_path}.")
        print("\nALL GOVERNED TASKS COMPLETED SUCCESSFULLY.")


if __name__ == "__main__":
    run_governed_exp017()
