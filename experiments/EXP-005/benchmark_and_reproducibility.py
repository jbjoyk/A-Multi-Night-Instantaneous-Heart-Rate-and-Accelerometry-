import os
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

from slamss_ifs_baseline import SLAMSS_IFS, make_reproducible_split
from slamss_ifs_train import load_night_features, SLAMSSDataset
from test_rccloss_audit import RCCLoss, get_official_weights

def run_10_step_benchmark_and_reproducibility():
    print("=" * 65)
    print("SECTIONS 10 & 11: 10-STEP REAL BENCHMARK & REPRODUCIBILITY")
    print("=" * 65)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    cache_dir = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
    data_dir = "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0"
    
    # Load 10 real training nights
    import glob
    mat_files = sorted(glob.glob(os.path.join(data_dir, "Bidslab*", "*", "labels.mat")))
    all_nights = []
    for mf in mat_files:
        parts = mf.split(os.sep)
        all_nights.append((parts[-3], parts[-2]))
        
    all_subjects = sorted(list(set(s for s, n in all_nights)))
    split = make_reproducible_split(all_subjects, split_type="paper", seed=0)
    train_subs = set(split["train"])
    
    real_records = []
    for s, n in all_nights:
        if s in train_subs:
            rec = load_night_features(cache_dir, s, n)
            if rec is not None:
                real_records.append(rec)
            if len(real_records) >= 20:
                break
                
    print(f"Loaded {len(real_records)} real training night records.")
    dataset = SLAMSSDataset(real_records, max_seq_len=1200, pad=True)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    fn_w, fp_w = get_official_weights()
    loss_rccl = RCCLoss(fn_weights=fn_w, fp_weights=fp_w, cls=4).to(device)
    loss_ce = nn.CrossEntropyLoss(reduction="none")
    
    def run_10_steps(initial_state_dict=None, seed=42):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        model = SLAMSS_IFS(n_classes=4).to(device)
        if initial_state_dict is not None:
            model.load_state_dict(initial_state_dict)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model.train()
        
        step_losses = []
        timings = {"load": [], "forward": [], "backward": [], "opt": []}
        
        torch.cuda.reset_peak_memory_stats(device)
        t_start = time.time()
        
        data_iter = iter(loader)
        for step in range(10):
            t0 = time.time()
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
                
            x_ihr = batch["ihr"].to(device)
            x_accel = batch["accel"].to(device)
            x_freq = batch["freq"].to(device)
            x_time = batch["time"].to(device)
            targets = batch["y"].to(device)
            mask = batch["mask"].to(device)
            torch.cuda.synchronize()
            t_load = time.time() - t0
            timings["load"].append(t_load)
            
            # Forward pass
            t1 = time.time()
            logits = model(x_ihr, x_accel, x_freq, x_time, mask=mask)
            
            # Composite Loss (CE + 0.01 * RWL)
            B, T, K = logits.shape
            flat_logits = logits.view(-1, K)
            flat_targets = targets.view(-1)
            flat_mask = mask.view(-1).float()
            
            ce_loss = loss_ce(flat_logits, flat_targets)
            tot_ce = torch.sum(ce_loss * flat_mask)
            
            rw_loss = loss_rccl(flat_logits, flat_targets)
            tot_rwl = -torch.sum(rw_loss * flat_mask.unsqueeze(1).repeat(1, 4))
            
            total_loss = tot_ce + 0.01 * tot_rwl
            
            torch.cuda.synchronize()
            t_fwd = time.time() - t1
            timings["forward"].append(t_fwd)
            
            # Backward pass
            t2 = time.time()
            optimizer.zero_grad()
            total_loss.backward()
            torch.cuda.synchronize()
            t_bwd = time.time() - t2
            timings["backward"].append(t_bwd)
            
            # Optimizer step
            t3 = time.time()
            optimizer.step()
            torch.cuda.synchronize()
            t_opt = time.time() - t3
            timings["opt"].append(t_opt)
            
            step_losses.append(total_loss.item())
            
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        return step_losses, timings, model.state_dict(), peak_vram_mb

    # Step 1: Initialize a reference model to get fixed initial weights
    torch.manual_seed(42)
    init_model = SLAMSS_IFS(n_classes=4).to(device)
    init_state = copy.deepcopy(init_model.state_dict())
    
    # Run 1: 10 steps
    print("\n--- RUN 1 (10 Steps, Seed 42) ---")
    run1_losses, run1_times, run1_state, peak_vram = run_10_steps(initial_state_dict=init_state, seed=42)
    for i, l in enumerate(run1_losses):
        print(f"  Step {i+1:02d}: Total Loss = {l:.4f}")
        
    avg_load = np.mean(run1_times["load"]) * 1000
    avg_fwd = np.mean(run1_times["forward"]) * 1000
    avg_bwd = np.mean(run1_times["backward"]) * 1000
    avg_opt = np.mean(run1_times["opt"]) * 1000
    total_step = avg_load + avg_fwd + avg_bwd + avg_opt
    
    print("\n--- SECTION 10: PROFILING BENCHMARK RESULTS ---")
    print(f"  Data Loading Time:      {avg_load:.2f} ms / step")
    print(f"  Forward Pass Time:      {avg_fwd:.2f} ms / step")
    print(f"  Backward Pass Time:     {avg_bwd:.2f} ms / step")
    print(f"  Optimizer Step Time:    {avg_opt:.2f} ms / step")
    print(f"  Total Step Time:        {total_step:.2f} ms / step ({1000.0/total_step:.2f} steps/s)")
    print(f"  Peak VRAM Allocated:    {peak_vram:.2f} MB ({peak_vram/1024:.2f} GB)")
    
    # Run 2: Repeat with exact same initial weights and seed 42
    print("\n--- RUN 2 (10 Steps, Seed 42, Exact Duplicate) ---")
    run2_losses, run2_times, run2_state, _ = run_10_steps(initial_state_dict=init_state, seed=42)
    for i, l in enumerate(run2_losses):
        print(f"  Step {i+1:02d}: Total Loss = {l:.4f}")
        
    # Verify Reproducibility
    print("\n--- SECTION 11: REPRODUCIBILITY VERIFICATION ---")
    max_loss_diff = max(abs(l1 - l2) for l1, l2 in zip(run1_losses, run2_losses))
    print(f"  Max Absolute Loss Difference between Run 1 and Run 2: {max_loss_diff:.2e}")
    
    # Check weight difference across all tensors
    weight_diffs = []
    for k in run1_state.keys():
        diff = (run1_state[k] - run2_state[k]).abs().max().item()
        weight_diffs.append(diff)
    max_weight_diff = max(weight_diffs)
    print(f"  Max Absolute Model Weight Difference:                {max_weight_diff:.2e}")
    
    is_reproducible = (max_loss_diff < 1e-5) and (max_weight_diff < 1e-6)
    print(f"  Reproducibility Status: {'100% BITWISE/NUMERICALLY REPRODUCIBLE (PASSED)' if is_reproducible else 'FAILED'}")
    print("=" * 65)
    return is_reproducible

if __name__ == "__main__":
    run_10_step_benchmark_and_reproducibility()
