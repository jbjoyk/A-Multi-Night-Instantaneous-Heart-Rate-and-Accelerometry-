"""
Empirical Hardware Profiling Script for Official SLAMSS-IFS Repository Architecture
Model Architecture: 5,465,768 parameters (Official BIDSLabUMass/SLAMSS-IFS repository)
Hardware Target: Workstation NVIDIA GeForce RTX 3060 (12 GB VRAM)
Precision: PyTorch Automatic Mixed Precision (AMP)
Batch Sizes: 1, 2, 4
Sequence Length: L = 1200 epochs (full night)
"""

import sys
import os
import glob
import time
import gc
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "official_slamss_ifs"))
from model.SLAMSS_IFS_Model import CNN, Attention, Encoder, Decoder, Seq2Seq

CACHE_DIR = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
DEVICE_ID = 1  # Dedicated GPU 1 (RTX 3060 12GB)

def load_real_bidsleep_batch(cache_files, batch_size, L=1200, device="cuda:1"):
    t_x_list, t_time_list, trg_list, y_list, mask_list = [], [], [], [], []
    for i in range(batch_size):
        fpath = cache_files[i % len(cache_files)]
        d = np.load(fpath)
        X, y_raw, meta = d['X'], d['y'], d['meta']
        T = min(len(y_raw), L)
        
        # 4-class mapping: Wake=0, N1/N2=1, N3=2, REM=3
        y4 = np.zeros(L, dtype=np.int64)
        m = np.zeros(L, dtype=np.float32)
        raw_y = y_raw[:T]
        y4[:T][raw_y == 0] = 0
        y4[:T][(raw_y == 1) | (raw_y == 2)] = 1
        y4[:T][raw_y == 3] = 2
        y4[:T][raw_y == 4] = 3
        m[:T] = 1.0
        
        # Channel 0: IHR (1 Hz, 30 samples/epoch)
        # Channel 6: Accel Magnitude (1 Hz, 30 samples/epoch)
        x_night = np.zeros((L * 30, 2), dtype=np.float32)
        x_night[:T*30, 0] = X[:T, 0, :].reshape(-1).astype(np.float32)
        x_night[:T*30, 1] = X[:T, 6, :].reshape(-1).astype(np.float32)
        
        # Meta: 5 channels (sampling rate, clock, elapsed time, hr2_min, hr2_max)
        time_night = np.zeros((L, 5), dtype=np.float32)
        time_night[:T, :] = meta[:T, :5].astype(np.float32)
        
        # Targets: one-hot (B, L, 4)
        trg_onehot = np.zeros((L, 4), dtype=np.float32)
        for t in range(T):
            trg_onehot[t, y4[t]] = 1.0
            
        t_x_list.append(x_night)
        t_time_list.append(time_night)
        trg_list.append(trg_onehot)
        y_list.append(y4)
        mask_list.append(m)
        
    return (
        torch.tensor(np.array(t_x_list), device=device),
        torch.tensor(np.array(t_time_list), device=device),
        torch.tensor(np.array(trg_list), device=device),
        torch.tensor(np.array(y_list), device=device),
        torch.tensor(np.array(mask_list), device=device)
    )

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def build_official_slamss_ifs(device):
    enc_hid_dim = 96
    dec_hid_dim = 96
    in_w = 2
    output_dim = 4
    
    attn = Attention(enc_hid_dim, dec_hid_dim)
    enc = Encoder(in_w, in_w, enc_hid_dim, dec_hid_dim, 0.0)
    dec = Decoder(output_dim, output_dim, enc_hid_dim, dec_hid_dim, 0.0, attn)
    cnn = CNN().to(device)
    s2s = Seq2Seq(enc, dec, device).to(device)
    return cnn, s2s

def compute_loss(outputs, y_labels, mask):
    # outputs: (B, L, 4)
    # y_labels: (B, L)
    # mask: (B, L)
    B, L, C = outputs.shape
    logits_flat = outputs.view(-1, C)
    labels_flat = y_labels.view(-1)
    mask_flat = mask.view(-1)
    
    loss_ce = F.cross_entropy(logits_flat, labels_flat, reduction='none')
    loss_masked = (loss_ce * mask_flat).sum() / (mask_flat.sum() + 1e-8)
    return loss_masked

def profile_batch_size(cache_files, batch_size, L=1200, n_warmup=2, n_profile=5):
    torch.cuda.set_device(DEVICE_ID)
    device = torch.device(f"cuda:{DEVICE_ID}")
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE_ID)
    
    # Measure baseline memory
    mem_baseline = torch.cuda.memory_allocated(DEVICE_ID) / (1024 ** 2)
    
    cnn, s2s = build_official_slamss_ifs(device)
    cnn.train()
    s2s.train()
    
    # Model parameters memory
    mem_model = torch.cuda.memory_allocated(DEVICE_ID) / (1024 ** 2) - mem_baseline
    
    # Setup Adam optimizers
    lr = 1e-4
    opt_cnn = torch.optim.Adam(cnn.parameters(), lr=lr)
    opt_s2s = torch.optim.Adam(s2s.parameters(), lr=lr)
    scaler = torch.amp.GradScaler('cuda')
    
    # Load batch
    t_x, t_time, trg, y_labels, mask = load_real_bidsleep_batch(cache_files, batch_size, L=L, device=device)
    
    # Warmup iterations
    print(f"\n--- Profiling Batch Size B={batch_size}, L={L} (AMP) ---")
    print(f"Running {n_warmup} warmup iterations...")
    for w in range(n_warmup):
        opt_cnn.zero_grad(set_to_none=True)
        opt_s2s.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            cnn_feat = cnn(t_x, t_time)
            outputs = s2s(cnn_feat, t_x, trg, teacher_forcing_ratio=0.0)
            loss = compute_loss(outputs, y_labels, mask)
            
        scaler.scale(loss).backward()
        scaler.step(opt_cnn)
        scaler.step(opt_s2s)
        scaler.update()
        torch.cuda.synchronize(DEVICE_ID)
        
    # Reset peak stats after warmup to measure actual active training memory including optimizer states
    torch.cuda.reset_peak_memory_stats(DEVICE_ID)
    
    # Profile iterations
    fwd_times, bwd_times, total_times = [], [], []
    losses = []
    has_nan_or_inf = False
    
    print(f"Running {n_profile} timed profiling iterations...")
    for step in range(n_profile):
        opt_cnn.zero_grad(set_to_none=True)
        opt_s2s.zero_grad(set_to_none=True)
        
        t0 = time.perf_counter()
        with torch.amp.autocast('cuda'):
            cnn_feat = cnn(t_x, t_time)
            outputs = s2s(cnn_feat, t_x, trg, teacher_forcing_ratio=0.0)
            loss = compute_loss(outputs, y_labels, mask)
        torch.cuda.synchronize(DEVICE_ID)
        t1 = time.perf_counter()
        
        # Check numerical stability
        if torch.isnan(loss) or torch.isinf(loss) or torch.isnan(outputs).any():
            has_nan_or_inf = True
            
        losses.append(loss.item())
        
        scaler.scale(loss).backward()
        torch.cuda.synchronize(DEVICE_ID)
        t2 = time.perf_counter()
        
        scaler.step(opt_cnn)
        scaler.step(opt_s2s)
        scaler.update()
        torch.cuda.synchronize(DEVICE_ID)
        t3 = time.perf_counter()
        
        fwd_times.append((t1 - t0) * 1000.0)
        bwd_times.append((t2 - t1) * 1000.0)
        total_times.append((t3 - t0) * 1000.0)
        
    peak_allocated_mb = torch.cuda.max_memory_allocated(DEVICE_ID) / (1024 ** 2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(DEVICE_ID) / (1024 ** 2)
    current_allocated_mb = torch.cuda.memory_allocated(DEVICE_ID) / (1024 ** 2)
    
    # Estimate optimizer state memory
    opt_state_bytes = 0
    for opt in [opt_cnn, opt_s2s]:
        for p, state in opt.state.items():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    opt_state_bytes += v.numel() * v.element_size()
    opt_state_mb = opt_state_bytes / (1024 ** 2)
    
    mean_fwd = np.mean(fwd_times)
    mean_bwd = np.mean(bwd_times)
    mean_total = np.mean(total_times)
    throughput_nights_per_sec = (batch_size / (mean_total / 1000.0))
    
    result = {
        "batch_size": batch_size,
        "sequence_length": L,
        "peak_allocated_mb": round(peak_allocated_mb, 2),
        "peak_allocated_gib": round(peak_allocated_mb / 1024, 3),
        "peak_reserved_mb": round(peak_reserved_mb, 2),
        "peak_reserved_gib": round(peak_reserved_mb / 1024, 3),
        "model_params_mb": round(mem_model, 2),
        "optimizer_state_mb": round(opt_state_mb, 2),
        "mean_fwd_ms": round(mean_fwd, 2),
        "mean_bwd_ms": round(mean_bwd, 2),
        "mean_step_ms": round(mean_total, 2),
        "throughput_nights_per_sec": round(throughput_nights_per_sec, 3),
        "loss_mean": round(float(np.mean(losses)), 4),
        "numerical_stability": "STABLE (No NaNs/Infs)" if not has_nan_or_inf else "UNSTABLE (NaN/Inf detected)"
    }
    
    print(f"Results for B={batch_size}:")
    print(f"  Peak Allocated: {result['peak_allocated_mb']} MiB ({result['peak_allocated_gib']} GiB)")
    print(f"  Peak Reserved:  {result['peak_reserved_mb']} MiB ({result['peak_reserved_gib']} GiB)")
    print(f"  Optimizer State: {result['optimizer_state_mb']} MiB")
    print(f"  Timing (ms):    Fwd: {result['mean_fwd_ms']} ms | Bwd: {result['mean_bwd_ms']} ms | Step: {result['mean_step_ms']} ms")
    print(f"  Throughput:     {result['throughput_nights_per_sec']} nights/sec")
    print(f"  Stability:      {result['numerical_stability']} (Loss: {result['loss_mean']})")
    
    # Clean up
    del cnn, s2s, opt_cnn, opt_s2s, scaler, t_x, t_time, trg, y_labels, mask
    gc.collect()
    torch.cuda.empty_cache()
    
    return result

def main():
    print("=" * 75)
    print("OFFICIAL SLAMSS-IFS (5,465,768 PARAMETERS) HARDWARE PROFILING")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available:  {torch.cuda.is_available()}")
    print(f"Target Device:   GPU {DEVICE_ID} ({torch.cuda.get_device_name(DEVICE_ID)})")
    total_mem = torch.cuda.get_device_properties(DEVICE_ID).total_memory / (1024 ** 2)
    print(f"Total Device VRAM: {total_mem:.1f} MiB")
    print("=" * 75)
    
    # 1. Verify Model Parameter Counts
    device = torch.device(f"cuda:{DEVICE_ID}")
    cnn, s2s = build_official_slamss_ifs(device)
    cnn_total, cnn_train = count_parameters(cnn)
    s2s_total, s2s_train = count_parameters(s2s)
    total_params = cnn_total + s2s_total
    
    print("\n--- ARCHITECTURAL PARAMETER VERIFICATION ---")
    print(f"Official CNN Subnetwork:       {cnn_total:,} parameters")
    print(f"Official Seq2Seq Subnetwork:   {s2s_total:,} parameters")
    print(f"Combined Official Total:       {total_params:,} parameters")
    print(f"Verified against DEC-011 spec: {total_params == 5465768} (Expected: 5,465,768)")
    
    del cnn, s2s
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Check Data Cache
    cache_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.npz")))
    assert len(cache_files) > 0, f"No cache files found at {CACHE_DIR}"
    print(f"\nFeature Cache Verified: {len(cache_files)} usable nights in {CACHE_DIR}")
    
    # 3. Profile batch sizes 1, 2, 4
    batch_sizes = [1, 2, 4]
    results = {}
    for bs in batch_sizes:
        try:
            res = profile_batch_size(cache_files, batch_size=bs, L=1200, n_warmup=2, n_profile=5)
            results[f"B_{bs}"] = res
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            print(f"ERROR profiling B={bs}: {e}")
            print(tb)
            results[f"B_{bs}"] = {"error": str(e), "traceback": tb}
            
    # Save results to JSON
    out_path = "experiments/EXP-005/official_slamss_ifs_hw_profile.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved profile results to {out_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
