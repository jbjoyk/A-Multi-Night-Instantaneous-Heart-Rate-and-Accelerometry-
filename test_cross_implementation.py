import sys
import os
import torch
import torch.nn as nn
import numpy as np

# Add official repo to path
sys.path.append(os.path.abspath("official_slamss_ifs"))
from model.SLAMSS_IFS_Model import CNN as OfficialCNN, Attention as OfficialAttention, Encoder as OfficialEncoder, Decoder as OfficialDecoder, Seq2Seq as OfficialSeq2Seq
from slamss_ifs_baseline import SLAMSS_IFS as OurSLAMSS_IFS

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def run_cross_implementation_test():
    print("=" * 65)
    print("SECTION 9: ONE-BATCH CROSS-IMPLEMENTATION TEST")
    print("=" * 65)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Executing on device: {device}")
    
    # 1. Instantiate Official Architecture
    enc_hid_dim = 96
    dec_hid_dim = 96
    in_w = 2
    output_dim = 4
    
    off_attn = OfficialAttention(enc_hid_dim, dec_hid_dim)
    off_enc = OfficialEncoder(in_w, in_w, enc_hid_dim, dec_hid_dim, 0.0)
    off_dec = OfficialDecoder(output_dim, output_dim, enc_hid_dim, dec_hid_dim, 0.0, off_attn)
    off_cnn = OfficialCNN().to(device)
    off_seq2seq = OfficialSeq2Seq(off_enc, off_dec, device).to(device)
    
    off_cnn_params, _ = count_parameters(off_cnn)
    off_s2s_params, _ = count_parameters(off_seq2seq)
    off_total_params = off_cnn_params + off_s2s_params
    
    # 2. Instantiate Our Baseline Architecture
    our_model = OurSLAMSS_IFS(n_classes=4).to(device)
    
    our_total_params, _ = count_parameters(our_model)
    
    print("\n--- PARAMETER COUNT COMPARISON ---")
    print(f"Official Repository Architecture:")
    print(f"  - CNN + Micro-BiGRU + CNN_ACT + CNN3: {off_cnn_params:,} parameters")
    print(f"  - Seq2Seq (Encoder GRU + Decoder GRU + Attn + Micro GRU): {off_s2s_params:,} parameters")
    print(f"  - Total Official Parameters:          {off_total_params:,}")
    print(f"\nOur Initial Paper-Faithful Architecture (Paper Text Spec):")
    print(f"  - Total Baseline Parameters:          {our_total_params:,}")
    print(f"  - Architectural Ratio (Official / Ours): {off_total_params / our_total_params:.2f}x")
    
    # 3. Construct 1 Deterministic Batch (T = 60 epochs for fast check, 1 night)
    B = 1
    T = 60
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Official inputs
    t_x = torch.randn(B, T * 30, 2, device=device) # (1, 1800, 2)
    t_time = torch.randn(B, T, 5, device=device)   # (1, 60, 5)
    trg = torch.zeros(B, T, 4, device=device)
    trg[:, :, 0] = 1.0 # dummy target
    
    # Our baseline inputs: (B, T, 30)
    x_ihr = t_x[:, :, 0].view(B, T, 30)
    x_accel = t_x[:, :, 1].view(B, T, 30)
    x_freq = t_time[:, :, 0:1]
    x_time = t_time[:, :, 2:3]
    
    print("\n--- FORWARD PASS SHAPE VERIFICATION ---")
    # Official forward pass
    off_cnn.eval()
    off_seq2seq.eval()
    with torch.no_grad():
        # Step 1: CNN feature extractor
        off_cnn_out = off_cnn(t_x, t_time) # (B, T, 517)
        print(f"Official CNN Output Shape:       {tuple(off_cnn_out.shape)} (Expected: [1, {T}, 517])")
        
        # Step 2: Seq2Seq forward pass
        off_logits = off_seq2seq(off_cnn_out, t_x, trg, teacher_forcing_ratio=0.0)
        print(f"Official Seq2Seq Logits Shape:   {tuple(off_logits.shape)} (Expected: [1, {T}, 4])")
        
    # Our baseline forward pass
    our_model.eval()
    with torch.no_grad():
        our_logits, our_attn = our_model(x_ihr, x_accel, x_freq, x_time, return_attention=True)
        print(f"Our Baseline Logits Shape:       {tuple(our_logits.shape)} (Expected: [1, {T}, 4])")
        print(f"Our Baseline Attention Shape:    {tuple(our_attn.shape)} (Expected: [1, {T}, {T}])")
        
    print("\n--- SHAPE EQUIVALENCE CONFIRMATION ---")
    assert off_logits.shape == our_logits.shape, f"Logit shapes differ! {off_logits.shape} vs {our_logits.shape}"
    print(f"Final output logits shape MATCH: {tuple(off_logits.shape)}")
    print("=" * 65)

if __name__ == "__main__":
    run_cross_implementation_test()
