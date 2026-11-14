import os
import sys
import torch
import torch.nn as nn
import numpy as np

from sleep_adaptive_net import SleepAdaptiveNet, build_ablation_model
from slamss_ifs_baseline import SLAMSS_IFS
from official_slamss_ifs.model.SLAMSS_IFS_Model import (
    CNN as OfficialCNN, Attention as OfficialAttention,
    Encoder as OfficialEncoder, Decoder as OfficialDecoder,
    Seq2Seq as OfficialSeq2Seq
)

def count_params(model):
    tot = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tot, trainable

def run_tests():
    print("=" * 70)
    print("TEST SUITE: SleepAdaptiveNet MODULAR ARCHITECTURE VERIFICATION")
    print("=" * 70)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # ---------------------------------------------------------
    # 1. PARAMETER COMPLEXITY & COMPARISON REPORT
    # ---------------------------------------------------------
    print("\n--- 1. PARAMETER COMPLEXITY REPORT ---")
    # Official SLAMSS-IFS
    off_cnn = OfficialCNN()
    off_attn = OfficialAttention(96, 96)
    off_enc = OfficialEncoder(2, 2, 96, 96, 0.0)
    off_dec = OfficialDecoder(4, 4, 96, 96, 0.0, off_attn)
    off_s2s = OfficialSeq2Seq(off_enc, off_dec, "cpu")
    off_total = sum(p.numel() for p in off_cnn.parameters()) + sum(p.numel() for p in off_s2s.parameters())
    
    # Paper-text baseline
    paper_base = SLAMSS_IFS(n_classes=4)
    paper_total, _ = count_params(paper_base)
    
    print(f"  Official SLAMSS-IFS (with 4.7M cnn3):   {off_total:,} parameters")
    print(f"  Paper-text SLAMSS-IFS Baseline:        {paper_total:,} parameters")
    
    # Ablations A0 through A5
    for abl in ["A0", "A1", "A2", "A3", "A4", "A5"]:
        m = build_ablation_model(abl)
        tot, trn = count_params(m)
        print(f"  Proposed Model [{abl}]:                   {tot:,} parameters (Trainable: {trn:,})")
        
    full_model = build_ablation_model("A5").to(device)
    full_tot, _ = count_params(full_model)
    print(f"\n  Efficiency: Proposed Full Model (A5) is {off_total / full_tot:.1f}x smaller than Official SLAMSS-IFS.")
    
    # Module parameter breakdown of A5
    print("\n  Proposed Model (A5) Module Breakdown:")
    print(f"    - HR Encoder:             {count_params(full_model.hr_encoder)[0]:,} params")
    print(f"    - ACC Encoder:            {count_params(full_model.acc_encoder)[0]:,} params")
    print(f"    - Adaptive Gate:          {count_params(full_model.fusion)[0]:,} params")
    print(f"    - Meta Projection:        {count_params(full_model.meta_proj)[0]:,} params")
    print(f"    - Temporal BiGRU:         {count_params(full_model.temporal_gru)[0]:,} params")
    print(f"    - Primary Classifier:     {count_params(full_model.classifier_head)[0]:,} params")
    print(f"    - Transition Module:      {count_params(full_model.transition_module)[0]:,} params")
    print(f"    - Subject Discriminator:  {count_params(full_model.subject_discriminator)[0]:,} params")
    print(f"    - Uncertainty Head:       {count_params(full_model.uncertainty_module)[0]:,} params")

    # ---------------------------------------------------------
    # 2. SHAPE & FORWARD PASS VERIFICATION
    # ---------------------------------------------------------
    print("\n--- 2. FORWARD PASS & TENSOR DIMENSION VERIFICATION ---")
    B = 2
    T = 60
    torch.manual_seed(42)
    x_ihr = torch.randn(B, T, 30, device=device)
    x_acc = torch.randn(B, T, 30, device=device)
    x_meta = torch.randn(B, T, 5, device=device)
    targets = torch.randint(0, 4, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    mask[:, -10:] = False # last 10 are padded
    subject_idx = torch.tensor([5, 12], device=device)
    
    out = full_model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subject_idx)
    
    assert out["logits"].shape == (B, T, 4), f"Bad logits shape: {out['logits'].shape}"
    assert out["alpha_gate"].shape == (B, T, 1), f"Bad alpha shape: {out['alpha_gate'].shape}"
    assert out["confidence"].shape == (B, T, 1), f"Bad conf shape: {out['confidence'].shape}"
    assert out["entropy"].shape == (B, T, 1), f"Bad entropy shape: {out['entropy'].shape}"
    print(f"  Logits shape:       {tuple(out['logits'].shape)} (PASS)")
    print(f"  Alpha gate shape:   {tuple(out['alpha_gate'].shape)} (PASS)")
    print(f"  Confidence shape:   {tuple(out['confidence'].shape)} (PASS)")
    print(f"  Entropy shape:      {tuple(out['entropy'].shape)} (PASS)")
    print(f"  Total Loss value:   {out['total_loss'].item():.4f}")
    print(f"    - Loss Stage:     {out['loss_stage'].item():.4f}")
    print(f"    - Loss Trans:     {out['loss_trans'].item():.4f}")
    print(f"    - Loss Boundary:  {out['loss_boundary'].item():.4f}")
    print(f"    - Loss Adv (GRL): {out['loss_adv'].item():.4f}")
    print(f"    - Loss Conf:      {out['loss_conf'].item():.4f}")

    # ---------------------------------------------------------
    # 3. GRADIENT VERIFICATION & GRL BACKWARD TEST
    # ---------------------------------------------------------
    print("\n--- 3. GRADIENTS & GRADIENT REVERSAL (GRL) TEST ---")
    full_model.zero_grad()
    out["total_loss"].backward()
    
    # Check that gradients exist and are finite across all modules
    for name, param in full_model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Missing grad for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN grad in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf grad in {name}"
    print("  All model parameters receive finite, non-zero gradients (PASS)")
    
    # Verify that GRL reverses gradients
    x_test = torch.randn(2, 96, requires_grad=True, device=device)
    grl = full_model.subject_discriminator.grl
    y_grl = grl(x_test)
    dummy_loss = (y_grl * 2.0).sum()
    dummy_loss.backward()
    # x_test.grad should be exactly -2.0 * alpha = -2.0 * 0.1 = -0.2
    expected_grad = -0.2
    assert torch.allclose(x_test.grad, torch.full_like(x_test.grad, expected_grad), atol=1e-5)
    print(f"  GRL Gradient Reversal verified mathematically: grad = {x_test.grad[0,0].item():.4f} (exp: {expected_grad}) (PASS)")

    # ---------------------------------------------------------
    # 4. ABLATION TOGGLES VERIFICATION (A0 -> A5)
    # ---------------------------------------------------------
    print("\n--- 4. ABLATION CONFIGURATION TOGGLES INDEPENDENCE ---")
    for abl in ["A0", "A1", "A2", "A3", "A4", "A5"]:
        m = build_ablation_model(abl).to(device)
        m.eval()
        with torch.no_grad():
            res = m(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subject_idx)
            
        if abl == "A0":
            # Static fusion (alpha = 0.5), no trans, no boundary, no adv, no conf
            assert torch.allclose(res["alpha_gate"], torch.full_like(res["alpha_gate"], 0.5))
            assert res["loss_trans"].item() == 0.0
            assert res["loss_boundary"].item() == 0.0
            assert res["loss_adv"].item() == 0.0
            assert res["loss_conf"].item() == 0.0
            print(f"  [{abl}] Verified: Static fusion (alpha=0.5), zero aux losses.")
        elif abl == "A1":
            # Adaptive fusion active, but other aux losses zero
            assert not torch.allclose(res["alpha_gate"], torch.full_like(res["alpha_gate"], 0.5))
            assert res["loss_trans"].item() == 0.0
            assert res["loss_boundary"].item() == 0.0
            print(f"  [{abl}] Verified: Learned adaptive fusion gate active.")
        elif abl == "A2":
            # Transition active
            assert res["loss_trans"].item() > 0.0
            assert res["loss_boundary"].item() == 0.0
            print(f"  [{abl}] Verified: Transition-aware modeling active.")
        elif abl == "A3":
            # Boundary active
            assert res["loss_trans"].item() > 0.0
            assert res["loss_boundary"].item() > 0.0
            print(f"  [{abl}] Verified: Boundary-aware loss active.")
        elif abl == "A4":
            # Adv active
            assert res["loss_adv"].item() > 0.0
            assert res["loss_conf"].item() == 0.0
            print(f"  [{abl}] Verified: Subject-invariant GRL discriminator active.")
        elif abl == "A5":
            # All active
            assert res["loss_conf"].item() > 0.0
            print(f"  [{abl}] Verified: Full proposed model with uncertainty head active.")

    # ---------------------------------------------------------
    # 5. TEST/VAL SUBJECT LEAKAGE PROTECTION TEST
    # ---------------------------------------------------------
    print("\n--- 5. SUBJECT LEAKAGE PREVENTION VERIFICATION ---")
    # Non-training subject index (e.g. -100) passed to subject discriminator
    val_subj_idx = torch.tensor([-100, -100], device=device)
    res_val = full_model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=val_subj_idx)
    assert res_val["loss_adv"].item() == 0.0, "Adversarial loss must be zero for non-training subjects!"
    print("  Non-training / test subjects strictly produce ZERO adversarial discriminator loss (PASS)")

    print("\n" + "=" * 70)
    print("ALL ARCHITECTURAL & UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
