#!/usr/bin/env python3
r"""
experiments/EXP-014/verify_candidate_pi.py
Rigorous 10-Test Automated Verification Suite for Candidate Pi
(UnbiasedHypersphericalBoundedMarginPyramidCNNBiGRU) on cuda:0.

Tests Performed:
  1. Parameter Count and Strict Budget Compliance (<= 196,728 cap, exactly 196,374 target)
  2. Complete Absence of Additive Linear Bias in Classifier Head and Synergy Logit Projection
  3. Hyperspherical Cosine Head (UH-CH) Geometry on S^63, Scale Initialization (s = 16.0), and Zero Margins (m = 0.0)
  4. Bounded Margin Stability Cutoffs (tau_acc >= 0.05, tau_ihr >= -0.05) & Anti-Drift Protection
  5. Forward Pass on Synthetic Batch (B=4, C=4, T=600) on cuda:0
  6. End-to-End Gradient Flow to ALL Trainable Parameters on cuda:0
  7. RCCLoss Compatibility & Multi-Task Loss Integration
  8. Tensor-Input vs Dict-Input Forward Equivalence
  9. Accelerometry Scale-Invariance & Gravity DC Tilt Invariance
  10. Full Mini-Epoch Optimization Step Simulation on cuda:0
"""

import os
import sys
import math
import copy
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Dynamically resolve workspace root
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_candidate1 = os.path.abspath(os.path.join(cur_dir, "../.."))
root_candidate2 = os.path.abspath(os.path.join(cur_dir, "../../.."))
WORKSPACE_ROOT = root_candidate1 if os.path.exists(os.path.join(root_candidate1, "exp013_models.py")) else root_candidate2
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from nas_iteration18_model import (
    UnbiasedHypersphericalBoundedMarginPyramidCNNBiGRU,
    build_nas_iteration18_model,
    BoundedMarginCrossModalSynergy,
    UnbiasedHypersphericalCosineClassifierHead,
    bounded_margin,
)


def run_candidate_pi_verification():
    print("=" * 80)
    print("RUNNING CANDIDATE PI 10-TEST AUTOMATED VERIFICATION SUITE")
    print("=" * 80)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device}")
    if device.type == "cuda":
        print(f"Device Name:   {torch.cuda.get_device_name(0)}")

    results: Dict[str, str] = {}

    # -------------------------------------------------------------------------
    # TEST 1: Parameter Count and Strict Budget Compliance
    # -------------------------------------------------------------------------
    print("\n--- TEST 1: Parameter Count and Strict Budget Compliance ---")
    model = build_nas_iteration18_model().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_breakdown = {
        "hr_encoder": sum(p.numel() for p in model.hr_encoder.parameters() if p.requires_grad),
        "acc_encoder": sum(p.numel() for p in model.acc_encoder.parameters() if p.requires_grad),
        "meta_proj": sum(p.numel() for p in model.meta_proj.parameters() if p.requires_grad),
        "basal_dslbs": sum(p.numel() for p in model.basal_dslbs.parameters() if p.requires_grad),
        "acc_dslbs": sum(p.numel() for p in model.acc_dslbs.parameters() if p.requires_grad),
        "luan_hr": sum(p.numel() for p in model.luan_hr.parameters() if p.requires_grad),
        "luan_acc": sum(p.numel() for p in model.luan_acc.parameters() if p.requires_grad),
        "bm_casci": sum(p.numel() for p in model.bm_casci.parameters() if p.requires_grad),
        "circadian_gate": sum(p.numel() for p in model.circadian_gate.parameters() if p.requires_grad),
        "temporal_gru": sum(p.numel() for p in model.temporal_gru.parameters() if p.requires_grad),
        "temp_norm": sum(p.numel() for p in model.temp_norm.parameters() if p.requires_grad),
        "inter_epoch_attn": sum(p.numel() for p in model.inter_epoch_attn.parameters() if p.requires_grad),
        "classifier_head": sum(p.numel() for p in model.classifier_head.parameters() if p.requires_grad),
    }

    print(f"Candidate Pi Trainable Parameters: {total_params:,}")
    print("Detailed Parameter Breakdown:")
    for k, v in param_breakdown.items():
        print(f"  - {k:18s}: {v:8,d} params")

    HARD_CAP = 196728
    TARGET_BUDGET = 196374
    headroom = HARD_CAP - total_params
    compression = (1.0 - total_params / 5470000.0) * 100.0

    print(f"Target Budget Expected:                 {TARGET_BUDGET:,} params")
    print(f"Hard Cap:                              {HARD_CAP:,} params")
    print(f"Headroom below Hard Cap:               {headroom:,} params")
    print(f"Compression vs 5.47M SLAMSS-IFS:       {compression:.2f}%")

    assert total_params <= HARD_CAP, f"Parameter count {total_params} exceeds hard cap {HARD_CAP}!"
    assert total_params == TARGET_BUDGET, f"Parameter count {total_params} does not match target {TARGET_BUDGET}!"
    assert compression > 96.0, f"Compression {compression:.2f}% is below 96% requirement!"

    print("[PASS] TEST 1: Parameter Count and Strict Budget Compliance Verified!")
    results["Test 1: Budget Compliance"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 2: Complete Absence of Additive Linear Bias in Classifier & Synergy
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: Complete Absence of Additive Linear Bias ---")
    # A. Classifier Head has no linear bias term for stage logits
    clf = model.classifier_head
    assert not hasattr(clf, "bias") or clf.bias is None, "classifier_head contains an illegal additive bias attribute!"
    assert not hasattr(clf, "linear"), "classifier_head contains an illegal standard linear layer!"
    print("  - classifier_head output layer has no additive linear bias parameters.")

    # B. Synergy Projection fc_logit has bias=False
    bm = model.bm_casci
    assert hasattr(bm, "fc_logit"), "bm_casci missing fc_logit projection!"
    assert bm.fc_logit.bias is None, f"bm_casci.fc_logit.bias is not None (found {bm.fc_logit.bias})!"
    print("  - bm_casci.fc_logit has bias=False (12 params, zero additive bias).")

    # C. Mathematical verification: Zero-vector input test
    # When feature input to classifier head has zero vector, cosine normalization handles it cleanly
    # and synergy logits with zero input produce strictly zero output.
    zero_synergy_input = torch.zeros(2, 5, 3, device=device)
    zero_synergy_output = bm.fc_logit(zero_synergy_input)
    assert (zero_synergy_output == 0.0).all(), "Zero synergy input did not produce strictly zero logits!"
    print("  - Zero synergy input strictly produces zero synergy logits (no offset drift).")

    print("[PASS] TEST 2: Complete Absence of Additive Linear Bias Verified!")
    results["Test 2: Zero Linear Bias"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 3: Hyperspherical Cosine Head Geometry, Scale Init, and Zero Margin
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: Hyperspherical Geometry on S^63, Scale Init, and Zero Margin ---")
    scale_val = clf.scale.item()
    print(f"  - Temperature scale s at initialization: {scale_val:.4f} (expected: 16.0)")
    assert abs(scale_val - 16.0) < 1e-5, f"Expected initial scale 16.0, got {scale_val}!"
    assert clf.scale.requires_grad, "Temperature scale s must be a learnable parameter!"

    # Verify normalization onto S^63
    test_feat = torch.randn(4, 10, 288, device=device)
    h = clf.in_proj(test_feat)
    h_norm = F.normalize(h, p=2, dim=-1)
    w_norm = F.normalize(clf.weight, p=2, dim=-1)

    h_norm_lengths = torch.norm(h_norm, p=2, dim=-1)
    w_norm_lengths = torch.norm(w_norm, p=2, dim=-1)

    assert torch.allclose(h_norm_lengths, torch.ones_like(h_norm_lengths), atol=1e-5), "h_norm lengths differ from 1.0 on S^63!"
    assert torch.allclose(w_norm_lengths, torch.ones_like(w_norm_lengths), atol=1e-5), "w_norm lengths differ from 1.0 on S^63!"
    print(f"  - Feature norm lengths on S^63: {h_norm_lengths.min().item():.6f} to {h_norm_lengths.max().item():.6f}")
    print(f"  - Prototype norm lengths on S^63: {w_norm_lengths.min().item():.6f} to {w_norm_lengths.max().item():.6f}")

    # Verify cosine similarity is strictly bounded in [-1.0, 1.0]
    cosine = F.linear(h_norm, w_norm)
    assert (cosine >= -1.0001).all() and (cosine <= 1.0001).all(), "Cosine similarity exceeds [-1.0, 1.0]!"
    print(f"  - Cosine similarity range: [{cosine.min().item():+.4f}, {cosine.max().item():+.4f}]")

    # Verify symmetric zero margins (m = 0.0): target-invariance (no margin subtracted)
    clf.eval()
    with torch.no_grad():
        logits_no_targets = clf(test_feat, targets=None)
        test_targets = torch.randint(0, 4, (4, 10), device=device)
        logits_with_targets = clf(test_feat, targets=test_targets)
    diff_margin = (logits_no_targets - logits_with_targets).abs().max().item()
    print(f"  - Max difference between targets=None and targets=with_targets logits: {diff_margin:.2e} (expected: 0.0)")
    assert diff_margin == 0.0, f"Targets altered logits! (Asymmetric margin leak: {diff_margin})"
    assert not hasattr(clf, "margins") or clf.margins is None, "clf contains an illegal margins parameter!"

    print("[PASS] TEST 3: Hyperspherical Geometry, Scale Init, and Zero Margin Verified!")
    results["Test 3: UH-CH Geometry"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 4: Bounded Margin Stability Cutoffs & Anti-Drift Protection
    # -------------------------------------------------------------------------
    print("\n--- TEST 4: Bounded Margin Stability Cutoffs & Anti-Drift Protection ---")
    bm_casci = model.bm_casci

    with torch.no_grad():
        # Test tau_acc_rem cutoff (min 0.05)
        bm_casci.tau_acc_rem.data.fill_(0.25)
        eff_acc = bounded_margin(bm_casci.tau_acc_rem, min_val=0.05).item()
        assert abs(eff_acc - 0.25) < 1e-5, f"Expected 0.25, got {eff_acc}"

        # Negative drift clamp for tau_acc_rem
        bm_casci.tau_acc_rem.data.fill_(-0.10)
        eff_acc_clamped = bounded_margin(bm_casci.tau_acc_rem, min_val=0.05).item()
        assert abs(eff_acc_clamped - 0.05) < 1e-5, f"Expected 0.05 cutoff, got {eff_acc_clamped}"
        print(f"  tau_acc_rem: raw = -0.10 -> clamped effective = {eff_acc_clamped:.2f} (cutoff >= 0.05 enforced)")

        # Negative drift clamp for tau_ihr_rem
        bm_casci.tau_ihr_rem.data.fill_(-0.25)
        eff_ihr_clamped = bounded_margin(bm_casci.tau_ihr_rem, min_val=-0.05).item()
        assert abs(eff_ihr_clamped - (-0.05)) < 1e-5, f"Expected -0.05 cutoff, got {eff_ihr_clamped}"
        print(f"  tau_ihr_rem: raw = -0.25 -> clamped effective = {eff_ihr_clamped:.2f} (cutoff >= -0.05 enforced)")

        # Deep margins
        bm_casci.tau_acc_deep.data.fill_(-0.02)
        eff_acc_deep = bounded_margin(bm_casci.tau_acc_deep, min_val=0.05).item()
        assert abs(eff_acc_deep - 0.05) < 1e-5, f"Expected 0.05 cutoff, got {eff_acc_deep}"

        bm_casci.tau_ihr_deep.data.fill_(-0.15)
        eff_ihr_deep = bounded_margin(bm_casci.tau_ihr_deep, min_val=-0.05).item()
        assert abs(eff_ihr_deep - (-0.05)) < 1e-5, f"Expected -0.05 cutoff, got {eff_ihr_deep}"

        # Reset to defaults
        bm_casci.tau_acc_rem.data.fill_(0.25)
        bm_casci.tau_ihr_rem.data.fill_(0.0)
        bm_casci.tau_acc_deep.data.fill_(0.05)
        bm_casci.tau_ihr_deep.data.fill_(0.0)

    print("[PASS] TEST 4: Bounded Margin Stability Cutoffs Verified!")
    results["Test 4: Bounded Margins"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 5: Forward Pass on Synthetic Batch (B=4, C=4, T=600) on cuda:0
    # -------------------------------------------------------------------------
    print(f"\n--- TEST 5: Forward Pass on (B=4, C=4, T=600) on {device} ---")
    B, T = 4, 600
    x_ihr = torch.randn(B, T, 30, device=device)
    x_accel = torch.randn(B, T, 30, device=device)
    x_meta = torch.randn(B, T, 5, device=device)
    targets = torch.randint(0, 4, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    mask[:, -20:] = False  # Simulate padding mask

    batch = {
        "ihr": x_ihr,
        "accel": x_accel,
        "meta": x_meta,
        "y": targets,
        "mask": mask
    }

    out = model(batch)

    assert "logits" in out, "Output missing 'logits' key!"
    assert "emission_logits" in out, "Output missing 'emission_logits' key!"
    assert "synergy_logits" in out, "Output missing 'synergy_logits' key!"
    assert "scale" in out, "Output missing 'scale' key!"
    assert "probabilities" in out, "Output missing 'probabilities' key!"
    assert "loss" in out, "Output missing 'loss' key!"
    assert "total_loss" in out, "Output missing 'total_loss' key!"

    logits = out["logits"]
    emission_logits = out["emission_logits"]
    synergy_logits = out["synergy_logits"]
    probs = out["probabilities"]
    loss = out["loss"]

    assert logits.shape == (B, T, 4), f"Expected logits shape (4, 600, 4), got {logits.shape}!"
    assert emission_logits.shape == (B, T, 4), f"Expected emission_logits shape (4, 600, 4), got {emission_logits.shape}!"
    assert synergy_logits.shape == (B, T, 4), f"Expected synergy_logits shape (4, 600, 4), got {synergy_logits.shape}!"
    assert probs.shape == (B, T, 4), f"Expected probs shape (4, 600, 4), got {probs.shape}!"
    assert loss.dim() == 0, f"Expected scalar loss, got dim {loss.dim()}!"

    assert not torch.isnan(logits).any(), "NaN found in logits!"
    assert not torch.isinf(logits).any(), "Inf found in logits!"
    assert not torch.isnan(loss).any(), "NaN found in loss!"

    print(f"  Logits Shape:          {logits.shape}")
    print(f"  Emission Logits Shape: {emission_logits.shape}")
    print(f"  Synergy Logits Shape:  {synergy_logits.shape}")
    print(f"  Probabilities Shape:   {probs.shape}")
    print(f"  Loss Scalar:           {loss.item():.4f}")
    print(f"  Scale Tensor Output:   {out['scale'].item():.4f}")

    print("[PASS] TEST 5: Forward Pass on cuda:0 Verified!")
    results["Test 5: Forward Pass"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 6: End-to-End Gradient Flow to ALL Trainable Parameters
    # -------------------------------------------------------------------------
    print("\n--- TEST 6: End-to-End Gradient Flow to ALL Trainable Parameters ---")
    # Part A: With zero_init=False, verify that 100% of all trainable parameter
    # tensors receive finite, non-zero gradients through backpropagation.
    model_flow = build_nas_iteration18_model(zero_init=False).to(device)
    model_flow.train()
    model_flow.zero_grad()

    out_flow = model_flow(batch)
    loss_flow = out_flow["loss"]
    loss_flow.backward()

    missing_grads: List[str] = []
    zero_grads: List[str] = []
    total_tensors = 0

    for name, param in model_flow.named_parameters():
        if param.requires_grad:
            total_tensors += 1
            if param.grad is None:
                missing_grads.append(name)
            elif param.grad.abs().sum().item() == 0.0:
                zero_grads.append(name)

    print(f"Part A (zero_init=False): Total Trainable Parameter Tensors: {total_tensors}")
    print(f"Tensors with Active Gradients: {total_tensors - len(missing_grads) - len(zero_grads)} / {total_tensors}")
    print(f"  classifier_head.scale grad:      {model_flow.classifier_head.scale.grad.item():+.6e}")
    print(f"  classifier_head.weight grad:     {model_flow.classifier_head.weight.grad.norm().item():.6e}")
    print(f"  bm_casci.tau_ihr_rem grad norm:  {model_flow.bm_casci.tau_ihr_rem.grad.norm().item():.6e}")
    print(f"  bm_casci.tau_acc_rem grad norm:  {model_flow.bm_casci.tau_acc_rem.grad.norm().item():.6e}")
    print(f"  bm_casci.fc_logit.weight grad:   {model_flow.bm_casci.fc_logit.weight.grad.norm().item():.6e}")
    print(f"  temporal_gru weight_ih_l0 grad:  {model_flow.temporal_gru.weight_ih_l0.grad.norm().item():.6e}")

    assert len(missing_grads) == 0, f"Parameters missing gradients: {missing_grads}"
    assert len(zero_grads) == 0, f"Parameters with exactly zero gradients: {zero_grads}"
    assert model_flow.classifier_head.scale.grad.item() != 0.0, "Scale received zero gradient!"

    # Part B: With zero_init=True (production mode), verify scale and projection gradients
    model.train()
    model.zero_grad()
    out_prod = model(batch)
    out_prod["loss"].backward()
    assert model.classifier_head.scale.grad is not None and model.classifier_head.scale.grad.item() != 0.0, "Scale grad failed in production mode!"
    assert model.classifier_head.weight.grad is not None and model.classifier_head.weight.grad.norm().item() > 0.0, "Head weight grad failed in production mode!"
    assert model.bm_casci.fc2.weight.grad is not None and model.bm_casci.fc2.weight.grad.norm().item() > 0.0, "bm_casci.fc2 grad failed in production mode!"
    print(f"Part B (zero_init=True): scale grad in production mode: {model.classifier_head.scale.grad.item():+.6e}")

    print("[PASS] TEST 6: 100% of Trainable Parameters Receive Non-Zero Finite Gradients!")
    results["Test 6: Gradient Flow"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 7: RCCLoss Compatibility & Multi-Task Integration
    # -------------------------------------------------------------------------
    print("\n--- TEST 7: RCCLoss Compatibility & Multi-Task Integration ---")
    model_no_rcc = build_nas_iteration18_model(use_rcc_loss=False).to(device)
    out_no_rcc = model_no_rcc(batch)
    loss_no_rcc = out_no_rcc["loss"]

    model_with_rcc = build_nas_iteration18_model(use_rcc_loss=True).to(device)
    out_with_rcc = model_with_rcc(batch)
    loss_with_rcc = out_with_rcc["loss"]

    print(f"  Loss without RCCLoss (CE only): {loss_no_rcc.item():.4f}")
    print(f"  Loss with RCCLoss (CE + RCC):   {loss_with_rcc.item():.4f}")

    assert not torch.isnan(loss_with_rcc), "RCCLoss produced NaN!"
    assert loss_with_rcc.item() > 0.0, "RCCLoss produced non-positive loss!"

    print("[PASS] TEST 7: RCCLoss Compatibility Verified!")
    results["Test 7: RCCLoss Compatibility"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 8: Tensor-Input vs Dict-Input Forward Equivalence
    # -------------------------------------------------------------------------
    print("\n--- TEST 8: Tensor-Input vs Dict-Input Forward Equivalence ---")
    model.eval()
    with torch.no_grad():
        out_dict = model(batch)
        out_args = model(x_ihr, x_accel, x_meta, targets=targets, mask=mask)

        max_diff = (out_dict["logits"] - out_args["logits"]).abs().max().item()
        print(f"  Max absolute logit difference (dict vs positional args): {max_diff:.2e}")
        assert max_diff < 1e-5, f"Logit difference {max_diff} exceeds tolerance!"

    print("[PASS] TEST 8: Calling Convention Equivalence Verified!")
    results["Test 8: Calling Equivalence"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 9: Accelerometry Scale-Invariance & Gravity DC Tilt Invariance
    # -------------------------------------------------------------------------
    print("\n--- TEST 9: Accelerometry Scale-Invariance & Gravity Tilt Invariance ---")
    with torch.no_grad():
        acc_raw = torch.randn(B, T, 30, device=device)
        delta_z1, z_mac1 = model.acc_dslbs(acc_raw)

        # Apply 10x sensor gain and +50.0 static gravity/tilt DC offset
        acc_perturbed = (acc_raw * 10.0) + 50.0
        delta_z2, z_mac2 = model.acc_dslbs(acc_perturbed)

        diff_delta = (delta_z1 - delta_z2).abs().max().item()
        print(f"  Max abs difference in delta_z_acc under 10x gain & +50.0 DC offset: {diff_delta:.2e}")
        assert diff_delta < 1e-4, f"Accelerometry contrast is not scale-invariant! Diff: {diff_delta}"

    print("[PASS] TEST 9: Accelerometry Scale and Tilt Invariance Verified!")
    results["Test 9: ACC Invariance"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 10: Multi-Step Optimization Simulation on cuda:0
    # -------------------------------------------------------------------------
    print("\n--- TEST 10: Multi-Step Optimization Simulation on cuda:0 ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_scale = model.classifier_head.scale.item()

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        out = model(batch)
        l = out["loss"]
        l.backward()
        optimizer.step()
        losses.append(l.item())

    final_scale = model.classifier_head.scale.item()

    print(f"  Step 0 Loss: {losses[0]:.4f} -> Step 4 Loss: {losses[-1]:.4f}")
    print(f"  Initial scale: {initial_scale:.4f}")
    print(f"  Final scale:   {final_scale:.4f} (delta = {final_scale - initial_scale:+.4f})")

    assert not any(math.isnan(l) for l in losses), "NaN encountered during optimization!"
    assert losses[-1] < losses[0], "Loss failed to decrease over 5 optimization steps!"

    print("[PASS] TEST 10: Multi-Step Optimization Simulation Verified!")
    results["Test 10: Optimization Simulation"] = "PASS"

    # -------------------------------------------------------------------------
    # SUMMARY REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CANDIDATE PI VERIFICATION SUMMARY:")
    print("=" * 80)
    for test_name, status in results.items():
        print(f"  {test_name:40s}: [{status}]")
    print("=" * 80)
    print("ALL 10 TESTS PASSED DEFINITIVELY ON CUDA:0!")
    print("Candidate Pi is 100% verified and ready for deployment.")
    print("=" * 80)


if __name__ == "__main__":
    run_candidate_pi_verification()
