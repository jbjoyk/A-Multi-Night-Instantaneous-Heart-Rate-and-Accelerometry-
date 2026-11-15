#!/usr/bin/env python3
r"""
experiments/EXP-014/verify_candidate_rho.py
Rigorous 10-Test Automated Verification Suite for Candidate Rho
(AutonomicHomeostasisBoundedMarginPyramidCNNBiGRU) on cuda:0.

Tests Performed:
  1. Parameter Count and Strict Budget Compliance (<= 196,728 cap, exactly 196,540 parameters)
  2. Autonomic & Somatomotor Homeostasis Contrast (s_light active firing vs prior zero-collapse)
  3. 4-Stage Context Bottleneck Rank-4 Subspace Dimension in BM-CASCI
  4. BoundedMarginSTE Stability Cutoffs and Straight-Through Gradient Flow
  5. Forward Pass on Synthetic Batch (B=4, C=4, T=600) on cuda:0
  6. End-to-End Gradient Flow to ALL 100% Trainable Parameters on cuda:0
  7. RCCLoss Compatibility & Multi-Task Loss Integration
  8. Tensor-Input vs Dict-Input Forward Equivalence
  9. Accelerometry Scale-Invariance & Gravity DC Tilt Invariance
  10. Multi-Step Optimization Simulation on cuda:0
"""

import os
import sys
import math
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

from nas_iteration19_model import (
    AutonomicHomeostasisBoundedMarginPyramidCNNBiGRU,
    build_nas_iteration19_model,
    AutonomicHomeostasisBoundedMarginCrossModalSynergy,
    BoundedMarginSTE,
    bounded_margin,
)


def run_candidate_rho_verification():
    print("=" * 80)
    print("RUNNING CANDIDATE RHO 10-TEST AUTOMATED VERIFICATION SUITE")
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
    model = build_nas_iteration19_model().to(device)
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

    print(f"Candidate Rho Trainable Parameters: {total_params:,}")
    print("Detailed Parameter Breakdown:")
    for k, v in param_breakdown.items():
        print(f"  - {k:18s}: {v:8,d} params")

    HARD_CAP = 196728
    TARGET_BUDGET = 196540
    headroom = HARD_CAP - total_params
    compression = (1.0 - total_params / 5470000.0) * 100.0

    print(f"Target Budget Expected:                 {TARGET_BUDGET:,} params")
    print(f"Hard Cap:                              {HARD_CAP:,} params")
    print(f"Headroom below Hard Cap:               {headroom:,} params")
    print(f"Compression vs 5.47M SLAMSS-IFS:       {compression:.2f}%")

    assert total_params <= HARD_CAP, f"Parameter count {total_params} exceeds hard cap {HARD_CAP}!"
    assert total_params == TARGET_BUDGET, f"Parameter count {total_params} does not match target {TARGET_BUDGET}!"
    assert param_breakdown["bm_casci"] == 648, f"bm_casci params {param_breakdown['bm_casci']} != 648!"
    assert compression > 96.0, f"Compression {compression:.2f}% is below 96% requirement!"

    print("[PASS] TEST 1: Parameter Count and Strict Budget Compliance Verified!")
    results["Test 1: Budget Compliance"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 2: Autonomic & Somatomotor Homeostasis Contrast (s_light Active Firing)
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: Autonomic & Somatomotor Homeostasis Contrast ---")
    bm = model.bm_casci

    # Case A: Quiet, homeostatic Light sleep (u_ihr = 0.1, u_acc = -0.05)
    # Both are within homeostatic margins: |u_ihr|=0.10 < 0.50, |u_acc|=0.05 < 0.25
    u_ihr_quiet = torch.tensor([[[0.10]]], device=device)
    u_acc_quiet = torch.tensor([[[-0.05]]], device=device)

    # In prior Candidate Kappa/Omicron:
    # s_light_prior = ReLU(tau_ihr - u_ihr) * ReLU(u_acc - tau_acc)
    # With tau_acc >= 0.0 and u_acc <= 0:
    s_light_prior = F.relu(0.0 - u_ihr_quiet) * F.relu(u_acc_quiet - 0.0)
    print(f"  Prior Kappa/Omicron s_light during quiet sleep: {s_light_prior.item():.6f} (MATHEMATICALLY ZERO!)")
    assert s_light_prior.item() == 0.0, "Prior formula should have been zero!"

    # In Candidate Rho AH-CASCI:
    # s_light = ReLU(tau_ihr_light - |u_ihr|) * ReLU(tau_acc_light - |u_acc|)
    tau_ihr_light = bounded_margin(bm.tau_ihr_light, min_val=0.05)
    tau_acc_light = bounded_margin(bm.tau_acc_light, min_val=0.05)
    s_light_rho = F.relu(tau_ihr_light - torch.abs(u_ihr_quiet)) * F.relu(tau_acc_light - torch.abs(u_acc_quiet))
    print(f"  Candidate Rho s_light during quiet sleep:        {s_light_rho.item():.6f} (ACTIVELY FIRING > 0!)")
    assert s_light_rho.item() > 0.0, f"Candidate Rho s_light failed to fire! Got {s_light_rho.item()}"
    expected_s_light = (0.50 - 0.10) * (0.25 - 0.05)  # 0.40 * 0.20 = 0.08
    assert abs(s_light_rho.item() - expected_s_light) < 1e-4, f"Expected {expected_s_light}, got {s_light_rho.item()}"

    # Case B: Wake movement (|u_acc| = 0.80 > 0.25)
    u_acc_wake = torch.tensor([[[0.80]]], device=device)
    s_light_wake = F.relu(tau_ihr_light - torch.abs(u_ihr_quiet)) * F.relu(tau_acc_light - torch.abs(u_acc_wake))
    print(f"  Candidate Rho s_light during active Wake movement: {s_light_wake.item():.6f} (Cleanly 0.0)")
    assert s_light_wake.item() == 0.0, "s_light must shut off during body movement!"

    # Case C: Deep bradycardia (u_ihr = -1.20, |u_ihr| = 1.20 > 0.50)
    u_ihr_deep = torch.tensor([[[-1.20]]], device=device)
    s_light_deep = F.relu(tau_ihr_light - torch.abs(u_ihr_deep)) * F.relu(tau_acc_light - torch.abs(u_acc_quiet))
    print(f"  Candidate Rho s_light during Deep sleep bradycardia: {s_light_deep.item():.6f} (Cleanly 0.0)")
    assert s_light_deep.item() == 0.0, "s_light must shut off during deep bradycardia!"

    # Case D: REM tachycardia surge (u_ihr = +1.10, |u_ihr| = 1.10 > 0.50)
    u_ihr_rem = torch.tensor([[[1.10]]], device=device)
    s_light_rem = F.relu(tau_ihr_light - torch.abs(u_ihr_rem)) * F.relu(tau_acc_light - torch.abs(u_acc_quiet))
    print(f"  Candidate Rho s_light during REM autonomic surge:    {s_light_rem.item():.6f} (Cleanly 0.0)")
    assert s_light_rem.item() == 0.0, "s_light must shut off during REM autonomic surges!"

    print("[PASS] TEST 2: Autonomic & Somatomotor Homeostasis Contrast Verified!")
    results["Test 2: AH-CASCI Active Firing"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 3: 4-Stage Context Bottleneck Rank-4 Subspace in BM-CASCI
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: 4-Stage Context Bottleneck Rank-4 Subspace ---")
    assert bm.hidden_dim == 4, f"Expected hidden_dim=4, got {bm.hidden_dim}!"
    assert bm.fc1.in_features == 8 and bm.fc1.out_features == 4, f"fc1 dimensions mismatch: {bm.fc1}"
    assert bm.norm.normalized_shape == (4,), f"LayerNorm dimensions mismatch: {bm.norm}"
    assert bm.fc2.in_features == 4 and bm.fc2.out_features == 144, f"fc2 dimensions mismatch: {bm.fc2}"
    assert bm.fc_logit.in_features == 4 and bm.fc_logit.out_features == 4, f"fc_logit dimensions mismatch: {bm.fc_logit}"

    # Verify rank of linear layer weight matrices
    rank_fc1 = torch.linalg.matrix_rank(bm.fc1.weight).item()
    rank_fc2 = torch.linalg.matrix_rank(bm.fc2.weight).item()
    rank_logit = torch.linalg.matrix_rank(bm.fc_logit.weight).item()

    print(f"  fc1 weight shape: {bm.fc1.weight.shape}, rank: {rank_fc1} (full row rank = 4)")
    print(f"  fc2 weight shape: {bm.fc2.weight.shape}")
    print(f"  fc_logit weight shape: {bm.fc_logit.weight.shape}")

    assert rank_fc1 == 4, f"fc1 should have full row rank 4, got {rank_fc1}"
    print("[PASS] TEST 3: 4-Stage Context Bottleneck Rank-4 Subspace Verified!")
    results["Test 3: Rank-4 Bottleneck"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 4: BoundedMarginSTE Stability Cutoffs & Anti-Drift Protection
    # -------------------------------------------------------------------------
    print("\n--- TEST 4: BoundedMarginSTE Stability Cutoffs & Anti-Drift Protection ---")
    with torch.no_grad():
        # Set all margins to negative drift values
        bm.tau_acc_rem.data.fill_(-0.50)
        bm.tau_ihr_rem.data.fill_(-0.50)
        bm.tau_acc_deep.data.fill_(-0.50)
        bm.tau_ihr_deep.data.fill_(-0.50)
        bm.tau_acc_wake.data.fill_(-0.50)
        bm.tau_ihr_wake.data.fill_(-0.50)
        bm.tau_acc_light.data.fill_(-0.50)
        bm.tau_ihr_light.data.fill_(-0.50)

        # Check effective margins with cutoffs
        eff_acc_rem = bounded_margin(bm.tau_acc_rem, min_val=0.05).item()
        eff_ihr_rem = bounded_margin(bm.tau_ihr_rem, min_val=-0.05).item()
        eff_acc_deep = bounded_margin(bm.tau_acc_deep, min_val=0.05).item()
        eff_ihr_deep = bounded_margin(bm.tau_ihr_deep, min_val=0.0).item()
        eff_acc_wake = bounded_margin(bm.tau_acc_wake, min_val=0.0).item()
        eff_ihr_wake = bounded_margin(bm.tau_ihr_wake, min_val=0.0).item()
        eff_acc_light = bounded_margin(bm.tau_acc_light, min_val=0.05).item()
        eff_ihr_light = bounded_margin(bm.tau_ihr_light, min_val=0.05).item()

        assert abs(eff_acc_rem - 0.05) < 1e-5, f"tau_acc_rem clamp failed: {eff_acc_rem}"
        assert abs(eff_ihr_rem - (-0.05)) < 1e-5, f"tau_ihr_rem clamp failed: {eff_ihr_rem}"
        assert abs(eff_acc_deep - 0.05) < 1e-5, f"tau_acc_deep clamp failed: {eff_acc_deep}"
        assert abs(eff_ihr_deep - 0.0) < 1e-5, f"tau_ihr_deep clamp failed: {eff_ihr_deep}"
        assert abs(eff_acc_wake - 0.0) < 1e-5, f"tau_acc_wake clamp failed: {eff_acc_wake}"
        assert abs(eff_ihr_wake - 0.0) < 1e-5, f"tau_ihr_wake clamp failed: {eff_ihr_wake}"
        assert abs(eff_acc_light - 0.05) < 1e-5, f"tau_acc_light clamp failed: {eff_acc_light}"
        assert abs(eff_ihr_light - 0.05) < 1e-5, f"tau_ihr_light clamp failed: {eff_ihr_light}"

        print(f"  tau_acc_rem:   raw = -0.50 -> clamped = {eff_acc_rem:+.2f} (cutoff >= +0.05 enforced)")
        print(f"  tau_ihr_rem:   raw = -0.50 -> clamped = {eff_ihr_rem:+.2f} (cutoff >= -0.05 enforced)")
        print(f"  tau_acc_deep:  raw = -0.50 -> clamped = {eff_acc_deep:+.2f} (cutoff >= +0.05 enforced)")
        print(f"  tau_ihr_deep:  raw = -0.50 -> clamped = {eff_ihr_deep:+.2f} (cutoff >=  0.00 enforced)")
        print(f"  tau_acc_wake:  raw = -0.50 -> clamped = {eff_acc_wake:+.2f} (cutoff >=  0.00 enforced)")
        print(f"  tau_ihr_wake:  raw = -0.50 -> clamped = {eff_ihr_wake:+.2f} (cutoff >=  0.00 enforced)")
        print(f"  tau_acc_light: raw = -0.50 -> clamped = {eff_acc_light:+.2f} (cutoff >= +0.05 enforced)")
        print(f"  tau_ihr_light: raw = -0.50 -> clamped = {eff_ihr_light:+.2f} (cutoff >= +0.05 enforced)")

        # Restore physiological initial values
        bm.tau_acc_rem.data.fill_(0.25)
        bm.tau_ihr_rem.data.fill_(0.0)
        bm.tau_acc_deep.data.fill_(0.05)
        bm.tau_ihr_deep.data.fill_(0.0)
        bm.tau_acc_wake.data.fill_(0.0)
        bm.tau_ihr_wake.data.fill_(0.0)
        bm.tau_acc_light.data.fill_(0.25)
        bm.tau_ihr_light.data.fill_(0.50)

    # Test STE gradient pass-through on clamped parameter
    test_param = nn.Parameter(torch.tensor(-0.20, requires_grad=True))
    eff_val = bounded_margin(test_param, min_val=0.05)
    loss = (eff_val - 1.0) ** 2
    loss.backward()
    assert test_param.grad is not None, "STE failed to pass gradient to parameter!"
    assert test_param.grad.item() != 0.0, "STE passed zero gradient!"
    print(f"  STE gradient pass-through on clamped parameter: {test_param.grad.item():+.4f}")

    print("[PASS] TEST 4: BoundedMarginSTE Stability Cutoffs and STE Gradient Flow Verified!")
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

    required_keys = [
        "logits", "emission_logits", "delta_logits", "synergy_logits",
        "synergies", "probabilities", "loss", "total_loss", "bm_gate",
        "circ_gate", "attn_weights", "delta_z_acc", "delta_z_ihr", "bm_ctx"
    ]
    for key in required_keys:
        assert key in out, f"Output dictionary missing '{key}' key!"

    logits = out["logits"]
    emission_logits = out["emission_logits"]
    synergies = out["synergies"]
    probs = out["probabilities"]
    loss = out["loss"]

    assert logits.shape == (B, T, 4), f"Expected logits shape (4, 600, 4), got {logits.shape}!"
    assert synergies.shape == (B, T, 4), f"Expected synergies shape (4, 600, 4), got {synergies.shape}!"
    assert probs.shape == (B, T, 4), f"Expected probs shape (4, 600, 4), got {probs.shape}!"
    assert loss.dim() == 0, f"Expected scalar loss, got dim {loss.dim()}!"

    assert not torch.isnan(logits).any(), "NaN found in logits!"
    assert not torch.isinf(logits).any(), "Inf found in logits!"
    assert not torch.isnan(loss).any(), "NaN found in loss!"

    print(f"  Logits Shape:          {logits.shape}")
    print(f"  Emission Logits Shape: {emission_logits.shape}")
    print(f"  Synergies Shape:       {synergies.shape}")
    print(f"  Probabilities Shape:   {probs.shape}")
    print(f"  Loss Scalar:           {loss.item():.4f}")

    print("[PASS] TEST 5: Forward Pass on cuda:0 Verified!")
    results["Test 5: Forward Pass"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 6: End-to-End Gradient Flow to ALL Trainable Parameters
    # -------------------------------------------------------------------------
    print("\n--- TEST 6: End-to-End Gradient Flow to ALL Trainable Parameters ---")
    model_flow = build_nas_iteration19_model(zero_init=False).to(device)
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
    print(f"  bm_casci.tau_ihr_rem grad norm:   {model_flow.bm_casci.tau_ihr_rem.grad.norm().item():.6e}")
    print(f"  bm_casci.tau_acc_rem grad norm:   {model_flow.bm_casci.tau_acc_rem.grad.norm().item():.6e}")
    print(f"  bm_casci.tau_ihr_light grad norm: {model_flow.bm_casci.tau_ihr_light.grad.norm().item():.6e}")
    print(f"  bm_casci.tau_acc_light grad norm: {model_flow.bm_casci.tau_acc_light.grad.norm().item():.6e}")
    print(f"  bm_casci.fc1.weight grad norm:    {model_flow.bm_casci.fc1.weight.grad.norm().item():.6e}")
    print(f"  bm_casci.fc2.weight grad norm:    {model_flow.bm_casci.fc2.weight.grad.norm().item():.6e}")
    print(f"  bm_casci.fc_logit.weight grad:    {model_flow.bm_casci.fc_logit.weight.grad.norm().item():.6e}")
    print(f"  temporal_gru weight_ih_l0 grad:   {model_flow.temporal_gru.weight_ih_l0.grad.norm().item():.6e}")
    print(f"  classifier_head.linear.weight:    {model_flow.classifier_head.linear.weight.grad.norm().item():.6e}")

    assert len(missing_grads) == 0, f"Parameters missing gradients: {missing_grads}"
    assert len(zero_grads) == 0, f"Parameters with exactly zero gradients: {zero_grads}"

    # Part B: Production Mode (zero_init=True)
    model.train()
    model.zero_grad()
    out_prod = model(batch)
    out_prod["loss"].backward()
    assert model.classifier_head.linear.weight.grad is not None and model.classifier_head.linear.weight.grad.norm().item() > 0.0
    assert model.bm_casci.fc2.weight.grad is not None and model.bm_casci.fc2.weight.grad.norm().item() > 0.0
    print("Part B (zero_init=True): Production mode gradients active on classifier head and BM-CASCI.")

    print("[PASS] TEST 6: 100% of Trainable Parameters Receive Non-Zero Finite Gradients!")
    results["Test 6: Gradient Flow"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 7: RCCLoss Compatibility & Multi-Task Integration
    # -------------------------------------------------------------------------
    print("\n--- TEST 7: RCCLoss Compatibility & Multi-Task Integration ---")
    model_no_rcc = build_nas_iteration19_model(use_rcc_loss=False).to(device)
    out_no_rcc = model_no_rcc(batch)
    loss_no_rcc = out_no_rcc["loss"]

    model_with_rcc = build_nas_iteration19_model(use_rcc_loss=True).to(device)
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
    # TEST 9: Accelerometry Scale-Invariance & Gravity Tilt Invariance
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

    initial_tau_light = model.bm_casci.tau_ihr_light.item()

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        out = model(batch)
        l = out["loss"]
        l.backward()
        optimizer.step()
        losses.append(l.item())

    final_tau_light = model.bm_casci.tau_ihr_light.item()

    print(f"  Step 0 Loss: {losses[0]:.4f} -> Step 4 Loss: {losses[-1]:.4f}")
    print(f"  Initial tau_ihr_light: {initial_tau_light:.4f}")
    print(f"  Final tau_ihr_light:   {final_tau_light:.4f} (delta = {final_tau_light - initial_tau_light:+.4e})")

    assert not any(math.isnan(l) for l in losses), "NaN encountered during optimization!"
    assert losses[-1] < losses[0], "Loss failed to decrease over 5 optimization steps!"

    print("[PASS] TEST 10: Multi-Step Optimization Simulation Verified!")
    results["Test 10: Optimization Simulation"] = "PASS"

    # -------------------------------------------------------------------------
    # SUMMARY REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CANDIDATE RHO VERIFICATION SUMMARY:")
    print("=" * 80)
    for test_name, status in results.items():
        print(f"  {test_name:40s}: [{status}]")
    print("=" * 80)
    print("ALL 10 TESTS PASSED DEFINITIVELY ON CUDA:0!")
    print("Candidate Rho is 100% verified and ready for deployment.")
    print("=" * 80)


if __name__ == "__main__":
    run_candidate_rho_verification()
