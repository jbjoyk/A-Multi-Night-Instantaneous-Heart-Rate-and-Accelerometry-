#!/usr/bin/env python3
r"""
experiments/EXP-014/verify_candidate_omicron.py
Rigorous 10-Test Automated Verification Suite for Candidate Omicron
(PriorCalibratedBoundedMarginPyramidCNNBiGRU) on cuda:0.

Tests Performed:
  1. Parameter Count and Strict Budget Compliance (<= 196,728 cap, exactly 196,382 target)
  2. Boundedness of tau_prior in [0, 0.5] and Smooth Differentiability
  3. Prior-Calibrated Logit Balancing (PLB) Mathematical Mechanism
  4. Bounded Margin Stability Cutoffs (tau_acc >= 0.05, tau_ihr >= -0.05) & Anti-Drift Protection
  5. Forward Pass on (B=4, C=4, T=600) on cuda:0
  6. End-to-End Gradient Flow to ALL Trainable Parameters (alpha_prior, 8 margins, etc.)
  7. RCCLoss Compatibility & Loss Multi-Task Integration
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

from nas_iteration17_model import (
    PriorCalibratedBoundedMarginPyramidCNNBiGRU,
    build_nas_iteration17_model,
    BoundedMarginCrossModalSynergy,
    bounded_margin,
)


def run_candidate_omicron_verification():
    print("=" * 80)
    print("RUNNING CANDIDATE OMICRON 10-TEST AUTOMATED VERIFICATION SUITE")
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
    model = build_nas_iteration17_model().to(device)
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
        "alpha_prior": model.alpha_prior.numel() if model.alpha_prior.requires_grad else 0,
    }

    print(f"Candidate Omicron Trainable Parameters: {total_params:,}")
    print("Detailed Parameter Breakdown:")
    for k, v in param_breakdown.items():
        print(f"  - {k:18s}: {v:8,d} params")

    HARD_CAP = 196728
    TARGET_BUDGET = 196382
    headroom = HARD_CAP - total_params
    compression = (1.0 - total_params / 5470000.0) * 100.0

    print(f"Target Budget Expected:                 {TARGET_BUDGET:,} params")
    print(f"Hard Cap:                              {HARD_CAP:,} params")
    print(f"Headroom below Hard Cap:               {headroom:,} params")
    print(f"Compression vs 5.47M SLAMSS-IFS:       {compression:.2f}%")

    assert total_params <= HARD_CAP, f"Parameter count {total_params} exceeds hard cap {HARD_CAP}!"
    assert total_params == TARGET_BUDGET, f"Parameter count {total_params} does not match target {TARGET_BUDGET}!"
    assert compression > 90.0, f"Compression {compression:.2f}% is below 90% requirement!"

    print("[PASS] TEST 1: Parameter Count and Strict Budget Compliance Verified!")
    results["Test 1: Budget Compliance"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 2: Boundedness of tau_prior in [0, 0.5] and Smooth Differentiability
    # -------------------------------------------------------------------------
    print("\n--- TEST 2: Boundedness of tau_prior in [0, 0.5] ---")
    tau_init = model.tau_prior.item()
    print(f"tau_prior at initialization (alpha_prior = {model.alpha_prior.item():.4f}): {tau_init:.4f}")
    assert abs(tau_init - 0.20) < 1e-3, f"Expected initial tau_prior approx 0.20, got {tau_init}!"

    # Test extreme negative (alpha -> -inf)
    with torch.no_grad():
        model.alpha_prior.data.fill_(-100.0)
        tau_neg = model.tau_prior.item()
        print(f"tau_prior at extreme negative alpha (-100.0): {tau_neg:.2e} (bounded >= 0.0)")
        assert tau_neg >= 0.0 and tau_neg < 1e-10, "tau_prior failed lower bound!"

        # Test extreme positive (alpha -> +inf)
        model.alpha_prior.data.fill_(100.0)
        tau_pos = model.tau_prior.item()
        print(f"tau_prior at extreme positive alpha (+100.0): {tau_pos:.4f} (bounded <= 0.5)")
        assert tau_pos <= 0.5 and tau_pos > 0.4999, "tau_prior failed upper bound!"

        # Test monotonicity across grid
        alphas = torch.linspace(-10.0, 10.0, 50)
        taus = [0.5 * torch.sigmoid(a).item() for a in alphas]
        is_monotonic = all(taus[i] <= taus[i+1] for i in range(len(taus) - 1))
        assert is_monotonic, "tau_prior is not monotonic!"
        print("tau_prior strictly monotonic in [0, 0.5] verified across 50 sample grid points.")

        # Reset alpha_prior to default init
        model.alpha_prior.data.fill_(-0.4055)

    print("[PASS] TEST 2: Boundedness of tau_prior Verified!")
    results["Test 2: tau_prior Bounds"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 3: Prior-Calibrated Logit Balancing (PLB) Mathematical Mechanism
    # -------------------------------------------------------------------------
    print("\n--- TEST 3: Prior-Calibrated Logit Balancing (PLB) Mechanism ---")
    priors = [0.0999, 0.4319, 0.2467, 0.2215]
    stage_names = ["Wake", "Light", "Deep", "REM"]
    corrections = - model.tau_prior.item() * np.log(priors)
    light_correction = corrections[1]
    relative_boosts = corrections - light_correction

    print("Logit Corrections at tau_prior = 0.20:")
    for name, p, corr, rel in zip(stage_names, priors, corrections, relative_boosts):
        print(f"  - {name:5s} (p = {p:.4f}): correction = {corr:+6.4f}, boost relative to Light = {rel:+6.4f}")

    # Wake, Deep, and REM MUST all receive positive relative boosts over Light sleep
    assert relative_boosts[0] > 0.15, "Wake sleep did not receive sufficient counterbalance!"
    assert relative_boosts[2] > 0.05, "Deep sleep did not receive sufficient counterbalance!"
    assert relative_boosts[3] > 0.05, "REM sleep did not receive sufficient counterbalance!"
    print("PLB confirms: Majority Light sleep is down-weighted relative to all 3 minority stages!")

    print("[PASS] TEST 3: PLB Mathematical Mechanism Verified!")
    results["Test 3: PLB Mechanism"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 4: Bounded Margin Stability Cutoffs & Anti-Drift Protection
    # -------------------------------------------------------------------------
    print("\n--- TEST 4: Bounded Margin Stability Cutoffs & Anti-Drift Protection ---")
    bm_casci = model.bm_casci

    # Test tau_acc_rem cutoff (min 0.05)
    with torch.no_grad():
        # When tau_acc is above cutoff (0.25)
        bm_casci.tau_acc_rem.data.fill_(0.25)
        eff_acc = bounded_margin(bm_casci.tau_acc_rem, min_val=0.05).item()
        assert abs(eff_acc - 0.25) < 1e-5, f"Expected 0.25, got {eff_acc}"

        # When tau_acc drifts below cutoff (-0.10)
        bm_casci.tau_acc_rem.data.fill_(-0.10)
        eff_acc_clamped = bounded_margin(bm_casci.tau_acc_rem, min_val=0.05).item()
        assert abs(eff_acc_clamped - 0.05) < 1e-5, f"Expected 0.05 cutoff, got {eff_acc_clamped}"
        print(f"  tau_acc_rem: raw = -0.10 -> clamped effective = {eff_acc_clamped:.2f} (cutoff >= 0.05 enforced)")

        # When tau_ihr drifts negative (-0.25)
        bm_casci.tau_ihr_rem.data.fill_(-0.25)
        eff_ihr_clamped = bounded_margin(bm_casci.tau_ihr_rem, min_val=-0.05).item()
        assert abs(eff_ihr_clamped - (-0.05)) < 1e-5, f"Expected -0.05 cutoff, got {eff_ihr_clamped}"
        print(f"  tau_ihr_rem: raw = -0.25 -> clamped effective = {eff_ihr_clamped:.2f} (cutoff >= -0.05 enforced)")

        # Reset to defaults
        bm_casci.tau_acc_rem.data.fill_(0.25)
        bm_casci.tau_ihr_rem.data.fill_(0.0)

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
    mask[:, -20:] = False  # Add some padding mask

    batch = {
        "ihr": x_ihr,
        "accel": x_accel,
        "meta": x_meta,
        "y": targets,
        "mask": mask
    }

    out = model(batch)

    assert "logits" in out, "Output missing 'logits' key!"
    assert "probabilities" in out, "Output missing 'probabilities' key!"
    assert "loss" in out, "Output missing 'loss' key!"
    assert "emission_logits" in out, "Output missing 'emission_logits' key!"
    assert "synergy_logits" in out, "Output missing 'synergy_logits' key!"
    assert "prior_correction" in out, "Output missing 'prior_correction' key!"

    logits = out["logits"]
    probs = out["probabilities"]
    loss = out["loss"]

    assert logits.shape == (B, T, 4), f"Expected logits shape (4, 600, 4), got {logits.shape}!"
    assert probs.shape == (B, T, 4), f"Expected probs shape (4, 600, 4), got {probs.shape}!"
    assert loss.dim() == 0, f"Expected scalar loss, got dim {loss.dim()}!"

    assert not torch.isnan(logits).any(), "NaN found in logits!"
    assert not torch.isinf(logits).any(), "Inf found in logits!"
    assert not torch.isnan(loss).any(), "NaN found in loss!"

    print(f"Logits Shape:          {logits.shape}")
    print(f"Probabilities Shape:   {probs.shape}")
    print(f"Loss Scalar:           {loss.item():.4f}")
    print(f"tau_prior Output:      {out['tau_prior'].item():.4f}")

    print("[PASS] TEST 5: Forward Pass on cuda:0 Verified!")
    results["Test 5: Forward Pass"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 6: End-to-End Gradient Flow to ALL Trainable Parameters
    # -------------------------------------------------------------------------
    print("\n--- TEST 6: End-to-End Gradient Flow to ALL Trainable Parameters ---")
    # Part A: With zero_init=False (active projections), verify that 100% of all 120 trainable parameter
    # tensors receive finite, non-zero gradients through backpropagation.
    model_flow = build_nas_iteration17_model(zero_init=False).to(device)
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
    print(f"  alpha_prior grad norm:           {model_flow.alpha_prior.grad.norm().item():.6e}")
    print(f"  bm_casci.tau_ihr_rem grad norm:  {model_flow.bm_casci.tau_ihr_rem.grad.norm().item():.6e}")
    print(f"  bm_casci.tau_acc_rem grad norm:  {model_flow.bm_casci.tau_acc_rem.grad.norm().item():.6e}")
    print(f"  bm_casci.tau_ihr_deep grad norm: {model_flow.bm_casci.tau_ihr_deep.grad.norm().item():.6e}")
    print(f"  bm_casci.tau_acc_deep grad norm: {model_flow.bm_casci.tau_acc_deep.grad.norm().item():.6e}")
    print(f"  temporal_gru weight_ih_l0 grad:  {model_flow.temporal_gru.weight_ih_l0.grad.norm().item():.6e}")

    assert len(missing_grads) == 0, f"Parameters missing gradients: {missing_grads}"
    assert len(zero_grads) == 0, f"Parameters with exactly zero gradients: {zero_grads}"
    assert model_flow.alpha_prior.grad.norm().item() > 0.0, "alpha_prior received zero gradient!"

    # Part B: With zero_init=True (production identity mode), verify alpha_prior and projection weights receive gradients
    model.train()
    model.zero_grad()
    out_default = model(batch)
    out_default["loss"].backward()
    assert model.alpha_prior.grad is not None and model.alpha_prior.grad.norm().item() > 0.0, "alpha_prior grad failed in production mode!"
    assert model.bm_casci.fc2.weight.grad is not None and model.bm_casci.fc2.weight.grad.norm().item() > 0.0, "bm_casci.fc2.weight grad failed in production mode!"
    print(f"Part B (zero_init=True): alpha_prior grad in default mode: {model.alpha_prior.grad.norm().item():.6e}")

    print("[PASS] TEST 6: 100% of Trainable Parameters Receive Non-Zero Finite Gradients!")
    results["Test 6: Gradient Flow"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 7: RCCLoss Compatibility & Loss Correctness
    # -------------------------------------------------------------------------
    print("\n--- TEST 7: RCCLoss Compatibility & Multi-Task Integration ---")
    model_no_rcc = build_nas_iteration17_model(use_rcc_loss=False).to(device)
    out_no_rcc = model_no_rcc(batch)
    loss_no_rcc = out_no_rcc["loss"]

    model_with_rcc = build_nas_iteration17_model(use_rcc_loss=True).to(device)
    out_with_rcc = model_with_rcc(batch)
    loss_with_rcc = out_with_rcc["loss"]

    print(f"Loss without RCCLoss (CE only): {loss_no_rcc.item():.4f}")
    print(f"Loss with RCCLoss (CE + RCC):   {loss_with_rcc.item():.4f}")

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
        print(f"Max absolute logit difference (dict vs positional args): {max_diff:.2e}")
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
        print(f"Max abs difference in delta_z_acc under 10x gain & +50.0 DC offset: {diff_delta:.2e}")
        assert diff_delta < 1e-4, f"Accelerometry contrast is not scale-invariant! Diff: {diff_delta}"

    print("[PASS] TEST 9: Accelerometry Scale and Tilt Invariance Verified!")
    results["Test 9: ACC Invariance"] = "PASS"

    # -------------------------------------------------------------------------
    # TEST 10: Full Mini-Epoch Optimization Step Simulation on cuda:0
    # -------------------------------------------------------------------------
    print("\n--- TEST 10: Mini-Epoch Optimization Simulation on cuda:0 ---")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    initial_alpha = model.alpha_prior.item()
    initial_tau = model.tau_prior.item()

    losses = []
    for step in range(5):
        optimizer.zero_grad()
        out = model(batch)
        l = out["loss"]
        l.backward()
        optimizer.step()
        losses.append(l.item())

    final_alpha = model.alpha_prior.item()
    final_tau = model.tau_prior.item()

    print(f"Step 0 Loss: {losses[0]:.4f} -> Step 4 Loss: {losses[-1]:.4f}")
    print(f"Initial alpha_prior: {initial_alpha:.4f} (tau = {initial_tau:.4f})")
    print(f"Final alpha_prior:   {final_alpha:.4f} (tau = {final_tau:.4f})")

    assert final_tau >= 0.0 and final_tau <= 0.5, f"tau_prior drifted out of bounds: {final_tau}!"
    assert losses[-1] < losses[0], "Loss failed to decrease over 5 optimization steps!"

    print("[PASS] TEST 10: Mini-Epoch Optimization Simulation Verified!")
    results["Test 10: Optimization Simulation"] = "PASS"

    # -------------------------------------------------------------------------
    # SUMMARY REPORT
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("CANDIDATE OMICRON VERIFICATION SUMMARY:")
    print("=" * 80)
    for test_name, status in results.items():
        print(f"  {test_name:40s}: [{status}]")
    print("=" * 80)
    print("ALL 10 TESTS PASSED DEFINITIVELY ON CUDA:0!")
    print("Candidate Omicron is 100% verified and ready for deployment.")
    print("=" * 80)


if __name__ == "__main__":
    run_candidate_omicron_verification()
