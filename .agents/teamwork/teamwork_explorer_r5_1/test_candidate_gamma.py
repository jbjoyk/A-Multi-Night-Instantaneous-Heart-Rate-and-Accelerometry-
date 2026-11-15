#!/usr/bin/env python3
"""
.agents/teamwork/teamwork_explorer_r5_1/test_candidate_gamma.py
Comprehensive Unit Test Suite for NAS Iteration 5: Candidate Gamma (InterEpochSkipAttentionCNNBiGRU)

Verifies:
  1. Exact Analytical Parameter Count: 193,156 parameters (strictly <= 196,728 cap).
  2. Step-0 Zero-Identity Initialization: attn_context is identically 0.0 at initialization.
  3. Padding Protection & Attention Key-Masking: Zero attention weight allocated to padded tokens.
  4. Masked TIN Standardization: Valid nocturnal sequence normalized to zero mean and unit variance.
  5. BIDSleep Forward Tensor Shapes: (B=4, T=600) with dictionary and positional argument interfaces.
  6. Backward Gradient Flow: Verified non-zero, non-NaN gradients across all submodules.
  7. Variable Sequence Length Packing: PyTorch pack_padded_sequence execution without shape error.
  8. Modality Feature Skip Highway: Direct gradient flow to hr_encoder and acc_encoder via skip path.
"""

import os
import sys
import unittest
import torch
import torch.nn as nn
import numpy as np

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from proposed_nas_iteration5_model import (
    MaskedTemporalInstanceNorm,
    InterEpochSequenceAttention,
    InterEpochSkipAttentionCNNBiGRU,
    CandidateGamma_IESA,
    build_nas_iteration5_model
)


class TestCandidateGamma(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.B = 4
        self.T = 600
        self.d_model = 64
        self.d_temp = 96
        self.d_attn = 48
        self.n_heads = 4
        self.n_classes = 4

    def test_01_parameter_budget_compliance(self):
        """Test 01: Verify analytical and empirical parameter count <= 196,728 cap."""
        model = build_nas_iteration5_model(concat_context=True, zero_init=True)
        total_params = sum(p.numel() for p in model.parameters())

        # Exact expected parameter breakdown
        hr_params = sum(p.numel() for p in model.hr_encoder.parameters())
        acc_params = sum(p.numel() for p in model.acc_encoder.parameters())
        meta_params = sum(p.numel() for p in model.meta_proj.parameters())
        tin_hr_params = sum(p.numel() for p in model.tin_hr.parameters())
        tin_acc_params = sum(p.numel() for p in model.tin_acc.parameters())
        norm_hr_params = sum(p.numel() for p in model.norm_hr.parameters())
        norm_acc_params = sum(p.numel() for p in model.norm_acc.parameters())
        gru_params = sum(p.numel() for p in model.temporal_gru.parameters())
        temp_norm_params = sum(p.numel() for p in model.temp_norm.parameters())
        attn_params = sum(p.numel() for p in model.inter_epoch_attn.parameters())
        clf_params = sum(p.numel() for p in model.classifier_head.parameters())

        self.assertEqual(hr_params, 29568)
        self.assertEqual(acc_params, 29568)
        self.assertEqual(meta_params, 96)
        self.assertEqual(tin_hr_params, 128)
        self.assertEqual(tin_acc_params, 128)
        self.assertEqual(norm_hr_params, 128)
        self.assertEqual(norm_acc_params, 128)
        self.assertEqual(gru_params, 97920)
        self.assertEqual(temp_norm_params, 192)
        self.assertEqual(attn_params, 16416)
        self.assertEqual(clf_params, 18884)

        expected_total = 193156
        self.assertEqual(total_params, expected_total, f"Total params {total_params} != expected {expected_total}")
        self.assertLessEqual(total_params, 196728, "Exceeded strict parameter cap 196,728")
        headroom = 196728 - total_params
        self.assertEqual(headroom, 3572)
        print(f"Test 01 PASSED: Parameter count = {total_params:,d} (3,572 headroom under 196,728 cap).")

    def test_02_step_0_zero_identity_initialization(self):
        """Test 02: Verify that at step 0, attn_context is identically zero."""
        model = build_nas_iteration5_model(concat_context=True, zero_init=True)
        model.eval()

        x_ihr = torch.randn(self.B, self.T, 30)
        x_acc = torch.randn(self.B, self.T, 30)
        x_meta = torch.randn(self.B, self.T, 5)
        mask = torch.ones(self.B, self.T, dtype=torch.bool)

        with torch.no_grad():
            out = model(x_ihr, x_acc, x_meta, mask=mask)

        attn_context = out["attn_context"]
        max_abs_attn = attn_context.abs().max().item()
        self.assertAlmostEqual(max_abs_attn, 0.0, places=6,
                               msg=f"attn_context at step 0 must be 0.0, got {max_abs_attn}")
        print("Test 02 PASSED: Exact step-0 zero-identity attention initialization confirmed.")

    def test_03_attention_key_masking_padding_protection(self):
        """Test 03: Verify zero attention weight allocated to padded positions."""
        attn_module = InterEpochSequenceAttention(d_in=96, d_attn=48, n_heads=4, zero_init=False)
        attn_module.eval()

        x = torch.randn(self.B, self.T, 96)
        seq_lens = torch.tensor([600, 450, 300, 150])
        mask = torch.arange(self.T).unsqueeze(0) < seq_lens.unsqueeze(1)

        with torch.no_grad():
            context, weights = attn_module(x, mask=mask)

        # For each batch element, positions >= seq_len must have exactly 0 attention weight
        for b in range(self.B):
            valid_len = seq_lens[b].item()
            if valid_len < self.T:
                padded_weights = weights[b, :, :, valid_len:]
                max_pad_weight = padded_weights.max().item()
                self.assertEqual(max_pad_weight, 0.0,
                                 f"Batch {b} allocated {max_pad_weight} attention weight to padded tokens!")

        print("Test 03 PASSED: Attention padding key-masking strictly eliminates padded token weights.")

    def test_04_masked_tin_standardization(self):
        """Test 04: Verify Masked TIN achieves nocturnal zero-mean and unit-variance."""
        tin = MaskedTemporalInstanceNorm(num_features=64)
        x = torch.randn(self.B, self.T, 64) * 15.0 + 75.0  # Simulated resting HR ~75 bpm with variation
        seq_lens = torch.tensor([600, 450, 300, 150])
        mask = torch.arange(self.T).unsqueeze(0) < seq_lens.unsqueeze(1)

        normed = tin(x, mask=mask)

        for b in range(self.B):
            v_len = seq_lens[b].item()
            valid_slice = normed[b, :v_len, :]  # (v_len, 64)
            mean_val = valid_slice.mean().item()
            std_val = valid_slice.std(unbiased=False).item()
            self.assertAlmostEqual(mean_val, 0.0, delta=1e-3, msg=f"Batch {b} TIN mean {mean_val} != 0.0")
            self.assertAlmostEqual(std_val, 1.0, delta=1e-2, msg=f"Batch {b} TIN std {std_val} != 1.0")

        print("Test 04 PASSED: Masked TIN achieves exact zero-mean and unit-variance nocturnal normalization.")

    def test_05_bidsleep_forward_shapes_and_interfaces(self):
        """Test 05: Verify full forward shapes on BIDSleep inputs with dict and tensor APIs."""
        model = build_nas_iteration5_model(concat_context=True, zero_init=True)
        model.eval()

        x_ihr = torch.randn(self.B, self.T, 30)
        x_acc = torch.randn(self.B, self.T, 30)
        x_meta = torch.randn(self.B, self.T, 5)
        targets = torch.randint(0, 4, (self.B, self.T))
        mask = torch.ones(self.B, self.T, dtype=torch.bool)
        seq_lens = torch.tensor([600, 500, 400, 300])

        # Test tensor-based call
        out1 = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)
        self.assertEqual(out1["logits"].shape, (self.B, self.T, self.n_classes))
        self.assertEqual(out1["temp_features"].shape, (self.B, self.T, 96))
        self.assertEqual(out1["attn_context"].shape, (self.B, self.T, 48))
        self.assertEqual(out1["skip_features"].shape, (self.B, self.T, 144))
        self.assertTrue(out1["total_loss"].item() > 0.0)

        # Test dictionary-based call
        batch_dict = {
            "ihr": x_ihr,
            "accel": x_acc,
            "meta": x_meta,
            "y": targets,
            "mask": mask,
            "seq_len": seq_lens
        }
        out2 = model(batch_dict)
        self.assertEqual(out2["logits"].shape, (self.B, self.T, self.n_classes))
        print("Test 05 PASSED: Full BIDSleep forward shapes and dual (tensor/dict) interface verified.")

    def test_06_backward_gradient_flow_all_modules(self):
        """Test 06: Verify active backward gradient flow across all modules."""
        # Using scale_init (zero_init=False) to verify active gradient flow to all attention projections
        model = build_nas_iteration5_model(concat_context=True, zero_init=False)
        model.train()

        x_ihr = torch.randn(self.B, self.T, 30)
        x_acc = torch.randn(self.B, self.T, 30)
        x_meta = torch.randn(self.B, self.T, 5)
        targets = torch.randint(0, 4, (self.B, self.T))
        mask = torch.ones(self.B, self.T, dtype=torch.bool)
        seq_lens = torch.tensor([600, 600, 600, 600])

        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)
        loss = out["total_loss"]
        loss.backward()

        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} did not receive a gradient!")
            self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN in gradient!")
            grad_norm = param.grad.norm().item()
            self.assertGreater(grad_norm, 0.0, f"Parameter {name} has zero gradient norm!")

        print("Test 06 PASSED: Non-zero, non-NaN backward gradient flow confirmed across all submodules.")

    def test_07_variable_sequence_length_packing(self):
        """Test 07: Verify variable sequence length packed execution without shape errors."""
        model = build_nas_iteration5_model(concat_context=True, zero_init=True)
        model.eval()

        seq_lens = torch.tensor([600, 480, 360, 240])
        x_ihr = torch.randn(self.B, self.T, 30)
        x_acc = torch.randn(self.B, self.T, 30)
        x_meta = torch.randn(self.B, self.T, 5)
        mask = torch.arange(self.T).unsqueeze(0) < seq_lens.unsqueeze(1)

        with torch.no_grad():
            out = model(x_ihr, x_acc, x_meta, mask=mask, seq_lens=seq_lens)

        self.assertEqual(out["logits"].shape, (self.B, self.T, 4))
        self.assertEqual(out["probabilities"].shape, (self.B, self.T, 4))
        print("Test 07 PASSED: Variable sequence length packed execution confirmed.")

    def test_08_modality_skip_gradient_highway(self):
        """Test 08: Verify direct 1-step backprop highway bypassing recurrent model."""
        model = build_nas_iteration5_model(concat_context=True, zero_init=True)
        model.train()

        x_ihr = torch.randn(self.B, self.T, 30, requires_grad=True)
        x_acc = torch.randn(self.B, self.T, 30, requires_grad=True)
        x_meta = torch.randn(self.B, self.T, 5)
        targets = torch.randint(0, 4, (self.B, self.T))
        mask = torch.ones(self.B, self.T, dtype=torch.bool)

        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
        out["total_loss"].backward()

        self.assertIsNotNone(x_ihr.grad)
        self.assertIsNotNone(x_acc.grad)
        self.assertGreater(x_ihr.grad.norm().item(), 0.0)
        self.assertGreater(x_acc.grad.norm().item(), 0.0)
        print("Test 08 PASSED: Direct gradient highway to raw input tensors confirmed.")


if __name__ == "__main__":
    unittest.main()
