"""
Unit Test Suite for NAS Iteration 4 Candidate Beta:
CircadianCrossAttentionCNNBiGRU & CircadianAnchoredCrossAttention (CACA)
Path: .agents/teamwork/teamwork_worker_r4_1/test_nas_iteration4.py

Verifies:
  1. Exact parameter count assertion (186,870 parameters, strictly within budget cap).
  2. Step-0 zero identity initialization (W_o == 0, b_o == 0, delta == 0).
  3. Cosine similarity logits strictly bounded in [-1, 1].
  4. Masked Temporal Instance Normalization: zero nocturnal mean and unit variance.
  5. Tensor forward pass on expected BIDSleep shapes (B=4, T=600, 30 for IHR/ACC, 5 for meta).
  6. Backward pass and gradient flow through all CACA components:
     attn_hr_from_acc, attn_acc_from_hr, tin_hr, ms_hr, ms_acc, circadian_proj.
  7. Variable sequence length packing with dictionary batch inputs.
"""

import sys
import os
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

EXP014_DIR = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014")
if EXP014_DIR not in sys.path:
    sys.path.insert(0, EXP014_DIR)

from nas_iteration4_model import (
    MaskedTemporalInstanceNorm,
    MultiScaleModalityBlock,
    LocalizedCosineCrossAttention,
    CircadianAnchoredCrossAttention,
    CircadianCrossAttentionCNNBiGRU,
    build_nas_iteration4_model
)


class TestNASIteration4(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        cls.model = build_nas_iteration4_model().to(cls.device)

    def test_01_parameter_counts(self):
        """Assert exact submodule and total parameter counts."""
        m = self.model
        p_hr = sum(p.numel() for p in m.hr_encoder.parameters())
        p_acc = sum(p.numel() for p in m.acc_encoder.parameters())
        p_meta = sum(p.numel() for p in m.meta_proj.parameters())
        p_caca = sum(p.numel() for p in m.cross_attention.parameters())
        p_gru = sum(p.numel() for p in m.temporal_gru.parameters())
        p_norm = sum(p.numel() for p in m.temp_norm.parameters())
        p_head = sum(p.numel() for p in m.classifier_head.parameters())
        total = sum(p.numel() for p in m.parameters())

        self.assertEqual(p_hr, 29568, f"Expected 29,568 hr_encoder, got {p_hr}")
        self.assertEqual(p_acc, 29568, f"Expected 29,568 acc_encoder, got {p_acc}")
        self.assertEqual(p_meta, 96, f"Expected 96 meta_proj, got {p_meta}")
        self.assertEqual(p_caca, 22930, f"Expected 22,930 cross_attention, got {p_caca}")
        self.assertEqual(p_gru, 97920, f"Expected 97,920 temporal_gru, got {p_gru}")
        self.assertEqual(p_norm, 192, f"Expected 192 temp_norm, got {p_norm}")
        self.assertEqual(p_head, 6596, f"Expected 6,596 classifier_head, got {p_head}")
        self.assertEqual(total, 186870, f"Expected 186,870 total, got {total}")
        self.assertLessEqual(total, 196728, "Total parameters must not exceed 196,728 cap")
        print(f"Test 01 PASSED: Parameter count = {total:,} (within budget cap 196,728)")

    def test_02_step0_identity_initialization(self):
        """Verify W_o == 0, b_o == 0 and output delta == 0 at step 0."""
        caca = self.model.cross_attention
        self.assertTrue(torch.all(caca.attn_hr_from_acc.out_proj.weight == 0))
        self.assertTrue(torch.all(caca.attn_hr_from_acc.out_proj.bias == 0))
        self.assertTrue(torch.all(caca.attn_acc_from_hr.out_proj.weight == 0))
        self.assertTrue(torch.all(caca.attn_acc_from_hr.out_proj.bias == 0))

        # Check that output delta is strictly zero
        q = torch.randn(2, 50, 96, device=self.device)
        k = torch.randn(2, 50, 96, device=self.device)
        v = torch.randn(2, 50, 64, device=self.device)
        delta_hr = caca.attn_hr_from_acc(q, k, v)
        delta_acc = caca.attn_acc_from_hr(q, k, v)
        self.assertAlmostEqual(delta_hr.abs().max().item(), 0.0, places=6)
        self.assertAlmostEqual(delta_acc.abs().max().item(), 0.0, places=6)
        print("Test 02 PASSED: Exact step-0 zero identity initialization verified.")

    def test_03_cosine_similarity_bounds(self):
        """Verify cosine similarity logits are strictly bounded in [-1, 1]."""
        attn = self.model.cross_attention.attn_hr_from_acc
        q = torch.randn(2, 100, 96, device=self.device) * 50.0  # huge scale
        k = torch.randn(2, 100, 96, device=self.device) * 0.01  # tiny scale
        v = torch.randn(2, 100, 64, device=self.device)
        _, sim = attn(q, k, v, return_sim=True)
        self.assertLessEqual(sim.max().item(), 1.0 + 1e-5)
        self.assertGreaterEqual(sim.min().item(), -1.0 - 1e-5)
        print("Test 03 PASSED: Cosine similarity strictly bounded in [-1.0, 1.0].")

    def test_04_masked_tin_normalization(self):
        """Verify nocturnal zero mean and unit variance per subject across valid sequence length."""
        tin = self.model.cross_attention.tin_hr
        x = torch.randn(3, 200, 64, device=self.device) * 12.0 + 45.0  # arbitrary baseline shift
        mask = torch.zeros(3, 200, dtype=torch.bool, device=self.device)
        mask[0, :150] = True
        mask[1, :120] = True
        mask[2, :180] = True
        out = tin(x, mask)
        for b in range(3):
            valid = out[b, mask[b]]
            mean = valid.mean().item()
            std = valid.std(unbiased=False).item()
            self.assertAlmostEqual(mean, 0.0, places=3)
            self.assertAlmostEqual(std, 1.0, places=2)
        print("Test 04 PASSED: Masked TIN nocturnal zero-mean and unit-variance confirmed.")

    def test_05_forward_bidsleep_shapes(self):
        """Verify forward pass on expected BIDSleep shapes (B=4, T=600, 30 for IHR/ACC, 5 for meta)."""
        self.model.eval()
        B, T = 4, 600
        x_ihr = torch.randn(B, T, 30, device=self.device)
        x_acc = torch.randn(B, T, 30, device=self.device)
        x_meta = torch.randn(B, T, 5, device=self.device)
        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)
        targets = torch.randint(0, 4, (B, T), device=self.device)

        with torch.no_grad():
            out = self.model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)

        self.assertEqual(out['logits'].shape, (B, T, 4))
        self.assertEqual(out['probabilities'].shape, (B, T, 4))
        self.assertEqual(out['temp_features'].shape, (B, T, 96))
        self.assertTrue(torch.isfinite(out['total_loss']))
        print("Test 05 PASSED: BIDSleep forward shapes (B=4, T=600) verified.")

    def test_06_backward_gradient_flow_all_caca_components(self):
        """Verify backward pass and gradient flow through all CACA submodules."""
        self.model.train()
        x_ihr = torch.randn(2, 100, 30, device=self.device)
        x_acc = torch.randn(2, 100, 30, device=self.device)
        x_meta = torch.randn(2, 100, 5, device=self.device)
        targets = torch.randint(0, 4, (2, 100), device=self.device)
        mask = torch.ones(2, 100, dtype=torch.bool, device=self.device)

        self.model.zero_grad()
        out = self.model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
        loss = out['total_loss']
        loss.backward()

        # Check gradients across all CACA submodules
        caca = self.model.cross_attention
        # attn_hr_from_acc
        self.assertIsNotNone(caca.attn_hr_from_acc.q_proj.weight.grad)
        self.assertIsNotNone(caca.attn_hr_from_acc.k_proj.weight.grad)
        self.assertIsNotNone(caca.attn_hr_from_acc.v_proj.weight.grad)
        self.assertIsNotNone(caca.attn_hr_from_acc.out_proj.weight.grad)
        self.assertIsNotNone(caca.attn_hr_from_acc.scale.grad)
        # attn_acc_from_hr
        self.assertIsNotNone(caca.attn_acc_from_hr.q_proj.weight.grad)
        self.assertIsNotNone(caca.attn_acc_from_hr.k_proj.weight.grad)
        self.assertIsNotNone(caca.attn_acc_from_hr.v_proj.weight.grad)
        self.assertIsNotNone(caca.attn_acc_from_hr.out_proj.weight.grad)
        self.assertIsNotNone(caca.attn_acc_from_hr.scale.grad)
        # tin_hr & tin_acc
        self.assertIsNotNone(caca.tin_hr.weight.grad)
        self.assertIsNotNone(caca.tin_hr.bias.grad)
        self.assertIsNotNone(caca.tin_acc.weight.grad)
        self.assertIsNotNone(caca.tin_acc.bias.grad)
        # ms_hr & ms_acc
        self.assertIsNotNone(caca.ms_hr.proj_in.weight.grad)
        self.assertIsNotNone(caca.ms_hr.proj_out.weight.grad)
        self.assertIsNotNone(caca.ms_acc.proj_in.weight.grad)
        self.assertIsNotNone(caca.ms_acc.proj_out.weight.grad)
        # circadian_proj
        self.assertIsNotNone(caca.circadian_proj[0].weight.grad)
        self.assertIsNotNone(caca.circadian_proj[0].bias.grad)
        print("Test 06 PASSED: Backward gradient flow across all CACA components confirmed.")

    def test_07_variable_length_packing(self):
        """Verify variable sequence length packing and dict batch input."""
        self.model.eval()
        x_ihr = torch.randn(3, 400, 30, device=self.device)
        x_acc = torch.randn(3, 400, 30, device=self.device)
        x_meta = torch.randn(3, 400, 5, device=self.device)
        seq_lens = torch.tensor([400, 250, 150], dtype=torch.int64)
        mask = torch.zeros(3, 400, dtype=torch.bool, device=self.device)
        for i, sl in enumerate(seq_lens):
            mask[i, :sl] = True

        batch = {
            'ihr': x_ihr,
            'accel': x_acc,
            'meta': x_meta,
            'mask': mask,
            'seq_len': seq_lens
        }
        with torch.no_grad():
            out = self.model(batch)
        self.assertEqual(out['logits'].shape, (3, 400, 4))
        print("Test 07 PASSED: Variable sequence length packed execution confirmed.")


if __name__ == '__main__':
    unittest.main()
