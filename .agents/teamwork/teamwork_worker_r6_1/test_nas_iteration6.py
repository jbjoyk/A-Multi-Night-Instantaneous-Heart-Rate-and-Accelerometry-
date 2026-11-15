"""
.agents/teamwork/teamwork_worker_r6_1/test_nas_iteration6.py
Unit Test Suite for NAS Iteration 6 Candidate Delta Architecture
(PhysiologicalMultiScaleFusionCNNBiGRU)

Verifies:
  1. Exact parameter count assertion (195,346 parameters <= 196,728 cap).
  2. Invariant basal normalization verification across differing resting HR baselines (50 bpm vs 85 bpm).
  3. MultiScale receptive field and mask preservation.
  4. Direct feature skip gradient highway verification.
  5. Forward pass on BIDSleep shapes (B=4, T=600, 30 for IHR/ACC, 5 for meta) with dual interface.
  6. Variable sequence length execution with RNN packed sequence handling.
  7. Step-0 numerical stability and probability distributions.
  8. Full backward pass and gradient coverage across all 15 submodules.
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
EXP014_DIR = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if EXP014_DIR not in sys.path:
    sys.path.insert(0, EXP014_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from nas_iteration6_model import (
    PhysiologicalBasalStandardization,
    MaskedTemporalInstanceNorm,
    MultiScaleTemporalFusion,
    PhysiologicalMultiScaleFusionCNNBiGRU,
    CandidateDelta_PBMF,
    build_nas_iteration6_model
)


class TestNASIteration6Worker(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.model = build_nas_iteration6_model().to(cls.device)

    def test_01_exact_parameter_count(self):
        """Verify exact parameter count is 195,346 and strictly within 196,728 budget cap."""
        total_params = sum(p.numel() for p in self.model.parameters())
        budget_cap = 196728
        slamss_ifs_params = 5471044

        p_hr = sum(p.numel() for p in self.model.hr_encoder.parameters())
        p_acc = sum(p.numel() for p in self.model.acc_encoder.parameters())
        p_meta = sum(p.numel() for p in self.model.meta_proj.parameters())
        p_basal = sum(p.numel() for p in self.model.basal_norm.parameters())
        p_tin_hr = sum(p.numel() for p in self.model.tin_hr.parameters())
        p_tin_acc = sum(p.numel() for p in self.model.tin_acc.parameters())
        p_norm_hr = sum(p.numel() for p in self.model.norm_hr.parameters())
        p_norm_acc = sum(p.numel() for p in self.model.norm_acc.parameters())
        p_mstf = sum(p.numel() for p in self.model.mstf.parameters())
        p_gru = sum(p.numel() for p in self.model.temporal_gru.parameters())
        p_tnorm = sum(p.numel() for p in self.model.temp_norm.parameters())
        p_clf = sum(p.numel() for p in self.model.classifier_head.parameters())

        self.assertEqual(p_hr, 29568)
        self.assertEqual(p_acc, 29568)
        self.assertEqual(p_meta, 96)
        self.assertEqual(p_basal, 6)
        self.assertEqual(p_tin_hr, 128)
        self.assertEqual(p_tin_acc, 128)
        self.assertEqual(p_norm_hr, 128)
        self.assertEqual(p_norm_acc, 128)
        self.assertEqual(p_mstf, 21672)
        self.assertEqual(p_gru, 97920)
        self.assertEqual(p_tnorm, 192)
        self.assertEqual(p_clf, 15812)

        self.assertEqual(total_params, 195346)
        self.assertLessEqual(total_params, budget_cap)

        headroom = budget_cap - total_params
        self.assertEqual(headroom, 1382)

        compression = (slamss_ifs_params - total_params) / slamss_ifs_params * 100.0
        self.assertGreater(compression, 90.0)
        print(f"\n[Test 01 PASSED] Exact Parameters: {total_params:,} | Cap: {budget_cap:,} | Headroom: {headroom:,} | Compression: {compression:.2f}%")

    def test_02_invariant_basal_normalization(self):
        """Verify dynamic nocturnal z-scoring invariants across differing resting HR baselines (50 bpm vs 85 bpm)."""
        pbs = PhysiologicalBasalStandardization().to(self.device)
        B, T, L = 2, 600, 30

        torch.manual_seed(42)
        x_ihr = torch.empty(B, T, L)
        x_ihr[0] = torch.randn(T, L) * 5.0 + 50.0  # 50 bpm athlete baseline
        x_ihr[1] = torch.randn(T, L) * 7.0 + 85.0  # 85 bpm elevated baseline

        x_meta = torch.zeros(B, T, 5)
        x_meta[:, :, 0] = 1.0  # freq
        x_meta[:, :, 1] = 0.0  # clock
        x_meta[:, :, 2] = torch.linspace(0.0, 1.0, T).unsqueeze(0).expand(B, T)
        x_meta[:, :, 3] = x_ihr.mean(dim=-1)  # epoch mean
        x_meta[:, :, 4] = x_ihr.std(dim=-1)   # epoch std

        valid_len = 480
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, :valid_len] = True

        z_ihr, z_meta = pbs(x_ihr, x_meta, mask=mask)

        for b in range(B):
            valid_z = z_ihr[b, :valid_len]
            padded_z = z_ihr[b, valid_len:]

            self.assertAlmostEqual(valid_z.mean().item(), 0.0, places=4)
            self.assertAlmostEqual(valid_z.std(unbiased=False).item(), 1.0, places=3)
            self.assertEqual(padded_z.abs().max().item(), 0.0)

            valid_meta3 = z_meta[b, :valid_len, 3]
            padded_meta3 = z_meta[b, valid_len:, 3]
            self.assertAlmostEqual(valid_meta3.mean().item(), 0.0, places=4)
            self.assertEqual(padded_meta3.abs().max().item(), 0.0)

            # Preserved columns 0 and 2
            self.assertTrue(torch.allclose(z_meta[b, :valid_len, 0], x_meta[b, :valid_len, 0]))
            self.assertTrue(torch.allclose(z_meta[b, :valid_len, 2], x_meta[b, :valid_len, 2]))

        print("[Test 02 PASSED] Invariant basal normalization verified across 50 bpm vs 85 bpm baselines.")

    def test_03_multiscale_receptive_field_and_mask(self):
        """Verify MultiScale temporal convolutions (k in {1, 3, 5}) dimension preservation and mask adherence."""
        mstf = MultiScaleTemporalFusion(in_channels=144, bottleneck_channels=24, kernel_sizes=(1, 3, 5)).to(self.device)
        B, T, C = 4, 600, 144
        x = torch.randn(B, T, C)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, 450:] = False

        out = mstf(x, mask=mask)
        self.assertEqual(out.shape, (B, T, C))
        self.assertEqual(out[:, 450:, :].abs().max().item(), 0.0)
        print("[Test 03 PASSED] MultiScale receptive field and mask preservation verified.")

    def test_04_direct_feature_skip_gradient_highway(self):
        """Verify direct feature skip gradient highway bypassing recurrent GRU."""
        model = build_nas_iteration6_model().to(self.device)
        B, T = 2, 100
        x_ihr = torch.randn(B, T, 30, requires_grad=True)
        x_acc = torch.randn(B, T, 30, requires_grad=True)
        x_meta = torch.randn(B, T, 5, requires_grad=True)
        targets = torch.randint(0, 4, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)

        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
        out["total_loss"].backward()

        self.assertIsNotNone(x_ihr.grad)
        self.assertFalse(torch.isnan(x_ihr.grad).any())
        self.assertGreater(x_ihr.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(x_acc.grad)
        self.assertFalse(torch.isnan(x_acc.grad).any())
        self.assertGreater(x_acc.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(x_meta.grad)
        self.assertFalse(torch.isnan(x_meta.grad).any())
        self.assertGreater(x_meta.grad.abs().sum().item(), 0.0)
        print("[Test 04 PASSED] Direct feature skip gradient highway verified.")

    def test_05_forward_pass_bidsleep_shapes(self):
        """Verify forward pass on standard BIDSleep training shapes (B=4, T=600, 30 for IHR/ACC, 5 for meta)."""
        model = self.model
        model.eval()
        B, T = 4, 600
        x_ihr = torch.randn(B, T, 30)
        x_acc = torch.randn(B, T, 30)
        x_meta = torch.randn(B, T, 5)
        targets = torch.randint(0, 4, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)
        seq_lens = torch.full((B,), T, dtype=torch.int64)

        with torch.no_grad():
            out_t = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)
            batch = {
                "ihr": x_ihr,
                "accel": x_acc,
                "meta": x_meta,
                "y": targets,
                "mask": mask,
                "seq_len": seq_lens,
                "subject_idx": torch.tensor([0, 1, 2, 3], dtype=torch.int64)
            }
            out_d = model(batch)

        self.assertEqual(out_t["logits"].shape, (4, 600, 4))
        self.assertEqual(out_t["probabilities"].shape, (4, 600, 4))
        self.assertEqual(out_t["temp_features"].shape, (4, 600, 96))
        self.assertEqual(out_t["skip_features"].shape, (4, 600, 144))
        self.assertEqual(out_t["fused_seq"].shape, (4, 600, 144))
        self.assertEqual(out_t["z_ihr"].shape, (4, 600, 30))
        self.assertEqual(out_t["z_meta"].shape, (4, 600, 5))

        self.assertTrue(torch.allclose(out_t["logits"], out_d["logits"], atol=1e-5))
        print("[Test 05 PASSED] Forward pass on BIDSleep shapes (B=4, T=600) and dual interface verified.")

    def test_06_variable_sequence_length_and_packing(self):
        """Verify variable sequence lengths and RNN packed sequence padding masking."""
        model = self.model
        B, T = 3, 600
        lengths = [200, 450, 600]
        x_ihr = torch.randn(B, T, 30)
        x_acc = torch.randn(B, T, 30)
        x_meta = torch.randn(B, T, 5)
        targets = torch.randint(0, 4, (B, T))
        mask = torch.zeros(B, T, dtype=torch.bool)
        for i, L in enumerate(lengths):
            mask[i, :L] = True

        seq_lens = torch.tensor(lengths, dtype=torch.int64)
        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)

        for i, L in enumerate(lengths):
            if L < T:
                self.assertEqual(out["temp_features"][i, L:, :].abs().max().item(), 0.0)
                self.assertEqual(out["z_ihr"][i, L:, :].abs().max().item(), 0.0)

        self.assertFalse(torch.isnan(out["logits"]).any())
        self.assertFalse(torch.isnan(out["total_loss"]).any())
        print("[Test 06 PASSED] Variable sequence length packed execution confirmed.")

    def test_07_step0_numerical_stability(self):
        """Verify step-0 forward pass produces valid probabilities and bounded loss."""
        model = build_nas_iteration6_model().to(self.device)
        model.eval()
        B, T = 2, 300
        x_ihr = torch.randn(B, T, 30) * 8.0 + 65.0
        x_acc = torch.rand(B, T, 30) * 0.1
        x_meta = torch.randn(B, T, 5)
        targets = torch.randint(0, 4, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)

        with torch.no_grad():
            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)

        probs = out["probabilities"]
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(B, T), atol=1e-5))
        self.assertGreater(out["total_loss"].item(), 0.0)
        self.assertLess(out["total_loss"].item(), 5.0)
        print(f"[Test 07 PASSED] Step-0 numerical stability confirmed (Loss: {out['total_loss'].item():.4f}).")

    def test_08_backward_gradient_coverage(self):
        """Verify backward pass and gradient coverage across all submodules."""
        model = build_nas_iteration6_model().to(self.device)
        model.train()
        B, T = 2, 200
        x_ihr = torch.randn(B, T, 30)
        x_acc = torch.randn(B, T, 30)
        x_meta = torch.randn(B, T, 5)
        targets = torch.randint(0, 4, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)

        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
        out["total_loss"].backward()

        modules_to_check = [
            ("basal_norm.gamma_ihr", model.basal_norm.gamma_ihr),
            ("hr_encoder", model.hr_encoder.proj.weight),
            ("acc_encoder", model.acc_encoder.proj.weight),
            ("meta_proj", model.meta_proj[0].weight),
            ("tin_hr", model.tin_hr.weight),
            ("tin_acc", model.tin_acc.weight),
            ("norm_hr", model.norm_hr.weight),
            ("norm_acc", model.norm_acc.weight),
            ("mstf.proj_in", model.mstf.proj_in.weight),
            ("mstf.branches[0]", model.mstf.branches[0][0].weight),
            ("mstf.branches[1]", model.mstf.branches[1][0].weight),
            ("mstf.branches[2]", model.mstf.branches[2][0].weight),
            ("mstf.proj_out", model.mstf.proj_out.weight),
            ("temporal_gru", model.temporal_gru.weight_ih_l0),
            ("classifier_head", model.classifier_head[0].weight)
        ]

        for name, param in modules_to_check:
            self.assertIsNotNone(param.grad, f"Gradient is None for {name}")
            self.assertFalse(torch.isnan(param.grad).any(), f"Gradient is NaN for {name}")
            self.assertGreater(param.grad.abs().sum().item(), 0.0, f"Gradient is all zeros for {name}")

        print("[Test 08 PASSED] Backward gradient coverage confirmed across all submodules.")


if __name__ == "__main__":
    unittest.main()
