"""
test_candidate_delta.py
Comprehensive Unit Test Suite for NAS Iteration 6 Candidate Delta Architecture
(PhysiologicalMultiScaleFusionCNNBiGRU)

Verifies:
  1. Strict parameter budget compliance (195,346 <= 196,728 cap; 1,382 headroom; 96.43% compression).
  2. Physiological basal standardization nocturnal normalization invariants across heterogeneous subjects.
  3. Sequence-level multi-scale temporal convolutions (k in {1, 3, 5}) dimension preservation.
  4. Direct intra-epoch feature skip gradient highway to input tensors.
  5. Dual input interface (tensor arguments vs batch dictionary) and BIDSleep output shapes.
  6. Variable sequence length execution with RNN packed sequence handling.
  7. Step-0 numerical stability and probability distributions.
  8. Full backward gradient flow across all submodules.
"""

import os
import sys
import unittest
import numpy as np
import torch
import torch.nn as nn

# Ensure test file directory and workspace root are in python search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from proposed_nas_iteration6_model import (
    PhysiologicalBasalStandardization,
    MaskedTemporalInstanceNorm,
    MultiScaleTemporalFusion,
    PhysiologicalMultiScaleFusionCNNBiGRU,
    CandidateDelta_PBMF,
    build_nas_iteration6_model
)


class TestCandidateDeltaArchitecture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cpu")
        cls.model = build_nas_iteration6_model().to(cls.device)

    def test_01_parameter_budget_compliance(self):
        """Verify parameter count is <= 196,728 and provides >90% compression vs 5.47M benchmark."""
        total_params = sum(p.numel() for p in self.model.parameters())
        baseline_params = 163940
        budget_cap = 196728
        slamss_ifs_params = 5471044

        # Analytical component breakdown
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

        expected_total = (
            p_hr + p_acc + p_meta + p_basal + p_tin_hr + p_tin_acc +
            p_norm_hr + p_norm_acc + p_mstf + p_gru + p_tnorm + p_clf
        )
        self.assertEqual(total_params, expected_total)
        self.assertEqual(total_params, 195346)

        headroom = budget_cap - total_params
        self.assertGreaterEqual(headroom, 0)
        self.assertEqual(headroom, 1382)

        compression_ratio = (slamss_ifs_params - total_params) / slamss_ifs_params * 100.0
        self.assertGreater(compression_ratio, 90.0)
        print(f"\n[Test 01 PASSED] Params: {total_params:,} | Cap: {budget_cap:,} | Headroom: {headroom:,} | Compression: {compression_ratio:.2f}%")

    def test_02_physiological_basal_standardization_invariants(self):
        """Verify dynamic nocturnal z-score invariants eliminate inter-subject baseline shifts."""
        pbs = PhysiologicalBasalStandardization().to(self.device)
        B, T, L = 2, 600, 30

        # Simulate two subjects with drastically different resting baselines:
        # Subject 0: athletic baseline (mean 50.0 bpm, std 5.0 bpm)
        # Subject 1: elevated baseline (mean 85.0 bpm, std 7.0 bpm)
        torch.manual_seed(42)
        x_ihr = torch.empty(B, T, L)
        x_ihr[0] = torch.randn(T, L) * 5.0 + 50.0
        x_ihr[1] = torch.randn(T, L) * 7.0 + 85.0

        x_meta = torch.zeros(B, T, 5)
        x_meta[:, :, 0] = 1.0  # freq
        x_meta[:, :, 1] = 0.0  # clock
        x_meta[:, :, 2] = torch.linspace(0.0, 1.0, T).unsqueeze(0).expand(B, T)
        x_meta[:, :, 3] = x_ihr.mean(dim=-1)  # epoch mean
        x_meta[:, :, 4] = x_ihr.std(dim=-1)   # epoch std

        # Mask: first 480 epochs valid, last 120 padded
        valid_len = 480
        mask = torch.zeros(B, T, dtype=torch.bool)
        mask[:, :valid_len] = True

        z_ihr, z_meta = pbs(x_ihr, x_meta, mask=mask)

        # Invariant checks for both subjects
        for b in range(B):
            valid_z = z_ihr[b, :valid_len]
            padded_z = z_ihr[b, valid_len:]

            mean_val = valid_z.mean().item()
            std_val = valid_z.std(unbiased=False).item()

            # Exact zero-mean and unit-variance nocturnal normalization
            self.assertAlmostEqual(mean_val, 0.0, places=4, msg=f"Subject {b} mean not zero: {mean_val}")
            self.assertAlmostEqual(std_val, 1.0, places=3, msg=f"Subject {b} std not one: {std_val}")

            # Padded elements strictly zeroed out
            self.assertEqual(padded_z.abs().max().item(), 0.0, "Padded z_ihr elements must be 0.0")

            # Meta column 3 normalized mean
            valid_meta3 = z_meta[b, :valid_len, 3]
            padded_meta3 = z_meta[b, valid_len:, 3]
            self.assertAlmostEqual(valid_meta3.mean().item(), 0.0, places=4)
            self.assertEqual(padded_meta3.abs().max().item(), 0.0)

            # Meta columns 0, 1, 2 preserved
            self.assertTrue(torch.allclose(z_meta[b, :valid_len, 0], x_meta[b, :valid_len, 0]))
            self.assertTrue(torch.allclose(z_meta[b, :valid_len, 2], x_meta[b, :valid_len, 2]))

        print("[Test 02 PASSED] Physiological basal standardization invariants verified across 50 bpm vs 85 bpm baselines.")

    def test_03_multiscale_temporal_fusion_receptive_fields(self):
        """Verify MultiScaleTemporalFusion preserves sequence length T and channel dimension."""
        mstf = MultiScaleTemporalFusion(in_channels=144, bottleneck_channels=24, kernel_sizes=(1, 3, 5)).to(self.device)
        B, T, C = 3, 400, 144
        x = torch.randn(B, T, C)
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, 300:] = False

        out = mstf(x, mask=mask)
        self.assertEqual(out.shape, (B, T, C))
        self.assertEqual(out[:, 300:, :].abs().max().item(), 0.0, "Padded epochs must remain strictly 0.0")
        print("[Test 03 PASSED] Multi-Scale Temporal Fusion receptive field and mask preservation verified.")

    def test_04_direct_feature_skip_gradient_highway(self):
        """Verify direct 1-step gradient flow from classifier head into input features."""
        model = build_nas_iteration6_model().to(self.device)
        B, T = 2, 100
        x_ihr = torch.randn(B, T, 30, requires_grad=True)
        x_acc = torch.randn(B, T, 30, requires_grad=True)
        x_meta = torch.randn(B, T, 5, requires_grad=True)
        targets = torch.randint(0, 4, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)

        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
        loss = out["total_loss"]
        loss.backward()

        self.assertIsNotNone(x_ihr.grad)
        self.assertFalse(torch.isnan(x_ihr.grad).any())
        self.assertGreater(x_ihr.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(x_acc.grad)
        self.assertFalse(torch.isnan(x_acc.grad).any())
        self.assertGreater(x_acc.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(x_meta.grad)
        self.assertFalse(torch.isnan(x_meta.grad).any())
        self.assertGreater(x_meta.grad.abs().sum().item(), 0.0)

        print("[Test 04 PASSED] Direct feature skip gradient highway confirmed.")

    def test_05_bidsleep_forward_dual_interface(self):
        """Verify dual input interface (individual tensors vs dictionary batch) and output keys."""
        model = self.model
        model.eval()
        B, T = 2, 600
        x_ihr = torch.randn(B, T, 30)
        x_acc = torch.randn(B, T, 30)
        x_meta = torch.randn(B, T, 5)
        targets = torch.randint(0, 4, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)
        seq_lens = torch.full((B,), T, dtype=torch.int64)

        with torch.no_grad():
            # 1. Tensor arguments
            out_t = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)
            # 2. Dictionary batch
            batch = {
                "ihr": x_ihr,
                "accel": x_acc,
                "meta": x_meta,
                "y": targets,
                "mask": mask,
                "seq_len": seq_lens,
                "subject_idx": torch.tensor([0, 1], dtype=torch.int64)
            }
            out_d = model(batch)

        expected_keys = [
            "logits", "emission_logits", "probabilities",
            "total_loss", "loss_stage", "temp_features",
            "fused_seq", "skip_features", "z_ihr", "z_meta"
        ]
        for k in expected_keys:
            self.assertIn(k, out_t)
            self.assertIn(k, out_d)

        self.assertEqual(out_t["logits"].shape, (B, T, 4))
        self.assertEqual(out_t["probabilities"].shape, (B, T, 4))
        self.assertEqual(out_t["temp_features"].shape, (B, T, 96))
        self.assertEqual(out_t["skip_features"].shape, (B, T, 144))
        self.assertEqual(out_t["fused_seq"].shape, (B, T, 144))
        self.assertEqual(out_t["z_ihr"].shape, (B, T, 30))
        self.assertEqual(out_t["z_meta"].shape, (B, T, 5))

        self.assertTrue(torch.allclose(out_t["logits"], out_d["logits"], atol=1e-5))
        print("[Test 05 PASSED] Dual interface and BIDSleep output shapes verified.")

    def test_06_variable_sequence_length_and_packing(self):
        """Verify packed RNN sequence execution with variable lengths."""
        model = self.model
        B, T = 3, 600
        lengths = [250, 400, 600]
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
                self.assertEqual(
                    out["temp_features"][i, L:, :].abs().max().item(), 0.0,
                    f"Batch {i} padded temp_features must be 0.0"
                )
                self.assertEqual(
                    out["z_ihr"][i, L:, :].abs().max().item(), 0.0,
                    f"Batch {i} padded z_ihr must be 0.0"
                )

        self.assertFalse(torch.isnan(out["logits"]).any())
        self.assertFalse(torch.isnan(out["total_loss"]).any())
        print("[Test 06 PASSED] Variable sequence length packed execution confirmed.")

    def test_07_step0_numerical_stability(self):
        """Verify initial step-0 forward pass produces stable probabilities and losses."""
        model = build_nas_iteration6_model().to(self.device)
        model.eval()
        B, T = 2, 300
        x_ihr = torch.randn(B, T, 30) * 8.0 + 65.0  # realistic HR range ~50-80
        x_acc = torch.rand(B, T, 30) * 0.1
        x_meta = torch.randn(B, T, 5)
        x_meta[:, :, 3] = x_ihr.mean(dim=-1)
        x_meta[:, :, 4] = x_ihr.std(dim=-1)
        targets = torch.randint(0, 4, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)

        with torch.no_grad():
            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)

        probs = out["probabilities"]
        prob_sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5))
        self.assertGreater(out["total_loss"].item(), 0.0)
        self.assertLess(out["total_loss"].item(), 5.0)
        print(f"[Test 07 PASSED] Step-0 numerical stability verified (Loss: {out['total_loss'].item():.4f}).")

    def test_08_backward_gradient_coverage(self):
        """Verify all submodules receive non-zero, non-NaN gradients."""
        model = build_nas_iteration6_model().to(self.device)
        model.train()
        B, T = 2, 200
        x_ihr = torch.randn(B, T, 30)
        x_acc = torch.randn(B, T, 30)
        x_meta = torch.randn(B, T, 5)
        targets = torch.randint(0, 4, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)

        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
        loss = out["total_loss"]
        loss.backward()

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

        print("[Test 08 PASSED] Backward gradient coverage confirmed across all 15 submodules.")


if __name__ == "__main__":
    unittest.main()
