"""
Unit Test Suite for NAS Iteration 3 Architecture: Candidate Alpha (CircadianCrossModalCNNBiGRU)
File: .agents/teamwork/teamwork_worker_r3_1/test_nas_iteration3.py

Tests:
1. Exact parameter count assertions (190,502 total; sub-module counts verified).
2. Zero-initialization & identity initial state verification.
3. Masked Temporal Instance Normalization (TIN) sequence standardization & masking behavior.
4. Forward pass tensor shapes for BIDSleep specifications (B=4, T=600, 30, 5).
5. Backward pass and gradient flow across all CACMF submodules.
6. Execution on GPU device (cuda:0).
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

EXP014_DIR = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014")
if EXP014_DIR not in sys.path:
    sys.path.insert(0, EXP014_DIR)

from nas_iteration3_model import (
    MaskedTemporalInstanceNorm,
    MultiScaleModalityBlock,
    CircadianAnchoredCrossModalFusion,
    CircadianCrossModalCNNBiGRU,
    build_nas_iteration3_model
)


class TestNASIteration3Architecture(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_01_parameter_counts(self):
        """Verify exact parameter counts for full model and all sub-modules."""
        model = build_nas_iteration3_model()
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n[Test 1] Total Model Parameters: {total_params:,}")

        # Sub-module parameter verification
        hr_enc_params = sum(p.numel() for p in model.hr_encoder.parameters())
        acc_enc_params = sum(p.numel() for p in model.acc_encoder.parameters())
        meta_proj_params = sum(p.numel() for p in model.meta_proj.parameters())
        fusion_params = sum(p.numel() for p in model.cross_fusion.parameters())
        gru_params = sum(p.numel() for p in model.temporal_gru.parameters())
        norm_params = sum(p.numel() for p in model.temp_norm.parameters())
        head_params = sum(p.numel() for p in model.classifier_head.parameters())

        print(f"  - hr_encoder:      {hr_enc_params:,} (expected 29,568)")
        print(f"  - acc_encoder:     {acc_enc_params:,} (expected 29,568)")
        print(f"  - meta_proj:       {meta_proj_params:,} (expected 96)")
        print(f"  - cross_fusion:    {fusion_params:,} (expected 26,562)")
        print(f"  - temporal_gru:    {gru_params:,} (expected 97,920)")
        print(f"  - temp_norm:       {norm_params:,} (expected 192)")
        print(f"  - classifier_head: {head_params:,} (expected 6,596)")

        self.assertEqual(hr_enc_params, 29568)
        self.assertEqual(acc_enc_params, 29568)
        self.assertEqual(meta_proj_params, 96)
        self.assertEqual(fusion_params, 26562)
        self.assertEqual(gru_params, 97920)
        self.assertEqual(norm_params, 192)
        self.assertEqual(head_params, 6596)

        # Budget cap checks
        self.assertEqual(total_params, 190502, f"Expected exactly 190,502 parameters, got {total_params}")
        self.assertLessEqual(total_params, 196728, "Exceeds +20% budget cap")

    def test_02_identity_initialization(self):
        """Verify zero initialization of gating MLP and gating scale parameters."""
        fusion = CircadianAnchoredCrossModalFusion()
        # gate_net[4] is the Linear(48, 128)
        self.assertTrue(torch.all(fusion.gate_net[4].weight == 0.0).item())
        self.assertTrue(torch.all(fusion.gate_net[4].bias == 0.0).item())
        self.assertAlmostEqual(fusion.alpha_hr.item(), 0.1, places=5)
        self.assertAlmostEqual(fusion.alpha_acc.item(), 0.1, places=5)
        print("[Test 2] Zero Identity Initialization: PASSED")

    def test_03_masked_temporal_instance_norm(self):
        """Verify Masked TIN standardizes valid sequence elements to zero mean and unit variance."""
        tin = MaskedTemporalInstanceNorm(64)
        B, T, C = 4, 600, 64
        # Synthetic shifted and scaled data with inter-batch variance
        x = torch.randn(B, T, C) * 15.0 + 72.0
        mask = torch.ones(B, T, dtype=torch.bool)
        # Pad tails differently per sample
        mask[0, 450:] = False
        mask[1, 500:] = False
        mask[2, 300:] = False
        mask[3, 550:] = False

        out = tin(x, mask)
        self.assertEqual(out.shape, (B, T, C))

        # Test valid positions per sample
        for b in range(B):
            valid_t = mask[b].nonzero(as_tuple=True)[0]
            valid_out = out[b, valid_t, :]
            mean_val = valid_out.mean(dim=0)
            std_val = valid_out.std(dim=0, unbiased=False)

            self.assertLess(torch.max(torch.abs(mean_val)).item(), 1e-4)
            self.assertLess(torch.max(torch.abs(std_val - 1.0)).item(), 1e-3)

        print("[Test 3] Masked TIN Standardization: PASSED")

    def test_04_forward_pass_shapes(self):
        """Verify forward pass output shapes and dictionary handling."""
        model = build_nas_iteration3_model()
        B, T = 4, 600
        x_ihr = torch.randn(B, T, 30)
        x_acc = torch.randn(B, T, 30)
        x_meta = torch.randn(B, T, 5)
        targets = torch.randint(0, 4, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)
        mask[:, 500:] = False
        seq_lens = torch.tensor([500, 500, 500, 500], dtype=torch.int64)

        # Standard args call
        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)
        self.assertIn("logits", out)
        self.assertEqual(out["logits"].shape, (B, T, 4))
        self.assertEqual(out["probabilities"].shape, (B, T, 4))
        self.assertEqual(out["total_loss"].dim(), 0)

        # Dictionary call
        batch_dict = {
            "ihr": x_ihr,
            "accel": x_acc,
            "meta": x_meta,
            "y": targets,
            "mask": mask,
            "seq_len": seq_lens
        }
        out_dict = model(batch_dict)
        self.assertEqual(out_dict["logits"].shape, (B, T, 4))
        self.assertEqual(out_dict["total_loss"].shape, out["total_loss"].shape)

        print("[Test 4] Forward Pass Shapes & Dictionary Calling: PASSED")

    def test_05_backward_pass_gradient_flow(self):
        """Verify backward pass flows gradients cleanly through all CACMF components."""
        model = build_nas_iteration3_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        B, T = 4, 600
        x_ihr = torch.randn(B, T, 30, requires_grad=False)
        x_acc = torch.randn(B, T, 30, requires_grad=False)
        x_meta = torch.randn(B, T, 5, requires_grad=False)
        targets = torch.randint(0, 4, (B, T))
        mask = torch.ones(B, T, dtype=torch.bool)

        # Step 1: Initial backward pass (verifies gate_net[4], tin, ms components)
        opt.zero_grad()
        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
        loss = out["total_loss"]
        loss.backward()

        cf = model.cross_fusion
        self.assertIsNotNone(cf.gate_net[4].weight.grad, "gate_net[4] missing grad")
        self.assertGreater(cf.gate_net[4].weight.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(cf.tin_hr.weight.grad, "tin_hr.weight missing grad")
        self.assertGreater(cf.tin_hr.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(cf.tin_acc.weight.grad, "tin_acc.weight missing grad")
        self.assertGreater(cf.tin_acc.weight.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(cf.ms_hr.proj_in.weight.grad, "ms_hr.proj_in missing grad")
        self.assertGreater(cf.ms_hr.proj_in.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(cf.ms_acc.proj_in.weight.grad, "ms_acc.proj_in missing grad")
        self.assertGreater(cf.ms_acc.proj_in.weight.grad.abs().sum().item(), 0.0)

        # Update weights (takes gate_net[4] off exact zero)
        opt.step()

        # Step 2: Second backward pass (verifies full gradient flow through gate_net[0], circadian_proj, and alpha)
        opt.zero_grad()
        out2 = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
        out2["total_loss"].backward()

        self.assertIsNotNone(cf.gate_net[0].weight.grad, "gate_net[0] missing grad")
        self.assertGreater(cf.gate_net[0].weight.grad.abs().sum().item(), 0.0)

        self.assertIsNotNone(cf.alpha_hr.grad, "alpha_hr missing grad")
        self.assertGreater(cf.alpha_hr.grad.abs().item(), 0.0)
        self.assertIsNotNone(cf.alpha_acc.grad, "alpha_acc missing grad")
        self.assertGreater(cf.alpha_acc.grad.abs().item(), 0.0)

        self.assertIsNotNone(cf.circadian_proj[0].weight.grad, "circadian_proj missing grad")
        self.assertGreater(cf.circadian_proj[0].weight.grad.abs().sum().item(), 0.0)

        print("[Test 5] Backward Pass & Multi-Step Gradient Flow: PASSED")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required for GPU verification")
    def test_06_gpu_execution(self):
        """Verify execution and backward pass on cuda:0."""
        device = torch.device("cuda:0")
        model = build_nas_iteration3_model().to(device)
        B, T = 4, 600
        x_ihr = torch.randn(B, T, 30, device=device)
        x_acc = torch.randn(B, T, 30, device=device)
        x_meta = torch.randn(B, T, 5, device=device)
        targets = torch.randint(0, 4, (B, T), device=device)
        mask = torch.ones(B, T, dtype=torch.bool, device=device)

        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
        loss = out["total_loss"]
        loss.backward()

        self.assertEqual(out["logits"].device.type, "cuda")
        self.assertIsNotNone(model.cross_fusion.gate_net[4].weight.grad)
        print(f"[Test 6] GPU Execution on {torch.cuda.get_device_name(0)}: PASSED")


if __name__ == "__main__":
    unittest.main(verbosity=2)
