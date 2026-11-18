"""
Unit Tests for NAS Iteration 2 (Candidate B: CrossModalFiLMCNNBiGRU)
File: .agents/teamwork/teamwork_worker_r2_1/test_nas_iteration2.py
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

from nas_iteration2_model import (
    CrossModalFiLMFusion,
    CrossModalFiLMCNNBiGRU,
    build_nas_iteration2_model
)


class TestNASIteration2(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_parameter_counts(self):
        """Verify exact parameter counts and budget compliance."""
        film = CrossModalFiLMFusion(d_model=64, d_cond=32)
        film_params = sum(p.numel() for p in film.parameters())
        self.assertEqual(
            film_params, 29440,
            f"CrossModalFiLMFusion parameters expected 29,440, got {film_params}"
        )

        model = CrossModalFiLMCNNBiGRU().to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        self.assertEqual(
            total_params, 193380,
            f"CrossModalFiLMCNNBiGRU parameters expected 193,380, got {total_params}"
        )
        self.assertLessEqual(
            total_params, 196728,
            f"Total parameters {total_params} exceed +20% cap (196,728)!"
        )

    def test_identity_initialization(self):
        """Verify second linear layer weights and biases in FiLM generators are zero-initialized."""
        film = CrossModalFiLMFusion(d_model=64, d_cond=32)
        self.assertTrue(torch.all(film.film_gen_hr[2].weight == 0.0))
        self.assertTrue(torch.all(film.film_gen_hr[2].bias == 0.0))
        self.assertTrue(torch.all(film.film_gen_acc[2].weight == 0.0))
        self.assertTrue(torch.all(film.film_gen_acc[2].bias == 0.0))

    def test_forward_shape(self):
        """Verify forward pass on BIDSleep standard shapes (B=4, T=600, 30)."""
        model = build_nas_iteration2_model().to(self.device)
        model.eval()

        B, T = 4, 600
        x_ihr = torch.randn(B, T, 30, device=self.device)
        x_acc = torch.randn(B, T, 30, device=self.device)
        x_meta = torch.randn(B, T, 5, device=self.device)
        seq_lens = torch.tensor([600, 550, 480, 600], dtype=torch.int64, device=self.device)
        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            out = model(x_ihr, x_acc, x_meta, mask=mask, seq_lens=seq_lens)

        self.assertIn("logits", out)
        self.assertEqual(out["logits"].shape, (B, T, 4))
        self.assertEqual(out["probabilities"].shape, (B, T, 4))
        self.assertEqual(out["temp_features"].shape, (B, T, 96))

    def test_forward_batch_dict(self):
        """Verify model accepts batch dictionary directly."""
        model = build_nas_iteration2_model().to(self.device)
        model.eval()

        B, T = 2, 100
        batch = {
            "ihr": torch.randn(B, T, 30, device=self.device),
            "accel": torch.randn(B, T, 30, device=self.device),
            "meta": torch.randn(B, T, 5, device=self.device),
            "y": torch.randint(0, 4, (B, T), device=self.device),
            "mask": torch.ones(B, T, dtype=torch.bool, device=self.device),
            "subject_idx": torch.tensor([0, 1], device=self.device),
            "seq_len": torch.tensor([100, 90], dtype=torch.int64, device=self.device)
        }

        with torch.no_grad():
            out = model(batch)

        self.assertEqual(out["logits"].shape, (B, T, 4))
        self.assertTrue(torch.is_tensor(out["total_loss"]))

    def test_backward_gradient_flow(self):
        """Verify backward pass and gradient flow through all FiLM and Gating parameters."""
        model = build_nas_iteration2_model().to(self.device)
        model.train()

        B, T = 4, 600
        x_ihr = torch.randn(B, T, 30, device=self.device)
        x_acc = torch.randn(B, T, 30, device=self.device)
        x_meta = torch.randn(B, T, 5, device=self.device)
        targets = torch.randint(0, 4, (B, T), device=self.device)
        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)
        seq_lens = torch.full((B,), T, dtype=torch.int64, device=self.device)

        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)
        loss = out["total_loss"]
        self.assertGreater(loss.item(), 0.0)

        loss.backward()

        # Check FiLM generators
        self.assertIsNotNone(model.cross_film.film_gen_hr[0].weight.grad)
        self.assertIsNotNone(model.cross_film.film_gen_hr[2].weight.grad)
        self.assertIsNotNone(model.cross_film.norm_hr.weight.grad)

        self.assertIsNotNone(model.cross_film.film_gen_acc[0].weight.grad)
        self.assertIsNotNone(model.cross_film.film_gen_acc[2].weight.grad)
        self.assertIsNotNone(model.cross_film.norm_acc.weight.grad)

        # Check Gating Network
        self.assertIsNotNone(model.cross_film.gate_net[0].weight.grad)
        self.assertIsNotNone(model.cross_film.gate_net[2].weight.grad)

        # Check Encoders and Heads
        self.assertIsNotNone(model.hr_encoder.net[0].weight.grad)
        self.assertIsNotNone(model.acc_encoder.net[0].weight.grad)
        self.assertIsNotNone(model.meta_proj[0].weight.grad)
        self.assertIsNotNone(model.temporal_gru.weight_ih_l0.grad)
        self.assertIsNotNone(model.classifier_head[0].weight.grad)


if __name__ == "__main__":
    unittest.main()
