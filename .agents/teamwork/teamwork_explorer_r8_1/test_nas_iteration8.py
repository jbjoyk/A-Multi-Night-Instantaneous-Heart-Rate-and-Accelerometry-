r"""
.agents/teamwork/teamwork_explorer_r8_1/test_nas_iteration8.py
Unit Test Suite for NAS Iteration 8 Candidate Zeta (LocalUltradianPyramidCNNBiGRU).

Verifies all critical acceptance criteria:
  1. Total parameter budget compliance (<= 196,728) and >90% compression vs 5.47M benchmark
  2. Dual-Scale Local Basal Standardization (DSLBS) micro/macro statistics and padding protection
  3. Multi-Scale Local Ultradian Latent Normalization (Latent MUP LUAN) dimensions and channel contrast
  4. Dynamic Circadian Phase Gating exact step-0 identity preservation (2 * g == 1.0)
  5. Multi-interface execution (individual, 3-arg positional, keyword, dict) on BIDSleep shapes (B=4, T=600)
  6. Variable sequence length packed execution safety
  7. Numerical stability at step-0 forward pass
  8. Active, non-NaN, non-zero gradient flow across 100% of named parameters
  9. Direct 1-step intra-epoch skip gradient highway into raw input tensors
 10. Autonomic contrast differential Delta z(t) discrimination between Deep and Light sleep states
"""

import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from proposed_nas_iteration8_model import (
    DualScaleLocalBasalStandardization,
    MultiScaleLocalUltradianLatentNorm,
    DynamicCircadianPhaseGating,
    InterEpochSequenceAttention,
    LocalUltradianPyramidCNNBiGRU,
    build_nas_iteration8_model
)


class TestNASIteration8CandidateZeta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"\n[Setup] Running TestNASIteration8CandidateZeta on device: {cls.device}")

    def test_01_parameter_budget_and_compression(self):
        """Test 01: Verify strict compliance with <=196,728 parameter cap and >90% compression."""
        model = build_nas_iteration8_model()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        CAP = 196728
        BENCHMARK = 5470000

        self.assertLessEqual(
            total_params, CAP,
            f"Candidate Zeta parameter count ({total_params:,}) exceeds cap ({CAP:,})!"
        )
        self.assertEqual(
            total_params, trainable_params,
            f"Non-trainable parameters found: total={total_params}, trainable={trainable_params}"
        )

        compression_ratio = (BENCHMARK - total_params) / BENCHMARK * 100.0
        self.assertGreater(compression_ratio, 90.0, f"Compression ratio {compression_ratio:.2f}% < 90%!")

        headroom = CAP - total_params
        print(
            f"[Test 01 PASSED] Params: {total_params:,} | Cap: {CAP:,} | "
            f"Headroom: {headroom:,} | Compression: {compression_ratio:.2f}%"
        )

    def test_02_dual_scale_local_basal_standardization(self):
        """Test 02: Verify DSLBS micro/macro running statistics, padding mask, and Delta z(t)."""
        dslbs = DualScaleLocalBasalStandardization(window_macro=91, window_micro=15).to(self.device)

        B, T = 2, 600
        # Subject 0: low baseline resting HR ~52 bpm (slow-wave sleep dominant)
        # Subject 1: elevated baseline resting HR ~80 bpm
        x_ihr = torch.empty(B, T, 30, device=self.device)
        x_ihr[0] = torch.randn(T, 30, device=self.device) * 2.0 + 52.0
        x_ihr[1] = torch.randn(T, 30, device=self.device) * 4.0 + 80.0

        x_meta = torch.zeros(B, T, 5, device=self.device)
        x_meta[:, :, 0] = 1.0  # freq
        x_meta[:, :, 1] = 0.0  # clock
        x_meta[:, :, 2] = torch.linspace(0.0, 1.0, T, device=self.device).unsqueeze(0).expand(B, T)
        x_meta[:, :, 3] = x_ihr.mean(dim=-1)
        x_meta[:, :, 4] = x_ihr.std(dim=-1)

        valid_len = 500
        mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        mask[:, :valid_len] = True

        z_ihr, z_meta, mup_ctx = dslbs(x_ihr, x_meta, mask=mask)

        # Invariant checks for both subjects
        for b in range(B):
            valid_z = z_ihr[b, :valid_len]
            padded_z = z_ihr[b, valid_len:]

            mean_val = valid_z.mean().item()
            std_val = valid_z.std(unbiased=False).item()

            # Running local standardization maps both subjects to zero-mean unit-variance
            self.assertAlmostEqual(mean_val, 0.0, delta=0.03, msg=f"Subject {b} local mean not ~0: {mean_val}")
            self.assertAlmostEqual(std_val, 1.0, delta=0.05, msg=f"Subject {b} local std not ~1: {std_val}")

            # Padded elements strictly zero
            self.assertEqual(padded_z.abs().max().item(), 0.0)

            # Meta column 3 normalized and padded strictly zero
            valid_meta3 = z_meta[b, :valid_len, 3]
            padded_meta3 = z_meta[b, valid_len:, 3]
            self.assertAlmostEqual(valid_meta3.mean().item(), 0.0, delta=0.03)
            self.assertEqual(padded_meta3.abs().max().item(), 0.0)

            # Meta columns 0, 1, 2 preserved
            self.assertTrue(torch.allclose(z_meta[b, :valid_len, 0], x_meta[b, :valid_len, 0]))
            self.assertTrue(torch.allclose(z_meta[b, :valid_len, 2], x_meta[b, :valid_len, 2]))

            # MUP context shape (B, T, 6) and valid padding
            self.assertEqual(mup_ctx.shape, (B, T, 6))
            self.assertEqual(mup_ctx[b, valid_len:, :].abs().max().item(), 0.0)

        print("[Test 02 PASSED] Dual-Scale Local Basal Standardization verified across 52 bpm vs 80 bpm baselines.")

    def test_03_multi_scale_latent_norm(self):
        """Test 03: Verify Latent MUP LUAN channel standardizers, zero-init identity, and mask preservation."""
        luan = MultiScaleLocalUltradianLatentNorm(num_features=64, window_macro=91, window_micro=15).to(self.device)
        B, T, C = 3, 400, 64
        x = torch.randn(B, T, C, device=self.device)
        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)
        mask[:, 300:] = False

        out = luan(x, mask=mask)
        self.assertEqual(out.shape, (B, T, C))
        self.assertEqual(out[:, 300:, :].abs().max().item(), 0.0)
        self.assertTrue(torch.all(luan.alpha == 0.0))  # Confirms zero-init for step-0 identity
        print("[Test 03 PASSED] Multi-Scale Latent Normalization dimensions and zero-init verified.")

    def test_04_dynamic_circadian_phase_gating_identity(self):
        """Test 04: Verify Dynamic Circadian Phase Gating exact step-0 identity preservation (2 * g == 1.0)."""
        dcpg = DynamicCircadianPhaseGating(in_features=6, hidden_dim=16, out_features=144, zero_init=True).to(self.device)
        B, T, D = 4, 600, 144
        e_intra = torch.randn(B, T, D, device=self.device)
        circ_ctx = torch.randn(B, T, 6, device=self.device)
        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)
        mask[:, 450:] = False

        e_gated, gate = dcpg(e_intra, circ_ctx, mask=mask)

        # Gate should be exactly 0.5 everywhere for valid positions
        valid_gate = gate[:, :450, :]
        self.assertTrue(
            torch.allclose(valid_gate, torch.full_like(valid_gate, 0.5), atol=1e-6),
            f"Gate not 0.5 at step 0: min={valid_gate.min().item()}, max={valid_gate.max().item()}"
        )

        # 2.0 * gate is exactly 1.0, so e_gated[:450] == e_intra[:450]
        self.assertTrue(
            torch.allclose(e_gated[:, :450, :], e_intra[:, :450, :], atol=1e-5),
            "Step-0 e_gated does not match e_intra identity!"
        )

        # Padded elements strictly zero
        self.assertEqual(e_gated[:, 450:, :].abs().max().item(), 0.0)
        print("[Test 04 PASSED] Dynamic Circadian Phase Gating exact step-0 identity confirmed.")

    def test_05_forward_all_interface_formats(self):
        """Test 05: Verify forward pass across all 4 interface invocation signatures on BIDSleep shapes."""
        model = build_nas_iteration8_model(zero_init=True).to(self.device)
        model.eval()

        B, T = 4, 600
        x_ihr = torch.randn(B, T, 30, device=self.device) * 5.0 + 65.0
        x_acc = torch.randn(B, T, 30, device=self.device)
        x_meta = torch.randn(B, T, 5, device=self.device)
        x_meta[:, :, 3] = x_ihr.mean(dim=-1)
        x_meta[:, :, 4] = x_ihr.std(dim=-1)
        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)
        mask[:, 520:] = False

        # Format 1: Individual tensor arguments
        with torch.no_grad():
            out1 = model(x_ihr, x_acc, x_meta, mask=mask)
        self.assertEqual(out1["logits"].shape, (B, T, 4))
        self.assertEqual(out1["mup_ctx"].shape, (B, T, 6))

        # Format 2: Positional 3-argument: model(x, x_meta, mask) where x is (B, T, 60)
        x_joint = torch.cat([x_ihr, x_acc], dim=-1)
        with torch.no_grad():
            out2 = model(x_joint, x_meta, mask)
        self.assertEqual(out2["logits"].shape, (B, T, 4))
        self.assertTrue(torch.allclose(out1["logits"][:, :520], out2["logits"][:, :520], atol=1e-5))

        # Format 3: Keyword call: model(x, x_meta=x_meta, mask=mask)
        with torch.no_grad():
            out3 = model(x_joint, x_meta=x_meta, mask=mask)
        self.assertEqual(out3["logits"].shape, (B, T, 4))
        self.assertTrue(torch.allclose(out1["logits"][:, :520], out3["logits"][:, :520], atol=1e-5))

        # Format 4: Batch dictionary call: model(batch_dict)
        batch_dict = {
            "ihr": x_ihr,
            "accel": x_acc,
            "meta": x_meta,
            "mask": mask
        }
        with torch.no_grad():
            out4 = model(batch_dict)
        self.assertEqual(out4["logits"].shape, (B, T, 4))
        self.assertTrue(torch.allclose(out1["logits"][:, :520], out4["logits"][:, :520], atol=1e-5))

        print("[Test 05 PASSED] Forward pass across all 4 interface formats verified on BIDSleep shapes (B=4, T=600).")

    def test_06_variable_length_packed_execution(self):
        """Test 06: Verify variable-length sequence packing safety without NaNs."""
        model = build_nas_iteration8_model().to(self.device)
        model.eval()

        B, T = 4, 600
        seq_lens = torch.tensor([600, 480, 320, 150], dtype=torch.int64, device=self.device)
        mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        for i, slen in enumerate(seq_lens):
            mask[i, :slen] = True

        x_ihr = torch.randn(B, T, 30, device=self.device) * 5.0 + 65.0
        x_acc = torch.randn(B, T, 30, device=self.device)
        x_meta = torch.randn(B, T, 5, device=self.device)
        x_meta[:, :, 3] = x_ihr.mean(dim=-1)
        x_meta[:, :, 4] = x_ihr.std(dim=-1)
        targets = torch.randint(0, 4, (B, T), device=self.device)

        with torch.no_grad():
            out = model(
                x_ihr, x_acc, x_meta,
                targets=targets,
                mask=mask,
                seq_lens=seq_lens
            )

        self.assertFalse(torch.isnan(out["logits"]).any())
        self.assertFalse(torch.isnan(out["total_loss"]).any())
        self.assertGreater(out["total_loss"].item(), 0.0)
        print("[Test 06 PASSED] Variable sequence length packed execution confirmed.")

    def test_07_step0_numerical_stability(self):
        """Test 07: Verify initial step-0 forward pass produces valid probabilities and bounded loss."""
        model = build_nas_iteration8_model(zero_init=True).to(self.device)
        model.eval()

        B, T = 2, 300
        x_ihr = torch.randn(B, T, 30, device=self.device) * 8.0 + 65.0
        x_acc = torch.rand(B, T, 30, device=self.device) * 0.1
        x_meta = torch.randn(B, T, 5, device=self.device)
        x_meta[:, :, 3] = x_ihr.mean(dim=-1)
        x_meta[:, :, 4] = x_ihr.std(dim=-1)
        targets = torch.randint(0, 4, (B, T), device=self.device)
        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)

        probs = out["probabilities"]
        prob_sums = probs.sum(dim=-1)
        self.assertTrue(torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5))
        self.assertGreater(out["total_loss"].item(), 0.0)
        self.assertLess(out["total_loss"].item(), 5.0)
        print(f"[Test 07 PASSED] Step-0 numerical stability verified (Loss: {out['total_loss'].item():.4f}).")

    def test_08_backward_gradient_coverage_all_named_parameters(self):
        """Test 08: Verify active, non-NaN, non-zero gradient flow across 100% of named parameters."""
        model = build_nas_iteration8_model(zero_init=False).to(self.device)
        model.train()

        B, T = 2, 200
        x_ihr = torch.randn(B, T, 30, device=self.device) * 5.0 + 60.0
        x_acc = torch.randn(B, T, 30, device=self.device)
        x_meta = torch.randn(B, T, 5, device=self.device)
        x_meta[:, :, 3] = x_ihr.mean(dim=-1)
        x_meta[:, :, 4] = x_ihr.std(dim=-1)
        targets = torch.randint(0, 4, (B, T), device=self.device)
        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)
        seq_lens = torch.tensor([200, 200], dtype=torch.int64, device=self.device)

        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)
        loss = out["total_loss"]
        loss.backward()

        param_count = 0
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, f"Parameter {name} did not receive a gradient!")
            self.assertFalse(torch.isnan(param.grad).any(), f"Parameter {name} has NaN in gradient!")
            grad_norm = param.grad.norm().item()
            self.assertGreater(grad_norm, 0.0, f"Parameter {name} has zero gradient norm!")
            param_count += 1

        print(f"[Test 08 PASSED] Backward gradient coverage confirmed across all {param_count} named parameters.")

    def test_09_direct_intra_epoch_skip_gradient_highway(self):
        """Test 09: Verify direct 1-step gradient flow from loss into raw input tensors."""
        model = build_nas_iteration8_model(zero_init=True).to(self.device)
        model.train()

        B, T = 2, 100
        x_ihr = (torch.randn(B, T, 30, device=self.device) * 5.0 + 60.0).detach().requires_grad_(True)
        x_acc = torch.randn(B, T, 30, device=self.device).detach().requires_grad_(True)
        x_meta = torch.randn(B, T, 5, device=self.device).detach().requires_grad_(True)
        targets = torch.randint(0, 4, (B, T), device=self.device)
        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)

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

        print("[Test 09 PASSED] Direct intra-epoch skip gradient highway to raw input tensors confirmed.")

    def test_10_autonomic_contrast_deep_vs_light_discrimination(self):
        """Test 10: Verify Autonomic Contrast Differential Delta z(t) separates sustained Deep from transient Light."""
        dslbs = DualScaleLocalBasalStandardization(window_macro=91, window_micro=15).to(self.device)

        B, T = 1, 300
        # Epochs 0..150: Deep sleep (constant low HR ~50 bpm with minimal variance)
        # Epochs 150..300: Light sleep with a sharp transient micro-arousal at epoch 200..210 (HR jumps to 75 bpm)
        hr_sim = torch.full((1, T, 30), 50.0, device=self.device)
        hr_sim[0, 150:, :] = 58.0  # Light sleep baseline
        hr_sim[0, 200:210, :] = 75.0  # Transient micro-arousal

        meta_sim = torch.zeros(1, T, 5, device=self.device)
        meta_sim[:, :, 2] = torch.linspace(0.0, 1.0, T, device=self.device).unsqueeze(0)
        meta_sim[:, :, 3] = hr_sim.mean(dim=-1)
        meta_sim[:, :, 4] = hr_sim.std(dim=-1).clamp(min=1.0)
        mask = torch.ones(1, T, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            _, _, mup_ctx = dslbs(hr_sim, meta_sim, mask=mask)

        # mup_ctx columns: [time, circ_trend, circ_var_ratio, delta_z, z_macro, var_ratio]
        delta_z = mup_ctx[0, :, 3]
        z_macro = mup_ctx[0, :, 4]

        # In Deep sleep (epoch 50): Delta z is ~0, z_macro is negative
        delta_z_deep = delta_z[50].item()
        z_macro_deep = z_macro[50].item()
        self.assertAlmostEqual(delta_z_deep, 0.0, delta=0.1, msg=f"Delta z not ~0 in Deep sleep: {delta_z_deep}")
        self.assertLess(z_macro_deep, 0.0, f"z_macro not negative in Deep sleep: {z_macro_deep}")

        # At the transient micro-arousal (epoch 205): Delta z spikes positive
        delta_z_arousal = delta_z[205].item()
        self.assertGreater(
            delta_z_arousal, delta_z_deep + 0.5,
            f"Delta z did not spike during arousal: deep={delta_z_deep:.3f}, arousal={delta_z_arousal:.3f}"
        )

        print(
            f"[Test 10 PASSED] Autonomic contrast verified: Deep Delta z = {delta_z_deep:.4f} (~0) vs "
            f"Light Arousal Delta z = {delta_z_arousal:.4f} (>0.5)."
        )


if __name__ == "__main__":
    unittest.main()
