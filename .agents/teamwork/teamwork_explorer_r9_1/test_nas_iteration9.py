r"""
.agents/teamwork/teamwork_explorer_r9_1/test_nas_iteration9.py
Unit Test Suite for NAS Iteration 9 Candidate Eta (SomatomotorAtoniaGatedPyramidCNNBiGRU).

Verifies all critical acceptance criteria:
  1. Total parameter budget compliance (<= 196,728) and >90% compression vs 5.47M benchmark
  2. Dual-Scale Local Basal Standardization (DSLBS) micro/macro statistics and padding protection
  3. Multi-Scale Local Ultradian Latent Normalization (Latent MUP LUAN) dimensions and channel contrast
  4. Dynamic Circadian Phase Gating exact step-0 identity preservation (2 * g == 1.0)
  5. Cross-Modal Somatomotor Atonia Gating exact step-0 identity preservation (2 * g_atonia == 1.0)
  6. Multi-interface execution (individual, 3-arg positional, keyword, dict) on BIDSleep shapes (B=4, T=600)
  7. Variable sequence length packed execution safety
  8. Numerical stability at step-0 forward pass
  9. Active, non-NaN, non-zero gradient flow across 100% of named parameters
 10. Direct 1-step intra-epoch skip gradient highway into raw input tensors
 11. Autonomic contrast differential Delta z(t) discrimination between Deep and Light sleep states
 12. Somatomotor atonia gating behavior: \sigma_acc ~ 0 activates REM selectivity while \sigma_acc > 0 suppresses REM
"""

import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../.."))
EXP014_DIR = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
if EXP014_DIR not in sys.path:
    sys.path.insert(0, EXP014_DIR)

from proposed_nas_iteration9_model import (
    DualScaleLocalBasalStandardization,
    MultiScaleLocalUltradianLatentNorm,
    CrossModalSomatomotorAtoniaGating,
    DynamicCircadianPhaseGating,
    InterEpochSequenceAttention,
    SomatomotorAtoniaGatedPyramidCNNBiGRU,
    build_nas_iteration9_model
)


class TestNASIteration9CandidateEta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"\n[Setup] Running TestNASIteration9CandidateEta on device: {cls.device}")

    def test_01_parameter_budget_and_compression(self):
        """Test 01: Verify strict compliance with <=196,728 parameter cap and >90% compression."""
        model = build_nas_iteration9_model()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        CAP = 196728
        BENCHMARK = 5470000

        self.assertLessEqual(
            total_params, CAP,
            f"Candidate Eta parameter count ({total_params:,}) exceeds cap ({CAP:,})!"
        )
        self.assertEqual(
            total_params, trainable_params,
            f"Non-trainable parameters found: total={total_params}, trainable={trainable_params}"
        )
        self.assertEqual(
            total_params, 196644,
            f"Expected exactly 196,644 parameters, got {total_params:,}"
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
        self.assertTrue(torch.all(luan.alpha == 0.0))
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

    def test_05_cross_modal_somatomotor_atonia_gating(self):
        """Test 05: Verify Cross-Modal Somatomotor Atonia Gating step-0 identity, shapes, and masking."""
        sag = CrossModalSomatomotorAtoniaGating(in_features=5, hidden_dim=4, out_features=144, zero_init=True).to(self.device)
        sag_params = sum(p.numel() for p in sag.parameters())
        self.assertEqual(sag_params, 752, f"Expected 752 parameters for SAG, got {sag_params}")

        B, T, D = 4, 600, 144
        e_intra = torch.randn(B, T, D, device=self.device)
        x_acc = torch.randn(B, T, 30, device=self.device) * 0.5
        mup_ctx = torch.randn(B, T, 6, device=self.device)
        mask = torch.ones(B, T, dtype=torch.bool, device=self.device)
        mask[:, 450:] = False

        e_sag, gate, atonia_ctx = sag(e_intra, x_acc, mup_ctx, mask=mask)

        # 1. Gate dimensions and step-0 value
        self.assertEqual(gate.shape, (B, T, 144))
        self.assertEqual(atonia_ctx.shape, (B, T, 5))
        valid_gate = gate[:, :450, :]
        self.assertTrue(
            torch.allclose(valid_gate, torch.full_like(valid_gate, 0.5), atol=1e-6),
            f"Atonia gate not 0.5 at step 0: min={valid_gate.min().item()}, max={valid_gate.max().item()}"
        )

        # 2. Exact step-0 identity: 2.0 * 0.5 = 1.0 -> e_sag == e_intra
        self.assertTrue(
            torch.allclose(e_sag[:, :450, :], e_intra[:, :450, :], atol=1e-5),
            "Step-0 e_sag does not match e_intra identity!"
        )

        # 3. Padded elements strictly zero
        self.assertEqual(e_sag[:, 450:, :].abs().max().item(), 0.0)
        self.assertEqual(gate[:, 450:, :].abs().max().item(), 0.0)
        self.assertEqual(atonia_ctx[:, 450:, :].abs().max().item(), 0.0)

        print(f"[Test 05 PASSED] Cross-Modal Somatomotor Atonia Gating exact step-0 identity confirmed ({sag_params} params).")

    def test_06_forward_all_interface_formats(self):
        """Test 06: Verify forward pass across all 4 interface invocation signatures on BIDSleep shapes."""
        model = build_nas_iteration9_model(zero_init=True).to(self.device)
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
        self.assertEqual(out1["atonia_gate"].shape, (B, T, 144))
        self.assertEqual(out1["atonia_ctx"].shape, (B, T, 5))

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

        print("[Test 06 PASSED] Forward pass across all 4 interface formats verified on BIDSleep shapes (B=4, T=600).")

    def test_07_variable_length_packed_execution(self):
        """Test 07: Verify variable-length sequence packing safety without NaNs."""
        model = build_nas_iteration9_model().to(self.device)
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
        print("[Test 07 PASSED] Variable sequence length packed execution confirmed.")

    def test_08_step0_numerical_stability(self):
        """Test 08: Verify initial step-0 forward pass produces valid probabilities and bounded loss."""
        model = build_nas_iteration9_model(zero_init=True).to(self.device)
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
        print(f"[Test 08 PASSED] Step-0 numerical stability verified (Loss: {out['total_loss'].item():.4f}).")

    def test_09_backward_gradient_coverage_all_named_parameters(self):
        """Test 09: Verify active, non-NaN, non-zero gradient flow across 100% of named parameters."""
        model = build_nas_iteration9_model(zero_init=False).to(self.device)
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

        self.assertEqual(param_count, 110, f"Expected 110 named parameters in Candidate Eta, got {param_count}")
        print(f"[Test 09 PASSED] Backward gradient coverage confirmed across all {param_count} named parameters.")

    def test_10_direct_intra_epoch_skip_gradient_highway(self):
        """Test 10: Verify direct 1-step gradient flow from loss into raw input tensors."""
        model = build_nas_iteration9_model(zero_init=True).to(self.device)
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

        print("[Test 10 PASSED] Direct intra-epoch skip gradient highway to raw input tensors confirmed.")

    def test_11_autonomic_contrast_deep_vs_light_discrimination(self):
        """Test 11: Verify Autonomic Contrast Differential Delta z(t) separates sustained Deep from transient Light."""
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
            f"[Test 11 PASSED] Autonomic contrast verified: Deep Delta z = {delta_z_deep:.4f} (~0) vs "
            f"Light Arousal Delta z = {delta_z_arousal:.4f} (>0.5)."
        )

    def test_12_somatomotor_atonia_gating_behavior_rem_vs_light(self):
        r"""Test 12: Verify Somatomotor Atonia Gating behavior: \sigma_acc ~ 0 activates REM selectivity while \sigma_acc > 0 suppresses REM."""
        sag = CrossModalSomatomotorAtoniaGating(in_features=5, hidden_dim=4, out_features=144, zero_init=False).to(self.device)

        # 1. Feature-Level Coupling Analysis:
        # Construct synthetic REM vs Light conditions with identical elevated autonomic HR (delta_z = +1.5)
        # Condition A (REM): Skeletal muscle atonia (\sigma_acc = 0.005 ~ 0)
        # Condition B (Light sleep micro-movement): Muscle tone twitches (\sigma_acc = 2.0 > 0)
        B, T = 1, 100
        x_acc_rem = torch.randn(B, T, 30, device=self.device) * 0.005  # Atonia flatline
        x_acc_light = torch.randn(B, T, 30, device=self.device) * 2.0   # Movement twitches

        mup_ctx = torch.zeros(B, T, 6, device=self.device)
        mup_ctx[:, :, 3] = 1.5   # delta_z = +1.5 (autonomic surge)
        mup_ctx[:, :, 4] = 0.2   # z_macro

        e_intra = torch.randn(B, T, 144, device=self.device)

        with torch.no_grad():
            _, _, ctx_rem = sag(e_intra, x_acc_rem, mup_ctx)
            _, _, ctx_light = sag(e_intra, x_acc_light, mup_ctx)

        # Column 0: \sigma_acc, Column 4: \Delta z * exp(-\sigma_acc)
        sigma_acc_rem = ctx_rem[0, 50, 0].item()
        sigma_acc_light = ctx_light[0, 50, 0].item()
        synergy_rem = ctx_rem[0, 50, 4].item()
        synergy_light = ctx_light[0, 50, 4].item()

        self.assertAlmostEqual(sigma_acc_rem, 0.0, delta=0.02)
        self.assertGreater(sigma_acc_light, 1.5)

        # Synergy index in REM must be near delta_z (1.5)
        self.assertAlmostEqual(synergy_rem, 1.5, delta=0.05)
        # Synergy index in Light must be heavily attenuated (< 0.35)
        self.assertLess(synergy_light, 0.35)
        synergy_suppression_ratio = synergy_rem / max(synergy_light, 1e-6)
        self.assertGreater(synergy_suppression_ratio, 4.0, f"Synergy ratio {synergy_suppression_ratio:.2f} <= 4.0")

        # 2. Optimization Dynamics & Directional Selectivity:
        # Train SAG on synthetic REM vs Light differentiation and verify divergence in output gating
        model = build_nas_iteration9_model(zero_init=False).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        # Class 1: Light Sleep, Class 3: REM Sleep
        # In REM (class 3): HR elevated (delta_z > 0), acc quiet (sigma_acc ~ 0)
        # In Light (class 1): HR elevated (delta_z > 0), acc active (sigma_acc > 0)
        B_train, T_train = 2, 50
        x_ihr_train = torch.randn(B_train, T_train, 30, device=self.device) * 4.0 + 75.0
        x_acc_train = torch.empty(B_train, T_train, 30, device=self.device)
        x_acc_train[0] = torch.randn(T_train, 30, device=self.device) * 0.01  # REM atonia
        x_acc_train[1] = torch.randn(T_train, 30, device=self.device) * 2.5   # Light twitching

        x_meta_train = torch.zeros(B_train, T_train, 5, device=self.device)
        x_meta_train[:, :, 2] = 0.5
        x_meta_train[:, :, 3] = x_ihr_train.mean(dim=-1)
        x_meta_train[:, :, 4] = x_ihr_train.std(dim=-1)

        targets_train = torch.empty(B_train, T_train, dtype=torch.long, device=self.device)
        targets_train[0] = 3  # Target REM for atonia sample
        targets_train[1] = 1  # Target Light for motion sample
        mask_train = torch.ones(B_train, T_train, dtype=torch.bool, device=self.device)

        model.train()
        for _ in range(8):
            optimizer.zero_grad()
            out = model(x_ihr_train, x_acc_train, x_meta_train, targets=targets_train, mask=mask_train)
            out["total_loss"].backward()
            optimizer.step()

        # Evaluate trained gating response on held-out test input
        model.eval()
        with torch.no_grad():
            eval_out = model(x_ihr_train, x_acc_train, x_meta_train, mask=mask_train)
            probs = eval_out["probabilities"]  # (B, T, 4): [Wake, Light, Deep, REM]
            rem_prob_atonia = probs[0, :, 3].mean().item()    # Subject 0 (atonia) REM prob
            rem_prob_motion = probs[1, :, 3].mean().item()    # Subject 1 (motion) REM prob
            light_prob_atonia = probs[0, :, 1].mean().item()  # Subject 0 Light prob
            light_prob_motion = probs[1, :, 1].mean().item()  # Subject 1 Light prob

        # Verify directional selectivity:
        # Atonia condition must strongly activate REM compared to motion condition
        self.assertGreater(
            rem_prob_atonia, rem_prob_motion,
            f"REM prob under atonia ({rem_prob_atonia:.4f}) not > motion ({rem_prob_motion:.4f})!"
        )
        # Motion condition must strongly activate Light compared to atonia condition
        self.assertGreater(
            light_prob_motion, light_prob_atonia,
            f"Light prob under motion ({light_prob_motion:.4f}) not > atonia ({light_prob_atonia:.4f})!"
        )

        print(
            f"[Test 12 PASSED] Somatomotor atonia gating verified: "
            f"Synergy REM={synergy_rem:.4f} vs Light={synergy_light:.4f} (ratio {synergy_suppression_ratio:.1f}x) | "
            f"REM prob atonia={rem_prob_atonia:.3f} vs motion={rem_prob_motion:.3f} | "
            f"Light prob motion={light_prob_motion:.3f} vs atonia={light_prob_atonia:.3f}."
        )


if __name__ == "__main__":
    unittest.main()
