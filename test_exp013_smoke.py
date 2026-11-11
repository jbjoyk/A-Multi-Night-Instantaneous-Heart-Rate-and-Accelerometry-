"""
Unit and Smoke Tests for EXP-013 Models
File: test_exp013_smoke.py

Verifies:
1. Instantiation of Models A, B, C, D.
2. Parameter count constraints (Model A within +/-10% of 162,325; B/C/D matched).
3. Forward pass tensor shapes (B, T, 4).
4. Loss computation and decomposition:
   - Model A: loss_trans == 0, loss_adv == 0, loss_boundary == 0, alpha is None
   - Model B: loss_trans > 0, loss_adv == 0, loss_boundary == 0, alpha is dynamic
   - Model C: loss_trans == 0, loss_adv > 0, loss_boundary == 0, alpha is dynamic
   - Model D: loss_trans > 0, loss_adv > 0, loss_boundary == 0, alpha is constant 0.5
5. Backward pass and gradient flow through all trainable parameters.
6. Error handling for Model E (asserting that training Model E raises ValueError).
"""

import unittest
import torch
from exp013_models import ConventionalCNNBiGRU, build_exp013_model


class TestEXP013Models(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.B, self.T = 2, 600
        self.x_ihr = torch.randn(self.B, self.T, 30)
        self.x_acc = torch.randn(self.B, self.T, 30)
        self.x_meta = torch.randn(self.B, self.T, 5)
        self.targets = torch.randint(0, 4, (self.B, self.T))
        self.mask = torch.ones(self.B, self.T, dtype=torch.bool)
        self.subj_idx = torch.tensor([0, 1])
        self.ref_params = 162325

    def test_parameter_counts(self):
        model_a = build_exp013_model("A", n_train_subjects=31)
        model_b = build_exp013_model("B", n_train_subjects=31)
        model_c = build_exp013_model("C", n_train_subjects=31)
        model_d = build_exp013_model("D", n_train_subjects=31)

        pa = sum(p.numel() for p in model_a.parameters())
        pb = sum(p.numel() for p in model_b.parameters())
        pc = sum(p.numel() for p in model_c.parameters())
        pd = sum(p.numel() for p in model_d.parameters())

        print(f"\n[Test] Model Parameter Audit:")
        print(f"  Model A: {pa:6d} (diff vs E: {pa - self.ref_params:+6d}, {(pa - self.ref_params)/self.ref_params*100:+.2f}%)")
        print(f"  Model B: {pb:6d} (diff vs E: {pb - self.ref_params:+6d}, {(pb - self.ref_params)/self.ref_params*100:+.2f}%)")
        print(f"  Model C: {pc:6d} (diff vs E: {pc - self.ref_params:+6d}, {(pc - self.ref_params)/self.ref_params*100:+.2f}%)")
        print(f"  Model D: {pd:6d} (diff vs E: {pd - self.ref_params:+6d}, {(pd - self.ref_params)/self.ref_params*100:+.2f}%)")
        print(f"  Model E: {self.ref_params:6d} (Frozen EXP-012 Reference)")

        # Model A constraint: within +/- 10% of 162,325
        self.assertTrue(abs(pa - self.ref_params) / self.ref_params <= 0.10, "Model A exceeds +/-10% parameter limit")
        # Models B, C, D approximately matched
        self.assertTrue(abs(pb - self.ref_params) / self.ref_params <= 0.10, "Model B exceeds +/-10% parameter limit")
        self.assertTrue(abs(pc - self.ref_params) / self.ref_params <= 0.10, "Model C exceeds +/-10% parameter limit")
        self.assertTrue(abs(pd - self.ref_params) / self.ref_params <= 0.10, "Model D exceeds +/-10% parameter limit")

    def test_model_a_forward_backward(self):
        model_a = build_exp013_model("A", n_train_subjects=31)
        out = model_a(self.x_ihr, self.x_acc, self.x_meta, targets=self.targets, mask=self.mask, subject_idx=self.subj_idx)

        self.assertEqual(out["logits"].shape, (self.B, self.T, 4))
        self.assertEqual(out["probabilities"].shape, (self.B, self.T, 4))
        self.assertIsNone(out["alpha_gate"])
        self.assertEqual(out["loss_trans"].item(), 0.0)
        self.assertEqual(out["loss_adv"].item(), 0.0)
        self.assertEqual(out["loss_boundary"].item(), 0.0)
        self.assertTrue(out["total_loss"].item() > 0.0)

        out["total_loss"].backward()
        for name, param in model_a.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient missing for Model A parameter {name}")

    def test_model_b_forward_backward(self):
        model_b = build_exp013_model("B", n_train_subjects=31)
        out = model_b(self.x_ihr, self.x_acc, self.x_meta, targets=self.targets, mask=self.mask, subject_idx=self.subj_idx)

        self.assertEqual(out["logits"].shape, (self.B, self.T, 4))
        self.assertIsNotNone(out["alpha_gate"])
        self.assertTrue(out["loss_trans"].item() > 0.0, "Model B must have active transition loss")
        self.assertEqual(out["loss_adv"].item(), 0.0, "Model B must have GRL loss == 0")
        self.assertEqual(out["loss_boundary"].item(), 0.0)

        out["total_loss"].backward()
        for name, param in model_b.named_parameters():
            if "subject_discriminator" not in name:
                self.assertIsNotNone(param.grad, f"Gradient missing for Model B parameter {name}")

    def test_model_c_forward_backward(self):
        model_c = build_exp013_model("C", n_train_subjects=31)
        out = model_c(self.x_ihr, self.x_acc, self.x_meta, targets=self.targets, mask=self.mask, subject_idx=self.subj_idx)

        self.assertEqual(out["logits"].shape, (self.B, self.T, 4))
        self.assertIsNotNone(out["alpha_gate"])
        self.assertEqual(out["loss_trans"].item(), 0.0, "Model C must have transition loss == 0")
        self.assertTrue(out["loss_adv"].item() > 0.0, "Model C must have active GRL loss")
        self.assertEqual(out["loss_boundary"].item(), 0.0)

        out["total_loss"].backward()
        for name, param in model_c.named_parameters():
            if "transition_matrix" not in name and "gamma" not in name:
                self.assertIsNotNone(param.grad, f"Gradient missing for Model C parameter {name}")

    def test_model_d_forward_backward(self):
        model_d = build_exp013_model("D", n_train_subjects=31)
        out = model_d(self.x_ihr, self.x_acc, self.x_meta, targets=self.targets, mask=self.mask, subject_idx=self.subj_idx)

        self.assertEqual(out["logits"].shape, (self.B, self.T, 4))
        self.assertIsNotNone(out["alpha_gate"])
        # Check that alpha is fixed equal 0.5
        self.assertTrue(torch.allclose(out["alpha_gate"], torch.tensor(0.5)), "Model D must use fixed 0.5 alpha")
        self.assertTrue(out["loss_trans"].item() > 0.0, "Model D must have active transition loss")
        self.assertTrue(out["loss_adv"].item() > 0.0, "Model D must have active GRL loss")
        self.assertEqual(out["loss_boundary"].item(), 0.0)

        out["total_loss"].backward()
        for name, param in model_d.named_parameters():
            self.assertIsNotNone(param.grad, f"Gradient missing for Model D parameter {name}")

    def test_model_e_freeze_protection(self):
        with self.assertRaises(ValueError):
            build_exp013_model("E")


if __name__ == "__main__":
    unittest.main()
