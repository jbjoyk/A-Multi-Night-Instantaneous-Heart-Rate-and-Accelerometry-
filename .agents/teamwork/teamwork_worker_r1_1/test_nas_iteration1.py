"""
Unit and Integration Tests for NAS Iteration 1 Architecture and Evaluation
File: .agents/teamwork/teamwork_worker_r1_1/test_nas_iteration1.py
"""

import os
import sys
import json
import pytest
import torch
import torch.nn as nn

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

EXP014_DIR = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014")
if EXP014_DIR not in sys.path:
    sys.path.insert(0, EXP014_DIR)

from exp013_models import ConventionalCNNBiGRU
from nas_iteration1_model import (
    MultiScaleConventionalCNNBiGRU,
    MutatedModelA_MSCNNBiGRU,
    MultiScaleTemporalFusion,
    MultiScaleTemporalConvBlock,
    build_nas_iteration1_model
)


def test_baseline_parameter_count():
    baseline = ConventionalCNNBiGRU()
    p_base = sum(p.numel() for p in baseline.parameters())
    assert p_base == 163940, f"Expected 163,940 baseline params, got {p_base}"


def test_mutated_parameter_count_budget():
    m_pre = MultiScaleConventionalCNNBiGRU(mutation_type="pre_recurrent")
    p_pre = sum(p.numel() for p in m_pre.parameters())
    assert p_pre == 192692, f"Expected 192,692 pre-recurrent params, got {p_pre}"
    assert p_pre <= 196728, f"Parameter count {p_pre} exceeds 196,728 cap!"
    delta_pct = (p_pre - 163940) / 163940 * 100.0
    assert delta_pct <= 20.0, f"Percentage increase {delta_pct:.2f}% exceeds 20%"

    m_intra = MultiScaleConventionalCNNBiGRU(mutation_type="intra_encoder")
    p_intra = sum(p.numel() for p in m_intra.parameters())
    assert p_intra == 173284, f"Expected 173,284 intra-encoder params, got {p_intra}"
    assert p_intra <= 196728, f"Parameter count {p_intra} exceeds 196,728 cap!"


def test_mstf_block_tensor_invariance():
    mstf = MultiScaleTemporalFusion(in_channels=144, bottleneck_channels=32, kernel_sizes=(1, 3, 5))
    p_mstf = sum(p.numel() for p in mstf.parameters())
    assert p_mstf == 28752, f"Expected 28,752 params in MSTF, got {p_mstf}"

    x = torch.randn(4, 600, 144)
    out = mstf(x)
    assert out.shape == (4, 600, 144), f"Expected shape (4, 600, 144), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN detected in MSTF output"


def test_forward_pass_individual_tensors():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_nas_iteration1_model(mutation_type="pre_recurrent").to(device)

    for B, T in [(2, 60), (4, 600)]:
        x_ihr = torch.randn(B, T, 30, device=device)
        x_acc = torch.randn(B, T, 30, device=device)
        x_meta = torch.randn(B, T, 5, device=device)
        targets = torch.randint(0, 4, (B, T), device=device)
        mask = torch.ones(B, T, dtype=torch.bool, device=device)
        seq_lens = torch.full((B,), T, dtype=torch.int64, device=device)

        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)
        assert "logits" in out
        assert out["logits"].shape == (B, T, 4)
        assert "total_loss" in out
        assert out["total_loss"].item() > 0.0


def test_forward_pass_batch_dict():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_nas_iteration1_model(mutation_type="pre_recurrent").to(device)

    B, T = 4, 100
    batch = {
        "ihr": torch.randn(B, T, 30, device=device),
        "accel": torch.randn(B, T, 30, device=device),
        "meta": torch.randn(B, T, 5, device=device),
        "y": torch.randint(0, 4, (B, T), device=device),
        "mask": torch.ones(B, T, dtype=torch.bool, device=device),
        "subject_idx": torch.zeros(B, dtype=torch.int64, device=device),
        "seq_len": torch.full((B,), T, dtype=torch.int64, device=device),
    }

    out = model(batch)
    assert out["logits"].shape == (B, T, 4)
    assert out["total_loss"].item() > 0.0


def test_gradient_flow():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_nas_iteration1_model(mutation_type="pre_recurrent").to(device)

    B, T = 2, 50
    x_ihr = torch.randn(B, T, 30, device=device, requires_grad=True)
    x_acc = torch.randn(B, T, 30, device=device)
    x_meta = torch.randn(B, T, 5, device=device)
    targets = torch.randint(0, 4, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)

    out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
    loss = out["total_loss"]
    loss.backward()

    # Verify gradients exist across all key mutated layers
    assert model.mstf.proj_in.weight.grad is not None
    assert model.mstf.proj_out.weight.grad is not None
    assert not torch.isnan(model.mstf.proj_in.weight.grad).any()
    for branch in model.mstf.branches:
        assert branch[0].weight.grad is not None


def test_evaluation_results_artifact():
    results_path = os.path.join(EXP014_DIR, "nas_iteration1_eval_results.json")
    assert os.path.exists(results_path), f"Missing results artifact {results_path}"

    with open(results_path, "r") as f:
        data = json.load(f)

    assert data["iteration"] == 1
    assert data["fold"] == 1
    assert data["baseline"]["parameters"] == 163940
    assert data["mutated"]["parameters"] == 192692
    assert data["mutated"]["budget_compliant"] is True
    assert "macro_f1" in data["mutated"]["results"]
    assert data["mutated"]["results"]["macro_f1"] > 0.0
    assert "per_class_f1" in data["mutated"]["results"]
    assert len(data["mutated"]["results"]["per_class_f1"]) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
