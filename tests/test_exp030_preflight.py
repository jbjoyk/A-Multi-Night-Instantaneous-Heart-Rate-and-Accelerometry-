"""
EXP-030 Pre-Flight Unit & Contract Tests
File: tests/test_exp030_preflight.py
"""

from __future__ import annotations

import copy
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.ao.quantization as quantization

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
for _p in (REPO_ROOT, EXP030_DIR, EXP029_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_exp030_downstream import (
    clone_scaled_model_with_state,
    compute_model_mmacs,
    select_tau_star_nested,
)
from exp013_models import ConventionalCNNBiGRU
from exp030_models import EXP030_GRID, EXP030_GRID_ORDER, build_exp030_model
from run_exp029_cv import compute_stage_loss_fp32
from select_exp030_elbow import select_elbow_from_scores

EXPECTED_14_KEYS = {
    "logits",
    "emission_logits",
    "probabilities",
    "alpha_gate",
    "confidence",
    "entropy",
    "total_loss",
    "loss_stage",
    "loss_trans",
    "loss_boundary",
    "loss_adv",
    "loss_conf",
    "boundary_mask",
    "temp_features",
}


@pytest.mark.parametrize(
    "tag,expected_params",
    [
        ("p1_100k", 98940),
        ("p2_164k", 163940),
        ("p3_268k", 268260),
        ("p4_404k", 403524),
        ("p5_666k", 665556),
        ("p6_1051k", 1051492),
    ],
)
def test_exact_parameter_counts(tag: str, expected_params: int) -> None:
    model = build_exp030_model(tag)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == expected_params, f"{tag}: expected {expected_params}, got {actual}"
    assert EXP030_GRID[tag]["expected_params"] == expected_params


def test_p2_164k_state_dict_and_forward_parity_with_model_a() -> None:
    torch.manual_seed(42)
    ref_model = ConventionalCNNBiGRU().eval()
    p2_model = build_exp030_model("p2_164k").eval()

    ref_sd = ref_model.state_dict()
    p2_sd = p2_model.state_dict()

    assert list(ref_sd.keys()) == list(p2_sd.keys())
    for k in ref_sd:
        assert ref_sd[k].shape == p2_sd[k].shape, f"Shape mismatch at {k}"
        assert ref_sd[k].dtype == p2_sd[k].dtype, f"Dtype mismatch at {k}"

    p2_model.load_state_dict(ref_sd, strict=True)

    torch.manual_seed(123)
    x_ihr = torch.randn(2, 40, 30)
    x_acc = torch.randn(2, 40, 30)
    x_meta = torch.randn(2, 40, 5)
    targets = torch.randint(0, 4, (2, 40))
    seq_lens = torch.tensor([40, 25], dtype=torch.int64)

    with torch.no_grad():
        out_ref = ref_model(x_ihr, x_acc, x_meta, targets=targets, seq_lens=seq_lens)
        out_p2 = p2_model(x_ihr, x_acc, x_meta, targets=targets, seq_lens=seq_lens)

    assert set(out_p2.keys()) == EXPECTED_14_KEYS
    assert set(out_ref.keys()) == EXPECTED_14_KEYS
    assert torch.allclose(out_ref["logits"], out_p2["logits"], atol=0.0, rtol=0.0)
    assert torch.allclose(out_ref["probabilities"], out_p2["probabilities"], atol=0.0, rtol=0.0)
    assert torch.allclose(out_ref["total_loss"], out_p2["total_loss"], atol=0.0, rtol=0.0)


def test_seq_lens_padding_invariance_and_none_fallback() -> None:
    torch.manual_seed(7)
    model = build_exp030_model("p1_100k").eval()
    B, T = 2, 32
    x_ihr = torch.randn(B, T, 30)
    x_acc = torch.randn(B, T, 30)
    x_meta = torch.randn(B, T, 5)
    seq_lens = torch.tensor([20, 32], dtype=torch.int64)

    # Perturb padded tail of sample 0 beyond epoch 20
    x_ihr_pert = x_ihr.clone()
    x_acc_pert = x_acc.clone()
    x_meta_pert = x_meta.clone()
    x_ihr_pert[0, 20:] = 999.0
    x_acc_pert[0, 20:] = -999.0
    x_meta_pert[0, 20:] = 123.0

    with torch.no_grad():
        out1 = model(x_ihr, x_acc, x_meta, seq_lens=seq_lens)
        out2 = model(x_ihr_pert, x_acc_pert, x_meta_pert, seq_lens=seq_lens)
        out_none = model(x_ihr, x_acc, x_meta, seq_lens=None)

    assert set(out_none.keys()) == EXPECTED_14_KEYS
    assert out_none["logits"].shape == (B, T, 4)
    # Valid positions [0, :20] must be invariant to padded tail
    assert torch.allclose(out1["logits"][0, :20], out2["logits"][0, :20], atol=1e-6)
    # Sample 1 (full length 32) must match seq_lens=None
    assert torch.allclose(out1["logits"][1], out_none["logits"][1], atol=1e-6)


def test_fp32_loss_outside_autocast() -> None:
    model = build_exp030_model("p3_268k", loss_type="rccloss").eval()
    logits_fp16 = torch.randn(2, 24, 4, dtype=torch.float16)
    targets = torch.randint(0, 4, (2, 24))
    targets[0, 20:] = -100
    mask = torch.ones(2, 24, dtype=torch.bool)
    mask[0, 20:] = False

    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        loss = compute_stage_loss_fp32(model, logits_fp16, targets, mask)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


@pytest.mark.parametrize("tag", EXP030_GRID_ORDER)
def test_dynamic_int8_quantization_compatibility(tag: str) -> None:
    model = build_exp030_model(tag).cpu().eval()
    q_model = quantization.quantize_dynamic(
        copy.deepcopy(model), {nn.Linear, nn.GRU}, dtype=torch.qint8
    ).eval()
    x_ihr = torch.randn(1, 1200, 30)
    x_acc = torch.randn(1, 1200, 30)
    x_meta = torch.randn(1, 1200, 5)
    with torch.no_grad():
        out = q_model(x_ihr, x_acc, x_meta)
    assert out["logits"].shape == (1, 1200, 4)
    assert torch.isfinite(out["logits"]).all()

    mmacs = compute_model_mmacs(model, seq_len=1200, epoch_len=30)
    assert mmacs > 100.0


def test_elbow_selection_rule_and_degenerate_guard() -> None:
    # Case 1: Degenerate flat sweep (delta_max <= 0.50 pp) -> p2_164k
    flat_scores = {
        "p1_100k": 64.10,
        "p2_164k": 64.80,
        "p3_268k": 65.10,
        "p4_404k": 65.25,  # +0.45 pp <= 0.50 pp
        "p5_666k": 65.20,
        "p6_1051k": 65.05,
    }
    res_flat = select_elbow_from_scores(flat_scores)
    assert res_flat["guard_triggered"] is True
    assert res_flat["star_tag"] == "p2_164k"
    assert res_flat["star_params"] == 163940
    assert res_flat["threshold_T"] is None

    # Case 2: Clear capacity knee -> smallest params meeting T = max(M_p2 + 0.9*delta, M_max - 0.20)
    knee_scores = {
        "p1_100k": 63.00,
        "p2_164k": 64.80,
        "p3_268k": 66.85,  # M_max=67.00 -> T=max(64.80+1.98=66.78, 66.80)=66.80 -> p3_268k qualifies!
        "p4_404k": 66.95,
        "p5_666k": 67.00,
        "p6_1051k": 66.90,
    }
    res_knee = select_elbow_from_scores(knee_scores)
    assert res_knee["guard_triggered"] is False
    assert np.isclose(res_knee["m_max"], 67.00)
    assert np.isclose(res_knee["delta_max"], 2.20)
    assert np.isclose(res_knee["threshold_T"], 66.80)
    assert res_knee["star_tag"] == "p3_268k"
    assert res_knee["star_params"] == 268260


def test_ft2_state_dict_clone_isolation_and_tau_tie_breaking() -> None:
    base_model = build_exp030_model("p3_268k").cpu()
    base_state = copy.deepcopy(base_model.state_dict())
    base_snapshot = {k: v.clone() for k, v in base_state.items()}

    cloned, trainable = clone_scaled_model_with_state(
        "p3_268k", base_state, torch.device("cpu")
    )
    assert len(trainable) > 0
    for name, param in cloned.named_parameters():
        if any(tok in name for tok in ("classifier_head", "temp_norm", "temporal_gru")):
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False

    # Mutate cloned trainable weights and verify base_state is untouched
    with torch.no_grad():
        for p in trainable:
            p.add_(1.0)
    for k in base_snapshot:
        assert torch.equal(base_state[k], base_snapshot[k]), f"base_state mutated at {k}"

    # Verify nested tau* selection tie-breaking (4-decimal round, smallest tau)
    subs = ["s1", "s2", "s3"]
    zs = {"s1": 0.60000, "s2": 0.65000, "s3": 0.70000}
    ft2 = {"s1": 0.55000, "s2": 0.68000, "s3": 0.72000}
    min_frac = {"s1": 0.005, "s2": 0.015, "s3": 0.060}
    # At tau=0.01: s1 abstains (0.60), s2/s3 adapt (0.68, 0.72) -> mean = 0.6667
    # At tau=0.02: s1, s2 abstain (0.60, 0.65), s3 adapts (0.72) -> mean = 0.6567
    tau_star, tie_used = select_tau_star_nested(subs, zs, ft2, min_frac)
    assert tau_star == 0.01
    assert tie_used is False
