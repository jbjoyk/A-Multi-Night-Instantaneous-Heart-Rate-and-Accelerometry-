"""
EXP-031 Pre-Flight Unit & Contract Tests
File: tests/test_exp031_preflight.py
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
EXP031_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-031")
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
for _p in (REPO_ROOT, EXP031_DIR, EXP030_DIR, EXP029_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_exp031_confirmatory import (  # noqa: E402
    compute_pages_l,
    evaluate_confirmatory_tests,
    evaluate_h_arch,
    profile_single_arm_cpu,
    select_backbone_section9,
)
from exp031_models import (  # noqa: E402
    EXP031_ARMS,
    EXP031_ARMS_ORDER,
    EXP031_ELIGIBLE_INT8_ARMS,
    FULL_V5_PARAMS,
    build_exp031_model,
)
from run_exp029_cv import compute_stage_loss_fp32  # noqa: E402
from run_exp031_width_sweep import validate_trainable_arm  # noqa: E402

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
    "arm,expected_params",
    [
        ("W050", 43108),
        ("W075", 93796),
        ("A", 163940),
        ("P190", 190940),
        ("B", 223268),
        ("W125", 253540),
        ("W150", 362596),
        ("W200", 639076),
        ("W250", 993380),
    ],
)
def test_exact_parameter_counts(arm: str, expected_params: int) -> None:
    model = build_exp031_model(arm)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == expected_params, f"{arm}: expected {expected_params}, got {actual}"
    assert EXP031_ARMS[arm]["expected_params"] == expected_params


def test_p190_within_one_percent_of_full_v5() -> None:
    p190_params = sum(p.numel() for p in build_exp031_model("P190").parameters())
    rel_diff = abs(p190_params - FULL_V5_PARAMS) / FULL_V5_PARAMS
    assert rel_diff <= 0.01, f"P190 ({p190_params}) exceeds +-1% of full_v5 ({FULL_V5_PARAMS})"


def test_int8_400kib_eligibility_mask_and_v5_exclusion() -> None:
    eligible = []
    ineligible = []
    for arm in EXP031_ARMS_ORDER:
        prof = profile_single_arm_cpu(arm, benchmark_latency=False)
        assert prof["eligible_int8_400kib"] == EXP031_ARMS[arm]["eligible_int8_400kib"]
        if prof["eligible_int8_400kib"]:
            eligible.append(arm)
        else:
            ineligible.append(arm)
    assert eligible == ["W050", "W075", "A", "P190", "B"]
    assert eligible == EXP031_ELIGIBLE_INT8_ARMS
    assert ineligible == ["W125", "W150", "W200", "W250"]
    assert "V5" not in EXP031_ELIGIBLE_INT8_ARMS
    assert "V5" not in EXP031_ARMS_ORDER


@pytest.mark.parametrize("arm", EXP031_ARMS_ORDER)
def test_forward_contract_and_dynamic_int8(arm: str) -> None:
    model = build_exp031_model(arm).cpu().eval()
    q_model = quantization.quantize_dynamic(
        copy.deepcopy(model), {nn.Linear, nn.GRU}, dtype=torch.qint8
    ).eval()
    x_ihr = torch.randn(1, 40, 30)
    x_acc = torch.randn(1, 40, 30)
    x_meta = torch.randn(1, 40, 5)
    with torch.no_grad():
        out_fp32 = model(x_ihr, x_acc, x_meta)
        out_int8 = q_model(x_ihr, x_acc, x_meta)
    assert set(out_fp32.keys()) == EXPECTED_14_KEYS
    assert out_int8["logits"].shape == (1, 40, 4)
    assert torch.isfinite(out_int8["logits"]).all()


def test_fp32_loss_outside_autocast() -> None:
    model = build_exp031_model("P190", loss_type="rccloss").eval()
    logits_fp16 = torch.randn(2, 24, 4, dtype=torch.float16)
    targets = torch.randint(0, 4, (2, 24))
    mask = torch.ones(2, 24, dtype=torch.bool)
    with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16, enabled=True):
        loss = compute_stage_loss_fp32(model, logits_fp16, targets, mask)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_h_arch_decision_rules() -> None:
    rng = np.random.default_rng(7)
    base = np.full(47, 0.65) + rng.normal(0.0, 0.005, size=47)

    # 1. SUPPORTED: delta >= +0.75 pp AND CI_low > 0
    res_sup = evaluate_h_arch(base + 0.012, base, practical_threshold_pp=0.75, n_boot=2000, seed=42)
    assert res_sup["verdict"] == "SUPPORTED"
    assert res_sup["delta_pp"] >= 0.75 and res_sup["ci95_low_pp"] > 0.0

    # 2. NOT SUPPORTED (practically): CI_high < +0.75 pp
    res_not = evaluate_h_arch(base + 0.002, base, practical_threshold_pp=0.75, n_boot=2000, seed=42)
    assert res_not["verdict"] == "NOT SUPPORTED (practically)"
    assert res_not["ci95_high_pp"] < 0.75

    # 3. INCONCLUSIVE: delta < +0.75 pp (or CI_low <= 0) while CI_high >= +0.75 pp
    noisy = base + np.linspace(-0.02, 0.032, 47)  # mean ~ +0.60 pp, wide CI straddling +0.75 pp
    res_inc = evaluate_h_arch(noisy, base, practical_threshold_pp=0.75, n_boot=2000, seed=42)
    assert res_inc["verdict"] == "INCONCLUSIVE"


def test_directional_pages_l_and_holm_bonferroni() -> None:
    N = 47
    # 1. Strictly increasing across 9 widths -> positive L, p < 0.05 -> SUPPORTED
    inc_mat = np.tile(np.linspace(0.60, 0.68, 9), (N, 1))
    scores_inc = {arm: inc_mat[:, i] for i, arm in enumerate(EXP031_ARMS_ORDER)}
    res_inc = evaluate_confirmatory_tests(
        full_v5_scores=scores_inc["P190"] + 0.001,
        width_scores_ordered=scores_inc,
        n_boot=1000,
        seed=42,
    )
    assert res_inc["h_scale"]["direction_positive"] is True
    assert res_inc["h_scale"]["L"] > res_inc["h_scale"]["mu_L"]
    assert res_inc["h_scale"]["holm_p_two_sided"] < 0.05
    assert res_inc["h_scale"]["verdict"] == "SUPPORTED"

    # 2. Strictly decreasing across 9 widths -> two-sided p < 0.05, but L < mu_L -> NOT SUPPORTED
    dec_mat = np.tile(np.linspace(0.68, 0.60, 9), (N, 1))
    pl_dec = compute_pages_l(dec_mat)
    assert pl_dec["L"] < pl_dec["mu_L"]
    assert pl_dec["z"] < 0.0
    assert pl_dec["p_two_sided"] < 0.05
    scores_dec = {arm: dec_mat[:, i] for i, arm in enumerate(EXP031_ARMS_ORDER)}
    res_dec = evaluate_confirmatory_tests(
        full_v5_scores=scores_dec["P190"] + 0.001,
        width_scores_ordered=scores_dec,
        n_boot=1000,
        seed=42,
    )
    assert res_dec["h_scale"]["verdict"] == "NOT SUPPORTED"


def test_section9_one_se_selection_rule() -> None:
    rng = np.random.default_rng(123)
    N = 47
    # Case 1: W075 is within 1-SE of m*=B, while W050 is far below -> selects W075
    b_scores = np.full(N, 0.666) + rng.normal(0.0, 0.02, N)
    scores_1 = {
        "W050": b_scores - 0.030,
        "W075": b_scores - 0.001 + rng.normal(0.0, 0.015, N),
        "A": b_scores - 0.018,
        "P190": b_scores - 0.005,
        "B": b_scores,
    }
    sel1 = select_backbone_section9(scores_1, n_boot=2000, seed=42)
    assert sel1["m_star"] == "B"
    assert sel1["selected_arm"] == "W075"

    # Case 2: Tie for m* between A and B -> m* breaks toward fewer params (A), selects A
    scores_tie = {
        "W050": np.full(N, 0.620),
        "W075": np.full(N, 0.640),
        "A": np.full(N, 0.665),
        "P190": np.full(N, 0.660),
        "B": np.full(N, 0.665),
    }
    sel2 = select_backbone_section9(scores_tie, n_boot=2000, seed=42)
    assert sel2["m_star"] == "A"
    assert sel2["selected_arm"] == "A"

    # Case 3: Passing non-plain V5 or ineligible W125 must raise
    with pytest.raises((KeyError, ValueError)):
        select_backbone_section9(scores_tie, eligible_arms=["W050", "W125"])


def test_guard_against_retraining_reused_arms() -> None:
    for reused in ("A", "B"):
        with pytest.raises(ValueError):
            validate_trainable_arm(reused)
    validate_trainable_arm("W050")
    validate_trainable_arm("P190")
