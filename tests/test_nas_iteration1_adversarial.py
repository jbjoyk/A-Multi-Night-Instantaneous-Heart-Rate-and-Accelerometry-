"""
Adversarial Stress Test Suite for NAS Iteration 1 Mutated Model
Target Architecture: MultiScaleConventionalCNNBiGRU (Pre-Recurrent MSTF)
File: tests/test_nas_iteration1_adversarial.py

Verification Dimensions:
1. Parameter Budget Edge Cases & Analytical Formula Verification
2. Tensor Shape & Boundary Stress Testing (B in {1, 2, 16}, T in {1, 5, 600, 1000}, seq_lens packing)
3. Gradient Flow & Numerical Stability (all branches k in {1,3,5}, projection layers, residual shortcut)
4. Extreme Input Stress (zeros, +/-1000.0, noise, NaNs/Infs)
5. Mode Switching & Determinism (train vs eval, dropout=0.0)
6. RCCLoss & Mask Boundary Edge Cases
"""

import os
import sys
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root and EXP-014 are in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

EXP014_DIR = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014")
if EXP014_DIR not in sys.path:
    sys.path.insert(0, EXP014_DIR)

from exp013_models import ConventionalCNNBiGRU
from nas_iteration1_model import (
    MultiScaleConventionalCNNBiGRU,
    MultiScaleTemporalFusion,
    MultiScaleTemporalConvBlock,
    build_nas_iteration1_model,
)
from test_rccloss_audit import RCCLoss, get_official_weights


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# SECTION 1: PARAMETER BUDGET & ANALYTICAL FORMULA VERIFICATION
# ==============================================================================

def compute_mstf_params_analytical(
    in_channels: int,
    bottleneck_channels: int,
    kernel_sizes: Tuple[int, ...]
) -> int:
    """
    Analytical parameter formula for MultiScaleTemporalFusion:
      proj_in: in_channels * mid_channels + mid_channels
      branches: sum_k [ bottleneck * 1 * k + bottleneck (conv) + 2 * bottleneck (GroupNorm) ]
      proj_out: mid_channels * in_channels + in_channels
      norm (LayerNorm): 2 * in_channels
    where mid_channels = bottleneck_channels * len(kernel_sizes)
    """
    k_len = len(kernel_sizes)
    mid_channels = bottleneck_channels * k_len
    p_proj_in = in_channels * mid_channels + mid_channels
    p_branches = sum(
        bottleneck_channels * k + bottleneck_channels + 2 * bottleneck_channels
        for k in kernel_sizes
    )
    p_proj_out = mid_channels * in_channels + in_channels
    p_norm = 2 * in_channels
    return p_proj_in + p_branches + p_proj_out + p_norm


def test_baseline_and_mutated_parameter_counts():
    """Verify exact parameter counts of baseline and mutated architectures."""
    baseline = ConventionalCNNBiGRU().to(DEVICE)
    baseline_params = sum(p.numel() for p in baseline.parameters())
    assert baseline_params == 163940, f"Baseline parameters {baseline_params} != 163,940"

    mutated = MultiScaleConventionalCNNBiGRU(mutation_type="pre_recurrent").to(DEVICE)
    mutated_params = sum(p.numel() for p in mutated.parameters())
    assert mutated_params == 192692, f"Mutated parameters {mutated_params} != 192,692"

    cap = 196728  # floor(163,940 * 1.20)
    assert mutated_params <= cap, f"Mutated params {mutated_params} exceeds budget cap {cap}"

    delta = mutated_params - baseline_params
    assert delta == 28752, f"Parameter delta {delta} != +28,752"
    pct_increase = (delta / baseline_params) * 100
    assert abs(pct_increase - 17.5381) < 0.01, f"Percentage increase {pct_increase:.2f}% != 17.54%"


def test_mstf_analytical_formula_grid():
    """Verify analytical formula matches actual PyTorch module across a parameter grid."""
    test_grid = [
        # (in_channels, bottleneck_channels, kernel_sizes)
        (144, 32, (1, 3, 5)),
        (144, 16, (1, 3, 5)),
        (144, 36, (1, 3, 5)),
        (144, 48, (1, 3, 5)),
        (96, 32, (1, 3, 5)),
        (144, 32, (1,)),
        (144, 32, (1, 3)),
        (144, 32, (1, 3, 5, 7)),
        (64, 16, (3, 5)),
    ]
    for c_in, c_b, k_sizes in test_grid:
        block = MultiScaleTemporalFusion(
            in_channels=c_in,
            bottleneck_channels=c_b,
            kernel_sizes=k_sizes
        ).to(DEVICE)
        actual_params = sum(p.numel() for p in block.parameters())
        expected_params = compute_mstf_params_analytical(c_in, c_b, k_sizes)
        assert actual_params == expected_params, (
            f"Formula mismatch for c_in={c_in}, c_b={c_b}, k_sizes={k_sizes}: "
            f"actual {actual_params} != expected {expected_params}"
        )


def test_channel_budget_boundary_cap():
    """
    Empirically verify the discrete channel search space under GroupNorm constraints:
    - cb=36 (multiple of 4): 196,232 params <= 196,728 cap (PASSED).
    - cb=40 (next multiple of 4): 199,772 params > 196,728 cap (VIOLATION).
    - cb=37 (not multiple of 4): raises ValueError due to GroupNorm divisibility constraint.
    """
    cap = 196728

    # 1. cb = 36: Valid multiple of 4, adheres strictly to +20% cap
    m_36 = MultiScaleConventionalCNNBiGRU(mutation_type="pre_recurrent").to(DEVICE)
    m_36.mstf = MultiScaleTemporalFusion(in_channels=144, bottleneck_channels=36, kernel_sizes=(1, 3, 5)).to(DEVICE)
    params_36 = sum(p.numel() for p in m_36.parameters())
    assert params_36 == 196232, f"Params with c_b=36 {params_36} != 196,232"
    assert params_36 <= cap, f"Params with c_b=36 {params_36} should be <= {cap}"

    # 2. cb = 40: Next valid multiple of 4, exceeds cap
    m_40 = MultiScaleConventionalCNNBiGRU(mutation_type="pre_recurrent").to(DEVICE)
    m_40.mstf = MultiScaleTemporalFusion(in_channels=144, bottleneck_channels=40, kernel_sizes=(1, 3, 5)).to(DEVICE)
    params_40 = sum(p.numel() for p in m_40.parameters())
    assert params_40 == 199772, f"Params with c_b=40 {params_40} != 199,772"
    assert params_40 > cap, f"Params with c_b=40 {params_40} should violate cap {cap}"

    # 3. cb = 37: PyTorch GroupNorm raises ValueError because 37 % 4 != 0
    with pytest.raises(ValueError, match=r"must be divisible by num_groups"):
        MultiScaleTemporalFusion(in_channels=144, bottleneck_channels=37, kernel_sizes=(1, 3, 5))


def test_even_kernel_size_dimension_incompatibility():
    """
    Verify that even kernel sizes (e.g. k=4) break sequence length invariance T_out == T_in,
    proving that MSTF strictly requires odd kernel sizes.
    """
    # k=4 with padding=2 gives T_out = T + 2*2 - 4 + 1 = T + 1
    mstf_even = MultiScaleTemporalFusion(in_channels=144, bottleneck_channels=32, kernel_sizes=(4,)).to(DEVICE)
    x = torch.randn(2, 60, 144, device=DEVICE)
    with pytest.raises(RuntimeError):
        # res + out will fail due to (2, 60, 144) vs (2, 61, 144)
        mstf_even(x)


def test_intra_encoder_parameter_budget():
    """Verify parameter count for the alternative intra-encoder mutation."""
    m_intra = MultiScaleConventionalCNNBiGRU(mutation_type="intra_encoder").to(DEVICE)
    p_intra = sum(p.numel() for p in m_intra.parameters())
    assert p_intra == 173284, f"Intra-encoder params {p_intra} != 173,284"
    assert p_intra <= 196728, f"Intra-encoder params {p_intra} exceeds cap"
    delta = p_intra - 163940
    assert delta == 9344, f"Intra-encoder delta {delta} != +9,344"


# ==============================================================================
# SECTION 2: TENSOR SHAPE & BOUNDARY STRESS TESTING
# ==============================================================================

@pytest.mark.parametrize("B", [1, 2, 16])
def test_extreme_batch_sizes(B: int):
    """Stress test extreme batch sizes (B=1, B=2, B=16) on GPU cuda:0."""
    T = 60
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.eval()

    x_ihr = torch.randn(B, T, 30, device=DEVICE)
    x_acc = torch.randn(B, T, 30, device=DEVICE)
    x_meta = torch.randn(B, T, 5, device=DEVICE)
    targets = torch.randint(0, 4, (B, T), device=DEVICE)
    mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
    seq_lens = torch.full((B,), T, dtype=torch.int64, device=DEVICE)

    with torch.no_grad():
        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)

    logits = out["logits"]
    assert logits.shape == (B, T, 4), f"Shape mismatch: {logits.shape} != {(B, T, 4)}"
    assert not torch.isnan(logits).any(), "NaN detected in logits"
    assert not torch.isinf(logits).any(), "Inf detected in logits"
    assert out["total_loss"].item() > 0.0, "Total loss <= 0.0"


@pytest.mark.parametrize("T", [1, 5, 600, 1000])
def test_extreme_sequence_lengths(T: int):
    """Stress test extreme sequence lengths (T=1, T=5, T=600, T=1000) on GPU cuda:0."""
    B = 2
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.eval()

    x_ihr = torch.randn(B, T, 30, device=DEVICE)
    x_acc = torch.randn(B, T, 30, device=DEVICE)
    x_meta = torch.randn(B, T, 5, device=DEVICE)
    targets = torch.randint(0, 4, (B, T), device=DEVICE)
    mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)
    seq_lens = torch.full((B,), T, dtype=torch.int64, device=DEVICE)

    with torch.no_grad():
        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)

    logits = out["logits"]
    assert logits.shape == (B, T, 4), f"Shape mismatch for T={T}: {logits.shape} != {(B, T, 4)}"
    assert not torch.isnan(logits).any(), f"NaN in logits for T={T}"
    assert not torch.isinf(logits).any(), f"Inf in logits for T={T}"


def test_joint_extreme_batch1_seqlen1():
    """Stress test simultaneous joint extremes: B=1 and T=1."""
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.eval()

    x_ihr = torch.randn(1, 1, 30, device=DEVICE)
    x_acc = torch.randn(1, 1, 30, device=DEVICE)
    x_meta = torch.randn(1, 1, 5, device=DEVICE)
    targets = torch.tensor([[1]], device=DEVICE)

    with torch.no_grad():
        out = model(x_ihr, x_acc, x_meta, targets=targets)

    assert out["logits"].shape == (1, 1, 4)
    assert not torch.isnan(out["logits"]).any()
    assert not torch.isnan(out["total_loss"])


def test_heterogeneous_sequence_lengths_with_packing():
    """Verify variable sequence lengths in batch with pack_padded_sequence."""
    B = 4
    T_max = 600
    lengths = [600, 450, 150, 10]
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.train()

    x_ihr = torch.randn(B, T_max, 30, device=DEVICE)
    x_acc = torch.randn(B, T_max, 30, device=DEVICE)
    x_meta = torch.randn(B, T_max, 5, device=DEVICE)
    targets = torch.randint(0, 4, (B, T_max), device=DEVICE)

    # Mask according to lengths
    mask = torch.zeros(B, T_max, dtype=torch.bool, device=DEVICE)
    for b, l in enumerate(lengths):
        mask[b, :l] = True

    seq_lens = torch.tensor(lengths, dtype=torch.int64, device=DEVICE)

    out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)
    loss = out["total_loss"]
    assert not torch.isnan(loss), "Loss is NaN with heterogeneous sequence lengths"
    loss.backward()

    # Verify gradients exist and are non-zero
    assert model.mstf.proj_in.weight.grad is not None
    assert model.mstf.proj_in.weight.grad.norm().item() > 0.0


def test_padding_isolation_and_receptive_field_vs_groupnorm():
    """
    Adversarial Boundary Analysis:
    1. Pure Conv1d(k=5, p=2) has exact spatial locality:
       Outputs at t < valid_len - 2 are 100% identical (diff = 0.0) regardless of padding.
    2. GroupNorm(4, 32) in MSTF pools statistics across the temporal dimension T:
       Consequently, padding values at t >= valid_len shift the GroupNorm mean and variance,
       introducing global temporal coupling across padded tokens.
    """
    B = 1
    T = 100
    valid_len = 50

    x_valid = torch.randn(B, 32, valid_len, device=DEVICE)
    x_pad0 = torch.cat([x_valid, torch.zeros(B, 32, T - valid_len, device=DEVICE)], dim=-1)
    x_pad100 = torch.cat([x_valid, torch.full((B, 32, T - valid_len), 100.0, device=DEVICE)], dim=-1)

    # 1. Pure Conv1d (k=5, p=2) test
    conv = nn.Conv1d(32, 32, kernel_size=5, padding=2, groups=32).to(DEVICE)
    with torch.no_grad():
        y0_conv = conv(x_pad0)
        y100_conv = conv(x_pad100)

    conv_interior_diff = (y0_conv[:, :, :valid_len - 2] - y100_conv[:, :, :valid_len - 2]).abs().max().item()
    assert conv_interior_diff == 0.0, f"Conv1d leaked beyond k=5 boundary! Diff: {conv_interior_diff}"

    conv_boundary_diff = (y0_conv[:, :, valid_len - 2:valid_len] - y100_conv[:, :, valid_len - 2:valid_len]).abs().max().item()
    assert conv_boundary_diff > 0.0, "Expected boundary position to reflect padding"

    # 2. GroupNorm(4, 32) global temporal pooling test
    gn = nn.GroupNorm(4, 32).to(DEVICE)
    with torch.no_grad():
        y0_gn = gn(x_pad0)
        y100_gn = gn(x_pad100)

    gn_interior_diff = (y0_gn[:, :, :valid_len - 2] - y100_gn[:, :, :valid_len - 2]).abs().max().item()
    # Empirically confirms that GroupNorm pools across T, coupling padded tokens to interior positions
    assert gn_interior_diff > 0.1, f"Expected GroupNorm temporal pooling coupling > 0.1, got {gn_interior_diff:.2e}"


# ==============================================================================
# SECTION 3: EXTREME INPUT VALUES STRESS TESTING
# ==============================================================================

def test_extreme_inputs_all_zeros():
    """Verify numerical stability when all inputs are exactly 0.0."""
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.eval()

    B, T = 2, 60
    x_ihr = torch.zeros(B, T, 30, device=DEVICE)
    x_acc = torch.zeros(B, T, 30, device=DEVICE)
    x_meta = torch.zeros(B, T, 5, device=DEVICE)
    targets = torch.randint(0, 4, (B, T), device=DEVICE)

    with torch.no_grad():
        out = model(x_ihr, x_acc, x_meta, targets=targets)

    logits = out["logits"]
    assert not torch.isnan(logits).any(), "NaN in logits with all-zero input"
    assert not torch.isinf(logits).any(), "Inf in logits with all-zero input"
    probs = out["probabilities"]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B, T, device=DEVICE), atol=1e-5)
    assert not torch.isnan(out["total_loss"]), "Loss is NaN with all-zero input"


@pytest.mark.parametrize("extreme_val", [1000.0, -1000.0])
def test_extreme_inputs_large_magnitude(extreme_val: float):
    """Verify numerical stability under extreme magnitude inputs (+/- 1000.0)."""
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.eval()

    B, T = 2, 60
    x_ihr = torch.full((B, T, 30), extreme_val, device=DEVICE)
    x_acc = torch.full((B, T, 30), extreme_val, device=DEVICE)
    x_meta = torch.full((B, T, 5), extreme_val, device=DEVICE)
    targets = torch.randint(0, 4, (B, T), device=DEVICE)

    with torch.no_grad():
        out = model(x_ihr, x_acc, x_meta, targets=targets)

    logits = out["logits"]
    assert not torch.isnan(logits).any(), f"NaN in logits with extreme val {extreme_val}"
    assert not torch.isinf(logits).any(), f"Inf in logits with extreme val {extreme_val}"
    probs = out["probabilities"]
    assert torch.allclose(probs.sum(dim=-1), torch.ones(B, T, device=DEVICE), atol=1e-5)


def test_extreme_inputs_high_variance_noise():
    """Verify numerical stability with large variance Gaussian noise (std=100.0)."""
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.eval()

    B, T = 2, 60
    torch.manual_seed(42)
    x_ihr = torch.randn(B, T, 30, device=DEVICE) * 100.0
    x_acc = torch.randn(B, T, 30, device=DEVICE) * 100.0
    x_meta = torch.randn(B, T, 5, device=DEVICE) * 100.0
    targets = torch.randint(0, 4, (B, T), device=DEVICE)

    with torch.no_grad():
        out = model(x_ihr, x_acc, x_meta, targets=targets)

    logits = out["logits"]
    assert not torch.isnan(logits).any(), "NaN in logits with std=100.0 noise"
    assert not torch.isinf(logits).any(), "Inf in logits with std=100.0 noise"


def test_nan_and_inf_input_behavior():
    """Verify that NaN / Inf inputs propagate cleanly without crashing Python/CUDA."""
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.eval()

    B, T = 2, 10
    x_ihr = torch.randn(B, T, 30, device=DEVICE)
    x_acc = torch.randn(B, T, 30, device=DEVICE)
    x_meta = torch.randn(B, T, 5, device=DEVICE)

    # Inject single NaN and single Inf
    x_ihr[0, 0, 0] = float("nan")
    x_acc[1, 5, 0] = float("inf")

    with torch.no_grad():
        out = model(x_ihr, x_acc, x_meta)

    # PyTorch expected behavior: NaNs/Infs propagate to logits without C++ aborts
    logits = out["logits"]
    assert torch.isnan(logits[0, 0]).any(), "NaN did not propagate to target position"


# ==============================================================================
# SECTION 4: GRADIENT FLOW & NUMERICAL STABILITY
# ==============================================================================

def test_gradient_propagation_all_parallel_branches():
    """
    Verify non-zero, finite gradients propagate through:
    - Parallel branches k in {1, 3, 5} (weights and biases of Conv1d and GroupNorm)
    - Projection layers (proj_in, proj_out)
    - Normalization layer (norm)
    - Residual shortcut
    """
    model = MultiScaleConventionalCNNBiGRU(mutation_type="pre_recurrent").to(DEVICE)
    model.train()

    B, T = 2, 60
    x_ihr = torch.randn(B, T, 30, device=DEVICE)
    x_acc = torch.randn(B, T, 30, device=DEVICE)
    x_meta = torch.randn(B, T, 5, device=DEVICE)
    targets = torch.randint(0, 4, (B, T), device=DEVICE)
    mask = torch.ones(B, T, dtype=torch.bool, device=DEVICE)

    out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
    loss = out["total_loss"]
    loss.backward()

    mstf = model.mstf

    # 1. Bottleneck projection
    assert mstf.proj_in.weight.grad is not None
    assert mstf.proj_in.weight.grad.norm().item() > 0.0
    assert mstf.proj_in.bias.grad is not None
    assert mstf.proj_in.bias.grad.norm().item() > 0.0

    # 2. Parallel branches k in {1, 3, 5}
    for idx, (k, branch) in enumerate(zip([1, 3, 5], mstf.branches)):
        conv_w_grad = branch[0].weight.grad
        conv_b_grad = branch[0].bias.grad
        gn_w_grad = branch[1].weight.grad
        gn_b_grad = branch[1].bias.grad

        assert conv_w_grad is not None, f"Branch k={k} conv weight grad is None"
        assert conv_w_grad.norm().item() > 0.0, f"Branch k={k} conv weight grad is 0.0"
        assert not torch.isnan(conv_w_grad).any(), f"Branch k={k} conv weight grad is NaN"

        assert conv_b_grad is not None, f"Branch k={k} conv bias grad is None"
        assert conv_b_grad.norm().item() > 0.0, f"Branch k={k} conv bias grad is 0.0"

        assert gn_w_grad is not None, f"Branch k={k} GN weight grad is None"
        assert gn_w_grad.norm().item() > 0.0, f"Branch k={k} GN weight grad is 0.0"

        assert gn_b_grad is not None, f"Branch k={k} GN bias grad is None"
        assert gn_b_grad.norm().item() > 0.0, f"Branch k={k} GN bias grad is 0.0"

    # 3. Restoration projection
    assert mstf.proj_out.weight.grad is not None
    assert mstf.proj_out.weight.grad.norm().item() > 0.0
    assert mstf.proj_out.bias.grad is not None
    assert mstf.proj_out.bias.grad.norm().item() > 0.0

    # 4. Residual normalization
    assert mstf.norm.weight.grad is not None
    assert mstf.norm.weight.grad.norm().item() > 0.0
    assert mstf.norm.bias.grad is not None
    assert mstf.norm.bias.grad.norm().item() > 0.0


def test_residual_shortcut_gradient_flow():
    """
    Directly verify that the identity residual shortcut in MultiScaleTemporalFusion
    actively passes gradient back to input features independently of the conv path.
    """
    mstf = MultiScaleTemporalFusion(in_channels=144, bottleneck_channels=32, kernel_sizes=(1, 3, 5)).to(DEVICE)
    x = torch.randn(2, 60, 144, device=DEVICE, requires_grad=True)

    out = mstf(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None
    grad_norm = x.grad.norm().item()
    assert grad_norm > 0.0, "Input x received zero gradient"
    assert not torch.isnan(x.grad).any(), "Input gradient is NaN"


def test_gradient_dynamic_range():
    """Verify that gradients across all model parameters are neither vanishing nor exploding."""
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.train()

    B, T = 4, 100
    x_ihr = torch.randn(B, T, 30, device=DEVICE)
    x_acc = torch.randn(B, T, 30, device=DEVICE)
    x_meta = torch.randn(B, T, 5, device=DEVICE)
    targets = torch.randint(0, 4, (B, T), device=DEVICE)

    out = model(x_ihr, x_acc, x_meta, targets=targets)
    out["total_loss"].backward()

    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm().item()
            grad_norms.append((name, norm))

    assert len(grad_norms) > 0, "No gradients computed"
    min_norm = min(n for _, n in grad_norms)
    max_norm = max(n for _, n in grad_norms)

    # Dynamic range sanity check: gradients must be > 1e-7 and < 1e3
    assert min_norm > 1e-7, f"Vanishing gradient detected! Min norm: {min_norm:.2e}"
    assert max_norm < 1e3, f"Exploding gradient detected! Max norm: {max_norm:.2e}"


# ==============================================================================
# SECTION 5: MODE SWITCHING & DETERMINISM
# ==============================================================================

def test_eval_mode_determinism():
    """Verify that model.eval() produces 100% bitwise deterministic outputs on identical inputs."""
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.eval()

    B, T = 2, 60
    x_ihr = torch.randn(B, T, 30, device=DEVICE)
    x_acc = torch.randn(B, T, 30, device=DEVICE)
    x_meta = torch.randn(B, T, 5, device=DEVICE)

    with torch.no_grad():
        out1 = model(x_ihr, x_acc, x_meta)
        out2 = model(x_ihr, x_acc, x_meta)

    # Exact bitwise equality check
    assert torch.equal(out1["logits"], out2["logits"]), "eval() mode forward pass is not deterministic!"
    assert torch.equal(out1["probabilities"], out2["probabilities"]), "Probabilities not deterministic in eval()"


def test_train_mode_dropout_regularization():
    """Verify that model.train() with default dropout produces stochastic outputs."""
    model = MultiScaleConventionalCNNBiGRU(dropout=0.2).to(DEVICE)
    model.train()

    B, T = 2, 60
    x_ihr = torch.randn(B, T, 30, device=DEVICE)
    x_acc = torch.randn(B, T, 30, device=DEVICE)
    x_meta = torch.randn(B, T, 5, device=DEVICE)

    with torch.no_grad():
        out1 = model(x_ihr, x_acc, x_meta)
        out2 = model(x_ihr, x_acc, x_meta)

    # Should NOT be bitwise equal due to active dropout
    assert not torch.equal(out1["logits"], out2["logits"]), "train() mode output is deterministic despite dropout=0.2"


def test_train_mode_zero_dropout_determinism():
    """Verify that model.train() with dropout=0.0 produces identical deterministic outputs."""
    model = MultiScaleConventionalCNNBiGRU(dropout=0.0).to(DEVICE)
    model.train()

    B, T = 2, 60
    x_ihr = torch.randn(B, T, 30, device=DEVICE)
    x_acc = torch.randn(B, T, 30, device=DEVICE)
    x_meta = torch.randn(B, T, 5, device=DEVICE)

    with torch.no_grad():
        out1 = model(x_ihr, x_acc, x_meta)
        out2 = model(x_ihr, x_acc, x_meta)

    assert torch.allclose(out1["logits"], out2["logits"], atol=1e-6), (
        "train() mode with dropout=0.0 produced non-deterministic outputs"
    )


# ==============================================================================
# SECTION 6: RCCLOSS & MASK BOUNDARY EDGE CASES
# ==============================================================================

@pytest.mark.parametrize("target_class", [0, 1, 2, 3])
def test_rccloss_single_class_batch(target_class: int):
    """Stress test RCCLoss and composite loss when the entire batch is one single class."""
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.train()

    B, T = 2, 60
    x_ihr = torch.randn(B, T, 30, device=DEVICE)
    x_acc = torch.randn(B, T, 30, device=DEVICE)
    x_meta = torch.randn(B, T, 5, device=DEVICE)
    targets = torch.full((B, T), target_class, dtype=torch.int64, device=DEVICE)

    out = model(x_ihr, x_acc, x_meta, targets=targets)
    loss = out["total_loss"]
    assert not torch.isnan(loss), f"Loss is NaN for single-class targets class={target_class}"
    loss.backward()
    assert model.classifier_head[0].weight.grad is not None


def test_zero_valid_mask_batch():
    """Verify that when mask is all zeros (no valid epochs), loss is 0.0 without crash."""
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.eval()

    B, T = 2, 60
    x_ihr = torch.randn(B, T, 30, device=DEVICE)
    x_acc = torch.randn(B, T, 30, device=DEVICE)
    x_meta = torch.randn(B, T, 5, device=DEVICE)
    targets = torch.randint(0, 4, (B, T), device=DEVICE)
    mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)

    assert out["total_loss"].item() == 0.0, f"Expected loss 0.0 for all-masked batch, got {out['total_loss'].item()}"


def test_single_valid_epoch():
    """Verify that when mask has only 1 valid epoch in the entire batch, loss is stable."""
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.train()

    B, T = 2, 60
    x_ihr = torch.randn(B, T, 30, device=DEVICE)
    x_acc = torch.randn(B, T, 30, device=DEVICE)
    x_meta = torch.randn(B, T, 5, device=DEVICE)
    targets = torch.randint(0, 4, (B, T), device=DEVICE)
    mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)
    mask[0, 0] = True  # exactly one valid epoch

    out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
    loss = out["total_loss"]
    assert not torch.isnan(loss), "Loss is NaN for single-valid-epoch batch"
    assert loss.item() > 0.0, "Loss <= 0.0 for single-valid-epoch batch"
    loss.backward()
    assert model.temporal_gru.weight_ih_l0.grad is not None


def test_negative_target_clamping_safety():
    """Verify that padded targets (-1 or -100) do not cause CUDA assertion crashes."""
    model = MultiScaleConventionalCNNBiGRU().to(DEVICE)
    model.train()

    B, T = 2, 30
    x_ihr = torch.randn(B, T, 30, device=DEVICE)
    x_acc = torch.randn(B, T, 30, device=DEVICE)
    x_meta = torch.randn(B, T, 5, device=DEVICE)

    targets = torch.full((B, T), -100, dtype=torch.int64, device=DEVICE)
    targets[0, :10] = torch.randint(0, 4, (10,), device=DEVICE)
    mask = torch.zeros(B, T, dtype=torch.bool, device=DEVICE)
    mask[0, :10] = True

    out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask)
    loss = out["total_loss"]
    assert not torch.isnan(loss), "Loss is NaN with negative target padding"
    loss.backward()
    assert model.classifier_head[0].weight.grad is not None


# ==============================================================================
# STANDALONE RUNNER
# ==============================================================================

def run_adversarial_suite():
    print("=" * 80)
    print(f"ADVERSARIAL STRESS TEST SUITE ON GPU: {DEVICE}")
    print(f"Device Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    tests = [
        ("1. Baseline & Mutated Parameter Counts", test_baseline_and_mutated_parameter_counts),
        ("2. MSTF Analytical Formula Grid", test_mstf_analytical_formula_grid),
        ("3. Channel Budget Boundary Cap (cb=36 vs cb=40 vs cb=37)", test_channel_budget_boundary_cap),
        ("4. Even Kernel Size Incompatibility (Rejection)", test_even_kernel_size_dimension_incompatibility),
        ("5. Intra-Encoder Parameter Budget", test_intra_encoder_parameter_budget),
        ("6. Extreme Batch Sizes (B=1)", lambda: test_extreme_batch_sizes(1)),
        ("7. Extreme Batch Sizes (B=2)", lambda: test_extreme_batch_sizes(2)),
        ("8. Extreme Batch Sizes (B=16)", lambda: test_extreme_batch_sizes(16)),
        ("9. Extreme Sequence Length (T=1)", lambda: test_extreme_sequence_lengths(1)),
        ("10. Extreme Sequence Length (T=5)", lambda: test_extreme_sequence_lengths(5)),
        ("11. Extreme Sequence Length (T=600)", lambda: test_extreme_sequence_lengths(600)),
        ("12. Extreme Sequence Length (T=1000)", lambda: test_extreme_sequence_lengths(1000)),
        ("13. Joint Extremes (B=1, T=1)", test_joint_extreme_batch1_seqlen1),
        ("14. Heterogeneous Sequence Lengths with Packing", test_heterogeneous_sequence_lengths_with_packing),
        ("15. Padding Isolation & Receptive Field vs GroupNorm", test_padding_isolation_and_receptive_field_vs_groupnorm),
        ("16. Extreme Inputs: All Zeros", test_extreme_inputs_all_zeros),
        ("17. Extreme Inputs: Large Positive (+1000.0)", lambda: test_extreme_inputs_large_magnitude(1000.0)),
        ("18. Extreme Inputs: Large Negative (-1000.0)", lambda: test_extreme_inputs_large_magnitude(-1000.0)),
        ("19. Extreme Inputs: High Variance Noise (std=100.0)", test_extreme_inputs_high_variance_noise),
        ("20. NaN / Inf Input Behavior", test_nan_and_inf_input_behavior),
        ("21. Gradient Flow: All Parallel Branches", test_gradient_propagation_all_parallel_branches),
        ("22. Residual Shortcut Gradient Flow", test_residual_shortcut_gradient_flow),
        ("23. Gradient Dynamic Range (No Vanish / Explode)", test_gradient_dynamic_range),
        ("24. Eval Mode Bitwise Determinism", test_eval_mode_determinism),
        ("25. Train Mode Dropout Stochasticity", test_train_mode_dropout_regularization),
        ("26. Train Mode Zero Dropout Determinism", test_train_mode_zero_dropout_determinism),
        ("27. Single Class Batch (Wake=0)", lambda: test_rccloss_single_class_batch(0)),
        ("28. Single Class Batch (Light=1)", lambda: test_rccloss_single_class_batch(1)),
        ("29. Single Class Batch (Deep=2)", lambda: test_rccloss_single_class_batch(2)),
        ("30. Single Class Batch (REM=3)", lambda: test_rccloss_single_class_batch(3)),
        ("31. Zero Valid Mask Batch Handling", test_zero_valid_mask_batch),
        ("32. Single Valid Epoch in Batch", test_single_valid_epoch),
        ("33. Negative Target Clamping Safety", test_negative_target_clamping_safety),
    ]

    all_passed = True
    passed_count = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  [PASS] {name}")
            passed_count += 1
        except Exception as e:
            print(f"  [FAIL] {name} -> {e}")
            all_passed = False

    print("=" * 80)
    print(f"ADVERSARIAL SUITE SUMMARY: {passed_count}/{len(tests)} PASSED | ALL PASSED: {all_passed}")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_adversarial_suite()
    sys.exit(0 if success else 1)
