"""
EXP-029 Pre-Flight Unit & Contract Test Suite
File: tests/test_exp029_preflight.py
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
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
for _p in (REPO_ROOT, EXP029_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp013_models import ConventionalCNNBiGRU
from exp029_calibration import DampedEMCalibrator, LogitStitchingSlidingWindowPredictor
from exp029_models import IntraEpochFiLM, UltradianPyramidalCoupler, build_exp029_model
from run_exp029_cv import compute_stage_loss_fp32


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


def test_exact_parameter_counts():
    m_a = ConventionalCNNBiGRU(d_model=64, d_temp=96, n_classes=4, n_meta=5, dropout=0.2)
    m_b = build_exp029_model("cnn_bigru_200k")
    m_c = build_exp029_model("film_only")
    m_d = build_exp029_model("pyramid_only")
    m_e = build_exp029_model("full_v5")

    assert sum(p.numel() for p in m_a.parameters()) == 163_940
    assert sum(p.numel() for p in m_b.parameters()) == 223_268
    assert sum(p.numel() for p in m_c.parameters()) == 172_260
    assert sum(p.numel() for p in m_d.parameters()) == 182_468
    assert sum(p.numel() for p in m_e.parameters()) == 190_788


def test_zero_init_film_identity():
    film = IntraEpochFiLM(d_model=64)
    assert torch.all(film.film_proj.weight == 0.0)
    assert torch.all(film.film_proj.bias == 0.0)

    h_hr = torch.randn(4, 64, 7)
    h_acc = torch.randn(4, 64, 7) * 5.0
    out = film(h_hr, h_acc)
    assert torch.allclose(out, h_hr, atol=1e-7)


@pytest.mark.parametrize("arm_id", ["film_only", "pyramid_only", "full_v5"])
def test_seq_lens_padding_invariance_and_none_fallback(arm_id: str):
    torch.manual_seed(42)
    model = build_exp029_model(arm_id).eval()
    B, T = 2, 25
    x_ihr = torch.randn(B, T, 30)
    x_acc = torch.randn(B, T, 30)
    x_meta = torch.randn(B, T, 5)
    seq_lens = torch.tensor([13, 19], dtype=torch.int64)

    # Perturb padded positions [seq_lens:T]
    x_ihr_pert = x_ihr.clone()
    x_acc_pert = x_acc.clone()
    x_meta_pert = x_meta.clone()
    x_ihr_pert[0, 13:] = 99.0
    x_acc_pert[0, 13:] = -99.0
    x_meta_pert[1, 19:] = 42.0

    with torch.no_grad():
        out1 = model(x_ihr, x_acc, x_meta, seq_lens=seq_lens)
        out2 = model(x_ihr_pert, x_acc_pert, x_meta_pert, seq_lens=seq_lens)

    assert torch.allclose(out1["logits"][0, :13], out2["logits"][0, :13], atol=1e-5)
    assert torch.allclose(out1["logits"][1, :19], out2["logits"][1, :19], atol=1e-5)

    # seq_lens=None fallback must match seq_lens=full(T)
    full_lens = torch.full((B,), T, dtype=torch.int64)
    with torch.no_grad():
        out_none = model(x_ihr, x_acc, x_meta, seq_lens=None)
        out_full = model(x_ihr, x_acc, x_meta, seq_lens=full_lens)
    assert torch.allclose(out_none["logits"], out_full["logits"], atol=1e-5)


@pytest.mark.parametrize("L", [1, 20, 60, 120, 300, 600, 1200])
def test_forward_14_keys_and_multi_horizon(L: int):
    model = build_exp029_model("full_v5").eval()
    B = 2
    x_ihr = torch.randn(B, L, 30)
    x_acc = torch.randn(B, L, 30)
    x_meta = torch.randn(B, L, 5)
    targets = torch.randint(0, 4, (B, L))

    out = model(x_ihr, x_acc, x_meta, targets=targets)
    assert set(out.keys()) == EXPECTED_14_KEYS
    assert out["logits"].shape == (B, L, 4)
    assert out["probabilities"].shape == (B, L, 4)
    assert out["temp_features"].shape == (B, L, 96)
    assert torch.isfinite(out["total_loss"])


def test_dynamic_int8_compatibility_and_360_50_kib_payload():
    model_fp32 = build_exp029_model("full_v5").cpu().eval()
    total_params = sum(p.numel() for p in model_fp32.parameters())
    quant_params = sum(
        p.numel()
        for mod in model_fp32.modules()
        if isinstance(mod, (nn.Linear, nn.GRU))
        for p in mod.parameters()
    )
    fp32_params = total_params - quant_params
    payload_kib = (quant_params * 1 + fp32_params * 4) / 1024.0

    assert total_params == 190_788
    assert quant_params == 131_332
    assert fp32_params == 59_456
    assert abs(payload_kib - 360.50390625) < 1e-6

    model_int8 = quantization.quantize_dynamic(
        copy.deepcopy(model_fp32), {nn.Linear, nn.GRU}, dtype=torch.qint8
    ).eval()

    x_ihr = torch.randn(1, 60, 30)
    x_acc = torch.randn(1, 60, 30)
    x_meta = torch.randn(1, 60, 5)
    with torch.no_grad():
        out_fp32 = model_fp32(x_ihr, x_acc, x_meta)
        out_int8 = model_int8(x_ihr, x_acc, x_meta)
    assert out_int8["logits"].shape == (1, 60, 4)
    assert torch.isfinite(out_int8["logits"]).all()
    assert torch.allclose(out_fp32["logits"], out_int8["logits"], atol=0.5)


@pytest.mark.parametrize("N", [45, 137])
def test_pre_softmax_logit_stitching_and_damped_em_zero_deep(N: int):
    model = build_exp029_model("full_v5").cpu().eval()
    L = 60
    rec = {
        "subject": "TestSub",
        "night": "1",
        "ihr": np.random.randn(N, 30).astype(np.float32),
        "accel": np.random.randn(N, 30).astype(np.float32),
        "freq": np.ones((N, 1), dtype=np.float32) * 0.2,
        "time": np.zeros((N, 1), dtype=np.float32),
        "y": np.random.choice([0, 1, 3], size=N).astype(np.int64),  # 0% Deep
    }

    pi_src = np.array([0.18, 0.44, 0.19, 0.19], dtype=np.float64)
    cal = DampedEMCalibrator(pi_src=pi_src, alpha=0.5, n_iter=3, temperature=1.0)

    pred_off = LogitStitchingSlidingWindowPredictor(
        model=model, context_len=L, device=torch.device("cpu"), prior_calibrator=None
    )
    pred_on = LogitStitchingSlidingWindowPredictor(
        model=model, context_len=L, device=torch.device("cpu"), prior_calibrator=cal
    )

    out_off = pred_off.predict_night(rec)
    out_on = pred_on.predict_night(rec)

    assert out_off["logits"].shape == (N, 4)
    assert out_off["probs"].shape == (N, 4)
    assert np.all(out_off["counts"] >= 1.0)
    assert np.allclose(out_off["probs"].sum(axis=1), 1.0, atol=1e-6)
    assert np.all(np.isfinite(out_on["probs"]))
    assert np.allclose(out_on["probs"].sum(axis=1), 1.0, atol=1e-6)


def test_fp32_loss_outside_autocast_and_cuda_amp_if_available():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = build_exp029_model("full_v5").to(device).train()

    x_ihr = torch.randn(2, 60, 30, device=device)
    x_acc = torch.randn(2, 60, 30, device=device)
    x_meta = torch.randn(2, 60, 5, device=device)
    targets = torch.randint(0, 4, (2, 60), device=device)
    mask = torch.ones((2, 60), dtype=torch.bool, device=device)

    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        out = model(x_ihr, x_acc, x_meta)

    loss = compute_stage_loss_fp32(model, out["logits"], targets, mask)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    loss.backward()
    grads_finite = all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert grads_finite
