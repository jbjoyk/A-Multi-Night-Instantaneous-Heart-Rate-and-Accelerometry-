"""
test_exp018_pipeline.py: Comprehensive Unit Test Suite for EXP-018.

Verifies:
  1. Random crop sampling bounds and short-night right-padding.
  2. Compute invariant: B * L * N_accum == 2,400 tokens/step for all 7 L values.
  3. Sliding-window inference on short nights (N <= L) with margin bypass.
  4. Sliding-window inference full-night coverage (N > L) with zero gaps (counts >= 1).
  5. Circadian neutralization control (time_feat == 0 in primary; preserved in ablation).
  6. Clinical PSG metrics calculation and zero-sleep edge cases.
  7. Model A capacity (163,940 parameters) and packed recurrent padding protection.
  8. Decision matrix coverage (MECE completeness).
  9. Data integrity and hard test-set leakage protection.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn

from dataset_gate import DatasetGate, TestSetLeakageError
from exp013_models import ConventionalCNNBiGRU
from evaluator import ClinicalHypnogramEvaluator
from exp018_dataset import (
    COMPUTE_SCHEDULE_LOOKUP,
    RandomCropSleepDataset,
    verify_compute_schedule_invariant
)
from exp018_inference import SlidingWindowPredictor
from run_exp018_execution import (
    evaluate_decision_matrix,
    compute_paired_subject_bootstrap_delta,
    fit_parametric_saturation_curve
)
from slamss_ifs_baseline import make_reproducible_split


def _create_mock_night(n_epochs: int, subject: str = "Bidslab00", night: str = "night_01") -> dict:
    """Helper creating synthetic night records with arbitrary duration."""
    return {
        "ihr": np.random.randn(n_epochs, 30).astype(np.float32),
        "accel": np.random.randn(n_epochs, 30).astype(np.float32),
        "freq": np.ones((n_epochs, 1), dtype=np.float32),
        "time": np.linspace(0.0, 1.0, n_epochs, dtype=np.float32).reshape(-1, 1),
        "y": np.random.randint(0, 4, size=n_epochs, dtype=np.int64),
        "subject": subject,
        "night": night
    }


def test_token_exposure_invariant():
    """Verify that every context length satisfies B * L * N_accum == 2,400."""
    verify_compute_schedule_invariant()
    expected_L = [1, 20, 60, 120, 300, 600, 1200]
    assert sorted(list(COMPUTE_SCHEDULE_LOOKUP.keys())) == expected_L

    for L in expected_L:
        cfg = COMPUTE_SCHEDULE_LOOKUP[L]
        effective_tokens = cfg["B"] * L * cfg["N_accum"]
        assert effective_tokens == 2400
        # Total tokens over 3200 steps
        assert 3200 * effective_tokens == 7_680_000


def test_random_crop_sampling_and_padding():
    """Verify uniform crop window extraction and short-night right-padding."""
    sub_map = {"Bidslab00": 0}

    # Case A: Native length N=800 > L=300
    rec_long = _create_mock_night(800)
    ds_long = RandomCropSleepDataset([rec_long], sub_map, context_len=300, seed=42)
    sample_long = ds_long[0]

    assert sample_long["ihr"].shape == (300, 30)
    assert sample_long["accel"].shape == (300, 30)
    assert sample_long["meta"].shape == (300, 5)
    assert sample_long["y"].shape == (300,)
    assert sample_long["mask"].shape == (300,)
    assert sample_long["seq_len"].item() == 300
    assert torch.all(sample_long["mask"])  # All 300 positions are valid
    assert not torch.any(sample_long["y"] == -100)

    # Case B: Native length N=150 <= L=300 (short night right-padding)
    rec_short = _create_mock_night(150)
    ds_short = RandomCropSleepDataset([rec_short], sub_map, context_len=300, seed=42)
    sample_short = ds_short[0]

    assert sample_short["ihr"].shape == (300, 30)
    assert sample_short["seq_len"].item() == 150
    assert torch.all(sample_short["mask"][:150])
    assert not torch.any(sample_short["mask"][150:])
    assert torch.all(sample_short["y"][150:] == -100)
    # Padded feature tensors should be zeroed
    assert torch.all(sample_short["ihr"][150:] == 0.0)
    assert torch.all(sample_short["accel"][150:] == 0.0)


def test_circadian_neutralization():
    """Verify that time_feat (index 2 of meta) is 0.0 in primary and active in ablation."""
    sub_map = {"Bidslab00": 0}
    rec = _create_mock_night(400)

    # Primary series: neutralized
    ds_prim = RandomCropSleepDataset([rec], sub_map, context_len=120, neutralize_circadian=True)
    sample_prim = ds_prim[0]
    # Meta layout: [freq, clock, time_feat, hr_mean, hr_std]
    assert torch.all(sample_prim["meta"][:, 2] == 0.0)

    # Ablation series: active
    ds_ablat = RandomCropSleepDataset([rec], sub_map, context_len=120, neutralize_circadian=False)
    sample_ablat = ds_ablat[0]
    assert not torch.all(sample_ablat["meta"][:, 2] == 0.0)


def test_sliding_inference_short_night():
    """Verify that N <= L inference bypasses margin discard and emits exact [0, N) hypnogram."""
    model = ConventionalCNNBiGRU()
    pred = SlidingWindowPredictor(model, context_len=600, device=torch.device("cpu"))

    rec_short = _create_mock_night(350)
    out = pred.predict_night(rec_short)

    assert len(out["y_pred"]) == 350
    assert out["probs"].shape == (350, 4)
    assert np.all(out["counts"] == 1.0)
    assert np.all(out["probs"] >= 0.0) and np.all(out["probs"] <= 1.0)


def test_sliding_inference_full_night_coverage():
    """Verify that N > L produces 100% nocturnal coverage without gaps across varied lengths."""
    model = ConventionalCNNBiGRU()
    pred_60 = SlidingWindowPredictor(model, context_len=60, device=torch.device("cpu"))
    pred_300 = SlidingWindowPredictor(model, context_len=300, device=torch.device("cpu"))
    pred_600 = SlidingWindowPredictor(model, context_len=600, device=torch.device("cpu"))

    # Test realistic recording durations: 610, 857, 1282 epochs
    for n_len in [610, 857, 1282]:
        rec = _create_mock_night(n_len)
        for predictor in [pred_60, pred_300, pred_600]:
            out = predictor.predict_night(rec)
            assert len(out["y_pred"]) == n_len
            assert out["probs"].shape == (n_len, 4)
            # Ensure every single epoch received at least one prediction
            assert np.all(out["counts"] >= 1.0), f"Gap detected for N={n_len}, L={predictor.context_len}"
            # Probabilities must sum to 1.0 per epoch
            prob_sums = np.sum(out["probs"], axis=-1)
            np.testing.assert_allclose(prob_sums, 1.0, atol=1e-5)


def test_clinical_metrics_synthetic():
    """Verify PSG clinical metrics calculation and zero-sleep edge cases."""
    evaluator = ClinicalHypnogramEvaluator

    # Case 1: Synthetic regular night (N=20 epochs = 10 minutes)
    # Epochs 0-1: Wake (1.0 min)
    # Epochs 2-4: Light (1.5 min) -> Sleep onset at epoch 2 (SOL = 1.0 min)
    # Epochs 5-7: Deep (1.5 min)
    # Epochs 8-9: Wake (1.0 min) -> WASO
    # Epochs 10-14: REM (2.5 min)
    # Epochs 15-19: Wake (2.5 min) -> final sleep is at 14
    hyp = np.array([
        0, 0,          # Wake (1.0m)
        1, 1, 1,       # Light (1.5m) -> onset at idx 2 (1.0 min)
        2, 2, 2,       # Deep (1.5m)
        0, 0,          # Wake (1.0m) -> WASO
        3, 3, 3, 3, 3, # REM (2.5m) -> final sleep at idx 14
        0, 0, 0, 0, 0  # Wake (2.5m)
    ])
    m = evaluator.compute_night_metrics(hyp)

    assert m["tib_min"] == 10.0
    # Sleep epochs: 3 Light + 3 Deep + 5 REM = 11 epochs = 5.5 min
    assert m["tst_min"] == 5.5
    assert m["sol_min"] == 1.0
    assert m["waso_min"] == 1.0
    assert pytest.approx(m["sleep_efficiency_pct"], 0.01) == 55.0
    assert m["deep_min"] == 1.5
    assert m["rem_min"] == 2.5
    assert pytest.approx(m["deep_pct_tst"], 0.01) == (1.5 / 5.5 * 100.0)

    # Case 2: Zero-sleep night (all Wake)
    hyp_zeros = np.zeros(100, dtype=np.int64)
    m_zero = evaluator.compute_night_metrics(hyp_zeros)
    assert m_zero["tib_min"] == 50.0
    assert m_zero["tst_min"] == 0.0
    assert m_zero["sol_min"] == 50.0  # SOL = TIB
    assert m_zero["waso_min"] == 0.0  # WASO = 0
    assert m_zero["sleep_efficiency_pct"] == 0.0


def test_model_a_capacity_and_forward():
    """Verify Model A has exactly 163,940 parameters and packed recurrent sequence handling."""
    model = ConventionalCNNBiGRU()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 163940

    B, T = 2, 60
    x_ihr = torch.randn(B, T, 30)
    x_acc = torch.randn(B, T, 30)
    x_meta = torch.randn(B, T, 5)
    seq_lens = torch.tensor([60, 40], dtype=torch.int64)
    targets = torch.randint(0, 4, (B, T))
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[1, 40:] = False

    out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)
    assert "logits" in out
    assert "total_loss" in out
    assert out["logits"].shape == (B, T, 4)
    assert not torch.isnan(out["total_loss"])


def test_decision_matrix_mece():
    """Verify that all possible outcome combinations map to exactly one category without dead zones."""
    evaluator = evaluate_decision_matrix

    # 1. SUPPORTED: delta >= 1.0 pp, CI_lower > 0
    res1 = evaluator(0.015, 0.005, 0.025)
    assert res1["verdict"] == "SUPPORTED (Context Benefit)"
    assert res1["action"] == "AUTHORIZE_PHASE_2"

    # 2. SUB-THRESHOLD GAIN: 0 < delta < 1.0 pp, CI_lower > 0
    res2 = evaluator(0.007, 0.001, 0.012)
    assert res2["verdict"] == "SUB-THRESHOLD GAIN"
    assert res2["action"] == "CONCLUDE_PHASE_1"

    # 3. TRUE EQUIVALENCE: CI strictly in [-0.5, +0.5] pp
    res3 = evaluator(0.001, -0.003, 0.004)
    assert res3["verdict"] == "TRUE EQUIVALENCE (Flat Horizon)"
    assert res3["action"] == "CONCLUDE_PHASE_1"

    # 4. SUB-THRESHOLD IMPAIRMENT: -1.0 < delta < 0, CI_upper < 0
    res4 = evaluator(-0.006, -0.009, -0.002)
    assert res4["verdict"] == "SUB-THRESHOLD IMPAIRMENT"
    assert res4["action"] == "CONCLUDE_PHASE_1"

    # 5. REJECTED: delta <= -1.0 pp, CI_upper < 0
    res5 = evaluator(-0.015, -0.022, -0.005)
    assert res5["verdict"] == "REJECTED (Context Impairment)"
    assert res5["action"] == "CONCLUDE_PHASE_1"

    # 6. INCONCLUSIVE: Spans 0 and not contained in [-0.5, +0.5] pp
    res6 = evaluator(0.002, -0.008, 0.012)
    assert res6["verdict"] == "INCONCLUSIVE (Insufficient Precision)"
    assert res6["action"] == "CONCLUDE_PHASE_1"


def test_data_integrity_and_no_leakage():
    """Verify that DatasetGate catches unauthorized access to held-out test subjects."""
    gate = DatasetGate()
    ds_status = gate.verify_dataset_structure()
    splits = make_reproducible_split(ds_status["subjects"], split_type="paper", seed=0)

    train_subs = splits["train"]
    val_subs = splits["val"]
    test_subs = splits["test"]

    assert len(train_subs) == 31
    assert len(val_subs) == 5
    assert len(test_subs) == 11

    # Gate must raise TestSetLeakageError if training touches any test subject
    with pytest.raises(TestSetLeakageError):
        gate.assert_no_test_leakage("training", [test_subs[0]], test_subs)

    # Gate must raise TestSetLeakageError if validation touches any test subject
    with pytest.raises(TestSetLeakageError):
        gate.assert_no_test_leakage("validation", [test_subs[0]], test_subs)
