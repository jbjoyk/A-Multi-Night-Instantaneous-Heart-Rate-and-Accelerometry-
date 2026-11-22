#!/usr/bin/env python3
"""
test_exp027_exp028_preflight.py: Mandatory pre-flight unit tests for EXP-027 & EXP-028.
"""

import os
import sys
import numpy as np
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [
    REPO_ROOT,
    os.path.join(REPO_ROOT, "experiments", "EXP-018"),
    os.path.join(REPO_ROOT, "experiments", "EXP-021"),
    os.path.join(REPO_ROOT, "experiments", "EXP-027"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluator import ClinicalHypnogramEvaluator
from exp013_models import ConventionalCNNBiGRU
from exp018_inference import SlidingWindowPredictor
from official_slamss_ifs_adapter import (
    OfficialSLAMSSIFSWrapper,
    SLAMSSSlidingWindowPredictor,
)
from prepare_sleepaccel import (
    ExternalValidationGateError,
    map_sleepaccel_labels,
    preprocess_sleepaccel_arrays,
    run_preflight_distribution_gate,
)
from run_exp028_sleepaccel_bland_altman import compute_bland_altman_stats


def _make_synthetic_sleepaccel(n_epochs: int = 650, acc_scale: float = 1.0, hr_offset: float = 65.0, t_neg_hours: float = 0.0):
    rng = np.random.default_rng(42)
    t_lbl = np.arange(n_epochs, dtype=np.float64) * 30.0
    stages = rng.choice([0, 1, 2, 3, 5], size=n_epochs)

    t_start = -t_neg_hours * 3600.0
    t_hr = np.arange(t_start, n_epochs * 30.0, 2.0)
    bpm = hr_offset + 5.0 * np.sin(t_hr / 100.0)
    if t_neg_hours > 0:
        # Pollute pre-PSG t < 0 with high ambulatory HR
        bpm[t_hr < 0] = 140.0

    t_mo = np.arange(max(t_start, -10.0), n_epochs * 30.0, 0.1)
    xyz = rng.normal(0.0, 0.05, size=(len(t_mo), 3)) * acc_scale
    xyz[:, 2] += 1.0 * acc_scale
    return t_hr, bpm, t_mo, xyz, t_lbl, stages


def test_sleepaccel_4class_label_mapping():
    raw = np.array([0, 1, 2, 3, 5, -1, 4, 9, np.nan])
    mapped = map_sleepaccel_labels(raw)
    assert np.array_equal(mapped, np.array([0, 1, 1, 2, 3, -100, -100, -100, -100]))


def test_1hz_aggregation_clipping_and_t_neg_trim_before_scale():
    t_hr, bpm, t_mo, xyz, t_lbl, stages = _make_synthetic_sleepaccel(n_epochs=650, t_neg_hours=2.0)
    # Inject extreme ACC spike to test [-10, 10] clipping
    xyz[len(xyz) // 2, 0] = 50.0
    rec = preprocess_sleepaccel_arrays(t_hr, bpm, t_mo, xyz, t_lbl, stages, "sub_test")
    assert rec["ihr"].shape == (650, 30)
    assert rec["accel"].shape == (650, 30)
    assert np.max(np.abs(rec["accel"])) <= 10.0
    assert np.max(np.abs(rec["ihr"])) <= 10.0
    # Pre-PSG 140 bpm must not shift raw_hr_median from ~65 bpm
    assert 60.0 <= rec["raw_hr_median"] <= 70.0


def test_meta_freq_formula_and_circadian_neutralization():
    t_hr, bpm, t_mo, xyz, t_lbl, stages = _make_synthetic_sleepaccel(n_epochs=650)
    rec = preprocess_sleepaccel_arrays(t_hr, bpm, t_mo, xyz, t_lbl, stages, "sub_meta")
    # 0.5 Hz HR -> 15 samples per 30s epoch -> log1p(15 / 30 * 10) = log1p(5.0)
    expected_freq = np.log1p(15.0 / 30.0 * 10.0)
    assert abs(float(np.median(rec["freq"])) - expected_freq) < 0.1
    assert np.all(rec["time"] == 0.0)


def test_coverage_qc_60pct_gate():
    t_hr, bpm, t_mo, xyz, t_lbl, stages = _make_synthetic_sleepaccel(n_epochs=650)
    # Drop 50% of HR recording so usable_frac < 0.60
    keep = t_hr < (300 * 30.0)
    rec_bad = preprocess_sleepaccel_arrays(t_hr[keep], bpm[keep], t_mo, xyz, t_lbl, stages, "sub_bad")
    assert rec_bad["qc_pass"] is False

    rec_ok = preprocess_sleepaccel_arrays(t_hr, bpm, t_mo, xyz, t_lbl, stages, "sub_ok")
    assert rec_ok["qc_pass"] is True


def test_preflight_unit_mismatch_aborts():
    t_hr, bpm, t_mo, xyz, t_lbl, stages = _make_synthetic_sleepaccel(n_epochs=650, acc_scale=9.81)
    rec_ms2 = preprocess_sleepaccel_arrays(t_hr, bpm, t_mo, xyz, t_lbl, stages, "sub_ms2")
    with pytest.raises(ExternalValidationGateError, match="ACC unit mismatch"):
        run_preflight_distribution_gate([rec_ms2], bidsleep_cache_dir=None)

    t_hr2, _, t_mo2, xyz2, t_lbl2, stages2 = _make_synthetic_sleepaccel(n_epochs=650)
    # Force RR-ms scale (~800) by bypassing physiological filter via raw_hr_median override
    rec_rrms = preprocess_sleepaccel_arrays(t_hr2, bpm, t_mo2, xyz2, t_lbl2, stages2, "sub_rr")
    rec_rrms["raw_hr_median"] = 800.0
    with pytest.raises(ExternalValidationGateError, match="IHR unit mismatch"):
        run_preflight_distribution_gate([rec_rrms], bidsleep_cache_dir=None)


def test_predictors_smoke_short_and_long_night():
    t_hr, bpm, t_mo, xyz, t_lbl, stages = _make_synthetic_sleepaccel(n_epochs=650)
    rec_short = preprocess_sleepaccel_arrays(t_hr, bpm, t_mo, xyz, t_lbl, stages, "sub_short")
    t_hr_l, bpm_l, t_mo_l, xyz_l, t_lbl_l, stages_l = _make_synthetic_sleepaccel(n_epochs=1300)
    rec_long = preprocess_sleepaccel_arrays(t_hr_l, bpm_l, t_mo_l, xyz_l, t_lbl_l, stages_l, "sub_long")

    model_a = ConventionalCNNBiGRU(d_model=64, d_temp=96, n_classes=4, n_meta=5, dropout=0.2, use_rcc_loss=True)
    pred_a = SlidingWindowPredictor(model=model_a, context_len=1200, neutralize_circadian=True, device=torch.device("cpu"))
    out_a_s = pred_a.predict_night(rec_short)
    out_a_l = pred_a.predict_night(rec_long)
    assert out_a_s["probs"].shape == (650, 4) and np.all(np.isfinite(out_a_s["probs"]))
    assert out_a_l["probs"].shape == (1300, 4) and np.all(np.isfinite(out_a_l["probs"]))

    model_s = OfficialSLAMSSIFSWrapper(device=torch.device("cpu"))
    pred_s = SLAMSSSlidingWindowPredictor(model=model_s, context_len=1200, neutralize_circadian=True, device=torch.device("cpu"))
    out_s_s = pred_s.predict_night(rec_short)
    assert out_s_s["probs"].shape == (650, 4) and np.all(np.isfinite(out_s_s["probs"]))


def test_clinical_hypnogram_evaluator_parity_and_bland_altman():
    # 3-consecutive-epoch SOL rule: single isolated sleep at idx 2, 3-epoch onset at idx 10
    hyp = np.zeros(100, dtype=np.int64)
    hyp[2] = 1
    hyp[10:20] = 1
    hyp[25] = 0
    m = ClinicalHypnogramEvaluator.compute_night_metrics(hyp)
    assert m["sol_min"] == 5.0  # idx 10 * 0.5 min

    # Zero-sleep night
    m_zero = ClinicalHypnogramEvaluator.compute_night_metrics(np.zeros(40, dtype=np.int64))
    assert m_zero["sol_min"] == 20.0 and m_zero["tst_min"] == 0.0

    # Bland-Altman 1999 SE(LoA) & proportional bias
    rng = np.random.default_rng(7)
    ref = rng.normal(350.0, 30.0, size=31)
    pred = ref + rng.normal(2.0, 10.0, size=31)
    ba = compute_bland_altman_stats(ref, pred)
    expected_se = np.sqrt((ba["sd_diff"] ** 2) * (1.0 / 31.0 + (1.96 ** 2) / (2.0 * 30.0)))
    assert abs(ba["se_loa"] - expected_se) < 1e-9
    assert np.isfinite(ba["prop_bias_slope"]) and np.isfinite(ba["shapiro_p"])
