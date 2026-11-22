#!/usr/bin/env python3
"""
test_exp019b_features.py: Mandatory CPU-fast unit tests for EXP-019B.
"""

import os
import sys
import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [
    REPO_ROOT,
    os.path.join(REPO_ROOT, "experiments", "EXP-019B"),
    os.path.join(REPO_ROOT, "experiments", "EXP-019"),
    os.path.join(REPO_ROOT, "experiments", "STAGE2_FINAL_MODEL"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_exp019b_heterogeneity_audit import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DIAG_TABLE,
    DEFAULT_EXP019_METRICS,
    DEFAULT_EXP021_DELTAS,
    DEFAULT_NIGHT_INVENTORY,
    DEFAULT_RAW_BIDSLEEP_DIR,
    compute_subject_window_features,
    load_usable_nights_by_subject,
    spearman_with_bootstrap,
)
from audit_heterogeneity_mechanisms import compute_bh_fdr
from exp019_adaptation_core import compute_wearable_quality_mask
from final_model_pipeline import audit_calibration_stage_coverage
from slamss_ifs_train import load_night_features


def test_cohort_subject_id_sets():
    diag_df = pd.read_csv(DEFAULT_DIAG_TABLE)
    exp019_df = pd.read_csv(DEFAULT_EXP019_METRICS)
    exp021_df = pd.read_csv(DEFAULT_EXP021_DELTAS)

    f1_subs = set(exp019_df["subject_id"].astype(str))
    f2_subs = set(exp021_df["subject_id"].astype(str))
    qual_subs = set(diag_df.loc[diag_df["qualifies_exp019_eval"].astype(bool), "subject_id"].astype(str))

    assert len(f1_subs) == 46
    assert "Bidslab49" not in f1_subs
    assert f1_subs == qual_subs
    assert len(f2_subs) == 47
    assert "Bidslab49" in f2_subs


def test_raw_hr_features_physiological_bounds_and_no_nans():
    usable = load_usable_nights_by_subject(DEFAULT_NIGHT_INVENTORY)
    exp019_subs = set(pd.read_csv(DEFAULT_EXP019_METRICS)["subject_id"].astype(str))

    for sub in sorted(usable.keys()):
        nights = usable[sub]
        f_all = compute_subject_window_features(sub, nights, DEFAULT_CACHE_DIR, DEFAULT_RAW_BIDSLEEP_DIR)
        for k, v in f_all.items():
            assert np.isfinite(v), f"Non-finite {k} for {sub} (all)"
        assert 35.0 <= f_all["baseline_hr_mean"] <= 120.0
        assert f_all["baseline_hr_iqr"] >= 0.0
        assert f_all["eff_ihr_hz"] > 0.0
        assert 0.0 <= f_all["missingness_rate"] < 1.0

        if sub in exp019_subs:
            f_cal = compute_subject_window_features(sub, nights[:2], DEFAULT_CACHE_DIR, DEFAULT_RAW_BIDSLEEP_DIR)
            for k, v in f_cal.items():
                assert np.isfinite(v), f"Non-finite {k} for {sub} (cal)"

    # Bidslab34 parity check against audit_calibration_stage_coverage()
    b34_cal = [load_night_features(DEFAULT_CACHE_DIR, "Bidslab34", n) for n in usable["Bidslab34"][:2]]
    cov_diag = audit_calibration_stage_coverage(b34_cal)
    f_b34 = compute_subject_window_features("Bidslab34", usable["Bidslab34"][:2], DEFAULT_CACHE_DIR, DEFAULT_RAW_BIDSLEEP_DIR)
    assert abs(f_b34["min_stage_frac"] - min(cov_diag["stage_fractions"].values())) < 1e-5


def test_synthetic_spearman_bh_fdr_and_bootstrap():
    p_vec = [0.001, 0.01, 0.04, 0.20, 0.50, 0.90]
    q_vec = compute_bh_fdr(p_vec)
    assert len(q_vec) == 6
    assert abs(q_vec[0] - 0.006) < 1e-9
    assert q_vec[0] <= q_vec[1] <= q_vec[2] <= q_vec[3] <= q_vec[4] <= q_vec[5] <= 1.0

    # Boolean union missingness check (no double counting)
    ihr = np.zeros((4, 30), dtype=np.float32)
    accel = np.ones((4, 30), dtype=np.float32)
    accel[1:] += np.linspace(0, 1, 30)[None, :]  # epoch 0 has zero variance
    y = np.array([0, -100, 1, 2], dtype=np.int64)  # epoch 0 fails wearable, epoch 1 fails label
    w_ok = compute_wearable_quality_mask(ihr, accel)
    missing = (~w_ok) | (y < 0) | (y >= 4)
    assert int(np.sum(missing)) == 2

    # Constant-vector bootstrap edge case
    x_const = np.ones(20, dtype=np.float64)
    y_vals = np.arange(20, dtype=np.float64)
    x_almost_const = np.zeros(20, dtype=np.float64)
    x_almost_const[0] = 1.0
    res = spearman_with_bootstrap(x_almost_const, y_vals, n_boot=200, seed=42)
    assert res["n_nan_resamples"] > 0
    assert np.isfinite(res["ci95_low"]) and np.isfinite(res["ci95_high"])
