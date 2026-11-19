"""
test_tsk06_tsk07_execution_readiness.py: Execution-readiness unit and integration test suite.

Validates:
  1. compute_wearable_quality_mask behavior and scale-invariance.
  2. Page's L-trend test mathematical exactness against theoretical null distribution.
  3. Pre-flight quality gate assertions (> 95% valid epochs, min epochs >= 600, zero non-physiological leakage).
  4. TSK-06 calibration-depth driver dry-run execution.
  5. TSK-07 UA-2 driver dry-run execution.
  6. Locked artifact immutability (SHA-256 hashes preserved).
"""

import os
import sys
import json
import hashlib
import subprocess
import pytest
import numpy as np
import scipy.stats as stats

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
exp019_dir = os.path.join(repo_root, "experiments", "EXP-019")
for p in [repo_root, exp019_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from exp019_adaptation_core import compute_wearable_quality_mask
from run_tsk06_k_curve import pages_l_test

LOCKED_HASHES = {
    "exp019_subject_metrics.csv": "87f0910ba72c346fd465bc64e5d6f32c6ef609454ea11eb99abccf5af15bb407",
    "exp019_sleep_summary_agreement.csv": "41a2d6afa73b3d65599a83c7c463e7cadbbbcabef6d1f71a6349194b489b35e3",
    "exp019_k_learning_curve.csv": "2bf08608c4685836facc52fefea5ccf8eb0070af93a81b2e0c35b5dde61af3e3",
}


def test_wearable_quality_mask_logic():
    """Verifies that flatlined signals are masked out while valid physiological variation passes."""
    # 1. Normal signals: 30 epochs
    ihr_normal = np.full((30, 1), 70.0)
    acc_normal = np.random.randn(30, 3)  # high variance
    mask = compute_wearable_quality_mask(ihr_normal, acc_normal)
    assert np.all(mask), "Expected all epochs to pass with normal variation"

    # 2. Zero-variance sensor flatline (e.g. sensor off wrist)
    acc_flat = np.full((30, 3), 0.05)
    mask_flat = compute_wearable_quality_mask(ihr_normal, acc_flat)
    assert not np.any(mask_flat), "Expected flatline acceleration to be masked out"

    # 3. Non-physiological raw heart rates (e.g. 10 bpm or 250 bpm if unstandardized)
    ihr_extreme = np.full((30, 1), 250.0)
    mask_extreme = compute_wearable_quality_mask(ihr_extreme, acc_normal)
    assert not np.any(mask_extreme), "Expected non-physiological HR > 220 to be masked out"

    # 4. Standardized z-score signals (mean ~ 0, std ~ 1)
    ihr_zscore = np.random.randn(30, 1)  # z-scores around 0
    mask_z = compute_wearable_quality_mask(ihr_zscore, acc_normal)
    assert np.all(mask_z), "Expected standardized z-scores with valid accel variance to pass"


def test_pages_l_test_mathematics():
    """
    Validates Page's L-trend test mathematical exactness for N=33, k=4 against theoretical values.
    Theoretical Null:
      mean(L) = N * k * (k + 1)^2 / 4 = 33 * 4 * 25 / 4 = 825.0
      var(L)  = N * (k^3 - k) * (k + 1) / 144 = 33 * 60 * 5 / 144 = 275.0
      std(L)  = sqrt(275) = 16.58312395...
    """
    N = 33
    k = 4
    # Perfect monotonic rank ordering: ranks 1, 2, 3, 4 for every subject
    perfect_ranks = np.tile(np.arange(1, k + 1), (N, 1))
    L, z, p, mean_L, std_L = pages_l_test(perfect_ranks)

    assert mean_L == 825.0, f"Expected theoretical mean 825.0, got {mean_L}"
    assert pytest.approx(std_L, rel=1e-5) == np.sqrt(275.0)
    assert L == 990.0, f"Expected perfect rank L = 990.0, got {L}"
    assert z > 9.9, f"Expected z-score > 9.9, got {z}"
    assert p < 1e-15, f"Expected infinitesimal p-value, got {p}"

    # Perfect reverse rank ordering: ranks 4, 3, 2, 1
    reverse_ranks = np.tile(np.arange(k, 0, -1), (N, 1))
    L_rev, z_rev, p_rev, _, _ = pages_l_test(reverse_ranks)
    assert L_rev == 660.0, f"Expected reverse rank L = 660.0, got {L_rev}"
    assert z_rev < -9.9, f"Expected negative z-score < -9.9, got {z_rev}"
    assert pytest.approx(p_rev, abs=1e-5) == 1.0


def test_preflight_gate_audit_log():
    """Verifies that the pre-flight gate audit log exists and all 3 mandatory assertions passed."""
    log_path = os.path.join(exp019_dir, "ua2_quality_gate_audit.log")
    assert os.path.exists(log_path), f"Audit log not found at {log_path}"

    with open(log_path, "r") as f:
        content = f.read()

    assert "PRE-FLIGHT GATE STATUS: PASSED" in content
    assert "Cohort Valid Epochs:" in content
    assert "Non-Physiological Leakage Count:  0" in content


def test_tsk07_dry_run_execution():
    """Runs TSK-07 driver with --dry-run on a single subject and verifies clean exit."""
    python_bin = sys.executable
    script_path = os.path.join(exp019_dir, "run_tsk07_ua2_rerun.py")

    cmd = [python_bin, script_path, "--dry-run", "--subjects", "Bidslab01", "--device", "cpu"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"TSK-07 dry-run failed:\n{res.stderr}"
    assert "Dry run completed successfully" in res.stdout


def test_tsk06_dry_run_execution():
    """Runs TSK-06 driver with --dry-run on a single subject and verifies clean exit."""
    python_bin = sys.executable
    script_path = os.path.join(exp019_dir, "run_tsk06_k_curve.py")

    cmd = [python_bin, script_path, "--dry-run", "--subjects", "Bidslab01", "--device", "cpu"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"TSK-06 dry-run failed:\n{res.stderr}"
    assert "Dry run completed successfully" in res.stdout


def test_locked_artifact_hashes_immutable():
    """Guarantees that locked historical files have NOT been modified or overwritten."""
    for fname, expected_hash in LOCKED_HASHES.items():
        fpath = os.path.join(exp019_dir, fname)
        assert os.path.exists(fpath), f"Locked file missing: {fpath}"
        with open(fpath, "rb") as f:
            actual_hash = hashlib.sha256(f.read()).hexdigest()
        assert actual_hash == expected_hash, f"CRITICAL: Locked file {fname} SHA-256 corrupted! Expected {expected_hash}, got {actual_hash}"
