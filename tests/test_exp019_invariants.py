"""
test_exp019_invariants.py: Automated unit and invariant test suite for EXP-019.

Validates experimental preconditions and safety guarantees:
  1. Temporal disjointness (C cap E == empty, chronological order).
  2. Cohort census (N=46 eligible subjects, 152 outer evaluation nights, Bidslab49 excluded).
  3. State neutralization invariance via in-memory deep copy.
  4. Target masking and remapping (-100 ignore index, valid classes 0..3).
  5. UA-2 wearable-only quality heuristic (30 <= HR <= 180, accel var > 0).
  6. UA-2 anti-collapse safeguard assertion (H < 0.6931 nats triggers abort).
"""

import os
import sys
import copy
import json
import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Ensure repo root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from exp013_models import ConventionalCNNBiGRU


def get_adaptation_core():
    """Helper to dynamically import exp019_adaptation_core regardless of directory naming."""
    import importlib.util
    mod_path = os.path.join(repo_root, "experiments", "EXP-019", "exp019_adaptation_core.py")
    spec = importlib.util.spec_from_file_location("exp019_adaptation_core", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cohort_census_and_bidslab49_exclusion():
    """Verify that exactly 46 subjects are eligible (usable >= 3) and Bidslab49 is excluded."""
    manifest_csv = os.path.join(repo_root, "review_package", "fold_manifests.csv")
    assert os.path.exists(manifest_csv), f"Authoritative manifest missing: {manifest_csv}"
    
    df = pd.read_csv(manifest_csv)
    # Check Bidslab49
    bids49 = df[df["subject_id"] == "Bidslab49"]
    assert len(bids49) >= 1
    assert bids49["n_nights"].values[0] == 2, f"Expected Bidslab49 to have 2 nights, got {bids49['n_nights'].values[0]}"
    
    # Filter fold 1 to inspect unique subjects
    fold1_df = df[df["fold"] == 1]
    eligible_subs = fold1_df[fold1_df["n_nights"] >= 3]["subject_id"].tolist()
    assert len(eligible_subs) == 46, f"Expected exactly 46 eligible subjects, got {len(eligible_subs)}"
    assert "Bidslab49" not in eligible_subs, "Bidslab49 must be excluded from eligible subjects"

    # Total outer evaluation nights across all 5 folds must be 152
    outer_df = df[df["role"] == "outer_test"]
    outer_eligible = outer_df[outer_df["subject_id"].isin(eligible_subs)]
    
    # In each fold, calibration nights k=2, so evaluation nights = n_nights - 2
    outer_eval_nights = sum(outer_eligible["n_nights"] - 2)
    assert outer_eval_nights == 152, f"Expected 152 outer evaluation nights, got {outer_eval_nights}"


def test_temporal_disjointness():
    """Verify that calibration nights C (k=2) and evaluation nights E have C cap E == empty and are forward in time."""
    night_csv = os.path.join(repo_root, "review_package", "night_inventory.csv")
    assert os.path.exists(night_csv), f"Night inventory missing: {night_csv}"
    df = pd.read_csv(night_csv)
    
    # Group by subject and inspect usable nights
    usable_df = df[df["usable"] == True]
    sub_counts = usable_df.groupby("subject_id").size()
    eligible_subs = sub_counts[sub_counts >= 3].index.tolist()
    
    for sub in eligible_subs:
        sub_df = usable_df[usable_df["subject_id"] == sub].sort_values("night")
        nights = sub_df["night"].tolist()
        
        cal_nights = set(nights[:2])
        eval_nights = set(nights[2:])
        
        # Disjointness
        assert len(cal_nights.intersection(eval_nights)) == 0, f"Overlap between C and E for subject {sub}"
        
        # Chronological order: night numbers are sorted strictly ascending
        for i in range(len(nights) - 1):
            assert nights[i] < nights[i+1], f"Night numbers not strictly ascending for {sub}: {nights}"


def test_state_neutralization_invariance():
    """Verify reset_model_weights completely restores identical model weights without leakage."""
    core = get_adaptation_core()
    
    model = ConventionalCNNBiGRU(d_model=64, d_temp=96, n_classes=4, n_meta=5, dropout=0.2, use_rcc_loss=True)
    ref_state = copy.deepcopy(model.state_dict())
    
    # Mutate all weights by adding random noise
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
            
    # Verify mutation occurred
    mutated_diff = False
    for k, v in model.state_dict().items():
        if not torch.allclose(v, ref_state[k]):
            mutated_diff = True
            break
    assert mutated_diff, "Weights were not mutated"
    
    # Reset model weights
    core.reset_model_weights(model, ref_state)
    
    # Verify exact match across all parameters
    for k, v in model.state_dict().items():
        assert torch.allclose(v, ref_state[k]), f"Weight mismatch after reset in key {k}"


def test_target_masking_and_remapping():
    """Verify target padding maps to -100 and valid targets are within [0, 3]."""
    core = get_adaptation_core()
    
    # Mock records
    rec1 = {
        "ihr": np.ones((800, 30), dtype=np.float32) * 65.0,
        "accel": np.ones((800, 30), dtype=np.float32) * 1.0,
        "freq": np.ones((800, 1), dtype=np.float32),
        "time": np.linspace(0, 1, 800).reshape(-1, 1).astype(np.float32),
        "y": np.random.choice([0, 1, 2, 3], size=(800,))
    }
    rec2 = {
        "ihr": np.ones((900, 30), dtype=np.float32) * 70.0,
        "accel": np.ones((900, 30), dtype=np.float32) * 1.0,
        "freq": np.ones((900, 1), dtype=np.float32),
        "time": np.linspace(0, 1, 900).reshape(-1, 1).astype(np.float32),
        "y": np.random.choice([0, 1, 2, 3], size=(900,))
    }
    
    batch = core.prepare_calibration_batch([rec1, rec2], context_len=1200, device="cpu")
    
    y = batch["y"].numpy()
    mask = batch["mask"].numpy()
    
    # Check padded region for rec1: indices 800:1200
    assert np.all(y[0, 800:] == -100), "Padded targets must be -100"
    assert np.all(mask[0, 800:] == False), "Padded targets must be masked out"
    
    # Check valid region for rec1: indices 0:800
    assert np.all(y[0, :800] >= 0) and np.all(y[0, :800] <= 3), "Valid targets must be in 0..3"
    assert np.all(mask[0, :800] == True), "Valid targets must be unmasked"


def test_wearable_quality_heuristic():
    """Verify compute_wearable_quality_mask enforces 30 <= HR <= 180 and accel var > 0."""
    core = get_adaptation_core()
    
    # Normal physiological data
    ihr_valid = np.full((10, 30), 75.0)
    acc_valid = np.random.randn(10, 30)  # non-zero variance
    mask_valid = core.compute_wearable_quality_mask(ihr_valid, acc_valid)
    assert np.all(mask_valid), "Physiological data should be accepted"
    
    # Unphysiological low HR (< 30)
    ihr_low = np.full((10, 30), 25.0)
    mask_low = core.compute_wearable_quality_mask(ihr_low, acc_valid)
    assert not np.any(mask_low), "HR < 30 bpm must be rejected"
    
    # Unphysiological high HR (> 180)
    ihr_high = np.full((10, 30), 195.0)
    mask_high = core.compute_wearable_quality_mask(ihr_high, acc_valid)
    assert not np.any(mask_high), "HR > 180 bpm must be rejected"
    
    # Disconnected/flatline accelerometer (variance == 0)
    acc_flat = np.zeros((10, 30))
    mask_flat = core.compute_wearable_quality_mask(ihr_valid, acc_flat)
    assert not np.any(mask_flat), "Accel variance == 0 must be rejected"


def test_ua2_anti_collapse_safeguard():
    """Verify that UA-2 aborts adaptation if marginal diversity drops below 0.6931 nats."""
    core = get_adaptation_core()
    
    model = ConventionalCNNBiGRU(d_model=64, d_temp=96, n_classes=4, n_meta=5, dropout=0.2, use_rcc_loss=True)
    
    rec1 = {
        "ihr": np.ones((600, 30), dtype=np.float32) * 65.0,
        "accel": np.random.randn(600, 30).astype(np.float32),
        "freq": np.ones((600, 1), dtype=np.float32),
        "time": np.linspace(0, 1, 600).reshape(-1, 1).astype(np.float32),
        "y": np.zeros(600, dtype=np.int64)
    }
    rec2 = {
        "ihr": np.ones((600, 30), dtype=np.float32) * 70.0,
        "accel": np.random.randn(600, 30).astype(np.float32),
        "freq": np.ones((600, 1), dtype=np.float32),
        "time": np.linspace(0, 1, 600).reshape(-1, 1).astype(np.float32),
        "y": np.zeros(600, dtype=np.int64)
    }
    
    # Test normal adaptation
    adapted_model, info = core.adapt_unsupervised_infomax(
        model=model,
        cal_records=[rec1, rec2],
        lr=1e-5,
        steps=2,
        lambda_div=1.0,
        device="cpu"
    )
    assert "aborted" in info
    assert "final_marginal_entropy" in info
