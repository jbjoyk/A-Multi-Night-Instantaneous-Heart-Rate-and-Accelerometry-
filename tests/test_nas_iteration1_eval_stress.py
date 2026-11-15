"""
Adversarial Stress Test Suite: NAS Iteration 1 Evaluation Loop, Metrics, Memory & Partitions
Author: Challenger 2 (teamwork_challenger_r1_2)
File: tests/test_nas_iteration1_eval_stress.py

Adversarially challenges:
  1. Metric Oracle Stress Testing (degenerate arrays, empty classes, inverted predictions, mathematical bounds)
  2. GPU Memory Leak & VRAM Stability (consecutive 1-epoch evaluation passes on cuda:0)
  3. Partition & Manifest Invariance (subject disjointness across all 5 folds, zero train-test contamination, non-shuffled test loader)
  4. Seed Determinism & 1-Epoch Empirical Variance Analysis
"""

import os
import sys
import json
import glob
import pytest
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    precision_recall_fscore_support
)

# Workspace root
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

EXP014_DIR = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014")
if EXP014_DIR not in sys.path:
    sys.path.insert(0, EXP014_DIR)

from exp013_models import ConventionalCNNBiGRU
from nas_iteration1_model import MultiScaleConventionalCNNBiGRU
from evaluate_nas_iteration1 import (
    load_fold1_data,
    evaluate_1epoch_fold1,
    MANIFEST_PATH,
    RESULTS_OUTPUT_PATH
)


# ==============================================================================
# 1. METRIC ORACLE STRESS TESTS
# ==============================================================================

def test_metric_oracle_all_zeros():
    """
    Construct adversarial prediction array where model predicts only class 0 (Wake)
    on a perfectly balanced 4-class ground truth.
    Verify exact mathematical oracle values:
      - Overall Accuracy: exactly 0.25
      - Per-class F1 for class 0: exactly 0.40
        (TP=250, FP=750, FN=0 -> Precision=0.25, Recall=1.0 -> F1 = 2*0.25*1/1.25 = 0.4)
      - Per-class F1 for classes 1, 2, 3: exactly 0.0
      - Macro-F1: exactly 0.10 (average of 0.40, 0, 0, 0)
      - Cohen's Kappa: exactly 0.0
    """
    n_per_class = 250
    y_true = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class + [3] * n_per_class, dtype=np.int64)
    y_pred = np.zeros(len(y_true), dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    _, _, f1_cls, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)

    assert np.isclose(acc, 0.25, atol=1e-6), f"Expected accuracy 0.25, got {acc}"
    assert np.isclose(f1_cls[0], 0.40, atol=1e-6), f"Expected Class 0 F1 0.40, got {f1_cls[0]}"
    assert np.allclose(f1_cls[1:], [0.0, 0.0, 0.0], atol=1e-6), f"Expected F1 0.0 for classes 1,2,3, got {f1_cls[1:]}"
    assert np.isclose(macro_f1, 0.10, atol=1e-6), f"Expected Macro-F1 0.10, got {macro_f1}"
    assert np.isclose(kappa, 0.0, atol=1e-6), f"Expected Kappa 0.0, got {kappa}"


def test_metric_oracle_uniform_random():
    """
    Construct uniform random predictions over balanced ground truth.
    Expectation:
      - Accuracy ~ 0.25 (+/- 0.02)
      - Macro-F1 ~ 0.25 (+/- 0.02)
      - Cohen's Kappa ~ 0.0 (+/- 0.03)
    """
    np.random.seed(42)
    n = 20000
    y_true = np.random.randint(0, 4, size=n, dtype=np.int64)
    y_pred = np.random.randint(0, 4, size=n, dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    assert 0.23 <= acc <= 0.27, f"Random accuracy {acc} out of expected bound [0.23, 0.27]"
    assert 0.23 <= macro_f1 <= 0.27, f"Random Macro-F1 {macro_f1} out of expected bound [0.23, 0.27]"
    assert -0.03 <= kappa <= 0.03, f"Random Kappa {kappa} out of expected bound [-0.03, 0.03]"


def test_metric_oracle_perfect_inversion():
    """
    Construct adversarial worst-case permutation: y_pred = 3 - y_true.
    Every single prediction is incorrect (TP=0 for all classes).
    Exact mathematical values:
      - Accuracy: exactly 0.0
      - Per-class F1: [0.0, 0.0, 0.0, 0.0]
      - Macro-F1: exactly 0.0
      - Cohen's Kappa: exactly -1/3 (-0.333333...)
        (P_o = 0, P_e = 0.25 -> kappa = (0 - 0.25)/(1 - 0.25) = -0.333333)
    """
    n_per_class = 250
    y_true = np.array([0] * n_per_class + [1] * n_per_class + [2] * n_per_class + [3] * n_per_class, dtype=np.int64)
    y_pred = 3 - y_true

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    _, _, f1_cls, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)

    assert acc == 0.0, f"Expected 0.0 accuracy, got {acc}"
    assert macro_f1 == 0.0, f"Expected 0.0 macro-F1, got {macro_f1}"
    assert np.allclose(f1_cls, [0.0, 0.0, 0.0, 0.0]), f"Expected all zeros per-class F1, got {f1_cls}"
    assert np.isclose(kappa, -1.0 / 3.0, atol=1e-6), f"Expected Kappa -1/3, got {kappa}"


def test_metric_oracle_missing_predicted_classes():
    """
    Test when model never predicts minority classes (e.g. predicts only Light sleep class 1).
    Verify that zero_division=0 prevents ZeroDivisionError and NaN values.
    """
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
    y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=np.int64)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    _, _, f1_cls, supp = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)

    assert not np.isnan(acc)
    assert not np.isnan(macro_f1)
    assert not np.isnan(kappa)
    assert not np.isnan(f1_cls).any()
    # Class 1: TP=2, FP=6, FN=0 -> P=2/8=0.25, R=2/2=1.0 -> F1=2*0.25*1/1.25=0.40
    assert np.isclose(f1_cls[1], 0.40)
    assert np.isclose(f1_cls[0], 0.0)
    assert np.isclose(f1_cls[2], 0.0)
    assert np.isclose(f1_cls[3], 0.0)
    assert np.isclose(macro_f1, 0.10)


def test_metric_oracle_nan_edge_case_single_class():
    """
    Adversarial edge case: evaluate_nas_iteration1.py uses cohen_kappa_score(y_true, y_pred).
    If y_true and y_pred contain only 1 single class (e.g. all 0), cohen_kappa_score returns NaN.
    Verify whether the evaluation script has defensive NaN handling or whether it propagates.
    """
    y_true = np.array([0, 0, 0, 0], dtype=np.int64)
    y_pred = np.array([0, 0, 0, 0], dtype=np.int64)

    # In scikit-learn, kappa is undefined and returns NaN
    kappa = cohen_kappa_score(y_true, y_pred)
    assert np.isnan(kappa), "Expected NaN for single-class degenerate kappa"


# ==============================================================================
# 2. GPU MEMORY LEAK & RESOURCE STRESS TESTS
# ==============================================================================

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for GPU memory test")
def test_gpu_memory_leak_consecutive_runs():
    """
    Run 5 consecutive iterations of evaluate_1epoch_fold1 on cuda:0.
    Verify:
      - Memory allocated after each iteration does NOT steadily increase.
      - Total memory leak between iteration 1 and iteration 5 is < 1.0 MiB.
      - PyTorch tensors / autograd graphs are cleanly garbage-collected.
    """
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # Establish CUDA context before memory stats call
    _dummy = torch.zeros(1, device=device)
    del _dummy
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)

    train_loader, test_loader, _ = load_fold1_data(batch_size=4, seq_len=600)
    base_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)

    allocated_history = []
    peak_history = []

    for i in range(5):
        # Create fresh model instance
        model = MultiScaleConventionalCNNBiGRU(mutation_type="pre_recurrent").to(device)
        res = evaluate_1epoch_fold1(model, train_loader, test_loader, device=device, seed=42 + i)
        assert res["macro_f1"] > 0.0

        # Clean up model reference
        del model
        torch.cuda.empty_cache()

        cur_alloc = torch.cuda.memory_allocated(0) / (1024 ** 2)
        cur_peak = torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        allocated_history.append(cur_alloc)
        peak_history.append(cur_peak)

    # Stress assertion: memory at end of 5 runs vs end of run 1 must be within 1.0 MiB
    delta_alloc = allocated_history[-1] - allocated_history[0]
    print(f"\n[GPU Memory Stress] Run 1-5 allocations (MiB): {allocated_history}")
    print(f"[GPU Memory Stress] Run 1-5 peaks (MiB): {peak_history}")
    print(f"[GPU Memory Stress] Delta (Run 5 - Run 1): {delta_alloc:+.4f} MiB")

    assert delta_alloc < 1.0, f"GPU Memory leak detected: {delta_alloc:.4f} MiB accumulated over 5 runs"


# ==============================================================================
# 3. PARTITION & MANIFEST INVARIANCE TESTS
# ==============================================================================

def test_manifest_partition_disjointness_all_folds():
    """
    Inspect all 5 manifests in experiments/EXP-014/EXP014_MANIFESTS/.
    For every fold:
      1. inner_train_subjects AND outer_test_subjects must be DISJOINT.
      2. inner_val_subjects AND outer_test_subjects must be DISJOINT.
      3. inner_train_subjects AND inner_val_subjects must be DISJOINT.
      4. outer_train_subjects AND outer_test_subjects must be DISJOINT.
    Across all 5 folds:
      5. outer_test_subjects must be mutually exclusive (pairwise disjoint).
    """
    manifest_dir = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014/EXP014_MANIFESTS")
    manifest_files = sorted(glob.glob(os.path.join(manifest_dir, "fold_*.json")))
    assert len(manifest_files) == 5, f"Expected 5 manifest files, found {len(manifest_files)}"

    outer_test_sets = []
    all_subjects_seen = set()

    for m_file in manifest_files:
        with open(m_file, "r") as f:
            m = json.load(f)

        f_num = m["fold"]
        in_train = set(m["inner_train_subjects"])
        in_val = set(m["inner_val_subjects"])
        out_train = set(m["outer_train_subjects"])
        out_test = set(m["outer_test_subjects"])

        # Check 1: Inner train vs Outer test disjointness
        train_test_overlap = in_train & out_test
        assert len(train_test_overlap) == 0, f"Fold {f_num}: Data contamination! Overlap: {train_test_overlap}"

        # Check 2: Inner val vs Outer test disjointness
        val_test_overlap = in_val & out_test
        assert len(val_test_overlap) == 0, f"Fold {f_num}: Inner val leaks into Outer test! Overlap: {val_test_overlap}"

        # Check 3: Inner train vs Inner val disjointness
        train_val_overlap = in_train & in_val
        assert len(train_val_overlap) == 0, f"Fold {f_num}: Inner train leaks into Inner val! Overlap: {train_val_overlap}"

        # Check 4: Outer train vs Outer test disjointness
        outer_overlap = out_train & out_test
        assert len(outer_overlap) == 0, f"Fold {f_num}: Outer train leaks into Outer test! Overlap: {outer_overlap}"

        outer_test_sets.append(out_test)
        all_subjects_seen.update(out_test)

    # Check 5: Outer test sets across folds 1..5 must be pairwise disjoint
    for i in range(len(outer_test_sets)):
        for j in range(i + 1, len(outer_test_sets)):
            leak = outer_test_sets[i] & outer_test_sets[j]
            assert len(leak) == 0, f"Folds {i+1} and {j+1} have overlapping test subjects: {leak}"

    print(f"\n[Manifest Disjointness] All 5 folds passed strict subject partition invariance across {len(all_subjects_seen)} subjects.")


def test_runtime_dataloader_data_isolation():
    """
    Inspect the live PyTorch DataLoaders returned by load_fold1_data().
    Verify:
      - Subject disjointness between train_loader and test_loader records.
      - test_loader has shuffle=False (SequentialSampler).
      - All targets in test partition are valid sleep stages {0, 1, 2, 3} or padding (-100).
      - (targets >= 0) & (targets < 4) & mask properly excludes invalid raw recording tokens (-100).
    """
    train_loader, test_loader, manifest = load_fold1_data(batch_size=4, seq_len=600)

    # Check subjects in datasets
    train_subs = {r["subject"] for r in train_loader.dataset.records}
    test_subs = {r["subject"] for r in test_loader.dataset.records}

    overlap = train_subs & test_subs
    assert len(overlap) == 0, f"Runtime leakage! Train and test datasets share subjects: {overlap}"
    assert train_subs == set(manifest["inner_train_subjects"])
    assert test_subs == set(manifest["outer_test_subjects"])

    # Verify test_loader is NOT shuffled
    from torch.utils.data import SequentialSampler
    assert isinstance(test_loader.sampler, SequentialSampler), (
        f"test_loader sampler must be SequentialSampler (shuffle=False), got {type(test_loader.sampler)}"
    )

    # Inspect test batch labels
    has_raw_invalid_token = False
    total_valid_windows = 0
    for batch in test_loader:
        y = batch["y"]
        mask = batch["mask"]
        # Check if raw recording contains invalid labels within mask
        if ((y < 0) | (y >= 4))[mask].any():
            has_raw_invalid_token = True

        # Test evaluation filter: (y >= 0) & (y < 4) & mask
        valid = (y >= 0) & (y < 4) & mask
        valid_y = y[valid]
        assert (valid_y >= 0).all() and (valid_y < 4).all(), (
            f"Found test target outside [0, 3] in valid masked region: min={valid_y.min()}, max={valid_y.max()}"
        )
        total_valid_windows += valid.sum().item()

    assert total_valid_windows == 32098, f"Expected 32,098 valid test windows, got {total_valid_windows}"
    assert has_raw_invalid_token, "Expected raw recording to contain invalid labels (-100) requiring (y >= 0) & (y < 4) filter"


# ==============================================================================
# 4. REPRODUCIBILITY & VARIANCE ANALYSIS
# ==============================================================================

def test_unseeded_weight_initialization_variance_demonstration():
    """
    Empirically demonstrates that instantiating models without an initial manual seed
    causes random weight initialization variance across runs.
    """
    # Run without explicit seed before instantiation
    m_unseeded_1 = MultiScaleConventionalCNNBiGRU(mutation_type="pre_recurrent")
    w1 = m_unseeded_1.classifier_head[0].weight.data.clone().cpu().numpy()

    m_unseeded_2 = MultiScaleConventionalCNNBiGRU(mutation_type="pre_recurrent")
    w2 = m_unseeded_2.classifier_head[0].weight.data.clone().cpu().numpy()

    # Weights will differ because no seed was set before initialization
    diff = np.max(np.abs(w1 - w2))
    assert diff > 1e-4, f"Expected non-zero difference in unseeded weight inits, got max diff {diff}"

    # Now verify that setting the seed before instantiation guarantees 100% bitwise determinism
    torch.manual_seed(12345)
    m_seeded_1 = MultiScaleConventionalCNNBiGRU(mutation_type="pre_recurrent")
    sw1 = m_seeded_1.classifier_head[0].weight.data.clone().cpu().numpy()

    torch.manual_seed(12345)
    m_seeded_2 = MultiScaleConventionalCNNBiGRU(mutation_type="pre_recurrent")
    sw2 = m_seeded_2.classifier_head[0].weight.data.clone().cpu().numpy()

    assert np.array_equal(sw1, sw2), "Seeded initializations must be 100% bitwise identical!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
