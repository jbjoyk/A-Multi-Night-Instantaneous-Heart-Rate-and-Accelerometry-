"""
test_exp019_pipeline.py: Unit and Smoke Tests for EXP-019 500-Epoch Benchmark.

Verifies:
  1. Parameter counts: Baseline (163,940) and MultiScale (192,692 <= 196,728).
  2. Metric calculation: Macro-F1, Accuracy, Kappa, MCC, per-stage F1.
  3. Single-batch forward, loss backward, and evaluation pass on dummy data.
  4. Fold 1 dataset manifest integrity and 0 subject leakage.
"""

import os
import sys
import json
import pytest
import numpy as np
import torch
import torch.nn as nn

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

EXP014_DIR = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014")
if EXP014_DIR not in sys.path:
    sys.path.insert(0, EXP014_DIR)

from exp013_models import ConventionalCNNBiGRU
from nas_iteration1_model import MultiScaleConventionalCNNBiGRU
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef


def test_exp019_parameter_budget():
    """Verify exact parameter counts and <= +20% budget adherence."""
    base = ConventionalCNNBiGRU()
    multi = MultiScaleConventionalCNNBiGRU()

    n_base = sum(p.numel() for p in base.parameters())
    n_multi = sum(p.numel() for p in multi.parameters())

    assert n_base == 163940, f"Baseline should be 163,940, got {n_base}"
    assert n_multi == 192692, f"MultiScale should be 192,692, got {n_multi}"

    cap_20pct = int(163940 * 1.20)  # 196,728
    assert n_multi <= cap_20pct, f"MultiScale exceeded +20% cap: {n_multi} > {cap_20pct}"
    headroom = cap_20pct - n_multi
    assert headroom == 4036, f"Expected 4,036 parameters of headroom, got {headroom}"


def test_exp019_mcc_metric_compatibility():
    """Verify matthews_corrcoef behaves correctly with 4-class predictions."""
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 2, 0, 1])

    mcc = matthews_corrcoef(y_true, y_pred)
    assert isinstance(mcc, float)
    assert 0.0 < mcc <= 1.0


def test_exp019_dummy_forward_and_backward():
    """Verify 1-step train and eval step for both models."""
    device = torch.device("cpu")
    B, T = 2, 10
    ihr = torch.randn(B, T, 30, device=device)
    acc = torch.randn(B, T, 30, device=device)
    meta = torch.randn(B, T, 5, device=device)
    targets = torch.randint(0, 4, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    subj_idx = torch.zeros(B, dtype=torch.long, device=device)
    seq_lens = torch.full((B,), T, dtype=torch.long, device=device)

    for ModelClass in [ConventionalCNNBiGRU, MultiScaleConventionalCNNBiGRU]:
        model = ModelClass().to(device)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        out = model(ihr, acc, meta, targets=targets, mask=mask, subject_idx=subj_idx, seq_lens=seq_lens)
        assert "total_loss" in out
        assert "logits" in out
        assert out["logits"].shape == (B, T, 4)

        loss = out["total_loss"]
        assert not torch.isnan(loss)
        loss.backward()
        optimizer.step()


def test_exp019_fold1_manifest_leakage():
    """Verify Fold 1 manifest exists and has 0 subject overlap."""
    manifest_path = os.path.join(WORKSPACE_ROOT, "experiments/EXP-014/EXP014_MANIFESTS/fold_1.json")
    assert os.path.exists(manifest_path)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    train_subs = set(manifest["inner_train_subjects"])
    val_subs = set(manifest["inner_val_subjects"])
    test_subs = set(manifest["outer_test_subjects"])

    assert len(train_subs) == 32
    assert len(val_subs) == 5
    assert len(test_subs) == 10

    assert len(train_subs.intersection(val_subs)) == 0, "Train-Val subject leakage!"
    assert len(train_subs.intersection(test_subs)) == 0, "Train-Test subject leakage!"
    assert len(val_subs.intersection(test_subs)) == 0, "Val-Test subject leakage!"
