"""
tests/test_evaluator.py: Unit tests for SleepEvaluator, Metric Hierarchy, and Test-Set Gate.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from evaluator import SleepEvaluator, MetricHierarchy, TestSetProtectionViolation, compute_ece


def test_metric_hierarchy_definitions():
    assert MetricHierarchy.PRIMARY == "macro_f1"
    assert "accuracy" in MetricHierarchy.SECONDARIES
    assert "kappa" in MetricHierarchy.SECONDARIES
    assert "confusion_matrix" in MetricHierarchy.DIAGNOSTICS


def test_evaluator_predictions_math():
    evaluator = SleepEvaluator()
    rng = np.random.RandomState(42)
    n = 400
    y_true = rng.randint(0, 4, size=n)
    y_pred = y_true.copy()
    # Perfect predictions test
    res = evaluator.evaluate_predictions(y_true, y_pred)
    assert res["primary"]["macro_f1"] == 1.0
    assert res["secondary"]["accuracy"] == 1.0
    assert res["secondary"]["kappa"] == 1.0

    # Introduce errors
    y_pred[:100] = (y_true[:100] + 1) % 4
    res_err = evaluator.evaluate_predictions(y_true, y_pred)
    assert 0.70 <= res_err["primary"]["macro_f1"] <= 0.80
    assert res_err["secondary"]["accuracy"] == 0.75


def test_compute_ece():
    n = 200
    probs = np.full((n, 4), 0.25)
    labels = np.zeros(n, dtype=int)
    ece = compute_ece(probs, labels, n_bins=10)
    assert isinstance(ece, float)
    assert 0.0 <= ece <= 1.0


def test_test_set_protection_gate():
    evaluator = SleepEvaluator()
    dummy_model = nn.Linear(5, 4)

    # Attempt to evaluate on test set without frozen authorization
    with pytest.raises(TestSetProtectionViolation) as exc_info:
        evaluator.evaluate_model(
            model=dummy_model,
            dataloader=None,
            device=torch.device("cpu"),
            split_role="test",
            frozen_evaluation_authorized=False
        )
    assert "HARD GATE VIOLATION" in str(exc_info.value)
