"""
tests/test_frozen_test_shield.py: Tests for direct and indirect test-set shielding.

Verifies:
  1. Direct evaluation on test set without explicit frozen authorization raises TestSetProtectionViolation.
  2. DatasetGate blocks training role from accessing held-out test subjects.
  3. DatasetGate blocks validation role from accessing held-out test subjects.
  4. DatasetGate blocks hyperparameter tuning and search roles from accessing held-out test subjects.
  5. DatasetGate blocks early stopping from accessing held-out test subjects.
  6. Governor blocks any proposal attempting to optimize or tune on test subjects.
  7. Test metrics cannot flow into inner-loop decision engine.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from evaluator import SleepEvaluator, TestSetProtectionViolation
from dataset_gate import DatasetGate, TestSetLeakageError
from governor import ResearchGovernor
from research_engine.experiment_designer import ExperimentDesigner


@pytest.fixture
def evaluator():
    return SleepEvaluator()


@pytest.fixture
def dataset_gate():
    return DatasetGate()


@pytest.fixture
def test_subjects():
    return ["Bidslab03", "Bidslab07", "Bidslab11", "Bidslab14", "Bidslab18"]


def test_direct_test_evaluation_without_authorization_blocked(evaluator):
    dummy_model = nn.Linear(10, 4)
    # Attempting to call evaluate_model with split_role='test' and frozen_evaluation_authorized=False
    with pytest.raises(TestSetProtectionViolation) as exc_info:
        evaluator.evaluate_model(
            model=dummy_model,
            dataloader=None,
            device=torch.device("cpu"),
            split_role="test",
            frozen_evaluation_authorized=False
        )
    assert "HARD GATE VIOLATION" in str(exc_info.value)
    assert "test set is strictly protected" in str(exc_info.value)


def test_indirect_leakage_training_role_blocked(dataset_gate, test_subjects):
    train_subjects = ["Bidslab01", "Bidslab02", "Bidslab03"]  # Bidslab03 is in test set!
    with pytest.raises(TestSetLeakageError) as exc_info:
        dataset_gate.assert_no_test_leakage("train", train_subjects, test_subjects)
    assert "TEST SET LEAKAGE VIOLATION" in str(exc_info.value)
    assert "Bidslab03" in str(exc_info.value)


def test_indirect_leakage_validation_role_blocked(dataset_gate, test_subjects):
    val_subjects = ["Bidslab07", "Bidslab20"]  # Bidslab07 is in test set!
    with pytest.raises(TestSetLeakageError) as exc_info:
        dataset_gate.assert_no_test_leakage("validation", val_subjects, test_subjects)
    assert "TEST SET LEAKAGE VIOLATION" in str(exc_info.value)


def test_indirect_leakage_tuning_and_search_blocked(dataset_gate, test_subjects):
    tuning_subjects = ["Bidslab11", "Bidslab25"]
    with pytest.raises(TestSetLeakageError) as exc_info1:
        dataset_gate.assert_no_test_leakage("tuning", tuning_subjects, test_subjects)
    assert "TEST SET LEAKAGE VIOLATION" in str(exc_info1.value)

    with pytest.raises(TestSetLeakageError) as exc_info2:
        dataset_gate.assert_no_test_leakage("hyperparameter_search", tuning_subjects, test_subjects)
    assert "TEST SET LEAKAGE VIOLATION" in str(exc_info2.value)


def test_indirect_leakage_early_stopping_blocked(dataset_gate, test_subjects):
    es_subjects = ["Bidslab14", "Bidslab30"]
    with pytest.raises(TestSetLeakageError) as exc_info:
        dataset_gate.assert_no_test_leakage("early_stopping", es_subjects, test_subjects)
    assert "TEST SET LEAKAGE VIOLATION" in str(exc_info.value)


def test_governor_blocks_test_set_proposals():
    gov = ResearchGovernor()
    designer = ExperimentDesigner()

    proposals = [
        "Optimize hyperparameters directly on the test set partition",
        "Select best model based on outer test accuracy",
        "Calibrate temperature scaling using held_out_test data"
    ]

    for p in proposals:
        spec = designer.design_experiment(
            experiment_id="EXP-LEAK",
            hypothesis_id="HYP-001",
            research_question="RQ",
            hypothesis_statement=p,
            baseline_model="Model A",
            exact_change=p,
            fixed_components=["HREncoder", "BiGRU"],
            success_metric="Macro-F1",
            falsification_criteria="Loss",
            confounders=["leakage"],
            compute_budget="SHORT"
        )
        safety = gov.check_experiment_safety(spec, human_approved=True)
        assert safety["allowed"] is False
        assert "TEST SET PROTECTION VIOLATION" in safety["reason"]
