"""
tests/test_scientific_reviewer.py: Unit tests for ScientificReviewer and Peer Review Audits.
"""

import pytest
from research_engine.experiment_designer import ExperimentDesigner, ExperimentDesignError
from research_engine.scientific_reviewer import ScientificReviewer, ReviewVerdict


def test_reviewer_accepts_clean_proposal():
    designer = ExperimentDesigner()
    reviewer = ScientificReviewer()

    clean_spec = designer.design_experiment(
        experiment_id="EXP-015",
        hypothesis_id="HYP-001",
        research_question="RQ-001: Pruning dummy clock channel",
        hypothesis_statement="Removing dummy zeros clock will maintain Macro-F1 parity.",
        baseline_model="Model A",
        exact_change="Prune dummy zeros clock from x_meta",
        fixed_components=["HREncoder", "ACCEncoder", "BiGRU", "RCCLoss"],
        success_metric="Validation Macro-F1 delta >= -0.003",
        falsification_criteria="Validation Macro-F1 degrades by > 0.010 or loss diverges.",
        confounders=["inter-subject variance"],
        compute_budget="SHORT"
    )

    review = reviewer.review_proposal(clean_spec)
    assert review["verdict"] == ReviewVerdict.ACCEPT.value
    assert review["checks_passed"] == 10


def test_reviewer_rejects_unanchored_baseline():
    designer = ExperimentDesigner()
    reviewer = ScientificReviewer()

    bad_baseline_spec = designer.design_experiment(
        experiment_id="EXP-BAD-BASE",
        hypothesis_id="HYP-001",
        research_question="RQ-001",
        hypothesis_statement="Compare with an unverified random network.",
        baseline_model="RandomCustomNetXYZ",  # unverified baseline
        exact_change="Change layer count",
        fixed_components=["HREncoder", "BiGRU"],
        success_metric="Macro-F1 > 0.60",
        falsification_criteria="Loss diverges",
        confounders=["baseline mismatch"],
        compute_budget="SHORT"
    )

    review = reviewer.review_proposal(bad_baseline_spec)
    assert review["verdict"] == ReviewVerdict.REJECT.value
    assert any("CHECK 2 FAILED" in f for f in review["flags"])


def test_designer_rejects_multi_variable_convoluted_change():
    designer = ExperimentDesigner()

    with pytest.raises(ExperimentDesignError) as exc_info:
        designer.design_experiment(
            experiment_id="EXP-CONFOUNDED",
            hypothesis_id="HYP-001",
            research_question="RQ-001",
            hypothesis_statement="Change everything at once.",
            baseline_model="Model A",
            exact_change="Double learning rate and also add new attention layer together with dropout modification",
            fixed_components=["HREncoder"],
            success_metric="Macro-F1",
            falsification_criteria="Loss diverges",
            confounders=["multi-variable"],
            compute_budget="SHORT"
        )
    assert "Multi-variable confounder detected" in str(exc_info.value)
