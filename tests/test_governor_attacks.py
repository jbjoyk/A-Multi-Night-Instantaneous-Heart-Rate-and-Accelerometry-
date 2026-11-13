"""
tests/test_governor_attacks.py: Adversarial Attack Suite testing Governor bypass attempts A through L.

Attacks:
  Attack A: 500 epochs -> BLOCKED_BY_RESEARCH_GOVERNANCE
  Attack B: >500 epochs (501 epochs, 1000 epochs) -> BLOCKED_BY_RESEARCH_GOVERNANCE
  Attack C: Unauthorized Level 4 escalation -> BLOCKED
  Attack D: Unauthorized Level 5 escalation -> BLOCKED
  Attack E: Direct runner bypass (calling engine without Governor token) -> BLOCKED
  Attack F: Fake approval token -> BLOCKED
  Attack G: Missing approval at Level 3 -> BLOCKED
  Attack H: Invalid experiment ID format -> BLOCKED
  Attack I: Invalid dataset fingerprint -> BLOCKED
  Attack J: Unregistered split / non-disjoint split -> BLOCKED
  Attack K: Duplicate experiment ID -> BLOCKED
  Attack L: Invalid compute budget / negative epochs -> BLOCKED
"""

import pytest
from governor import ResearchGovernor, AutonomyLevel, GovernanceBlockError
from research_engine.experiment_designer import ExperimentDesigner
from research_engine.experiment_engine import ExperimentEngine
from research_engine.hypothesis_agent import HypothesisAgent
from dataset_gate import SplitDisjointnessError, DatasetIntegrityError


@pytest.fixture
def governor():
    return ResearchGovernor()


@pytest.fixture
def designer():
    return ExperimentDesigner()


@pytest.fixture
def valid_spec(designer):
    return designer.design_experiment(
        experiment_id="EXP-991",
        hypothesis_id="HYP-001",
        research_question="RQ-001",
        hypothesis_statement="Test statement",
        baseline_model="Model A",
        exact_change="Prune dummy clock feature",
        fixed_components=["HREncoder", "BiGRU"],
        success_metric="Macro-F1",
        falsification_criteria="Loss diverges",
        confounders=["variance"],
        compute_budget="SHORT"
    )


# Attack A: 500 epochs
def test_attack_a_500_epochs(governor, designer):
    spec = designer.design_experiment(
        experiment_id="EXP-992",
        hypothesis_id="HYP-001",
        research_question="RQ-001",
        hypothesis_statement="Train for 500 epochs exactly",
        baseline_model="Model A",
        exact_change="Train for 500 epochs",
        fixed_components=["HREncoder", "BiGRU"],
        success_metric="Macro-F1",
        falsification_criteria="Loss diverges",
        confounders=["variance"],
        compute_budget="STANDARD"
    )
    safety = governor.check_experiment_safety(spec, human_approved=True)
    assert safety["allowed"] is False
    assert safety["status"] == "BLOCKED_BY_RESEARCH_GOVERNANCE"
    assert "500-epoch training paused under DEC-019/020" in safety["reason"]


# Attack B: >500 epochs (501 epochs, 1000 epochs)
def test_attack_b_greater_than_500_epochs(governor, designer):
    # Case 1: 501 epochs
    spec_501 = designer.design_experiment(
        experiment_id="EXP-993",
        hypothesis_id="HYP-001",
        research_question="RQ-001",
        hypothesis_statement="Train for 501 epochs to probe post-convergence stability",
        baseline_model="Model A",
        exact_change="Train for 501 epochs",
        fixed_components=["HREncoder", "BiGRU"],
        success_metric="Macro-F1",
        falsification_criteria="Loss diverges",
        confounders=["variance"],
        compute_budget="STANDARD"
    )
    safety_501 = governor.check_experiment_safety(spec_501, human_approved=True)
    assert safety_501["allowed"] is False
    assert safety_501["status"] == "BLOCKED_BY_RESEARCH_GOVERNANCE"
    assert "Training >= 500 epochs is paused" in safety_501["reason"]

    # Case 2: 1000 epochs passed numerically
    spec_1000 = designer.design_experiment(
        experiment_id="EXP-994",
        hypothesis_id="HYP-001",
        research_question="RQ-001",
        hypothesis_statement="Train for 1000 epochs massive run",
        baseline_model="Model A",
        exact_change="Train for 1000 epochs",
        fixed_components=["HREncoder", "BiGRU"],
        success_metric="Macro-F1",
        falsification_criteria="Loss diverges",
        confounders=["variance"],
        compute_budget="STANDARD"
    )
    safety_1000 = governor.check_experiment_safety(spec_1000, human_approved=True, requested_epochs=1000)
    assert safety_1000["allowed"] is False
    assert safety_1000["status"] == "BLOCKED_BY_RESEARCH_GOVERNANCE"


# Attack C: Unauthorized Level 4
def test_attack_c_unauthorized_level_4(governor):
    with pytest.raises(GovernanceBlockError) as exc_info:
        governor.escalate_autonomy(AutonomyLevel.LEVEL_4_SUPERVISED_EXECUTION, human_approval_token=None)
    assert "REJECTED" in str(exc_info.value)

    with pytest.raises(GovernanceBlockError) as exc_info2:
        governor.escalate_autonomy(AutonomyLevel.LEVEL_4_SUPERVISED_EXECUTION, human_approval_token="")
    assert "REJECTED" in str(exc_info2.value)


# Attack D: Unauthorized Level 5
def test_attack_d_unauthorized_level_5(governor):
    with pytest.raises(GovernanceBlockError) as exc_info:
        governor.escalate_autonomy(AutonomyLevel.LEVEL_5_AUTONOMOUS_LOOP, human_approval_token=None)
    assert "REJECTED" in str(exc_info.value)

    with pytest.raises(GovernanceBlockError) as exc_info2:
        governor.escalate_autonomy(AutonomyLevel.LEVEL_5_AUTONOMOUS_LOOP, human_approval_token="UNAUTHORIZED_HACK")
    assert "REJECTED" in str(exc_info2.value)


# Attack E: Direct runner bypass
def test_attack_e_direct_runner_bypass(governor):
    engine = ExperimentEngine()
    hyp = HypothesisAgent().get_grounded_candidates()[0]

    # Calling engine.execute_lifecycle_step directly without Governor token MUST raise GovernanceBlockError
    with pytest.raises(GovernanceBlockError) as exc_info:
        engine.execute_lifecycle_step(
            hypothesis=hyp,
            experiment_id="EXP-BYPASS-001",
            parent_experiment="EXP-014",
            baseline_model="Model A",
            exact_change="Attempt direct bypass without Governor token",
            fixed_components=["HREncoder", "BiGRU"],
            compute_budget="SHORT",
            dry_run=True,
            governance_token=None  # Missing token!
        )
    assert "DIRECT RUNNER BYPASS BLOCKED" in str(exc_info.value)


# Attack F: Fake approval token
def test_attack_f_fake_approval_token(governor):
    # Fake human token
    with pytest.raises(GovernanceBlockError) as exc_info:
        governor.escalate_autonomy(AutonomyLevel.LEVEL_4_SUPERVISED_EXECUTION, human_approval_token="FORGED_SECRET_123")
    assert "REJECTED: Invalid or missing human authorization token" in str(exc_info.value)

    # Fake governor execution token
    engine = ExperimentEngine()
    hyp = HypothesisAgent().get_grounded_candidates()[0]
    with pytest.raises(GovernanceBlockError) as exc_info2:
        engine.execute_lifecycle_step(
            hypothesis=hyp,
            experiment_id="EXP-BYPASS-002",
            parent_experiment="EXP-014",
            baseline_model="Model A",
            exact_change="Attempt execution with fake token",
            fixed_components=["HREncoder", "BiGRU"],
            compute_budget="SHORT",
            dry_run=True,
            governance_token="fake_forged_governor_token_hash"
        )
    assert "DIRECT RUNNER BYPASS BLOCKED" in str(exc_info2.value)


# Attack G: Missing approval at Level 3
def test_attack_g_missing_approval_at_level_3(governor, valid_spec):
    assert governor.autonomy_level == AutonomyLevel.LEVEL_3_IMPLEMENT_REQUIRE_APPROVAL
    # Human approval not given
    safety = governor.check_experiment_safety(valid_spec, human_approved=False)
    assert safety["allowed"] is False
    assert safety["status"] == "BLOCKED_PENDING_APPROVAL"
    assert "requires explicit human approval before execution" in safety["reason"]


# Attack H: Invalid experiment ID
def test_attack_h_invalid_experiment_id(governor, designer):
    # Invalid ID: non-standard format
    for bad_id in ["exp-1", "HACK_EXP", "EXP-99999-EXTRA-LONG-INVALID", ""]:
        try:
            bad_spec = designer.design_experiment(
                experiment_id=bad_id,
                hypothesis_id="HYP-001",
                research_question="RQ",
                hypothesis_statement="Test",
                baseline_model="Model A",
                exact_change="Prune dummy clock feature",
                fixed_components=["HREncoder", "BiGRU"],
                success_metric="Macro-F1",
                falsification_criteria="Loss",
                confounders=["v"],
                compute_budget="SHORT"
            )
            safety = governor.check_experiment_safety(bad_spec, human_approved=True)
            assert safety["allowed"] is False
            assert "Invalid experiment ID" in safety["reason"]
        except Exception as e:
            # Rejection during design or check is acceptable
            assert "Invalid" in str(e) or "experiment ID" in str(e)


# Attack I: Invalid dataset fingerprint
def test_attack_i_invalid_dataset_fingerprint(governor, valid_spec, monkeypatch):
    # Mock dataset gate failure
    def mock_verify_structure():
        raise DatasetIntegrityError("Corrupted dataset directory structure!")
    monkeypatch.setattr(governor.dataset_gate, "verify_dataset_structure", mock_verify_structure)

    safety = governor.check_experiment_safety(valid_spec, human_approved=True)
    assert safety["allowed"] is False
    assert safety["status"] == "BLOCKED_BY_DATASET_GATE"
    assert "Corrupted dataset" in safety["reason"]


# Attack J: Unregistered split / non-disjoint split
def test_attack_j_unregistered_split(governor):
    # Overlapping train and test partition
    with pytest.raises(SplitDisjointnessError) as exc_info:
        governor.dataset_gate.verify_split(
            train_subjects=["Bidslab01", "Bidslab02"],
            val_subjects=["Bidslab03"],
            test_subjects=["Bidslab01"]  # Overlap!
        )
    assert "Subject overlap between Train and Test" in str(exc_info.value)


# Attack K: Duplicate experiment
def test_attack_k_duplicate_experiment(governor, designer):
    # Attempting to re-run EXP-014 (already archived in research_memory.json)
    dup_spec = designer.design_experiment(
        experiment_id="EXP-014",
        hypothesis_id="HYP-001",
        research_question="RQ",
        hypothesis_statement="Test duplicate",
        baseline_model="Model A",
        exact_change="Prune dummy clock feature",
        fixed_components=["HREncoder", "BiGRU"],
        success_metric="Macro-F1",
        falsification_criteria="Loss",
        confounders=["v"],
        compute_budget="SHORT"
    )
    safety = governor.check_experiment_safety(dup_spec, human_approved=True)
    assert safety["allowed"] is False
    assert safety["status"] == "BLOCKED_DUPLICATE_EXPERIMENT"
    assert "already exists in research memory" in safety["reason"]


# Attack L: Invalid budget & negative epochs
def test_attack_l_invalid_budget_and_negative_epochs(governor, designer, valid_spec):
    # Negative epoch count
    safety_neg = governor.check_experiment_safety(valid_spec, human_approved=True, requested_epochs=-5)
    assert safety_neg["allowed"] is False
    assert safety_neg["status"] == "BLOCKED_BY_RESEARCH_GOVERNANCE"
    assert "negative epoch" in safety_neg["reason"].lower()

    # Budget exceeding ceiling (e.g. 50 epochs in SHORT budget where ceiling is 10)
    spec_exceed = designer.design_experiment(
        experiment_id="EXP-995",
        hypothesis_id="HYP-001",
        research_question="RQ",
        hypothesis_statement="50 epochs under SHORT budget",
        baseline_model="Model A",
        exact_change="Train for 50 epochs",
        fixed_components=["HREncoder", "BiGRU"],
        success_metric="Macro-F1",
        falsification_criteria="Loss",
        confounders=["v"],
        compute_budget="SHORT"
    )
    safety_exceed = governor.check_experiment_safety(spec_exceed, human_approved=True)
    assert safety_exceed["allowed"] is False
    assert safety_exceed["status"] == "BLOCKED_BY_RESEARCH_GOVERNANCE"
    assert "exceeds ceiling" in safety_exceed["reason"]
