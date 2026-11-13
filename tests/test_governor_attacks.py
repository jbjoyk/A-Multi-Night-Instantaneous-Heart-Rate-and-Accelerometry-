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
import time
import hashlib
import hmac
from governor import ResearchGovernor, AutonomyLevel, GovernanceBlockError
from research_engine.experiment_designer import ExperimentDesigner
from research_engine.experiment_engine import ExperimentEngine
from research_engine.hypothesis_agent import HypothesisAgent
from dataset_gate import SplitDisjointnessError, DatasetIntegrityError


@pytest.fixture
def governor():
    return ResearchGovernor()


@pytest.fixture
def isolated_engine(tmp_path, governor):
    import json
    mem_file = str(tmp_path / "test_memory.json")
    with open(mem_file, "w") as f:
        json.dump({"experiments": []}, f)
    return ExperimentEngine(
        base_dir=str(tmp_path),
        memory_path=mem_file,
        dataset_gate=governor.dataset_gate
    )


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
def test_attack_e_direct_runner_bypass(governor, isolated_engine):
    hyp = HypothesisAgent().get_grounded_candidates()[0]

    # Calling engine.execute_lifecycle_step directly without Governor token MUST raise GovernanceBlockError
    with pytest.raises(GovernanceBlockError) as exc_info:
        isolated_engine.execute_lifecycle_step(
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
def test_attack_f_fake_approval_token(governor, isolated_engine):
    # Fake human token
    with pytest.raises(GovernanceBlockError) as exc_info:
        governor.escalate_autonomy(AutonomyLevel.LEVEL_4_SUPERVISED_EXECUTION, human_approval_token="FORGED_SECRET_123")
    assert "REJECTED: Invalid or missing human authorization token" in str(exc_info.value)

    # Fake governor execution token
    hyp = HypothesisAgent().get_grounded_candidates()[0]
    with pytest.raises(GovernanceBlockError) as exc_info2:
        isolated_engine.execute_lifecycle_step(
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


# Numerical Bypass: Hidden and key-value epoch declarations
def test_attack_b_hidden_and_keyvalue_epochs(governor, designer):
    hidden_variants = [
        "epochs: 500",
        "num_epochs=501",
        "max_epochs: 1000",
        "5e2 epochs"
    ]
    for variant in hidden_variants:
        spec = designer.design_experiment(
            experiment_id="EXP-980",
            hypothesis_id="HYP-001",
            research_question="RQ",
            hypothesis_statement=f"Probe with hidden declaration: {variant}",
            baseline_model="Model A",
            exact_change=f"Configuration includes {variant}",
            fixed_components=["HREncoder", "BiGRU"],
            success_metric="Macro-F1",
            falsification_criteria="Loss",
            confounders=["v"],
            compute_budget="STANDARD"
        )
        safety = governor.check_experiment_safety(spec, human_approved=True)
        assert safety["allowed"] is False
        assert safety["status"] == "BLOCKED_BY_RESEARCH_GOVERNANCE"


# Numerical Bypass: Floating-point epoch declarations
def test_attack_b_floating_point_epochs(governor, designer, valid_spec):
    # Float in text
    spec_float = designer.design_experiment(
        experiment_id="EXP-981",
        hypothesis_id="HYP-001",
        research_question="RQ",
        hypothesis_statement="Probe with float epochs: 10.5 epochs",
        baseline_model="Model A",
        exact_change="Train for 10.5 epochs",
        fixed_components=["HREncoder", "BiGRU"],
        success_metric="Macro-F1",
        falsification_criteria="Loss",
        confounders=["v"],
        compute_budget="SHORT"
    )
    safety_float = governor.check_experiment_safety(spec_float, human_approved=True)
    assert safety_float["allowed"] is False
    assert safety_float["status"] == "BLOCKED_BY_RESEARCH_GOVERNANCE"
    assert "Floating-point epoch" in safety_float["reason"]

    # Float requested_epochs parameter
    safety_param = governor.check_experiment_safety(valid_spec, human_approved=True, requested_epochs=5.5)
    assert safety_param["allowed"] is False
    assert safety_param["status"] == "BLOCKED_BY_RESEARCH_GOVERNANCE"
    assert "Floating-point epoch" in safety_param["reason"]


# Numerical Bypass: Conflicting epoch declarations taking maximum
def test_attack_b_conflicting_epoch_declarations(governor, designer):
    # Mentions 10 epochs (legal for SHORT) but also 80 epochs (illegal for SHORT ceiling of 10)
    spec_conflict = designer.design_experiment(
        experiment_id="EXP-982",
        hypothesis_id="HYP-001",
        research_question="RQ",
        hypothesis_statement="Initial test mentions 10 epochs but later declares 80 epochs",
        baseline_model="Model A",
        exact_change="Start at 10 epochs and extend to 80 epochs",
        fixed_components=["HREncoder", "BiGRU"],
        success_metric="Macro-F1",
        falsification_criteria="Loss",
        confounders=["v"],
        compute_budget="SHORT"
    )
    safety_conflict = governor.check_experiment_safety(spec_conflict, human_approved=True)
    assert safety_conflict["allowed"] is False
    assert safety_conflict["status"] == "BLOCKED_BY_RESEARCH_GOVERNANCE"
    assert "exceeds ceiling" in safety_conflict["reason"]


# Token Security: Reused token blocked
def test_attack_f_reused_token_blocked(governor, isolated_engine):
    hyp = HypothesisAgent().get_grounded_candidates()[0]
    ds_fp = governor.dataset_gate.compute_dataset_fingerprint()

    token = governor.generate_governance_token("EXP-983", "SHORT", exact_change="Change", dataset_fingerprint=ds_fp)

    # First run succeeds
    isolated_engine.execute_lifecycle_step(
        hypothesis=hyp,
        experiment_id="EXP-983",
        parent_experiment="EXP-014",
        baseline_model="Model A",
        exact_change="Change",
        fixed_components=["HREncoder", "BiGRU"],
        compute_budget="SHORT",
        dry_run=True,
        governance_token=token
    )

    # Replay attempt with same token must fail
    with pytest.raises(GovernanceBlockError) as exc_info:
        isolated_engine.execute_lifecycle_step(
            hypothesis=hyp,
            experiment_id="EXP-983",
            parent_experiment="EXP-014",
            baseline_model="Model A",
            exact_change="Change",
            fixed_components=["HREncoder", "BiGRU"],
            compute_budget="SHORT",
            dry_run=True,
            governance_token=token
        )
    assert "REUSED_GOVERNOR_TOKEN" in str(exc_info.value)


# Token Security: Altered proposal after token generation blocked
def test_attack_f_altered_proposal_after_token_generation(governor, isolated_engine):
    hyp = HypothesisAgent().get_grounded_candidates()[0]
    ds_fp = governor.dataset_gate.compute_dataset_fingerprint()

    token = governor.generate_governance_token("EXP-984", "SHORT", exact_change="Original Change", dataset_fingerprint=ds_fp)

    # Attempt execution with altered exact_change
    with pytest.raises(GovernanceBlockError) as exc_info:
        isolated_engine.execute_lifecycle_step(
            hypothesis=hyp,
            experiment_id="EXP-984",
            parent_experiment="EXP-014",
            baseline_model="Model A",
            exact_change="TAMPERED_MALICIOUS_CHANGE",
            fixed_components=["HREncoder", "BiGRU"],
            compute_budget="SHORT",
            dry_run=True,
            governance_token=token
        )
    assert "ALTERED_PROPOSAL_AFTER_TOKEN_GENERATION" in str(exc_info.value)


# Token Security: Wrong experiment ID or budget blocked
def test_attack_f_wrong_experiment_id_or_budget(governor, isolated_engine):
    hyp = HypothesisAgent().get_grounded_candidates()[0]
    ds_fp = governor.dataset_gate.compute_dataset_fingerprint()

    token = governor.generate_governance_token("EXP-985", "SHORT", exact_change="Change", dataset_fingerprint=ds_fp)

    # Try applying to different experiment ID
    with pytest.raises(GovernanceBlockError) as exc_info1:
        isolated_engine.execute_lifecycle_step(
            hypothesis=hyp,
            experiment_id="EXP-986",  # Different ID!
            parent_experiment="EXP-014",
            baseline_model="Model A",
            exact_change="Change",
            fixed_components=["HREncoder", "BiGRU"],
            compute_budget="SHORT",
            dry_run=True,
            governance_token=token
        )
    assert "FORGED_SIGNATURE" in str(exc_info1.value)

    # Try applying to different budget
    with pytest.raises(GovernanceBlockError) as exc_info2:
        isolated_engine.execute_lifecycle_step(
            hypothesis=hyp,
            experiment_id="EXP-985",
            parent_experiment="EXP-014",
            baseline_model="Model A",
            exact_change="Change",
            fixed_components=["HREncoder", "BiGRU"],
            compute_budget="STANDARD",  # Different budget!
            dry_run=True,
            governance_token=token
        )
    assert "FORGED_SIGNATURE" in str(exc_info2.value)


# Token Security: Expired token blocked
def test_attack_f_expired_token_blocked(governor, isolated_engine):
    hyp = HypothesisAgent().get_grounded_candidates()[0]
    ds_fp = governor.dataset_gate.compute_dataset_fingerprint()

    # Create expired token (timestamp 4000 seconds in the past)
    expired_ts = int(time.time()) - 4000
    content_hash = hashlib.sha256(f"Change:{ds_fp}".encode("utf-8")).hexdigest()[:16]
    payload = f"EXP-987:SHORT:{governor.autonomy_level.value}:{content_hash}:{expired_ts}".encode("utf-8")
    sig = hmac.new(governor.GOVERNOR_SECRET_KEY, payload, hashlib.sha256).hexdigest()
    expired_token = f"{expired_ts}:{content_hash}:{sig}"

    with pytest.raises(GovernanceBlockError) as exc_info:
        isolated_engine.execute_lifecycle_step(
            hypothesis=hyp,
            experiment_id="EXP-987",
            parent_experiment="EXP-014",
            baseline_model="Model A",
            exact_change="Change",
            fixed_components=["HREncoder", "BiGRU"],
            compute_budget="SHORT",
            dry_run=True,
            governance_token=expired_token
        )
    assert "EXPIRED_GOVERNOR_TOKEN" in str(exc_info.value)
