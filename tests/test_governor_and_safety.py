"""
tests/test_governor_and_safety.py: Unit tests for ResearchGovernor, Autonomy Levels, 500-Epoch Rule, and GPU Allocation.
"""

import pytest
from governor import ResearchGovernor, AutonomyLevel, GovernanceBlockError
from research_engine.experiment_designer import ExperimentDesigner
from research_engine.resource_manager import GPUResourceManager, ResourceAllocationError


def test_autonomy_level_default():
    gov = ResearchGovernor()
    assert gov.autonomy_level == AutonomyLevel.LEVEL_3_IMPLEMENT_REQUIRE_APPROVAL


def test_autonomy_escalation_security():
    gov = ResearchGovernor()

    # Attempt to escalate to Level 4 without valid token
    with pytest.raises(GovernanceBlockError) as exc_info:
        gov.escalate_autonomy(AutonomyLevel.LEVEL_4_SUPERVISED_EXECUTION, human_approval_token="wrong_token")
    assert "REJECTED" in str(exc_info.value)

    # Escalation with valid token
    gov.escalate_autonomy(AutonomyLevel.LEVEL_4_SUPERVISED_EXECUTION, human_approval_token="HUMAN_AUTHORIZED_ESCALATION")
    assert gov.autonomy_level == AutonomyLevel.LEVEL_4_SUPERVISED_EXECUTION


def test_500_epoch_hard_block():
    gov = ResearchGovernor()
    designer = ExperimentDesigner()

    bad_spec = designer.design_experiment(
        experiment_id="EXP-BAD-500",
        hypothesis_id="HYP-001",
        research_question="RQ-001",
        hypothesis_statement="Train for 500 epochs to confirm long-term convergence.",
        baseline_model="Model A",
        exact_change="Increase training epochs to 500",
        fixed_components=["HREncoder", "BiGRU"],
        success_metric="Macro-F1 > 0.70",
        falsification_criteria="Loss diverges",
        confounders=["overfitting"],
        compute_budget="STANDARD"
    )
    bad_spec.compute_budget = "500_EPOCHS"

    safety = gov.check_experiment_safety(bad_spec, human_approved=True)
    assert safety["allowed"] is False
    assert safety["status"] == "BLOCKED_BY_RESEARCH_GOVERNANCE"
    assert "500-epoch training paused under DEC-019/020" in safety["reason"]


def test_test_set_proposal_protection():
    gov = ResearchGovernor()
    designer = ExperimentDesigner()

    leaky_spec = designer.design_experiment(
        experiment_id="EXP-LEAKY",
        hypothesis_id="HYP-001",
        research_question="RQ-001",
        hypothesis_statement="Tune hyperparameter on the test set partition directly.",
        baseline_model="Model A",
        exact_change="Tuning on test partition",
        fixed_components=["HREncoder", "BiGRU"],
        success_metric="Macro-F1",
        falsification_criteria="Loss diverges",
        confounders=["leakage"],
        compute_budget="SHORT"
    )

    safety = gov.check_experiment_safety(leaky_spec)
    assert safety["allowed"] is False
    assert "TEST SET PROTECTION VIOLATION" in safety["reason"]


def test_gpu_resource_manager_allocation_and_release():
    rm = GPUResourceManager()
    status = rm.get_device_status()
    assert len(status) >= 1

    # Allocate GPU for EXP-015
    dev = rm.allocate_gpu("EXP-015", "SHORT")
    assert dev in ("cuda:0", "cuda:1", "cpu")

    # If 2 CUDA devices exist, test allocation of second GPU and refusal of third concurrent heavy task
    if rm.device_count == 2:
        dev2 = rm.allocate_gpu("EXP-016", "SHORT")
        assert dev2 != dev
        with pytest.raises(ResourceAllocationError) as exc_info:
            rm.allocate_gpu("EXP-017", "STANDARD")  # no free GPU
        assert "All 2 GPUs are currently busy" in str(exc_info.value)
        rm.release_gpu(dev2)

    rm.release_gpu(dev)
    updated_status = rm.get_device_status()
    for d in updated_status:
        assert d["is_available"] is True
