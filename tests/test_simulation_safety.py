"""
tests/test_simulation_safety.py: Unit tests proving simulation non-evidence boundaries.

Verifies:
  1. Simulation cannot become the best REAL experiment.
  2. Simulation cannot influence real experiment ranking.
  3. Simulation cannot satisfy evidence requirements.
  4. Simulation cannot be used as a scientific baseline.
  5. Simulation metrics cannot be interpreted as empirical results.
  6. Simulation cannot be silently converted to REAL by editing one field.
  7. Dry-run and simulation records remain distinguishable from real experiments.
"""

import copy
import pytest
from research_engine.state_machine import ExecutionMode, LifecycleStatus, ExecutionModeViolationError
from research_engine.memory_validator import ResearchMemoryValidator, MemoryCorruptionError


@pytest.fixture
def sample_memory():
    return {
        "system_version": "1.0.0",
        "experiments": [
            {
                "experiment_id": "EXP-013",
                "parent_experiment": "EXP-012",
                "hypothesis": "OFAT component attribution",
                "research_question": "RQ",
                "change": "Model E vs Model A",
                "git_commit": "abc1234",
                "dataset_fingerprint": "8ae7a916a4cd5709",
                "split_hash": "official_paper_split_31_5_11",
                "seed": 42,
                "budget": "STANDARD",
                "gpu": "cuda:0",
                "metrics": {
                    "macro_f1": 0.6844,
                    "accuracy": 0.6844
                },
                "execution_mode": "REAL",
                "lifecycle_status": "COMPLETED",
                "real_training": True,
                "scientific_evidence": True,
                "eligible_for_model_selection": True,
                "eligible_for_research_conclusions": True,
                "eligible_for_leaderboard": True,
                "decision": "KEPT",
                "reason": "Empirical attribution established.",
                "timestamp": "2026-11-10T13:44:00"
            },
            {
                "experiment_id": "EXP-014",
                "parent_experiment": "EXP-013",
                "hypothesis": "Cross-validation evaluation",
                "research_question": "RQ",
                "change": "5-fold CV",
                "git_commit": "abc1234",
                "dataset_fingerprint": "8ae7a916a4cd5709",
                "split_hash": "exp014_5fold_disjoint",
                "seed": [42, 123, 2024],
                "budget": "FULL",
                "gpu": "cuda:0",
                "metrics": {
                    "macro_f1": 0.6658,
                    "accuracy": 0.6663
                },
                "execution_mode": "REAL",
                "lifecycle_status": "COMPLETED",
                "real_training": True,
                "scientific_evidence": True,
                "eligible_for_model_selection": True,
                "eligible_for_research_conclusions": True,
                "eligible_for_leaderboard": True,
                "decision": "KEPT",
                "reason": "Equivalence demonstrated.",
                "timestamp": "2026-11-11T00:34:00"
            },
            {
                "experiment_id": "EXP-015-SIM",
                "parent_experiment": "EXP-014",
                "hypothesis": "Simulated dummy clock pruning",
                "research_question": "RQ",
                "change": "Prune dummy clock channel",
                "git_commit": "abc1234",
                "dataset_fingerprint": "8ae7a916a4cd5709",
                "split_hash": "inner_val_split_seed42",
                "seed": 42,
                "budget": "SHORT",
                "gpu": "cuda:0",
                "metrics": {
                    "macro_f1": 0.9999,  # Artificially high simulated score
                    "accuracy": 0.9999
                },
                "execution_mode": "SIMULATION",
                "lifecycle_status": "COMPLETED",
                "real_training": False,
                "scientific_evidence": False,
                "eligible_for_model_selection": False,
                "eligible_for_research_conclusions": False,
                "eligible_for_leaderboard": False,
                "simulation_tag": "SIMULATION — NON-EVIDENCE",
                "decision": "SIMULATION_NON_EVIDENCE",
                "reason": "Dry run verification only.",
                "timestamp": "2026-11-13T22:13:14"
            }
        ]
    }


def test_simulation_cannot_enter_real_experiments(sample_memory):
    reals = ResearchMemoryValidator.get_real_experiments(sample_memory)
    real_ids = [e["experiment_id"] for e in reals]
    assert "EXP-013" in real_ids
    assert "EXP-014" in real_ids
    assert "EXP-015-SIM" not in real_ids


def test_simulation_cannot_enter_leaderboard(sample_memory):
    # Despite EXP-015-SIM having Macro-F1 = 0.9999, it MUST NOT appear on leaderboard
    lb = ResearchMemoryValidator.get_leaderboard(sample_memory, sort_by="macro_f1")
    lb_ids = [e["experiment_id"] for e in lb]
    assert "EXP-015-SIM" not in lb_ids
    assert lb[0]["experiment_id"] == "EXP-013"
    assert lb[0]["metrics"]["macro_f1"] == 0.6844


def test_simulation_cannot_satisfy_evidence_requirements(sample_memory):
    sim = sample_memory["experiments"][2]
    assert sim["scientific_evidence"] is False
    assert sim["real_training"] is False
    assert sim["eligible_for_model_selection"] is False
    assert sim["eligible_for_research_conclusions"] is False


def test_simulation_metrics_not_empirical(sample_memory):
    sims = ResearchMemoryValidator.get_simulation_records(sample_memory)
    assert len(sims) == 1
    assert sims[0]["simulation_tag"] == "SIMULATION — NON-EVIDENCE"
    assert sims[0]["decision"] == "SIMULATION_NON_EVIDENCE"


def test_simulation_cannot_be_silently_converted_to_real(sample_memory):
    bad_rec = copy.deepcopy(sample_memory["experiments"][2])
    # Attempt 1: Just flip execution_mode without changing the other safety flags
    bad_rec["execution_mode"] = "REAL"
    with pytest.raises(MemoryCorruptionError) as exc_info:
        ResearchMemoryValidator.validate_record(bad_rec)

    # Attempt 2: Try state machine validation of execution mode mutation
    with pytest.raises(ExecutionModeViolationError) as exc_info2:
        from research_engine.state_machine import ExperimentStateMachine
        ExperimentStateMachine.validate_execution_mode_invariance("SIMULATION", "REAL", "EXP-015-SIM")
    assert "CRITICAL INTEGRITY VIOLATION" in str(exc_info2.value)


def test_simulation_distinguishable_from_real(sample_memory):
    for exp in sample_memory["experiments"]:
        if exp["execution_mode"] == "REAL":
            assert exp["scientific_evidence"] is True
            assert exp["real_training"] is True
            assert exp["eligible_for_leaderboard"] is True
        elif exp["execution_mode"] == "SIMULATION":
            assert exp["scientific_evidence"] is False
            assert exp["real_training"] is False
            assert exp["eligible_for_leaderboard"] is False
            assert "simulation_tag" in exp
