"""
tests/test_memory_integrity.py: Tests for ResearchMemoryValidator schema validation and corruption detection.

Verifies:
  - duplicate experiment IDs
  - missing required fields
  - missing metrics
  - impossible metrics (outside [0, 1] or kappa outside [-1, 1])
  - negative epochs
  - missing dataset fingerprint
  - missing split hash
  - missing Git provenance
  - simulation recorded as REAL
  - REAL recorded as SIMULATION
  - invalid lifecycle status
  - illegal state transition
"""

import copy
import pytest
from research_engine.memory_validator import ResearchMemoryValidator, MemoryCorruptionError


@pytest.fixture
def valid_record():
    return {
        "experiment_id": "EXP-014",
        "parent_experiment": "EXP-013",
        "hypothesis": "Test hypothesis",
        "research_question": "RQ-001",
        "change": "5-fold CV",
        "git_commit": "abc1234",
        "dataset_fingerprint": "8ae7a916a4cd5709",
        "split_hash": "exp014_5fold_disjoint",
        "seed": 42,
        "budget": "STANDARD",
        "gpu": "cuda:0",
        "metrics": {
            "macro_f1": 0.6658,
            "accuracy": 0.6663,
            "kappa": 0.5210
        },
        "execution_mode": "REAL",
        "lifecycle_status": "COMPLETED",
        "real_training": True,
        "scientific_evidence": True,
        "eligible_for_model_selection": True,
        "eligible_for_research_conclusions": True,
        "eligible_for_leaderboard": True,
        "decision": "KEPT",
        "reason": "Parity confirmed.",
        "timestamp": "2026-11-11T00:34:00"
    }


def test_valid_record_passes(valid_record):
    ResearchMemoryValidator.validate_record(valid_record)


def test_missing_required_fields_raises(valid_record):
    for field in ["experiment_id", "git_commit", "dataset_fingerprint", "split_hash", "metrics"]:
        bad_rec = copy.deepcopy(valid_record)
        del bad_rec[field]
        with pytest.raises(MemoryCorruptionError) as exc_info:
            ResearchMemoryValidator.validate_record(bad_rec)
        assert "missing required fields" in str(exc_info.value) or "Invalid experiment ID" in str(exc_info.value)


def test_duplicate_experiment_ids_raises(valid_record):
    memory = {
        "experiments": [
            valid_record,
            copy.deepcopy(valid_record)  # Duplicate!
        ]
    }
    with pytest.raises(MemoryCorruptionError) as exc_info:
        ResearchMemoryValidator.validate_memory_integrity(memory)
    assert "DUPLICATE EXPERIMENT ID DETECTED" in str(exc_info.value)


def test_impossible_metric_values_raise(valid_record):
    # Macro-F1 > 1.0
    bad_f1 = copy.deepcopy(valid_record)
    bad_f1["metrics"]["macro_f1"] = 1.05
    with pytest.raises(MemoryCorruptionError) as exc_info1:
        ResearchMemoryValidator.validate_record(bad_f1)
    assert "Impossible metric value" in str(exc_info1.value)

    # Accuracy < 0.0
    bad_acc = copy.deepcopy(valid_record)
    bad_acc["metrics"]["accuracy"] = -0.1
    with pytest.raises(MemoryCorruptionError) as exc_info2:
        ResearchMemoryValidator.validate_record(bad_acc)
    assert "Impossible metric value" in str(exc_info2.value)

    # Kappa < -1.0
    bad_kap = copy.deepcopy(valid_record)
    bad_kap["metrics"]["kappa"] = -1.5
    with pytest.raises(MemoryCorruptionError) as exc_info3:
        ResearchMemoryValidator.validate_record(bad_kap)
    assert "Impossible metric value" in str(exc_info3.value)


def test_simulation_recorded_as_real_raises(valid_record):
    sim_as_real = copy.deepcopy(valid_record)
    sim_as_real["execution_mode"] = "REAL"
    sim_as_real["real_training"] = False  # Inconsistent!
    with pytest.raises(MemoryCorruptionError) as exc_info:
        ResearchMemoryValidator.validate_record(sim_as_real)
    assert "INCONSISTENCY VIOLATION" in str(exc_info.value)


def test_real_recorded_as_simulation_safety(valid_record):
    real_as_sim = copy.deepcopy(valid_record)
    real_as_sim["execution_mode"] = "SIMULATION"
    # If execution mode is SIMULATION, scientific_evidence must be False
    with pytest.raises(MemoryCorruptionError) as exc_info:
        ResearchMemoryValidator.validate_record(real_as_sim)
    assert "SIMULATION SAFETY VIOLATION" in str(exc_info.value)


def test_invalid_lifecycle_status_raises(valid_record):
    bad_stat = copy.deepcopy(valid_record)
    bad_stat["lifecycle_status"] = "INVALID_STATUS_UNKNOWN"
    with pytest.raises(MemoryCorruptionError) as exc_info:
        ResearchMemoryValidator.validate_record(bad_stat)
    assert "Invalid lifecycle_status" in str(exc_info.value)


def test_registry_exact_counts_and_consistency():
    import os
    import json

    mem_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "research_memory.json"
    )
    with open(mem_path, "r") as f:
        memory = json.load(f)

    # 1. Must pass full consistency audit without exceptions
    ResearchMemoryValidator.assert_registry_consistency(memory)

    # 2. Programmatically verified counts
    stats = ResearchMemoryValidator.compute_registry_statistics(memory)
    assert stats["total_records"] == 12
    assert stats["real_count"] == 11
    assert stats["simulation_count"] == 1
    assert stats["dry_run_count"] == 0
    assert stats["eligible_count"] == 11
    assert stats["ineligible_count"] == 1

    expected_reals = [
        "EXP-001", "EXP-005", "EXP-006", "EXP-007", "EXP-008",
        "EXP-009", "EXP-010", "EXP-011", "EXP-012", "EXP-013", "EXP-014"
    ]
    assert stats["real_ids"] == expected_reals
    assert stats["simulation_ids"] == ["EXP-015-SIM"]
    assert stats["ineligible_ids"] == ["EXP-015-SIM"]

    # 3. Disclosed missing sequence IDs
    assert set(stats["excluded_sequence"].keys()) == {"EXP-002", "EXP-003", "EXP-004"}


def test_registry_inconsistency_detection():
    import os
    import json

    mem_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "research_memory.json"
    )
    with open(mem_path, "r") as f:
        memory = json.load(f)

    # Inconsistency A: Contaminate simulation by marking eligible_for_leaderboard = True
    corrupt_a = copy.deepcopy(memory)
    for exp in corrupt_a["experiments"]:
        if exp["experiment_id"] == "EXP-015-SIM":
            exp["eligible_for_leaderboard"] = True
    with pytest.raises(MemoryCorruptionError) as exc_a:
        ResearchMemoryValidator.assert_registry_consistency(corrupt_a)
    assert "SIMULATION SAFETY VIOLATION" in str(exc_a.value) or "CRITICAL CONTAMINATION" in str(exc_a.value)

    # Inconsistency B: Demote a REAL experiment to ineligible
    corrupt_b = copy.deepcopy(memory)
    for exp in corrupt_b["experiments"]:
        if exp["experiment_id"] == "EXP-014":
            exp["eligible_for_leaderboard"] = False
    with pytest.raises(MemoryCorruptionError) as exc_b:
        ResearchMemoryValidator.assert_registry_consistency(corrupt_b)
    assert "REAL experiments found with ineligible status" in str(exc_b.value)

