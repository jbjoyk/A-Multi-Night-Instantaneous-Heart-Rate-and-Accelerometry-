"""
tests/test_experiment_engine_simulated.py: Unit tests for ExperimentEngine lifecycle simulation and DecisionEngine.
"""

import os
import json
import tempfile
import pytest
from governor import ResearchGovernor
from research_engine.hypothesis_agent import HypothesisAgent
from research_engine.experiment_engine import ExperimentEngine
from research_engine.decision_logic import DecisionEngine, ExperimentDecision


def test_decision_engine_rules():
    de = DecisionEngine()

    # Rule 1: Substantial gain with controlled capacity -> KEPT
    res_gain = de.evaluate_outcome(
        baseline_metrics={"macro_f1": 0.6658, "accuracy": 0.6663},
        candidate_metrics={"macro_f1": 0.6780, "accuracy": 0.6790},
        baseline_params=163940,
        candidate_params=164000
    )
    assert res_gain["decision"] == ExperimentDecision.KEPT.value
    assert "Statistically meaningful Macro-F1 improvement" in res_gain["rationale"]

    # Rule 2: Complexity trap (tiny gain with massive parameter spike) -> NEEDS_REVIEW
    res_bloat = de.evaluate_outcome(
        baseline_metrics={"macro_f1": 0.6658, "accuracy": 0.6663},
        candidate_metrics={"macro_f1": 0.6670, "accuracy": 0.6675},
        baseline_params=163940,
        candidate_params=350000  # +113% parameters
    )
    assert res_bloat["decision"] == ExperimentDecision.NEEDS_REVIEW.value
    assert "Complexity Trap" in res_bloat["rationale"]

    # Rule 3: Beneficial simplification (parity with lower parameter count) -> KEPT
    res_simple = de.evaluate_outcome(
        baseline_metrics={"macro_f1": 0.6658, "accuracy": 0.6663},
        candidate_metrics={"macro_f1": 0.6650, "accuracy": 0.6655},
        baseline_params=163940,
        candidate_params=125000  # -24% parameters
    )
    assert res_simple["decision"] == ExperimentDecision.KEPT.value
    assert "Beneficial Simplification" in res_simple["rationale"]

    # Rule 4: Clear degradation -> DISCARDED
    res_drop = de.evaluate_outcome(
        baseline_metrics={"macro_f1": 0.6658, "accuracy": 0.6663},
        candidate_metrics={"macro_f1": 0.6450, "accuracy": 0.6460},
        baseline_params=163940,
        candidate_params=163940
    )
    assert res_drop["decision"] == ExperimentDecision.DISCARDED.value
    assert "Clear performance regression" in res_drop["rationale"]


def test_engine_simulated_cycle_success():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem_file = os.path.join(tmpdir, "test_memory.json")
        with open(mem_file, "w") as f:
            json.dump({"experiments": []}, f)

        gov = ResearchGovernor()
        engine = ExperimentEngine(memory_path=mem_file)
        hyp_agent = HypothesisAgent()
        hyp = hyp_agent.get_grounded_candidates()[0]

        token = gov.generate_governance_token("EXP-996", "SHORT")

        res = engine.execute_lifecycle_step(
            hypothesis=hyp,
            experiment_id="EXP-996",
            parent_experiment="EXP-014",
            baseline_model="Model A",
            exact_change="Prune dummy clock channel",
            fixed_components=["HREncoder", "BiGRU", "RCCLoss"],
            compute_budget="SHORT",
            dry_run=True,
            governance_token=token,
            simulate_failure=False,
            simulate_blocked=False
        )

        assert res["record"]["lifecycle_status"] == "COMPLETED"
        assert res["record"]["execution_mode"] == "DRY_RUN"
        assert res["record"]["scientific_evidence"] is False
        assert res["record"]["experiment_id"] == "EXP-996"
        assert "macro_f1" in res["record"]["metrics"]

        # Verify persisted to memory
        with open(mem_file, "r") as f:
            loaded_mem = json.load(f)
        assert len(loaded_mem["experiments"]) == 1
        assert loaded_mem["experiments"][0]["experiment_id"] == "EXP-996"


def test_engine_simulated_cycle_failure():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem_file = os.path.join(tmpdir, "test_memory.json")
        with open(mem_file, "w") as f:
            json.dump({"experiments": []}, f)

        gov = ResearchGovernor()
        engine = ExperimentEngine(memory_path=mem_file)
        hyp_agent = HypothesisAgent()
        hyp = hyp_agent.get_grounded_candidates()[0]

        token = gov.generate_governance_token("EXP-997", "SHORT")

        res = engine.execute_lifecycle_step(
            hypothesis=hyp,
            experiment_id="EXP-997",
            parent_experiment="EXP-014",
            baseline_model="Model A",
            exact_change="Prune dummy clock channel",
            fixed_components=["HREncoder", "BiGRU", "RCCLoss"],
            compute_budget="SHORT",
            dry_run=True,
            governance_token=token,
            simulate_failure=True,
            simulate_blocked=False
        )

        assert res["record"]["lifecycle_status"] == "FAILED"
        assert "Simulated execution failure" in res["record"]["reason"]


def test_engine_simulated_cycle_blocked():
    with tempfile.TemporaryDirectory() as tmpdir:
        mem_file = os.path.join(tmpdir, "test_memory.json")
        with open(mem_file, "w") as f:
            json.dump({"experiments": []}, f)

        gov = ResearchGovernor()
        engine = ExperimentEngine(memory_path=mem_file)
        hyp_agent = HypothesisAgent()
        hyp = hyp_agent.get_grounded_candidates()[0]

        token = gov.generate_governance_token("EXP-998", "SHORT")

        res = engine.execute_lifecycle_step(
            hypothesis=hyp,
            experiment_id="EXP-998",
            parent_experiment="EXP-014",
            baseline_model="Model A",
            exact_change="Prune dummy clock channel",
            fixed_components=["HREncoder", "BiGRU", "RCCLoss"],
            compute_budget="SHORT",
            dry_run=True,
            governance_token=token,
            simulate_failure=False,
            simulate_blocked=True
        )

        assert res["record"]["lifecycle_status"] == "BLOCKED"
        assert "BLOCKED_BY_RESEARCH_GOVERNANCE" in res["record"]["reason"]
