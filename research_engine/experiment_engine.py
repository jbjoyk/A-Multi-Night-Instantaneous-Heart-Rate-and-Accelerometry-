"""
research_engine/experiment_engine.py: Autonomous Experimentation Engine.

Implements Karpathy's Autoresearch experiment execution loop adapted for Sleep Staging:
  PROPOSE
  -> GOVERN (Safety, Budget, Test-Set Protection, 500-Epoch Rule, Token Verification)
  -> DESIGN (Controlled OFAT Specification)
  -> REVIEW (Autonomous Scientific Peer Review)
  -> IMPLEMENT (Isolated Git Tracking & Artifact Directory)
  -> RUN (Supervised Training or Dry-Run Simulation under Exclusive GPU Lease)
  -> EVALUATE (SleepEvaluator: Primary Macro-F1, Secondary, Diagnostics)
  -> DECISION (Multi-factor Keep / Discard / Investigate)
  -> MEMORY (Structured Registration in research_memory.json with Simulation Shielding)

Hard Invariants:
  1. Direct runner execution without a valid Governor authorization token is strictly blocked.
  2. Sequential state transitions enforced via ExperimentStateMachine.
  3. Execution modes (REAL, SIMULATION, DRY_RUN) strictly decoupled from lifecycle statuses.
  4. Simulation results cannot enter leaderboards, model selection, or scientific evidence.
"""

import os
import json
import time
import hmac
import hashlib
import subprocess
from typing import Dict, List, Optional, Any
import numpy as np

from dataset_gate import DatasetGate, TestSetLeakageError
from evaluator import SleepEvaluator, TestSetProtectionViolation
from research_engine.resource_manager import GPUResourceManager, ResourceAllocationError
from research_engine.literature_agent import LiteratureAgent
from research_engine.hypothesis_agent import HypothesisAgent, ResearchHypothesis
from research_engine.experiment_designer import ExperimentDesigner, ExperimentSpecification, ExperimentDesignError
from research_engine.scientific_reviewer import ScientificReviewer, ReviewVerdict
from research_engine.decision_logic import DecisionEngine, ExperimentDecision
from research_engine.state_machine import (
    ExecutionMode, LifecycleStatus, ExperimentStateMachine,
    IllegalStateTransitionError, ExecutionModeViolationError
)
from research_engine.memory_validator import ResearchMemoryValidator, MemoryCorruptionError


class GovernanceBlockError(Exception):
    """Raised when an operation is blocked by Research Governance."""
    pass


class ExperimentEngine:
    """
    Coordinates the controlled, research-grade autonomous experimentation lifecycle.
    """
    GOVERNOR_SECRET_KEY = b"sleep_research_governor_internal_auth_key_dec021"

    def __init__(
        self,
        base_dir: Optional[str] = None,
        memory_path: Optional[str] = None,
        dataset_gate: Optional[DatasetGate] = None,
        evaluator: Optional[SleepEvaluator] = None,
        resource_manager: Optional[GPUResourceManager] = None,
        reviewer: Optional[ScientificReviewer] = None,
        decision_engine: Optional[DecisionEngine] = None
    ):
        self.base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.memory_path = memory_path or os.path.join(self.base_dir, "research_memory.json")
        self.dataset_gate = dataset_gate or DatasetGate(base_dir=self.base_dir)
        self.evaluator = evaluator or SleepEvaluator()
        self.resource_manager = resource_manager or GPUResourceManager()
        self.literature_agent = LiteratureAgent()
        self.hypothesis_agent = HypothesisAgent(literature_agent=self.literature_agent)
        self.designer = ExperimentDesigner()
        self.reviewer = reviewer or ScientificReviewer()
        self.decision_engine = decision_engine or DecisionEngine()
        self.used_tokens: set[str] = set()

    def get_current_git_commit(self) -> str:
        """Retrieves the current git commit hash."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self.base_dir,
                capture_output=True,
                text=True,
                check=True
            )
            return res.stdout.strip()
        except Exception:
            return "unknown"

    def load_memory(self) -> Dict[str, Any]:
        """Loads research memory from disk."""
        if not os.path.exists(self.memory_path):
            return {"experiments": []}
        with open(self.memory_path, "r") as f:
            return json.load(f)

    def save_memory(self, memory: Dict[str, Any]) -> None:
        """Persists updated research memory to disk after validation."""
        ResearchMemoryValidator.validate_memory_integrity(memory)
        with open(self.memory_path, "w") as f:
            json.dump(memory, f, indent=2)

    def verify_governance_token(
        self,
        experiment_id: str,
        compute_budget: str,
        exact_change: str,
        dataset_fingerprint: str,
        token: Optional[str]
    ) -> tuple[bool, str]:
        """
        Verifies if the provided governance token is valid for this experiment.
        Checks:
          - Token existence
          - Expiration (valid for 3600 seconds)
          - Single-use replay protection
          - Proposal content hash invariance
          - HMAC signature validity
        """
        if not token:
            return False, "MISSING_GOVERNOR_TOKEN"

        parts = token.split(":")
        # Hardened format: ts:content_hash:sig (3 parts)
        if len(parts) == 3:
            ts_str, content_hash, sig = parts
            try:
                ts = int(ts_str)
            except ValueError:
                return False, "MALFORMED_TIMESTAMP"

            # Check expiration (1 hour)
            if time.time() - ts > 3600:
                return False, "EXPIRED_GOVERNOR_TOKEN"

            # Check replay prevention
            if token in self.used_tokens:
                return False, "REUSED_GOVERNOR_TOKEN"

            # Check proposal content invariance
            expected_content_hash = hashlib.sha256(f"{exact_change}:{dataset_fingerprint}".encode("utf-8")).hexdigest()[:16]
            if not hmac.compare_digest(expected_content_hash, content_hash):
                return False, "ALTERED_PROPOSAL_AFTER_TOKEN_GENERATION"

            # Verify HMAC signature against allowed autonomy levels
            for aut_level in (0, 1, 2, 3, 4, 5):
                payload = f"{experiment_id}:{compute_budget.upper()}:{aut_level}:{content_hash}:{ts}".encode("utf-8")
                expected_sig = hmac.new(self.GOVERNOR_SECRET_KEY, payload, hashlib.sha256).hexdigest()
                if hmac.compare_digest(expected_sig, sig):
                    return True, "VALID_TOKEN"

            return False, "FORGED_SIGNATURE"

        # Legacy 1-part hex digest fallback
        for aut_level in (0, 1, 2, 3, 4, 5):
            payload = f"{experiment_id}:{compute_budget.upper()}:{aut_level}".encode("utf-8")
            expected = hmac.new(self.GOVERNOR_SECRET_KEY, payload, hashlib.sha256).hexdigest()
            if hmac.compare_digest(expected, token):
                if token in self.used_tokens:
                    return False, "REUSED_GOVERNOR_TOKEN"
                return True, "VALID_LEGACY_TOKEN"

        return False, "INVALID_TOKEN_FORMAT"

    def execute_lifecycle_step(
        self,
        hypothesis: ResearchHypothesis,
        experiment_id: str,
        parent_experiment: str,
        baseline_model: str,
        exact_change: str,
        fixed_components: List[str],
        compute_budget: str,
        dry_run: bool = True,
        governance_token: Optional[str] = None,
        simulate_failure: bool = False,
        simulate_blocked: bool = False
    ) -> Dict[str, Any]:
        """
        Executes one full controlled experimentation cycle:
        PROPOSE -> GOVERN -> DESIGN -> REVIEW -> IMPLEMENT -> RUN -> EVALUATE -> DECISION -> MEMORY
        """
        cycle_log: List[str] = []
        cycle_log.append(f"Starting Governed Experiment Cycle for {experiment_id} (Hypothesis: {hypothesis.hypothesis_id})")

        # Determine execution mode upfront
        exec_mode = ExecutionMode.DRY_RUN.value if dry_run else ExecutionMode.REAL.value
        is_simulation = dry_run

        # 1. PROPOSE
        current_status = LifecycleStatus.PROPOSED
        cycle_log.append(f"Step 1 [{current_status.value}]: Hypothesis registered.")

        # 2. GOVERN (Pre-flight safety & direct runner bypass check)
        next_status = LifecycleStatus.GOVERNED
        ExperimentStateMachine.validate_transition(current_status, next_status, experiment_id)
        current_status = next_status

        # Anti-bypass check: Direct runner execution without Governor token is blocked
        ds_fp = self.dataset_gate.compute_dataset_fingerprint()
        valid_tok, token_reason = self.verify_governance_token(
            experiment_id=experiment_id,
            compute_budget=compute_budget,
            exact_change=exact_change,
            dataset_fingerprint=ds_fp,
            token=governance_token
        )
        if not valid_tok:
            current_status = LifecycleStatus.BLOCKED
            cycle_log.append(f"Step 2 [GOVERN]: BLOCKED - Direct runner bypass attempt or invalid governance token ({token_reason}).")
            raise GovernanceBlockError(
                f"DIRECT RUNNER BYPASS BLOCKED: Direct runner execution for {experiment_id} "
                f"rejected by Governor token verification gate ({token_reason})."
            )

        if governance_token:
            self.used_tokens.add(governance_token)

        if simulate_blocked:
            current_status = LifecycleStatus.BLOCKED
            record = {
                "experiment_id": experiment_id,
                "parent_experiment": parent_experiment,
                "hypothesis": hypothesis.hypothesis,
                "research_question": hypothesis.research_question,
                "change": exact_change,
                "git_commit": self.get_current_git_commit(),
                "dataset_fingerprint": self.dataset_gate.compute_dataset_fingerprint(),
                "split_hash": "inner_val_split_seed42",
                "seed": 42,
                "budget": compute_budget,
                "gpu": "none",
                "metrics": {},
                "execution_mode": exec_mode,
                "lifecycle_status": current_status.value,
                "real_training": False,
                "scientific_evidence": False,
                "eligible_for_model_selection": False,
                "eligible_for_research_conclusions": False,
                "eligible_for_leaderboard": False,
                "status": current_status.value,
                "decision": "DISCARDED",
                "reason": "BLOCKED_BY_RESEARCH_GOVERNANCE: Simulated governance hard block.",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            cycle_log.append(f"Step 2 [GOVERN]: BLOCKED - {record['reason']}")
            return {"record": record, "log": cycle_log}

        # Check 500-epoch restriction
        if "500" in compute_budget or "500" in exact_change:
            current_status = LifecycleStatus.BLOCKED
            record = {
                "experiment_id": experiment_id,
                "parent_experiment": parent_experiment,
                "hypothesis": hypothesis.hypothesis,
                "research_question": hypothesis.research_question,
                "change": exact_change,
                "git_commit": self.get_current_git_commit(),
                "dataset_fingerprint": self.dataset_gate.compute_dataset_fingerprint(),
                "split_hash": "inner_val_split_seed42",
                "seed": 42,
                "budget": compute_budget,
                "gpu": "none",
                "metrics": {},
                "execution_mode": exec_mode,
                "lifecycle_status": current_status.value,
                "real_training": False,
                "scientific_evidence": False,
                "eligible_for_model_selection": False,
                "eligible_for_research_conclusions": False,
                "eligible_for_leaderboard": False,
                "status": current_status.value,
                "decision": "DISCARDED",
                "reason": "BLOCKED_BY_RESEARCH_GOVERNANCE: 500-epoch training paused under DEC-019/020 and user governance.",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            cycle_log.append(f"Step 2 [GOVERN]: BLOCKED - {record['reason']}")
            return {"record": record, "log": cycle_log}

        # Verify dataset fingerprint
        ds_fp = self.dataset_gate.compute_dataset_fingerprint()
        cycle_log.append(f"Step 2 [GOVERN]: Dataset gate passed (Fingerprint: {ds_fp}).")

        # 3. DESIGN
        next_status = LifecycleStatus.DESIGNED
        ExperimentStateMachine.validate_transition(current_status, next_status, experiment_id)
        current_status = next_status

        spec = self.designer.design_experiment(
            experiment_id=experiment_id,
            hypothesis_id=hypothesis.hypothesis_id,
            research_question=hypothesis.research_question,
            hypothesis_statement=hypothesis.hypothesis,
            baseline_model=baseline_model,
            exact_change=exact_change,
            fixed_components=fixed_components,
            success_metric=hypothesis.evaluation_criteria,
            falsification_criteria="Validation Macro-F1 degrades by > 0.010 or loss diverges.",
            confounders=["inter-subject physiological variance", "class imbalance"],
            compute_budget=compute_budget,
            target_device="cuda:0"
        )
        cycle_log.append(f"Step 3 [DESIGN]: Specification created for budget {spec.compute_budget}.")

        # 4. REVIEW
        next_status = LifecycleStatus.REVIEWED
        ExperimentStateMachine.validate_transition(current_status, next_status, experiment_id)
        current_status = next_status

        memory = self.load_memory()
        existing_ids = [exp["experiment_id"] for exp in memory.get("experiments", [])]
        review = self.reviewer.review_proposal(spec, existing_experiment_ids=existing_ids)
        cycle_log.append(f"Step 4 [REVIEW]: Verdict = {review['verdict']} ({review['checks_passed']}/10 checks passed).")

        if review["verdict"] == ReviewVerdict.REJECT.value:
            current_status = LifecycleStatus.DISCARDED
            record = {
                "experiment_id": experiment_id,
                "parent_experiment": parent_experiment,
                "hypothesis": hypothesis.hypothesis,
                "research_question": hypothesis.research_question,
                "change": exact_change,
                "git_commit": self.get_current_git_commit(),
                "dataset_fingerprint": ds_fp,
                "split_hash": "inner_val_split_seed42",
                "seed": 42,
                "budget": compute_budget,
                "gpu": "none",
                "metrics": {},
                "execution_mode": exec_mode,
                "lifecycle_status": current_status.value,
                "real_training": False,
                "scientific_evidence": False,
                "eligible_for_model_selection": False,
                "eligible_for_research_conclusions": False,
                "eligible_for_leaderboard": False,
                "status": current_status.value,
                "decision": "DISCARDED",
                "reason": f"Rejected by Scientific Reviewer: {'; '.join(review['flags'])}",
                "review": review,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            return {"record": record, "log": cycle_log}

        # 5. APPROVE
        next_status = LifecycleStatus.APPROVED
        ExperimentStateMachine.validate_transition(current_status, next_status, experiment_id)
        current_status = next_status

        # 6. IMPLEMENT
        next_status = LifecycleStatus.IMPLEMENTING
        ExperimentStateMachine.validate_transition(current_status, next_status, experiment_id)
        current_status = next_status

        exp_dir = os.path.join(self.base_dir, "research", "experiments", experiment_id)
        os.makedirs(exp_dir, exist_ok=True)
        git_commit = self.get_current_git_commit()
        cycle_log.append(f"Step 5 [IMPLEMENT]: Workspace ready at {exp_dir} (Commit: {git_commit}).")

        # 7. RUN (Under safe resource lease context)
        next_status = LifecycleStatus.RUNNING
        ExperimentStateMachine.validate_transition(current_status, next_status, experiment_id)
        current_status = next_status

        try:
            with self.resource_manager.lease(experiment_id, compute_budget) as allocated_device:
                cycle_log.append(f"Step 6 [RUN]: Leased device {allocated_device} for budget {compute_budget}.")
                if simulate_failure:
                    raise RuntimeError("Simulated execution failure or out-of-memory error.")

                if dry_run:
                    time.sleep(0.05)  # Simulated execution
                    cycle_log.append("Step 6 [RUN]: Dry-run simulation completed successfully.")
                else:
                    raise NotImplementedError("LIVE_EXECUTION_INTERFACE_NOT_ENABLED: Live model training is disabled during Phase 2 audit.")
        except Exception as e:
            current_status = LifecycleStatus.FAILED
            record = {
                "experiment_id": experiment_id,
                "parent_experiment": parent_experiment,
                "hypothesis": hypothesis.hypothesis,
                "research_question": hypothesis.research_question,
                "change": exact_change,
                "git_commit": git_commit,
                "dataset_fingerprint": ds_fp,
                "split_hash": "inner_val_split_seed42",
                "seed": 42,
                "budget": compute_budget,
                "gpu": "none",
                "metrics": {},
                "execution_mode": exec_mode,
                "lifecycle_status": current_status.value,
                "real_training": False,
                "scientific_evidence": False,
                "eligible_for_model_selection": False,
                "eligible_for_research_conclusions": False,
                "eligible_for_leaderboard": False,
                "status": current_status.value,
                "decision": "DISCARDED",
                "reason": f"Runtime execution failure: {str(e)}",
                "git_commit": git_commit,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
            }
            cycle_log.append(f"Step 6 [RUN]: FAILED - {str(e)}")
            return {"record": record, "log": cycle_log}

        # 8. EVALUATE
        next_status = LifecycleStatus.EVALUATING
        ExperimentStateMachine.validate_transition(current_status, next_status, experiment_id)
        current_status = next_status

        # Simulated evaluation on inner validation partition
        rng = np.random.RandomState(42)
        n_val = 1000
        y_val_true = rng.randint(0, 4, size=n_val)
        y_val_pred = y_val_true.copy()
        y_val_pred[rng.choice(n_val, size=320, replace=False)] = rng.randint(0, 4, size=320)
        
        eval_metrics = self.evaluator.evaluate_predictions(y_val_true, y_val_pred)
        cycle_log.append(
            f"Step 7 [EVALUATE]: Validation Macro-F1 = {eval_metrics['primary']['macro_f1']:.4f}, "
            f"Accuracy = {eval_metrics['secondary']['accuracy']:.4f}."
        )

        # Baseline metrics (Model A reference from EXP-014: Macro-F1 = 0.6658, Acc = 0.6663)
        baseline_metrics = {
            "macro_f1": 0.6658,
            "accuracy": 0.6663,
            "secondary": {"rem_f1": 0.6120}
        }

        # 9. DECISION
        next_status = LifecycleStatus.COMPLETED
        ExperimentStateMachine.validate_transition(current_status, next_status, experiment_id)
        current_status = next_status

        next_status = LifecycleStatus.DECISION
        ExperimentStateMachine.validate_transition(current_status, next_status, experiment_id)
        current_status = next_status

        outcome = self.decision_engine.evaluate_outcome(
            baseline_metrics=baseline_metrics,
            candidate_metrics=eval_metrics,
            baseline_params=163940,
            candidate_params=162325,
            baseline_vram_mb=4000,
            candidate_vram_mb=3900,
            is_simulation=is_simulation
        )
        decision = outcome["decision"]
        cycle_log.append(f"Step 8 [DECISION]: Verdict = {decision}. Rationale: {outcome['rationale']}")

        # 10. MEMORY
        next_status = LifecycleStatus.MEMORY
        ExperimentStateMachine.validate_transition(current_status, next_status, experiment_id)
        current_status = next_status

        record = {
            "experiment_id": experiment_id,
            "parent_experiment": parent_experiment,
            "hypothesis": hypothesis.hypothesis,
            "research_question": hypothesis.research_question,
            "change": exact_change,
            "files_changed": [f"research/experiments/{experiment_id}/config.json"],
            "git_commit": git_commit,
            "dataset_fingerprint": ds_fp,
            "split_hash": "inner_val_split_seed42",
            "seed": 42,
            "budget": compute_budget,
            "gpu": allocated_device,
            "metrics": {
                "macro_f1": eval_metrics["primary"]["macro_f1"],
                "accuracy": eval_metrics["secondary"]["accuracy"],
                "kappa": eval_metrics["secondary"]["kappa"]
            },
            "execution_mode": exec_mode,
            "lifecycle_status": LifecycleStatus.COMPLETED.value,
            "real_training": not is_simulation,
            "scientific_evidence": not is_simulation,
            "eligible_for_model_selection": not is_simulation,
            "eligible_for_research_conclusions": not is_simulation,
            "eligible_for_leaderboard": not is_simulation,
            "status": LifecycleStatus.COMPLETED.value,
            "decision": "SIMULATION_NON_EVIDENCE" if is_simulation else decision,
            "reason": (
                f"SIMULATION — NON-EVIDENCE: {outcome['rationale']}"
                if is_simulation else outcome["rationale"]
            ),
            "review": review["summary"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        if is_simulation:
            record["simulation_tag"] = "SIMULATION — NON-EVIDENCE"

        # Validate with memory validator before persisting
        ResearchMemoryValidator.validate_record(record)
        memory["experiments"].append(record)
        self.save_memory(memory)
        cycle_log.append(f"Step 9 [MEMORY]: Experiment {experiment_id} successfully persisted to research_memory.json.")

        return {"record": record, "log": cycle_log}
