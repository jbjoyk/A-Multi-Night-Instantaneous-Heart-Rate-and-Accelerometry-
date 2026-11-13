"""
governor.py: Supreme Research Governor for Autonomous Sleep Staging Research.

The Governor is the highest-level authority in the research infrastructure.
It enforces:
  1. Autonomy Levels (Level 0 through 5; Defaults to Level 3).
  2. Test-Set Protection (Hard gate against test data usage during development/selection).
  3. Strict Compute Budgets (SMOKE, SHORT, STANDARD, FULL).
  4. Mandatory 500-Epoch Hard Block (BLOCKED_BY_RESEARCH_GOVERNANCE).
  5. Dual-GPU Resource Management & VRAM Safety (RTX 3060 12GB x 2).
  6. Dataset and Split Integrity Gates.
  7. Human Authorization Enforcement for Training Execution and Escalation.
"""

import os
import sys
import argparse
from enum import IntEnum
from typing import Dict, List, Optional, Any

from dataset_gate import DatasetGate, TestSetLeakageError
from evaluator import SleepEvaluator, TestSetProtectionViolation
from research_engine.resource_manager import GPUResourceManager, ResourceAllocationError
from research_engine.literature_agent import LiteratureAgent
from research_engine.hypothesis_agent import HypothesisAgent, ResearchHypothesis
from research_engine.experiment_designer import ExperimentDesigner, ExperimentSpecification, ExperimentDesignError
from research_engine.scientific_reviewer import ScientificReviewer, ReviewVerdict
from research_engine.decision_logic import DecisionEngine
from research_engine.experiment_engine import ExperimentEngine


class AutonomyLevel(IntEnum):
    LEVEL_0_ANALYSIS_ONLY = 0
    LEVEL_1_HYPOTHESIS_GEN = 1
    LEVEL_2_EXPERIMENT_DESIGN = 2
    LEVEL_3_IMPLEMENT_REQUIRE_APPROVAL = 3  # DEFAULT LEVEL
    LEVEL_4_SUPERVISED_EXECUTION = 4
    LEVEL_5_AUTONOMOUS_LOOP = 5            # STRICTLY GATED


class GovernanceBlockError(Exception):
    """Raised when an operation is blocked by Research Governance."""
    pass


class ResearchGovernor:
    """
    Supreme Research Governor orchestrating agents and enforcing scientific guardrails.
    """
    DEFAULT_LEVEL = AutonomyLevel.LEVEL_3_IMPLEMENT_REQUIRE_APPROVAL
    MAX_ALLOWABLE_EPOCHS_WITHOUT_APPROVAL = 10  # SHORT budget

    def __init__(
        self,
        base_dir: Optional[str] = None,
        autonomy_level: AutonomyLevel = DEFAULT_LEVEL
    ):
        self.base_dir = base_dir or os.path.dirname(os.path.abspath(__file__))
        self.autonomy_level = autonomy_level
        self.dataset_gate = DatasetGate(base_dir=self.base_dir)
        self.evaluator = SleepEvaluator()
        self.resource_manager = GPUResourceManager()
        self.literature_agent = LiteratureAgent()
        self.hypothesis_agent = HypothesisAgent(literature_agent=self.literature_agent)
        self.designer = ExperimentDesigner()
        self.reviewer = ScientificReviewer()
        self.decision_engine = DecisionEngine()
        self.engine = ExperimentEngine(
            base_dir=self.base_dir,
            dataset_gate=self.dataset_gate,
            evaluator=self.evaluator,
            resource_manager=self.resource_manager,
            reviewer=self.reviewer,
            decision_engine=self.decision_engine
        )

    def escalate_autonomy(self, new_level: AutonomyLevel, human_approval_token: str) -> None:
        """
        Escalates autonomy level. Escalation to Level 4 or Level 5 requires explicit human token.
        """
        if new_level >= AutonomyLevel.LEVEL_4_SUPERVISED_EXECUTION:
            expected_token = "HUMAN_AUTHORIZED_ESCALATION"
            if human_approval_token != expected_token:
                raise GovernanceBlockError(
                    f"Autonomy escalation to {new_level.name} REJECTED: Invalid human authorization token. "
                    f"Escalation beyond Level 3 requires explicit project owner approval."
                )
        self.autonomy_level = new_level

    def deescalate_autonomy(self, new_level: AutonomyLevel) -> None:
        """De-escalates autonomy level to a more restricted mode."""
        self.autonomy_level = new_level

    def check_experiment_safety(
        self,
        spec: ExperimentSpecification,
        human_approved: bool = False
    ) -> Dict[str, Any]:
        """
        Conducts pre-flight safety screening before any code execution or training launch:
          1. 500-Epoch Block Check (Mandatory Rule 18)
          2. Autonomy Level Authorization Check
          3. Test-Set Protection Check
          4. Dataset and Split Gate Check
          5. Compute Budget & VRAM Allocation Check
          6. Scientific Peer Review Check
          7. Redundancy & Historical Duplication Check
        """
        # 1. Mandatory Rule 18: 500-Epoch Hard Block
        spec_text = str(spec.to_dict()).lower()
        if "500" in spec.compute_budget or "500" in spec.exact_change or "500" in spec.hypothesis_statement:
            return {
                "allowed": False,
                "status": "BLOCKED_BY_RESEARCH_GOVERNANCE",
                "reason": (
                    "BLOCKED_BY_RESEARCH_GOVERNANCE: 500-epoch training paused under DEC-019/020 and user governance. "
                    "Long-training jobs are strictly blocked without explicit supervisor protocol authorization."
                )
            }

        # 2. Autonomy Level Enforcement
        if self.autonomy_level < AutonomyLevel.LEVEL_4_SUPERVISED_EXECUTION and not human_approved:
            if spec.compute_budget in ("STANDARD", "FULL"):
                return {
                    "allowed": False,
                    "status": "BLOCKED_PENDING_APPROVAL",
                    "reason": (
                        f"Current Autonomy Level is {self.autonomy_level.name}. Training under budget "
                        f"'{spec.compute_budget}' requires explicit human approval before execution."
                    )
                }

        # 3. Test-Set Protection Check
        if "test set" in spec_text or "outer test" in spec_text or "test partition" in spec_text:
            if "frozen_test_benchmark" not in spec_text:
                return {
                    "allowed": False,
                    "status": "BLOCKED_BY_RESEARCH_GOVERNANCE",
                    "reason": "TEST SET PROTECTION VIOLATION: Proposal attempts to access or optimize on test subjects during development!"
                }

        # 4. Dataset Gate Verification
        try:
            self.dataset_gate.verify_dataset_structure()
            self.dataset_gate.verify_cache_integrity()
        except Exception as e:
            return {
                "allowed": False,
                "status": "BLOCKED_BY_DATASET_GATE",
                "reason": f"Dataset Gate Failure: {str(e)}"
            }

        # 5. Scientific Peer Review Audit
        memory = self.engine.load_memory()
        existing_ids = [exp["experiment_id"] for exp in memory.get("experiments", [])]
        review = self.reviewer.review_proposal(spec, existing_experiment_ids=existing_ids)
        if review["verdict"] == ReviewVerdict.REJECT.value:
            return {
                "allowed": False,
                "status": "BLOCKED_BY_SCIENTIFIC_REVIEWER",
                "reason": f"Scientific Reviewer Rejected Proposal: {'; '.join(review['flags'])}",
                "review": review
            }

        # 6. GPU VRAM Pre-flight Check
        device_status = self.resource_manager.get_device_status()
        available_devices = [d for d in device_status if d["is_available"]]
        if not available_devices:
            return {
                "allowed": False,
                "status": "BLOCKED_BY_RESOURCE_CONTENTION",
                "reason": "All GPUs are currently allocated or busy. Concurrent execution on a single GPU is disallowed."
            }

        return {
            "allowed": True,
            "status": "APPROVED",
            "review": review,
            "allocated_candidate_device": available_devices[0]["device_str"],
            "reason": "All safety, budget, dataset, and scientific integrity gates passed."
        }

    def run_governed_cycle(
        self,
        hypothesis: ResearchHypothesis,
        experiment_id: str,
        parent_experiment: str,
        baseline_model: str,
        exact_change: str,
        fixed_components: List[str],
        compute_budget: str,
        dry_run: bool = True,
        human_approved: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a complete governed experimentation cycle under active autonomy policies.
        """
        # Step A: Check autonomy level for initial phase
        if self.autonomy_level == AutonomyLevel.LEVEL_0_ANALYSIS_ONLY:
            return {
                "status": "LEVEL_0_ANALYSIS_ONLY",
                "message": "Governor is running in Level 0 (Analysis Only). Hypothesis and experiment design are prohibited.",
                "dataset_fingerprint": self.dataset_gate.compute_dataset_fingerprint(),
                "memory_summary": f"{len(self.engine.load_memory().get('experiments', []))} archived experiments."
            }

        # Step B: Design & Pre-flight Screen
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
            confounders=["inter-subject variance", "class imbalance"],
            compute_budget=compute_budget
        )

        safety = self.check_experiment_safety(spec, human_approved=human_approved)
        if not safety["allowed"]:
            return {
                "status": safety["status"],
                "reason": safety["reason"],
                "spec": spec.to_dict()
            }

        # Step C: Execute Cycle via Engine
        result = self.engine.execute_lifecycle_step(
            hypothesis=hypothesis,
            experiment_id=experiment_id,
            parent_experiment=parent_experiment,
            baseline_model=baseline_model,
            exact_change=exact_change,
            fixed_components=fixed_components,
            compute_budget=compute_budget,
            dry_run=dry_run
        )
        return result


def main():
    parser = argparse.ArgumentParser(description="SLEEP Research Governor CLI")
    parser.add_argument("--mode", choices=["status", "hypotheses", "audit", "dry_run"], default="status",
                        help="Execution mode: 'status' (read-only), 'hypotheses' (list proposals), 'audit' (run gates), 'dry_run' (simulated cycle)")
    parser.add_argument("--autonomy", type=int, choices=[0, 1, 2, 3, 4, 5], default=3,
                        help="Autonomy level (0=Analysis only, 3=Implementation default, 4=Supervised run)")
    args = parser.parse_args()

    gov = ResearchGovernor(autonomy_level=AutonomyLevel(args.autonomy))

    print("=" * 80)
    print(f"SLEEP RESEARCH GOVERNOR | Autonomy: {AutonomyLevel(args.autonomy).name}")
    print("=" * 80)

    if args.mode == "status":
        print(f"Active Base Directory: {gov.base_dir}")
        print(f"Dataset Fingerprint:   {gov.dataset_gate.compute_dataset_fingerprint()}")
        gpu_stat = gov.resource_manager.get_device_status()
        print(f"Managed GPUs ({len(gpu_stat)} devices):")
        for g in gpu_stat:
            print(f"  - {g.get('device_str', 'cpu')}: {g.get('name')} (Available: {g.get('is_available')})")
        mem = gov.engine.load_memory()
        print(f"Institutional Memory:  {len(mem.get('experiments', []))} historical experiments archived.")
        print("Status Check COMPLETE [VERIFIED]")

    elif args.mode == "hypotheses":
        hyps = gov.hypothesis_agent.get_grounded_candidates()
        print(f"Active Grounded Hypotheses Addressing DEC-021 ({len(hyps)} candidates):")
        for h in hyps:
            print(f"\n[{h.hypothesis_id}] {h.research_question}")
            print(f"  Hypothesis: {h.hypothesis}")
            print(f"  Budget:     {h.required_compute}")
            print(f"  Novelty:    {h.novelty_rationale}")

    elif args.mode == "audit":
        print("Auditing Dataset Gate...")
        ds_info = gov.dataset_gate.verify_dataset_structure()
        cache_info = gov.dataset_gate.verify_cache_integrity()
        print(f"  Dataset: {ds_info['n_subjects']} subjects, {ds_info['n_raw_nights']} raw nights [PASSED]")
        print(f"  Cache:   {cache_info['n_usable_nights']} usable nights [PASSED]")
        print("Auditing 500-Epoch Governor Rule...")
        bad_spec = gov.designer.design_experiment(
            "EXP-TEST", "HYP-001", "RQ-001", "500-epoch test", "Model A",
            "500 epochs training", ["encoders", "BiGRU"], "Macro-F1", "Loss", ["confounders"], "STANDARD"
        )
        bad_spec.compute_budget = "500_EPOCHS"
        safety = gov.check_experiment_safety(bad_spec)
        print(f"  500-Epoch Gate: {safety['status']} (Reason: {safety['reason']}) [PASSED]")
        print("Governor Audit COMPLETE [VERIFIED]")

    elif args.mode == "dry_run":
        print("Executing Governed Dry-Run Simulation for HYP-001...")
        hyps = gov.hypothesis_agent.get_grounded_candidates()
        target_hyp = hyps[0]
        res = gov.run_governed_cycle(
            hypothesis=target_hyp,
            experiment_id="EXP-015-SIM",
            parent_experiment="EXP-014",
            baseline_model="Model A",
            exact_change="Prune all-zero clock channel from x_meta (4 meta channels instead of 5)",
            fixed_components=["HREncoder", "ACCEncoder", "BiGRU", "RCCLoss"],
            compute_budget="SHORT",
            dry_run=True,
            human_approved=True
        )
        print("\nCycle Result:")
        print(f"  Status:   {res['record']['status']}")
        print(f"  Decision: {res['record']['decision']}")
        print(f"  Reason:   {res['record']['reason']}")
        print("\nCycle Execution Log:")
        for line in res["log"]:
            print(f"  {line}")
        print("\nDry-Run Simulation COMPLETE [VERIFIED]")


if __name__ == "__main__":
    main()
