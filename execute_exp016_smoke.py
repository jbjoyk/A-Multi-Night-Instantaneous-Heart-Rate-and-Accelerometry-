"""
execute_exp016_smoke.py: Single controlled execution of EXP-016-SMOKE.
Authorized under human instruction for Phase 3 infrastructure smoke test:
  - 1 epoch
  - cuda:0
  - budget: SMOKE
  - seed: 42
  - zero test set access
  - classification: NOT SCIENTIFIC EVIDENCE
"""
import os
import sys
import json
import time
import torch

from governor import ResearchGovernor, AutonomyLevel
from research_engine.hypothesis_agent import ResearchHypothesis


def main():
    print("=" * 80)
    print("GOVERNED PHASE-3 INFRASTRUCTURE SMOKE TEST: EXP-016-SMOKE")
    print("=" * 80)

    # Define authorized smoke test hypothesis
    hyp = ResearchHypothesis(
        hypothesis_id="HYP-INFRA-016",
        research_question="RQ-INFRA: Can the governed training lifecycle execute 1 real epoch on cuda:0 without violating safety gates?",
        hypothesis="Governed execution pipeline runs 1 epoch of Model A training within budget and resource limits without held-out leakage.",
        motivation="Phase-3 infrastructure readiness validation following Phase 2.1 hardening.",
        expected_mechanism="PyTorch training loop executes 1 epoch on cuda:0 with Adam optimizer, logs metrics, and releases GPU lock.",
        experiment_design="Train Model A for 1 epoch on 31 train subjects using batch size 4 on cuda:0 and evaluate on 5 validation subjects.",
        expected_outcome="Completion of 1 epoch with valid training and validation metrics, peak VRAM under 10240 MB, and zero leakage.",
        novelty_rationale="Infrastructure pipeline verification [candidate novelty requiring literature verification]",
        feasibility="High; verified cache and environment.",
        risks="CUDA OOM or unhandled exception during backward pass.",
        required_compute="SMOKE (1 epoch, < 600 seconds on 1x RTX 3060)",
        evaluation_criteria="Validation Macro-F1 parity (delta >= -0.010) on inner validation split"
    )

    # Initialize Governor at Level 4 (Supervised Execution) with explicit human authorization
    gov = ResearchGovernor(autonomy_level=AutonomyLevel.LEVEL_4_SUPERVISED_EXECUTION)

    t0 = time.time()
    result = gov.run_governed_cycle(
        hypothesis=hyp,
        experiment_id="EXP-016-SMOKE",
        parent_experiment="EXP-014",
        baseline_model="Model A",
        exact_change="Infrastructure smoke test 1 epoch on cuda:0",
        fixed_components=["HREncoder", "BiGRU", "RCCLoss"],
        compute_budget="SMOKE",
        dry_run=False,
        human_approved=True,
        execution_mode="INFRASTRUCTURE_SMOKE_TEST"
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 80)
    print("EXP-016-SMOKE EXECUTION COMPLETED")
    print(f"Total Wallclock Elapsed: {elapsed:.2f} seconds")
    print("=" * 80)

    log = result.get("log", [])
    for entry in log:
        print(f"  {entry}")

    rec = result.get("record", {})
    print("\nRecord Summary:")
    print(f"  Experiment ID:        {rec.get('experiment_id')}")
    print(f"  Execution Mode:       {rec.get('execution_mode')}")
    print(f"  Lifecycle Status:     {rec.get('lifecycle_status')}")
    print(f"  Real Training:        {rec.get('real_training')}")
    print(f"  Scientific Evidence:  {rec.get('scientific_evidence')}")
    print(f"  Leaderboard Eligible: {rec.get('eligible_for_leaderboard')}")
    print(f"  Decision:             {rec.get('decision')}")
    print(f"  Simulation Tag:       {rec.get('simulation_tag')}")
    print(f"  Metrics:              {rec.get('metrics')}")
    print(f"  GPU:                  {rec.get('gpu')}")
    print(f"  Git Commit:           {rec.get('git_commit')}")
    print(f"  Dataset Fingerprint:  {rec.get('dataset_fingerprint')}")
    print(f"  Split Hash:           {rec.get('split_hash')}")

    return result


if __name__ == "__main__":
    main()
