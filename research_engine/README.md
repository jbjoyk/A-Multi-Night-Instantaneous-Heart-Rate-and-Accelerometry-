# research_engine

## Overview
The `research_engine` component implements the autonomous, governed scientific research pipeline. It provides strict state-machine lifecycle enforcement, hypothesis generation, experimental design validation, peer-review simulation, dual-GPU resource arbitration, and immutable research memory validation.

## Interface
- `DecisionEngine` (`decision_logic.py`): Evaluates empirical outcomes (`evaluate_outcome`) into triage actions (`PROMOTE`, `ITERATE`, `REJECT`).
- `ExperimentDesigner` (`experiment_designer.py`): Validates and generates structured specifications (`design_experiment`, `ExperimentSpecification`).
- `ExperimentEngine` (`experiment_engine.py`): Orchestrates execution lifecycle steps (`execute_lifecycle_step`, `load_memory`, `save_memory`, `verify_governance_token`).
- `HypothesisAgent` (`hypothesis_agent.py`): Formulates grounded hypotheses against research priors (`create_hypothesis`, `get_grounded_candidates`).
- `LiteratureAgent` (`literature_agent.py`): Cross-references claims against scientific databases (`audit_claim`, `search_external_literature`).
- `ResearchMemoryValidator` (`memory_validator.py`): Enforces structural integrity of `research_memory.json`, leaderboard extraction, and partition between simulation and real experiments.
- `GPUResourceManager` (`resource_manager.py`): Manages lockfiles, device validation, and context-managed leases (`allocate_gpu`, `release_gpu`, `lease`) for dual RTX 3060 GPUs.
- `ScientificReviewer` (`scientific_reviewer.py`): Automated peer-review gatekeeper validating experimental proposals against historical baselines (`review_proposal`).
- `ExperimentStateMachine` (`state_machine.py`): State lifecycle transition validation (`validate_transition`, `validate_execution_mode_invariance`).

## Invariants
- `ResearchMemoryValidator.assert_registry_consistency`: Simulations (`ExecutionMode.SIMULATION`) and live trials (`ExecutionMode.REAL`) are strictly partitioned; simulation data can never enter leaderboards or satisfy empirical evidence criteria.
- `ExperimentStateMachine.validate_transition`: All experiments must transition monotonically through `PROPOSED -> REVIEWED -> APPROVED -> IN_PROGRESS -> COMPLETED / FAILED / BLOCKED`. Out-of-order transitions raise `IllegalStateTransitionError`.
- `ExperimentStateMachine.validate_execution_mode_invariance`: `execution_mode` is immutable once an experiment record is created.
- `GPUResourceManager.lease`: GPU allocation requires an active non-expired lease, preventing multi-process hardware contention across `cuda:0` and `cuda:1`.

## Side Effects
- Reads and updates `research_memory.json` on disk via `ExperimentEngine.save_memory`.
- Executes local git subprocesses to inspect commit hashes (`git rev-parse HEAD`).
- Creates and deletes file-based lock and lease tracking markers during GPU allocation.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_experiment_engine_simulated.py tests/test_memory_integrity.py tests/test_scientific_reviewer.py tests/test_state_machine.py tests/test_gpu_safety_advanced.py`
