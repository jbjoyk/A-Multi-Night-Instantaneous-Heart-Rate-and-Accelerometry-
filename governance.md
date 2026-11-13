# Research Governance, Autonomy Policies, and Scientific Safeguards

## 1. Governance Principles

The **Research Governor** (`governor.py`) serves as the supreme operational and ethical authority in the SLEEP research infrastructure. Autonomous agents, experiment loops, and optimization scripts are strictly subordinate to the Governor's policies.

The core mandate of Research Governance is:
1. Prevent data leakage and protect the held-out test cohort.
2. Prevent un-monitored or excessive compute expenditure.
3. Enforce deterministic, reproducible science over speculative score-chasing.
4. Guarantee that every code modification tests an explicit, falsifiable scientific hypothesis.
5. Strictly shield empirical leaderboards and baseline selection from simulation and dry-run contamination.

---

## 2. Autonomy Levels and Escalation Protocol

The system defines six explicit, non-overlapping autonomy levels:

| Level | Name | Permissions and Scope | Human Approval Required? |
| :---: | :--- | :--- | :--- |
| **LEVEL 0** | **Analysis Only** | Read-only inspection of repository state, research memory, historical metrics, and audit dossiers. Cannot propose or execute experiments. | No |
| **LEVEL 1** | **Hypothesis Generation** | Proposes scientific hypotheses formatted according to the 10-point schema. Identifies research gaps from memory and literature. | No |
| **LEVEL 2** | **Experiment Design** | Transforms hypotheses into controlled OFAT experiment specifications. Assesses feasibility and compute budgets. | No |
| **LEVEL 3** | **Implementation & Preparation** *(DEFAULT)* | Prepares code diffs, configuration files, and experiment branches. Passes through pre-flight safety screening. **Execution is paused until human approval.** | **YES (for training launch)** |
| **LEVEL 4** | **Supervised Execution** | Automatically launches approved `SMOKE` (1–2 epochs) and `SHORT` (10 epochs) experiments. `STANDARD` and `FULL` budgets require explicit human authorization. | **YES (for STANDARD/FULL)** |
| **LEVEL 5** | **Autonomous Experiment Loop** | Unrestricted, iterative loop proposing, training, evaluating, and pruning experiments autonomously. | **STRICTLY GATED** |

### Escalation Safeguard:
- The system **strictly defaults to Level 3**.
- Autonomous escalation to Level 4 or Level 5 is blocked unless the user explicitly supplies the cryptographic token:
  `HUMAN_AUTHORIZED_ESCALATION`
- The system will never self-escalate autonomy levels.

---

## 3. Compute Budgets and Execution Limits

To prevent accidental resource exhaustion and run-away loops, all experiments must be assigned to one of four frozen compute budgets:

| Budget Tier | Max Epochs | Batch Size | Max Runtime | VRAM Ceiling | Target Use Case |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **`SMOKE`** | 1–2 | 4 | 60 seconds | 2.0 GB | Unit tests, gradient flow verification, shape assertions, dry-run profiling. |
| **`SHORT`** | 10 | 4 | 5 minutes | 4.0 GB | Rapid component screening, ablation screening (e.g. EXP-007, boundary weight sweep). |
| **`STANDARD`** | 80 | 4 | 20 minutes | 6.0 GB | Single-seed convergence benchmarks (matching locked EXP-010/011 peak). |
| **`FULL`** | 80 $\times$ 5 folds | 4 | 120 minutes | 8.0 GB | Comprehensive 5-fold cross-validation or multi-seed stability evaluation (e.g. EXP-014). |

### Mandatory Rule: 500-Epoch Hard Block
- Under **DEC-019**, **DEC-020**, and user governance, the 500-epoch experimental protocol is explicitly paused.
- The Governor enforces numerical parsing of requested epochs:
  - Any request with $\ge 500$ epochs (e.g., 500, 501, 1000) or negative epochs is immediately blocked:
    ```
    Status: BLOCKED_BY_RESEARCH_GOVERNANCE
    Reason: 500-epoch training requested. 500-epoch training paused under DEC-019/020 and user governance. Training >= 500 epochs is paused and strictly blocked.
    ```
  - Any epoch count exceeding the budget tier ceiling is blocked.

---

## 4. Hardware Allocation and Dual-GPU Guardrails

The workstation is equipped with:
- **GPU 0**: NVIDIA GeForce RTX 3060 12 GB
- **GPU 1**: NVIDIA GeForce RTX 3060 12 GB
- **Host RAM**: 32 GB | **CPU**: AMD Ryzen 7 7700 (16 threads)

### Resource Management Guardrails:
1. **Hard VRAM Ceiling**: The `GPUResourceManager` enforces a maximum allocation ceiling of **10.0 GB per device**, guaranteeing a 2.0 GB buffer for CUDA runtime contexts and OS display overhead.
2. **Single-Job Exclusivity**: Concurrent execution of two memory-intensive training runs on the same physical GPU is strictly prohibited to eliminate out-of-memory errors and context switching thrash.
3. **Safe Context Manager Leases**: GPUs must be leased via `with resource_manager.lease(...):`, guaranteeing immediate cleanup on success, runtime failure, or unexpected exception.
4. **Stale Lease Recovery**: An automated garbage collector releases locks older than 3600 seconds, preventing hung tasks from locking hardware.

---

## 5. Test-Set Protection Gates (Hard Non-Bypassable Safeguards)

Protecting the held-out test cohort from leakage or subtle selection bias is of paramount scientific importance.

### Safeguards Implemented:
1. **Dataset Gate (`DatasetGate.assert_no_test_leakage`)**:
   - Compares the subject IDs accessed by the active dataloader against the held-out test partition.
   - If any subject from the test partition appears in an operation labeled `train`, `validation`, `search`, `tuning`, `feature_selection`, `calibration`, or `early_stopping`, a `TestSetLeakageError` is thrown immediately, killing the job.
2. **Evaluator Gate (`SleepEvaluator.evaluate_model`)**:
   - If `split_role == 'test'`, evaluation is blocked unless the flag `frozen_evaluation_authorized=True` is provided.
   - Any attempt to compute test metrics during development or hypothesis ranking raises `TestSetProtectionViolation`.
3. **No Optimization on Test Scores**:
   - The Decision Engine operates strictly on inner-validation metrics. Test metrics cannot be fed into the keep/discard decision loop.

---

## 6. Simulation Non-Evidence Boundaries & Data Model Decoupling

To prevent simulated or dry-run metrics from contaminating real scientific evidence:
1. **Decoupled Attributes**:
   - `execution_mode`: `REAL`, `SIMULATION`, `DRY_RUN`
   - `lifecycle_status`: `PROPOSED`, `GOVERNED`, `DESIGNED`, `REVIEWED`, `APPROVED`, `IMPLEMENTING`, `RUNNING`, `EVALUATING`, `COMPLETED`, `FAILED`, `BLOCKED`, `DISCARDED`, `NEEDS_REVIEW`, `MEMORY`
2. **Explicit Semantic Invariants**:
   Every simulation record MUST explicitly declare:
   ```json
   "real_training": false,
   "scientific_evidence": false,
   "eligible_for_model_selection": false,
   "eligible_for_research_conclusions": false,
   "eligible_for_leaderboard": false,
   "simulation_tag": "SIMULATION — NON-EVIDENCE"
   ```
3. **Leaderboard Shielding**:
   - The institutional memory leaderboard (`ResearchMemoryValidator.get_leaderboard`) strictly filters for `execution_mode == 'REAL'` and `scientific_evidence == true`.
   - Simulation results cannot become best models, influence rankings, or serve as baselines.
4. **Anti-Mutation Invariant**:
   - Attempting to mutate `execution_mode` from `SIMULATION` to `REAL` in-place raises `ExecutionModeViolationError`.

---

## 7. The Governed Experiment Lifecycle State Machine

Every research investigation follows the strict 11-stage finite state machine enforced by `ExperimentStateMachine`:

```
[PROPOSE] ---> [GOVERN] ---> [DESIGN] ---> [REVIEW] ---> [APPROVE] ---> [IMPLEMENT]
                                                                             |
[MEMORY] <--- [DECISION] <--- [COMPLETED] <--- [EVALUATE] <--- [RUN] <-------+
```

Terminal Exception States:
- `BLOCKED`: Triggered by safety, 500-epoch, or dataset gate violations.
- `DISCARDED`: Triggered by reviewer veto or model performance degradation.
- `FAILED`: Triggered by execution crash or numerical instability.
- `NEEDS_REVIEW`: Triggered by ambiguous trade-offs or complexity traps.

### Anti-Bypass Execution Token:
- Direct execution of `ExperimentEngine` without Governor screening is strictly prohibited.
- When an experiment is approved by the Governor, an HMAC execution token is issued:
  ```python
  token = governor.generate_governance_token(experiment_id, compute_budget)
  ```
- The engine validates this token; missing or forged tokens immediately raise `GovernanceBlockError`.

---

## 8. Git Governance and Branch Isolation

- Current State: Commits are tracked locally on `main` for Claude/collaborator synchronization.
- Governance Risk: Direct autonomous modification of `main` without PR or human review risks publishing unvetted code.
- Recommended Staged Branch Workflow:
  ```text
  feature branch (research/EXP-XXX)
          ↓
  automated verification (67/67 unit tests pass)
          ↓
  scientific peer review (ReviewVerdict.ACCEPT)
          ↓
  human supervisor authorization
          ↓
  merge to main
  ```
