# Research Governance, Autonomy Policies, and Scientific Safeguards

## 1. Governance Principles

The **Research Governor** (`governor.py`) serves as the supreme operational and ethical authority in the SLEEP research infrastructure. Autonomous agents, experiment loops, and optimization scripts are strictly subordinate to the Governor's policies.

The core mandate of Research Governance is:
1. Prevent data leakage and protect the held-out test cohort.
2. Prevent un-monitored or excessive compute expenditure.
3. Enforce deterministic, reproducible science over speculative score-chasing.
4. Guarantee that every code modification tests an explicit, falsifiable scientific hypothesis.

---

## 2. Autonomy Levels and Escalation Protocol

The system defines six explicit, non-overlapping autonomy levels:

| Level | Name | Permissions and Scope | Human Approval Required? |
| :---: | :--- | :--- | :---: |
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
- If any proposed experiment requests $\ge 500$ epochs, or attempts to resume the historical long-run convergence protocol, the Governor immediately blocks execution:
  ```
  Status: BLOCKED_BY_RESEARCH_GOVERNANCE
  Reason: 500-epoch training paused under DEC-019/020 and user governance. Long-training jobs require explicit supervisor protocol authorization.
  ```

---

## 4. Hardware Allocation and Dual-GPU Guardrails

The workstation is equipped with:
- **GPU 0**: NVIDIA GeForce RTX 3060 12 GB
- **GPU 1**: NVIDIA GeForce RTX 3060 12 GB
- **Host RAM**: 32 GB | **CPU**: AMD Ryzen 7 7700 (16 threads)

### Resource Management Guardrails:
1. **Hard VRAM Ceiling**: The `GPUResourceManager` enforces a maximum allocation ceiling of **10.0 GB per device**, guaranteeing a 2.0 GB buffer for CUDA runtime contexts and OS display overhead.
2. **Single-Job Exclusivity**: Concurrent execution of two memory-intensive training runs on the same physical GPU is strictly prohibited to eliminate out-of-memory errors and context switching thrash.
3. **Dual-Device Distribution**: Independent tasks (e.g., parallel validation screening or independent fold evaluations) must be explicitly mapped to disjoint devices (`cuda:0` and `cuda:1`).

---

## 5. Test-Set Protection Gates (Hard Non-Bypassable Safeguards)

Protecting the held-out test cohort from leakage or subtle selection bias is of paramount scientific importance.

### Safeguards Implemented:
1. **Dataset Gate (`DatasetGate.assert_no_test_leakage`)**:
   - Compares the subject IDs accessed by the active dataloader against the held-out test partition.
   - If any subject from the test partition appears in an operation labeled `train`, `validation`, `search`, `tuning`, `feature_selection`, or `early_stopping`, a `TestSetLeakageError` is thrown immediately, killing the job.
2. **Evaluator Gate (`SleepEvaluator.evaluate_model`)**:
   - If `split_role == 'test'`, evaluation is blocked unless the flag `frozen_evaluation_authorized=True` is provided.
   - Any attempt to compute test metrics during development or hypothesis ranking raises `TestSetProtectionViolation`.
3. **No Optimization on Test Scores**:
   - The Decision Engine operates strictly on inner-validation metrics. Test metrics cannot be fed into the keep/discard decision loop.

---

## 6. The Governed Experiment Lifecycle

Every research investigation follows the strict 9-stage finite state machine:

```
[PROPOSE] ---> [GOVERN] ---> [DESIGN] ---> [REVIEW] ---> [IMPLEMENT]
                                                            |
[MEMORY] <--- [DECISION] <--- [EVALUATE] <--- [RUN] <-------+
```

1. **`PROPOSE`**: Hypothesis Agent drafts a proposal addressing an open gap from `research_queue.yaml` adhering to the 10-point schema.
2. **`GOVERN`**: Governor screens for budget compliance, 500-epoch rule, and autonomy permissions.
3. **`DESIGN`**: Experiment Designer specifies exact single-factor modification and fixed controls.
4. **`REVIEW`**: Scientific Reviewer audits proposal across 10 critical standards (`ACCEPT`, `REVISE`, or `REJECT`).
5. **`IMPLEMENT`**: Workspace prepared under Git commit tracking (`research/experiments/EXP-XXX/`).
6. **`RUN`**: GPU allocated via `GPUResourceManager`; training executed under monitored budget.
7. **`EVALUATE`**: `SleepEvaluator` scores model against inner-validation split using Macro-F1, accuracy, $\kappa$, and ECE.
8. **`DECISION`**: `DecisionEngine` evaluates multi-criteria trade-offs (`KEPT`, `DISCARDED`, `NEEDS_REVIEW`, `INVESTIGATE`).
9. **`MEMORY`**: Comprehensive experiment metadata and metrics appended to `research_memory.json`.
