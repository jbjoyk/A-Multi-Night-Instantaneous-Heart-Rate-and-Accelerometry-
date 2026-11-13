# Autonomous Research Infrastructure Architecture

## 1. System Overview

This document specifies the autonomous experimentation and scientific research architecture implemented for the **SLEEP Research Project** (wearable sleep staging on the PhysioNet BIDSLab dataset).

The architecture synthesizes the ideation and peer-review principles of **SakanaAI's AI-Scientist** with the iterative experiment lifecycle and Git-based isolation of **Karpathy's Autoresearch**, while introducing a mandatory, overarching **Research Governor** that prevents unauthorized training, data leakage, and resource contention.

```
                             +-----------------------------------+
                             |      SLEEP Research Governor      |
                             |  (Supreme Authority & Guardrails) |
                             +-----------------+-----------------+
                                               |
         +--------------------+----------------+--------------------+--------------------+
         |                    |                |                    |                    |
+--------v-------+   +--------v-------+ +------v------+   +---------v--------+  +--------v-------+
|   Literature   |   |   Hypothesis   | | Experiment  |   |    Scientific    |  |    Resource    |
|     Agent      |   |     Agent      | |  Designer   |   |     Reviewer     |  |    Manager     |
| (Grounding &   |   | (10-Pt Schema  | | (OFAT Single|   | (10-Point Audit; |  | (Dual RTX 3060 |
|  Evidence QC)  |   | & Cand. Nov.)  | |  Variable)  |   |  Autonomous Veto)|  |  VRAM Limits)  |
+--------+-------+   +--------+-------+ +------+------+   +---------+--------+  +--------+-------+
         |                    |                |                    |                    |
         +--------------------+----------------+--------------------+--------------------+
                                               |
                                     +---------v--------+
                                     |   Dataset Gate   |
                                     | (Split Hash &    |
                                     | Test Protection) |
                                     +---------+--------+
                                               |
                                     +---------v--------+
                                     |   SleepEvaluator |
                                     | (Macro-F1 / ECE /|
                                     | Bootstrap CIs)   |
                                     +---------+--------+
                                               |
                                     +---------v--------+
                                     | Research Memory  |
                                     | (Institutional   |
                                     | JSON Registry)   |
                                     +------------------+
```

---

## 2. Core Architectural Components

### 2.1 The Research Governor (`governor.py`)
The Governor is the supreme authority in the system. Subagents and execution loops cannot bypass it.
- **Autonomy Level Enforcement**: Operates across Levels 0 to 5; defaults to **Level 3 (Implementation / Require Approval)**. Unsupervised execution of large runs (Level 4/5) is strictly gated.
- **Pre-Flight Safety Screening**:
  1. *500-Epoch Rule*: Blocks any experiment requesting $\ge 500$ epochs (`BLOCKED_BY_RESEARCH_GOVERNANCE`).
  2. *Test-Set Protection Gate*: Rejects any proposal accessing held-out test subjects during development, tuning, or ranking.
  3. *Dataset & Split Gate*: Validates cohort integrity (47 subjects, 253 raw nights, 246 usable nights) and verifies strict subject disjointness.
  4. *Resource Contention Gate*: Precludes concurrent execution of heavy training tasks on the same GPU.
  5. *Reviewer Veto Gate*: Requires unanimous approval from the Scientific Reviewer.

### 2.2 Literature Agent (`research_engine/literature_agent.py`)
Inspired by SakanaAI/AI-Scientist:
- Maintains a curated index of verified literature (e.g., IEEE TBME 2026 SLAMSS-IFS, Ganin & Lempitsky 2015 GRL, PhysioNet BIDSLab).
- Classifies every statement into 4 mutually exclusive evidence categories:
  1. `VERIFIED_LITERATURE_EVIDENCE`: Published peer-reviewed findings with verified citations.
  2. `PROJECT_OBSERVATION`: Empirically verified findings within this repository (EXP-001 through EXP-014, AUDIT_ANSWERS).
  3. `LLM_INFERENCE`: Logical deduction derived from data/architecture properties.
  4. `SPECULATIVE_HYPOTHESIS`: Untested conjectures requiring empirical experimentation.
- **Novelty Grounding Rule**: Subagents are strictly prohibited from declaring an idea "novel" on their own authority. All novelty claims are stamped as **"candidate novelty requiring literature verification"**.

### 2.3 Hypothesis Agent (`research_engine/hypothesis_agent.py`)
Formulates testable hypotheses anchored in verified research gaps (e.g. DEC-021: sequence truncation, dummy clock feature pruning, GRL gradient scaling, multi-night longitudinal variance).
Every hypothesis must satisfy the **10-Point Schema**:
1. `hypothesis`
2. `motivation`
3. `expected_mechanism`
4. `experiment_design`
5. `expected_outcome`
6. `novelty_rationale` (candidate novelty label enforced)
7. `feasibility`
8. `risks`
9. `required_compute`
10. `evaluation_criteria`

### 2.4 Experiment Designer (`research_engine/experiment_designer.py`)
Converts hypotheses into executable, controlled experiment specifications.
- **Single-Variable Control**: Enforces One-Factor-At-A-Time (OFAT) ablations. Detects and rejects multi-variable kitchen-sink mutations (e.g., changing learning rate, architecture, and loss function simultaneously).
- **Control Invariance**: Requires explicit declaration of fixed components (`fixed_components`) alongside the single mutated factor (`exact_change`).

### 2.5 Scientific Reviewer (`research_engine/scientific_reviewer.py`)
Acts as an autonomous, hostile peer reviewer before compute is allocated. Audits proposals across 10 critical dimensions:
1. *Data & Split Leakage* (checking test subject isolation).
2. *Invalid Comparisons* (mismatched parameter capacities or metrics).
3. *Metric Inconsistencies* (mandating Macro-F1 priority on class-imbalanced sleep data).
4. *Overfitting & VRAM Risk* (rejecting oversized models).
5. *Baseline Adequacy* (must anchor to audited Model A or Model E baselines).
6. *Clean Ablation Control* (rejecting confounded mutations).
7. *Unsupported Novelty Assertions* (enforcing candidate novelty labels).
8. *Excessive Complexity* (rejecting parameter bloat for negligible potential gain).
9. *Statistical Uncertainty & Falsifiability* (requiring clear falsification criteria).
10. *Deterministic Reproducibility* (checking seed and split hash).
- Returns structured verdicts: `ACCEPT`, `REVISE`, or `REJECT`.

### 2.6 Dual-GPU Resource Manager (`research_engine/resource_manager.py`)
Manages the local hardware:
- **GPU 0**: NVIDIA GeForce RTX 3060 12 GB
- **GPU 1**: NVIDIA GeForce RTX 3060 12 GB
- **Host RAM**: 32 GB | **CPU**: AMD Ryzen 7 7700
- Allocates VRAM dynamically based on budget (SMOKE: 1.5GB, SHORT: 3.0GB, STANDARD: 4.5GB, FULL: 6.0GB).
- Enforces an absolute safe ceiling of 10.0 GB per device (leaving 2 GB for system/CUDA context).
- Device locking: Precludes running two concurrent heavy training jobs on the same physical GPU.

### 2.7 Dataset and Split Gate (`dataset_gate.py`)
Ensures that models cannot accidentally train on corrupt data or leak test subjects:
- Validates the 47 subjects and 253 raw nights against PhysioNet `labels.mat`.
- Validates that the preprocessing cache (`prep_57738d534b`) contains the 246 usable nights.
- Enforces subject-level disjointness: $\text{Train} \cap \text{Val} = \emptyset$, $\text{Train} \cap \text{Test} = \emptyset$, $\text{Val} \cap \text{Test} = \emptyset$.
- **Hard Gate**: Raises `TestSetLeakageError` immediately if training, validation, or tuning touches held-out test subjects.

### 2.8 Standardized Evaluator (`evaluator.py`)
Provides uniform scoring across all experiments:
- **Primary Metric**: Macro-F1 (unweighted mean across Wake, Light, Deep, REM).
- **Secondary Metrics**: Accuracy, Cohen's $\kappa$, MCC, Expected Calibration Error (ECE, 10 bins), per-class F1/Recall/Precision, boundary error rate.
- **Diagnostic Metrics**: Raw and normalized confusion matrices, parameter count, peak VRAM, gate weighting statistics ($\alpha_{\text{Wake}}$ vs $\alpha_{\text{Sleep}}$).
- **Hard Gate**: Raises `TestSetProtectionViolation` if evaluation is attempted on the test partition without explicit frozen authorization.

### 2.9 Decision Engine (`research_engine/decision_logic.py`)
Multi-criteria evaluation replacing naive `if score > previous: keep`:
- **Kept**: Substantial Macro-F1 gain ($\ge +0.010$) under controlled capacity, OR meaningful model simplification ($\ge -15\%$ parameters) while maintaining parity ($\Delta \ge -0.003$).
- **Discarded**: Performance drop ($\Delta < -0.015$) without compensating simplicity, or rejected by reviewer.
- **Needs Review**: Marginal gain ($< +0.005$) achieved through significant parameter bloat ($> +30\%$).
- **Investigate**: Overall parity accompanied by marked selective class improvement (e.g. REM F1 gain $\ge +0.030$).

### 2.10 Experiment Engine & Institutional Memory (`research_engine/experiment_engine.py` & `research_memory.json`)
Coordinates the end-to-end experiment lifecycle:
$$\text{PROPOSE} \to \text{GOVERN} \to \text{DESIGN} \to \text{REVIEW} \to \text{IMPLEMENT} \to \text{RUN} \to \text{EVALUATE} \to \text{DECISION} \to \text{MEMORY}$$
- Records git commits, split fingerprints, random seeds, budgets, and hardware allocations.
- Persists all historical and new runs into `research_memory.json`.
