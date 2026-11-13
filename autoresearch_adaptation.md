# Autoresearch & AI-Scientist Adaptation Document

## 1. Executive Summary

This document details the adaptation of concepts from **Karpathy's Autoresearch** and **SakanaAI's AI-Scientist** into the existing **SLEEP Research Infrastructure**, along with the adversarial audit and hardening conducted in **Phase 2**.

Neither framework was blindly cloned or adopted in its raw form. Wearable physiological sleep staging involves domain-specific complexities—such as strong inter-subject physiological heterogeneity, severe class imbalance, multi-night circadian structures, and strict test-set isolation—that render generic LLM-loop frameworks vulnerable to data leakage, compute waste, and misleading point-estimate optimizations.

---

## 2. What Was Borrowed

### From SakanaAI / AI-Scientist:
1. **Structured Scientific Ideation**:
   - Replaced unguided prompt tinkering with a formal 10-point hypothesis schema: `hypothesis`, `motivation`, `expected_mechanism`, `experiment_design`, `expected_outcome`, `novelty_rationale`, `feasibility`, `risks`, `required_compute`, and `evaluation_criteria`.
2. **Literature-Grounded Evidence Categorization**:
   - Adopted explicit categorization of scientific claims into:
     - `VERIFIED_LITERATURE_EVIDENCE`
     - `PROJECT_OBSERVATION`
     - `LLM_INFERENCE`
     - `SPECULATIVE_HYPOTHESIS`
3. **Autonomous Hostile Peer Review**:
   - Implemented an automated pre-execution reviewer (`ScientificReviewer`) that evaluates proposals against 10 critical checks (data leakage, baseline adequacy, metric consistency, overfitting risk, and ablation control) with the authority to issue `REJECT` verdicts before compute is consumed.

### From Karpathy / Autoresearch:
1. **Deterministic Experiment Lifecycle**:
   - Adopted the structured sequential state machine:
     $$\text{PROPOSE} \to \text{GOVERN} \to \text{DESIGN} \to \text{REVIEW} \to \text{IMPLEMENT} \to \text{RUN} \to \text{EVALUATE} \to \text{DECISION} \to \text{MEMORY}$$
2. **Git-Based Experiment Isolation**:
   - Every experiment is tagged with its parent experiment ID, exact git commit hash, and isolated artifact directory (`research/experiments/EXP-XXX/`).
3. **Explicit Experiment Lifecycle Statuses & Decoupled Modes**:
   - Decoupled `execution_mode` (`REAL`, `SIMULATION`, `DRY_RUN`) from `lifecycle_status` (`PROPOSED`, `GOVERNED`, `DESIGNED`, `REVIEWED`, `APPROVED`, `IMPLEMENTING`, `RUNNING`, `EVALUATING`, `COMPLETED`, `FAILED`, `BLOCKED`, `DISCARDED`, `NEEDS_REVIEW`, `MEMORY`).
4. **Structured Experiment Registry**:
   - Maintained a single, inspectable institutional memory file (`research_memory.json`) and queue (`research_queue.yaml`) validated by an adversarial schema gate.

---

## 3. What Was Changed and Why

| Autoresearch / AI-Scientist Feature | Adapted SLEEP Implementation | Scientific / Engineering Rationale |
| :--- | :--- | :--- |
| **Fixed 5-Minute Training Budget** | **Configurable Experiment Budgets (`SMOKE`, `SHORT`, `STANDARD`, `FULL`)** | Sleep staging models cannot be trained meaningfully on a fixed 5-minute clock. A 10-epoch screening run takes ~2.5 minutes, while an 80-epoch convergence benchmark requires ~12–15 minutes. Epoch counts are physiologically grounded in circadian batch passes, not arbitrary wall-clock timeouts. |
| **Naive Keep/Discard (`if score > prev: keep`)** | **Multi-Criteria Decision Engine (`DecisionEngine`)** | Sleep staging is multi-objective. A model that gains +0.002 Macro-F1 by adding 5 million parameters is a "complexity trap". Conversely, an architecture that drops parameters by 20% while maintaining parity is scientifically valuable. The decision engine evaluates primary Macro-F1, secondary metrics, parameter delta, VRAM, and clinical relevance. |
| **Accuracy as Primary Metric** | **Macro-F1 as Primary Metric** | Sleep datasets exhibit severe class imbalance (Light sleep represents ~44% of epochs, while Wake represents ~11% and Deep ~19%). Maximizing raw accuracy rewards models that overpredict Light sleep and ignore minority Deep/Wake episodes. Macro-F1 gives equal weight to all 4 clinical stages. |
| **Unrestricted Level 5 Autonomy** | **Gated Autonomy (Default Level 3: Implement / Require Approval)** | Unrestricted LLM loops generate speculative, ungrounded architectural variants that consume massive GPU cycles without testing clear scientific questions. Defaulting to Level 3 ensures human oversight over compute expenditure while automating proposal, review, and validation logic. |
| **Single GPU Assumption** | **Dual-GPU Resource Manager (`GPUResourceManager`)** | The local workstation features two physical NVIDIA RTX 3060 12GB cards. The resource manager tracks VRAM allocations, device leases, and enforces single-job exclusivity per GPU with automated stale lease recovery to prevent CUDA out-of-memory crashes and deadlocks. |
| **Universal OFAT vs Multi-variable** | **Policy-Gated OFAT Mode (`CONTROLLED_OFAT_MODE`)** | OFAT is treated as the current controlled policy for DEC-021 mechanism attribution rather than an immutable law. Multi-factor protocols remain supported in theory if explicitly approved under a future documented protocol. |

---

## 4. What Was Intentionally NOT Adopted & Capability Disclosures

1. **Autonomous Web Scraping / Automated Paper Writing**:
   - *AI-Scientist* includes automated LaTeX manuscript generation and automated Semantic Scholar crawling.
   - *Why Rejected*: The SLEEP project requires rigorous empirical verification of wearable sensor physics, not automated paper mill outputs. Literature context is curated and verified locally (`research_engine/literature_agent.py`) to prevent LLM hallucinations from entering the scientific record.
   - **Capability Status**: `LITERATURE_RETRIEVAL_NOT_IMPLEMENTED`.
2. **Live Training CLI Execution without Human Gate**:
   - *Autoresearch* launches training commands automatically.
   - *Why Rejected*: To prevent unauthorized GPU compute expenditure, long-running convergence runs, and accidental test set access.
   - **Capability Status**: `LIVE_EXECUTION_INTERFACE_NOT_ENABLED` (Live training runner interface is deliberately gated pending human authorization).
3. **500-Epoch Long-Training Execution**:
   - *Why Rejected*: Under DEC-019/020 and user governance, 500-epoch training is explicitly paused. Long training past epoch 100 was proven in EXP-010/011 to cause severe post-peak overfitting. The Governor implements a numerical hard block (`BLOCKED_BY_RESEARCH_GOVERNANCE`) against any $\ge 500$ epoch request.
4. **Development-Time Test-Set Access**:
   - *Autoresearch* frequently evaluates candidates on a held-out benchmark during the inner search loop.
   - *Why Rejected*: In clinical and machine learning research, using test data for model selection or hyperparameter tuning causes optimistic selection bias and invalidates publication claims. Test-set evaluation is strictly blocked by hard gates (`TestSetProtectionViolation` and `TestSetLeakageError`).

---

## 5. Governance Risk: Direct Autonomous Git Push to Main

In the current setup, all git commits occur locally on branch `main`. While this facilitates synchronization, autonomous pushes directly to `main` pose a governance risk:
- An errant agent could commit unvetted code directly to the authoritative branch.
- Recommended future hardening:
  ```text
  development / experiment branch: research/EXP-XXX
          ↓
  automated unit & safety tests (67/67 pass)
          ↓
  autonomous scientific review (ReviewVerdict.ACCEPT)
          ↓
  human supervisor authorization
          ↓
  merge to main
  ```
