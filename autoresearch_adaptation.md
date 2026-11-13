# Autoresearch & AI-Scientist Adaptation Document

## 1. Executive Summary

This document details the adaptation of concepts from **Karpathy's Autoresearch** and **SakanaAI's AI-Scientist** into the existing **SLEEP Research Infrastructure**. 

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
3. **Explicit Experiment Lifecycle Statuses**:
   - Adopted formal status tracking: `PROPOSED`, `APPROVED`, `RUNNING`, `COMPLETED`, `FAILED`, `DISCARDED`, `KEPT`, `NEEDS_REVIEW`, and `BLOCKED`.
4. **Structured Experiment Registry**:
   - Maintained a single, inspectable institutional memory file (`research_memory.json`) and queue (`research_queue.yaml`) rather than complex external databases.

---

## 3. What Was Changed and Why

| Autoresearch / AI-Scientist Feature | Adapted SLEEP Implementation | Scientific / Engineering Rationale |
| :--- | :--- | :--- |
| **Fixed 5-Minute Training Budget** | **Configurable Experiment Budgets (`SMOKE`, `SHORT`, `STANDARD`, `FULL`)** | Sleep staging models cannot be trained meaningfully on a fixed 5-minute clock. A 10-epoch screening run takes ~2.5 minutes, while an 80-epoch convergence benchmark requires ~12–15 minutes. Epoch counts are physiologically grounded in circadian batch passes, not arbitrary wall-clock timeouts. |
| **Naive Keep/Discard (`if score > prev: keep`)** | **Multi-Criteria Decision Engine (`DecisionEngine`)** | Sleep staging is multi-objective. A model that gains +0.002 Macro-F1 by adding 5 million parameters is a "complexity trap". Conversely, an architecture that drops parameters by 20% while maintaining parity is scientifically valuable. The decision engine evaluates primary Macro-F1, secondary metrics, parameter delta, VRAM, and clinical relevance. |
| **Accuracy as Primary Metric** | **Macro-F1 as Primary Metric** | Sleep datasets exhibit severe class imbalance (Light sleep represents ~44% of epochs, while Wake represents ~11% and Deep ~19%). Maximizing raw accuracy rewards models that overpredict Light sleep and ignore minority Deep/Wake episodes. Macro-F1 gives equal weight to all 4 clinical stages. |
| **Unrestricted Level 5 Autonomy** | **Gated Autonomy (Default Level 3: Implement / Require Approval)** | Unrestricted LLM loops generate speculative, ungrounded architectural variants that consume massive GPU cycles without testing clear scientific questions. Defaulting to Level 3 ensures human oversight over compute expenditure while automating proposal, review, and validation logic. |
| **Single GPU Assumption** | **Dual-GPU Resource Manager (`GPUResourceManager`)** | The local workstation features two physical NVIDIA RTX 3060 12GB cards. The resource manager tracks VRAM allocations, device leases, and enforces single-job exclusivity per GPU to prevent CUDA out-of-memory crashes. |

---

## 4. What Was Intentionally NOT Adopted

1. **Autonomous Web Scraping / Automated Paper Writing**:
   - *AI-Scientist* includes automated LaTeX manuscript generation and automated Semantic Scholar crawling.
   - *Why Rejected*: The SLEEP project requires rigorous empirical verification of wearable sensor physics, not automated paper mill outputs. Literature context is curated and verified locally (`research_engine/literature_agent.py`) to prevent LLM hallucinations from entering the scientific record.
2. **5-Minute Training Window**:
   - *Autoresearch* strictly enforces 5 minutes of training per iteration.
   - *Why Rejected*: Prematurely truncated training obscures real convergence dynamics. In EXP-010/011, models exhibited peak validation Macro-F1 specifically at Epoch 78–80. Forcing a 5-minute cutoff would evaluate models during early under-fitting, yielding invalid conclusions regarding architecture quality.
3. **500-Epoch Long-Training Execution**:
   - *Why Rejected*: Under DEC-019/020 and user governance, 500-epoch training is explicitly paused. Long training past epoch 100 was proven in EXP-010/011 to cause severe post-peak overfitting (validation loss rising from 0.72 to 1.29). The Governor implements a hard block (`BLOCKED_BY_RESEARCH_GOVERNANCE`) against any 500-epoch request.
4. **Development-Time Test-Set Access**:
   - *Autoresearch* frequently evaluates candidates on a held-out benchmark during the inner search loop.
   - *Why Rejected*: In clinical and machine learning research, using test data for model selection or hyperparameter tuning causes optimistic selection bias and invalidates publication claims. Test-set evaluation is strictly blocked by hard gates (`TestSetProtectionViolation` and `TestSetLeakageError`).
5. **Kitchen-Sink Architecture Mutation**:
   - *Why Rejected*: Generic LLM coding agents tend to mutate loss weights, learning rates, convolutional depths, and attention heads simultaneously. The Experiment Designer enforces strict One-Factor-At-A-Time (OFAT) ablations, rejecting confounded multi-variable proposals.
