# Stage 5: Initial Zoom Out Review Report

* **Target Document:** [`docs/designs/exp019_adaptation.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/docs/designs/exp019_adaptation.md)
* **Standard Applied:** [Zoom Out Standard](file:///home/rj2826/.gemini/config/plugins/egm/resources/document_standards/zoom-out-standard.md) & TRIZ Contradiction Analysis
* **Reviewers:** Skeptical Architect (`egm-zoom-out`), Inventive Problem-Solving Goldfish (`egm-inventive-catalyst`)
* **Review Date:** 2026-11-16
* **Overall Assessment:** **CONDITIONAL PASS**

---

## 1. Executive Summary

The initial zoom-out review evaluated the high-altitude architecture, scientific alignment, complexity control, and system contradictions of the EXP-019 within-person sleep staging adaptation specification. The document demonstrates high discipline by freezing model architecture (Model A, 163,940 parameters), restricting primary claims to 4-class staging, decoupling Stage D developer selection from Stage C confirmatory evaluation, and establishing an empirical negative control ($FT\text{-}other\text{-}2$).

The review grants a **Conditional Pass**, identifying two actionable architectural improvements and two inventive risk mitigations to be incorporated before final implementation.

---

## 2. Key Review Findings by Persona

### A. Skeptical Architect (`egm-zoom-out`) Findings
* **Finding 1 (Major - Code Duplication in Step 1 Checkpoint Regeneration):**
  * *Observation:* The plan proposed writing `experiments/EXP-018/regenerate_modelA_checkpoints.py` from scratch. However, `experiments/EXP-018/run_exp018_execution.py` already contains the audited, tested Phase 2 training engine, compute-invariant scheduling logic ($B \times L \times N_{\text{accum}} = 2400$), random cropping dataset, and optimizer setup.
  * *Actionable Recommendation:* Mandate that the regeneration script imports and reuses `train_model` directly from `run_exp018_execution.py`, appending a checkpoint-saving hook to guarantee 100% protocol and weight parity with prior Phase 2 runs.
* **Finding 2 (Major - Filesystem Namespace Collision in `experiments/EXP-019/`):**
  * *Observation:* The workspace directory `experiments/EXP-019/` already exists on disk containing files from the historical 500-epoch exploratory benchmark (`run_exp019_500epochs.py`, `EXP019_REPORT.md`, `exp019_results/`). Directly adding confirmatory adaptation scripts into this directory risks mixing exploratory and confirmatory artifacts.
  * *Actionable Recommendation:* Formalize a Step 0 pre-execution cleanup in the implementation plan: archive existing exploratory files into `experiments/EXP-019-EXPLORATORY/` so that `experiments/EXP-019/` is dedicated cleanly to the pre-registered confirmatory H-A adaptation pipeline.
* **Finding 3 (Nit - Batch Sequence Feeding Specification):**
  * *Observation:* Fine-tuning on calibration nights $C=\{N_1, N_2\}$ should be explicitly documented as full-night padded sequences passed to `pack_padded_sequence` (as $N \le 1200$), rather than taking random sub-crops.

### B. Inventive Problem-Solving Goldfish (`egm-inventive-catalyst`) Findings
* **Finding 4 (Major - Mode Collapse Risk in Arm 5 $UA\text{-}2$):**
  * *Observation:* In Arm 5 ($UA\text{-}2$), unconstrained Shannon entropy minimization ($\mathcal{L}_{\text{ent}} = -\sum p \log p$) on imbalanced stages contains a degenerate attractor: collapsing all predictions to the majority Light-sleep class achieves zero entropy but destroys Macro-F1.
  * *Actionable Recommendation:* In `run_exp019_stage_d.py`, include an automated safeguard check verifying that $UA\text{-}2$ predictions do not collapse to a single class, and evaluate bounded step counts ($\le 20$ steps) or diversity-regularized entropy ($\mathcal{L}_{\text{ent}} - \mathcal{H}(\bar{p})$).
* **Finding 5 (Major - Missing Stages in 2-Night Calibration Sets):**
  * *Observation:* Subjects exhibiting disrupted sleep or first-night effects on $N_1$ or $N_2$ may have zero REM epochs in calibration, causing cross-entropy gradients to severely depress unobserved class logits.
  * *Actionable Recommendation:* Stage D should monitor per-stage recall across inner-validation subjects and prioritize parameter-selective adaptation (`head + last BiGRU layer`) as a primary hedge against catastrophic class logit erasure.
* **Finding 6 (Nit - State Neutralization via In-Memory Cloning):**
  * *Observation:* Re-reading binary `.pt` files from physical disk 690 times creates unnecessary I/O overhead.
  * *Actionable Recommendation:* Cache the outer-fold backbone in host RAM (`state_dict_ref`) and clone via `copy.deepcopy` inside the subject loop, accompanied by an explicit unit test verifying state invariance.
* **Finding 7 (Nit - Donor Night Validation for $FT\text{-}other\text{-}2$):**
  * *Observation:* Ensure donor night sampling asserts $\ge 300$ valid epochs to prevent pairing with an artifact-corrupted donor recording.

---

## 3. Recommended Remediation & Next Steps

All findings are compatible with the locked research direction and enhance experimental robustness. The findings will be incorporated into the design document during resolution.
