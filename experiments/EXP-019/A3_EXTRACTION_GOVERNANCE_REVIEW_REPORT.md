# Independent Governance and Scientific Review: Phase A3 Diagnostic Extraction Specification

**Target Document:** `experiments/EXP-019/design_phase_a3_extraction.md`  
**Review Skill:** `egm-review`  
**Date:** 2026-11-17  
**Review Status:** COMPLETE  
**Review Verdict:** **APPROVED WITH REQUIRED CHANGES (Strict Scope Reduction)**  

---

## Executive Summary

The proposed Phase A3 extraction specification ([`experiments/EXP-019/design_phase_a3_extraction.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-019/design_phase_a3_extraction.md)) conflates two fundamentally distinct activities:
1. **Legitimate Diagnostic Recovery ($k=2$):** Reconstructing unpersisted epoch predictions, confusion matrices, and hypnograms for the *already executed and locked* $k=2$ Stage C confirmatory experiment, strictly verified against locked scalar metrics.
2. **Prohibited New Experimentation ($k \in \{1, 3, 4\}$):** Executing 414 new neural network fine-tuning runs ($46\text{ subjects} \times 3\text{ seeds} \times 3\text{ scaling arms}$) using un-preregistered hyperparameter transfer across varying calibration depths.

Forensic audit of the repository proves that **$k \in \{1, 3, 4\}$ was never executed in EXP-019**. Authorizing fine-tuning for $k \ne 2$ under the guise of "diagnostic extraction" is a direct governance violation of the anti-premature training gate in [`RESEARCH_DIRECTION_v4_FULL.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/RESEARCH_DIRECTION_v4_FULL.md).

The specification must be restricted strictly to $k=2$ diagnostic recovery and locked scalar re-analysis. The learning curve for $k \in \{1, 3, 4\}$ must be officially classified as **UNRECOVERABLE FROM LOCKED ARTIFACTS — REQUIRES NEW PROTOCOL** and deferred to Phase D.

---

## Question 1: Are $k=1, 3, 4$ Already Executed?

A forensic trace of the repository code, git history, logs, and artifacts was conducted:
- `run_exp019_stage_c.py`: Lines 197–198 hardcode `cal_nights = nights[:2]` and `eval_nights = nights[2:]`. No parameter or loop for $k$ exists.
- `run_exp019_stage_d.py`: Line 11 hardcodes `cal_nights = nights[:2]`. Grid search evaluated only $k=2$.
- `exp019_stage_c_results.json`: `eval_nights` contains only nights beginning at index 3.
- Checkpoints on disk: Only zero-shot EXP-018 backbones exist (`experiments/EXP-018/checkpoints/modelA_L1200_fold{1..5}_seed{42,123,2024}.pt`). No adapted checkpoints exist for any $k$.

### Execution Status Table

| Calibration Depth | Execution Classification | Empirical Evidence / Rationale |
| :---: | :---: | :--- |
| **$k = 1$** | **C. Never executed** | Zero code, logs, or results exist for $k=1$. `run_exp019_stage_c.py` only sliced `nights[:2]`. |
| **$k = 2$** | **B. Executed, but intermediate outputs unpersisted** | Executed in Stage C ($n=46, N=152$). Macro-F1 and deltas preserved in `exp019_stage_c_results.json`, but epoch predictions and hypnograms were discarded in-memory. |
| **$k = 3$** | **C. Never executed** | Zero code, logs, or results exist for $k=3$. `run_exp019_stage_c.py` only sliced `nights[:2]`. |
| **$k = 4$** | **C. Never executed** | Zero code, logs, or results exist for $k=4$. `run_exp019_stage_c.py` only sliced `nights[:2]`. |

---

## Question 2: Was $k=1–4$ Preregistered as an Executable Experiment?

Inspection of [`PREREG_EXP-019_H-A.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/PREREG_EXP-019_H-A.md) and [`RESEARCH_DIRECTION_v4_FULL.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/RESEARCH_DIRECTION_v4_FULL.md) reveals:

1. **Preregistration Arm Definitions (Section 4):**
   The locked protocol defined exactly five arms: `ZS`, `FT-2`, `FT-other-2`, `PS-2`, and `UA-2`. All four adaptation arms were explicitly and exclusively locked at $k=2$ calibration nights (`FT-2`). There were no arms defined for `FT-1`, `FT-3`, or `FT-4`.
2. **Hyperparameter Selection (Section 5):**
   Stage D evaluated candidate grids strictly on $k=2$ calibration nights to select optimal learning rates, epochs, and trainable scopes. Stage D never evaluated or selected hyperparameters for $k \in \{1, 3, 4\}$.
3. **Descriptive Intent vs. Experimental Protocol:**
   Section 2 lists "learning curve over k = 1, 2, 3, 4" under descriptive analyses without confirmatory claims. However, this was an unexecuted aspirational output. The preregistration did not establish an execution protocol, hyperparameter mapping, or script implementation for $k \ne 2$.

**Conclusion:** The $k=1–4$ learning curve was a preregistered descriptive output requirement, but it was **never an executed experimental condition**. It cannot be treated as existing data.

---

## Question 3: Deliverable Classification Table

| Proposed Deliverable | Classification Category | Feasibility under Phase A | Technical Mechanism & Governance Rule |
| :--- | :--- | :---: | :--- |
| **$k=2$ Confirmatory Anchor** | **1. Pure extraction from locked results** | **ALLOWED** | Extracted directly from `exp019_stage_c_results.json` and `exp019_subject_metrics.csv` ($\bar{\Delta} = +1.01\text{ pp}, n=46$). Zero new computation. |
| **$k \in \{1, 3, 4\}$ Scaling Curve** | **5. New experiment requiring new protocol** | **PROHIBITED** | Requires 414 new neural network fine-tuning runs across test subjects with un-preregistered hyperparameter transfer. Violates anti-premature training gate. |
| **Four-Class Confusion Matrices ($k=2$)** | **3/4. Deterministic reconstruction pass** | **CONDITIONALLY ALLOWED** | In-memory fine-tuning for $k=2$ only, strictly matching locked Stage C code, verified by bitwise matching to locked macro-F1 ($\le 0.0001$). |
| **Per-Stage F1 Metrics ($k=2$)** | **3/4. Deterministic reconstruction pass** | **CONDITIONALLY ALLOWED** | Derived from reconstructed $k=2$ epoch predictions. Includes FDR multiplicity control ($q = 0.05$). |
| **Sleep-Summary Agreement ($k=2$)** | **3/4. Deterministic reconstruction pass** | **CONDITIONALLY ALLOWED** | Derived from reconstructed $k=2$ whole-night hypnograms using `ClinicalHypnogramEvaluator`. Evaluates TST, SOL, WASO, SE, REM%, Deep%. |
| **Night-Index Breakdown ($k=2$)** | **3/4. Deterministic reconstruction pass** | **CONDITIONALLY ALLOWED** | Derived by evaluating reconstructed predictions night-by-night across outer evaluation nights (indices 3–7). Strictly descriptive. |
| **Bootstrap & Wilcoxon Tests ($k=2$)** | **1. Pure extraction / re-analysis** | **ALLOWED** | Re-computed directly on preserved subject metrics in `exp019_subject_metrics.csv`. |

*Definitions:*
- **Category 1 (Pure Extraction):** Arithmetic or statistical calculation directly on existing CSV/JSON tables without executing PyTorch models.
- **Category 2 (Deterministic Re-analysis):** Processing saved epoch/prediction tensors (not available, as predictions were not saved).
- **Category 3/4 (Deterministic Reconstruction Pass):** Running the exact frozen Stage C code with fixed seeds to recover discarded intermediate epoch predictions, gated by strict regression assertions against locked scalar values.
- **Category 5 (New Experiment):** Running model training or adaptation on previously unexecuted conditions.

---

## Question 4: Governance Violations in `design_phase_a3_extraction.md`

The draft specification contains three major governance violations:

1. **Section 1, Section 2, and Section 5 (Stage 2): Authorization of New Training**
   - *Draft Text:* Authorizes "scaling adaptation passes for $k \in \{1, 3, 4\}$ using locked fold hyperparameters from Stage D."
   - *Violation:* Stage D hyperparameters were tuned exclusively for $k=2$. Applying them to $k \in \{1, 3, 4\}$ is un-preregistered ad-hoc training. Running 414 new fine-tuning runs violates the anti-premature training gate of `RESEARCH_DIRECTION_v4_FULL.md`.
2. **Section 6: Formalizing $k \in \{1, 3, 4\}$ Execution as "Extraction"**
   - *Draft Text:* Outlines fine-tuning on $n_1$ ($k=1$), $[n_1, n_2, n_3]$ ($k=3$), and $[n_1, \dots, n_4]$ ($k=4$).
   - *Violation:* Calling neural network gradient descent optimization an "extraction" misrepresents new model training as retrospective data mining.
3. **Section 14.1: Requirement to Populate `exp019_k_learning_curve.csv` for $k=1, 3, 4$**
   - *Draft Text:* Mandates populating numerical results for $k=1, 3, 4$.
   - *Violation:* Forces the execution of unexecuted models to satisfy an artifact schema.

---

## Required Modifications to `design_phase_a3_extraction.md`

To achieve governance compliance, the specification must be amended as follows:

1. **Section 1 (Objective):** Remove references to computing new scaling passes for $k \in \{1, 3, 4\}$. State that A3 extracts $k=2$ diagnostics and formally records the unexecuted status of $k \in \{1, 3, 4\}$.
2. **Section 2 (Scope):** Explicitly prohibit fine-tuning for $k \ne 2$. Add $k \in \{1, 3, 4\}$ to Out-of-Scope.
3. **Section 4 & Section 6 ($k=1–4$ Methodology):**
   - Retain $k=2$ as the sole executed confirmatory benchmark ($n=46, N=152$).
   - Document $k \in \{1, 3, 4\}$ as **UNRECOVERABLE FROM LOCKED ARTIFACTS — REQUIRES NEW PROTOCOL**.
   - Provide the theoretical subject census ($n=46, 42, 33$) as a descriptive cohort structure, but forbid model execution.
4. **Section 5 (Extraction Procedure):** Scope the deterministic reconstruction pass strictly to $k=2$ for arms `ZS`, `FT-2`, `FT-other-2`, and `PS-2`.
5. **Section 14.1 (`exp019_k_learning_curve.csv`):**
   - Populate numerical metrics exclusively for row $k=2$.
   - For rows $k \in \{1, 3, 4\}$, populate census fields (`n_subjects`, `n_eval_nights`) and mark performance metrics as `UNEXECUTED_IN_EXP019`.
6. **Section 18 (Prohibitions):** Add explicit prohibition: *"DO NOT execute adaptation or fine-tuning for $k \in \{1, 3, 4\}$."*

---

## Final Review Decision and Next Safe Action

### 1. Governance Verdict
**APPROVED WITH REQUIRED CHANGES (Strict Scope Reduction)**

### 2. Exact Next Safe Action
Apply the required scope reductions to [`experiments/EXP-019/design_phase_a3_extraction.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-019/design_phase_a3_extraction.md) using `egm-resolve`:
1. Restrict reconstruction strictly to $k=2$ ($ZS$, $FT\text{-}2$, $FT\text{-}other\text{-}2$, $PS\text{-}2$).
2. Classify $k \in \{1, 3, 4\}$ as unexecuted and unrecoverable from locked artifacts.
3. Keep the anti-premature training gate intact.
4. Await user confirmation before authoring the extraction script.
