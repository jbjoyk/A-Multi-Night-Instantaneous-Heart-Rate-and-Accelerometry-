# Independent Adversarial Methodology Review: Phase D Experimental Decision Plan
## Critical Audit of `PHASE_D_EXPERIMENTAL_DECISION_PLAN.md` Prior to Phase D Authorization

**Audit Date:** 2026-11-19  
**Auditor Persona:** Independent Adversarial Research Reviewer & Q1 Methodology Auditor (EGM Protocol)  
**Target Document Reviewed:** [`PHASE_D_EXPERIMENTAL_DECISION_PLAN.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/PHASE_D_EXPERIMENTAL_DECISION_PLAN.md)  
**Governing Baselines & Authorities:**
- Authoritative Research Direction: [`RESEARCH_DIRECTION_v4_FULL.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/RESEARCH_DIRECTION_v4_FULL.md) (DEC-029)
- Goal Attainment Audit: [`RESEARCH_GOAL_ATTAINMENT_AUDIT.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/RESEARCH_GOAL_ATTAINMENT_AUDIT.md)
- Governance Decisions: [`DECISIONS.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/DECISIONS.md) (DEC-029, DEC-030, DEC-031, DEC-032)
- Preregistration & Extraction Records: `PREREG_EXP-019_H-A.md` (Commit `4dff5b7`), `experiments/EXP-019/A3_EXTRACTION_GOVERNANCE_REVIEW_REPORT.md`

---

## 1. Executive Summary & Verdict

### **AUDIT VERDICT: PASS WITH MANDATORY GOVERNANCE & METHODOLOGICAL CORRECTIONS**

The proposed `PHASE_D_EXPERIMENTAL_DECISION_PLAN.md` correctly resolves the historical discrepancy regarding the $k=1, 2, 3, 4$ learning curve (confirming that $k \in \{1, 3, 4\}$ were never historically trained in EXP-019) and properly enforces the immutability of all locked EXP-019 artifacts.

However, an adversarial audit reveals **three critical methodological flaws** and **one strategic misalignment** that must be corrected before any experimental code or training runs can be authorized:

1. **Governance Misclassification of TSK-06:** The decision plan classified TSK-06 ($k=1, 3, 4$) as "covered by an existing preregistration." This is a severe governance error. Because $k \in \{1, 3, 4\}$ requires 121 new fine-tuning runs, hyperparameter inheritance rules, and stopping criteria not specified in `PREREG_EXP-019_H-A.md`, it represents **new model training** that requires a dedicated preregistration addendum before execution.
2. **Inverted Priority Hierarchy (TSK-08 vs. TSK-06):** The decision plan ranked baseline comparison (TSK-08) above the primary contribution C1 (TSK-06). In `RESEARCH_DIRECTION_v4_FULL.md`, Contribution C1 is the primary research contribution; completing the personalization learning curve must take precedence over secondary baseline benchmarking.
3. **Severe Statistical Fallacy in TSK-08 Decision Criteria:** The decision plan proposed that "if the 95% CI spans 0, Model A proves parity/compression." This conflates absence of evidence with evidence of absence. To claim non-inferiority or parity against SLAMSS-IFS, a formal **non-inferiority margin** ($\delta_{\text{NI}}$) and one-sided testing rule must be pre-registered.
4. **Superficial Mask Fix in TSK-07:** The proposed fix for the unsupervised quality mask risked substituting an arbitrary statistical z-score cutoff for true physiological quality control. Quality screening must be evaluated on unstandardized raw units before normalization.

---

## 2. Itemized Adversarial Audit Findings

### 2.1 Audit Item 1: Governance Misclassification of TSK-06 ($k=1, 3, 4$)
- **Decision Plan Position:** Classified TSK-06 as *Category 2: Missing but Covered by Existing Preregistration*.
- **Adversarial Critique:**
  * In `PREREG_EXP-019_H-A.md` (locked at commit `4dff5b7`), Question K was listed merely as an unelaborated descriptive objective ("Gain vs number of calibration nights $k=1 \text{ to } 4$").
  * The locked preregistration text contained no formal decision rules, no stopping criteria, and no hyperparameter tuning specification for $k \ne 2$. Stage D hyperparameter tuning was conducted strictly for 2-night calibration ($k=2$).
  * Running gradient descent across 121 subject-conditions ($46 + 42 + 33$) is unequivocally **new model training**, as explicitly ratified in `A3_EXTRACTION_GOVERNANCE_REVIEW_REPORT.md`.
  * In computational research governance, new model training cannot be grandfathered under an old preregistration that lacked hyperparameter specifications for those conditions.
- **Mandatory Correction:** Reclassify TSK-06 to **Category 3: Requires Dedicated Preregistration Addendum**. Before executing $k \in \{1, 3, 4\}$, a formal addendum (`PREREG_EXP-019_K_CURVE.md`) must be authored and committed, specifying: (i) whether hyperparameters are inherited directly from $k=2$ or swept on inner validation, (ii) evaluable cohort definitions, and (iii) statistical decision thresholds.

---

### 2.2 Audit Item 2: Strategic Priority Alignment (TSK-06 vs. TSK-08)
- **Decision Plan Position:** Ranked TSK-08 (SLAMSS-IFS matched re-run) as Rank 1, demoting TSK-06 ($k$-learning curve) to Rank 2.
- **Adversarial Critique:**
  * In `RESEARCH_DIRECTION_v4_FULL.md` Section 4 (line 52), **Contribution C1 is explicitly designated as the PRIMARY contribution** of the entire study ("*C1 (primary): a pre-registered, leakage-free replication and extension of multi-night supervised personalization...*").
  * The working title of the manuscript is: "*Does a few labeled nights of personal data help wearable sleep staging?*"
  * If the project submits without $k=3$ and $k=4$, it leaves its primary scientific question unanswered and fails to test the central finding of van der Aar et al. (2025), which claimed that wearable personalization only reaches statistical significance at $k \ge 3$ nights.
  * By contrast, SLAMSS-IFS is a secondary baseline comparison addressing Gap G2. While important, a baseline comparison cannot supersede the paper's primary research question.
  * Furthermore, TSK-06 operates on our existing, audited pipeline and frozen Model A weights (~1.5 GPU hours), whereas TSK-08 involves running third-party code with higher implementation friction (~7.5–15 GPU hours).
- **Mandatory Correction:** **Invert the priority ranking**. TSK-06 ($k$-learning curve) must be designated **Priority 1**, and TSK-08 (SLAMSS-IFS matched baseline) designated **Priority 2**.

---

### 2.3 Audit Item 3: Biostatistical Rigor of TSK-08 (Non-Inferiority vs. Equivalence Fallacy)
- **Decision Plan Position:** Section 4.1 stated:
  > *"Parity / Compression Supported: Out-of-fold Macro-F1 difference is within $\pm 1.0\text{ pp}$ (CI spans 0). $\implies$ Manuscript proves that Model A achieves 33.3-fold parameter compression with zero statistically significant performance loss."*
- **Adversarial Critique:**
  * This is the classic biostatistical fallacy: **absence of evidence is not evidence of absence** (Altman & Bland, 1995).
  * If the 95% bootstrap confidence interval of $\Delta = \text{Macro-F1}(\text{Model A}) - \text{Macro-F1}(\text{SLAMSS-IFS})$ is $[-2.40, +0.40]\text{ pp}$, the interval spans 0 ($p > 0.05$). However, this interval is entirely consistent with Model A being $2.4\text{ pp}$ worse than SLAMSS-IFS! Claiming this as "proof of parity" will be rejected by any Q1 statistical reviewer.
  * To scientifically claim that a 163.9k parameter model matches a 5.47M parameter model without performance loss, one must formulate a formal **non-inferiority trial**.
- **Mandatory Correction:**
  * Pre-specify a formal **non-inferiority margin** $\delta_{\text{NI}}$ a priori. A defensible margin in wearable sleep staging is **$\delta_{\text{NI}} = -1.50\text{ pp}$ Macro-F1** (matching the effect size threshold in H-A).
  * Formal Decision Rule:
    1. **Non-Inferiority Supported (Parity Proven):** The lower bound of the two-sided 95% bootstrap confidence interval strictly exceeds $\delta_{\text{NI}}$:
       $$\text{CI}_{\text{lower}}(\text{Model A} - \text{SLAMSS-IFS}) > -1.50\text{ pp}$$
    2. **Inferiority Demonstrated:** The upper bound is strictly below $0.0\text{ pp}$, or the lower bound is below $\delta_{\text{NI}}$ with statistical significance.
    3. **Inconclusive:** The confidence interval spans $\delta_{\text{NI}} = -1.50\text{ pp}$ (e.g. $[-2.20, +0.30]\text{ pp}$).

---

### 2.4 Audit Item 4: Physiological Quality Control Integrity in TSK-07 (UA-2 Re-Run)
- **Decision Plan Position:** Proposed fixing `compute_wearable_quality_mask` by either:
  1. Evaluating $30 \le \text{BPM} \le 180$ on unstandardized heart rate; OR
  2. Evaluating the mask on z-scored IHR using cutoffs $[-4.0, +4.0]$.
- **Adversarial Critique:**
  * The second option (applying $[-4.0, +4.0]$ to z-scores) is unacceptable. Statistical z-score truncation is not physiological quality control; it merely clips distributional tails and ignores raw sensor dropouts.
  * Furthermore, completely removing the quality mask would allow sensor dropouts, detached watch periods, and imputation artifacts to contaminate unsupervised entropy loss, causing catastrophic divergence.
- **Mandatory Correction:** Formally mandate that **physiological quality control must be evaluated exclusively on raw physical units prior to feature standardization**:
  * Heart rate: $30 \le \text{IHR}_{\text{raw}} \le 180\text{ BPM}$.
  * Accelerometry: $0.5\text{ g} \le m_{\text{raw}} \le 3.0\text{ g}$.
  * The resulting binary validity mask must be serialized alongside normalized features and applied during adaptation.

---

### 2.5 Audit Item 5: Computational Estimates & Experimental Protocol Justification
- **TSK-06 ($k$-curve):** 121 adaptation runs ($46 + 42 + 33$ subjects $\times$ 15 epochs) estimated at ~1.5 GPU hours on an A100. **Adequately Justified.** (Each run processes ~800 to ~3,200 epochs, taking ~35–45 seconds per subject).
- **TSK-07 (UA-2):** 46 subjects $\times$ 5 epochs estimated at ~15 minutes GPU time. **Adequately Justified.**
- **TSK-08 (SLAMSS-IFS):** 15 runs (5 folds $\times$ 3 seeds) estimated at ~7.5 GPU hours.
  * *Audit Warning:* SLAMSS-IFS has 5,465,768 parameters, multi-scale convolutions, bidirectional LSTMs, attention modules, and class-weighted RCCLoss processing full-night sequences ($L = 1200$). If run on a single mid-range GPU (e.g. RTX 3080/4080 or T4), runtime will exceed 1.5 hours per fold/seed, requiring **15 to 22 GPU hours**.
  * The protocol must explicitly specify hardware requirements (minimum 24 GB VRAM GPU, e.g. A100 or RTX 3090/4090) and batch sizing.

---

### 2.6 Audit Item 6: Compliance with Governance Decisions (DEC-029, DEC-030, DEC-031, DEC-032)
- **DEC-029 Compliance:** All locked EXP-019 confirmatory files remain bitwise immutable. No unapproved training occurred during plan creation. **PASS.**
- **DEC-030 Compliance:** TSK-07 explicitly addresses the S3 implementation no-op without attempting to misrepresent historical data. **PASS.**
- **DEC-031 Compliance:** Model A (163,940 parameters) is maintained as the baseline reference size. No NAS or arbitrary architecture expansion is included. TSK-08 tests whether larger capacity (5.47M) is scientifically justified. **PASS.**
- **DEC-032 Compliance:** The plan preserves the distinction between multi-night average agreement and single-night bounds. **PASS.**

---

## 3. Corrected Experimental Roadmap & Phase D Execution Sequence

To align strictly with `RESEARCH_DIRECTION_v4_FULL.md` and satisfy Q1 methodology standards, the Phase D execution sequence is reorganized into three sequential stages:

```
================================================================================
CORRECTED PHASE D EXPERIMENTAL EXECUTION SEQUENCE
================================================================================

STAGE D1: PRIMARY CONTRIBUTION C1 CLOSURE (LOW COMPUTE, HIGH SCIENTIFIC LEVERAGE)
  Step 1.1: TSK-06 — Calibration-Depth Learning Curve (k = 1, 3, 4)
            * Protocol: PREREG_EXP-019_ADDENDUM.md
            * Evaluates k-trajectory against van der Aar 2025; completes C1.
            * Compute: ~1.5 GPU hours.
  Step 1.2: TSK-07 — Re-Run of Unsupervised Adaptation (UA-2) with Corrected Mask
            * Protocol: PREREG_EXP-019_ADDENDUM.md
            * Replaces S3 execution no-op with true label-free empirical evidence.
            * Compute: ~15 minutes GPU time.

STAGE D2: MATCHED BASELINE BENCHMARKING (CRITICAL FOR BASELINE RIGOR)
  Step 2.1: TSK-08 — Matched 5-Fold Cross-Validation Re-Run of SLAMSS-IFS
            * Protocol: PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md
            * Tests non-inferiority of Model A against 5.47M parameter baseline (delta_NI = -1.50 pp).
            * Compute: ~7.5–15 GPU hours.

STAGE D3: DIAGNOSTIC & EXTERNAL GENERALIZATION (OPTIONAL / CONDITIONAL GATES)
  Step 3.1: TSK-09 — EXP-019B Subject-Level Heterogeneity Hypothesis Testing
            * Pure Python statistical testing on existing extracted features; zero GPU time.
  Step 3.2: TSK-11 — EXP-027 External Zero-Shot Transfer to SleepAccel (31 PSG subjects)
            * Tests cross-sensor generalizability; zero training (inference only).
================================================================================
```

---

## 4. Exact Approvals and Preregistration Requirements Prior to Execution

Before a single Python process or GPU training run is launched:

1. **User Approval of Corrected Sequence:** The user must explicitly approve the re-sequenced Phase D roadmap (Stage D1 $\to$ Stage D2 $\to$ Stage D3).
2. **Approval & Commit of `PREREG_EXP-019_ADDENDUM.md`:** A formal preregistration addendum covering Stage D1 must be committed to the repository, specifying:
   - Evaluables cohorts for $k \in \{1, 3, 4\}$ ($N = 46, 42, 33$ subjects; $198, 106, 64$ evaluation nights).
   - Fixed hyperparameter inheritance from $k=2$ (15 epochs, AdamW, $\eta = 10^{-4}$, weight decay $10^{-2}$, batch size 4).
   - Raw-unit quality mask logic for UA-2 ($30 \le \text{IHR}_{\text{raw}} \le 180\text{ BPM}$).
   - Decision criteria and bootstrap confidence interval definitions.
3. **Approval & Commit of `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md`:** A dedicated preregistration covering Stage D2 must be committed, specifying:
   - Non-inferiority margin: $\delta_{\text{NI}} = -1.50\text{ pp}$ Macro-F1.
   - Fold manifests, random seeds (42, 123, 2024), full-night context ($L = 1200$).
   - Official code repository commit hash and parameter count verification.

---

*(Audit complete. No code executed; no training runs launched; locked files preserved.)*
