# Comprehension Test Report (Stage 6)

**Target:** `experiments/EXP-019/HETEROGENEITY_AUDIT_REPORT.md` (Lines 1–51)  
**Reviewer:** `egm-comprehension` (Naive Reader)  
**Evaluation Mode:** Isolated Document Audit (Zero Assumed Institutional Knowledge)

---

## What this proposes
The document proposes an exploratory diagnostic framework (EXP-019) evaluating nine candidate pre-adaptation predictors to explain performance variance across 46 subjects undergoing subject-specific model adaptation. It reports that none of the tested pre-adaptation predictors (divergence from population prior, sleep stage scarcity, heart rate volatility, loss headroom, or parameter displacement) correlate significantly with adaptation gain after multiple-testing correction ($q < 0.05$). Retrospectively, however, it finds that divergence between calibration nights and evaluation nights strongly degrades adaptation performance. Based on this, the document proposes an exploratory, zero-leakage pre-adaptation deployment safety gate requiring minimum stage prevalence ($\ge 3.0\%$) and bounded population divergence ($D_{JS} < 0.35$), demonstrating that it successfully filters out catastrophic non-responders while boosting mean adaptation gain across accepted subjects.

## How the existing system works, per this document
Per this document, the existing system evaluates sleep staging adaptation across 46 subjects and 152 outer evaluation nights partitioned into a 5-fold cross-validation scheme. Subjects begin with a zero-shot (ZS) baseline model and undergo subject-specific fine-tuning (FT) using calibration night data, producing performance gains in 73.9% of subjects and performance degradation in 26.1%. Model fine-tuning operates under varying architectural update scopes, where some evaluation folds adapt all model parameters while at least one fold adapts only the recurrent classification head. Adaptation is currently applied uniformly to all incoming subjects without pre-adaptation gating or screening, leaving the system vulnerable to severe negative performance drops when calibration stage distributions are unrepresentative.

---

## Detailed Findings

### 1. Undefined Dependent Metric & Delta Scale — Severity: 🟡 Major
- **Affected Scope**: Section 1 (Table Header & Rows, Lines 9–19), Section 2.A (Line 27), Section 3 (Lines 47–49); Terms: `Delta`, `pp`
- **Discrepancy Description & Rationale**: The primary dependent variable `Delta` is used throughout the audit report without being mathematically or operationally defined. While units of `pp` (percentage points) appear in lines 27 and 47–49, the underlying performance metric is never identified (Macro-F1 score on 4-stage sleep classification: Wake, Light, Deep, REM). A fresh reader cannot determine what metric is being adapted, how `Delta` is calculated ($\text{Metric}_{\text{FT}} - \text{Metric}_{\text{ZS}}$), or what magnitude of change represents clinically or practically meaningful performance.

### 2. Undefined Distribution Priors & Divergence Bounds — Severity: 🟡 Major
- **Affected Scope**: Section 1 (Table, Lines 11–19), Section 2.A (Lines 24–29), Section 3 (Line 45); Terms: $D_{\text{JS}}$, $D_{\text{JS\_pop}}$, $P_C$, $P_E$, $P_{\text{pop}}$
- **Discrepancy Description & Rationale**: The probability distributions and divergence metrics are introduced without mathematical definitions or domain groundings. The text references $D_{JS}(P_C \parallel P_{\text{pop}})$ and $D_{JS}(P_C \parallel P_E)$ without explaining:
  1. What entity the probability distributions $P$ represent (discrete 4-class categorical distributions over sleep stages: Wake, Light, Deep, REM).
  2. How $P_C$ (calibration prior), $P_E$ (evaluation prior), and $P_{\text{pop}}$ (population prior) are estimated (empirical stage frequencies across 30-second epochs in calibration nights vs. training folds).
  3. The base of the logarithm or bounds used for Jensen-Shannon divergence ($D_{JS} \in [0, 1]$ base 2).

### 3. Undefined Project Acronym: H-A — Severity: 🟡 Major
- **Affected Scope**: Header (Line 3); Term: `H-A`
- **Discrepancy Description & Rationale**: Line 3 states: *"Zero Modification to Confirmatory H-A Results"*. `H-A` is an undefined project-specific acronym. A new reader cannot discern whether `H-A` refers to "Hypothesis A", "Phase A", "Heart-Rate Adaptation", or a primary benchmark protocol. It should be explicitly defined as Confirmatory Hypothesis H-A.

### 4. Unstated Responder Threshold — Severity: 🟡 Major
- **Affected Scope**: Header (Line 5); Terms: `Responders`, `Non-Responders`
- **Discrepancy Description & Rationale**: The report partitions 46 subjects into 34 "Responders" (73.9%) and 12 "Non-Responders" (26.1%) without stating the classification threshold. It is unclear whether classification as a responder strictly requires $\text{Delta} > 0.0\text{ pp}$, or if a non-zero margin/statistical significance threshold is required.

### 5. Incompletely Defined Oldham Predictor & Baselines — Severity: 🟡 Major
- **Affected Scope**: Section 1 (Line 15), Section 2.B (Lines 32–35); Terms: `oldham_mean`, `Oldham-Corrected Headroom`, `ZS`, `FT`, `RTM`
- **Discrepancy Description & Rationale**: Section 2.B introduces formulas involving `ZS` and `FT` without defining them as Zero-Shot and Fine-Tuning baselines. Furthermore, the report references "Oldham's method" and `oldham_mean` as an adjustment for regression to the mean ($\text{Cov}(ZS, FT - ZS) = -\text{Var}(\epsilon) < 0$), but fails to provide the explicit mathematical formulation of the Oldham predictor variable ($\frac{\text{ZS} + \text{FT}}{2}$).

### 6. Architectural Context for Adaptation Scope Discrepancy — Severity: 🟡 Major
- **Affected Scope**: Section 1 (Line 18), Section 2.C (Lines 37–40); Terms: `scope=all`, `scope=head_gru`, `param_displacement_norm`
- **Discrepancy Description & Rationale**: The document introduces parameter displacement analysis across model adaptation scopes without providing neural network architectural context (Model A: ConventionalCNNBiGRU). Furthermore, line 40 notes that Fold 4 adapted only `head_gru` while Folds 1, 2, 3, and 5 adapted all layers, but fails to clarify that this difference was an outcome of nested inner-validation hyperparameter selection (Stage D). Additionally, `param_displacement_norm` does not specify which norm (relative Frobenius $L_2$ displacement) was computed.

### 7. Missing Units and Definitions for Calibration Predictors — Severity: 🟡 Major
- **Affected Scope**: Section 1 (Table, Lines 12–19); Terms: `deep_frac_cal`, `min_stage_frac_cal`, `hr_volatility_cal`, `loss_zs_cal`, `transition_rate_cal`, `n_valid_tokens_cal`
- **Discrepancy Description & Rationale**: Several primary predictor variables in Table 1 lack inline definitions or units:
  - `loss_zs_cal`: Which loss function is evaluated (zero-shot cross-entropy on calibration nights)?
  - `hr_volatility_cal`: How is volatility computed (standard deviation of valid 1 Hz instantaneous heart rate across calibration nights)?
  - `transition_rate_cal`: How is stage transition rate normalized (state changes per 30-second epoch)?
  - `n_valid_tokens_cal`: What constitutes a "token" (number of valid 30-second epochs in calibration nights)?

### 8. Dataset Subject Identifier Context — Severity: 🟢 Nit
- **Affected Scope**: Section 2.A (Line 27), Section 3 (Line 49); Term: `Bidslab34`
- **Discrepancy Description & Rationale**: The subject identifier `Bidslab34` is presented without clarifying that it is a pseudonymized participant ID from the underlying BIDSleep cohort.

### 9. Multiple Hypothesis Correction Scope — Severity: 🟢 Nit
- **Affected Scope**: Section 1 (Table, Line 9); Term: `FDR q-value`
- **Discrepancy Description & Rationale**: The table reports FDR $q$-values without explicitly documenting the correction algorithm applied (Benjamini-Hochberg) or defining the exact family of hypotheses over which the correction was adjusted ($M=9$ primary pre-adaptation predictors).
