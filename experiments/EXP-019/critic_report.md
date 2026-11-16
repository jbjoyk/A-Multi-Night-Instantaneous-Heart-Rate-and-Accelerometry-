# Critic Review Report (Stage 7)

**Target:** `experiments/EXP-019/HETEROGENEITY_AUDIT_REPORT.md` (Lines 1–51)  
**Standard:** `feature-spec-standard`  
**Reviewers:** `egm-design` (Hostile Design Critic), `egm-structural` (Document Architect & Layout Validator), `egm-style` (Technical Copy Editor)  
**Context:** `experiments/EXP-019/heterogeneity_predictors_table.csv`, `experiments/EXP-019/audit_heterogeneity_mechanisms.py`

---

## Reviewer Status

| Reviewer | Lens | Status |
|---|---|---|
| `egm-design` | Hostile Design Critic (Statistical rigor, failure modes, data snooping) | Completed |
| `egm-structural` | Document Architect (Symmetry, heading hierarchy, table formatting) | Completed |
| `egm-style` | Technical Copy Editor (Objectivity, conciseness, uncalibrated claims) | Completed |

---

## Prioritized Findings (Blockers -> Major -> Nits)

### 1. Exploratory Gating Rule Suffers from Post-Hoc Data Snooping and Rejects Genuine Responders — Severity: 🔴 Blocker / 🟡 Major
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** Section 3 (Lines 42–51, "Deployment Gating Simulation")
* **Discrepancy Description**: 
  1. *Post-hoc threshold tailoring:* The compound condition ($\min_c P_C(y=c) \ge 3.0\%$ AND $D_{JS}(P_C \parallel P_{\text{pop}}) < 0.35$) was reverse-engineered knowing `Bidslab34`'s calibration metrics (`deep_frac_cal` = 0.0068, `D_JS_pop` = 0.375).
  2. *False positive rejection of responders:* Out of 46 subjects, the gate rejects exactly $N=2$: `Bidslab34` ($\Delta = -16.34\text{ pp}$) and `Bidslab15` ($\Delta = +1.27\text{ pp}$). `Bidslab15` is a confirmed positive responder (`deep_frac_cal` = 0.017, Fold 4). The report advertises "Mean Delta Among Rejected Subjects: -7.54 pp", masking that half of the rejected subjects were genuine responders.
  3. *Fragile boundary margins:* High responders `Bidslab42` ($\min_c = 3.07\%$, $\Delta = +6.31\text{ pp}$) and `Bidslab62` ($\min_c = 3.17\%$, $\Delta = +4.48\text{ pp}$) barely cleared the gate. A minor threshold shift from 3.0% to 3.5% would have erroneously discarded two of the cohort's strongest responders.
  4. *Near-zero sensitivity to non-responders:* The gate fails to catch 11 of the 12 non-responders in the cohort (including severe degradation cases like `Bidslab65` [$-6.57\text{ pp}$] and `Bidslab10` [$-6.47\text{ pp}$]). True sensitivity to non-responders is only $1/12 = 8.3\%$.
* **Rationale & Recommendation**: Explicitly document in Section 3 that the rule was tuned post-hoc with knowledge of test outcomes. Add full disclosure that the gate rejected `Bidslab15` (+1.27 pp) and missed 11 of 12 non-responders. State clearly that establishing a production safety gate requires nested cross-validation or pre-registered thresholding on an independent holdout.

### 2. Methodological Inconsistency in Oldham's Method for RTM Evaluation — Severity: 🟡 Major
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** Section 2.B (Lines 32–36; `audit_heterogeneity_mechanisms.py`, Line 311)
* **Discrepancy Description**: 
  1. *Violation of mathematical covariance cancellation:* Oldham's method (Oldham, 1962) relies on the algebraic identity:
     $$\text{Cov}\left(\frac{X+Y}{2}, Y-X\right) = \frac{1}{2}\left(\text{Var}(Y) - \text{Var}(X)\right)$$
     This cancellation theorem strictly requires **linear Pearson covariance**. In `audit_heterogeneity_mechanisms.py` (Line 311), the audit computes Spearman rank correlation (`stats.spearmanr(df_out["oldham_mean"], df_out["delta_HA"])`). Rank transformations do not preserve linear variance cancellation, compromising the theoretical guarantee of Oldham's adjustment.
  2. *Over-interpreting a null baseline correlation:* Line 33 claims: "Naive correlation is negative purely due to measurement error variance... proving that the apparent 'ceiling effect' is partially an RTM artifact!" However, the naive correlation was $\rho = -0.1710$ with $p = 0.2559$. Because $p > 0.25$, the naive correlation was never statistically distinguishable from zero; it is scientifically unwarranted to claim an artifact was "resolved" or "proven" when no significant relationship existed initially.
* **Rationale & Recommendation**: In Section 2.B, compute and report both Pearson and Spearman correlations for Oldham's mean. Qualify the interpretation to state that baseline Zero-Shot performance showed no statistically significant relationship with $\Delta$ before ($p = 0.256$) or after ($p = 0.804$) adjustment, confirming the absence of an empirical ceiling effect in this cohort.

### 3. Incomplete Parameter Displacement Stratification (Fold Hyperparameter Confounder) — Severity: 🟡 Major
* **Reviewer Lens:** `egm-design`
* **Affected Scope:** Section 2.C (Lines 37–41)
* **Discrepancy Description**: Section 2.C correctly observes that Fold 4 used `scope=head_gru` (freezing the CNN, yielding parameter displacement norms $\approx 0.028–0.030$), creating an artificial bimodal split when pooled with Folds 1–3, 5 (`scope=all`). However, stratifying by `scope=all` ($n=37$) does not eliminate fold-level optimization confounding:
  - Fold 1: $\text{lr} = 10^{-4}$, $\text{epochs} = 5$ (displacements $\approx 0.11–0.15$)
  - Folds 2 & 3: $\text{lr} = 3 \times 10^{-5}$, $\text{epochs} = 15$ (displacements $\approx 0.10–0.15$)
  - Fold 5: $\text{lr} = 3 \times 10^{-5}$, $\text{epochs} = 5$ (displacements $\approx 0.044–0.054$)
  Fold 5 adapted for only 5 epochs at low lr, compressing its displacement to one-third of Folds 1–3. Thus, within `scope=all`, parameter displacement is still confounded by fold-level optimization hyperparameters ($\text{lr} \times \text{epochs}$).
* **Rationale & Recommendation**: Acknowledge this second-layer confounder in Section 2.C. Note that within `scope=all`, displacement variations reflect fold-level Stage-D hyperparameter selection in addition to subject-level dynamics.

### 4. Severe Structural Asymmetry & Omission of Mechanisms 2, 4, and 6 — Severity: 🟡 Major
* **Reviewer Lens:** `egm-structural`
* **Affected Scope:** Section 2 (Lines 21–41)
* **Discrepancy Description**: Section 1 tests 9 hypotheses across 6 distinct mechanisms (M1: Prior Shift / Stage Scarcity, M2: Autonomic HR Volatility, M3: Baseline Headroom / RTM, M4: Transition Friction, M5: Parameter Drift, M6: Token Exposure / Signal Coverage). However, Section 2 only provides narrative subsections for Mechanisms 1, 3, and 5 (`### A. Prior Probability Shift...`, `### B. Regression to the Mean...`, `### C. Scope-Stratified Parameter Displacement...`). Mechanisms 2, 4, and 6 are completely omitted without explanatory justification.
* **Rationale & Recommendation**: Expand Section 2 to include subsections for Mechanisms 2, 4, and 6 (even if brief summaries of negative findings), or rename Section 2 to explicitly state it highlights the 3 mechanisms showing notable forensic phenomena.

### 5. Taxonomy & Nomenclature Disconnect Between Section 1 and Section 2 — Severity: 🟡 Major
* **Reviewer Lens:** `egm-structural`
* **Affected Scope:** Section 1 Table (Lines 9–19) vs Section 2 Headings (Lines 23, 32, 37)
* **Discrepancy Description**: Section 1 labels hypotheses using letter prefixes (`H-D1..3`, `H-P1`, `H-C1..2`, `H-L1`, `H-R1`, `H-S1`) without defining the acronym key or grouping rows by mechanism number. Conversely, Section 2 headings reference mechanisms by ordinal integer (`Mechanism 1`, `Mechanism 3`, `Mechanism 5`), but never mention the corresponding hypothesis IDs (`H-D1`, `H-C1`, `H-R1`).
* **Rationale & Recommendation**: Group the table in Section 1 by **Mechanism 1–6** and explicitly correlate the hypothesis IDs with the narrative headings in Section 2.

### 6. Emotional Typography and Promotional Framing in Gating Results — Severity: 🟡 Major
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** Section 3, Line 49 (`Severe Negative Outliers Blocked: Bidslab34 ... was **REJECTED** by the gate because its Deep sleep was < 3.0%!`)
* **Discrepancy Description**: The sentence uses promotional framing ("Severe Negative Outliers Blocked"), capitalized shouting boldface (`**REJECTED**`), and an exclamation mark (`!`). Engineering and scientific technical reports require dry, objective reporting.
* **Rationale & Recommendation**: Change to neutral, declarative phrasing: `- **Outlier Exclusion**: Subject Bidslab34 (-16.34 pp) was excluded by the gate due to calibration deep sleep frequency < 3.0%.`

### 7. Epistemic Overreach and Uncalibrated Causal Claims — Severity: 🟡 Major
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** Section 2.A (Line 30), Section 2.B (Line 34)
* **Discrepancy Description**: 
  1. Section 2.A Line 30 labels an observational case as `*Causal Diagnosis*` and claims fine-tuning drove emission bias to "negative infinity".
  2. Section 2.B Line 34 claims Oldham's correction is "proving that the apparent ceiling effect is partially an RTM artifact!".
* **Rationale & Recommendation**: Reclassify `*Causal Diagnosis*` to `**Diagnostic Interpretation:**` and calibrate the Oldham explanation to state that baseline dependence is consistent with mathematical coupling rather than empirical headroom.

### 8. Escaped TeX Formatting Artifacts and Unit Standardization — Severity: 🟢 Nit
* **Reviewer Lens:** `egm-structural`, `egm-style`
* **Affected Scope:** Lines 27, 28, 30, 45, 47–49
* **Discrepancy Description**: Doubled curly braces (`$D_{{JS}}(P_C \parallel P_{{\text{{pop}}}})$`) from Python raw strings, inconsistent percentage styling (`0.7%` vs `$0.0\%$`), and un-texted covariance terms (`\text{Cov}(ZS, FT - ZS)`).
* **Rationale & Recommendation**: Normalize to standard TeX: `$D_{JS}(P_C \parallel P_{\text{pop}})$` and `\text{Cov}(\text{ZS}, \text{FT} - \text{ZS})`.
