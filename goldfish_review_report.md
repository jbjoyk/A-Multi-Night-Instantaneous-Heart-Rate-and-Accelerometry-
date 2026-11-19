# Goldfish Annotated Review Report

**Target:** `EXP019_WITHIN_COHORT_SLEEP_SUMMARY_AGREEMENT_REPORT.md` (`full file`)  
**Status:** RESOLVED (Incorporated into report, script, and governance records under DEC-032)  
**Resolution Date:** 2026-11-19

## Prioritized Findings & Resolution Actions

### [FIND-B-01] [`egm-critic`] - Severity: Major — Status: RESOLVED
* **Affected Scope:** Section 1, Section 3.2, Section 3.3, Section 4 (Limits of Agreement scope)
* **Discrepancy Description:** Subject-averaged limits of agreement describe agreement for a participant's multi-night monitoring mean, not for an isolated single night. Describing them without this distinction produces an overly optimistic interpretation of spot measurement precision.
* **Rationale:** In repeated-measures method comparison where true values fluctuate (Case 2, Bland & Altman 1999/2007), averaging multiple nights per subject divides the within-subject variance by the number of nights ($n_i$). The resulting LoA quantifies agreement of multi-night habitual sleep duration, not single-night spot measurements.
* ***AI Suggestion:*** **PROPOSE FIX (APPLIED)**
* ***Justification:*** Essential biostatistical distinction required for Q1 publication integrity.
* ***Resolution Applied:***
  * Updated Section 1, Section 3.2, Section 3.3, and Table 1 in `EXP019_WITHIN_COHORT_SLEEP_SUMMARY_AGREEMENT_REPORT.md` and `research_engine/bidsleep_within_cohort_agreement_analysis.py` to explicitly state that the reported LoA ($[-30.11, +34.67]\text{ min}$ for TST) applies strictly to multi-night participant means ($\sim 3$ nights).
  * Explicitly documented that single-night agreement requires an independent variance-components model ($\text{LoA}_{\text{single}} = \bar{d} \pm 1.96 \sqrt{\sigma_b^2 + \sigma_w^2}$).

---

### [FIND-B-02] [`egm-critic`] - Severity: Major — Status: RESOLVED
* **Affected Scope:** Section 1, Section 3.1 (Micro-Data Information Loss Inventory)
* **Discrepancy Description:** The authoritative source artifact (`exp019_sleep_summary_agreement.csv`) is an aggregate summary table containing 48 cohort-level moments, not raw participant-night paired micro-data. Five secondary analyses could not be performed.
* **Rationale:** Without paired micro-data tuples $(y_{ij}^{\text{pred}}, y_{ij}^{\text{ref}})$, two-way error variance decomposition, linear mixed-effects proportional bias regression, formal heteroscedasticity tests, non-parametric participant-level cluster bootstrapping, and residual outlier leverage diagnostics cannot be evaluated.
* ***AI Suggestion:*** **PROPOSE FIX (APPLIED)**
* ***Justification:*** Transparently catalogs analytical boundaries and prevents misleading claims of unadjusted surrogates.
* ***Resolution Applied:***
  * Formalized a dedicated inventory in Section 3.1 of `EXP019_WITHIN_COHORT_SLEEP_SUMMARY_AGREEMENT_REPORT.md` explicitly documenting all five blocked micro-data analyses.
  * Established the subject-averaged independent cohort ($N=46, df=45$) as the primary scientifically defensible surrogate under available data.

---

### [FIND-B-03] [`egm-critic`] - Severity: Minor — Status: DEFERRED TO PHASE 3
* **Affected Scope:** `MANUSCRIPT_DRAFT_v0.1.md` Section 6.2 Table 1 (LoA Uncertainty Reporting)
* **Discrepancy Description:** Draft manuscript Table 1 reports parametric limits of agreement but omits standard errors and confidence intervals on the LoA bounds ($\text{SE}(\text{LoA})$ and 95% CIs on $\text{LoA}_{\text{low}}$ and $\text{LoA}_{\text{high}}$).
* **Rationale:** Reporting uncertainty around limits of agreement is recommended by Bland & Altman (1999) and clinical journal guidelines (*Lancet Digital Health*).
* ***AI Suggestion:*** **DEFER TO PHASE 3 (EDITORIAL POLISH)**
* ***Justification:*** Per Phase B governance rules, manuscript Sections 6–8 remain locked and unedited until Phase 3 synthesis reopens.
* ***Resolution Scheduled:***
  * In Phase 3, expand manuscript Table 1 to include $\text{SE}(\text{LoA})$ and explicit 95% confidence intervals on LoA bounds for all arms, and add explicit multi-night header scope.

---

### [FIND-B-04] [`egm-critic`] - Severity: Minor — Status: RESOLVED
* **Affected Scope:** Section 6.2 (WASO Underestimation Causal Framing)
* **Discrepancy Description:** Describing WASO underestimation as an "intrinsic sensor limitation" (inability to detect cortical micro-arousals) is an unsupported causal claim.
* **Rationale:** While cortical micro-arousals without gross motor activity can evade actigraphy broadly, asserting that the specific $-4.87\text{ min}$ model bias in FT-2 is biologically or hardware-driven lacks direct empirical proof. The bias could equally stem from model decision thresholds, training loss class weighting (wake class imbalance), or 30-second epoch feature representations.
* ***AI Suggestion:*** **PROPOSE FIX (APPLIED)**
* ***Justification:*** Eliminates unsubstantiated causal claims and aligns with clinical epidemiology reporting standards.
* ***Resolution Applied:***
  * Retracted the causal claim of an "intrinsic sensor limitation" in Section 6.2 of `EXP019_WITHIN_COHORT_SLEEP_SUMMARY_AGREEMENT_REPORT.md` and related documentation.
  * Re-anchored the discussion strictly on an **observed empirical model underestimation ($-4.87\text{ min}$, $p=0.0104$)**, cataloging the technical model factors (scoring thresholds, loss weighting, temporal feature resolution) alongside sensor sensitivity.

---

### [Additional Correction] [`egm-critic`] - Status: RESOLVED
* **Affected Scope:** Section 3.4, Section 5 (Unadjusted Pooled Night Interval Framing)
* **Discrepancy Description:** The unadjusted pooled-night interval (e.g. $[-39.60, +46.95]\text{ min}$ for TST) must not be presented as a validated single-night prediction interval.
* **Rationale:** In `night_level_pooled_n152_unadjusted`, all 152 nights are pooled without accounting for subject clustering ($df=151$). This confounds between-subject and within-subject variance.
* ***AI Suggestion:*** **PROPOSE FIX (APPLIED)**
* ***Justification:*** Prevents conflating unadjusted descriptive pooling with a validated inferential prediction interval.
* ***Resolution Applied:***
  * Added prominent cautionary banners in Section 3.4 and Section 5 of `EXP019_WITHIN_COHORT_SLEEP_SUMMARY_AGREEMENT_REPORT.md` and updated script print titles in `research_engine/bidsleep_within_cohort_agreement_analysis.py`.
  * Explicitly stated that unadjusted pooled night numbers are descriptive only and must never be cited as validated single-night prediction intervals.

---

### [FIND-B-05] & [FIND-B-06] [`egm-critic`] - Status: VERIFIED
* **FIND-B-05:** 100% numerical consistency verified across source CSV, Python script, Phase B report, and manuscript draft.
* **FIND-B-06:** Denominators ($N=46$ subjects, $N=152$ nights) and anti-leakage ($C \cap E = \emptyset$) confirmed intact.

---

## Post-Resolution Summary

- **FIND-B-01:** RESOLVED (Multi-night LoA scope clarified).
- **FIND-B-02:** RESOLVED (Micro-data information loss inventory documented).
- **FIND-B-03:** DEFERRED TO PHASE 3 (Table 1 LoA standard errors and CIs scheduled for Phase 3).
- **FIND-B-04:** RESOLVED (WASO causal sensor claim retracted; empirical model underestimation framed).
- **Additional Correction:** RESOLVED (Pooled-night interval framed as descriptive, not single-night prediction interval).
- **FIND-B-05 & FIND-B-06:** VERIFIED (Numerical consistency and denominator/leakage integrity).
- **Governance Ratification:** Formally enacted under **DEC-032** (2026-11-19).
- **Phase Gate Status:** **PHASE B FULLY RECONCILED AND CLEARED FOR PHASE 3**.
