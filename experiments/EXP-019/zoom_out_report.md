# Goldfish Review Report: Initial Zoom Out Review (Stage 5)

**Target:** `experiments/EXP-019/HETEROGENEITY_AUDIT_REPORT.md` (Lines 1–51)  
**Reviewers:** `egm-zoom-out` (Skeptical Architect), `egm-inventive-catalyst` (Inventive Problem-Solving Goldfish)  
**Context:** `experiments/EXP-019/heterogeneity_predictors_table.csv`, `experiments/EXP-019/audit_heterogeneity_mechanisms.py`

---

## Reviewer Status

| Reviewer | Lens | Status |
|---|---|---|
| `egm-zoom-out` | Skeptical Architect (Simplicity, YAGNI, Codebase Reuse) | Completed |
| `egm-inventive-catalyst` | Inventive Problem Solving (TRIZ Contradiction Analysis) | Completed |

---

## Prioritized Findings (Blockers -> Major -> Nits)

### 1. Context Insufficiency: Missing High-Altitude Motivation & Confirmatory Outcome Framing — Severity: 🟡 Major
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** Lines 1–6 (Document Header and Intro)
* **Discrepancy Description**: The document plunges immediately into raw hypothesis tables without explaining *why* this audit exists. A fresh reader has no high-level context on:
  1. What EXP-019 was (within-person supervised adaptation on nights 2–3 evaluated on nights 4+).
  2. The locked confirmatory H-A outcome (+1.01 pp mean gain, 73.9% responder rate, but non-significant under two-tailed Wilcoxon signed-rank $W = 654.0, p = 0.1706$ vs the $+1.50\text{ pp}$ threshold).
  3. The core problem: The confirmatory test was pulled below significance by severe negative outliers (e.g. `Bidslab34` at $-16.34\text{ pp}$), which triggered this exploratory root-cause audit under the DEC-021 mandate.
* **Rationale & Recommendation**: Add a concise executive context block at the top defining the research setting, the confirmatory outcome, and the specific investigative purpose of this exploratory audit.

### 2. Binary Discard vs Prior-Preserving Adaptation in Deployment Gating — Severity: 🟡 Major
* **Reviewer Lens:** `egm-inventive-catalyst` & `egm-zoom-out`
* **Affected Scope:** Section 3 (Lines 42–51, "Deployment Gating Simulation")
* **Discrepancy Description**: The report accepts an all-or-nothing trade-off for deployment safety: either permit unconstrained fine-tuning (risking catastrophic collapse on scarce classes, e.g., `Bidslab34` at $-16.34\text{ pp}$), or reject subjects completely via hard threshold gating ($\min_c P_C(y=c) \ge 3\%$).
  Furthermore, `egm-zoom-out` identified that this gate was tailored post-hoc to catch `Bidslab34` ($0.68\%$ Deep sleep) but also silently discards legitimate positive responders (e.g., `Bidslab15`, $+1.27\text{ pp}$ gain; while narrowly missing `Bidslab42` at $3.07\%$ with $+6.31\text{ pp}$ gain). Across the whole cohort, pre-adaptation stage scarcity lacks significant monotonic predictive power ($q = 0.9711$).
* **Rationale & Recommendation**: 
  - **Inventive Pattern #4 (Beforehand Cushioning) & Pattern #6 (Dynamics)**: Rather than binary discard, resolve the contradiction by formulating prior-regularized adaptation (e.g., Bayesian prior penalty against population distribution $D_{KL}(P_\theta(y) \parallel P_{\text{pop}}(y))$ or class-frequency logit adjustment).
  - Explicitly document in the report that the exploratory gating rule rejected 2 subjects (1 outlier, 1 responder) and acknowledge its post-hoc heuristic limitations.

### 3. Monolithic Coupling of Feature Personalization and Class Prior Fitting — Severity: 🟡 Major
* **Reviewer Lens:** `egm-inventive-catalyst`
* **Affected Scope:** Section 2.A (Lines 23–31, "Prior Probability Shift & Stage Absence")
* **Discrepancy Description**: The report treats catastrophic forgetting of Deep sleep as an inevitable property of the data rather than an architectural coupling flaw. Because standard SGD optimizes cross-entropy across all layers, calibrating the feature extractor to subject-specific autonomic signals (HRV/motion dynamics) simultaneously corrupts the classification layer with short-term empirical stage frequencies ($P_C(y)$).
* **Rationale & Recommendation**: **Inventive Pattern #2 (Extraction / Taking Out)**: Note in the diagnostic discussion that decoupling feature adaptation (e.g. updating encoder/GRU while freezing or regularizing classifier logits to population priors) could prevent emission bias collapse when short calibration windows lack specific sleep stages.

### 4. Explicit Contrast Between Static Predictors and Retrospective Divergence — Severity: 🟢 Nit
* **Reviewer Lens:** `egm-zoom-out`
* **Affected Scope:** Section 1 (Lines 7–20) & Section 2.A (Lines 26–30)
* **Discrepancy Description**: Table 1 lists only static pre-adaptation predictors (all $q \approx 0.97$, unsupported), while the single statistically significant relationship discovered—retrospective evaluation divergence $D_{JS}(P_C \parallel P_E)$ ($\rho = -0.2933, p = 0.0479$)—is placed in Section 2 prose.
* **Rationale & Recommendation**: Add a concluding comparative summary note clarifying the contrast between static pre-adaptation predictors (incapable of predicting adaptation failure) and dynamic retrospective shift (which explains why failure occurred).
