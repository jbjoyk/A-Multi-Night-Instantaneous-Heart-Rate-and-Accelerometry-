# Intermediate Zoom Out Review Report (Stage 8)

**Target:** `experiments/EXP-019/HETEROGENEITY_AUDIT_REPORT.md`  
**Reviewer:** `egm-zoom-out` (Skeptical Architect)  
**Context:** `zoom_out_report.md` (Stage 5), `comprehension_report.md` (Stage 6), `critic_report.md` (Stage 7), `heterogeneity_predictors_table.csv`, and `audit_heterogeneity_mechanisms.py`

---

## Executive Architectural Assessment

The exploratory audit report for EXP-019 (`HETEROGENEITY_AUDIT_REPORT.md`) captures a vital scientific outcome: **pre-adaptation subject features offer essentially zero foreknowledge of adaptation gain ($q \approx 0.97$), whereas inter-night non-stationary label shift ($D_{JS}(P_C \parallel P_E)$, $p = 0.048$) governs catastrophic degradation.**

However, the previous review stages (Comprehension and Critic) generated an expansive list of 17 discrete findings, ranging from missing baseline acronyms to mathematical covariance proofs for Oldham's method and demands for full structural symmetry across all 6 mechanisms. 

If all reviewer recommendations were applied literally, this compact 51-line diagnostic report would balloon into a 300-line pedagogical monograph. As Skeptical Architect, my role is to prevent textbook bloat, reject overfitted symptom patches, and enforce minimalist, high-altitude architectural clarity.

---

## Prioritized Findings & Strategic Synthesis

### 1. Reframing Section 3: Rejection of Overfitted "Gating Solution" in Favor of Diagnostic Negative Finding — Severity: 🔴 Blocker / 🟡 Major
* **Affected Scope:** Section 3 (Lines 42–51, "Deployment Gating Simulation")
* **Simplicity & Architectural Rationale:**
  - *The Critic's finding is fatal to Section 3 as currently framed:* The compound gate ($\min_c P_C(y=c) \ge 3.0\%$ AND $D_{JS}(P_C \parallel P_{\text{pop}}) < 0.35$) was reverse-engineered around subject `Bidslab34`. In doing so, it discards a confirmed positive responder (`Bidslab15`, $+1.27\text{ pp}$), misses 11 of 12 non-responders (sensitivity = $8.3\%$), and sits on an unstable threshold boundary where a $+0.5\%$ shift would drop high responders `Bidslab42` ($+6.31\text{ pp}$) and `Bidslab62` ($+4.48\text{ pp}$).
  - *Zoom-Out Diagnosis:* Presenting this post-hoc gate as an exploratory "safety simulation" with triumphant metrics ("Severe Outliers Blocked!", "+1.39 pp Gain") is classic YAGNI symptom-patching. It papers over an architectural deficiency (monolithic SGD fine-tuning collapsing emission logits when a class is absent) with a brittle client-exclusion heuristic.
* **Strategic Recommendation:**
  - **Do NOT delete Section 3 entirely**, as the extreme failure of `Bidslab34` ($0.68\%$ Deep sleep) provides essential empirical insight.
  - **Reframe Section 3 as a Diagnostic Counterfactual / Stress Test**:
    - Rename section to: `## 3. Stress-Testing Static Pre-Adaptation Gating (Counterfactual Analysis)`.
    - Present the data neutrally: A static stage-scarcity threshold ($\min_c \ge 3.0\%$) excludes `Bidslab34` ($-16.34\text{ pp}$), but also excludes positive responder `Bidslab15` ($+1.27\text{ pp}$) while failing to detect 11 of 12 non-responders.
    - **Architectural Conclusion**: Conclude decisively that static pre-adaptation screening is an ineffective, blunt mechanism. The principled long-term architectural solution is **algorithmic regularization** (e.g., prior-preserving logit penalties or feature-only adaptation) rather than hard client gating.

### 2. Guarding Against Scope Creep: Modular Context vs. Textbook Bloat — Severity: 🟡 Major
* **Affected Scope:** Document Header (Lines 1–6) & Section 1 (Lines 7–20)
* **Simplicity & Architectural Rationale:**
  - The Comprehension review noted that a reader cannot tell what metric `Delta` represents (Macro-F1 on 4-stage sleep), what `H-A` stands for, or how $D_{JS}$ is bounded.
  - However, resolving this by writing exhaustive textbook paragraphs explaining Jensen-Shannon divergence formulas, Benjamini-Hochberg FDR theory, and BIDSleep cohort history violates the Zoom-Out standard. The report is an experimental artifact, not a tutorial.
* **Strategic Recommendation:**
  - Implement a **single, unified 4-line metadata callout block** at the top of the report:
    - **Task & Baseline:** EXP-019 subject-specific adaptation of Model A (CNN-BiGRU) on nights 2–3, evaluated on nights 4+ across 46 subjects (5-fold CV).
    - **Target Metric:** Sleep staging 4-class Macro-F1 (Wake, Light, Deep, REM); $\Delta = \text{Macro-F1}_{\text{FT}} - \text{Macro-F1}_{\text{ZS}}$ in percentage points (`pp`).
    - **Responder Definition:** $\Delta > 0.0\text{ pp}$ ($N=34$ responders, $N=12$ non-responders). Confirmatory H-A mean gain was $+1.01\text{ pp}$ (non-significant vs $+1.50\text{ pp}$ target, $p=0.171$).
    - **Divergence Metric:** $D_{JS}$ is base-2 Jensen-Shannon divergence ($\in [0, 1]$) between 4-stage empirical probability vectors ($P_C$ calibration, $P_E$ evaluation, $P_{\text{pop}}$ training fold prior).
  - Defer detailed variable schemas directly to the source code and artifact (`heterogeneity_predictors_table.csv`, `audit_heterogeneity_mechanisms.py`) rather than reproducing data dictionaries in markdown text.

### 3. Preventing Symmetrical Over-Engineering: Consolidating Null Mechanisms — Severity: 🟡 Major
* **Affected Scope:** Section 2 (Lines 21–41)
* **Simplicity & Architectural Rationale:**
  - The Critic review demanded "structural symmetry" by expanding Section 2 from 3 subsections (A, B, C) to 6 subsections covering all 6 mechanisms (adding M2: Autonomic HR Volatility, M4: Transition Friction, and M6: Token Exposure).
  - Writing three full narrative subsections just to repeat that $\rho = -0.21, -0.02, -0.06$ with $q \approx 0.97$ (all flat nulls) adds zero architectural value and dilutes reader focus away from the meaningful forensic insights (M1 label shift, M3 RTM artifacts, M5 fold confounding).
* **Strategic Recommendation:**
  - Maintain focus on the three mechanisms that yielded diagnostic insight, while explicitly acknowledging the others in a single, compact subsection:
    - Keep: `### A. Prior Probability Shift & Stage Absence (Mechanism 1)`
    - Keep: `### B. Regression to the Mean vs Headroom (Mechanism 3)`
    - Keep: `### C. Scope-Stratified Parameter Displacement (Mechanism 5)`
    - Add single consolidated subsection: `### D. Uniformly Null Autonomic, Structural, and Exposure Predictors (Mechanisms 2, 4, 6)`:
      - 2 concise sentences summarizing that HR volatility ($\rho = -0.21$), stage transition rate ($\rho = -0.02$), and calibration token volume ($\rho = -0.06$) demonstrated no predictive association with adaptation gain after FDR adjustment ($q = 0.9711$).
  - Group Section 1 table cleanly by Mechanism ID (M1–M6) to establish 1:1 nomenclature symmetry without prose bloat.

### 4. Methodological & Epistemic Calibration (Oldham & Causal De-escalation) — Severity: 🟡 Major
* **Affected Scope:** Section 2.A (Line 30) & Section 2.B (Lines 32–35)
* **Simplicity & Architectural Rationale:**
  - Critic correctly noted two methodological overreaches:
    1. Line 30 states `*Causal Diagnosis*` and claims fine-tuning drove emission bias to "negative infinity". Observational sleep record audits cannot assert strict mathematical infinity or definitive causal proof.
    2. Line 34 claims Oldham's correction "proves that the apparent ceiling effect is an RTM artifact", despite the naive correlation being statistically insignificant ($\rho = -0.1710, p = 0.2559$). Furthermore, Oldham's variance cancellation theorem strictly holds for linear Pearson covariance, not rank-order Spearman $\rho$.
* **Strategic Recommendation:**
  - Tone down uncalibrated causal language: Reclassify `*Causal Diagnosis*` to `**Forensic Case Interpretation (Bidslab34):**` and describe emission bias as "severely suppressed" rather than "driven to negative infinity".
  - Clarify Section 2.B: Report both Spearman ($\rho = +0.038, p = 0.804$) and Pearson ($r = +0.032, p = 0.832$) for Oldham's mean ($\frac{\text{ZS} + \text{FT}}{2}$). State cleanly that baseline Zero-Shot performance showed no significant relationship with adaptation gain before ($p = 0.256$) or after ($p = 0.804$) Oldham adjustment, demonstrating that neither true ceiling effects nor RTM artifacts explain the outcome variance.

### 5. Architectural Altitude: Preserving the Root Scientific Insight — Severity: 🟡 Major
* **Affected Scope:** Concluding synthesis of the document
* **Simplicity & Architectural Rationale:**
  - The document must not lose its core architectural punchline among statistical nuances:
    - Static pre-adaptation predictors are blind ($q \approx 0.97$).
    - Retrospective label shift is decisive ($D_{JS}(P_C \parallel P_E)$, $\rho = -0.2933, p = 0.0479$).
* **Strategic Recommendation:**
  - Conclude Section 2 with a punchy, high-altitude architectural takeaway:
    - **Takeaway:** Adaptation failures are driven by dynamic, non-stationary stage shifts between calibration and evaluation nights (e.g., absent stages in calibration reappearing in evaluation), not by baseline subject physiology or static pre-adaptation scarcity.
    - Future adaptation protocols must focus on **robustness to temporal non-stationarity** (e.g. prior-anchored loss functions) rather than static pre-screening.

---

## Action Plan for Report Refactoring

| Priority | Action | Target Location |
|---|---|---|
| **1. Reframe Gating** | Convert "Deployment Gating Simulation" to "Stress-Testing Static Pre-Adaptation Gating", documenting false-positive responder rejection ($N=1$) and low sensitivity ($8.3\%$). | Section 3 |
| **2. Add Minimalist Context** | Add a compact 4-bullet executive metadata box defining Task, Macro-F1 $\Delta$, Responder Threshold, and $D_{JS}$ bounds. | Header (Lines 1–6) |
| **3. Consolidate Nulls** | Group Table 1 by Mechanism 1–6 and add a single unified Subsection 2.D summarizing null findings for Mechanisms 2, 4, and 6. | Sections 1 & 2.D |
| **4. Calibrate Statistics** | Add Pearson $r$ to Oldham analysis; de-escalate "causal" and "negative infinity" rhetoric to objective diagnostic terminology. | Section 2.A & 2.B |
| **5. Clean Typography** | Strip double braces `$D_{{JS}}$` and standardize TeX notation throughout. | Global |
