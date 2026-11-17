# Goldfish Annotated Review Report

**Status:** RESOLVED (All 10 accepted remediation items C-1 through C-10 implemented; D-1 contested; D-2 deferred)
**Target:** `experiments/EXP-019/HETEROGENEITY_AUDIT_REPORT.md` and `experiments/EXP-019/audit_heterogeneity_mechanisms.py`  
**Review Standard:** `feature-spec-standard` & `implementation-readiness-standard`  
**Resolution Skill:** `egm-resolve`  
**Date:** November 17, 2026

---

## Prioritized Findings & Resolution Decisions

### Finding C-1 [`egm-design`, `egm-zoom-out`, `egm-inventive-catalyst`] - Severity: 🔴 Blocker / 🟡 Major
* **Affected Scope:** `HETEROGENEITY_AUDIT_REPORT.md` Section 3; `audit_heterogeneity_mechanisms.py` Section 3 printouts
* **Discrepancy Description:** The compound gate ($\min_c P_C(y=c) \ge 3.0\%$ AND $D_{JS}(P_C \parallel P_{\text{pop}}) < 0.35$) is post-hoc data-snooped around `Bidslab34`. It discards a confirmed positive responder (`Bidslab15`, $+1.27\text{ pp}$), misses 11 of 12 non-responders in the cohort (sensitivity = $8.3\%$), sits on a brittle boundary margin, and papers over an unregularized loss formulation with a client-exclusion heuristic.
* **Rationale:** Presenting this gate as a triumphant exploratory safety deployment solution is misleading. It must be reframed as an empirical demonstration of why static pre-screening fails.
* ***AI Decision:*** **ACCEPT (PROPOSE FIX)**
* ***Justification:*** Acknowledging false positive responder exclusion and low sensitivity converts an overfitted heuristic into a defensible scientific finding, motivating algorithmic regularization (e.g., prior-preserving logit penalties) for future protocols.
* ***Proposed Edit:***
  * Location: `HETEROGENEITY_AUDIT_REPORT.md` Section 3
  * Change: Rename section to `## 3. Stress-Testing Static Pre-Adaptation Gating (Counterfactual Analysis)`. Neutrally document that the gate excluded positive responder `Bidslab15` ($+1.27\text{ pp}$) and failed to detect 11 of 12 non-responders ($8.3\%$ sensitivity). Conclude that static pre-adaptation screening is an ineffective blunt filter, and that principled resolution requires algorithmic loss regularization.

---

### Finding C-2 [`egm-comprehension`, `egm-zoom-out`, `egm-style`] - Severity: 🟡 Major
* **Affected Scope:** `HETEROGENEITY_AUDIT_REPORT.md` Header (Lines 1–6)
* **Discrepancy Description:** A fresh reader cannot identify the underlying staging task, model backbone, metric ($\Delta$ Macro-F1 percentage points on 4-stage sleep), responder threshold ($\Delta > 0.0\text{ pp}$), or probability distribution entities.
* **Rationale:** The document drops altitude too rapidly without establishing baseline experimental facts.
* ***AI Decision:*** **ACCEPT (PROPOSE FIX)**
* ***Justification:*** Adding a unified 4-bullet executive metadata box resolves all comprehension gaps without ballooning the document into textbook tutorial text.
* ***Proposed Edit:***
  * Location: `HETEROGENEITY_AUDIT_REPORT.md` Header
  * Change: Insert compact executive metadata block defining Task (4-stage sleep staging), Model (Model A CNN-BiGRU), Metric ($\Delta$ Macro-F1 in `pp`), Responder criterion ($\Delta > 0.0\text{ pp}$), and $D_{JS}$ bounds ($[0, 1]$ base 2).

---

### Finding C-3 [`egm-structural`, `egm-zoom-out`] - Severity: 🟡 Major
* **Affected Scope:** `HETEROGENEITY_AUDIT_REPORT.md` Section 1 Table & Section 2 Headings
* **Discrepancy Description:** Section 1 tested 6 mechanisms across 9 hypotheses, but Section 2 provided narrative subsections only for M1, M3, and M5. M2 (autonomic HR volatility), M4 (transition friction), and M6 (token exposure) were omitted without explanation.
* **Rationale:** Structural asymmetry obscures whether M2, M4, and M6 were evaluated or abandoned.
* ***AI Decision:*** **ACCEPT (PROPOSE FIX)**
* ***Justification:*** Sub-heading Table 1 by Mechanism 1–6 and adding a single compact Subsection 2.D maintains 1:1 symmetry without adding dilutive prose.
* ***Proposed Edit:***
  * Location: `HETEROGENEITY_AUDIT_REPORT.md` Section 1 & Section 2
  * Change: Sub-header Table 1 rows by Mechanism 1–6. Add `### D. Uniformly Null Autonomic, Structural, and Exposure Predictors (Mechanisms 2, 4, 6)` summarizing that HR volatility ($\rho = -0.21$), transition rate ($\rho = -0.02$), and token volume ($\rho = -0.06$) demonstrated no predictive association after FDR correction ($q = 0.9711$).

---

### Finding C-4 [`egm-design`, `egm-style`, `egm-zoom-out`] - Severity: 🟡 Major
* **Affected Scope:** `HETEROGENEITY_AUDIT_REPORT.md` Section 2.A & 2.B; `audit_heterogeneity_mechanisms.py` Line 311
* **Discrepancy Description:** 
  1. Oldham's variance cancellation theorem strictly holds for linear Pearson covariance, not rank-order Spearman $\rho$. Furthermore, claiming Oldham's method "proves the ceiling effect is an RTM artifact" is uncalibrated because the baseline correlation was already statistically insignificant ($\rho = -0.1710, p = 0.2559$).
  2. Section 2.A Line 30 labels an observational case as `*Causal Diagnosis*` and claims fine-tuning drove emission bias to "negative infinity".
* **Rationale:** Uncalibrated causal assertions and mathematical inaccuracies undermine scientific defensibility.
* ***AI Decision:*** **ACCEPT (PROPOSE FIX)**
* ***Justification:*** Reporting both Pearson and Spearman correlations establishes statistical completeness; replacing causal hyperbole with objective diagnostic phrasing ensures academic rigor.
* ***Proposed Edit:***
  * Location: `HETEROGENEITY_AUDIT_REPORT.md` Section 2.A & 2.B; `audit_heterogeneity_mechanisms.py`
  * Change: In Section 2.B, report Pearson ($r = +0.032, p = 0.832$) and Spearman ($\rho = +0.038, p = 0.804$) for Oldham's mean ($\frac{\text{ZS} + \text{FT}}{2}$). Clarify that baseline performance showed no significant relationship before or after adjustment. Reclassify `*Causal Diagnosis*` to `**Diagnostic Case Interpretation (Bidslab34):**` and calibrate emission bias description.

---

### Finding C-5 [`egm-implementation-readiness`] - Severity: 🟡 Major
* **Affected Scope:** `audit_heterogeneity_mechanisms.py` Lines 232–246
* **Discrepancy Description:** `adapt_supervised_ft()` puts the model in `model.train()`, consuming PyTorch RNG during dropout forward passes. Without explicit seeding, re-running the script produces subtle stochastic fluctuations in `param_displacement_norm`.
* **Rationale:** Bitwise exact reproduction of the predictor table requires deterministic RNG seeding.
* ***AI Decision:*** **ACCEPT (PROPOSE FIX)**
* ***Justification:*** Essential engineering hardening for reproducibility.
* ***Proposed Edit:***
  * Location: `audit_heterogeneity_mechanisms.py` entry point and subject adaptation loop
  * Change: Set `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, and `np.random.seed(42)` before adaptation.

---

### Finding C-6 [`egm-codebase-details`, `egm-implementation-readiness`] - Severity: 🟡 Major
* **Affected Scope:** `audit_heterogeneity_mechanisms.py` Lines 30–35 & Lines 426–433
* **Discrepancy Description:** `sys.path` omitted `exp019_dir`, and CLI default arguments were relative paths from CWD. Running the script from `experiments/EXP-019/` caused `FileNotFoundError` and `ModuleNotFoundError`.
* **Rationale:** CLI tools must be runnable from any directory within the repository.
* ***AI Decision:*** **ACCEPT (PROPOSE FIX)**
* ***Justification:*** Standardizing path resolution prevents broken executions.
* ***Proposed Edit:***
  * Location: `audit_heterogeneity_mechanisms.py` module setup and `argparse` definitions
  * Change: Add `exp019_dir = os.path.abspath(os.path.dirname(__file__))` to `sys.path`. Anchor all CLI defaults using `os.path.join(repo_root, ...)`.

---

### Finding C-7 [`egm-style`] - Severity: 🟡 Major
* **Affected Scope:** `audit_heterogeneity_mechanisms.py` Lines 80–90 & Lines 423–435
* **Discrepancy Description:** `run_heterogeneity_audit` lacked a docstring, and all `argparse` options lacked `--help` text.
* **Rationale:** Public script entrypoints must be documented for maintainability and automated execution.
* ***AI Decision:*** **ACCEPT (PROPOSE FIX)**
* ***Justification:*** Non-invasive documentation improvement.
* ***Proposed Edit:***
  * Location: `audit_heterogeneity_mechanisms.py`
  * Change: Add NumPy/Google-style docstring to `run_heterogeneity_audit()` and descriptive `help=` strings to all CLI flags.

---

### Finding C-8 [`egm-implementation-readiness`, `egm-codebase-details`, `egm-style`] - Severity: 🟢 Nit
* **Affected Scope:** `audit_heterogeneity_mechanisms.py` Lines 223–227
* **Discrepancy Description:** Lines 223–225 compute an unused fallback correlation into `f1_zs_cal` that is immediately overwritten on line 227 by `f1_score`. In addition, `from sklearn.metrics import f1_score` is re-imported 46 times inside the subject loop.
* **Rationale:** Dead code and in-loop redundant imports violate code cleanliness.
* ***AI Decision:*** **ACCEPT (PROPOSE FIX)**
* ***Justification:*** Straightforward cleanup.
* ***Proposed Edit:***
  * Location: `audit_heterogeneity_mechanisms.py`
  * Change: Move `from sklearn.metrics import f1_score` to module header, remove dead lines 223–225, and add defensive check for empty valid tokens.

---

### Finding C-9 [`egm-structural`, `egm-style`, `egm-codebase-details`] - Severity: 🟢 Nit
* **Affected Scope:** `audit_heterogeneity_mechanisms.py` Line 408; `HETEROGENEITY_AUDIT_REPORT.md` Line 45
* **Discrepancy Description:** Double curly braces `$D_{{JS}}(P_C \parallel P_{{\text{{pop}}}})$` emitted verbatim into markdown.
* **Rationale:** Formatting artifact that clutters LaTeX rendering.
* ***AI Decision:*** **ACCEPT (PROPOSE FIX)**
* ***Justification:*** Formatting correction.
* ***Proposed Edit:***
  * Location: `audit_heterogeneity_mechanisms.py` and `HETEROGENEITY_AUDIT_REPORT.md`
  * Change: Replace with single curly braces: `$D_{JS}(P_C \parallel P_{\text{pop}})$`.

---

### Finding C-10 [`egm-implementation-readiness`, `egm-codebase-details`] - Severity: 🟢 Nit
* **Affected Scope:** `audit_heterogeneity_mechanisms.py` Lines 103–106
* **Discrepancy Description:** Population stage prior averages across unusable nights (`usable == False`), where fractions are 0.0.
* **Rationale:** Data hygiene improvement.
* ***AI Decision:*** **ACCEPT (PROPOSE FIX)**
* ***Justification:*** Ensures population demographic prior strictly reflects valid sleep recordings.
* ***Proposed Edit:***
  * Location: `audit_heterogeneity_mechanisms.py`
  * Change: Filter `df_inv[df_inv["usable"] == True]` before computing mean stage fractions.

---

### Item D-1 [Epoch-Level HRV Extraction from Raw PPG/ECG] - Severity: Nit / External
* **Lenses:** `egm-design` (Critic Finding 4)
* **Scope:** Autonomic HR Volatility Mechanism
* **Description:** Critic suggested computing beat-to-beat HRV (RMSSD, pNN50, HF power) instead of nightly median HR differences.
* ***AI Decision:*** **REJECT WITH JUSTIFICATION (CONTEST)**
* ***Justification:*** The BIDSleep cohort dataset provides instantaneous heart rate (IHR) sampled at 1 Hz from wearable photoplethysmography / pulse signals, not raw microsecond R-peak intervals (RR tachogram). Frequency-domain HRV and RMSSD cannot be mathematically calculated on 1 Hz interpolated IHR without introducing interpolation artifacts. Nightly median HR difference is the valid, methodologically grounded metric available on this dataset. This data constraint is explicitly noted in Subsection 2.D.

---

### Item D-2 [Implementation of Prior-Regularized Training / Dynamic Test-Time Adaptation] - Severity: Strategic / Exploratory
* **Lenses:** `egm-inventive-catalyst` (Inventive Finding 1 & 2)
* **Scope:** Adaptation loss formulation and streaming test-time loops
* **Description:** Proposing concrete Bayesian prior penalties ($D_{KL}(P_\theta(y) \parallel P_{\text{pop}})$) or streaming entropy watchdogs.
* ***AI Decision:*** **DEFER (FUTURE PROTOCOL DESIGN)**
* ***Justification:*** The user instruction mandates: *"Do not change H-A, EXP-019 results, preregistration, or the locked research direction. Treat the heterogeneity/gating analysis as exploratory... Do not implement speculative new mechanisms or redesign the experiment."* Implementing active training regularizations or new test-time architectures exceeds the exploratory audit scope. These resolutions are recorded as prospective design recommendations for future experimental protocol proposals under DEC-021.

---

## Summary of Resolution Actions

- **Total Review Items Evaluated:** 12
- **Accepted (PROPOSE FIX):** 10 (Findings C-1 through C-10)
- **Rejected with Justification (CONTEST):** 1 (Item D-1, 1 Hz IHR data constraint)
- **Deferred to Future Protocols (DEFER):** 1 (Item D-2, retraining / dynamic test-time loops)
- **Target Files to Change:**
  1. `experiments/EXP-019/HETEROGENEITY_AUDIT_REPORT.md`
  2. `experiments/EXP-019/audit_heterogeneity_mechanisms.py`
  3. `HANDOFF.md`
