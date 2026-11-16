# Consolidated EGM Goldfish Review Report (Stages 5–9)

**Target Document:** [`experiments/EXP-019/HETEROGENEITY_AUDIT_REPORT.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-019/HETEROGENEITY_AUDIT_REPORT.md)  
**Implementation Script:** [`experiments/EXP-019/audit_heterogeneity_mechanisms.py`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-019/audit_heterogeneity_mechanisms.py)  
**Data Artifact:** [`experiments/EXP-019/heterogeneity_predictors_table.csv`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-019/heterogeneity_predictors_table.csv)  
**Review Standard:** EGM Goldfish Protocol (Full Uninterrupted Flow: Stages 5 → 6 → 7 → 8 → 9)  
**Date:** November 17, 2026

---

## 1. Executive Summary & Review Dashboard

The exploratory heterogeneity diagnostic audit delivers critical empirical insights into the EXP-019 multi-night adaptation experiment. It establishes two fundamental scientific conclusions:
1. **Pre-adaptation subject features offer essentially zero predictive foreknowledge of adaptation gain** ($q \approx 0.9711$ across all 9 pre-adaptation hypotheses).
2. **Subject performance degradation is governed by inter-night non-stationary label shift** ($D_{JS}(P_C \parallel P_E)$, $\rho = -0.2933, p = 0.0479$). In particular, the severe collapse of `Bidslab34` ($-16.34\text{ pp}$) is traced directly to extreme stage absence during calibration ($0.7\%$ Deep sleep) reappearing in evaluation nights (~$18\%$).

Across all five stages of the EGM Goldfish Review Protocol, the mathematical execution, zero-leakage invariants, and numerical reproduction of results were **verified with 100% parity**. 

However, the review swarm identified critical methodological, architectural, and presentation vulnerabilities:
- **Flawed Gating Framing (Major):** Section 3 presented a post-hoc compound gate as an exploratory "safety simulation". Critic and Zoom-Out reviews proved that this rule was reverse-engineered to catch `Bidslab34`, discards a confirmed positive responder (`Bidslab15`, $+1.27\text{ pp}$), misses 11 of 12 non-responders (sensitivity = $8.3\%$), and represents brittle symptom-patching rather than principled algorithmic resolution.
- **Context Insufficiency vs. Textbook Bloat (Major):** The document lacked high-altitude framing of the EXP-019 task, metric ($\Delta$ Macro-F1), and responder threshold, but must resolve this via a concise metadata block rather than sprawling textbook text.
- **Statistical Calibration (Major):** Oldham's method requires reporting Pearson $r$ alongside Spearman $\rho$, while recognizing that the baseline correlation was already non-significant ($p = 0.256$). Causal overreach ("negative infinity", "causal diagnosis") must be de-escalated to objective diagnostic language.
- **Implementation Portability & Determinism (Major):** The audit script lacked deterministic RNG seeding for PyTorch dropout during parameter displacement extraction, contained brittle working-directory-dependent CLI defaults, and omitted `sys.path` entries.

---

## 2. Reviewer Status Matrix

| Stage | Reviewer Lens | Persona / Agent | Target Scope | Status |
|---|---|---|---|---|
| **Stage 5** | Skeptical Architect | `egm-zoom-out` | Full Report (Lines 1–51) | **Completed** |
| **Stage 5** | Inventive Problem Solving | `egm-inventive-catalyst` | Full Report (Lines 1–51) | **Completed** |
| **Stage 6** | Naive Reader (Comprehension) | `egm-comprehension` | Report (Isolated Reading) | **Completed** |
| **Stage 7** | Hostile Design Critic | `egm-design` | Feature Scope (Lines 1–51) | **Completed** |
| **Stage 7** | Document Architect (Layout) | `egm-structural` | Feature Scope (Lines 1–51) | **Completed** |
| **Stage 7** | Technical Copy Editor | `egm-style` | Feature Scope (Lines 1–51) | **Completed** |
| **Stage 8** | Intermediate Zoom Out | `egm-zoom-out` | Full Synthesis & Anti-Bloat | **Completed** |
| **Stage 9** | Pragmatic Tech Lead | `egm-implementation-readiness` | `audit_heterogeneity_mechanisms.py` | **Completed** |
| **Stage 9** | Codebase Details Auditor | `egm-codebase-details` | Code vs Doc & Tabular Parity | **Completed** |
| **Stage 9** | Technical Copy Editor | `egm-style` | Docstrings, CLI & Comments | **Completed** |

---

## 3. Consolidated Findings by Severity

### 🔴 Blocker / Critical Architectural Decisions

#### Finding C-1: Reframe Section 3 from "Deployment Gating Solution" to "Diagnostic Counterfactual Stress Test"
* **Lenses:** `egm-design`, `egm-zoom-out`, `egm-inventive-catalyst`
* **Affected Scope:** `HETEROGENEITY_AUDIT_REPORT.md` Section 3; `audit_heterogeneity_mechanisms.py` Section 3 printouts
* **Discrepancy:** The compound gate ($\min_c P_C(y=c) \ge 3.0\%$ AND $D_{JS}(P_C \parallel P_{\text{pop}}) < 0.35$) is post-hoc data-snooped around `Bidslab34`. In doing so, it:
  1. Erroneously discards confirmed positive responder `Bidslab15` ($+1.27\text{ pp}$).
  2. Misses 11 of 12 non-responders in the cohort (sensitivity = $8.3\%$).
  3. Sits on an unstable threshold boundary where a $+0.5\%$ shift would drop high responders `Bidslab42` ($+6.31\text{ pp}$) and `Bidslab62` ($+4.48\text{ pp}$).
  4. Masks an architectural limitation (unregularized cross-entropy driving absent-class logits negative) with a brittle client-exclusion heuristic.
* **Consolidated Action:**
  - Rename Section 3 to: `## 3. Stress-Testing Static Pre-Adaptation Gating (Counterfactual Analysis)`.
  - Disclose that the gate excluded positive responder `Bidslab15` and failed to catch 11 of 12 non-responders.
  - Conclude that static pre-adaptation screening is an ineffective, blunt tool; the principled architectural solution is algorithmic regularization (prior-preserving loss penalties or feature-only adaptation).

---

### 🟡 Major Findings

#### Finding C-2: Add Unified 4-Line Executive Context Box (Preventing Textbook Bloat)
* **Lenses:** `egm-comprehension`, `egm-zoom-out`, `egm-style`
* **Affected Scope:** `HETEROGENEITY_AUDIT_REPORT.md` Header (Lines 1–6)
* **Discrepancy:** A reader cannot tell what metric `Delta` represents (Macro-F1 percentage points on 4-stage sleep), what `H-A` stands for, what the responder threshold is, or what entity probability distributions $P$ represent.
* **Consolidated Action:** Add a concise metadata box immediately following the title:
  - **Task & Model:** EXP-019 subject-specific adaptation of Model A (CNN-BiGRU) on nights 2–3, evaluated on nights 4+ across 46 subjects (5-fold CV).
  - **Target Metric:** 4-class sleep staging Macro-F1 (Wake, Light, Deep, REM); $\Delta = \text{Macro-F1}_{\text{FT}} - \text{Macro-F1}_{\text{ZS}}$ in percentage points (`pp`).
  - **Responder Definition:** $\Delta > 0.0\text{ pp}$ ($N=34$ responders, $N=12$ non-responders). Confirmatory H-A mean gain was $+1.01\text{ pp}$ (non-significant vs $+1.50\text{ pp}$ target, $p=0.171$).
  - **Divergence Metrics:** $D_{JS}$ is base-2 Jensen-Shannon divergence ($\in [0, 1]$) between 4-stage empirical probability vectors ($P_C$ calibration, $P_E$ evaluation, $P_{\text{pop}}$ training fold prior).

#### Finding C-3: Consolidate Null Mechanisms (M2, M4, M6) into a Single Subsection
* **Lenses:** `egm-structural`, `egm-zoom-out`
* **Affected Scope:** `HETEROGENEITY_AUDIT_REPORT.md` Section 1 Table & Section 2 Headings
* **Discrepancy:** Section 1 tested 6 mechanisms, but Section 2 provided narrative subsections only for M1, M3, and M5. M2 (autonomic HR volatility), M4 (transition friction), and M6 (token exposure) were omitted without explanation.
* **Consolidated Action:**
  - Sub-header Table 1 explicitly by **Mechanism 1 through 6** and correlate hypothesis IDs (`H-D1..3`, `H-P1`, `H-C1..2`, `H-L1`, `H-R1`, `H-S1`).
  - Add a compact subsection `### D. Uniformly Null Autonomic, Structural, and Exposure Predictors (Mechanisms 2, 4, 6)` summarizing that HR volatility ($\rho = -0.21$), transition rate ($\rho = -0.02$), and calibration token volume ($\rho = -0.06$) demonstrated no predictive power after FDR adjustment ($q = 0.9711$).

#### Finding C-4: Methodological & Epistemic Calibration (Oldham's Method & Causal Language)
* **Lenses:** `egm-design`, `egm-style`, `egm-zoom-out`
* **Affected Scope:** `HETEROGENEITY_AUDIT_REPORT.md` Section 2.A & 2.B; `audit_heterogeneity_mechanisms.py` Line 311
* **Discrepancy:** 
  1. Oldham's variance cancellation theorem strictly holds for linear Pearson covariance, not rank-order Spearman $\rho$. Furthermore, claiming Oldham's method "proves the ceiling effect is an RTM artifact" is uncalibrated because the baseline correlation was already statistically insignificant ($\rho = -0.1710, p = 0.2559$).
  2. Section 2.A Line 30 labels an observational case as `*Causal Diagnosis*` and claims fine-tuning drove emission bias to "negative infinity".
* **Consolidated Action:**
  - In Section 2.B, report both Pearson ($r = +0.032, p = 0.832$) and Spearman ($\rho = +0.038, p = 0.804$) for Oldham's mean ($\frac{\text{ZS} + \text{FT}}{2}$). State cleanly that baseline performance showed no significant relationship with $\Delta$ before or after adjustment.
  - Reclassify `*Causal Diagnosis*` to `**Forensic Case Interpretation (Bidslab34):**` and describe emission bias as "severely suppressed" rather than "driven to negative infinity".

#### Finding C-5: Deterministic RNG Seeding for PyTorch Dropout in Parameter Displacement
* **Lenses:** `egm-implementation-readiness`
* **Affected Scope:** `audit_heterogeneity_mechanisms.py` Lines 232–246
* **Discrepancy:** `adapt_supervised_ft()` puts the model in `model.train()`, consuming PyTorch RNG during dropout forward passes. Without explicit seeding, re-running the script produces subtle stochastic fluctuations in `param_displacement_norm`.
* **Consolidated Action:** Add `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, and `np.random.seed(42)` at script initialization and before each subject's adaptation.

#### Finding C-6: Portability & Path Anchoring for CLI Execution
* **Lenses:** `egm-codebase-details`, `egm-implementation-readiness`
* **Affected Scope:** `audit_heterogeneity_mechanisms.py` Lines 30–35 & Lines 426–433
* **Discrepancy:** `sys.path` omitted `exp019_dir`, and CLI default arguments were relative paths from CWD. Running the script from `experiments/EXP-019/` caused `FileNotFoundError` and `ModuleNotFoundError`.
* **Consolidated Action:** Add `exp019_dir` to `sys.path` and anchor all CLI default paths to `repo_root` using `os.path.join(repo_root, ...)`.

#### Finding C-7: Docstrings and CLI Argument Help Strings
* **Lenses:** `egm-style`
* **Affected Scope:** `audit_heterogeneity_mechanisms.py` Lines 80–90 & Lines 423–435
* **Discrepancy:** `run_heterogeneity_audit` lacked a docstring, and all `argparse` options lacked `--help` text.
* **Consolidated Action:** Add Google-style docstrings with parameter definitions, return types, and descriptive CLI `--help` strings.

---

### 🟢 Nits & Code Quality

#### Finding C-8: Dead Code and In-Loop Import Removal
* **Lenses:** `egm-implementation-readiness`, `egm-codebase-details`, `egm-style`
* **Affected Scope:** `audit_heterogeneity_mechanisms.py` Lines 223–227
* **Consolidated Action:** Relocate `from sklearn.metrics import f1_score` to module header, delete dead lines 223–225 (`pd.Series(...).corr(...)`), and add defensive handling for empty calibration slices.

#### Finding C-9: Normalize TeX Escaping & Standardization
* **Lenses:** `egm-structural`, `egm-style`, `egm-codebase-details`
* **Affected Scope:** `audit_heterogeneity_mechanisms.py` Line 408; `HETEROGENEITY_AUDIT_REPORT.md` Line 45
* **Consolidated Action:** Replace double braces `$D_{{JS}}(P_C \parallel P_{{\text{{pop}}}})$` with standard TeX `$D_{JS}(P_C \parallel P_{\text{pop}})$` and format covariance terms as `\text{Cov}(\text{ZS}, \text{FT} - \text{ZS})`.

#### Finding C-10: Filter Usable Nights in Population Prior Calculation
* **Lenses:** `egm-implementation-readiness`, `egm-codebase-details`
* **Affected Scope:** `audit_heterogeneity_mechanisms.py` Lines 103–106
* **Consolidated Action:** Filter `df_inv[df_inv["usable"] == True]` before averaging demographic stage fractions.

---

## 4. Traceability Matrix

| Raw Finding Source | Primary Lens | Severity | Consolidated Action Item |
|---|---|---|---|
| Stage 5 Finding 1 | `egm-zoom-out` | Major | **Finding C-2**: Add 4-line executive context block |
| Stage 5 Finding 2 | `egm-inventive-catalyst` | Major | **Finding C-1**: Reframe gating to counterfactual analysis & prior regularization |
| Stage 5 Finding 3 | `egm-inventive-catalyst` | Major | **Finding C-1**: Note architectural extraction/decoupling of prior from features |
| Stage 5 Finding 4 | `egm-zoom-out` | Nit | **Finding C-2**: Clarify contrast between static predictors and dynamic shift |
| Stage 6 Finding 1 | `egm-comprehension` | Major | **Finding C-2**: Define `Delta` and Macro-F1 metric in context block |
| Stage 6 Finding 2 | `egm-comprehension` | Major | **Finding C-2**: Define $P_C, P_E, P_{\text{pop}}$ and $D_{JS}$ bounds |
| Stage 6 Finding 3 | `egm-comprehension` | Major | **Finding C-2**: Define `H-A` project acronym |
| Stage 6 Finding 4 | `egm-comprehension` | Major | **Finding C-2**: State responder threshold ($\Delta > 0.0\text{ pp}$) |
| Stage 6 Finding 5 | `egm-comprehension` | Major | **Finding C-4**: Define Oldham formula and baselines |
| Stage 6 Finding 6 | `egm-comprehension` | Major | **Finding C-3**: Document Model A and Stage-D hyperparameter origin |
| Stage 6 Finding 7 | `egm-comprehension` | Major | **Finding C-2**: Defer variable definitions to artifact schemas |
| Stage 7 Finding 1 | `egm-design` | Blocker/Major | **Finding C-1**: Reframe gating; report responder rejection and low sensitivity |
| Stage 7 Finding 2 | `egm-design` | Major | **Finding C-4**: Compute Pearson $r$ for Oldham; calibrate null baseline |
| Stage 7 Finding 3 | `egm-design` | Major | **Finding C-3**: Acknowledge fold-level optimization hyperparameter confounder |
| Stage 7 Finding 4 | `egm-structural` | Major | **Finding C-3**: Consolidate M2, M4, M6 in narrative subsection |
| Stage 7 Finding 5 | `egm-structural` | Major | **Finding C-3**: Group Table 1 by Mechanism 1–6 with hypothesis IDs |
| Stage 7 Finding 6 | `egm-style` | Major | **Finding C-1**: Strip shouting bold and exclamation marks from gating |
| Stage 7 Finding 7 | `egm-style` | Major | **Finding C-4**: Replace "Causal Diagnosis" and "negative infinity" |
| Stage 7 Finding 8 | `egm-structural` | Nit | **Finding C-9**: Normalize TeX formatting and percentage styles |
| Stage 8 Finding 1 | `egm-zoom-out` | Blocker/Major | **Finding C-1**: Counterfactual reframe of Section 3 (anti-symptom-patching) |
| Stage 8 Finding 2 | `egm-zoom-out` | Major | **Finding C-2**: Anti-bloat 4-bullet executive metadata box |
| Stage 8 Finding 3 | `egm-zoom-out` | Major | **Finding C-3**: Consolidate null mechanisms without textbook bloat |
| Stage 8 Finding 4 | `egm-zoom-out` | Major | **Finding C-4**: Statistical and epistemic de-escalation |
| Stage 9 Finding 1 | `egm-readiness` | Major | **Finding C-5**: Seed PyTorch/NumPy RNG for deterministic displacement |
| Stage 9 Finding 2 | `egm-readiness` / `codebase` | Major | **Finding C-6**: Anchor CLI default paths and include `exp019_dir` in `sys.path` |
| Stage 9 Finding 3 | `egm-style` | Major | **Finding C-7**: Add docstring to `run_heterogeneity_audit` and CLI `--help` |
| Stage 9 Finding 4 | `egm-readiness` / `codebase` | Nit | **Finding C-8**: Remove dead code, move `f1_score` to top import |
| Stage 9 Finding 5 | `egm-codebase` | Nit | **Finding C-9**: Fix string-escape double braces in line 408 |
| Stage 9 Finding 6 | `egm-readiness` / `codebase` | Nit | **Finding C-10**: Filter usable nights for population prior |

---

## 5. Concrete Action Plan & Implementation Guidance

### A. Modifications to `experiments/EXP-019/audit_heterogeneity_mechanisms.py`
1. **Paths & Imports:**
   - Add `exp019_dir = os.path.abspath(os.path.dirname(__file__))` to the `sys.path` insertion list.
   - Anchor all default CLI arguments to `os.path.join(repo_root, ...)`.
   - Move `from sklearn.metrics import f1_score` to module-level imports.
2. **Determinism:**
   - Add `torch.manual_seed(42)`, `torch.cuda.manual_seed_all(42)`, and `np.random.seed(42)` at the entry point of `run_heterogeneity_audit()` and before each subject's adaptation.
3. **Statistical Robustness:**
   - Filter `df_inv[df_inv["usable"] == True]` for population prior calculation.
   - Compute and log both Pearson $r$ and Spearman $\rho$ for Oldham's mean.
   - Cast `cal_y_valid.astype(int)` defensively before `np.bincount`.
4. **Code Quality & Docstrings:**
   - Add NumPy/Google-style docstring to `run_heterogeneity_audit()` documenting inputs and outputs.
   - Add descriptive `help=` strings to all `argparse` arguments.
   - Delete dead lines 223–225 (`pd.Series(...).corr(...)`).
   - Fix double curly-brace string formatting in markdown report emission.
   - Reframe generated Section 3 markdown according to Finding C-1.

### B. Modifications to `experiments/EXP-019/HETEROGENEITY_AUDIT_REPORT.md`
1. **Executive Context Box:** Insert the 4-bullet executive context box at the top of the report.
2. **Table 1 Symmetry:** Sub-header Table 1 by Mechanism 1 through 6, displaying explicit hypothesis IDs.
3. **Mechanism Discussion:**
   - Section 2.A: Reclassify `*Causal Diagnosis*` to `**Forensic Case Interpretation (Bidslab34):**` and calibrate emission bias description.
   - Section 2.B: Report Pearson $r$ alongside Spearman $\rho$ for Oldham's mean; clarify that baseline performance was non-significant both before and after adjustment.
   - Section 2.C: Acknowledge fold-level hyperparameter budget confounding within `scope=all`.
   - Section 2.D: Add compact summary of null findings for Mechanisms 2, 4, and 6.
4. **Section 3 Refactoring:** Reframe "Deployment Gating Simulation" into "Stress-Testing Static Pre-Adaptation Gating (Counterfactual Analysis)", disclosing the rejection of responder `Bidslab15`, the $8.3\%$ sensitivity to non-responders, and motivating algorithmic regularization over static client exclusion.
