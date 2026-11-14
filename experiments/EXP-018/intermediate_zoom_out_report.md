# Stage 8 Intermediate Zoom Out Review: EXP-018 Phase 1 Report

**Target Document**: [`experiments/EXP-018/EXP018_PHASE1_REPORT.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/EXP018_PHASE1_REPORT.md)  
**Auditor**: Skeptical Architect (`egm-zoom-out`)  
**Input Context**: Stage 5 Zoom Out (`zoom_out_report.md`), Stage 6 Comprehension (`comprehension_report.md`), Stage 7 Critic (`critic_report.md`), Protocol Spec ([`design_draft.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/design_draft.md)), and Raw Summary ([`phase1_summary.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/phase1_results/phase1_summary.json)).  
**Review Status**: 🟡 **CONDITIONAL PASS — REVISIONS REQUIRED BEFORE FINALIZATION**

---

## 1. Cross-Stage Synthesis (Zoom Out, Comprehension, Critic)

A thorough synthesis across all three review stages reveals complete, unanimous convergence on the core architectural, mathematical, and reporting tensions in `EXP018_PHASE1_REPORT.md`:

1. **The Parametric Saturation Paradox (Unanimous Core Finding)**:
   - *Stage 5 (`ZOOM-5-1`), Stage 6 (Finding 1), Stage 7 (`CRIT-7-1`)*: Reporting an asymptotic ceiling of $A=0.6435$ while empirical performance reaches $0.6529$ ($L=600$) and $0.6993$ ($L=1200$) is mathematically incoherent on its face. The least-squares single-exponential model latched onto the rapid transition rise ($L=1 \to 60$) and mistook the intermediate plateau ($L=60 \to 120$) for final saturation ($L^* \approx 46.3$ epochs / 23.1 min). Sleep physiology operates across two distinct temporal regimes: local epoch smoothing ($\tau_1 \approx 10\text{--}30\text{ min}$) and multi-hour ultradian NREM-REM cycles ($\tau_2 \approx 90\text{--}120\text{ min}$).
2. **Cohort Scope & Population Overclaiming (Unanimous)**:
   - *Stage 5 (`ZOOM-5-2`), Stage 6 (Finding 3), Stage 7 (`CRIT-7-3`)*: Section 1 declared broad population authorization without explicitly qualifying that Phase 1 was evaluated on the 5-subject discovery validation set ($N=5$, 28 nights). While the $+4.27\text{ pp}$ gain and 95% CI $[+2.53, +6.46\text{ pp}]$ decisively clear the pre-registered gating rule ($\Delta \ge +1.0\text{ pp}$, $\text{CI}_{\text{lower}} > 0$), Section 1 must contextualize this as a validation gate authorizing the true 47-subject Phase 2 CV.
3. **Missing Architectural Identity & Acronyms (Comprehension & Critic)**:
   - *Stage 6 (Findings 2, 4), Stage 7 (`CRIT-7-7`)*: "Model A" was introduced without definition, and clinical metrics (TST, SOL, WASO, SE) and statistical terms (MCC, CV) lacked expansions.
4. **Clinical PSG Metric Dissociation & Inference Confound (Critic & Zoom Out)**:
   - *Stage 5 (`ZOOM-5-3`), Stage 7 (`CRIT-7-2`, `CRIT-7-4`)*: Macro-F1 improves from $L=60$ to $L=600$, yet TST error degrades ($31.04\text{m} \to 36.61\text{m}$) due to intermediate low-pass smoothing of micro-arousals. Furthermore, $L=1200$ bypasses window margin discarding (Case 1 single-pass), creating an operational asymmetry relative to $L \le 600$ (Case 2 sliding window).
5. **Git Portability & Table Hygiene (Critic)**:
   - *Stage 7 (`CRIT-7-5`, `CRIT-7-6`, `CRIT-7-8`, `CRIT-7-13`)*: Machine-specific absolute URIs (`file:///home/rj2826/...`), missing SDs in Table 2, redundant unit suffixes (`m`, `%`, `s`), and center-aligned numeric columns.

---

## 2. Residual Architectural and Scientific Risks Before Phase 2

1. **YAGNI on Complex Curve Re-Fitting**:
   - *Risk*: Implementing complex bi-exponential or piecewise curve-fitting algorithms in Python right now would be premature over-engineering.
   - *Architectural Guidance*: Keep the codebase lean. For the Phase 1 report, simply annotate Section 3 as a single-scale baseline fit that captures local epoch inertia ($L \le 60$), explaining that multi-hour ultradian dynamics supersede this asymptote. Defer multi-scale curve fitting until the complete 47-subject Phase 2 dataset is collected.
2. **Phase 2 Compute Budget & Architectural Invariant**:
   - *Risk*: Phase 2 involves 105 runs ($7 L \times 5 \text{ folds} \times 3 \text{ seeds}$). At $\approx 3$ minutes per run, total compute is $\approx 5.25$ GPU hours on `cuda:1`.
   - *Architectural Guidance*: The pre-registered gating threshold was passed cleanly ($\Delta = +4.27\text{ pp}$ vs gate of $+1.0\text{ pp}$), and all 5 individual subjects exhibited positive deltas ($+1.51$ to $+8.33\text{ pp}$). The fixed compute invariant ($B \times L \times N_{\text{accum}} = 2,400$) held perfectly across all 21 runs. Phase 2 execution is low-risk and structurally sound.
3. **Inference Confound Transparency**:
   - *Risk*: In Phase 2, if $L=1200$ again shows a leap over $L=600$, reviewers may question whether this is physiological or an artifact of Case 1 (whole-night single forward pass) vs Case 2 (sliding-window margin slicing).
   - *Architectural Guidance*: Explicitly pre-register in the report that Phase 2 will analyze whether boundary stitching loss at $L=600$ contributes to the performance delta.

---

## 3. Categorized Actionable Findings for Report Finalization

### 🔴 Blocker (Must fix before finalizing report)

- **Severity**: **Blocker**
- **Affected Scope**: Section 3 (Lines 33–41, `Parametric Saturation Curve Fit`)
- **Simplicity & Architectural Rationale**:
  Presenting an asymptotic ceiling $A = 0.6435$ alongside observed empirical performance of $0.6529$ ($L=600$) and $0.6993$ ($L=1200$) creates an immediate mathematical contradiction that destroys scientific credibility. 
  *Required Fix*: Retitle Section 3 to `Parametric Saturation Curve Fit (Single-Scale Baseline)`. Add a concise physiological note explaining that the single-exponential fit captures the primary local transition inertia regime ($L^* = 46.3$ epochs / 23.1 min), while multi-hour ultradian NREM-REM cycles produce a second scaling regime ($L \ge 300$) that breaks through this single-scale asymptote.

---

### 🟡 Major (Core improvements for clarity and scientific accuracy)

1. **Explicit Validation Cohort Scope & Subject Consistency**:
   - **Severity**: **Major**
   - **Affected Scope**: Section 1 (Lines 10–18, `Executive Summary & Gating Decision`)
   - **Rationale**: Prevent overclaiming population generalization from an $N=5$ sample.
   - *Required Fix*: Explicitly state: `Evaluated on Phase 1 Validation Partition (N=5 subjects, 28 nights; split hash bf33573623f47f3f)`. State the pre-registered gating criterion ($\Delta \ge +1.0\text{ pp}$, $\text{CI}_{\text{lower}} > 0$). Cite the 5 individual subject deltas ($+1.51$ to $+8.33\text{ pp}$, 100% positive concordance) to demonstrate that the $+4.27\text{ pp}$ gain is uniform across subjects.
2. **Clinical PSG Metric Dissociation & Inference Confound**:
   - **Severity**: **Major**
   - **Affected Scope**: Section 4 (Lines 43–54, `Clinical Polysomnography Metrics`) & Section 2
   - **Rationale**: A reader needs to know why TST MAE worsens from $L=60$ ($31.04\text{m}$) to $L=600$ ($36.61\text{m}$) even as Macro-F1 improves, and why $L=1200$ behaves differently.
   - *Required Fix*: Add a brief physiological note in Section 4 explaining that intermediate BiGRU context functions as a low-pass filter, over-smoothing micro-arousals (Wake recall drops from 50.8% to 46.4%) and inflating TST error, until full-night boundary anchors take effect at $L=1200$. In Section 2, briefly note that $L=1200$ evaluates via Case 1 (whole-night single pass without margin truncation).
3. **Define Architectural Identity & Acronyms**:
   - **Severity**: **Major**
   - **Affected Scope**: Section 6 (Line 85) & Section 4
   - **Rationale**: Self-containment. A new collaborator cannot identify "Model A" or domain acronyms.
   - *Required Fix*: Define Model A as `ConventionalCNNBiGRU` (163,940 parameters; [`exp013_models.py`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/exp013_models.py#L69-L72)). Expand TST (Total Sleep Time), SOL (Sleep Onset Latency), WASO (Wake After Sleep Onset), SE (Sleep Efficiency), MCC (Matthews Correlation Coefficient), and CV (Cross-Validation).
4. **Repository-Relative Artifact Links**:
   - **Severity**: **Major**
   - **Affected Scope**: Section 5 (Lines 58–80, Table 5)
   - **Rationale**: Local machine paths (`file:///home/rj2826/...`) break on GitHub and across collaborator environments.
   - *Required Fix*: Convert all 21 links to repo-relative paths (`phase1_results/run_LXXXX_seedYY.json`).
5. **Complete Dispersion Reporting in Table 2**:
   - **Severity**: **Major**
   - **Affected Scope**: Section 2 (Lines 20–30, Table 2)
   - **Rationale**: Section title promises `(Mean ± SD across 3 Seeds)`, but Accuracy, $\kappa$, and MCC omit SD.
   - *Required Fix*: Add `± SD` for Accuracy, Cohen's $\kappa$, and MCC using the verified figures from `phase1_summary.json`.

---

### 🟢 Nit (Presentation & formatting polish)

1. **Table Column Numeric Alignment & Redundant Unit Suffixes**:
   - **Affected Scope**: Sections 2, 4, 5
   - *Fix*: Right-align numeric columns (`---:`) and strip repetitive unit characters (`m`, `%`, `s`) from data cells when already specified in column headers.
2. **Subject-Paired (+4.27 pp) vs Marginal Difference (+4.12 pp)**:
   - **Affected Scope**: Section 1 (Line 13)
   - *Fix*: Add a 1-line clarification that $+4.27\text{ pp}$ is the subject-paired mean delta (equal subject weighting), whereas $+4.12\text{ pp}$ is the marginal difference of pooled means.
3. **Consolidate Redundant Authorization Line**:
   - **Affected Scope**: Section 1 (Lines 15–16)
   - *Fix*: Remove the redundant repetition of "Phase 2 CV is AUTHORIZED".
4. **Calibrate Governance Tone**:
   - **Affected Scope**: Section 6 (Line 87)
   - *Fix*: Replace "strictly LOCKED" with objective technical prose: "11 held-out test subjects unaccessed; zero test evaluations performed."

---

## 4. Final Verdict & Recommendation

The underlying research and empirical findings are sound, mathematically verified against `phase1_summary.json`, and satisfy the pre-registered gating threshold authorizing Phase 2. However, **`EXP018_PHASE1_REPORT.md` is not ready to be finalized in its current form**. 

Applying the specific, non-disruptive revisions above will resolve the saturation paradox, restore mathematical coherence, ensure full self-containment, and provide an unassailable scientific foundation before Phase 2 CV is launched.
