# Stage 6 Comprehension Test: EXP-018 Phase 1 Final Report

**Target Document**: [`experiments/EXP-018/EXP018_PHASE1_REPORT.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-018/EXP018_PHASE1_REPORT.md) (Lines 1–88)  
**Auditor Persona**: Naive Reader / Comprehension Goldfish (`egm-comprehension`)  
**Standard**: Self-Containment & Zero Assumed Institutional Knowledge  

---

## High-Altitude Paraphrase

This document reports the empirical outcome of Phase 1 of experiment EXP-018, which evaluates how varying temporal context windows ($L$, ranging from a single 30-second epoch up to 10 hours) impacts neural-network-based sleep stage classification from wearable signals. Because expanding context from 30 minutes ($L=60$) to 5 hours ($L=600$) produced a statistically credible performance improvement (+4.27 percentage points in Macro-F1), the project has declared context benefit supported, formally authorizing progression to multi-fold cross-validation (Phase 2) and an optional circadian ablation study (Phase 1b).

---

## Component / Structural Paraphrase

Across 21 experimental runs (7 context lengths $\times$ 3 fixed random seeds), an unnamed deep learning architecture ("Model A", 163,940 parameters) was trained under a fixed compute invariant ($B \times L \times N_{\text{accum}} = 2,400$) while keeping 11 validation/test subjects strictly locked. Classification fidelity was tracked using standard statistical metrics (Macro-F1, Accuracy, Cohen's $\kappa$, MCC, and individual stage F1 scores) alongside clinical polysomnography error metrics (TST, SOL, WASO, and Sleep Efficiency MAE). To characterize how performance scales with horizon length, the authors fitted a 3-parameter exponential saturation curve ($F_1(L) = A - B e^{-k L}$) estimating saturation threshold and asymptotic bounds, and linked JSON artifact records for all individual runs.

---

## Comprehension Findings

### 1. Paradoxical Asymptotic Fit Ceiling Below Observed Data
- **Severity**: **Major**
- **Affected Scope**: Section 3 (Lines 35–41, `Parametric Saturation Curve Fit`)
- **Discrepancy Description & Rationale**: The document reports a fitted saturation curve $F_1(L) = A - B e^{-k L}$ with an **Asymptotic Ceiling ($A$) of 0.6435**. However, in Section 2, the empirical Macro-F1 at $L=600$ is **0.6529** and at $L=1200$ is **0.6993**, both well above the supposed theoretical ceiling ($A$). To a fresh reader, a model asymptotic maximum that is lower than the actual observed performance creates severe cognitive dissonance and calls into question the validity of the reported $R^2 = 0.9258$ and the 95% saturation threshold ($L^* = 46.3$ epochs). The document must explain whether the fit was restricted to a subset of horizons (e.g., $L \le 300$) or why the data supersedes the asymptotic model.

### 2. Undefined Architectural Identity ("Model A")
- **Severity**: **Major**
- **Affected Scope**: Section 6 (Line 85, `Model A Parameter Count`)
- **Discrepancy Description & Rationale**: Section 6 states "Model A Parameter Count: 163,940 parameters verified across all 21 runs," but "Model A" is never defined inline, accompanied by an architectural description (e.g., CNN-BiGRU baseline), or hyperlinked to a specification file or codebase module. A reader joining the project cannot determine what model architecture was evaluated.

### 3. Undefined Gating Thresholds and Unlinked Downstream Phases
- **Severity**: **Major**
- **Affected Scope**: Section 1 (Lines 10–17, `Executive Summary & Gating Decision`)
- **Discrepancy Description & Rationale**: The executive summary declares a frozen decision category of `SUPPORTED (Context Benefit)` and authorizes `Phase 2 CV` and `Phase 1b Circadian Ablation`. However, the document does not define the pre-registered decision criteria or quantitative threshold required to trigger `SUPPORTED` versus other outcomes. Furthermore, neither `Phase 2 CV` nor `Phase 1b Circadian Ablation` is defined or linked to an experiment plan explaining what those phases do.

### 4. Unexpanded Clinical Polysomnography Acronyms
- **Severity**: **Nit**
- **Affected Scope**: Section 4 (Lines 43–54, `Clinical Polysomnography Metrics`)
- **Discrepancy Description & Rationale**: The table uses clinical sleep staging acronyms: **TST**, **SOL**, and **WASO**. While standard in clinical polysomnography (Total Sleep Time, Sleep Onset Latency, Wake After Sleep Onset), they are domain-specific acronyms that are not defined or expanded anywhere in the document.

### 5. Informal Compute Invariant and Token Terminology
- **Severity**: **Nit**
- **Affected Scope**: Section 6 (Line 86, `Compute Invariant`)
- **Discrepancy Description & Rationale**: The invariant $B \times L \times N_{\text{accum}} = 2,400$ introduces variables $B$ and $N_{\text{accum}}$ without explicit definitions (batch size and gradient accumulation steps). Additionally, the phrase "7,680,000 total tokens per run" borrows NLP terminology without stating whether 1 "token" corresponds to one 30-second epoch or a raw sensor time-step.
