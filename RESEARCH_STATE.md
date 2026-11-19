# Research State

**Last updated:** 2026-11-18 (Governance Milestone: Master Execution of Research Candidates 1–5 Completed under DEC-029)

---

## 1. Authoritative Research Direction

- **Current Working Document:** `RESEARCH_DIRECTION_v4_FULL.md` (formally supersedes `RESEARCH_DIRECTION_v3_FULL.md`, the v3.1 amendment, and `RESEARCH_DIRECTION_v4_AMENDMENT_PROPOSAL.md`).
- **Research Framing:** A pre-registered, leakage-free replication and extension of multi-night supervised personalization using consumer-smartwatch instantaneous heart rate (IHR, ~0.2 Hz) and accelerometry in healthy adults, with chronological evaluation, specificity controls, calibration-night learning curves, heterogeneity/failure analysis, external validation, and clinical sleep-summary agreement.
- **Withdrawal of Novelty Claim:** The previous framing ("multi-night personalization is an open/underexplored lane") is formally withdrawn following literature discovery of van der Aar et al. 2025 (*Physiological Measurement*, DOI: 10.1088/1361-6579/ae0119), which established multi-night wrist-worn PPG + accelerometry personalization in sleep-disordered adults (82.5% of subjects improved, significant at $\ge 3$ training nights).
- **Core Scientific Question:**
  > How much do two or more labeled earlier nights help consumer-smartwatch IHR and accelerometry sleep staging in healthy adults, who is harmed, and why, under strict chronological, subject-level, leakage-free evaluation?

---

## 2. Contribution Structure & Boundaries (v4)

1. **C1 (Primary):** Pre-registered, leakage-free replication and extension of multi-night supervised personalization with consumer IHR + accelerometry in healthy adults.
   - Specificity control (`FT-other-2`) separating personalization from generic optimization.
   - Calibration efficiency learning curve ($k = 1 \text{ to } 4$).
   - Heterogeneity and failure-mode analysis across subjects.
2. **C2 (Secondary, Exploratory):** Calibration-night class coverage (e.g. stage absence, extreme label shift) as a candidate failure mode, with nested evaluation of regularized or gated adaptation.
3. **C3:** External zero-shot validation (SleepAccel cohort, EXP-027) and clinical sleep-summary agreement (TST, SOL, WASO, SE, REM%, Deep%, EXP-028).

**Claim Boundaries:**
- Must NOT claim "first" multi-night personalization or novelty of supervised fine-tuning.
- Must NOT claim that personalization generally "works" unconditionally (pre-registered H-A was not met; effect is heterogeneous).
- Must NOT claim superiority over SLAMSS-IFS without an independent protocol-matched re-run.
- Must NOT claim clinical-grade accuracy or disorder generalization (cohort is strictly healthy adults).

---

## 3. Dataset & Task Definition (Locked)

- **Dataset:** PhysioNet BIDSleep dataset (Song et al., 2026, DOI: 10.13026/rees-1092; IEEE TBME 2025, DOI: 10.1109/TBME.2025.3612158).
- **Cohort:** 47 healthy adults, 253 recorded nights, 246 usable nights after QC.
- **Signals:** Apple Watch instantaneous heart rate (IHR, ~0.2 Hz, HealthKit) and 3-axis wrist accelerometry, segmented into 30-second epochs.
- **Staging Task:** 4-class classification: Wake (0), Light (1; N1+N2 combined), Deep (2; N3), REM (3). Unknown stages and padded positions mapped to `-100` (`ignore_index`).
- **Context Horizon:** Full-night context ($L = 1200$, sliding-window inference with probability stitching). The historical 5-hour crop ($T = 600$) dropped ~28% of labeled epochs with REM heavily depleted (DEC-021). All cropped results must be reported as cropped.
- **Model Backbone & Parameter Baseline:** Model A (`ConventionalCNNBiGRU`, exactly 163,940 parameters). Under DEC-031, 163,940 parameters is the baseline reference size. Future models should preferentially be lighter than this baseline. Larger models require explicit scientific justification and transparent reporting. Architecture search is permanently closed for completed experiments.
- **Test Set Status:** The historical EXP-012 test set (11 subjects, 58 nights) and early EXP-001/004 test set (7 subjects, 38 nights) are formally **SPENT** (`EXP001_EXP004_SPLIT_LEAKAGE_AUDIT.md`, DEC-029 Section 6). The two test splits represent disjoint cohorts rotated across training and test roles. All final claims rest on 5-fold out-of-fold nested CV across all 47 subjects plus external validation.

---

## 4. Current Experimental State & Hypothesis Status

| ID | Hypothesis / Question | Status | Key Evidence & Governance Standing |
|---|---|---|---|
| **H-A** | Fine-tuning on held-out subject's first 2 labeled nights ($FT\text{-}2$) improves Macro-F1 on later nights vs zero-shot ($ZS$) | **LOCKED (EXP-019)** | $\bar{\Delta} = \mathbf{+1.01\text{ pp}}$ ($64.58\% \to 65.59\%$), 95% bootstrap CI: $\mathbf{[-0.17, +1.99]\text{ pp}}$, Wilcoxon $p = 0.00131$, win rate: 73.9% (34/46). <br>**Governance Reconciled Label:** *INCONCLUSIVE* under the exact locked decision rule (`git show 4dff5b7:PREREG_EXP-019_H-A.md` Section 6: NOT SUPPORTED requires CI upper bound $< +1.50\text{ pp}$; since $+1.99 > +1.50$, it evaluates to INCONCLUSIVE). The EXP-019 report labeled it NOT SUPPORTED. Reconciled via erratum. Numerical results immutable. |
| **S1** | $FT\text{-}2$ outperforms donor control $FT\text{-}other\text{-}2$ (Specificity Control) | **LOCKED (EXP-019)** | **SUPPORTED:** $\bar{\Delta} = \mathbf{+2.73\text{ pp}}$ [95% CI: $+1.05, +4.32\text{ pp}$], $p = 9.0 \times 10^{-6}$. *Nuance:* $FT\text{-}other\text{-}2$ is $-1.72\text{ pp}$ below $ZS$ (harmful), proving fine-tuning on another subject degrades performance and adaptation is subject-specific. |
| **S2** | $FT\text{-}2$ outperforms per-subject model trained from scratch ($PS\text{-}2$) | **LOCKED (EXP-019)** | **SUPPORTED:** $\bar{\Delta} = \mathbf{+20.32\text{ pp}}$ [95% CI: $+17.70, +22.94\text{ pp}$], $p < 10^{-10}$. Cohort pre-training is indispensable. Treated as a weak baseline check, not a major claim. |
| **S3** | Label-free unsupervised InfoMax adaptation ($UA\text{-}2$) outperforms $ZS$ | **LOCKED (EXP-019)**<br>*(Amendment Preregistered)* | $\bar{\Delta} = \mathbf{0.00\text{ pp}}$ identically across all 46 subjects. **UNRESOLVED / INVALIDATED (Implementation No-Op):** Phase A2 forensic audit (`S3_INFOMAX_NOOP_AUDIT_REPORT.md`, DEC-030) confirmed that `compute_wearable_quality_mask` checked `30 <= IHR <= 180` (raw BPM) on standardized z-score IHR ($[-3.95, +9.94]$). Exactly 0 out of 82,762 calibration epochs passed the mask, terminating adaptation on step 0 (0 optimizer steps, 0/163,940 parameters changed, identical evaluation logits/predictions). S3 is scientifically uninformative; it must NOT be interpreted as evidence that unsupervised adaptation fails (nor that it works). Dedicated Phase D amendment preregistered under `PREREG_EXP-019_S3_AMENDMENT.md` (DEC-033; pending user execution authorization). |
| **K** | Learning curve over calibration nights $k = 1, 2, 3, 4$ | **Pre-registered descriptive**<br>*(Phase D Addendum)* | Phase A3 completed (`experiments/EXP-019/exp019_k_learning_curve.csv`). $k=2$ populated with confirmatory metrics ($\bar{\Delta} = +1.01\text{ pp}$); $k \in \{1, 3, 4\}$ explicitly marked `UNEXECUTED_IN_EXP019` ($n \in \{46, 42, 33\}$, $|E_k| \in \{198, 106, 64\}$). Formal Phase D confirmatory protocol preregistered under `PREREG_EXP-019_K_CURVE.md` (DEC-033; $N=33$ common-cohort on $E_4$, Page's $L$-trend test, Holm-Bonferroni correction; pending user execution authorization). |
| **EXP-021** | SLAMSS-IFS Matched 5-Fold Cross-Validation Paired Comparative Benchmark | **DEFERRED (DEC-034)** | Evaluates lightweight Model A (163,940 params) vs. official SLAMSS-IFS (5.47M repo; empirically measured peak VRAM: 5.12 GiB reserved at $B=4$ on RTX 3060). Under DEC-034, TSK-08 is formally deferred until the project's final model is developed and frozen, to be executed exactly once as the definitive matched benchmark. |
| **H-C** | Context scaling: $L=600$ beats $L=60$ | **EXP-018 Phase 2** | $+5.33\text{ pp}$ Macro-F1 [95% CI: $+4.28, +6.39\text{ pp}$], 46/47 subjects. Counts as confirmatory if pre-decision commit verified (`TO VERIFY`); exploratory otherwise. |
| **EXP-019B** | Heterogeneity and Failure Mechanisms | **Exploratory** | Diagnostic audit completed (`HETEROGENEITY_AUDIT_REPORT.md`). Causal DAG separates root causes (M1: stage absence, M2: autonomic drift) from mediators (M3: calibration loss, M5: parameter drift). Bidslab34 identified with 0.00% deep sleep on Night 2. |
| **EXP-020** | Regularized / Gated Adaptation | **Exploratory (Gated)** | Strictly gated under Phase D1. Requires Phase A, B, C prerequisites and committed preregistration before launch. |
| **EXP-027** | External Zero-Shot Validation (SleepAccel) | **Pre-registered** | Planned under Phase D3. |
| **EXP-028** | Clinical Sleep-Summary Agreement | **Pre-registered** | Phase A3 extracted baseline agreement (`exp019_sleep_summary_agreement.csv`) with symmetric artifact masking. Planned under Phase D4 for external validation. |

---

## 5. Work Plan & Phase Execution Structure (v4)

- **Phase A — Close out EXP-019 (COMPLETED):**
  - **A1:** Reconcile H-A verdict label against locked preregistration decision rule via erratum (`INCONCLUSIVE`).
  - **A2:** S3 no-op audit completed (`S3_INFOMAX_NOOP_AUDIT_REPORT.md`, DEC-030).
  - **A3:** Extract and report missing pre-registered outputs: 6 artifacts materialized and verified in `experiments/EXP-019/` (`PHASE_A3_DIAGNOSTIC_REPORT.md`, $k$-curve, confusion matrices, per-stage F1, Bland-Altman agreement, night index breakdown).
  - **A4:** Update S1 documentation to clarify donor-night fine-tuning harm.
  - **A5:** Complete NAS Iteration 1 erratum (confirming architecture search is closed).
- **Phase B — Evidence Base, Literature Intelligence & EXP-019 Sleep-Summary Agreement (RESOLVED & CLEARED):**
  - **B1 (COMPLETE):** Apply KB round-2 corrections (Heremans PMID `35508121` across 6 files, P011/P009 downgraded to MODERATE, P101 author Colin Quirk and text fixed, gating labels clarified, Lorenzen verified; `updates/2026-11-18_phase_b_and_c_execution.md`).
  - **B2 (PARTIALLY COMPLETE):** Systematic search executed and logged across PubMed, Europe PMC, arXiv, and medRxiv/bioRxiv tracking 4 query blocks and forward citations of Song 2026, Phan 2020, Mikkelsen 2018, and van der Aar 2025 (`search_log.md`). IEEE Xplore and Scopus/Web of Science coverage bounded by institutional subscription credentials.
  - **B3 & Exit (PARTIALLY COMPLETE):** Assessed 7 core papers; full text verified for 5 papers (`P001`, `P002`, `P006`, `P056`, `P106`), while `P007` (IEEE) and `P054` (IOP) remain abstract-verified due to paywall boundaries. "Closest ten" comparison matrix and dated novelty statement produced without priority claims, retaining the C1/C2/C3 hierarchy (`updates/2026-11-18_phase_b_and_c_execution.md`).
  - **B4 — EXP-019 Sleep-Summary Agreement Analysis (COMPLETE, RESOLVED & RATIFIED under DEC-032):**
    - Executed reproducible analysis script `research_engine/bidsleep_within_cohort_agreement_analysis.py` on `exp019_sleep_summary_agreement.csv` (SHA-256: `41a2d6af...`).
    - Established primary agreement on subject-averaged multi-night participant means ($N=46$ independent participants, $df=45$), resolving repeated-measures clustering by construction (Bland & Altman 1999/2007).
    - **FIND-B-01 (RESOLVED):** Explicitly distinguished multi-night participant-average limits of agreement (e.g. TST $[-30.11, +34.67]\text{ min}$) from single-night spot agreement bounds (which are broader due to unmodeled within-subject variance $\sigma_w^2$).
    - **FIND-B-02 (RESOLVED):** Formally documented the five analyses blocked by aggregate-only format (variance components, linear mixed-effects proportional bias, heteroscedasticity rank tests, participant-level cluster bootstrap, outlier leverage diagnostics).
    - **FIND-B-03 (RESOLVED IN PHASE 3):** Incorporated LoA standard errors ($\text{SE}(\text{LoA})$) and 95% confidence intervals on LoA bounds into manuscript Table 1 in Section 6.2 per Bland & Altman (1999) and *Lancet Digital Health* reporting guidelines.
    - **FIND-B-04 (RESOLVED):** Retracted causal sensor attribution for WASO underestimation; confirmed empirical model underestimation ($-4.87\text{ min}$, $p=0.0104$) which may stem from scoring thresholds, loss class weights, or feature representations.
    - **Additional Methodological Correction (RESOLVED):** Clarified that the unadjusted pooled-night interval ($[-39.60, +46.95]\text{ min}$ for TST) is an unadjusted descriptive summary, NOT a validated single-night prediction interval.
    - **FIND-B-05 & FIND-B-06 (VERIFIED):** 100% numerical reproduction of draft manuscript Table 1 verified; denominators ($N=46$ subjects, $N=152$ nights) and anti-leakage ($C \cap E = \emptyset$) intact.
    - Independent review completed: `PHASE_B_INDEPENDENT_AUDIT_REPORT.md` (Verdict: PASS WITH CORRECTIONS; CLEARED FOR PHASE 3).

- **Phase C — Data Search (BOUNDED COMPLETE / DECISION GATE ENFORCED):**
  - **C1 (BOUNDED COMPLETE):** Systematic search for a second multi-night labeled wearable dataset ($\ge 3$ labeled nights/subject, wrist PPG/IHR + accelerometry, EEG/PSG ground truth). Census across PhysioNet, NSRR, OpenNeuro, and literature confirmed no qualifying second dataset exists among surveyed sources (van der Aar 2025 cohort is restricted/private; SleepAccel and DREAMT are 1 night/subject). Decision gate enforced: single-dataset paper framed as replication, extension, and failure analysis; limitation reported openly.
- **Phase D — Experiments (Ratified Governance Framework under DEC-033 & DEC-034):**
  - **Stage 1 (Immediate Contributions C1 & C2 Closure — Active Focus):**
    * **Experiment 1 (TSK-06 / EXP-019-K):** Calibration-depth learning curve ($k = 1, 3, 4$) under `PREREG_EXP-019_K_CURVE.md` (Common-cohort $N=33$ on $E_4$, Page's $L$-test, Holm-Bonferroni correction). Status: *ACTIVE / PRIORITIZED FOR EXECUTION*.
    * **Experiment 2 (TSK-07 / EXP-019-S3):** Unsupervised adaptation re-run ($UA\text{-}2$) under `PREREG_EXP-019_S3_AMENDMENT.md` (DEC-030 unit fix, pre-flight gate, InfoMax). Status: *ACTIVE / PRIORITIZED FOR EXECUTION*.
  - **Stage 2 (Final Model Synthesis & Freezing):**
    * Synthesize findings across TSK-06, TSK-07, context scaling (EXP-018), and agreement (EXP-019); develop, validate, and freeze final project model.
  - **Stage 3 (Definitive Matched Baseline Benchmark — Run Once):**
    * **Experiment 3 (TSK-08 / EXP-021):** SLAMSS-IFS matched 5-fold cross-validation paired comparative benchmark under `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md`. Status: *DEFERRED PENDING FINAL MODEL FREEZE (DEC-034)*.
  - **Deferred Exploratory Tasks:** EXP-019B (heterogeneity tests), EXP-020 (gated adaptation), and EXP-027 (SleepAccel transfer) remain deferred under DEC-029.
- **Phase E — Synthesis & Paper Writing (COMPLETE FOR DRAFT v0.1):**
  - **Candidate 1 (COMPLETE):** Historical split and data leakage audit across EXP-001 to EXP-004 (`EXP001_EXP004_SPLIT_LEAKAGE_AUDIT.md`); test access events TA-01 to TA-06 formally recorded in `EXPERIMENTS.md`.
  - **Candidate 2 (COMPLETE):** SLAMSS-IFS scientific baseline audit (`SLAMSS_IFS_BASELINE_PROTOCOL_AUDIT.md`); resolved 10-dimension comparison matrix, parameter discrepancy (666k paper-text vs 5.47M repo vs 163.9k Model A), and holdout variance vs 5-fold CV.
  - **Candidate 3 (COMPLETE):** Design-informed exploratory preregistration drafts authored with prior-access disclosures: EXP-019B (`EXP019B_PREREGISTRATION_DRAFT.md`, SHA-256: `30ca80f1...`) and EXP-020 (`EXP020_PREREGISTRATION_DRAFT.md`, SHA-256: `c226cc41...`).
  - **Candidate 4 (COMPLETE / AUDITED):** BIDSleep cohort data and signal integrity audit (`BIDSLEEP_COHORT_DATA_INTEGRITY_AUDIT.md`); reconciled 47 vs 46 subject denominators, 7 excluded nights across 6 subjects, HealthKit IHR effective sampling rate (median 0.197 Hz, IQR 0.073 Hz), `Bidslab34` calibration Deep sleep scarcity (0.68%, 15 epochs), and evaluable cohort denominators across $k=1..4$ ($47, 46, 42, 33$).
  - **Candidate 5 / Phase 3 Polish (COMPLETE):** Full working manuscript draft Sections 1–8 authored (`MANUSCRIPT_DRAFT_v0.1.md`) with evidence-to-claim matrix (`MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv` claims CLM-001 through CLM-040) and open unresolved evidence register (`MANUSCRIPT_UNRESOLVED_EVIDENCE.md`). Resolved FIND-B-01 (multi-night LoA scope), FIND-B-02 (micro-data information loss), FIND-B-03 (LoA uncertainty bounds in Table 1), and FIND-B-04 (empirical WASO model underestimation).
  - Write manuscript adhering strictly to pre-registered rules and honest negative/inconclusive reporting.

---

## 6. Strict Governance Guardrails

1. **Immutable Preregistration & Data**: EXP-019 preregistration (`4dff5b7`), numerical metrics, and stored checkpoints remain strictly frozen.
2. **No Premature Training**: Do NOT launch EXP-020 or any adaptation training until Phase A and B prerequisites are completed and user approval is obtained under a committed preregistration block.
3. **No Architecture Search**: Model A (`ConventionalCNNBiGRU`, 163,940 params) is frozen. NAS is permanently terminated.
4. **Leakage Protection**: All evaluations enforce strict subject-level disjointness and chronological calibration/evaluation splits.
5. **Continuous Git Synchronization**: All documentation updates, errata, and audit scripts are committed and pushed to `origin main`.
6. **Parameter-Count Governance (DEC-031)**: The historical +20% (196,728) cap is removed. 163,940 parameters is the baseline reference size. Future models should preferentially be lighter than this baseline; larger models require explicit scientific justification and transparent reporting.
