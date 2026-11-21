# Research Handoff

## Current Task

**EGM RESEARCH CONTEXT SYNCHRONIZATION: RESEARCH DIRECTION v4 & REVISED KNOWLEDGE BASE**

Following explicit user authorization per the KB Governance Rule, the workspace context is formally synchronized to **`RESEARCH_DIRECTION_v4_FULL.md`**, superseding `RESEARCH_DIRECTION_v3_FULL.md`, the v3.1 amendment, and `RESEARCH_DIRECTION_v4_AMENDMENT_PROPOSAL.md`.

---

## 1. Key Direction Changes (v4 Summary)

1. **Reframed Research Position**:
   - The prior assumption that "multi-night personalization is an open lane" is formally withdrawn following the identification of van der Aar et al. 2025 (*Physiological Measurement*, DOI: 10.1088/1361-6579/ae0119), which demonstrated multi-night wrist PPG + accelerometry personalization in $n=59$ adults across up to 7 home nights (82.5% improved; significant at $\ge 3$ nights).
   - The research is now positioned as:
     > A pre-registered, leakage-free replication and extension of multi-night supervised personalization using consumer-smartwatch instantaneous heart rate (IHR, ~0.2 Hz) and accelerometry in healthy adults, with chronological evaluation, specificity controls, calibration-night learning curves, heterogeneity/failure analysis, external validation, and clinical sleep-summary agreement.
2. **Ratified Contribution Structure**:
   - **C1 (Primary)**: Pre-registered, leakage-free replication and extension with consumer IHR + accelerometry in healthy adults (FT-other specificity control, $k=1–4$ learning curve, heterogeneity analysis).
   - **C2 (Secondary, Exploratory)**: Calibration-night class coverage as a candidate failure mode, with nested evaluation of regularized or gated adaptation.
   - **C3**: External zero-shot validation (EXP-027) and clinical sleep-summary agreement (EXP-028).
3. **Primary Question**:
   > How much do two or more labeled earlier nights help consumer-smartwatch IHR and accelerometry sleep staging in healthy adults, who is harmed, and why, under strict chronological, subject-level, leakage-free evaluation?

---

## 2. Experimental Status & Governance Reconciliation (EXP-019)

All EXP-019 numerical results, split manifests, model checkpoints, and preregistration files remain strictly immutable.

| Arm / Hypothesis | Estimand & Metric | Status & Reconciled Interpretation |
|---|---|---|
| **H-A ($FT\text{-}2$ vs $ZS$)** | $\bar{\Delta} = \mathbf{+1.01\text{ pp}}$ [95% CI: $\mathbf{-0.17, +1.99\text{ pp}}$], $p = 0.00131$, win rate 73.9% (34/46) | **LOCKED (Reconciled Governance Label: INCONCLUSIVE)**<br>• Section 6 of `git show 4dff5b7:PREREG_EXP-019_H-A.md` defines SUPPORTED ($\bar{\Delta} \ge +1.50$ & CI_low > 0), NOT SUPPORTED ($\text{CI}_{\text{high}} < +1.50$), and INCONCLUSIVE (otherwise).<br>• Because the 95% CI upper bound (+1.99 pp) is NOT $< +1.50\text{ pp}$, the preregistered rule evaluates strictly to **INCONCLUSIVE**, whereas the report labeled it NOT SUPPORTED.<br>• Reconciled via erratum. Safe summary: "Supervised 2-night adaptation did not reach the pre-registered practical threshold (+1.50 pp) and is inconclusive for effects up to ~2 pp." |
| **S1 ($FT\text{-}2$ vs $FT\text{-}other\text{-}2$)** | $\bar{\Delta} = \mathbf{+2.73\text{ pp}}$ [95% CI: $\mathbf{+1.05, +4.32\text{ pp}}$], $p = 9.0 \times 10^{-6}$ | **SUPPORTED (Specificity Confirmed)**<br>• Fine-tuning on a person's own nights outperforms fine-tuning on another person's nights.<br>• *Critical nuance:* $FT\text{-}other\text{-}2$ degrades performance by $-1.72\text{ pp}$ below $ZS$ ($62.86\%$ vs $64.58\%$). Much of the $+2.73\text{ pp}$ gap reflects donor fine-tuning harm. |
| **S2 ($FT\text{-}2$ vs $PS\text{-}2$)** | $\bar{\Delta} = \mathbf{+20.32\text{ pp}}$ [95% CI: $\mathbf{+17.70, +22.94\text{ pp}}$], $p < 10^{-10}$ | **SUPPORTED (Cohort Foundation Indispensable)**<br>• Per-subject scratch training achieves only $45.27\%$ Macro-F1. Treated as a baseline check. |
| **S3 ($UA\text{-}2$ vs $ZS$)** | $\bar{\Delta} = \mathbf{+0.21\text{ pp}}$ [95% CI: $\mathbf{+0.04, +0.39\text{ pp}}$], $p = 0.0322$, win rate 58.7% (27/46) | **EXECUTED / RESOLVED (TSK-07 / DEC-036)**<br>• Re-run with corrected wearable quality mask ($\text{Var}_t(m) > 10^{-6}$, 97.26% valid epochs) resolved the DEC-030 implementation no-op.<br>• Officially classified as **SUB-BENCHMARK GAIN** under `PREREG_EXP-019_S3_AMENDMENT.md` Section 6.<br>• Proves label-free InfoMax yields genuine positive transfer without representation collapse, but magnitude is insufficient on its own for the $+1.00\text{ pp}$ benchmark. |
| **$k=1–4$ Learning Curve** | Monotonic depth scaling across $k \in \{1, 2, 3, 4\}$ | **EXECUTED / RESOLVED (TSK-06 / DEC-035)**<br>• Common cohort ($N=33$, $E_4$): $k=1$ (+3.00 pp), $k=2$ (+3.13 pp), $k=3$ (+3.54 pp), $k=4$ (+3.96 pp).<br>• Page's $L$-trend test: $L = 863.0, z = 2.261, p = 0.01187$ (**Holm-Bonferroni Significant**).<br>• van der Aar et al. (2025) 3-night replication confirmed ($\Delta = +3.54\text{ pp}, p = 0.00030$). |

---

## 3. Work Plan Phasing (v4)

- **Phase A: Close out EXP-019 (COMPLETED)**
  - A1: Settle H-A verdict label against locked preregistration decision rule via formal erratum (`INCONCLUSIVE`).
  - A2: S3 no-op audit completed (`experiments/EXP-019/S3_INFOMAX_NOOP_AUDIT_REPORT.md`, DEC-030).
  - A3: Diagnostic extraction completed (`experiments/EXP-019/PHASE_A3_DIAGNOSTIC_REPORT.md` + 5 CSV/JSON deliverables).
  - A4: S1 documentation updated with donor fine-tuning harm clarification.
  - A5: NAS Iteration 1 erratum completed (architecture search permanently closed).
- **Phase B: Evidence Base, Literature Intelligence & EXP-019 Sleep-Summary Agreement (RESOLVED & RATIFIED under DEC-032)**
  - B1: KB round-2 corrections applied and verified (Heremans PMID `35508121` across 6 files, P011/P009 downgraded to MODERATE, P101 author Colin Quirk and text fixed, gating labels clarified, Lorenzen verified). **Status: COMPLETE.**
  - B2: Systematic search executed and logged across PubMed, Europe PMC, arXiv, and medRxiv/bioRxiv tracking 4 query blocks and forward citations (`search_log.md`). IEEE Xplore and Scopus/WoS bounded by institutional access. **Status: PARTIALLY COMPLETE.**
  - B3 & Exit: Core papers assessed; 5 verified via full text (`P001`, `P002`, `P006`, `P056`, `P106`), 2 verified via abstract/excerpts under paywalls (`P007`, `P054`). "Closest ten" comparison matrix and dated novelty statement produced without priority claims (`research_knowledge/updates/2026-11-18_phase_b_and_c_execution.md`). **Status: PARTIALLY COMPLETE.**
  - B4: EXP-019 Sleep-Summary Agreement Analysis & Methodological Evaluation (`EXP019_WITHIN_COHORT_SLEEP_SUMMARY_AGREEMENT_REPORT.md`, `research_engine/bidsleep_within_cohort_agreement_analysis.py`).
    - Subject-averaging resolved repeated-measures clustering by construction ($N=46$ independent participants, $df=45$).
    - Independent audit completed: `PHASE_B_INDEPENDENT_AUDIT_REPORT.md` (Verdict: PASS WITH CORRECTIONS; CLEARED FOR PHASE 3).
    - All audit findings formally resolved and ratified under DEC-032:
      * FIND-B-01 (RESOLVED): Explicitly distinguished multi-night participant-average limits of agreement from single-night agreement.
      * FIND-B-02 (RESOLVED): Documented the inventory of five analyses blocked without subject-night microdata.
      * FIND-B-03 (RESOLVED IN PHASE 3): LoA standard errors and 95% confidence intervals added to manuscript Table 1 in Section 6.2 per Bland & Altman (1999).
      * FIND-B-04 (RESOLVED): Retracted unsupported causal sensor claims regarding WASO underestimation; confirmed empirical model underestimation.
      * Additional Methodological Correction (RESOLVED): Prohibited conflating unadjusted pooled nights ($N=152$) with single-night prediction intervals.
      * FIND-B-05 & FIND-B-06 (VERIFIED): 100% numerical reproduction of draft manuscript Table 1 verified; denominators and anti-leakage intact. **Status: COMPLETE & RESOLVED.**
- **Phase C: Data Search (BOUNDED COMPLETE / DECISION GATE ENFORCED)**
  - C1: Systematic search for second multi-night labeled wearable dataset completed. Survey across public repositories (PhysioNet, NSRR, OpenNeuro) and literature confirmed no qualifying second dataset exists (van der Aar 2025 cohort is restricted/private; SleepAccel and DREAMT are 1 night/subject). Decision gate enforced: single-dataset paper framed as replication, extension, and failure analysis; limitation reported openly. **Status: BOUNDED COMPLETE.**
- **Phase D: Experiments (Governance Ratified under DEC-033, DEC-034, DEC-035, DEC-036)**
  - Stage 1 (Primary C1 & C2 Closure — COMPLETED):
    * Experiment 1 (TSK-06 / EXP-019-K): Calibration-depth learning curve ($k = 1, 3, 4$) under `PREREG_EXP-019_K_CURVE.md` (Common-cohort $N=33$ on $E_4$, Page's $L$-test, Holm-Bonferroni correction). Status: **COMPLETED (DEC-035)**. Monotonic scaling confirmed ($p = 0.01187$); van der Aar replication confirmed ($\Delta_{k=3} = +3.54\text{ pp}$, $p = 0.00030$); marginal saturation steps inconclusive.
    * Experiment 2 (TSK-07 / EXP-019-S3): Unsupervised adaptation re-run ($UA\text{-}2$) under `PREREG_EXP-019_S3_AMENDMENT.md` (DEC-030 unit fix, pre-flight gate, InfoMax). Status: **COMPLETED (DEC-036)**. Sub-benchmark gain ($\Delta = +0.21\text{ pp}$ [95% CI: $+0.04, +0.39$], $p = 0.0322$).
  - Stage 2 (Final Model Synthesis & Freezing — COMPLETED & SEALED under DEC-037):
    * Executed `freeze_final_model.py --device cuda:0` across all 15 frozen Model A checkpoints (`163,940` params). All 15 checkpoints SHA-256 sealed (`frozen_checkpoint_manifest.json`), 141 subject-seed OOF rows extracted (`final_model_oof_subject_metrics.csv`, max fold-pooled parity diff `0.000099`, max per-subject diff `0.000655`), and `STAGE2_MODEL_FREEZE_REPORT.md` generated.
  - Stage 3 (Definitive Matched Baseline Benchmark — EGM-VERIFIED & LAUNCHED under DEC-037):
    * Experiment 3 (TSK-08 / EXP-021): Restored 246-night `prep_57738d534b` feature cache, verified 12/12 unit & CUDA AMP tests (`tests/test_stage2_stage3_implementation.py`) and end-to-end dual-GPU `--dry-run`, ratified `DEC-037`, and launched the full 15-run dual-GPU SLAMSS-IFS benchmark (`./experiments/EXP-021/run_stage3_dual_gpu.sh`).
  - Deferred Tasks: EXP-019B (heterogeneity tests), EXP-020 (gated adaptation), and EXP-027 (SleepAccel transfer) remain deferred under DEC-029.
- **Phase E: Synthesis & Writing (COMPLETE FOR DRAFT v0.1)**
  - Candidate 1 (COMPLETE): Historical split & data leakage audit across EXP-001 to EXP-004 (`EXP001_EXP004_SPLIT_LEAKAGE_AUDIT.md`, `EXPERIMENTS.md` TA-01 to TA-06).
  - Candidate 2 (COMPLETE / AUDITED): SLAMSS-IFS scientific baseline audit (`SLAMSS_IFS_BASELINE_PROTOCOL_AUDIT.md`; EXP-012 3-seed mean 68.44% ± 1.35% verified; 33.3x repo vs 4.06x paper text distinguished).
  - Candidate 3 (COMPLETE / GATED): EXP-019B and EXP-020 preregistration drafts authored with prior-access disclosures (`EXP019B_PREREGISTRATION_DRAFT.md`, `EXP020_PREREGISTRATION_DRAFT.md`; hashes preserved). Strictly unapproved for execution.
  - Candidate 4 (COMPLETE / AUDITED): BIDSleep cohort data and signal integrity audit (`BIDSLEEP_COHORT_DATA_INTEGRITY_AUDIT.md`, 3 CSVs; 7 nights across 6 subjects; Bidslab34 15 Deep epochs [0.68%]; evaluable denominators $k=1..4$: $47, 46, 42, 33$).
  - Candidate 5 / Phase 3 Polish (COMPLETE): Full working manuscript draft Sections 1–8 authored (`MANUSCRIPT_DRAFT_v0.1.md`), Table 1 enhanced with verified LoA bounds standard errors and 95% CIs (`FIND-B-03`), multi-night agreement distinguished from single-night spot bounds (`FIND-B-01`), aggregate micro-data loss cataloged (`FIND-B-02`), WASO underestimation empirically framed (`FIND-B-04`), and claim matrix synchronized across all 40 claims (`MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv`).

---

## 4. Next Concrete Tasks

1. **Stage 1 Closure & Results Integration (COMPLETED)**:
   - TSK-06 ($k$-learning curve) and TSK-07 (UA-2 rerun) successfully executed and verified on dual-GPU workstation.
   - All empirical metrics serialized to `exp019_k_learning_curve_executed.csv`, `exp019_k_curve_subject_metrics.csv`, and `exp019_ua2_rerun_subject_metrics.csv`.
   - Governance decisions DEC-035 and DEC-036 ratified in `DECISIONS.md`.
2. **Stage 2 Final Model Freeze (COMPLETED under DEC-037)**:
   - All 15 frozen Model A checkpoints SHA-256 hashed (`frozen_checkpoint_manifest.json`) and 141 subject-seed OOF rows verified (`final_model_oof_subject_metrics.csv`, `STAGE2_MODEL_FREEZE_REPORT.md`).
3. **Stage 3 Execution (TSK-08 / EXP-021 — IN PROGRESS on Dual RTX 3060 GPUs)**:
   - 15-run official 5,465,768-parameter SLAMSS-IFS benchmark running concurrently on `cuda:0` (7 runs) and `cuda:1` (8 runs) via `./experiments/EXP-021/run_stage3_dual_gpu.sh`, followed automatically by `analyze_tsk08_paired_benchmark.py`.
4. **Post-Stage 3 Action**:
   - Inspect `experiments/EXP-021/slamss_ifs_paired_comparative_report.md`, record final benchmark decision in `DECISIONS.md` and `EXPERIMENTS.md`, and push all results to `origin main`.
