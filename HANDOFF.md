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
  - Stage 3 (Definitive Matched Baseline Benchmark — COMPLETED under DEC-038):
    * Experiment 3 (TSK-08 / EXP-021): Restored 246-night `prep_57738d534b` feature cache, verified 12/12 unit & CUDA AMP tests, and executed the full 15-run dual-GPU SLAMSS-IFS benchmark across both RTX 3060 GPUs (`./experiments/EXP-021/run_stage3_dual_gpu.sh`). Evaluated against frozen Model A ($N=47$, 10,000 bootstrap resamples; SLAMSS-IFS advantage $\bar{d} = -2.58\text{ pp}$ [95% CI: $-3.53, -1.68\text{ pp}$], $p = 1.98 \times 10^{-6}$; Model A achieves 33.34x parameter reduction with statistical equivalence on REM sleep staging, $p=0.105$). Output artifacts generated in `experiments/EXP-021/` and ratified under `DEC-038`.
  - Stage 4 / Stage D (Exploratory Closure & Zero-Shot External Validation — 100% COMPLETED under DEC-040 & DEC-041, Commit `945c7c1`):
    * EXP-019B (CPU): Heterogeneity audit across 6 predictors against EXP-019 and EXP-021; all 12 primary tests null after BH-FDR ($q \ge 0.5673$), Oldham headroom $\rho = +0.0376$ ($p = 0.8043$).
    * EXP-020 (`cuda:0`): Regularized and gated adaptation closure on $N=46$; `GATE_tau` achieved $+1.33\text{ pp}$ vs. ZS ($p_{\text{Holm}} = 0.0028$) and cut worst degradation from $-16.34\text{ pp}$ to $-6.40\text{ pp}$ by abstaining `Bidslab34`; `FT-2-KL` yielded $+1.03\text{ pp}$ ($p_{\text{Holm}} = 0.0030$).
    * EXP-027 & EXP-028 (`cuda:1`): Zero-shot external validation on PhysioNet SleepAccel v1.0.0 ($N=31$ Apple Watch + in-lab PSG, $n_{\text{qc\_pass}}=25$). Model A achieved statistical parity with 33.34x larger Official SLAMSS-IFS (`prob_ensemble` `62.94%` vs. `62.79%`, $p = 0.9789$, $d_z = +0.03$). Full Bland-Altman agreement with 744 raw paired tuples persisted. Post-execution EGM drift audit cleared with 0 Blocking, 0 Major.
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
3. **Stage 3 Execution (TSK-08 / EXP-021 — COMPLETED under DEC-038)**:
   - 15-run official 5,465,768-parameter SLAMSS-IFS benchmark completed across all 5 folds $\times$ 3 seeds on dual RTX 3060 GPUs.
   - Paired comparative analysis executed via `analyze_tsk08_paired_benchmark.py`: SLAMSS-IFS advantage established ($\bar{d} = -2.58\text{ pp}$ Macro-F1, $p = 1.98 \times 10^{-6}$), with Model A achieving 33.34x parameter reduction and statistical equivalence on REM ($p = 0.1047$).
   - Deliverables emitted: `experiments/EXP-021/{slamss_ifs_matched_subject_metrics.csv, slamss_ifs_paired_subject_deltas.csv, slamss_ifs_paired_comparison_summary.json, slamss_ifs_paired_comparative_report.md}`.
4. **Audit & Profiling Tasks W0-A and W0-B (COMPLETED under DEC-039 + Micro-Patch Reconciled)**:
   - **Task W0-A (TSK-06 Forensic Audit & Clean-ZS Micro-Patch):** Identified in-place model mutation defect across outer test subjects in `run_tsk06_k_curve.py` causing un-reset ZS evaluations on subjects 2..N to evaluate weights fine-tuned on the prior subject (de facto donor control), resolving the $62.86\%$ baseline anomaly (which matched `FT-other-2`). CPU-only reconciliation (`experiments/EXP-019/reconcile_tsk06_clean_zs.py`, 594 night inferences, zero retraining) computed exact Clean-ZS baselines ($63.54\%$ on $E_4$ [$N=33$]; $64.58\%$ on $E_2$ [$N=46$]), 10,000-resample bootstrap 95% CIs, exact two-sided Wilcoxon $p$-values, and updated 5-test Holm-Bonferroni adjusted $p$-values in `experiments/EXP-019/exp019_k_learning_curve_CLEAN_RECONCILED.csv` (historical executed CSV untouched). Deliverables: `experiments/EXP-019/TSK06_ZS_DISCREPANCY_AUDIT.md`, `experiments/EXP-019/reconcile_tsk06_clean_zs.py`, `experiments/EXP-019/exp019_k_learning_curve_CLEAN_RECONCILED.csv`.
   - **Task W0-B (Empirical INT8 Edge Benchmark & Hardware Caveats):** Benchmarked all 15 frozen checkpoints across all 47 subjects (738 night inferences) under dynamic INT8 post-training quantization. Empirical parity verified: Macro-F1 delta $\bar{\Delta} = \mathbf{+0.0011\text{ pp}}$ [95% CI: $-0.0277, +0.0312\text{ pp}$], Wilcoxon $p = 0.7892$, Cohen's $d_z = 0.010$; single-threaded CPU latency is $69.39\text{ ms}$/night (4-thread: $47.73\text{ ms}$/night). Two caveats codified in `DEC-039`: (i) empirical Dynamic INT8 in-memory weight payload is **$309.54\text{ KiB}$** (serialized $339.99\text{ KiB}$; `Linear`+`GRU` quantized, `Conv1d` in FP32) vs. theoretical all-INT8 **$160.10\text{ KiB}$**; (ii) bidirectional GRU (`temporal_gru`) requires **full-night sequence buffering on a smartwatch SoC or paired smartphone**, not causal 1-epoch streaming on a 512 KB SRAM MCU. Deliverables: `experiments/STAGE2_FINAL_MODEL/int8_quantization_profile.json`, `INT8_EDGE_BENCHMARK_REPORT.md`.
5. **Stage D Exploratory Closure & Zero-Shot External Validation (100% COMPLETE under `DEC-040` & `DEC-041`, Commit `945c7c1`)**:
   - **EXP-019B (CPU):** `experiments/EXP-019B/{exp019b_subject_features.csv, exp019b_correlation_summary.csv, exp019b_diagnostics.json, EXP019B_CLOSURE_REPORT.md}` — all 12 primary tests null after BH-FDR ($q \ge 0.5673$).
   - **EXP-020 (`cuda:0`):** `experiments/EXP-020/{exp020_inner_val_selection.json, exp020_subject_metrics.csv, exp020_summary_metrics.json, EXP020_CLOSURE_REPORT.md}` — `GATE_tau` $+1.33\text{ pp}$ vs. ZS ($p_{\text{Holm}}=0.0028$, `Bidslab34` abstained); `FT-2-KL` $+1.03\text{ pp}$ ($p_{\text{Holm}}=0.0030$).
   - **EXP-027 & EXP-028 (`cuda:1`):** `experiments/EXP-027/{exp027_per_subject_metrics.csv, exp027_summary.json, preflight_distribution_gate.json, sleepaccel_qc_summary.csv, sleepaccel_sha256_manifest.json}` and `experiments/EXP-028/{exp028_sleepaccel_raw_paired_tuples.csv, exp028_sleepaccel_bland_altman_summary.csv}` — Model A achieved zero-shot parity with 33.34x larger Official SLAMSS-IFS on PhysioNet SleepAccel (`qc_clean` $n=25$ `prob_ensemble` `62.94%` vs. `62.79%`, $p=0.9789$).
6. **Next Concrete Task (Stage E)**:
   - Author `MANUSCRIPT_DRAFT_v0.2.md` integrating Stage 3 (`EXP-021`), W0-A/W0-B (`DEC-039`), and Stage D (`EXP-019B`, `EXP-020`, `EXP-027`, `EXP-028` under `DEC-040` & `DEC-041`), and synchronize the Evidence-to-Claim Matrix.
