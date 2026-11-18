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
| **S3 ($UA\text{-}2$ vs $ZS$)** | $\bar{\Delta} = \mathbf{0.00\text{ pp}}$ [95% CI: $\mathbf{0.00, 0.00\text{ pp}}$] identically across 46 subjects | **UNRESOLVED / INVALIDATED (Implementation No-Op)**<br>• Phase A2 forensic audit (`S3_INFOMAX_NOOP_AUDIT_REPORT.md`, DEC-030) confirmed that `compute_wearable_quality_mask` checked raw BPM limits (`30 <= IHR <= 180`) on standardized z-score IHR. Exactly 0 out of 82,762 calibration epochs passed, causing adaptation to abort on step 0 (0 updates, 0/163,940 params changed, identical evaluation logits/predictions).<br>• Scientifically uninformative: must NOT be cited as evidence that unsupervised adaptation fails (nor that it works). |
| **$k=1–4$ Learning Curve** | Gain vs number of calibration nights | **Pre-registered Descriptive Output (Pending Phase A3)**<br>• Critical for comparing with van der Aar et al. 2025's finding of significance at $\ge 3$ nights. |

---

## 3. Work Plan Phasing (v4)

- **Phase A: Close out EXP-019 (COMPLETED)**
  - A1: Settle H-A verdict label against locked preregistration decision rule via formal erratum (`INCONCLUSIVE`).
  - A2: S3 no-op audit completed (`experiments/EXP-019/S3_INFOMAX_NOOP_AUDIT_REPORT.md`, DEC-030).
  - A3: Diagnostic extraction completed (`experiments/EXP-019/PHASE_A3_DIAGNOSTIC_REPORT.md` + 5 CSV/JSON deliverables).
  - A4: S1 documentation updated with donor fine-tuning harm clarification.
  - A5: NAS Iteration 1 erratum completed (architecture search permanently closed).
- **Phase B: Evidence Base & Systematic Literature Search (PARTIALLY COMPLETE / AUDITED)**
  - B1: KB round-2 corrections applied and verified (Heremans PMID `35508121` across 6 files, P011/P009 downgraded to MODERATE, P101 author Colin Quirk and text fixed, gating labels clarified, Lorenzen verified). **Status: COMPLETE.**
  - B2: Systematic search executed and logged across PubMed, Europe PMC, arXiv, and medRxiv/bioRxiv tracking 4 query blocks and forward citations (`search_log.md`). IEEE Xplore and Scopus/WoS bounded by institutional access. **Status: PARTIALLY COMPLETE.**
  - B3 & Exit: Core papers assessed; 5 verified via full text (`P001`, `P002`, `P006`, `P056`, `P106`), 2 verified via abstract/excerpts under paywalls (`P007`, `P054`). "Closest ten" comparison matrix and dated novelty statement produced without priority claims (`research_knowledge/updates/2026-11-18_phase_b_and_c_execution.md`). **Status: PARTIALLY COMPLETE.**
- **Phase C: Data Search (BOUNDED COMPLETE / DECISION GATE ENFORCED)**
  - C1: Systematic search for second multi-night labeled wearable dataset completed. Survey across public repositories (PhysioNet, NSRR, OpenNeuro) and literature confirmed no qualifying second dataset exists (van der Aar 2025 cohort is restricted/private; SleepAccel and DREAMT are 1 night/subject). Decision gate enforced: single-dataset paper framed as replication, extension, and failure analysis; limitation reported openly. **Status: BOUNDED COMPLETE.**
- **Phase D: Experiments (Strictly Gated)**
  - D1: EXP-020 (exploratory, design-informed regularized/gated adaptation) — strictly gated until committed preregistration block is approved.
  - D2: EXP-019B heterogeneity hypothesis testing.
  - D3: EXP-027 external zero-shot transfer (SleepAccel).
  - D4: EXP-028 clinical sleep-summary agreement.
- **Phase E: Synthesis & Writing (IN PROGRESS)**
  - Candidate 1 (COMPLETE): Historical split & data leakage audit across EXP-001 to EXP-004 (`EXP001_EXP004_SPLIT_LEAKAGE_AUDIT.md`, `EXPERIMENTS.md` TA-01 to TA-06).
  - Candidate 2 (COMPLETE / AUDITED): SLAMSS-IFS scientific baseline audit (`SLAMSS_IFS_BASELINE_PROTOCOL_AUDIT.md`; EXP-012 3-seed mean 68.44% ± 1.35% verified; 33.3x repo vs 4.06x paper text distinguished).
  - Candidate 3 (COMPLETE / GATED): EXP-019B and EXP-020 preregistration drafts authored with prior-access disclosures (`EXP019B_PREREGISTRATION_DRAFT.md`, `EXP020_PREREGISTRATION_DRAFT.md`; hashes preserved). Strictly unapproved for execution.
  - Candidate 4 (COMPLETE / AUDITED): BIDSleep cohort data and signal integrity audit (`BIDSLEEP_COHORT_DATA_INTEGRITY_AUDIT.md`, 3 CSVs; 7 nights across 6 subjects; Bidslab34 15 Deep epochs [0.68%]; evaluable denominators $k=1..4$: $47, 46, 42, 33$).
  - Candidate 5 (COMPLETE / SYNCHRONIZED): Manuscript draft v0.1 Sections 1–5 authored and synchronized (`MANUSCRIPT_DRAFT_v0.1.md`, `MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv`, `MANUSCRIPT_UNRESOLVED_EVIDENCE.md`).

---

## 4. Next Concrete Tasks

1. **Independent Verification (`/egm-review`)**:
   - Perform an independent `/egm-review` of the completed Candidate 2, 4, and 5 resolutions to confirm all corrections prior to any Git commit considerations.
2. **Phase D Experimental Gating (Strictly Gated)**:
   - EXP-019B and EXP-020 remain strictly unauthorized.
   - Do NOT launch or schedule any training or adaptation jobs without explicit written user approval.
   - Execute EXP-028 (formal Bland-Altman clinical sleep-summary analysis from `exp019_sleep_summary_agreement.csv`).
3. **Continuous Git Synchronization**:
   - Ready to commit and push all newly generated audit, preregistration, diagnostic, and manuscript deliverables upon user instruction.
