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

- **Phase A: Close out EXP-019 (Immediate Priority)**
  - A1: Settle H-A verdict label against locked preregistration decision rule via formal erratum.
  - A2: S3 no-op audit (inspect weight drift, gradient flow, and inference checkpoint assignment).
  - A3: Report missing preregistered outputs ($k=1–4$ curve, sleep-summary agreement, per-stage F1, confusion matrices, Holm correction).
  - A4: Update S1 documentation with donor fine-tuning harm clarification.
  - A5: Complete NAS Iteration 1 erratum (confirming architecture search is closed).
- **Phase B: Evidence Base & Systematic Literature Search**
  - B1: Apply KB round-2 corrections (Heremans PMID, P011/P009 grading, P101 text, Lorenzen DOI).
  - B2: Conduct systematic search tracking forward citations of Song 2026, Phan 2020, Mikkelsen 2018, van der Aar 2025.
  - B3: Full-text review of core papers; build "closest ten" comparison matrix and dated novelty statement.
- **Phase C: Data Search**
  - C1: Search for an independent, second multi-night labeled wearable dataset ($\ge 3$ nights/subject, wrist IHR/PPG + accelerometry, PSG/EEG labels). Strongest Q1 lever.
- **Phase D: Experiments (Strictly Gated)**
  - D1: EXP-020 (exploratory, design-informed regularized/gated adaptation) — strictly gated until Phases A–C and a committed preregistration block are completed.
  - D2: EXP-019B heterogeneity hypothesis testing.
  - D3: EXP-027 external zero-shot transfer (SleepAccel).
  - D4: EXP-028 clinical sleep-summary agreement.
  - D5: Replication on second dataset if found.
- **Phase E: Synthesis & Writing**

---

## 4. Next Concrete Tasks

1. **Phase A1**: Author and commit formal erratum documenting the H-A verdict reconciliation (INCONCLUSIVE under preregistration rule vs NOT SUPPORTED in early report).
2. **Phase A2**: Completed forensic diagnosis of S3 (`experiments/EXP-019/S3_INFOMAX_NOOP_AUDIT_REPORT.md`, DEC-030). Confirmed complete adaptation no-op due to quality-mask unit mismatch (0/82,762 epochs valid, 0 optimizer steps, 0 parameter updates). Status synchronized to UNRESOLVED / INVALIDATED across all workspace governance records.
3. **Phase A3**: Extract the $k=1–4$ learning curve from existing EXP-019 artifacts / predictions across eligible subjects.
4. **Phase B1**: Verify and apply pending Knowledge Base round-2 errata fixes.
5. **Continuous Git Synchronization**: Commit all updated governance documents and push to GitHub (`origin main`).
