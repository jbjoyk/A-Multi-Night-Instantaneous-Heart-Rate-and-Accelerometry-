# Research State

**Last updated:** 2026-11-17 (Governance Milestone: Synchronized to Research Direction v4 and Revised Research Knowledge Base)

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
- **Test Set Status:** The historical EXP-012 test set (11 subjects, 58 nights) is formally **SPENT** (used across prior benchmarks and folded into later CV splits). All final claims rest on 5-fold out-of-fold nested CV across all 47 subjects plus external validation.

---

## 4. Current Experimental State & Hypothesis Status

| ID | Hypothesis / Question | Status | Key Evidence & Governance Standing |
|---|---|---|---|
| **H-A** | Fine-tuning on held-out subject's first 2 labeled nights ($FT\text{-}2$) improves Macro-F1 on later nights vs zero-shot ($ZS$) | **LOCKED (EXP-019)** | $\bar{\Delta} = \mathbf{+1.01\text{ pp}}$ ($64.58\% \to 65.59\%$), 95% bootstrap CI: $\mathbf{[-0.17, +1.99]\text{ pp}}$, Wilcoxon $p = 0.00131$, win rate: 73.9% (34/46). <br>**Governance Reconciled Label:** *INCONCLUSIVE* under the exact locked decision rule (`git show 4dff5b7:PREREG_EXP-019_H-A.md` Section 6: NOT SUPPORTED requires CI upper bound $< +1.50\text{ pp}$; since $+1.99 > +1.50$, it evaluates to INCONCLUSIVE). The EXP-019 report labeled it NOT SUPPORTED. Reconciled via erratum. Numerical results immutable. |
| **S1** | $FT\text{-}2$ outperforms donor control $FT\text{-}other\text{-}2$ (Specificity Control) | **LOCKED (EXP-019)** | **SUPPORTED:** $\bar{\Delta} = \mathbf{+2.73\text{ pp}}$ [95% CI: $+1.05, +4.32\text{ pp}$], $p = 9.0 \times 10^{-6}$. *Nuance:* $FT\text{-}other\text{-}2$ is $-1.72\text{ pp}$ below $ZS$ (harmful), proving fine-tuning on another subject degrades performance and adaptation is subject-specific. |
| **S2** | $FT\text{-}2$ outperforms per-subject model trained from scratch ($PS\text{-}2$) | **LOCKED (EXP-019)** | **SUPPORTED:** $\bar{\Delta} = \mathbf{+20.32\text{ pp}}$ [95% CI: $+17.70, +22.94\text{ pp}$], $p < 10^{-10}$. Cohort pre-training is indispensable. Treated as a weak baseline check, not a major claim. |
| **S3** | Label-free unsupervised InfoMax adaptation ($UA\text{-}2$) outperforms $ZS$ | **LOCKED (EXP-019)** | $\bar{\Delta} = \mathbf{0.00\text{ pp}}$ identically across all 46 subjects. **UNRESOLVED / INVALIDATED (Implementation No-Op):** Phase A2 forensic audit (`S3_INFOMAX_NOOP_AUDIT_REPORT.md`, DEC-030) confirmed that `compute_wearable_quality_mask` checked `30 <= IHR <= 180` (raw BPM) on standardized z-score IHR ($[-3.95, +9.94]$). Exactly 0 out of 82,762 calibration epochs passed the mask, terminating adaptation on step 0 (0 optimizer steps, 0/163,940 parameters changed, identical evaluation logits/predictions). S3 is scientifically uninformative; it must NOT be interpreted as evidence that unsupervised adaptation fails (nor that it works). |
| **K** | Learning curve over calibration nights $k = 1, 2, 3, 4$ | **Pre-registered descriptive** | High priority for Phase A3 close-out. Directly connects with van der Aar et al. 2025 finding of significance at $\ge 3$ nights. |
| **H-C** | Context scaling: $L=600$ beats $L=60$ | **EXP-018 Phase 2** | $+5.33\text{ pp}$ Macro-F1 [95% CI: $+4.28, +6.39\text{ pp}$], 46/47 subjects. Counts as confirmatory if pre-decision commit verified (`TO VERIFY`); exploratory otherwise. |
| **EXP-019B** | Heterogeneity and Failure Mechanisms | **Exploratory** | Diagnostic audit completed (`HETEROGENEITY_AUDIT_REPORT.md`). Causal DAG separates root causes (M1: stage absence, M2: autonomic drift) from mediators (M3: calibration loss, M5: parameter drift). Bidslab34 identified with 0.00% deep sleep on Night 2. |
| **EXP-020** | Regularized / Gated Adaptation | **Exploratory (Gated)** | Strictly gated under Phase D1. Requires Phase A, B, C prerequisites and committed preregistration before launch. |
| **EXP-027** | External Zero-Shot Validation (SleepAccel) | **Pre-registered** | Planned under Phase D3. |
| **EXP-028** | Clinical Sleep-Summary Agreement | **Pre-registered** | Planned under Phase D4 (and Phase A3 descriptive analysis). |

---

## 5. Work Plan & Phase Execution Structure (v4)

- **Phase A — Close out EXP-019 (Mandatory Prerequisite, No Numerical Alterations):**
  - **A1:** Reconcile H-A verdict label against locked preregistration decision rule via erratum.
  - **A2:** S3 no-op audit completed (confirmed implementation no-op via quality-mask standardized unit mismatch; S3 invalidated as scientifically uninformative; `experiments/EXP-019/S3_INFOMAX_NOOP_AUDIT_REPORT.md`, DEC-030).
  - **A3:** Extract and report missing pre-registered outputs: $k=1–4$ learning curve, sleep-summary agreement, per-stage F1, confusion matrices, and Holm-adjusted p-values.
  - **A4:** Update S1 documentation to clarify donor-night fine-tuning harm.
  - **A5:** Complete NAS Iteration 1 erratum (confirming architecture search is closed).
- **Phase B — Evidence Base & Literature Intelligence:**
  - **B1:** Apply KB round-2 corrections (Heremans PMID, P011/P009 grading, P101 text, Lorenzen DOI).
  - **B2:** Execute systematic search across PubMed, IEEE Xplore, Scopus, arXiv, medRxiv (2018–present) tracking forward citations of Song 2026, Phan 2020, Mikkelsen 2018, and van der Aar 2025.
  - **B3:** Read core papers in full text; construct "closest ten" comparison matrix and dated novelty statement.
- **Phase C — Data Search:**
  - **C1:** Systematically search for a second multi-night labeled wearable dataset ($\ge 3$ labeled nights/subject, wrist PPG/IHR + accelerometry, EEG/PSG ground truth). This constitutes the strongest Q1 lever.
- **Phase D — Experiments (Strictly Gated):**
  - **D1:** EXP-020 (exploratory, design-informed regularized/gated adaptation) — only after Phase A–C completion and committed preregistration.
  - **D2:** EXP-019B formal heterogeneity test protocol.
  - **D3:** EXP-027 external zero-shot transfer to SleepAccel.
  - **D4:** EXP-028 clinical sleep-summary agreement with Bland-Altman analysis.
  - **D5:** If a second dataset is found, pre-register and execute confirmatory replication.
- **Phase E — Synthesis & Paper Writing:**
  - Write manuscript adhering strictly to pre-registered rules and honest negative/inconclusive reporting.

---

## 6. Strict Governance Guardrails

1. **Immutable Preregistration & Data**: EXP-019 preregistration (`4dff5b7`), numerical metrics, and stored checkpoints remain strictly frozen.
2. **No Premature Training**: Do NOT launch EXP-020 or any adaptation training until Phase A and B prerequisites are completed and user approval is obtained under a committed preregistration block.
3. **No Architecture Search**: Model A (`ConventionalCNNBiGRU`, 163,940 params) is frozen. NAS is permanently terminated.
4. **Leakage Protection**: All evaluations enforce strict subject-level disjointness and chronological calibration/evaluation splits.
5. **Continuous Git Synchronization**: All documentation updates, errata, and audit scripts are committed and pushed to `origin main`.
6. **Parameter-Count Governance (DEC-031)**: The historical +20% (196,728) cap is removed. 163,940 parameters is the baseline reference size. Future models should preferentially be lighter than this baseline; larger models require explicit scientific justification and transparent reporting.
