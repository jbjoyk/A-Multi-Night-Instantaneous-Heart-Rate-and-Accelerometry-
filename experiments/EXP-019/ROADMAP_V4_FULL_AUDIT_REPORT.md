# Master Roadmap Audit: Research Direction v4 Comprehensive Assessment

**Audit Date:** 2026-11-17  
**Auditor:** EGM Review System (Top-Level Forensic & Strategic Roadmap Audit)  
**Governing Document:** `RESEARCH_DIRECTION_v4_FULL.md` (Ratified under `DEC-029` and `DEC-030`)  
**Scope:** Exhaustive audit across all five phases: **Phase A, Phase B, Phase C, Phase D, and Phase E**.

---

## 1. Executive Summary & Global Phase Assessment

This audit establishes the empirical, algorithmic, and governance status of the entire research program under `RESEARCH_DIRECTION_v4_FULL.md`.

```
Roadmap Progression State:
[ PHASE A: EXP-019 Closeout ] ───▶ NOT READY TO CLOSE (A3 missing outputs pending)
[ PHASE B: Evidence Base    ] ───▶ PARTIALLY READY   (Full-text review & closest-10 matrix pending)
[ PHASE C: Second Dataset   ] ───▶ NOT STARTED       (No eligible multi-night dataset identified)
[ PHASE D: Extensions       ] ───▶ STRICTLY BLOCKED  (Gated behind Phase A, B, C prerequisites)
[ PHASE E: Paper Synthesis  ] ───▶ BLOCKED           (Gated behind Phase A3, B3, D3 evidence)
```

**Overall Research Readiness:** **`PARTIALLY READY`** (Solidified foundation on Phase A1/A2/A4/A5; blocked from advancing to training/extension phases until Phase A descriptive closure and Phase B evidence-base prerequisites are met).

---

## 2. Phase A Audit — EXP-019 Confirmatory Closure

### A1 — Hypothesis H-A Verdict Reconciliation
- **Locked Preregistration Protocol:** Commit `4dff5b7` (`PREREG_EXP-019_H-A.md` Section 6).
- **Exact Locked Decision Rule:**
  - **SUPPORTED:** $\bar{\Delta} \ge +1.50\text{ pp}$ AND 95% bootstrap CI lower bound $> 0$.
  - **NOT SUPPORTED:** 95% bootstrap CI upper bound $< +1.50\text{ pp}$ (ruling out an effect of practical size).
  - **INCONCLUSIVE:** Anything else.
- **Observed Empirical Result:** $\bar{\Delta} = \mathbf{+1.01\text{ pp}}$ ($64.58\% \to 65.59\%$), 95% bootstrap CI: $\mathbf{[-0.17, +1.99]\text{ pp}}$, Wilcoxon signed-rank $p = 0.00131$, subject win rate: $73.9\%$ (34/46 improved).
- **Correct Verdict Classification:** **`INCONCLUSIVE`**.
  - The observed mean gain ($+1.01\text{ pp}$) fell short of $+1.50\text{ pp}$ (failing SUPPORTED).
  - However, the upper confidence limit ($+1.99\text{ pp}$) is not below $+1.50\text{ pp}$ (failing NOT SUPPORTED).
  - Statistical ruling out of a practical $+1.50\text{ pp}$ effect was not achieved with $N=46$.
- **Governance Reconciliation:** Early reporting checked only a binary `mean >= 1.50`, outputting `"supported": false` to JSON, which was mislabeled in prose as "NOT SUPPORTED." Reconciled to `INCONCLUSIVE` under `DEC-029` and `EXP019_FINAL_REPORT.md` Section 6 erratum.
- **Numerical Result Integrity:** Unchanged; byte-for-byte identical across all JSON/CSV artifacts.

### A2 — Hypothesis S3 Forensic Validity
- **Audit Authority:** `experiments/EXP-019/S3_INFOMAX_NOOP_AUDIT_REPORT.md` (DEC-030).
- **Forensic Diagnosis:** Secondary Hypothesis S3 ($UA\text{-}2$ vs $ZS = 0.00\text{ pp}$, 95% CI: $[0.00, 0.00]\text{ pp}$, all 46 subjects identical) was a complete **implementation no-op**.
- **Root Cause:** In `exp019_adaptation_core.py::compute_wearable_quality_mask`, the plausibility check expected raw beats-per-minute ($30 \le \text{IHR} \le 180$), whereas preprocessed cached records provided standardized z-scores strictly bounded within $[-3.95, +9.94]$.
- **Forensic Measurements:**
  - 82,762 calibration epochs audited across all 94 calibration nights for all 47 subjects;
  - Exactly 0 epochs passed the mask ($0.00\%$);
  - In `adapt_unsupervised_infomax`, `if valid_probs.shape[0] == 0: break` triggered on step 0;
  - 0 optimizer steps, 0 backward passes, 0/163,940 parameters changed (displacement $\equiv 0.00000000$);
  - Evaluation logits and stage predictions were identical to $ZS$ down to floating-point precision.
- **Stage D Consequence:** In inner-validation tuning, all 4 grid candidates in `UA_GRID` broke out on step 0 and returned the zero-shot score; tie-breaker selected `(lr=1e-5, steps=20)` purely due to sorting on minimum hyperparameters.
- **Current Governance Status:** **`UNRESOLVED / INVALIDATED (Implementation No-Op)`**.
- **Scientific Implication:** S3 is scientifically uninformative. It provides zero evidence against (or for) unsupervised InfoMax adaptation. No document currently treats S3 as valid negative evidence. Code remains unmodified; re-run deferred to Phase D.

### EXP-019 S1 — Specificity Control
- **Estimand & Metrics:** $\bar{\Delta} = \mathbf{+2.73\text{ pp}}$ (from $62.86\% \pm 8.85\%$ to $65.59\% \pm 8.59\%$), 95% bootstrap CI: $[+1.05, +4.32]\text{ pp}$, Wilcoxon $p = 9.0 \times 10^{-6}$, win rate: $78.3\%$ (36/46).
- **Donor Harm Nuance:** $FT\text{-}other\text{-}2$ ($62.86\%$) performs **$-1.72\text{ pp}$ below the zero-shot baseline ($64.58\%$)**.
- **Scientific Interpretation:** The $+2.73\text{ pp}$ gap CANNOT legitimately be claimed as pure personalization benefit. It is composed of:
  - Personal fine-tuning benefit over zero-shot: $\mathbf{+1.01\text{ pp}}$;
  - Donor fine-tuning degradation below zero-shot: $\mathbf{+1.72\text{ pp}}$.
- Cross-subject fine-tuning actively harms performance, proving biological specificity.

### EXP-019 S2 — Cohort Transfer Foundation
- **Estimand & Metrics:** $\bar{\Delta} = \mathbf{+20.32\text{ pp}}$ (from $45.27\% \pm 9.42\%$ to $65.59\% \pm 8.59\%$), 95% bootstrap CI: $[+17.70, +22.94]\text{ pp}$, Wilcoxon $p < 10^{-10}$, win rate: $100\%$ (46/46).
- **Scientific Claim:** Training from scratch per-subject on 2 nights fails completely ($45.27\%$). Cohort pre-training is indispensable. Accurately treated in governance as a weak sanity baseline, not a major claim.

### A3 — Preregistered Missing Outputs Audit

| Requirement | Description | Current Availability | Generation Feasibility | Classification |
|---|---|---|---|---|
| **1. $k=1–4$ Learning Curve** | Gain vs number of calibration nights ($k \in \{1, 2, 3, 4\}$) | **Missing** (Stage C only ran $k=2$) | **Requires computation** (run lightweight deterministic adaptation on eligible cohorts: $n=46, 42, 33$) | Preregistered descriptive |
| **2. Heterogeneity vs ZS** | Oldham's baseline-headroom correlation | **Available** (`HETEROGENEITY_AUDIT_REPORT.md`, $\rho = +0.0376$) | Already extracted | Exploratory diagnostic |
| **2b. Night-Index Effect** | Gain broken down by evaluation night position (Night 1, 2, 3+) | **Missing** (results pooled across nights) | **Requires computation** (evaluate night-by-night from saved checkpoints) | Preregistered descriptive |
| **3. Per-Stage F1 & Conf. Matrices** | Wake, Light, Deep, REM F1 and $4 \times 4$ matrices | **Missing** (only Macro-F1 stored in results JSON) | **Requires computation** (inference forward pass on saved checkpoints) | Preregistered descriptive |
| **4. Sleep-Summary Agreement** | Bland-Altman bias & 95% limits (TST, SOL, WASO, SE, REM%, Deep%) | **Missing** (not computed in Stage C driver) | **Requires computation** (epoch-to-summary conversion on test nights) | Preregistered descriptive |
| **5. Selected Configurations** | Stage D hyperparameter choices per fold | **Available** (`exp019_stage_d_selected_configs.json`) | Already extracted | Confirmatory protocol |
| **6. Holm-Adjusted p-values** | Multiplicity control on secondary tests | **Reconstructable** (S1: $p_{\text{adj}} = 1.8 \times 10^{-5}$; S2: $p_{\text{adj}} < 3 \times 10^{-10}$) | Mathematical calculation | Confirmatory protocol |
| **7. Stage D Spread** | Inner-val paired difference SD | **Available** (`s_delta = 2.2432 pp`, 25 subjects) | Already extracted | Confirmatory protocol |
| **8. Donor-Harm Interpretation** | Decomposition of S1 specificity gap | **Available** (`DEC-029`, `HANDOFF.md`, `EXPERIMENTS.md`) | Already extracted | Confirmatory protocol |

### Phase A Exit Recommendation: **`NOT READY TO CLOSE`**
- **Exact Blockers:** Phase A3 descriptive analyses (1, 2b, 3, 4) must be extracted and published in a formal Phase A close-out report.

---

## 3. Phase B Audit — Prior-Art & Evidence Base

### Audit Findings against v4 Mandates

1. **Systematic Search Execution:** Partially completed. 106 paper records exist in `research_knowledge/papers/`, but queries were conducted primarily via web and PubMed sweeps; formal Boolean query strings across IEEE Xplore, Scopus/Web of Science, and medRxiv are not fully documented in `search_log.md`.
2. **Databases / Sources Searched:** PubMed, PMC, Google Search, arXiv, Frontiers, MDPI. IEEE Xplore and Scopus were not systematically scraped.
3. **Systematic Rigor:** Lacks formal PRISMA-style tracking (duplicates, hit counts per exact string, exclusion logs).
4. **Full-Text Availability:** **Major deficiency:** Only **21 of 106 papers (19.8%)** have `retrieval_depth: full-text`. **85 of 106 papers (80.2%)** rely strictly on `abstract-only` or search metadata.
5. **Key Core Papers on Abstract Only:**
   - [van der Aar et al. (2025)](https://pubmed.ncbi.nlm.nih.gov/40882681/) (`P106`) is `abstract-only`.
   - [Phan et al. (2020)](https://lirias.kuleuven.be/retrieve/616969) (`P006`) is `abstract-only`.
   - [Lorenzen et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/38833404/) (`P007`) is `abstract-only`.
   - [Heremans et al. (2022)](https://pubmed.ncbi.nlm.nih.gov/35508121/) (`P054`) is `abstract-only`.
6. **van der Aar et al. (2025) Integration:** Successfully ingested as `P106` in `evidence_matrix.csv`, cited in `RESEARCH_DIRECTION_v4_FULL.md` and `DEC-029`. However, the full text has not been reviewed; exact non-responder rates (17.5%) and kappa values are marked `TO FILL`.
7. **Novelty Claim Correction:** The prior claim that "multi-night personalization is an open lane" was formally withdrawn in `DEC-029`. The research is now correctly framed as replication and extension with consumer IHR + accelerometry in healthy adults.
8. **Closest-Ten Comparison Matrix:** **Missing**. No dedicated table comparing our protocol directly against the ten closest papers exists in `research_knowledge/`.
9. **Dated Novelty Statement:** **Missing**. `novelty_registry.md` contains concept rows, but no signed, dated novelty statement defining our exact contribution boundaries against 2025/2026 literature.
10. **KB Contradictions & Quality Issues:**
    - **Wrong Heremans PMID:** PMID `35877770` remains present in 5 summary files (`literature_map.md`, `methods_catalog.md`, `competing_hypotheses.md`, `research_gaps.md`, `novelty_registry.md`), whereas `P054` corrected it to `35508121`.
    - **P101 Authorship:** Author Colin Quirk is missing from `P101`.
    - **Placeholder Metadata:** Multiple scaled records (`P020`–`P100`) contain `UNKNOWN` for subject counts, night counts, and preprocessing details.
11. **Evidence Strength Calibration:** Many papers without statistical tests were initially graded `STRONG`; need to be capped at `MODERATE` per v4 evidence criteria.

### Phase B Exit Recommendation: **`NOT READY TO CLOSE`**
- **Exact Blockers:**
  1. Retrieve and read full texts for core papers: van der Aar 2025 (`P106`), Phan 2020 (`P006`), Lorenzen 2024 (`P007`), Heremans 2022 (`P054`);
  2. Apply KB round-2 errata (PMID 35508121, P101 author, evidence strength caps);
  3. Construct the "closest ten" comparison matrix;
  4. Author the signed, dated novelty statement.

---

## 4. Phase C Audit — Second Dataset Search

### Audit against v4 Eligibility Criteria

Required criteria for confirmatory replication:
1. $\ge 3$ labeled nights per subject;
2. Wrist PPG or instantaneous heart rate (IHR);
3. Accelerometry;
4. EEG or PSG-derived sleep stage labels.

### Evaluation of Cataloged Datasets (`datasets_catalog.md`)

| Dataset ID | Dataset Name | Nights / Subject | Wearable Sensors | Labels | Suitable for Multi-Night Confirmatory? | Status / Limitation |
|---|---|---|---|---|---|---|
| `D001` | **BIDSleep** | 3–7 nights | Apple Watch IHR + Accel | Dreem 2 EEG | Primary Dataset | Already spent in EXP-019 Stage C ($N=46$). Cannot provide independent replication. |
| `D002` | **SleepAccel** | 1 night | Apple Watch Accel + PPG | Attended PSG | **NO** | 1 PSG night per subject. High for zero-shot external transfer (EXP-027), but cannot test multi-night adaptation. |
| `D003` | **MESA Sleep** | 1 night | Actigraphy + Oximetry | Unattended PSG | **NO** | 1 PSG night per subject; lacks raw continuous IHR/PPG waveforms. |
| `D004` | **MrOS Sleep** | 1 night | Actigraphy (60s) + ECG | Central PSG | **NO** | 1 PSG night per subject; older male cohort; coarse actigraphy. |
| `D005` | **SHHS** | 1–2 visits | ECG-derived IHR | PSG | **NO** | Visits separated by years; lacks wrist accelerometry. |
| `D006` | **Sleep-EDF** | 2 nights | EEG only | PSG | **NO** | Lacks wearable PPG / IHR and accelerometry. |
| `D007` | **DREAMT** | `UNKNOWN` | Wrist wearable | PSG | `UNKNOWN` | Requires verification in Phase C. |
| `P106` | **van der Aar Cohort** | Up to 7 nights | Wrist PPG + Accel | Wearable EEG + PSG | **NO (Proprietary)** | Philips / Kempenhaeghe clinical trial data; not publicly available. |

### Phase C Classification: **`NOT STARTED`**
- **Findings:** No external dataset in the research workspace meets all four criteria. Every public dataset currently available has either 1 night per subject (SleepAccel, MESA), lacks accelerometry (SHHS), or is EEG-only (Sleep-EDF).
- **Implication:** Independent confirmatory replication of multi-night adaptation on a second dataset is currently blocked by data availability.

---

## 5. Phase D Audit — Exploratory Extensions Governance

### Candidate Experiments Governance Table

| Experiment | Scientific Purpose | Mandatory Prerequisites | Formally Authorized? | Currently Blocked By | Scientific Status |
|---|---|---|---|---|---|
| **EXP-020** | Regularized (KL) & Gated Adaptation to prevent negative transfer | Phase A closure, Phase B core reading (Phan, Lorenzen, van der Aar), committed preregistration block | **NO** | Phase A3 missing outputs; Phase B full texts; uncommitted preregistration | Exploratory (Design-informed, descriptive endpoints only) |
| **EXP-019B** | Heterogeneity failure-mode hypothesis testing (DAG mechanisms) | EXP-019 frozen artifacts, committed testing plan | **NO** (Audit completed, formal tests gated) | Phase A closure | Exploratory diagnostic |
| **EXP-027** | Zero-shot transfer to SleepAccel (31 subjects, 1 PSG night) | Label mapping, sensor alignment, committed evaluation script | **NO** | Phase A closure; Phase B evidence base | Descriptive external validation |
| **EXP-028** | Clinical sleep-summary agreement (Bland-Altman) | Clinically acceptable margin definitions | **NO** for standalone Phase D; part of Phase A3 for BIDSleep | Phase A closure | Descriptive clinical agreement |
| **D5 Replication** | Confirmatory replication of EXP-019 on second cohort | Phase C1 dataset acquisition, committed preregistration | **NO** | Phase C dataset identification | Confirmatory replication |

### Critical Governance Enforcements:
- **EXP-020 Anti-Premature Gate:** v4 explicitly forbids running EXP-020 until Phase A close-out and Phase B full-text reading are complete.
- **NAS Closure:** Neural Architecture Search is permanently closed (Model A frozen at 163,940 parameters; NAS Iterations 2–19 remain historical exploratory smoke tests).
- **EXP-012 Test Set:** Formally designated as **SPENT**; cannot be reused for model selection.

---

## 6. Phase E Audit — Paper Synthesis & Reporting Readiness

### Readiness Assessment: **`BLOCKED`**

| Required Component | Current Status | Assessment |
|---|---|---|
| **Final Research Question** | Established | "How much do 2+ labeled earlier nights help smartwatch IHR+Accel staging in healthy adults, who is harmed, and why?" |
| **Primary Contribution (C1)** | Established | Pre-registered replication & extension with consumer IHR+Accel, specificity control, learning curves, failure analysis. |
| **Secondary Contribution (C2)** | Emerging | Stage absence & autonomic instability as failure mechanisms; gating/regularization. |
| **Tertiary Contribution (C3)** | Blocked | External validation (EXP-027) not yet executed. |
| **Prior-Art Positioning** | Partially Established | Positioned relative to van der Aar 2025; closest-ten matrix still missing. |
| **Confirmatory Metrics** | Established | H-A (+1.01 pp, Inconclusive), S1 (+2.73 pp, Supported), S2 (+20.32 pp, Supported). |
| **Descriptive Outputs** | **Missing** | $k=1–4$ curve, confusion matrices, Bland-Altman sleep metrics missing. |
| **S3 Invalidation Reporting** | Established | Implementation no-op documented (`S3_INFOMAX_NOOP_AUDIT_REPORT.md`). |
| **Negative Transfer Analysis** | Established | Donor harm (-1.72 pp) and negative responders (Bidslab34 -16.34 pp) documented. |

**Phase E Classification:** **`BLOCKED`** until Phase A3 descriptive outputs and Phase B closest-ten comparison are landed.

---

## 7. Cross-Phase Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                         PHASE A                             │
│  [A1: H-A Reconciled] ───▶ [A2: S3 Invalidated]             │
│            │                       │                        │
│            ▼                       ▼                        │
│  [A4: S1 Donor Harm]  ───▶ [A5: NAS Closed]                 │
│            │                                                │
│            ▼                                                │
│  [A3: Descriptive Outputs (k=1-4, Conf. Mat, Bland-Altman)] │
└──────────────────────────────┬──────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌─────────────────────────────┐ ┌─────────────────────────────┐
│           PHASE B           │ │           PHASE C           │
│  [B1: KB Errata / PMIDs]    │ │  [C1: Second Dataset Search]│
│  [B2: Core Full Texts]      │ │  (PPG/IHR + Accel + PSG     │
│  [B3: Closest-10 Table]     │ │   >= 3 nights/subject)      │
│  [B4: Dated Novelty Stmt]   │ │                             │
└──────────────┬──────────────┘ └──────────────┬──────────────┘
               │                               │
               └───────────────┬───────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                         PHASE D                             │
│  [D1: EXP-020 (KL / Gating, Exploratory, Prereg Committed)] │
│  [D2: EXP-019B (Heterogeneity Tests, Benjamini-Hochberg)]   │
│  [D3: EXP-027 (SleepAccel Zero-Shot Transfer)]              │
│  [D4: EXP-028 (Clinical Agreement Margins)]                 │
│  [D5: Confirmatory Replication (IF Phase C succeeds)]       │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                         PHASE E                             │
│  [Paper Synthesis, Artifact Archive, Final Manuscript]      │
└─────────────────────────────────────────────────────────────┘
```

### Dependency Rules:
1. **Phase A3 is the Immediate Serial Bottleneck:** Phase D cannot begin until Phase A is closed.
2. **Phase B can run in Parallel with Phase A3:** Literature retrieval and KB cleanup do not require GPU execution.
3. **Phase C is an Independent Track:** If a second dataset is found, D5 provides a confirmatory replication; if not, the paper is framed on C1 + C2 + external zero-shot (SleepAccel).

---

## 8. Master Status Matrix

| Phase | Objective | Current Status | Completed | Missing | Blocked By | Required Next Action | Correct Top-Level EGM Skill |
|---|---|---|---|---|---|---|---|
| **Phase A** | EXP-019 Closure & Descriptive Reporting | **NOT READY TO CLOSE** | A1 (H-A reconciled), A2 (S3 invalidated), A4 (S1 donor harm), A5 (NAS closed) | A3 ($k=1–4$ curve, confusion matrices, Bland-Altman agreement, night index) | Uncomputed descriptive metrics | Author and execute Phase A3 diagnostic extraction script | **`egm-draft`** (to draft Phase A3 extraction plan) / **`egm-review`** (to audit outputs) |
| **Phase B** | Prior-Art Evidence Base & Novelty Rigor | **PARTIALLY READY** | Novelty claim reframed, van der Aar ingested (`P106`), 106 records cataloged | Full-text retrieval (85/106 pending), Heremans PMID fixes, Closest-10 table, Novelty statement | Paywalled / unretrieved full texts | Retrieve full texts of core 4 papers and build closest-ten matrix | **`egm-draft`** (to construct closest-ten comparison) |
| **Phase C** | Second Multi-Night Wearable Dataset | **NOT STARTED** | Cataloged single-night PSG cohorts (SleepAccel, MESA) | External dataset with $\ge 3$ nights, PPG/IHR, accel, and PSG labels | No open dataset identified | Systematic search of data availability in forward citations | **`egm-review`** (to audit candidate datasets) |
| **Phase D** | Exploratory Extensions & Mitigations | **STRICTLY BLOCKED** | Architecture freeze, Model A checkpoints, DAG hypothesis formulation | Pre-registration blocks for EXP-020, EXP-027, EXP-028 | Phase A close-out; Phase B full texts; Phase C search | Complete Phase A3 and Phase B before formulating preregistrations | **BLOCKED** (Do not invoke `egm-brainstorm`) |
| **Phase E** | Paper Synthesis & Final Manuscript | **BLOCKED** | Primary contribution structure, statistical results, negative transfer framing | A3 figures/tables, Closest-10 table, SleepAccel transfer results | Phase A3, Phase B, Phase D3 | Complete experimental and literature deliverables | **BLOCKED** |

---

## 9. Governance Audit & Contradictions Report

### Contradiction Findings Across Documents:

1. **Heremans et al. (2022) PubMed ID Discrepancy (Phase B Contradiction):**
   - In `research_knowledge/papers/P054_from_unsupervised_to_semi_supervised_adv.md`, the PubMed ID is correctly updated to `35508121`.
   - However, in `literature_map.md`, `methods_catalog.md`, `competing_hypotheses.md`, `research_gaps.md`, and `novelty_registry.md`, the wrong URL `https://pubmed.ncbi.nlm.nih.gov/35877770/` remains.
   - **Resolution Required:** Update the five summary files to reference PMID `35508121` during Phase B1.

2. **EXP-019 Final Report Inline Label:**
   - Section 1 summary table lists H-A as "NOT SUPPORTED" and S3 as "NOT SUPPORTED" with footnote references pointing to Section 6 Erratum.
   - **Status:** Epistemically valid because Section 6 formalizes the erratum; no conflict as long as Section 6 is read as governing.

3. **Preregistration File vs Early Report:**
   - The locked preregistration text (`4dff5b7`) explicitly requires `CI_high < +1.50` for NOT SUPPORTED.
   - The early report's check of only `mean >= 1.50` was the single root cause of the labeling contradiction, now resolved by `DEC-029`.

---

## 10. Scientific Claim Audit

### 1. Claims Currently Supported by Verified Evidence
- Supervised two-night within-person adaptation ($FT\text{-}2$) improves Macro-F1 by a population mean of $+1.01\text{ pp}$ ($64.58\% \to 65.59\%$, $p = 0.00131$), with $73.9\%$ of subjects (34/46) improving.
- Personal fine-tuning is biologically specific: adapting on an individual's own nights provides a $+2.73\text{ pp}$ advantage over adapting on another person's nights ($p = 9.0 \times 10^{-6}$).
- Fine-tuning on a donor individual actively degrades performance by $-1.72\text{ pp}$ below the zero-shot baseline ($62.86\%$ vs $64.58\%$).
- Cohort pre-training provides an indispensable foundation ($+20.32\text{ pp}$ advantage over per-subject scratch training, $p < 10^{-10}$).
- Full-night context ($L=1200$) significantly improves representation over truncated crops ($L=60$, $+5.33\text{ pp}$).
- Model A (`ConventionalCNNBiGRU`, 163.9k params) matches complex gated and transition architectures, establishing that architectural capacity was not the performance bottleneck.

### 2. Claims Currently Unsupported or Invalidated
- **INVALIDATED:** Claim that EXP-019 S3 proved "unsupervised InfoMax adaptation fails" or that "unsupervised adaptation requires supervision." (Invalidated as an implementation no-op).
- **UNSUPPORTED:** Claiming multi-night personalization on wearables is an "open or underexplored lane." (Preceded by van der Aar et al. 2025).
- **UNSUPPORTED:** Claiming parity with published SLAMSS-IFS (71.04%) without an independent re-run on identical splits under matched class definitions.
- **UNSUPPORTED:** Claiming that supervised adaptation is ready for clinical deployment without evaluating Bland-Altman agreement on clinical sleep summary metrics.
- **UNSUPPORTED:** Claiming real-time sleep staging capability (model uses bidirectional GRU requiring full future sequence context).
- **UNSUPPORTED:** Generalizing findings to sleep-disordered clinical populations (BIDSleep consists strictly of healthy adults).

### 3. Claims that Remain Unresolved (True Uncertainty)
- Whether supervised two-night adaptation achieves practical significance ($\ge +1.50\text{ pp}$) in a larger cohort (H-A is `INCONCLUSIVE`, CI $[-0.17, +1.99]\text{ pp}$).
- Whether unsupervised InfoMax adaptation succeeds or fails when implemented with a valid quality mask (untested).
- Whether calibration-night stage absence is the sole or primary driver of catastrophic failure in negative outliers (exploratory hypothesis from EXP-019B).
- Whether KL-regularization or calibration gating prevents outlier degradation without dampening responder gains.

---

## 11. Final Decision & Next EGM Action

### 1. Current Research Phase
$$\mathbf{PHASE\ A\ (Phase\ A3\ Execution)}$$

### 2. Immediate Next Action
Author and execute the **Phase A3 Diagnostic Extraction Plan** (`extract_exp019_phase_a3_outputs.py`) to compute and report:
1. $k=1–4$ calibration learning curve across eligible cohorts ($n=46, 42, 33$);
2. Cohort confusion matrices and per-stage F1 scores (Wake, Light, Deep, REM);
3. Bland-Altman clinical sleep-summary agreement (TST, SOL, WASO, SE, REM%, Deep%);
4. Night-index evaluation breakdown.

### 3. Correct Top-Level EGM Skill
$$\mathbf{egm\text{-}draft}$$

### 4. Why This is the Correct Skill
`egm-draft` is the mandated EGM skill to construct the formal, structured technical specification and execution plan for Phase A3 (`design_phase_a3_extraction.md`) before running any diagnostic script. It enforces plain-English mechanics, tensor invariants, artifact persistence, and leakage-free controls without premature training.

### 5. What NOT to Do
- **Do NOT invoke `egm-brainstorm`** (the research direction is locked under v4; brainstorming is prohibited).
- **Do NOT launch EXP-020 or train any new neural networks** (strictly gated behind Phase A, B, and C).
- **Do NOT modify locked EXP-019 artifacts, checkpoints, or preregistration files**.
- **Do NOT implement code fixes for S3 in `exp019_adaptation_core.py` or re-run S3**.
- **Do NOT proceed to Phase B full-text reading or Phase D until Phase A3 execution is approved and landed**.

### 6. Overall Research Readiness
$$\mathbf{PARTIALLY\ READY\ (Phase\ A\ Execution\ In\ Progress)}$$
