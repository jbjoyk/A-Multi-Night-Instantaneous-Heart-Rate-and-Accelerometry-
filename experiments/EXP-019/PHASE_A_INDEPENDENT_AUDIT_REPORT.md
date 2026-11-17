# Independent Forensic & Governance Audit: Phase A Close-Out Evaluation

**Audit Date:** 2026-11-17  
**Auditor:** EGM Review System (Independent Technical Quality & Governance Auditor)  
**Governing Direction:** `RESEARCH_DIRECTION_v4_FULL.md` (Phase A Mandate)  
**Governing Preregistration:** `PREREG_EXP-019_H-A.md` (Locked at Git commit `4dff5b7`)  
**Target Scope:** Complete Phase A Work across A1, A2, S1, S2, Cohort Integrity, Leakage, Statistics, and Phase A3 Prerequisites.

---

## 1. Executive Summary & Phase A Exit Recommendation

### **RECOMMENDATION: NOT READY TO CLOSE**

While Phase A1 (H-A decision-rule label reconciliation under DEC-029), Phase A2 (S3 implementation no-op forensic diagnosis under DEC-030), Phase A4 (S1 donor-harm clarification), and Phase A5 (NAS Iteration 1 erratum) have been executed with forensic precision and governance integrity, **Phase A cannot be closed because Phase A3 remains unfulfilled**. 

Preregistered descriptive deliverables—specifically the $k=1–4$ calibration learning curve, full per-stage F1 scores and confusion matrices, and Bland-Altman clinical sleep-summary agreement (TST, SOL, WASO, SE, REM%, Deep%)—have not yet been computed or reported from the outer test cohort.

---

## 2. Phase A Status Table

| Audit Area | Item | Status | Concrete Empirical Evidence | Problem Identified | Required Action to Close |
|---|---|---|---|---|---|
| **A1** | **H-A Verdict Reconciliation** | **VERIFIED & CLOSED** | Locked rule in `PREREG_EXP-019_H-A.md` (commit `4dff5b7`, Sec. 6); observed $\bar{\Delta} = +1.01\text{ pp}$, 95% CI $[-0.17, +1.99]\text{ pp}$. | Early report checked only binary `mean >= 1.50`, writing `"supported": false` to JSON, which was misreported as "NOT SUPPORTED". Because CI upper bound $+1.99 \ge +1.50$, it fails the NOT SUPPORTED condition. | None. Reconciled to `INCONCLUSIVE` in `DEC-029` and `EXP019_FINAL_REPORT.md` Section 6 erratum. Numerical data remain untouched. |
| **A2** | **S3 Forensic Audit** | **VERIFIED & CLOSED** | Forensic report `experiments/EXP-019/S3_INFOMAX_NOOP_AUDIT_REPORT.md`; `DEC-030`. | `compute_wearable_quality_mask` checked raw BPM limits `30 <= IHR <= 180` on standardized z-score IHR. 0/82,762 epochs passed; adaptation aborted on step 0 (0 updates, 0/163,940 params changed, identical evaluation logits). | None for Phase A. Formally classified as `UNRESOLVED / INVALIDATED (Implementation No-Op)`. Code fixes and re-runs deferred to Phase D. |
| **S1** | **Specificity Control ($FT\text{-}2$ vs $FT\text{-}other\text{-}2$)** | **VERIFIED & CLOSED** | $\bar{\Delta} = +2.73\text{ pp}$, 95% CI $[+1.05, +4.32]\text{ pp}$, Wilcoxon $p = 9.0 \times 10^{-6}$, 36/46 wins. | Risk of overstating $+2.73\text{ pp}$ as pure personal gain. | Clarified in `DEC-029`, `HANDOFF.md`, and `EXPERIMENTS.md`: $FT\text{-}other\text{-}2$ degrades by $-1.72\text{ pp}$ below $ZS$ ($62.86\%$ vs $64.58\%$). Personal gain is $+1.01\text{ pp}$; donor fine-tuning harm is $+1.72\text{ pp}$. |
| **S2** | **Transfer Foundation ($FT\text{-}2$ vs $PS\text{-}2$)** | **VERIFIED & CLOSED** | $\bar{\Delta} = +20.32\text{ pp}$, 95% CI $[+17.70, +22.94]\text{ pp}$, $p < 10^{-10}$, 46/46 wins. | None. Cohort pre-training is indispensable compared to training from scratch on 2 nights. | Framed appropriately in governance documents as a sanity baseline check rather than a novel contribution. |
| **Cohort** | **Subject & Night Counts** | **VERIFIED & INTACT** | Manifest: 47 subjects; evaluated: 46 subjects; evaluation nights: 152; calibration nights: 92. | `Bidslab49` has only 2 usable nights. | Properly excluded per preregistration Section 4 ($n_{\text{usable}} \ge 3$ required). 100% compliant. |
| **Leakage** | **Partitioning & Preprocessing Separation** | **VERIFIED & LEAK-FREE** | 5-fold cross-validation; $C \cap E = \emptyset$; circadian features neutralized; in-memory state restoration via `reset_model_weights`. | None detected. | Evaluation labels were strictly isolated from adaptation and Stage D tuning. |
| **Stats** | **Multiplicity & Threshold Integrity** | **VERIFIED & SOUND** | Stage D $s_{\Delta} = 2.2432\text{ pp}$, continuous power threshold $0.9267\text{ pp}$, locked threshold $+1.50\text{ pp}$; Oldham's $\rho = +0.0376, p = 0.8043$. | S1/S2 p-values were unadjusted in original report table. | Compute and document Holm-adjusted secondary p-values (S1: $p_{\text{adj}} = 1.8 \times 10^{-5}$; S2: $p_{\text{adj}} < 3 \times 10^{-10}$). Both easily survive adjustment. |
| **A3-1** | **$k=1–4$ Learning Curve** | **MISSING** | `run_exp019_stage_c.py` only adapted with $k=2$. | Preregistered descriptive output omitted from Stage C. | **MANDATORY**: Run deterministic lightweight adaptation pass for $k \in \{1, 3, 4\}$ on eligible cohorts ($n=46, 42, 33$). |
| **A3-2** | **Heterogeneity vs ZS & Night Index** | **PARTIAL** | Heterogeneity vs ZS completed in `HETEROGENEITY_AUDIT_REPORT.md` (Oldham's $\rho = +0.0376$). | Night-index breakdown (effect across chronological evaluation nights 1, 2, 3+) was not logged or extracted. | **MANDATORY**: Extract evaluation metrics stratified by evaluation night position (Night 1, Night 2, etc.). |
| **A3-3** | **Per-Stage F1 & Confusion Matrices** | **MISSING** | `exp019_stage_c_results.json` only stored aggregate Macro-F1 per subject; epoch-level confusion matrices were discarded. | Preregistered descriptive requirement omitted. | **MANDATORY**: Run evaluation forward pass with saved checkpoints to log cohort-level confusion matrices and per-stage F1 (Wake, Light, Deep, REM). |
| **A3-4** | **Sleep-Summary Agreement (Bland-Altman)** | **MISSING** | Not computed in `run_exp019_stage_c.py`. | Preregistered clinical agreement analysis (TST, SOL, WASO, SE, REM%, Deep%) omitted. | **MANDATORY**: Compute Bland-Altman mean bias and 95% limits of agreement before vs after adaptation against EEG ground truth. |
| **A3-5** | **Selected Configurations per Fold** | **AVAILABLE** | Complete JSON file `experiments/EXP-019/exp019_stage_d_selected_configs.json`. | None. | Document explicitly in the consolidated Phase A close-out summary. |
| **A3-6** | **Holm-Adjusted Secondary p-values** | **RECONSTRUCTABLE** | S1 ($p = 9.0 \times 10^{-6}$), S2 ($p < 10^{-10}$), S3 ($p = \text{NaN}$, no-op). | Formal adjusted table omitted from report. | Formalize in Phase A close-out table: S2 $p_{\text{adj}} < 3 \times 10^{-10}$, S1 $p_{\text{adj}} = 1.8 \times 10^{-5}$. |
| **A3-7** | **Stage-D Spread Estimate** | **AVAILABLE** | `empirical_s_delta = 2.2432 pp` recorded in `exp019_stage_d_selected_configs.json` across 25 subjects. | None. | Document explicitly alongside Phase A report. |
| **A3-8** | **Donor-Harm Interpretation** | **AVAILABLE** | Recorded in `DEC-029`, `HANDOFF.md`, and `EXPERIMENTS.md`. | None. | Document the exact $+1.01\text{ pp}$ personal gain vs $+1.72\text{ pp}$ donor degradation decomposition. |
| **A4** | **S1 Donor-Harm Documentation** | **VERIFIED & CLOSED** | Documented in `DEC-029`, `HANDOFF.md`, and `EXPERIMENTS.md`. | None. | Already completed. |
| **A5** | **NAS Iteration 1 Erratum** | **VERIFIED & CLOSED** | `EXPERIMENTS.md` lines 724–729; `DEC-026` erratum; `RESEARCH_DIRECTION_v4_FULL.md`. | None. | Architecture search confirmed closed; Model A frozen at 163,940 parameters. |

---

## 3. Scientific Validity Assessment

1. **Hypothesis H-A ($FT\text{-}2$ vs $ZS$): VALID & CORRECTLY CLASSIFIED**
   - The confirmatory numerical results ($\bar{\Delta} = +1.01\text{ pp}$, 95% CI: $[-0.17, +1.99]\text{ pp}$, Wilcoxon $p = 0.00131$, win rate 73.9%) are methodologically robust and free of temporal/spatial leakage.
   - Classification under the locked preregistered decision rule (`4dff5b7`, Section 6) is strictly **`INCONCLUSIVE`** because the upper confidence bound ($+1.99\text{ pp}$) is greater than the practical significance threshold ($+1.50\text{ pp}$). It failed to meet the support criterion ($\bar{\Delta} \ge +1.50\text{ pp}$ and $\text{CI}_{\text{low}} > 0$), but also failed to rule out an effect of practical size ($\text{CI}_{\text{high}} < +1.50\text{ pp}$).
   - The governance label reconciliation is mathematically accurate.

2. **Secondary Hypothesis S1 (Specificity: $FT\text{-}2$ vs $FT\text{-}other\text{-}2$): VALID & PROPERLY INTERPRETED**
   - Personal fine-tuning significantly outperforms donor fine-tuning ($\bar{\Delta} = +2.73\text{ pp}$, 95% CI: $[+1.05, +4.32]\text{ pp}$, $p = 9.0 \times 10^{-6}$).
   - Epistemic nuance is properly observed: fine-tuning on another individual's nights causes active harm ($-1.72\text{ pp}$ below zero-shot baseline, $62.86\%$ vs $64.58\%$). The $+2.73\text{ pp}$ difference is correctly decomposed as $+1.01\text{ pp}$ personal benefit plus $+1.72\text{ pp}$ avoidance of donor degradation.

3. **Secondary Hypothesis S2 (Transfer Foundation: $FT\text{-}2$ vs $PS\text{-}2$): VALID & PROPERLY FRAMED**
   - $FT\text{-}2$ vastly outperforms training per-subject from scratch ($\bar{\Delta} = +20.32\text{ pp}$, 95% CI: $[+17.70, +22.94]\text{ pp}$, $p < 10^{-10}$, 46/46 wins).
   - Validates that population-level representation learning is indispensable. Properly characterized as a baseline sanity check rather than a primary claim.

4. **Secondary Hypothesis S3 (Unsupervised InfoMax: $UA\text{-}2$ vs $ZS$): INVALIDATED (NO-OP)**
   - The Phase A2 forensic audit proved unequivocally that S3 was an implementation no-op. Due to a physiological unit mismatch in `compute_wearable_quality_mask` ($30 \le \text{IHR} \le 180$ raw BPM vs $[-3.95, +9.94]$ standardized z-score), 0 out of 82,762 calibration epochs passed the mask.
   - Adaptation terminated at step 0 across all 46 subjects. Zero optimizer steps were taken; zero parameters were updated.
   - S3 is **scientifically uninformative**. It provides zero evidence against (or for) unsupervised adaptation.

5. **No Other Latent Validity Problems in Confirmatory Runs:**
   - Pretrained checkpoints (`modelA_L1200_fold{1..5}_seed{42,123,2024}.pt`) are intact and verifiable.
   - Fold splits, night orderings, and subject-level aggregations are strictly leak-free.

---

## 4. Governance Assessment

The repository state was audited across all governance records:
- **`DEC-029` (Adoption of v4 & Reconciliation):** Accurate and consistent.
- **`DEC-030` (S3 Invalidation):** Accurately ratifies the forensic audit; correctly mandates epistemic neutrality on unsupervised adaptation.
- **Preregistration (`PREREG_EXP-019_H-A.md`):** Bitwise identical to Git commit `4dff5b7`.
- **Numerical Result Artifacts:** `exp019_stage_c_results.json`, `exp019_subject_metrics.csv`, `exp019_summary.csv`, and `exp019_stage_d_selected_configs.json` remain completely unmodified.
- **Current Research State (`RESEARCH_STATE.md`) & Handoff (`HANDOFF.md`):** Accurately reflect v4 direction, H-A inconclusive status, S3 invalidation, and pending Phase A3 work.
- **Research Knowledge Base:** `novelty_registry.md`, `research_gaps.md`, `competing_hypotheses.md`, and `P007` review notes have been scrubbed of premature negative claims regarding unsupervised adaptation.
- **Contradictions Identified:** **Zero residual contradictions**.

---

## 5. Missing Phase A Work

To formally close Phase A, the following tasks must be addressed:

### A. Mandatory Blocking Tasks (Required by Preregistration & v4)
1. **Extract Missing Preregistered Descriptive Outputs (Phase A3):**
   - **$k=1–4$ Learning Curve:** Execute lightweight deterministic adaptation for $k \in \{1, 3, 4\}$ on eligible cohorts ($n=46$ for $k=1$, $n=42$ for $k=3$, $n=33$ for $k=4$) using the locked Stage D configurations. Report mean gain, 95% CI, and responder count per $k$.
   - **Per-Stage F1 & Confusion Matrices:** Execute an inference forward pass with the saved checkpoints and deterministic FT-2 weights to output pooled and per-fold confusion matrices, alongside stage-specific F1 scores (Wake, Light, Deep, REM).
   - **Bland-Altman Clinical Sleep-Summary Agreement:** Compute Total Sleep Time (TST), Sleep Onset Latency (SOL), Wake After Sleep Onset (WASO), Sleep Efficiency (SE), REM%, and Deep% before vs after adaptation against EEG ground truth.
   - **Night-Index Effect:** Calculate adaptation gain as a function of evaluation night position (Evaluation Night 1 vs 2 vs 3+).
2. **Author Formal Phase A Close-Out Report:**
   - Consolidate all Phase A deliverables (A1 reconciliation erratum, A2 no-op report, A3 descriptive results, A4 donor harm decomposition, A5 NAS closure) into a single final close-out document (`experiments/EXP-019/PHASE_A_CLOSEOUT_REPORT.md`).

### B. Optional Non-Blocking Tasks
1. **Holm Multiplicity Formatted Output:** Present the formalized Holm-Bonferroni table ($p_{\text{adj}}$ for S1, S2, S3) in the close-out document (already mathematically verified).
2. **Stage D Variance Summary:** Formalize the reporting of the 25 inner-validation subject deltas and $s_{\Delta} = 2.2432\text{ pp}$.

### C. Exploratory Non-Blocking Tasks
1. **EXP-019B Heterogeneity Integration:** Reference findings from `HETEROGENEITY_AUDIT_REPORT.md` (stage absence as candidate root cause; Oldham's correlation refuting RTM) as contextual background for future Phase D1/D2 designs.

---

## 6. Exit Criteria Checklist for Phase A Close-Out

```text
[x] Phase A1: Settle H-A verdict label under preregistered rule (INCONCLUSIVE) and log erratum.
[x] Phase A2: Forensic audit of S3 execution path completed and ratified (DEC-030).
[x] Phase A4: Clarify S1 specificity nuance (avoid overstating personal gain; isolate donor harm).
[x] Phase A5: Confirm NAS Iteration 1 erratum and permanent closure of architecture search.
[ ] Phase A3: Extract and report k=1–4 learning curve across eligible cohorts.
[ ] Phase A3: Extract and report cohort confusion matrices and per-stage F1 scores.
[ ] Phase A3: Compute and report Bland-Altman clinical sleep-summary agreement.
[ ] Phase A3: Extract and report evaluation night-index breakdown.
[ ] Final Phase A Close-Out Report published and committed.
```

**Verdict:** Maintain **Phase A Open** until Phase A3 descriptive analyses are executed and reported. Do NOT advance to Phase B or Phase D until Phase A3 execution is authorized and finalized.
