# Stage 9 Implementation Readiness & Codebase Details Audit Report (`MANUSCRIPT_DRAFT_v0.2.md`)

**Auditor Roles:** `egm-codebase-details`, `egm-implementation-readiness`, `egm-style` (TRIPOD+AI Biostatistician & Technical Quality Auditor)  
**Target Document:** [`research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md) (Lines 1–252)  
**Audit Verdict:** **CONDITIONAL PASS PENDING 2 BLOCKERS, 5 MAJORS, AND 3 NITS**

---

## 1. Executive Verification Summary across Audit Axes

| Audit Vector / Specific Prompt Check | Status | Verification Summary |
| :--- | :---: | :--- |
| **`mean_single_model` (PRIMARY) vs. `prob_ensemble` (SECONDARY) on `EXP-027` & `EXP-028`** | **PASS** | Verified in Structured Summary (L20), Research in context (L36), Methods §2.4 (L93), Results §3.6 prose (L179), Table 3 (L184–196), Table 4 (L205–223), and Discussion §4.1 (L230). |
| **`all_n31` `prob_ensemble` Exact Values (`EXP-027`)** | **PASS** | Table 3 (L196) & §3.6 (L179) report `62.94%` vs. `64.14%`, `-1.20 pp`, `95% CI [-3.96, +1.04] pp`, `p=0.5551`, `d_z=-0.17`, `13 / 18 / 0` — exact match to [`exp027_summary.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-027/exp027_summary.json#L313-L324). |
| **Table 1 (`EXP-019` BA, $N=46$, $df=45$)** | **PASS (100%)** | Every cell across all 4 arms $\times$ 6 metrics (24 rows $\times$ 10 numerical columns = 240 values) verified 100% bitwise accurate against [`exp019_sleep_summary_agreement.csv`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-019/exp019_sleep_summary_agreement.csv#L1-L43) and [`bidsleep_within_cohort_agreement_analysis.py`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_engine/bidsleep_within_cohort_agreement_analysis.py#L62-L107) ($t_{0.975,45}=2.0141$). |
| **Table 2 (`EXP-021` & `W0-B`, $N=47$)** | **PASS (100%)** | All 8 classification metrics, parameter counts, footprints (`309.54 KiB` in-memory, `339.99 KiB` file, `160.10 KiB` floor), MMACs (`794.05`), latencies (`47.73 ms` 4T, `69.39 ms` 1T), CIs, and $p$-values verified against [`slamss_ifs_paired_comparison_summary.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-021/slamss_ifs_paired_comparison_summary.json#L8-L184) and [`int8_quantization_profile.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/STAGE2_FINAL_MODEL/int8_quantization_profile.json#L7-L216). |
| **Table 3 (`EXP-027` SleepAccel, $n=25$ & $N=31$)** | **🔴 BLOCKER / 🟡 MAJOR** | Primary `qc_clean` and `all_n31` numbers match [`exp027_summary.json`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-027/exp027_summary.json), **but** 4 of the 5 excluded short-PSG subject IDs in L73 & L198 contradict [`sleepaccel_qc_summary.csv`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-027/sleepaccel_qc_summary.csv) (🔴 Blocker), and L194 compresses Wake/Deep/REM F1 while omitting CIs, $d_z$, wins/losses, and Kappa (🟡 Major). |
| **Table 4 (`EXP-028` SleepAccel BA, $n=25$, $df=24$)** | **🟡 MAJOR** | Primary rows match [`exp028_sleepaccel_bland_altman_summary.csv`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-028/exp028_sleepaccel_bland_altman_summary.csv), **but** secondary rows have a rounding error (`10.25` vs `10.2553 -> 10.26%` on L220), omit the non-parametric LoA bounds `[-19.90, +10.30]` for `sol_min` on L221, omit proportional-bias $p$-values (including significant $p=0.0316$ on L221 and $p=0.0265$ on L222), and use inconsistent decimal precision (1-decimal LoA on L219; 3-decimal $r$ on L219–223). |
| **Section 3.2 `S1`/`S2` CIs & $p$-values vs. `DEC-028` / `CLM-022`** | **🟡 MAJOR** | Section 3.2 (L128–129) reports parametric $t$-intervals (`[+1.01, +4.44]`, `W=169.0, p=1.89e-5` for `S1`; `[+17.60, +23.03]` for `S2`) from `bidsleep_sleep_summary_and_specificity_analysis.py`, contradicting §2.4 (L93, which specifies 10,000-resample bootstrap CIs), `exp019_stage_c_results.json` (`[+1.05, +4.32]`, `p=9.0e-6` for `S1`; `[+17.70, +22.94]` for `S2`), `DEC-028`, and `CLM-022`. |
| **5 Mandatory Disclosures** | **🔴 BLOCKER / 🟡 MAJOR** | Disclosures 1, 3, and 5 pass; Disclosure 4 has an internal manuscript contradiction on BIDSleep Dreem 2 reference priors (`46.2%` Light / `18.6%` Deep on L68 vs. `44.2%` Light / `18.9%` Deep on L36, L200, L232; 🔴 Blocker); Disclosure 2 cites `DEC-039` as an internal code without explaining the prior-subject weight mutation reconciliation (🟡 Major). |

---

## 2. Detailed Findings (Ordered by Severity)

### 🔴 BLOCKER-01: Four of Five Excluded SleepAccel Short-PSG Subject IDs Contradict `sleepaccel_qc_summary.csv`
* **Severity:** 🔴 Blocker (`Contradicts the Code / Non-Existent References`)
* **Affected Scope:** [`MANUSCRIPT_DRAFT_v0.2.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L73) (Section 2.1, Line 73) and [`MANUSCRIPT_DRAFT_v0.2.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L198) (Table 3 footnote, Line 198).
* **Discrepancy Description & Rationale:**
  - Both Line 73 and Line 198 state that the 5 short PSG recordings (`<600` epochs) excluded from SleepAccel `qc_clean` are:
    > ``3951299`, `5132496`, `7596675`, `8443592`, `8692249``
  - Inspection of [`experiments/EXP-027/sleepaccel_qc_summary.csv`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-027/sleepaccel_qc_summary.csv#L1-L33) proves that 4 of these 5 subject IDs **do not exist in PhysioNet SleepAccel v1.0.0** (or are corrupted versions of passing/failing IDs). The actual 6 rows in `sleepaccel_qc_summary.csv` with `qc_pass=False` are:
    1. **`3509524`** (`n_epochs = 417 < 600`, `usable_frac = 1.0`, `qc_pass = False`) — *omitted from manuscript*
    2. **`4018081`** (`n_epochs = 510 < 600`, `usable_frac = 0.9725`, `qc_pass = False`) — *omitted from manuscript*
    3. **`46343`** (`n_epochs = 567 < 600`, `usable_frac = 0.9771`, `qc_pass = False`) — *omitted from manuscript*
    4. **`5132496`** (`n_epochs = 465 < 600`, `usable_frac = 1.0`, `qc_pass = False`) — *correct*
    5. **`759667`** (`n_epochs = 475 < 600`, `usable_frac = 0.9558`, `qc_pass = False`) — *manuscript wrote `7596675` with an extra trailing `5`*
    6. **`7749105`** (`n_epochs = 960`, `usable_frac = 0.1240 < 0.60`, `cov_mo_mean = 0.1365`, `qc_pass = False`) — *correct*
  - Meanwhile, `844359` (manuscript wrote `8443592`) has `n_epochs = 949` and `qc_pass = True`, and `8692923` (manuscript wrote `8692249`) has `n_epochs = 965` and `qc_pass = True`.
  - **Required Fix:** Replace ``3951299`, `5132496`, `7596675`, `8443592`, `8692249`` on Line 73 and Line 198 with the exact verified IDs from `sleepaccel_qc_summary.csv`:
    > ``3509524` [417 epochs], `4018081` [510], `46343` [567], `5132496` [465], `759667` [475]`

---

### 🔴 BLOCKER-02: Internal Manuscript Contradiction on BIDSleep Dreem 2 Reference Stage Distribution (`46.2%` Light, `18.6%` Deep vs. `44.2%` Light, `18.9%` Deep)
* **Severity:** 🔴 Blocker (`Contradicts the Code / Internal Inconsistency in Mandatory Disclosure #4`)
* **Affected Scope:** [`MANUSCRIPT_DRAFT_v0.2.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L68) (Section 2.1, Line 68) vs. Lines 36 (Research in context), 200 (Section 3.6), and 232 (Section 4.1); plus [`MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_EVIDENCE_TO_CLAIM_MATRIX.csv#L20) (`CLM-019` vs. `CLM-047`).
* **Discrepancy Description & Rationale:**
  - In **Section 2.1 (Line 68)** and **`CLM-019`**, BIDSleep's 246 usable nights (`137,801` valid 30-second epochs) are reported as **Wake `11.9%`, Light `46.2%`, Deep `18.6%`, REM `23.2%`**.
  - However, in **Research in context (Line 36)**, **Section 3.6 (Line 200)**, **Section 4.1 (Line 232)**, and **`CLM-047`**, the BIDSleep Dreem 2 reference distribution is cited as **`44.2%` Light, `18.9%` Deep** (the unweighted mean across the 246 usable nights of all labeled EEG epochs in `review_package/night_inventory.csv` before sensor-coverage masking to 137,801 epochs).
  - **Required Fix:** Explicitly disambiguate or harmonize both references (e.g., state `44.2% Light, 18.9% Deep across night-averaged Dreem 2 EEG [46.2% Light, 18.6% Deep across 137,801 sensor-valid pooled epochs] vs. 57.7% Light, 11.1% Deep in SleepAccel in-lab PSG`).

---

### 🟡 MAJOR-01: Table 4 (`EXP-028`) Rounding Error (`10.25` vs. `10.26%`), Omitted Non-Parametric LoA Bounds, Omitted Significant Proportional-Bias $p$-Values, and Inconsistent Precision in Secondary Rows
* **Severity:** 🟡 Major (`Code/Artifact Drift & Incomplete Statistical Reporting`)
* **Affected Scope:** [`MANUSCRIPT_DRAFT_v0.2.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L218-L223) (Table 4, Lines 218–223) vs. [`experiments/EXP-028/exp028_sleepaccel_bland_altman_summary.csv`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-028/exp028_sleepaccel_bland_altman_summary.csv#L8-L25).
* **Required Fix:** Unpack the 5 compressed slash-separated rows in Table 4 (Lines 219–223) into standard 1-metric-per-row entries matching the primary rows, fix `10.25` $\to$ `10.26%`, restore all proportional-bias $p$-values (including significant `-0.531 (0.032)` for `sol_min` on L221 and `-0.672 (0.026)` for `SE` on L222), and restore the `[-19.90, +10.30]` non-parametric LoA fallback on L221.

---

### 🟡 MAJOR-02: Section 3.2 `S1` and `S2` CIs and Wilcoxon $p$-Values Use Parametric $t$-Intervals from `bidsleep_sleep_summary_and_specificity_analysis.py`, Contradicting Section 2.4, `exp019_stage_c_results.json`, `DEC-028`, and `CLM-022`
* **Severity:** 🟡 Major (`Statistical Method & Internal KB Drift`)
* **Affected Scope:** [`MANUSCRIPT_DRAFT_v0.2.md`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/research_knowledge/updates/MANUSCRIPT_DRAFT_v0.2.md#L128-L129) (Section 3.2, Lines 128–129).
* **Required Fix:** Report the locked 10,000-resample percentile bootstrap 95% CIs as primary (`95% bootstrap CI +1.05 to +4.32 pp` for `S1`; `+17.70 to +22.94 pp` for `S2`) alongside the parametric $t$-test / two-sided Wilcoxon values (`t-based 95% CI +1.01 to +4.44, t=3.204, p=0.00249; two-sided Wilcoxon W=169.0, p=1.89×10^-5`, one-sided `p=9.0×10^-6`), harmonizing `CLM-022` and `CLM-035`/`CLM-036`.

---

### 🟡 MAJOR-03: Table 3 (`EXP-027`) `qc_clean` Secondary (`prob_ensemble`) Compresses Wake/Deep/REM F1 (Omitting CIs, $d_z$, and Win/Loss/Tie Counts) and Omits Cohen's Kappa
* **Severity:** 🟡 Major (`Incomplete Tabular Reporting`)
* **Affected Scope:** Table 3 (Lines 191–194).
* **Required Fix:** Expand `qc_clean` `prob_ensemble` in Table 3 to include `Kappa` (`0.5122` vs. `0.5202`, `-0.0079`, `[-0.0349, +0.0188]`, `p=0.6915`, `d_z=-0.11`, `11 / 14 / 0`) and separate rows for `Wake F1`, `Deep F1`, and `REM F1` with their complete CIs, 4-decimal $p$-values, $d_z$, and Win/Loss/Tie counts.

---

### 🟡 MAJOR-04: Mandatory Disclosure #2 (`DEC-039` Forensic Audit Transparency) Cited Only as an Internal Code Without Explaining the Prior-Subject Baseline Mutation Reconciliation
* **Severity:** 🟡 Major (`Missing Mandatory Disclosure Detail / Unstated Assumption`)
* **Affected Scope:** Section 2.3 (Line 89) and Section 3.4 (Lines 163–164).
* **Required Fix:** Add one explicit sentence explaining that a forensic audit of the initial calibration-depth driver (`DEC-039`) identified an in-place weight-mutation defect where un-reset zero-shot evaluations on subjects $2\dots N$ of each fold inherited prior-subject fine-tuned weights ($62.86\%$, matching `FT-other-2`), which was resolved via CPU-only clean zero-shot baseline reconciliation across all 15 frozen checkpoints (`63.54%` on $E_4$; `64.58%` on $E_2$).

---

### 🟡 MAJOR-05: Internal Governance Jargon Codes (`DEC-xxx`, `TSK-xx`, `W0-B`, `H-A`, `E_4`) Used in Manuscript Prose Without Standalone Scientific Definitions
* **Severity:** 🟡 Major (`Naive Reader / Journal Style Readiness`)

---

### 🟢 NIT-01 to NIT-03
- **NIT-01:** Available Cohort $k=2$ Delta (`+0.99 pp`, `33/46` wins) vs. Locked `EXP-019` `FT-2` (`+1.01 pp`, `34/46` wins) harmonization note on Line 171.
- **NIT-02:** `exp019_stage_c_results.json` upper CI bound for `H-A` (`+1.987` vs. `+1.9928 pp`) on Line 125.
- **NIT-03:** Reference 5 incomplete citation format (`Quirk C, et al. Data leakage in deep learning sleep staging. 2024.`) on Line 251.
