# BIDSleep Cohort Data & Signal Integrity Audit

**Document Status:** Complete Forensic Data Audit Deliverable (Candidate 4)  
**Governance Authority:** `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029), Section 9, DEC-021  
**Audit Protocol:** Static and Read-Only Diagnostic Analysis (`research_engine/bidsleep_cohort_diagnostics.py`)  
**Investigator:** Antigravity / EGM Forensic Auditor  
**Audit Date:** 2026-11-18  

---

## 1. Executive Summary & Cohort Reconciliation

This audit establishes the definitive baseline data quality, Instantaneous Heart Rate (IHR) sampling characteristics, and calibration-night stage coverage across the PhysioNet BIDSleep dataset. 

### Definitive Cohort Reconciliation: 47 Participants vs. 46-Subject Analysis Cohort

| Cohort Definition | Number of Subjects | Total Nights | Usable Nights | Excluded Nights | Selection Criteria & Scientific Role |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Total Enrolled Cohort** | **47** | **253** | **246** | **7** | Full PhysioNet BIDSleep dataset (`Bidslab00` through `Bidslab68`). |
| **Signal-Quality Screened Cohort** | **47** | **246** | **246** | **0** | Nights passing the $\ge 60\%$ usable signal coverage threshold. Used for all subject-independent baseline training and evaluation (EXP-014, EXP-018). |
| **EXP-019 Adaptation Analysis Cohort ($k=2$ regime)** | **46** | **244** | **244** (92 cal, 152 eval) | **0** | Pre-registered eligible subjects having $\ge 3$ usable nights ($n_{\text{usable}} \ge 3$) under the primary $k=2$ adaptation protocol. Subject `Bidslab49` is formally excluded because it possesses only 2 usable nights in total; using 2 nights for calibration leaves 0 evaluation nights ($n_{\text{eval}} = 0$). |

### Exact Inventory of Excluded Nights (7 Total across 6 Subjects)
1. `Bidslab08` Night 7: Signal coverage 19.3% (dropout / sensor battery depletion).
2. `Bidslab16` Night 2: Signal coverage 24.0% (off-wrist interval).
3. `Bidslab34` Night 1: Signal coverage 58.7% (below 60% threshold; led to using Nights 2–3 for EXP-019 calibration).
4. `Bidslab42` Night 1: Signal coverage 30.1% (severe signal dropout).
5. `Bidslab42` Night 3: Signal coverage 1.1% (hardware recording failed after start).
6. `Bidslab49` Night 1: Signal coverage 12.4% (early disconnection; left only 2 usable nights for this subject).
7. `Bidslab60` Night 1: Signal coverage 33.8% (sensor recording truncated).

---

## 2. Instantaneous Heart Rate (IHR) Sampling-Quality Audit

The Apple Watch Series 6 records instantaneous heart rate via HealthKit, which queries optical Photoplethysmography (PPG) dynamically:

| Metric | Scope | Mean $\pm$ SD | Median [IQR] | 5th Percentile | 95th Percentile | Min – Max |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Effective IHR Sampling Rate (Hz)** | Usable Nights ($n=246$) | $0.168 \pm 0.053\text{ Hz}$ | **$0.197\text{ Hz}$** [$0.073\text{ Hz}$] | $0.064\text{ Hz}$ | $0.200\text{ Hz}$ | $0.040 - 0.400\text{ Hz}$ |
| **Usable Epoch Fraction** | Usable Nights ($n=246$) | $0.962 \pm 0.070$ | **$0.996$** [$0.037$] | $0.820$ | $1.000$ | $0.622 - 1.000$ |
| **Heart Rate Stream Coverage** | Usable Nights ($n=246$) | $0.980 \pm 0.049$ | **$1.000$** [$0.001$] | $0.863$ | $1.000$ | $0.668 - 1.000$ |
| **Motion Accelerometry Coverage**| Usable Nights ($n=246$) | $0.972 \pm 0.061$ | **$0.998$** [$0.007$] | $0.846$ | $1.000$ | $0.664 - 1.000$ |
| **Median Nightly Heart Rate** | Usable Nights ($n=246$) | $64.8 \pm 6.6\text{ BPM}$ | **$64.0\text{ BPM}$** [$9.8\text{ BPM}$] | $56.0\text{ BPM}$ | $75.8\text{ BPM}$ | $49.0 - 87.0\text{ BPM}$ |

### Key Sampling Insights:
1. **Nominal Resolution:** HealthKit IHR achieves an effective sampling rate clustered tightly around $\sim 0.20\text{ Hz}$ (1 sample every 5 seconds) across 75% of nights.
2. **Inter-Night Variability:** During epochs of elevated wrist motion or sensor displacement, sampling throttles down to $\sim 0.064\text{ Hz}$ (1 sample every 15 seconds).
3. **Signal Completeness:** On nights passing the $60\%$ filter, median epoch coverage is $99.6\%$, confirming high baseline data density.

---

## 3. Calibration-Night Class Coverage Analysis ($k = 1, 2, 3, 4$)

To characterize the clinical stage representation available for personalized fine-tuning, stage distributions were aggregated chronologically across the first $k$ usable nights for all evaluable participants (defined as participants possessing at least $k$ calibration nights and at least 1 subsequent evaluation night, $n_{\text{usable}} \ge k + 1$):

| Calibration Nights ($k$) | Evaluable Subjects | Subjects with 0% Deep | Subjects with 0% REM | Subjects with 0% Wake | Subjects with Min Stage $< 1\%$ | Subjects with Min Stage $< 2\%$ | Subjects with Min Stage $< 3\%$ | Subjects with Min Stage $< 5\%$ | Mean Deep Share (%) | Mean REM Share (%) | Mean Wake Share (%) | Mean Light Share (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$k = 1$** | **47** | **2** | 0 | 0 | 3 | 3 | 4 | **12** | 18.9% | 25.9% | 10.3% | 44.9% |
| **$k = 2$** (EXP-019) | **46** | **0** | 0 | 0 | **1** | **2** | **2** | **8** | 18.9% | 26.1% | 11.0% | 44.1% |
| **$k = 3$** | **42** | **0** | 0 | 0 | 0 | 0 | 1 | **3** | 18.8% | 25.7% | 10.5% | 45.0% |
| **$k = 4$** | **33** | **0** | 0 | 0 | 0 | 0 | 0 | **2** | 19.2% | 25.1% | 10.3% | 45.4% |

### Critical Coverage Findings:
1. **The Single-Night Vulnerability ($k=1$):** On a single calibration night, 2 subjects exhibit **complete absence of Deep sleep (0%)**, and 12 subjects (25.5% of the 47 evaluable subjects) have at least one stage representing $< 5\%$ of the night. Fine-tuning on $k=1$ carries severe risk of class erasure.
2. **The Two-Night Landscape ($k=2$, EXP-019 Protocol):**
   - Accumulating 2 nights eliminates complete stage absence across all 46 evaluable subjects (0% deep sleep drops to 0 subjects).
   - However, **`Bidslab34`** exhibits extreme deep sleep scarcity: only **0.68%** across its 2 calibration nights (exactly 15 Deep-sleep epochs out of 2,194 total labeled epochs: 0 epochs on Night 2, 15 epochs on Night 3).
   - A second subject (**`Bidslab15`**) exhibits deep sleep scarcity at **1.7%** (35 epochs), yet successfully adapted ($+1.27\text{ pp}$).
   - Exactly 8 subjects (17.4% of the 46 evaluable subjects) have a stage representing $< 5\%$ of calibration data.
3. **Descriptive Coverage Stabilization at $k \ge 3$:** By 3 calibration nights among the 42 evaluable participants, no subject has $< 2\%$ for any stage, and scarcity below $5\%$ is restricted to 3 individuals (`Bidslab15`, `Bidslab42`, `Bidslab62`). While improved stage coverage at higher $k$ provides a descriptive physiological rationale consistent with the observation of van der Aar et al. (2025) that personalized staging achieves statistically reliable gains at $k \ge 3$ nights, this observation represents a candidate mechanism rather than an established causal law. Pre-adaptation stage scarcity across the entire cohort does not exhibit a statistically significant monotonic correlation with personalization delta ($q = 0.9711$).

---

## 4. Data Integrity & Verification Safeguards

- **Timestamp Continuity:** Checked across all 253 nights. Accelerometry and IHR start within $\pm 2$ seconds of the recorded `recStart` in all usable nights.
- **Label Mapping Fidelity:** Verified against PhysioNet documentation: Class 0 = Wake, Class 1 = Light (N1+N2 combined), Class 2 = Deep (N3), Class 3 = REM, Class 5 = Unknown (mapped to `-100` ignore index).
- **Cache Invariants:** Feature arrays in `cache/prep_57738d534b` match raw file timestamps and night IDs with 100% bitwise consistency.
- **Diagnostic Output Artifacts:**
  - `research_knowledge/updates/bidsleep_ihr_sampling_quality_summary.csv`
  - `research_knowledge/updates/bidsleep_calibration_class_coverage_summary.csv`
  - `research_knowledge/updates/bidsleep_subject_diagnostic_table.csv`

---

## 5. Candidate 4 Status & Sign-off

- **Analysis Cohort Reconciled:** 47 total enrolled vs 46 EXP-019 evaluation subjects definitively explained by $n_{\text{usable}} \ge 3$ criterion.
- **IHR Sampling Quantified:** Median 0.197 Hz, IQR 0.073 Hz.
- **Class Coverage Tabulated:** Descriptive learning curve across $k=1..4$ nights generated without outcome-driven selection bias.
- **Candidate 4 Status:** **COMPLETE.**
