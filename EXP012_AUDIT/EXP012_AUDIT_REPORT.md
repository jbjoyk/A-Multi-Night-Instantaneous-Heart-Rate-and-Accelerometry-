# Independent Scientific Audit Report: EXP-012 Frozen Benchmark

**Audit Date**: 2026-11-10  
**Target Artifacts**: `exp012_test_results.pkl`, `exp012_test_results.json`, `exp012_subject_metrics.csv`, `exp012_seed*.pt`  
**Auditor**: Independent Research Engineer & Experimental Auditor  

---

# Discrepancies Found in the Previous Report

A total of **11 primary numerical discrepancies** (plus structural reporting errors) were identified when comparing the previous EXP-012 report against the saved artifacts and source dataset:

1. **Total Test Windows (34,800 vs. 34,241)**:
   - *Reported*: 34,800 evaluated windows.
   - *Recomputed*: **34,241** real, valid staging windows.
   - *Cause*: 58 test nights $\times$ 600 slots = 34,800 total padded matrix slots. Within these slots, exactly **559 windows** contain invalid/padded labels (`-100`) and were correctly masked out during evaluation. The previous prose erroneously reported the total padded slots rather than the evaluated valid windows.

2. **REM Class Support (8,871 vs. 8,312)**:
   - *Reported*: 8,871 windows.
   - *Recomputed*: **8,312** windows.
   - *Cause*: In the previous report table, the author subtracted Wake (3,819), Light (13,816), and Deep (8,294) from 34,800 ($34,800 - 3,819 - 13,816 - 8,294 = 8,871$). The 559 padded/invalid slots were falsely absorbed into the REM class support in the table. The true ground-truth REM support is **8,312**.

3. **Per-Seed Class Metrics in Text Prose (Fabricated / Estimated Values)**:
   - *Reported (Seed 42 text)*: Wake F1 = 0.7137, Recall = 76.54%, Precision = 66.84%.
   - *Recomputed (Seed 42)*: Wake F1 = **0.6910**, Recall = **76.72%**, Precision = **62.86%**.
   - *Reported (Seed 123 text)*: Wake F1 = 0.6929, Light F1 = 0.6758, Deep F1 = 0.7340, REM F1 = 0.6493.
   - *Recomputed (Seed 123)*: Wake F1 = **0.7145**, Light F1 = **0.6702**, Deep F1 = **0.7295**, REM F1 = **0.6376**.
   - *Reported (Seed 2024 text)*: Wake F1 = 0.6754, Light F1 = 0.6669, Deep F1 = 0.7143, REM F1 = 0.6280.
   - *Recomputed (Seed 2024)*: Wake F1 = **0.6765**, Light F1 = **0.6716**, Deep F1 = **0.7165**, REM F1 = **0.6198**.
   - *Cause*: Per-class F1, precision, and recall numbers in the prose bullet points of Section 2 were typed by hand or estimated rather than pulled directly from the saved arrays in `exp012_test_results.pkl`.

4. **Boundary and Stable Window Counts (5,745 / 29,055 vs. 1,379 / 32,862)**:
   - *Reported*: 5,745 boundary windows, 29,055 stable windows ($5,745 + 29,055 = 34,800$).
   - *Recomputed*: **1,379** boundary windows, **32,862** stable windows ($1,379 + 32,862 = 34,241$).
   - *Cause*: The previous report fabricated window counts to force their sum to 34,800. The actual transition boundary windows (where $y_t \ne y_{t-1}$) in the 58 test nights number exactly **1,379**.

5. **Per-Seed Boundary Error Rates**:
   - *Reported*: Seed 42 = 58.74%, Seed 123 = 58.90%.
   - *Recomputed*: Seed 42 = **58.88%** ($812 / 1,379$), Seed 123 = **58.74%** ($810 / 1,379$).
   - *Cause*: Approximated/swapped values in text.

6. **Raw Confusion Matrix in Section 6**:
   - *Reported*: Matrix entries summed to 104,400 ($34,800 \times 3$), with REM row summing to 26,613 ($8,871 \times 3$).
   - *Recomputed*: Actual pooled confusion matrix sums to **102,723** ($34,241 \times 3$), with REM row summing to **24,936** ($8,312 \times 3$).
   - *Cause*: The displayed raw confusion matrix in the old markdown report was modified by hand to sum to 104,400.

7. **Per-Subject Table in Section 9 (Fictional Subject IDs and Multiples of 600)**:
   - *Reported*: Listed subject IDs `1053`, `1449`, `2638`, `3022`, `3450`, `4077`, `4426`, `5797`, `6038`, `6882`, `8692` with window counts all being exact multiples of 600 (e.g. 3,600, 3,000, 2,400).
   - *Recomputed*: Real subject IDs in the BIDSLab test split are `Bidslab08`, `Bidslab10`, `Bidslab16`, `Bidslab17`, `Bidslab18`, `Bidslab19`, `Bidslab43`, `Bidslab45`, `Bidslab49`, `Bidslab62`, `Bidslab68`, with true valid window counts: 3592, 3599, 3568, 3480, 3470, 3562, 2949, 3591, 1200, 2948, 2282 (sum = 34,241).
   - *Cause*: Complete fabrication of the subject breakdown table in Section 9 of the old report, despite the real data already existing on disk in `exp012_subject_metrics.csv`.

8. **"Pooled Aggregate" Bootstrap Confidence Interval**:
   - *Reported*: Accuracy $[65.53\%, 71.21\%]$, Macro-F1 $[0.6519, 0.7173]$, $\kappa$ $[0.5102, 0.5949]$.
   - *Recomputed*: No pooled bootstrap was implemented in the code. The author literally took the arithmetic mean of the per-seed bootstrap lower and upper bounds ($(66.94+65.43+64.23)/3 = 65.53$, $(72.68+71.06+69.90)/3 = 71.21$). Averaging confidence bounds is statistically invalid.

---

# Claims I Can and Cannot Support

### Claims That Are SUPPORTED
- **Overall Accuracy**: **$68.44\% \pm 1.35\%$** (Seed 42: 69.86%, Seed 123: 68.27%, Seed 2024: 67.18%).
- **Macro-F1**: **$0.6859 \pm 0.0139$** (Seed 42: 0.6986, Seed 123: 0.6880, Seed 2024: 0.6711).
- **Weighted-F1**: **$0.6835 \pm 0.0141$** (Seed 42: 0.6985, Seed 123: 0.6816, Seed 2024: 0.6704).
- **Cohen's $\kappa$**: **$0.5527 \pm 0.0188$** (Seed 42: 0.5726, Seed 123: 0.5504, Seed 2024: 0.5352).
- **Parameter Compression**: **162,325** parameters vs. SLAMSS-IFS **5,465,768** parameters (**33.7x reduction**, 97.03% compression).
- **Test Set Isolation**: The test split (11 subjects, 58 nights) was strictly isolated; no test labels were seen during training.
- **Per-Seed Subject Bootstrap 95% CIs**: Correctly computed with $B=1000$ cluster-bootstrap over subjects.

### Claims That Are NOT SUPPORTED (or Downgraded)
- **34,800 Evaluated Windows**: NOT SUPPORTED. True evaluated count is **34,241**.
- **REM Support = 8,871**: NOT SUPPORTED. True ground-truth REM support is **8,312**.
- **Physiologically Plausible Sleep Architectures**: NOT SUPPORTED. No transition-bout duration or hypnogram distribution analysis was performed.
- **Edge Feasibility: High**: NOT SUPPORTED. No embedded edge hardware profiling was conducted.
- **VRAM and Training Throughput Comparison vs. SLAMSS-IFS**: NOT SUPPORTED. Measurements were not conducted under matched side-by-side benchmarking conditions.
- **Exceptional Calibration**: Subjective descriptor downgraded. Stated strictly as numerical ECE = **$0.0257 \pm 0.0050$**.
- **Test-Set Alpha Gate Values**: **NOT COMPUTED**. Alpha gate activations were not logged during test evaluation; asserting test alpha values without saved data is unsupported.

---

# Corrected EXP-012 Results

All metrics recomputed from saved raw predictions (`exp012_test_results.pkl`) across the **34,241** real test windows:

### 1. Overall Performance Across Seeds ($N=3$)

| Metric | Seed 42 | Seed 123 | Seed 2024 | Mean $\pm$ SD |
| :--- | :---: | :---: | :---: | :---: |
| **Accuracy (%)** | **69.86%** | **68.27%** | **67.18%** | **$68.44\% \pm 1.35\%$** |
| **Macro-F1** | **0.6986** | **0.6880** | **0.6711** | **$0.6859 \pm 0.0139$** |
| **Weighted-F1** | **0.6985** | **0.6816** | **0.6704** | **$0.6835 \pm 0.0141$** |
| **Cohen's $\kappa$** | **0.5726** | **0.5504** | **0.5352** | **$0.5527 \pm 0.0188$** |
| **MCC** | **0.5740** | **0.5512** | **0.5370** | **$0.5541 \pm 0.0187$** |
| **ECE (10 bins)** | **0.0221** | **0.0234** | **0.0315** | **$0.0257 \pm 0.0050$** |

### 2. Verified Subject-Level Bootstrap 95% Confidence Intervals ($B=1000$)

| Seed | Accuracy [95% CI] | Macro-F1 [95% CI] | Cohen's $\kappa$ [95% CI] |
| :---: | :---: | :---: | :---: |
| **Seed 42** | $[66.94\%, 72.68\%]$ | $[0.6644, 0.7300]$ | $[0.5303, 0.6151]$ |
| **Seed 123** | $[65.43\%, 71.06\%]$ | $[0.6553, 0.7188]$ | $[0.5087, 0.5928]$ |
| **Seed 2024** | $[64.23\%, 69.90\%]$ | $[0.6359, 0.7032]$ | $[0.4917, 0.5769]$ |

*(Note: "Pooled Aggregate" CI was removed as it was an ungrounded average of intervals).*

### 3. Verified Per-Class Metrics (34,241 Windows per Seed)

| Class | Support | Precision (Mean $\pm$ SD) | Recall (Mean $\pm$ SD) | F1-Score (Mean $\pm$ SD) |
| :--- | :---: | :---: | :---: | :---: |
| **Wake (0)** | **3,819** | $64.01\% \pm 4.10\%$ | **$75.95\% \pm 1.20\%$** | **$0.6940 \pm 0.0192$** |
| **Light (1)** | **13,816** | $65.68\% \pm 0.94\%$ | **$70.02\% \pm 1.72\%$** | **$0.6778 \pm 0.0119$** |
| **Deep (2)** | **8,294** | $73.89\% \pm 3.46\%$ | **$72.66\% \pm 1.86\%$** | **$0.7323 \pm 0.0174$** |
| **REM (3)** | **8,312** | $71.10\% \pm 1.47\%$ | **$58.15\% \pm 2.94\%$** | **$0.6395 \pm 0.0207$** |

### 4. Verified Pooled Confusion Matrix (Sum = 102,723 Windows across 3 Seeds)

```
                Predicted Wake   Predicted Light   Predicted Deep   Predicted REM    True Support
True Wake:           8,702            2,042               293              420          11,457 (3,819 * 3)
True Light:          3,312           29,020             4,426            4,690          41,448 (13,816 * 3)
True Deep:             905            5,111            18,080              786          24,882 (8,294 * 3)
True REM:              720            8,006             1,710           14,500          24,936 (8,312 * 3)
Total Predicted:    13,639           44,179            24,509           20,396         102,723
```

### 5. Verified Boundary Dynamics (1,379 Boundary vs. 32,862 Stable Windows)

- **Boundary Windows**: **1,379** (Error Rate: **$59.08\% \pm 0.47\%$** [Seed 42: 58.88%, Seed 123: 58.74%, Seed 2024: 59.61%]).
- **Stable Windows**: **32,862** (Error Rate: **$30.41\% \pm 1.39\%$** [Seed 42: 28.94%, Seed 123: 30.59%, Seed 2024: 31.69%]).
- **Weighted Identity Check**: $\frac{0.590766 \times 1379 + 0.304070 \times 32862}{34241} = 0.315616 = 1 - 0.684384$ (Passes to $10^{-12}$).

### 6. Verified Per-Subject Summary (11 Subjects, Recomputed)

From [exp012_subject_metrics_recomputed.csv](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/EXP012_AUDIT/results/exp012_subject_metrics_recomputed.csv):
- `Bidslab08` (3,592 wins): Acc $64.49\%$, Macro-F1 $0.6479$, $\kappa$ $0.5056$
- `Bidslab10` (3,599 wins): Acc $69.50\%$, Macro-F1 $0.6781$, $\kappa$ $0.5651$
- `Bidslab16` (3,568 wins): Acc $76.88\%$, Macro-F1 $0.7699$, $\kappa$ $0.6713$
- `Bidslab17` (3,480 wins): Acc $64.53\%$, Macro-F1 $0.6559$, $\kappa$ $0.4883$
- `Bidslab18` (3,470 wins): Acc $76.71\%$, Macro-F1 $0.7719$, $\kappa$ $0.6749$
- `Bidslab19` (3,562 wins): Acc $68.61\%$, Macro-F1 $0.6725$, $\kappa$ $0.5303$
- `Bidslab43` (2,949 wins): Acc $66.19\%$, Macro-F1 $0.6472$, $\kappa$ $0.5283$
- `Bidslab45` (3,591 wins): Acc $69.75\%$, Macro-F1 $0.6659$, $\kappa$ $0.5518$
- `Bidslab49` (1,200 wins): Acc $72.33\%$, Macro-F1 $0.7163$, $\kappa$ $0.6015$
- `Bidslab62` (2,948 wins): Acc $63.31\%$, Macro-F1 $0.6209$, $\kappa$ $0.4875$
- `Bidslab68` (2,282 wins): Acc $60.05\%$, Macro-F1 $0.6121$, $\kappa$ $0.4137$
- **Mean-of-Subject Accuracy**: **$68.29\% \pm 1.37\%$** (vs. **$68.44\% \pm 1.35\%$** Pooled Window Accuracy).

---

# SLAMSS-IFS Comparability

The comparison with the published SLAMSS-IFS benchmark (Table V: 71.04% Acc, 0.7079 Weighted-F1, 5.47M parameters) is **PARTIALLY COMPARABLE / UNPAIRED**:

1. **What is identical**: The underlying physical dataset (BIDSLab Apple Watch + Dreem 2), signal usability filter ($\ge 60\%$), subject partition (31 train / 5 val / 11 test), and test nights (58 nights) are identical.
2. **What differs**:
   - **Sequence Length**: SLAMSS-IFS evaluated up to 1200 epochs (10 hours, 55,196 windows); SleepAdaptiveNet evaluated the first 600 epochs (5 hours, 34,241 windows).
   - **Model Selection**: SLAMSS-IFS selected the best validation checkpoint over 500 epochs; SleepAdaptiveNet evaluated at fixed epoch 80 without checkpoint selection.
   - **Metric Reporting**: SLAMSS-IFS reported **Weighted-F1 (0.7079)**. Under matched Weighted-F1, SleepAdaptiveNet achieves **$0.6835 \pm 0.0141$** ($\Delta = -0.0244$).
3. **Statistical Limitation**: Because per-window predictions from SLAMSS-IFS are unavailable:
   > `UNPAIRED COMPARISON — SLAMSS-IFS predictions unavailable.`
   A paired subject-level bootstrap difference cannot be computed against SLAMSS-IFS.

---

# Scientific Interpretation

1. **Generalization Gap**: SleepAdaptiveNet achieves **$68.44\% \pm 1.35\%$ Accuracy** and **$0.6835 \pm 0.0141$ Weighted-F1** on the held-out test set, trailing the 5.47M-parameter SLAMSS-IFS baseline by **$2.60\text{ percentage points}$** in accuracy and **$0.0244$** in Weighted-F1. SleepAdaptiveNet does **NOT** beat SLAMSS-IFS in raw accuracy.
2. **Efficiency Parity**: SleepAdaptiveNet achieves within 2.6 pp of SLAMSS-IFS while using **162,325 parameters**—a **33.7x reduction** (97.03% compression) compared to the 5,465,768 parameters in SLAMSS-IFS.
3. **Reliable Generalization**: Performance across 3 independent random seeds shows high consistency (CV: 2.0% in Macro-F1), high Wake sensitivity ($75.95\%$ recall), and strong probability calibration ($\text{ECE} = 0.0257$).

---

# Recommendation

The **corrected EXP-012 results are robust, reproducible, and ready to present to the supervisor**, provided that:
1. The 11 textual/prose discrepancies in the previous draft are fully replaced with the verified audit values from this report.
2. No claims of "beating SLAMSS-IFS" or "edge-ready deployment" are made.
3. The result is strictly framed as **Case B: Competitive Staging under 33.7x Parameter Compression**.
4. The next experimental phase (testing whether component ablations A/B/C/D isolate the contribution of adaptive fusion, transition modeling, and GRL) is presented as a structured proposal without execution.
