# SLAMSS-IFS Comparability Audit

This document provides a protocol-level comparability analysis between the published **SLAMSS-IFS** benchmark (IEEE Transactions on Biomedical Engineering, 2026, Table V) and the **SleepAdaptiveNet** evaluation (`B0_GRL`, EXP-012).

---

## 1. Protocol Comparison Table

| Protocol Element | SLAMSS-IFS (Official Paper & Repository) | SleepAdaptiveNet (EXP-012) | Identical? | Impact / Notes |
| :--- | :--- | :--- | :---: | :--- |
| **Dataset Source** | BIDSLab Apple Watch + Dreem 2 headband | BIDSLab Apple Watch + Dreem 2 headband | **YES** | Exact same cohort (47 physical subjects, 253 raw nights). |
| **Signal Usability Filter** | $\ge 60\%$ usable Dreem EEG signal fraction per night | $\ge 60\%$ usable Dreem EEG signal fraction per night | **YES** | Excludes exactly 7 defective nights; leaves 246 usable nights. |
| **Cohort Size** | 47 subjects, 246 usable nights | 47 subjects, 246 usable nights | **YES** | Verified exact match. |
| **Subject Split** | 31 Train / 5 Validation / 11 Test | 31 Train / 5 Validation / 11 Test | **YES** | Exact subject IDs in Test: Bidslab08, 10, 16, 17, 18, 19, 43, 45, 49, 62, 68. |
| **Test Partition Size** | 11 subjects, 58 usable nights | 11 subjects, 58 usable nights | **YES** | Both evaluate on the exact same 58 test nights. |
| **Label Mapping** | 0: Wake, 1: Light, 2: Deep, 3: REM, 5: Excluded | 0: Wake, 1: Light, 2: Deep, 3: REM, 5: Excluded | **YES** | 4-class classification standard (AASM-aligned). |
| **Sequence Length ($T$)** | **$T = 1200$ epochs** (10 continuous hours) in official testing notebook (`sqe1200=1200`) | **$T = 600$ epochs** (5 continuous hours, 0-padded / truncated) | **NO (DIFFERENCE)** | SLAMSS-IFS evaluates up to 10 hours of recording per night ($55,196$ test windows evaluated in repo notebook); SleepAdaptiveNet evaluated the first 5 hours ($34,241$ valid windows). |
| **Padding & Masking** | Padded with zeros to 1200; mask vector $z$ marks valid epochs ($z=1$ for stages 0..4, $z=0$ for artifact/unknown stage 5). Only valid epochs evaluated. | Padded with zeros to 600; mask vector marks valid epochs within sequence, invalid/unknown epochs marked $-100$. Only valid epochs evaluated. | **PARTIALLY (SAME PRINCIPLE, DIFFERENT $T$)** | Both correctly mask out invalid/padding epochs; but SLAMSS-IFS tests longer sequence horizons. |
| **Accuracy Aggregation** | **Pooled Window Accuracy** ($\sum \text{TP} / \sum N$ across all windows in test batches) | **Pooled Window Accuracy** ($\sum \text{TP} / \sum N$ across all test windows) | **YES** | Both use pooled window accuracy as primary metric. |
| **Reported Metric Scope** | Overall Accuracy, Sensitivity, Specificity, Precision, Weighted-F1, Weighted-MCC | Overall Accuracy, Macro-F1, Weighted-F1, Cohen's $\kappa$, MCC, ECE | **PARTIALLY** | SLAMSS-IFS reported **Weighted-F1** (0.7079); did not report unweighted Macro-F1 or Cohen's $\kappa$. |
| **Model Selection Protocol** | **Validation-Selected Checkpoint** over 500 epochs: "validation dataset was utilized to identify optimal model parameters" | **Frozen Fixed Budget**: Exactly 80 epochs, zero early stopping, single checkpoint evaluated per seed | **NO (DIFFERENCE)** | SLAMSS-IFS peak validation checkpoint selection on test; SleepAdaptiveNet frozen at epoch 80. |
| **Random Initializations** | **Single Seed** reported in Table V (plus separate 5-fold CV robustness test: $69.76\% \pm 1.31\%$) | **Multi-Seed Evaluation** ($N=3$ seeds: 42, 123, 2024, Mean $\pm$ SD) | **NO (DIFFERENCE)** | SleepAdaptiveNet provides multi-seed distribution and subject-level bootstrap CIs; SLAMSS-IFS Table V is single-run. |
| **Preprocessing & Inputs** | Instantaneous Heart Rate (IHR, mean/std normalized), 3-axis accelerometry, time features, frequency features | Instantaneous Heart Rate (IHR, mean/std normalized), 3-axis accelerometry, time features, frequency features | **YES** | Exact preprocessed feature cache matching repository. |
| **Architecture Scale** | **5,465,768 parameters** (heavy multi-layer CNNs, intra-epoch GRU, cross-modal attention) | **162,325 parameters** (lightweight 1D CNNs, BiGRU, adaptive fusion gate, GRL discriminator) | **NO (DELIBERATE DESIGN)** | SleepAdaptiveNet uses **33.7x fewer parameters** (97.03% compression). |

---

## 2. In-Depth Analysis of Differences

### A. Sequence Truncation ($T=600$ vs $T=1200$)
- In the official SLAMSS-IFS repository (`official_slamss_ifs/SLAMSS_IFS_Testing.ipynb`), the testing loop sets `sqe1200 = 1200` epochs ($1200 \times 30\text{ s} = 10\text{ hours}$). All available epochs in each recording up to 1200 are evaluated. In their notebook output, this yielded **55,196 evaluated test windows**.
- In the SleepAdaptiveNet evaluation pipeline, sequences were standardized to $T=600$ epochs ($600 \times 30\text{ s} = 5\text{ hours}$), evaluating the first 5 hours of each test night. This yielded **34,241 valid test windows**.
- *Impact*: Evaluating the entire 10-hour night encompasses late-morning periods and additional REM sleep cycles (which typically concentrate in the later hours of sleep). Truncating to 5 hours primarily captures the first half of the night (rich in N3 Deep sleep, with relatively fewer REM cycles).

### B. Validation Model Selection vs. Locked Budget
- The SLAMSS-IFS paper explicitly states: *"The model was trained for 500 optimization epochs using the Adam optimizer. The validation dataset was utilized to identify the optimal model parameters, which were then applied to generate the final results from the test set."*
- In contrast, EXP-012 applied a locked budget of **exactly 80 epochs** across all seeds, without early stopping or post-hoc validation checkpoint picking.
- *Impact*: In our EXP-011 convergence study, peak validation performance occurred between Epoch 72 and 92. Evaluating at a single fixed epoch (80) avoids validation-based test-set selection, but means individual seeds may be evaluated slightly before or after their exact validation peak.

### C. Primary Reported F1 Metric: Weighted vs. Macro
- Table V in the SLAMSS-IFS paper reports **Weighted F1 Score = 0.7079**, not unweighted Macro-F1.
- In imbalanced sleep classification (where Light sleep accounts for ~40% of epochs and Wake accounts for ~11%), Weighted-F1 is substantially higher than unweighted Macro-F1 because it weights the high-performing majority class (Light sleep) more heavily.
- For SleepAdaptiveNet (`B0_GRL`):
  - **Weighted-F1**: **$0.6835 \pm 0.0141$** (Seed 42: 0.6985, Seed 123: 0.6816, Seed 2024: 0.6704).
  - **Macro-F1**: **$0.6859 \pm 0.0139$** (Seed 42: 0.6986, Seed 123: 0.6880, Seed 2024: 0.6711).
- Comparing SleepAdaptiveNet Macro-F1 against SLAMSS-IFS Weighted-F1 is a metric mismatch. The proper comparison is **Weighted-F1 to Weighted-F1**: **0.6835 vs. 0.7079** ($\Delta = -0.0244$).

---

## 3. Aggregation Matching (`SLAMSS_IFS_MATCHED_AGGREGATION`)

To verify whether accuracy aggregation affects the comparison:
- **SLAMSS-IFS Official Testing Notebook**: Uses **Pooled Window Accuracy** ($\sum \text{TP} / \sum N$).
- **SleepAdaptiveNet Primary Metric**: Uses **Pooled Window Accuracy** ($68.44\% \pm 1.35\%$).
- **Alternative: Mean-of-Subject Accuracy**:
  - Seed 42: $69.72\%$ (vs $69.86\%$ pooled)
  - Seed 123: $68.18\%$ (vs $68.27\%$ pooled)
  - Seed 2024: $66.98\%$ (vs $67.18\%$ pooled)
  - Mean across seeds: **$68.29\% \pm 1.37\%$** (vs $68.44\% \pm 1.35\%$ pooled)
- *Conclusion*: The aggregation method produces an insignificant difference ($-0.15\text{ pp}$). Both works primarily utilize **Pooled Window Accuracy**.

---

## 4. Comparability Verdict

- **Classification**: **PARTIALLY COMPARABLE / UNPAIRED BENCHMARK**
- **Statement**:
  > `UNPAIRED COMPARISON — SLAMSS-IFS predictions unavailable.`
  The underlying physical cohort (47 subjects), signal usability threshold ($\ge 60\%$), subject split (31 train / 5 val / 11 test), and test nights (58 nights) are identical. However, the evaluation sequence duration differs ($T=600$ vs $T=1200$), model selection differs (fixed 80 epochs vs validation-selected over 500 epochs), and per-window predictions from SLAMSS-IFS are not saved, preventing a paired subject-level statistical test.
