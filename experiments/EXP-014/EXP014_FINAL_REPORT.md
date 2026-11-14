# EXP-014: 75-Run Subject-Level Cross-Validation Final Audit Report

**Date**: 2026-11-10 / 2026-11-11  
**Auditor**: ML Research Engineer / Experimental Auditor  
**Dataset**: BIDSLab Apple Watch + Dreem 2 (All 47 subjects, 246 usable nights, 145,283 valid evaluated 30-second windows)  
**Execution Matrix**: 5 Models $\times$ 5 Outer Folds $\times$ 3 Seeds = **75 Independent Training Runs**  
**Hardware**: NVIDIA GeForce RTX 3060 12GB (CUDA 13.0, PyTorch 2.14.1+cu130, Python 3.14.7)  
**Primary Status**: COMPLETED & BITWISE VERIFIED (Zero Run Failures, Zero Reruns, Zero Protocol Deviations)

---

## 1. Executive Summary & Core Scientific Findings

EXP-014 was executed as an independent **subject-level 5-fold cross-validation experiment** across all 47 subjects to investigate whether the mechanisms proposed in SleepAdaptiveNet (Adaptive Modality Fusion, Transition Modeling, and Domain-Adversarial Subject-Invariant GRL) provide robust generalizability across different held-out subject cohorts, or whether their apparent advantage in EXP-012/013 was split-dependent.

### Critical Empirical Takeaways:

1. **Outcome B: Global Equivalence Across Subject Folds ($E \approx A$)**:
   - Across all 75 training runs (15 evaluations per model covering all 47 subjects), the overall performance differences between the 5 models are minimal:
     - **Model A** (Conventional CNN-BiGRU, 163.9k params): Accuracy = **$66.63\% \pm 3.33\%$**, Macro-F1 = **$0.6658 \pm 0.0365$**, Cohen's $\kappa$ = **$0.5251 \pm 0.0505$**.
     - **Model B** (Minus GRL, 154.0k params): Accuracy = **$66.68\% \pm 3.31\%$**, Macro-F1 = **$0.6664 \pm 0.0345$**, Cohen's $\kappa$ = **$0.5244 \pm 0.0505$**.
     - **Model C** (Minus Transition, 162.3k params): Accuracy = **$66.07\% \pm 4.04\%$**, Macro-F1 = **$0.6596 \pm 0.0424$**, Cohen's $\kappa$ = **$0.5154 \pm 0.0600$**.
     - **Model D** (Minus Adaptive Gating, 153.9k params): Accuracy = **$66.68\% \pm 3.59\%$**, Macro-F1 = **$0.6684 \pm 0.0371$**, Cohen's $\kappa$ = **$0.5270 \pm 0.0514$**.
     - **Model E** (Full SleepAdaptiveNet, 162.3k params): Accuracy = **$66.06\% \pm 4.08\%$**, Macro-F1 = **$0.6596 \pm 0.0425$**, Cohen's $\kappa$ = **$0.5153 \pm 0.0601$**.

2. **Paired Subject-Level Cluster Bootstrap Analysis ($B=1000$, $N=47$ Subjects)**:
   - Resampling subjects identically between paired models across the entire cohort reveals that **all 95% confidence intervals span zero**:
     - **$E \text{ vs } A$**: $\Delta \text{Accuracy} = \mathbf{-0.62\%}$ [95% CI: $-1.90\%, +0.50\%$], $\Delta \text{Macro-F1} = \mathbf{-0.0069}$ [95% CI: $-0.0213, +0.0059$], $\Delta \kappa = \mathbf{-0.0106}$ [95% CI: $-0.0332, +0.0065$].
     - **$E \text{ vs } B$** (GRL): $\Delta \text{Accuracy} = \mathbf{-0.66\%}$ [95% CI: $-1.36\%, +0.16\%$], $\Delta \text{Macro-F1} = \mathbf{-0.0067}$ [95% CI: $-0.0141, +0.0032$].
     - **$E \text{ vs } C$** (Transition): $\Delta \text{Accuracy} = \mathbf{-0.01\%}$ [95% CI: $-0.15\%, +0.12\%$], $\Delta \text{Macro-F1} = \mathbf{-0.0001}$ [95% CI: $-0.0014, +0.0013$].
     - **$E \text{ vs } D$** (Adaptive Fusion): $\Delta \text{Accuracy} = \mathbf{-0.63\%}$ [95% CI: $-1.99\%, +1.02\%$], $\Delta \text{Macro-F1} = \mathbf{-0.0081}$ [95% CI: $-0.0196, +0.0051$].
   - **Conclusion**: The paired bootstrap evidence does **not** support a generalizable performance difference between the proposed SleepAdaptiveNet and the matched-capacity conventional baseline across the general cohort.

3. **Subject-Level Win/Loss Distribution**:
   - Out of 47 independent subjects:
     - Model E achieved higher accuracy than Model A in **23 subjects** ($48.9\%$).
     - Model A achieved higher accuracy than Model E in **23 subjects** ($48.9\%$).
     - Exactly 1 subject was tied ($2.2\%$).
   - This exact 50/50 subject-level split firmly categorizes the cross-validation outcome as **Outcome B (Equivalence / Parity)**. Small models of matched capacity (~160k parameters) achieve equivalent staging performance, and the additional specialized modules provide limited generalizable benefit across random subject partitions.

4. **Component Attribution Synthesis**:
   - **Transition Module ($E - C$)**: Yielded virtually zero marginal difference across all 5 folds ($\Delta \text{Acc} = -0.01\%$, 95% CI strictly centered on zero: $[-0.15\%, +0.12\%]$). In sequence models employing a bidirectional GRU over $T=600$ epochs, inter-epoch transitions are already effectively modeled by the recurrent hidden states.
   - **Adaptive Fusion ($E - D$) & GRL ($E - B$)**: Neither mechanism produced a statistically distinguishable advantage across the 47 subjects under standard 5-fold cross-validation. Fixed 0.5/0.5 fusion (Model D) performed on par with or marginally ahead of dynamic gating ($66.68\%$ vs $66.06\%$).

---

## 2. Experimental Execution Audit

- **Total Runs Planned**: 75
- **Total Runs Executed Successfully**: 75
- **Failed Runs**: 0
- **Reruns**: 0
- **Protocol Violations**: 0
- **Total Compute Duration**: 5,745 seconds (~95.7 minutes) on NVIDIA RTX 3060.
- **Outer Test Isolation Verification**: Outer test sets were evaluated strictly once per run at locked epoch 80. Zero test data was used for inner validation, early stopping, or checkpoint selection.

---

## 3. Overall Performance Summary Table (15 Evaluations per Model, Mean ± SD)

Metrics computed across 5 outer folds $\times$ 3 seeds (145,283 valid evaluated windows):

| Model ID | Model Description | Parameters | Accuracy (%) | Macro-F1 | Weighted-F1 | Cohen's $\kappa$ | MCC | ECE | Boundary Err (%) | Non-Bound Err (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** | Matched Conventional CNN-BiGRU | 163,940 | 66.63 ± 3.33 | 0.6658 ± 0.0365 | 0.6654 ± 0.0339 | 0.5251 ± 0.0505 | 0.5278 ± 0.0502 | 0.0549 ± 0.0322 | 58.42 ± 0.93 | 32.31 ± 3.39 |
| **Model B** | SleepAdaptiveNet minus GRL | 153,974 | 66.68 ± 3.31 | 0.6664 ± 0.0345 | 0.6666 ± 0.0334 | 0.5244 ± 0.0505 | 0.5271 ± 0.0499 | 0.0502 ± 0.0323 | 57.84 ± 1.38 | 32.28 ± 3.36 |
| **Model C** | SleepAdaptiveNet minus Transition | 162,325 | 66.07 ± 4.04 | 0.6596 ± 0.0424 | 0.6602 ± 0.0412 | 0.5154 ± 0.0600 | 0.5193 ± 0.0581 | 0.0536 ± 0.0328 | 58.19 ± 1.53 | 32.91 ± 4.10 |
| **Model D** | SleepAdaptiveNet minus Dynamic Gate | 153,876 | 66.68 ± 3.59 | 0.6684 ± 0.0371 | 0.6668 ± 0.0359 | 0.5270 ± 0.0514 | 0.5299 ± 0.0518 | 0.0510 ± 0.0261 | 58.46 ± 1.67 | 32.27 ± 3.63 |
| **Model E** | Full SleepAdaptiveNet | 162,325 | 66.06 ± 4.08 | 0.6596 ± 0.0425 | 0.6601 ± 0.0415 | 0.5153 ± 0.0601 | 0.5193 ± 0.0583 | 0.0565 ± 0.0353 | 58.25 ± 1.73 | 32.92 ± 4.13 |

---

## 4. Paired Subject-Level Cluster Bootstrap Analysis ($B=1000$)

Paired bootstrap deltas ($\Delta = \text{Model E} - \text{Comparator}$) across all 47 subjects using identical resampled subject indices:

| Comparison | Target Mechanism Evaluated | Mean $\Delta \text{Acc}$ | 95% Bootstrap CI (Acc) | Mean $\Delta \text{Macro-F1}$ | 95% Bootstrap CI (F1) | Mean $\Delta \kappa$ | 95% Bootstrap CI ($\kappa$) | Excludes Zero? |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E vs A** | Full Architecture vs. Baseline | -0.62% | [-1.90%, +0.50%] | -0.0069 | [-0.0213, +0.0059] | -0.0106 | [-0.0332, +0.0065] | **NO** |
| **E vs B** | Subject-Invariant GRL | -0.66% | [-1.36%, +0.16%] | -0.0067 | [-0.0141, +0.0032] | -0.0101 | [-0.0229, +0.0056] | **NO** |
| **E vs C** | Transition Modeling | -0.01% | [-0.15%, +0.12%] | -0.0001 | [-0.0014, +0.0013] | -0.0001 | [-0.0020, +0.0018] | **NO** |
| **E vs D** | Adaptive Reliability Gating | -0.63% | [-1.99%, +1.02%] | -0.0081 | [-0.0196, +0.0051] | -0.0119 | [-0.0337, +0.0146] | **NO** |

---

## 5. Fold-by-Fold Staging Performance

Detailed breakdown of outer test accuracy and Macro-F1 across the 5 outer folds (averaged across 3 seeds per fold):

| Outer Fold | Outer Test Size | Model A (Acc / F1) | Model B (Acc / F1) | Model C (Acc / F1) | Model D (Acc / F1) | Model E (Acc / F1) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 10 subs (32,098 wins) | 63.31% / 0.6263 | 63.96% / 0.6318 | 62.57% / 0.6157 | 63.45% / 0.6334 | 62.54% / 0.6161 |
| **Fold 2** | 10 subs (31,291 wins) | 65.67% / 0.6517 | 65.25% / 0.6497 | 64.37% / 0.6411 | 64.40% / 0.6410 | 64.26% / 0.6403 |
| **Fold 3** | 9 subs (26,599 wins) | 71.79% / 0.7203 | 71.81% / 0.7177 | 72.13% / 0.7220 | 72.31% / 0.7255 | 72.14% / 0.7220 |
| **Fold 4** | 9 subs (27,630 wins) | 63.88% / 0.6402 | 63.60% / 0.6408 | 62.49% / 0.6296 | 64.46% / 0.6499 | 62.49% / 0.6295 |
| **Fold 5** | 9 subs (27,665 wins) | 68.51% / 0.6903 | 68.79% / 0.6921 | 68.79% / 0.6896 | 68.75% / 0.6923 | 68.87% / 0.6899 |

*Note: In Fold 3 and Fold 5, all models achieve high accuracy (~68.5–72.3%), whereas in Folds 1, 2, and 4, performance drops to ~62.5–65.7%. This inter-fold variance ($\approx 8\text{--}9\text{ pp}$) is driven by subject heterogeneity across the cohort rather than model architecture.*

---

## 6. Per-Stage Metrics Breakdown (Cohort-Level Mean ± SD Across 3 Seeds)

Ground truth support across all 47 subjects: Wake = 17,219 ($11.85\%$), Light = 59,207 ($40.75\%$), Deep = 34,704 ($23.89\%$), REM = 34,153 ($23.51\%$), Total = 145,283 valid evaluated windows.

| Model ID | Stage | Precision (Mean ± SD) | Recall (Mean ± SD) | F1-Score (Mean ± SD) |
| :--- | :--- | :---: | :---: | :---: |
| **Model A** | Wake | 0.6925 ± 0.0040 | 0.6779 ± 0.0144 | 0.6851 ± 0.0093 |
| | Light | 0.6353 ± 0.0123 | 0.6883 ± 0.0202 | 0.6605 ± 0.0041 |
| | Deep | 0.7181 ± 0.0357 | 0.6832 ± 0.0524 | 0.6984 ± 0.0150 |
| | REM | 0.6522 ± 0.0232 | 0.5898 ± 0.0215 | 0.6191 ± 0.0124 |
| **Model B** | Wake | 0.7079 ± 0.0153 | 0.6503 ± 0.0085 | 0.6777 ± 0.0035 |
| | Light | 0.6328 ± 0.0084 | 0.6962 ± 0.0219 | 0.6628 ± 0.0070 |
| | Deep | 0.7252 ± 0.0097 | 0.6734 ± 0.0175 | 0.6982 ± 0.0061 |
| | REM | 0.6465 ± 0.0377 | 0.6057 ± 0.0399 | 0.6240 ± 0.0110 |
| **Model C** | Wake | 0.6898 ± 0.0460 | 0.6552 ± 0.0192 | 0.6710 ± 0.0139 |
| | Light | 0.6235 ± 0.0168 | 0.7022 ± 0.0155 | 0.6602 ± 0.0024 |
| | Deep | 0.7403 ± 0.0087 | 0.6306 ± 0.0343 | 0.6806 ± 0.0168 |
| | REM | 0.6373 ± 0.0222 | 0.6114 ± 0.0109 | 0.6241 ± 0.0162 |
| **Model D** | Wake | 0.7041 ± 0.0402 | 0.6620 ± 0.0214 | 0.6815 ± 0.0087 |
| | Light | 0.6391 ± 0.0102 | 0.6803 ± 0.0162 | 0.6589 ± 0.0053 |
| | Deep | 0.7340 ± 0.0208 | 0.6649 ± 0.0398 | 0.6968 ± 0.0123 |
| | REM | 0.6257 ± 0.0229 | 0.6368 ± 0.0420 | 0.6303 ± 0.0152 |
| **Model E** | Wake | 0.6889 ± 0.0487 | 0.6555 ± 0.0207 | 0.6707 ± 0.0143 |
| | Light | 0.6234 ± 0.0161 | 0.7015 ± 0.0162 | 0.6599 ± 0.0017 |
| | Deep | 0.7396 ± 0.0095 | 0.6298 ± 0.0328 | 0.6799 ± 0.0161 |
| | REM | 0.6382 ± 0.0236 | 0.6128 ± 0.0109 | 0.6252 ± 0.0167 |

---

## 7. Sleep Stage Transition Boundary Dynamics

Across all 47 subjects and 246 usable nights:
- **Boundary Windows**: **5,920 windows** ($4.07\%$)
- **Non-Boundary Windows**: **139,363 windows** ($95.93\%$)

| Model ID | Boundary Error Rate (%) | Non-Boundary Error Rate (%) | Boundary / Non-Boundary Ratio |
| :--- | :---: | :---: | :---: |
| **Model A** | 58.42 ± 0.93 | 32.31 ± 3.39 | **1.81x** |
| **Model B** | 57.84 ± 1.38 | 32.28 ± 3.36 | **1.79x** |
| **Model C** | 58.19 ± 1.53 | 32.91 ± 4.10 | **1.77x** |
| **Model D** | 58.46 ± 1.67 | 32.27 ± 3.63 | **1.81x** |
| **Model E** | 58.25 ± 1.73 | 32.92 ± 4.13 | **1.77x** |

*Finding*: Across all 5 models, the error rate at transition arrival epochs remains consistently around ~58%, compared to ~32% during non-transition epochs. None of the tested architectural variants resolved the boundary transition barrier under the 80-epoch cross-validation regime.

---

## 8. Saved Artifacts Manifest

All primary, machine-readable, and derived artifacts are preserved in the root and `EXP014_RESULTS/`:
- `EXP014_RAW_RESULTS.csv`: Full 75-run execution table with run-level metrics, training times, and loss components.
- `EXP014_SUMMARY.csv`: Model-level aggregation table (mean ± SD across 15 runs per model).
- `EXP014_PER_CLASS.csv`: Class-specific precision, recall, F1, and support for all 4 classes across all models and seeds.
- `EXP014_SUBJECT_METRICS.csv`: Granular per-subject metrics (accuracy, macro-F1, kappa) for all 47 subjects across all 75 runs.
- `EXP014_PAIRED_BOOTSTRAP.csv`: Subject-level cluster bootstrap table ($B=1000$) reporting deltas and 95% confidence intervals.
- `EXP014_CONFUSION_MATRICES/`: Raw and normalized confusion matrices for Models A, B, C, D, and E (`cm_raw_model_{M}.csv`, `cm_norm_model_{M}.csv`).
- `EXP014_TRAINING_HISTORY/`: Validation histories recorded at epoch 80.
- `EXP014_RESULTS/checkpoints/`: 75 trained PyTorch model checkpoints (`model_{M}_fold{F}_seed{S}.pt`).
- `EXP014_RESULTS/predictions/`: 75 serialized prediction pickle files containing window-level ground truth, predictions, and subject labels.
