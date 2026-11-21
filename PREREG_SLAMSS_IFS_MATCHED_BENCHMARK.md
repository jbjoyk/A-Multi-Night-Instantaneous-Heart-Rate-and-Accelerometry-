# PRE-REGISTRATION: SLAMSS-IFS Matched 5-Fold Cross-Validation Paired Comparative Benchmark

**Document ID:** `PREREG_SLAMSS_IFS_MATCHED_BENCHMARK`  
**Experiment ID:** EXP-021 (Phase D, Stage 3 / Task TSK-08)  
**Status:** FINALIZED & EGM-VERIFIED (Pending Explicit User Execution Authorization under DEC-033 & DEC-034; zero models trained; locked baselines preserved)  
**Governing Authority:** [`RESEARCH_DIRECTION_v4_FULL.md`](./RESEARCH_DIRECTION_v4_FULL.md), Section 8 & Section 11; [`DECISIONS.md`](./DECISIONS.md); [`PHASE_D_EXPERIMENTAL_DECISION_PLAN.md`](./PHASE_D_EXPERIMENTAL_DECISION_PLAN.md)  
**Date:** 2026-11-21 (Pre-Execution EGM Amendment: Circadian Feature Parity & Training Budget Locked)  

---

## 1. Scientific Motivation & The Published Baseline Protocol

### 1.1 The Published Benchmark (Song et al., 2026)
The foundational baseline for sleep staging on the PhysioNet BIDSleep dataset was established by Song et al. (2026):
- **Citation:** Song TA, Zhang Y, Zhou Z, Hou L, Malekzadeh M, Behzad A, Dutta J. *AI-Driven Sleep Staging Using Instantaneous Heart Rate and Accelerometry: Insights From an Apple Watch Study*, IEEE Transactions on Biomedical Engineering (TBME), 73(4):1596–1608, 2026 (published online Sept 2025; DOI: `10.1109/TBME.2025.3612158`).
- **Published Metrics:** 
  - Overall Accuracy: **71.04%**
  - Weighted F1: **0.7079**
  - Class Recalls: Deep (N3) = 76.4%, REM = 70.4%, Wake accuracy $\approx 60\%$.
  - *Note:* Song et al. did **not** report unweighted Macro-F1 (the apparent "78.1%" figure occasionally cited in secondary informal summaries was an unverified erroneous attribution).
- **Published Evaluation Protocol:**
  - Evaluated on a **single fixed subject-disjoint holdout partition** (`seed = 0`): 31 train subjects, 5 validation subjects, and **11 test subjects** (Cohort Beta: `Bidslab08, Bidslab10, Bidslab16, Bidslab17, Bidslab18, Bidslab19, Bidslab43, Bidslab45, Bidslab49, Bidslab62, Bidslab68`, comprising 58 evaluation nights).
  - Song et al. enforced strict subject disjointness (no subject overlap between train and test). There was **no subject-night leakage** in their evaluation.
  - However, because their test partition consisted of a single 11-subject holdout without cross-validation or bootstrap confidence intervals, the published 71.04% accuracy carries significant single-split sampling variance.

### 1.2 Distinguishing the Published Architecture from the Repository Implementation
Forensic code audit (`SLAMSS_IFS_BASELINE_PROTOCOL_AUDIT.md`) revealed a major parameter discrepancy between the published paper text and the official author repository (`github.com/BIDSLabUMass/SLAMSS-IFS`):
1. **Published Paper Text Architecture:**
   - 5 Conv1d layers (128 channels), MaxPool (30 $\to$ 5 $\to$ 1) + 2-layer BiLSTM (hidden size 32) + 258-dim feature fusion + Seq2Seq LSTM (hidden size 96) with Bahdanau attention.
   - Total Trainable Parameters: **666,060 parameters** (~2.66 MB in float32).
   - Relative to this paper-text specification, Model A (`ConventionalCNNBiGRU`, 163,940 parameters) achieves a **4.06-fold parameter compression** ($666,060 / 163,940 = 4.06\times$).
2. **Official Author Repository Codebase (`SLAMSS_IFS_Model.py`):**
   - Incorporates a massive 4.70M-parameter multi-scale `cnn3` convolutional pyramid block alongside BiGRU and Seq2Seq modules.
   - Total Trainable Parameters: **5,465,768 parameters** (~21.86 MB in float32).
   - Relative to this official repository implementation, Model A achieves a **33.34-fold parameter compression** ($5,465,768 / 163,940 = 33.34\times$).

### 1.3 The Need for a Protocol-Matched Benchmark
Our project benchmark (Model A) achieved out-of-fold accuracy of **66.63%–67.12%** and unweighted Macro-F1 of **64.44%–64.58%** under rigorous 5-fold cross-validation across all 47 subjects. 

When evaluated on the exact same 11-subject test partition (Cohort Beta) in historical experiment EXP-012, our lightweight model achieved an empirical 3-seed mean accuracy of **$68.44\% \pm 1.35\%$** (Seed 42: 69.86%, Seed 123: 68.27%, Seed 2024: 67.18%; Weighted F1: $0.6835 \pm 0.0141$; Macro-F1: $0.6859 \pm 0.0139$), demonstrating that Cohort Beta is an easier-than-average partition.

Directly comparing our 47-subject cross-validation results to Song et al.'s single 11-subject holdout is scientifically invalid. To eliminate baseline ambiguity, this preregistration establishes an identical, protocol-matched 5-fold cross-validation re-evaluation of SLAMSS-IFS under the exact data splits, preprocessing, sequence context, and evaluation metrics used for Model A.

---

## 2. Evaluation Framework: Paired Comparative Estimation with Confidence Intervals

In the absence of a pre-existing clinical consensus or regulatory standard defining a formal non-inferiority margin for wearable sleep staging, imposing an arbitrary threshold (such as $-1.50\text{ pp}$) is methodologically unjustified. 

Instead, this benchmark adopts an **estimation and paired comparative framework** with bootstrap confidence intervals and effect sizes, adhering strictly to open-science reporting standards:
1. **Primary Focus on Estimation:** Rather than forcing a binary non-inferiority verdict against an artificial threshold, the experiment focuses on precisely estimating the magnitude, sign, and uncertainty of the participant-level performance difference between Model A and SLAMSS-IFS.
2. **Contextualizing the Parameter-Accuracy Trade-off:** The empirical performance difference is reported alongside the concrete hardware efficiency gains:
   - Model A requires 163,940 parameters (~164 KB in INT8 quantization), fitting into on-chip SRAM of wearable microcontrollers (e.g., Nordic nRF5340) for real-time edge inference.
   - SLAMSS-IFS requires 666,060 parameters (paper text; 4.06x larger) or 5,465,768 parameters (repository; 33.34x larger), requiring external DRAM.

---

## 3. Matched Experimental Protocol & Constants

SLAMSS-IFS will be trained and evaluated under the identical experimental harness used for Model A:

```
+-----------------------------------------------------------------------------------------+
|                                EXPERIMENTAL CONSTANTS                                   |
+-----------------------------------------------------------------------------------------+
| 1. Cohort: Exactly the 47 BIDSleep participants (246 total usable nights).             |
| 2. Split Manifests: Identical subject-independent 5-fold CV splits:                     |
|    - Fold 1: 10 subjects                                                               |
|    - Fold 2: 10 subjects                                                               |
|    - Fold 3: 9 subjects                                                                |
|    - Fold 4: 9 subjects                                                                |
|    - Fold 5: 9 subjects                                                                |
|    - Manifest Hash: review_package/fold_manifests.csv (SHA256: 36d84020...)            |
| 3. Sequence Context: Full-night context L = 1200 epochs (10 hours) with sliding-        |
|    window probability stitching for sequence length N > L.                              |
| 4. Input Features: Resampled 1 Hz instantaneous heart rate + accelerometry magnitude     |
|    (B, L*30, 2) plus matched 5-channel epoch metadata [freq, clock=0, time_feat=0,     |
|    hr_mean, hr_std] (neutralize_circadian=True) for strict feature parity with locked  |
|    EXP-018 Model A (DEC-024 proved unneutralized time_feat confounds architecture by   |
|    +1.36 pp). Two of cnn_ACT's 5 channels (clock, time_feat) are zeroed identically    |
|    to Model A's meta_proj(5->16).                                                      |
| 5. Target Mapping: 4-class classification:                                              |
|    Wake = 0, Light (N1+N2) = 1, Deep (N3) = 2, REM = 3; unscored/padded = -100.         |
| 6. Random Seeds: Exactly the 3 locked project seeds: 42, 123, 2024.                    |
| 7. Loss Function: DEC-011 composite loss L_total = L_CE(masked) + 0.01 * L_RCC(masked)  |
|    using official asymmetric fn/fp matrices, computed strictly in FP32 outside AMP.    |
| 8. Decoder Input Integrity: Zero-leakage trg_zeros with teacher_forcing_ratio = 0.0     |
|    at both train and inference time (remediating upstream t=0 target leakage).         |
+-----------------------------------------------------------------------------------------+
```

---

## 4. Formal Estimands & Statistical Inference Protocol

### 4.1 Participant-Level Paired Estimand
Because repeated nights within a participant are statistically correlated, statistical inference is conducted strictly at the independent participant level ($N = 47$):

1. For each participant $i \in \{1, \dots, 47\}$, compute out-of-fold Macro-F1 across all valid evaluation epochs for Model A and SLAMSS-IFS, averaged across the 3 seeds:
   $$\bar{F}_{i, \text{Model A}} = \frac{1}{3} \sum_{s \in \{42, 123, 2024\}} \text{Macro-F1}_{i, s}(\text{Model A})$$
   $$\bar{F}_{i, \text{SLAMSS-IFS}} = \frac{1}{3} \sum_{s \in \{42, 123, 2024\}} \text{Macro-F1}_{i, s}(\text{SLAMSS-IFS})$$
2. Compute the paired difference for each participant:
   $$d_i = \bar{F}_{i, \text{Model A}} - \bar{F}_{i, \text{SLAMSS-IFS}}$$
3. The primary estimand is the cohort mean paired difference:
   $$\bar{d} = \frac{1}{47} \sum_{i=1}^{47} d_i$$

### 4.2 Bootstrap & Non-Parametric Inference
- **10,000-Resample Participant-Level Percentile Bootstrap:** Resample participants with replacement 10,000 times. Construct the two-sided 95% bootstrap confidence interval $[\text{CI}_{\text{lower}}, \text{CI}_{\text{upper}}] = [\text{Percentile}_{2.5}(\bar{d}^*), \text{Percentile}_{97.5}(\bar{d}^*)]$.
- **Wilcoxon Signed-Rank Test:** Two-sided non-parametric test on the paired differences $\{d_1, \dots, d_{47}\}$ testing the null hypothesis $H_0: \text{Median}(d) = 0$.
- **Effect Size Metric:** Standardized paired effect size (Cohen's $d_z$):
  $$d_z = \frac{\bar{d}}{s_d}$$
  where $s_d$ is the sample standard deviation of the paired participant differences.
- **Secondary Metrics:** Paired differences in overall accuracy, Cohen's kappa, per-stage F1 scores (Wake, Light, Deep, REM), and weighted F1 are computed and reported with corresponding 95% bootstrap confidence intervals.

---

## 5. Decision & Interpretation Framework

Rather than imposing arbitrary regulatory pass/fail boundaries, the experimental outcome will be interpreted strictly according to empirical confidence intervals and effect sizes:

```
+-----------------------------------------------------------------------------------------+
|                               INTERPRETATION FRAMEWORK                                  |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  1. MODEL A ADVANTAGE (STATISTICALLY SIGNIFICANT):                                      |
|     - Criterion: Bootstrap 95% CI lower bound > 0.0 pp AND Wilcoxon p < 0.05.           |
|     - Scientific Meaning: Model A statistically outperforms SLAMSS-IFS despite operating |
|       at 4.06x to 33.34x parameter compression.                                         |
|                                                                                         |
|  2. SLAMSS-IFS ADVANTAGE (STATISTICALLY SIGNIFICANT):                                   |
|     - Criterion: Bootstrap 95% CI upper bound < 0.0 pp AND Wilcoxon p < 0.05.           |
|     - Scientific Meaning: SLAMSS-IFS statistically outperforms Model A on matched        |
|       splits. The performance gap is precisely quantified by [\bar{d}, 95% CI],        |
|       allowing an objective assessment of whether the accuracy gain justifies its       |
|       higher parameter and memory requirements.                                         |
|                                                                                         |
|  3. PERFORMANCE INDISTINGUISHABLE (STATISTICAL PARITY):                                 |
|     - Criterion: Bootstrap 95% CI spans 0.0 pp (Wilcoxon p >= 0.05).                     |
|     - Scientific Meaning: Model A and SLAMSS-IFS exhibit no statistically significant   |
|       performance difference under matched 5-fold cross-validation. The confidence      |
|       interval bounds the maximum plausible performance disparity, establishing that    |
|       lightweight recurrent modeling achieves comparable accuracy to complex attention. |
+-----------------------------------------------------------------------------------------+
```

---

## 6. Computational Resource Allocation

- **Model Execution:** 15 training runs (5 outer folds $\times$ 3 seeds).
- **Execution Precision:** SLAMSS-IFS architecture executed with PyTorch AMP (`torch.cuda.amp.autocast`) and GradientScaler.
- **Hardware Architecture & Empirical Memory Profiling:**
  - Workstation: 2 $\times$ NVIDIA GeForce RTX 3060 (12 GB VRAM each; independent PCIe devices, no pooled memory assumed), AMD Ryzen 7 7700 (8C/16T), 32 GB RAM.
  - Architecture Profiled: Official BIDSLabUMass/SLAMSS-IFS repository model (DEC-011): 5,166,272-param CNN + 299,496-param Seq2Seq = **5,465,768 parameters**.
  - Empirically measured metrics on RTX 3060 ($L = 1200$, real BIDSleep tensors, PyTorch AMP):
    * Batch size $B = 1$: **1,416.32 MiB (1.38 GiB) allocated / 1,674.00 MiB (1.64 GiB) reserved VRAM**; throughput: 0.31 nights/s.
    * Batch size $B = 2$: **2,510.50 MiB (2.45 GiB) allocated / 3,020.00 MiB (2.95 GiB) reserved VRAM**; throughput: 0.60 nights/s.
    * Batch size $B = 4$: **4,737.44 MiB (4.63 GiB) allocated / 5,242.00 MiB (5.12 GiB) reserved VRAM**; throughput: 1.23 nights/s (leaves $> 6.8\text{ GiB}$ free VRAM on GPU 1 and $> 6.0\text{ GiB}$ on GPU 0).
    * Model Parameters: 20.85 MiB; Adam Optimizer State: **41.70 MiB** (1st and 2nd momentum states).
    * Numerical Stability: Verified fully stable across all batch sizes (zero NaNs/Infs; mean loss: ~1.21–1.23).
  - **Verdict:** The official 5.47M SLAMSS-IFS model fits comfortably on a **single 12 GB RTX 3060 GPU** even at $B = 4$; the previous assumption requiring $\ge 24\text{ GB}$ VRAM (A100 or RTX 3090/4090) was an unbatched theoretical overestimate.
- **Training Budget & Optimizer-Step Parity:**
  - Micro-batch size $B = 4$, effective batch size $B_{\text{eff}} = 4$ (`accum_steps = 1`, zero tail-accumulation scaling bias), yielding **42–44 optimizer steps/epoch** across 168–175 training nights/fold.
  - Budget: `max_epochs = 100` (up to **4,200–4,400 optimizer steps**, strictly matching/exceeding Model A's locked 3,200 steps and `B0_GRL`'s ~80-epoch convergence window under DEC-017), Adam (`lr = 1e-4`), gradient clipping `max_norm = 1.0`.
  - Early stopping on inner-validation Macro-F1 (zero outer-test access) with `min_epochs = 40` (~1,700-step floor) and `patience = 25` epochs. Note: Model A EXP-018 used fixed 3,200 steps without validation checkpointing; granting inner-val early stopping to SLAMSS-IFS is transparently disclosed as conservative in SLAMSS-IFS's favor.
  - Validation & outer-test inference use batched sliding-window probability stitching (`val_batch_size = 16` under `torch.no_grad()`, ~2 s/epoch vs. ~30 s/epoch unbatched) with identical mathematical outputs.
- **Runtime & Parallel Allocation:**
  - Empirical step timing at $B=4$: ~145 s/epoch training + ~2.5 s/epoch batched validation $\approx$ **~148 s/epoch (~2.5 min/epoch)**.
  - Expected runtime per fold-seed run (early stopping at ~65–85 epochs): **~2.7 to 3.5 hours** on RTX 3060 with AMP and CPU thread pinning (`OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`).
  - Dual-GPU parallel distribution:
    * GPU 0 (`cuda:0`): 7 runs (Folds 1–2 all seeds + Fold 3 seed 42).
    * GPU 1 (`cuda:1`): 8 runs (Fold 3 seeds 123, 2024 + Folds 4–5 all seeds).
  - Expected wall-clock duration: **~21 to 28 hours** total execution time (up to ~33 hours if all 8 GPU-1 runs hit `max_epochs = 100`).
- **Output Artifacts:**
  - Per-run telemetry JSONs written atomically to `experiments/EXP-021/run_results/slamss_ifs_fold{fold}_seed{seed}.json`.
  - Aggregated out-of-fold metrics written to `experiments/EXP-021/slamss_ifs_matched_subject_metrics.csv`.
  - Paired comparative analysis serialized to `experiments/EXP-021/slamss_ifs_paired_subject_deltas.csv`, `experiments/EXP-021/slamss_ifs_paired_comparison_summary.json`, and `experiments/EXP-021/slamss_ifs_paired_comparative_report.md`.
  - Locked EXP-018 and EXP-019 Model A baselines remain strictly untouched.
