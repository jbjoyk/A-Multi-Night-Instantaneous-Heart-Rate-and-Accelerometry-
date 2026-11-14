# Supervisor Executive Summary: EXP-012 Audit & Research State

**Project**: Wearable Sleep Staging using Apple Watch (Instantaneous Heart Rate + 3-Axis Accelerometry)  
**Evaluated Architecture**: `SleepAdaptiveNet` (`B0_GRL`, 162,325 parameters)  
**Dataset**: BIDSLab Apple Watch + Dreem 2 (Official 11-Subject Held-Out Test Set)  
**Audit Status**: **AUDIT COMPLETE — ALL METRICS VERIFIED & RESOLVED**  

---

## 1. What Was Originally Reported
The initial draft of the EXP-012 report reported:
- Generalization performance on the held-out test set across 3 seeds (Accuracy: $68.44\% \pm 1.35\%$, Macro-F1: $0.6859 \pm 0.0139$, Weighted-F1: $0.6835 \pm 0.0141$, Cohen's $\kappa$: $0.5527 \pm 0.0188$).
- Comparison against the 5.47M parameter SLAMSS-IFS benchmark (IEEE TBME 2026 Table V: 71.04% Acc, 0.7079 Weighted-F1).
- Stated 34,800 evaluated windows, 8,871 REM windows, and various text-level class/boundary breakdowns.

---

## 2. What Was Independently Recomputed
An independent audit script (`EXP012_AUDIT/audit_exp012.py`) was executed to recalculate every metric directly from the raw saved prediction artifacts (`exp012_test_results.pkl`) and raw dataset labels:
- Real vs. padded window audit using the actual test sequence masks.
- Overall accuracy, Macro-F1, Weighted-F1, Cohen's $\kappa$, MCC, and ECE across all 3 seeds.
- Per-class precision, recall, F1, and support for all 4 sleep stages.
- Exact raw and row-normalized confusion matrices.
- Real boundary vs. non-boundary window error rates and verification of the weighted error identity.
- Cluster-bootstrap 95% confidence intervals ($B=1000$ iterations resampled at the subject level).
- Per-subject staging breakdown across all 11 real BIDSLab test subjects.

---

## 3. Discrepancies Found in the Previous Report
The audit identified **11 primary numerical and textual discrepancies** in the old draft (documented in detail in [EXP012_AUDIT_REPORT.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/SUPERVISOR_REVIEW_EXP012/EXP012_AUDIT_REPORT.md)):
1. **Window Count (34,800 vs. 34,241)**: 34,800 represents the total padded matrix allocation ($58 \text{ nights} \times 600 \text{ slots}$). Within these, **559 slots** contain invalid/padding labels (`-100`) and were properly masked out. The true number of evaluated valid staging windows is **34,241**.
2. **REM Class Support (8,871 vs. 8,312)**: The old text table subtracted Wake, Light, and Deep from 34,800, erroneously absorbing the 559 padded slots into REM. Ground-truth REM support is **8,312**.
3. **Prose Class Metrics**: Per-class F1/recall numbers in the text prose of Section 2 were typed by hand or estimated rather than pulled from the saved arrays.
4. **Boundary Window Counts**: Window counts in Section 7 were fabricated (5,745 / 29,055) to sum to 34,800. True transition boundary windows count is **1,379** (error rate: **$59.08\% \pm 0.47\%$**) and stable windows count is **32,862** (error rate: **$30.41\% \pm 1.39\%$**).
5. **Raw Confusion Matrix**: The text matrix in Section 6 was altered to sum to 104,400 ($34,800 \times 3$). True pooled matrix sum is **102,723** ($34,241 \times 3$).
6. **Per-Subject Table**: Section 9 in the old report displayed fabricated subject IDs (`1053`, `1449`, etc.) with multiples of 600 windows. The real per-subject metrics from `Bidslab08` through `Bidslab68` are now correctly provided.
7. **"Pooled Aggregate" Bootstrap CI**: The old report averaged bootstrap lower and upper bounds, which is statistically invalid. Per-seed bootstrap CIs are verified and preserved.

---

## 4. Status of the Final Test Benchmark
- **THE FINAL TEST BENCHMARK REMAINS STRICTLY FROZEN.**
- The model weights (`exp012_seed{42,123,2024}_final.pt`), predictions, targets, and test split were **NOT altered**.
- Zero retraining, zero hyperparameter adjustment, and zero post-hoc test selection took place.
- EXP-013 was **NOT launched**.

---

## 5. What Is Currently Supported
- **Verified Primary Test Results** ($N=3$ seeds, Mean $\pm$ SD on 34,241 real windows):
  - **Accuracy**: **$68.44\% \pm 1.35\%$** [Seed 42: 69.86%, Seed 123: 68.27%, Seed 2024: 67.18%]
  - **Macro-F1**: **$0.6859 \pm 0.0139$** [Seed 42: 0.6986, Seed 123: 0.6880, Seed 2024: 0.6711]
  - **Weighted-F1**: **$0.6835 \pm 0.0141$** [Seed 42: 0.6985, Seed 123: 0.6816, Seed 2024: 0.6704]
  - **Cohen's $\kappa$**: **$0.5527 \pm 0.0188$** [Seed 42: 0.5726, Seed 123: 0.5504, Seed 2024: 0.5352]
  - **Matthews Corr. (MCC)**: **$0.5541 \pm 0.0187$**
  - **Calibration (ECE)**: **$0.0257 \pm 0.0050$**
- **Verified Wake Sensitivity**: High Wake recall of **$75.95\% \pm 1.20\%$** (essential for detecting nocturnal arousals).
- **Verified Parameter Efficiency**: **162,325 parameters** vs. SLAMSS-IFS **5,465,768 parameters** (**33.7x compression**, 97.03% smaller).
- **Scientific Conclusion**: **Case B (Competitive / Approximate Parity under 33.7x Parameter Compression)**. SleepAdaptiveNet trails SLAMSS-IFS by 2.6 pp in accuracy while operating in an ultra-compact regime.

---

## 6. What Is Proposed Next
A complete proposal for the next research phase is detailed in [NEXT_PHASE_PROPOSAL.md](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/SUPERVISOR_REVIEW_EXP012/NEXT_PHASE_PROPOSAL.md):
- **Model A**: Matched-size plain CNN-BiGRU baseline (~160k params) without adaptive gating, transition loss, or GRL.
- **Model B**: Ablation without GRL (isolates subject-invariance benefit).
- **Model C**: Ablation without transition loss (isolates temporal regularization benefit).
- **Model D**: Ablation with fixed equal fusion (isolates adaptive reliability gate benefit).
- **Model E**: Frozen `B0_GRL` (audited EXP-012 results reused without retraining).
- **Paired Subject-Level Bootstrap Protocol**: $B=1000$ paired resamples comparing $E - M$ with 95% CIs.

**STATUS**: The next phase is fully specified but **STRICTLY ON HOLD** pending your review and authorization.
