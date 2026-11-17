# Technical and Reproducibility Review: Phase A3 Diagnostic Extraction Specification

**Target Document:** `experiments/EXP-019/design_phase_a3_extraction.md`  
**Target Commit:** `60d28590991b45156f6cd7d73bb959bfd4b8968c`  
**Review Authority:** `experiments/EXP-019/A3_EXTRACTION_GOVERNANCE_REVIEW_REPORT.md`  
**Review Skill:** `egm-review`  
**Date:** 2026-11-17  
**Review Verdict:** **APPROVED WITH REQUIRED CHANGES**  

---

## Executive Summary

This review evaluated the technical feasibility, statistical rigor, numerical determinism, and software interfaces specified in `experiments/EXP-019/design_phase_a3_extraction.md` for reconstructing EXP-019 Phase A3 diagnostics.

The review establishes:
1. **Zero-Shot (`ZS`) is 100% Bitwise Reproducible:** `ZS` relies on saved checkpoints (`modelA_L1200_fold{fold}_seed{seed}.pt`) and runs in `model.eval()` mode with zero stochasticity. Empirical verification confirms exact 6-decimal Macro-F1 equivalence ($0.000000$ difference) across outer test subjects.
2. **Adapted Arms (`FT-2`, `FT-other-2`, `PS-2`) Cannot Guarantee Bitwise Prediction Match:** In the historical `run_exp019_stage_c.py`, fine-tuning executed in `model.train()` with `dropout=0.2` without setting an explicit PyTorch random seed inside `run_stage_c()`. Adapted weights and epoch predictions were discarded in-memory. Consequently, minor stochastic variation ($\pm 0.0002$ to $\pm 0.012$ subject-level Macro-F1) occurs when re-executing adaptation. The proposed integrity gate requiring $\le 0.0001$ agreement for fine-tuned models is technically unachievable and must be reformed.
3. **Bland–Altman Analysis Has Unaddressed Clustered Data Structure:** The 152 evaluation nights are repeated measures nested within 46 subjects ($1$ to $5$ nights/subject). Naive pooled Bland–Altman statistics violate observation independence and underestimate agreement interval variance.
4. **Hypnogram Evaluator Artifact Mismatch:** `ClinicalHypnogramEvaluator.compute_night_metrics()` calculates Time in Bed as $\text{TIB} = N \times 0.5\text{ min}$. Dataset records contain significant fractions of unscored/artifact epochs (e.g., 171 unscored epochs out of 991 in `Bidslab36__2.npz`). Passing unmasked model predictions causes artificial inflation of predicted Total Sleep Time (TST) and Sleep Efficiency (SE).

---

## 1. Reconstruction Feasibility Assessment

### 1.1 Detailed Arm-by-Arm Analysis

| Arm | Reconstruction Mechanism | Reproducibility Type | Historical Missing Artifacts | Feasibility Status |
| :--- | :--- | :--- | :--- | :--- |
| **ZS** | **Inference-Only Pass** on locked backbone checkpoints | **Exact Bitwise Equivalence** | Per-epoch predictions and night hypnograms (recoverable to $0.000000$) | **FEASIBLE (Bitwise)** |
| **FT-2** | **Re-running Adaptation Procedure** ($5$ or $15$ epochs, locked Stage D hyperparameters) | **Protocol-Equivalent / Stochastic Alignment** | Adapted checkpoints, dropout RNG states, epoch predictions, hypnograms | **FEASIBLE (Distributional/Protocol)** |
| **FT-other-2** | **Re-running Adaptation Procedure** with deterministic donor selection | **Protocol-Equivalent / Stochastic Alignment** | Adapted checkpoints, dropout RNG states, epoch predictions, hypnograms | **FEASIBLE (Distributional/Protocol)** |
| **PS-2** | **Re-running Training from Scratch** ($300$ or $500$ steps) | **Protocol-Equivalent / Stochastic Alignment** | Trained checkpoints, dropout RNG states, epoch predictions, hypnograms | **FEASIBLE (Distributional/Protocol)** |

### 1.2 Root Cause of Adaptation Stochasticity
Empirical testing on Outer Fold 1 revealed:
- `ZS` subject Macro-F1 values matched `exp019_subject_metrics.csv` to 6 decimal places ($0.000000$ difference across all 10 test subjects).
- `FT-2` subject Macro-F1 values matched with small residuals:
  - `Bidslab08`: $0.6351$ vs $0.6343$ ($\Delta = +0.0008$)
  - `Bidslab41`: $0.6657$ vs $0.6655$ ($\Delta = +0.0002$)
  - `Bidslab65`: $0.4192$ vs $0.4071$ ($\Delta = +0.0121$)
- **Mechanism:** In `run_exp019_stage_c.py`, the CLI argument `--seeds 42 123 2024` was used solely as a filename index to select backbone checkpoint files. The script never called `torch.manual_seed(seed)` or `torch.cuda.manual_seed_all(seed)` inside `run_stage_c()`. Because `ConventionalCNNBiGRU` incorporates dropout ($p=0.2$) in `hr_encoder`, `acc_encoder`, `temporal_gru`, and `classifier_head`, the forward/backward passes during `adapt_supervised_ft` and `train_per_subject_scratch` drew from the process's unseeded CUDA RNG stream.

---

## 2. Integrity Gate Reformulation

### 2.1 Critical Defect in Proposed Gate (Section 15.1)
Section 15.1 specifies:
> "The recomputed Macro-F1 values for all 46 subjects across arms ZS, FT-2, FT-other-2, and PS-2 must match `exp019_subject_metrics.csv` within an absolute tolerance of $\le 0.0001$."
Section 15.4 mandates:
> "If the reconstructed outputs deviate from locked metrics by $> 0.0001$, the script must immediately halt, output an error log, and exit with non-zero code without writing outputs."

**Finding (CRITICAL):** Because the historical execution did not record the CUDA RNG state or save intermediate checkpoints, re-running adaptation on any hardware will produce small variations ($\Delta > 0.0001$) on adapted arms. Enforcing a hard threshold of $\le 0.0001$ across all arms will guarantee immediate script abortion.

### 2.2 Scientifically Defensible Tiered Integrity Gate
The integrity gate must be reformed into a rigorous, two-tiered standard:

1. **Tier 1 (Deterministic Arms — Bitwise Invariant):**
   - For `ZS`, recomputed Macro-F1 must match `exp019_subject_metrics.csv` to $\le 0.0001$ for all 46 subjects.
   - For `ZS`, epoch-level predictions and hypnograms are certified bitwise identical to the historical run.
2. **Tier 2 (Stochastic Adaptation Arms — Protocol & Distributional Invariant):**
   - Fine-tuning must use the exact locked Stage D configuration per fold and exact deterministic data batches from `prepare_calibration_batch()`.
   - Reconstructed cohort-level mean Macro-F1 for `FT-2` must match historical mean ($0.6559$) within $\pm 0.0050\text{ pp}$.
   - Reconstructed primary paired delta ($\bar{\Delta}$) must match historical delta ($+1.01\text{ pp}$) within $\pm 0.0050\text{ pp}$ and preserve the qualitative conclusion ($H_A$ Inconclusive).
   - Reconstructed per-subject correlation with locked metrics must exceed $r \ge 0.98$.
   - The diagnostic report must explicitly disclose that fine-tuned predictions represent a protocol-faithful re-execution rather than bitwise recovered historical arrays.

---

## 3. Statistical Methodology Review

### 3.1 Clustered Structure in Bland–Altman Agreement (MAJOR)
Section 9.3 proposes computing Bland–Altman limits of agreement across all $N=152$ evaluation nights pooled together:
$$\bar{d} = \frac{1}{N} \sum d_i, \quad s_d = \sqrt{\frac{1}{N-1} \sum (d_i - \bar{d})^2}, \quad \text{LoA} = [\bar{d} \pm 1.96 s_d]$$

**Defect:** The 152 evaluation nights are repeated measures nested within 46 subjects. Subjects have between 1 and 5 evaluation nights. Pooled calculation treats repeated observations as independent, which:
1. Gives disproportionate weight to subjects with 5 evaluation nights (e.g., 7-night subjects) compared to subjects with 1 evaluation night (3-night subjects);
2. Conflates between-subject bias with within-subject night-to-night variation;
3. Produces confidence intervals and limits of agreement that are statistically invalid under standard polysomnographic validation guidelines (Bland & Altman, 2007).

**Required Correction:**
- Primary reporting must compute **subject-averaged Bland–Altman statistics** ($n=46$ independent clusters): average metric differences within subject across their evaluation nights, then compute mean bias and limits of agreement over the 46 subjects.
- Night-level pooled statistics ($N=152$) may be reported as a secondary, descriptive sensitivity check, explicitly labeled as *unadjusted for clustering*.

### 3.2 Unit of Analysis for Hypothesis Tests (MINOR)
Section 11 correctly maintains subjects as the unit of analysis ($n=46$) for the 10,000-resample cluster bootstrap and Wilcoxon signed-rank tests.
Section 12 correctly controls False Discovery Rate across the 4 sleep stages and 6 clinical sleep metrics using Benjamini–Hochberg ($q=0.05$).

---

## 4. Technical Consistency Review

### 4.1 Masking of Unscored and Artifact Epochs in Hypnograms (CRITICAL)
In `cache/prep_57738d534b`, expert labels contain unscored and movement artifact epochs marked as invalid (`target < 0` or `target >= 4`). For example, in `Bidslab36__2.npz`, 171 of 991 epochs are invalid.

In [`ClinicalHypnogramEvaluator`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/evaluator.py):
```python
N = len(hyp)
tib = float(N * cls.EPOCH_DURATION_MIN)
is_sleep = (hyp == 1) | (hyp == 2) | (hyp == 3)
tst = float(np.sum(is_sleep) * cls.EPOCH_DURATION_MIN)
se = float((tst / tib * 100.0) if tib > 0 else 0.0)
```

**Defect:** `SlidingWindowPredictor.predict_night()` emits continuous predictions in $\{0, 1, 2, 3\}$ for all $N$ epochs.
- If $y_{\text{true}}$ contains $-100$ on artifact epochs, `is_sleep` is False for those epochs in the reference hypnogram.
- In the predicted hypnogram, the model classifies those artifact epochs as Wake, Light, Deep, or REM.
- If passed unmasked to `ClinicalHypnogramEvaluator`, the predicted hypnogram will count artifact epochs as sleep, creating a severe artificial positive bias in TST ($+10$ to $+60$ minutes) and artificially depressing Sleep Efficiency relative to reference.

**Required Correction:**
Before passing hypnograms to `ClinicalHypnogramEvaluator`:
- Mask invalid reference epochs symmetrically in predicted hypnograms:
  ```python
  valid_mask = (y_true >= 0) & (y_true < 4)
  y_pred_masked = np.where(valid_mask, y_pred, -100)
  ```
- Or, update the evaluator call to compute metrics strictly on the subset of valid scored epochs.

### 4.2 Resource and Execution Feasibility
- Reconstructing `ZS` requires only 15 backbone forward passes across 152 nights ($\approx 45\text{ seconds}$ on GPU).
- Re-running adaptation for $k=2$ across 46 subjects $\times 3$ seeds requires $\approx 3\text{ minutes}$ on CUDA GPU (`cuda:0`).
- Total extraction runtime is estimated at $< 5\text{ minutes}$, confirming hardware feasibility.

---

## 5. Governance and Scope Verification

The specification satisfies all core governance constraints established by `A3_EXTRACTION_GOVERNANCE_REVIEW_REPORT.md`:
1. **$k=2$ Exclusivity:** All adaptation execution is strictly confined to $k=2$.
2. **$k \in \{1, 3, 4\}$ Status:** Formally designated `UNEXECUTED_IN_EXP019 / UNRECOVERABLE FROM LOCKED ARTIFACTS — REQUIRES NEW PROTOCOL`.
3. **`UA-2` Exclusion:** Correctly designated as an invalidated implementation no-op.
4. **Locked Artifact Immutability:** No locked files are modified.
5. **Anti-Premature Training Gate:** Phase D remains blocked.

---

## 6. Summary of Findings by Severity

### Critical Issues
1. **Unattainable $\le 0.0001$ Gate for Adapted Arms (Section 15.1, 15.4):** Fine-tuning stochasticity due to unseeded dropout RNG states in historical code prevents exact bitwise reproduction of `FT-2`, `FT-other-2`, and `PS-2`. The gate will cause immediate abortion.
2. **Hypnogram Evaluator Artifact Mismatch (Section 9.1, 9.2):** Failure to symmetrically mask unscored/artifact epochs before computing clinical sleep summaries will bias TST and SE calculations.

### Major Issues
3. **Clustered Bland–Altman Structure (Section 9.3):** Pooled night-level Bland–Altman analysis violates independence across repeated measures within subjects. Must add subject-averaged Bland–Altman statistics ($n=46$) as primary.

### Minor Issues
4. **Disclosure of Reconstruction Claim (Section 1, Section 14.6):** Must distinguish bitwise recovered `ZS` diagnostics from protocol-re-executed `FT-2` diagnostics in report metadata.

---

## 7. Required Document Corrections for `design_phase_a3_extraction.md`

1. **Section 9.1 & 9.2 (Hypnogram Preprocessing):** Add explicit symmetric artifact masking rule: `y_pred_masked = np.where((y_true >= 0) & (y_true < 4), y_pred, -100)` before invoking `ClinicalHypnogramEvaluator`.
2. **Section 9.3 (Clustered Bland–Altman):** Restructure to primary subject-averaged Bland–Altman ($n=46$) with secondary night-level pooled Bland–Altman ($N=152$) labeled as unadjusted.
3. **Section 15.1 & 15.4 (Integrity Gate):**
   - Tier 1: `ZS` subject Macro-F1 must match locked metrics to $\le 0.0001$.
   - Tier 2: `FT-2` cohort mean must match locked metric ($0.6559$) within $\pm 0.0050$, with primary paired delta matching $+1.01\text{ pp} \pm 0.0050\text{ pp}$ and subject correlation $r \ge 0.98$.
4. **Section 14.6 (Metadata):** Require explicit labeling in generated reports distinguishing bitwise reconstructed outputs (`ZS`) from protocol-faithful stochastic re-executions (`FT-2`, `FT-other-2`, `PS-2`).

---

## 8. Final Decision and Next Action

**Final Verdict:** **APPROVED WITH REQUIRED CHANGES**

**Top-Level EGM Skill:** `egm-resolve`  
**Next Safe Action:** Authorize `egm-resolve` to incorporate these four technical corrections into `experiments/EXP-019/design_phase_a3_extraction.md`. Once the specification is updated, proceed directly to script implementation.
