# PRE-REGISTRATION AMENDMENT: EXP-019 Secondary Hypothesis S3 (UA-2 Unsupervised Adaptation Re-Run)

**Document ID:** `PREREG_EXP-019_S3_AMENDMENT`  
**Parent Preregistration:** [`PREREG_EXP-019_H-A.md`](./PREREG_EXP-019_H-A.md) (Locked at commit `4dff5b7`)  
**Experiment ID:** EXP-019 (Phase D, Experiment 2 / Task TSK-07)  
**Status:** DRAFT FOR USER REVIEW (No models trained; zero GPU execution; locked baselines preserved)  
**Governing Authority:** [`RESEARCH_DIRECTION_v4_FULL.md`](./RESEARCH_DIRECTION_v4_FULL.md), line 131; [`DECISIONS.md`](./DECISIONS.md), DEC-030; [`PHASE_D_EXPERIMENTAL_DECISION_PLAN.md`](./PHASE_D_EXPERIMENTAL_DECISION_PLAN.md)  
**Date:** 2026-11-19 (Revised)  

---

## 1. Background & Forensic Preservation of the Historical No-Op

In `PREREG_EXP-019_H-A.md`, secondary hypothesis **S3** was pre-registered to test unsupervised test-time adaptation ($UA\text{-}2$):
> "Label-free adaptation UA-2 (InfoMax entropy minimization with marginal diversity penalty on the 2 calibration nights' unlabeled inputs) outperforms ZS."

### 1.1 Forensic Diagnosis & Preservation of Historical Records
During the independent reproducibility audit of Stage C execution artifacts (`experiments/EXP-019/exp019_subject_metrics.csv`), it was discovered that $UA\text{-}2$ produced performance metrics bitwise identical to population zero-shot ($ZS$) across all 46 participants and all 3 seeds:
$$\text{Macro-F1}_i(UA\text{-}2) \equiv \text{Macro-F1}_i(ZS) \quad \forall i \in \{1, \dots, 46\}, \forall s \in \{42, 123, 2024\}$$

Inspection of `exp019_adaptation_core.py` (lines 34–59, 121) uncovered the exact root cause:
1. The quality screening function `compute_wearable_quality_mask` applied physical sanity bounds designed for raw sensor data:
   $$30 \le \text{IHR} \le 180\text{ beats per minute (BPM)}$$
2. However, the data passed into `compute_wearable_quality_mask` from the preprocessed cache (`prep_57738d534b`) had **already undergone robust z-score standardization**:
   $$z = \frac{x - \text{median}(x)}{\text{IQR}(x) + 10^{-6}}$$
   Consequently, the instantaneous heart rate features exhibited empirical values strictly within $[-3.95, +9.94]$.
3. Because every normalized epoch fell below $30$, the boolean quality mask evaluated to `False` for 100% of calibration epochs across all subjects and nights (0 out of 82,762 epochs passed).
4. The unsupervised loss was computed over an empty tensor, zero gradient steps were executed, and the model remained bitwise identical to the unadapted zero-shot backbone.

**Governance Invariant (DEC-030):** The historical result ($\bar{\Delta} = 0.00\text{ pp}$ identically) remains permanently locked in `exp019_subject_metrics.csv` as a documented implementation no-op. It must **not** be interpreted as evidence that unsupervised adaptation fails (nor that it succeeds). This preregistration amendment governs a separate, clean re-test serialized to an independent artifact (`exp019_ua2_rerun_subject_metrics.csv`).

---

## 2. Scientific Motivation & Core Hypothesis

### 2.1 Scientific Question
In real-world wearable deployment, collecting ground-truth EEG sleep labels for supervised calibration is costly, intrusive, and unfeasible for consumer wearables. Unsupervised test-time adaptation offers a potential alternative: adjusting the model's feature representations or classifier margins using only the participant's unlabeled sensor streams (cardiac and accelerometry vitals).

However, low-rate autonomic vitals have high noise and subtle stage separation. Does unsupervised entropy minimization successfully adapt a sequence model to an individual's autonomic baseline, or does unsupervised optimization without labels lead to representation collapse?

### 2.2 Formal Hypothesis S3 (Revised)
Fine-tuning the population zero-shot model ($ZS$) on a participant's first 2 unlabeled calibration nights using Information Maximization (Shannon entropy minimization with marginal class diversity penalty) improves Macro-F1 on subsequent evaluation nights relative to $ZS$:
- **Statistical Superiority Hypothesis:**
  $$H_0: \bar{\Delta}_{\text{UA-2}} \le 0.0\text{ pp} \quad \text{vs.} \quad H_1: \bar{\Delta}_{\text{UA-2}} > 0.0\text{ pp}$$
- **Clinical / Practical Relevance Benchmark:**
  $$\bar{\Delta}_{\text{UA-2}} \ge +1.00\text{ pp Macro-F1}$$

---

## 3. Sensor Normalization, Physical Units, and Quality Masking

To rectify the implementation fault without repeating the unit-mismatch error on other channels:

### 3.1 Physical Units vs. Preprocessed Cache Realities
1. **Instantaneous Heart Rate (Channel 0):**
   - *Raw Sensor Unit:* Beats per minute (BPM) derived from Apple Watch photoplethysmography (PPG). Normal nocturnal range is $40–90\text{ BPM}$ (bradycardia in deep sleep $\approx 35–40\text{ BPM}$; transient arousals $\approx 120–140\text{ BPM}$).
   - *Cache Representation:* Robust-standardized z-score ($z_{\text{IHR}}$ with mean $\approx 0$, std $\approx 1$, empirical range $[-3.95, +9.94]$).
   - *Validation Rule:* To evaluate physical bounds ($30 \le \text{BPM} \le 180$), raw BPM must be reconstructed via inverse transform using stored night metadata ($\text{IHR}_{\text{raw}} = z \cdot \text{IQR} + \text{median}$) or read from raw `hr.csv` prior to standardization.
2. **Accelerometry Magnitude (Channel 6):**
   - *Raw Sensor Unit:* Triaxial acceleration $(a_x, a_y, a_z)$ in units of Earth's gravitational acceleration $g$ ($1\text{ g} \approx 9.81\text{ m/s}^2$). When the wrist is stationary, Earth's gravity yields a raw vector magnitude $m_{\text{raw}} = \sqrt{a_x^2 + a_y^2 + a_z^2} \approx 1.0\text{ g}$. Normal sleep movements produce up to $2.0–2.5\text{ g}$.
   - *Cache Representation:* Channel 6 (`acc_mag`) in `prep_57738d534b` has **also undergone robust standardization** (median-centered and scaled, with values bounded within $[-10.0, +10.0]$).
   - *Critical Distinction:* Applying physical gravity thresholds ($0.2\text{ g} \le m \le 3.5\text{ g}$) to cached `X[:, 6, :]` would repeat the exact unit-mismatch bug!
   - *Corrected Accelerometer Validation:*
     - On cached data: Accelerometer validity is established via non-zero intra-epoch variance:
       $$\text{Var}_{t}(m) > 0.0$$
       Because variance is scale-invariant, this correctly screens out sensor freeze, disconnected hardware, or zero-padding.
     - On raw data (if available at ingestion): Physical bounds $0.2\text{ g} \le m_{\text{raw}} \le 3.5\text{ g}$ are evaluated strictly on raw $g$ before standardization.

### 3.2 Mandatory Pre-Flight Quality Gate
Before any optimization loop or GPU gradient step is executed, the implementation must pass the following automated assertions across all outer test calibration nights:

```python
# Mandatory Pre-Flight Quality Gate Assertions
assert valid_epoch_ratio > 0.95, (
    f"Quality filter rejected too many epochs ({valid_epoch_ratio:.2%}). "
    "Check whether z-scores are being passed to raw thresholds!"
)
assert non_physiological_leakage_count == 0, (
    "Non-physiological epochs detected entering adaptation loss!"
)
assert min_epochs_per_night >= 600, (
    f"Calibration night has fewer than 600 valid epochs ({min_epochs_per_night})."
)
```
- **Logging:** All validation statistics must be serialized to `experiments/EXP-019/ua2_quality_gate_audit.log` before training begins.

---

## 4. Unsupervised Adaptation Objective & Known Limitations

### 4.1 Information Maximization (InfoMax) Loss
$$\mathcal{L}_{\text{unsupervised}} = \mathcal{L}_{\text{entropy}} - \lambda \mathcal{L}_{\text{diversity}}$$

1. **Conditional Entropy Minimization (Prediction Confidence):**
   $$\mathcal{L}_{\text{entropy}} = -\frac{1}{|\mathcal{C}|} \sum_{t \in \mathcal{C}} \sum_{c=1}^4 p(y_t = c | x_t) \log p(y_t = c | x_t)$$
2. **Marginal Diversity Regularization (Anti-Collapse Penalty):**
   $$\bar{p}_c = \frac{1}{|\mathcal{C}|} \sum_{t \in \mathcal{C}} p(y_t = c | x_t), \quad \mathcal{L}_{\text{diversity}} = -\sum_{c=1}^4 \bar{p}_c \log \bar{p}_c$$
   Diversity weight is fixed at $\lambda = 1.0$.

### 4.2 Documented Limitation: The Uniform Marginal Regularizer
- **The Issue:** The Shannon entropy of the marginal class distribution $-\sum_c \bar{p}_c \log \bar{p}_c$ achieves its theoretical maximum when $\bar{p}$ is **uniform across all 4 classes** ($25\%$ Wake, $25\%$ Light, $25\%$ Deep, $25\%$ REM).
- **The Physiological Conflict:** Normal human sleep is heavily class-imbalanced: Light sleep (N1+N2) comprises ~50% of epochs, Deep sleep (N3) ~20%, REM ~20%, and Wake ~10%.
- **Methodological Impact:** Forcing the marginal distribution toward a uniform 25% target actively penalizes the natural biological dominance of Light sleep and can artificially inflate Wake and REM predictions.
- **Reporting Requirement:** This experiment explicitly records class marginal shifts ($\bar{p}_c$) to document whether uniform marginal regularization induces class-prior distortion on low-rate vitals.

---

## 5. Hyperparameter Inheritance & Statistical Inference

- **Backbone:** Frozen EXP-018 Phase 2 Model A checkpoints ($5\text{ folds} \times 3\text{ seeds}$).
- **Trainable Parameters:** Prediction head and last recurrent layer (`head_gru`), keeping convolutional feature extractors frozen to prevent distortion of temporal wave filters.
- **Hyperparameters (Stage D Selected):** Learning rate $\eta = 1 \times 10^{-5}$, 20 gradient steps, AdamW (weight decay $10^{-2}$), full-night context ($L = 1200$).
- **Cohort:** Exactly the $N = 46$ eligible test participants ($n_{\text{usable}} \ge 3$), evaluated on $E_2$ (152 nights), identical to $FT\text{-}2$.
- **Inference:** Participant-level paired differences $\Delta_i(\text{UA-2}) = \text{Macro-F1}_i(UA\text{-}2) - \text{Macro-F1}_i(ZS)$ evaluated via 10,000-resample bootstrap and two-sided Wilcoxon signed-rank test.

---

## 6. Formal Decision Rules

The experimental outcome will be classified strictly according to the following mutually exclusive and exhaustive criteria:

```
+-----------------------------------------------------------------------------------------+
|                                    DECISION MATRIX                                      |
+-----------------------------------------------------------------------------------------+
|                                                                                         |
|  1. SUPPORTED (Clinically Relevant Label-Free Gain):                                    |
|     - Condition: 95% bootstrap CI lower bound > 0.0 pp                                  |
|       AND point estimate \bar{\Delta}_{UA-2} >= +1.00 pp.                               |
|     - Scientific Meaning: Confirms that unsupervised test-time adaptation calibrates to |
|       individual autonomic baselines without ground-truth hypnograms.                   |
|                                                                                         |
|  2. SUB-BENCHMARK GAIN (Statistically Reliable but < +1.00 pp):                         |
|     - Condition: 95% bootstrap CI lower bound > 0.0 pp                                  |
|       AND point estimate \bar{\Delta}_{UA-2} < +1.00 pp.                                |
|     - Scientific Meaning: Unsupervised adaptation yields a genuine positive effect,     |
|       but the magnitude is modest.                                                      |
|                                                                                         |
|  3. NEUTRAL / PRACTICAL EQUIVALENCE:                                                    |
|     - Condition: 95% bootstrap CI spans 0.0 pp AND is entirely contained within         |
|       [-0.50, +0.50] pp.                                                                |
|     - Scientific Meaning: Proves that label-free adaptation neither helps nor harms      |
|       the model; representations remain stable but unenhanced.                          |
|                                                                                         |
|  4. INCONCLUSIVE:                                                                       |
|     - Condition: 95% bootstrap CI spans 0.0 pp BUT extends outside [-0.50, +0.50] pp.   |
|     - Scientific Meaning: Sample variance is too large to establish equivalence or      |
|       superiority.                                                                      |
|                                                                                         |
|  5. DESTRUCTIVE DRIFT (Negative Transfer / Mode Collapse):                              |
|     - Condition: 95% bootstrap CI upper bound < 0.0 pp.                                 |
|     - Scientific Meaning: Entropy minimization causes representation collapse,          |
|       statistically degrading performance relative to unadapted zero-shot.             |
+-----------------------------------------------------------------------------------------+
```

---

## 7. Computational Execution Requirements

- **Total Adaptation Runs:** 46 subjects $\times$ 3 seeds = 138 runs (20 gradient steps each).
- **Target Hardware & Memory Profile:**
  - Hardware: NVIDIA GeForce RTX 3060 (12 GB VRAM).
  - Memory consumption: Peak VRAM is **460 MiB** at batch size $B=4, L=1200$.
  - System RAM impact: Feature cache occupies ~1.8 GB RAM; total Python process memory $< 3\text{ GB}$.
- **Execution Strategy & Runtime:**
  - Runtime per subject-seed run: ~12 seconds on RTX 3060.
  - Total wall-clock time: **~28 minutes** on a single RTX 3060.
  - Concurrency: Can run simultaneously on GPU 0 while TSK-06 executes on GPU 1, introducing zero execution bottlenecks or kernel contention.
- **Output Artifacts:**
  - `experiments/EXP-019/exp019_ua2_rerun_subject_metrics.csv`
  - `experiments/EXP-019/ua2_quality_gate_audit.log`
  - Historical artifact `exp019_subject_metrics.csv` remains strictly preserved and locked.
