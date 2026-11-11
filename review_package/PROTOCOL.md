# PROTOCOL.md: Experimental Protocol and Data Pipeline Specification

This document provides a complete, plain-language description of the end-to-end data pipeline, feature engineering, model architectures, loss formulations, and training configurations implemented across the SLEEP research project on the BIDSLab Apple Watch + Dreem 2 dataset.

---

## 1. Dataset Overview and Hardware Signals

The study investigates automated 4-class sleep staging using consumer wearable sensors evaluated against clinical-grade portable polysomnography (PSG).

- **Cohort**: 47 healthy human subjects recorded across multiple nights in a free-living home sleep environment.
- **Wearable Device (Input Signals)**: Apple Watch Series 3 / 4 worn on the non-dominant wrist.
  - **Photoplethysmography (PPG) / Instantaneous Heart Rate (IHR)**: Emits non-uniform pulse event timestamps at irregular intervals (average effective sampling frequency $\sim 0.2\text{ Hz}$, typical interval 3–10 seconds).
  - **Triaxial Accelerometer**: Captures wrist motion at a nominal sampling frequency of 50 Hz along three orthogonal spatial axes $(a_x, a_y, a_z)$ in units of $g$.
- **Reference Ground Truth**: Dreem 2 EEG headband worn concurrently.
  - Derives sleep stage classifications according to the American Academy of Sleep Medicine (AASM) standards for consecutive, non-overlapping 30-second epochs.
  - Raw label mapping:
    - `0`: Wake
    - `1`: N1 (Stage 1 Light NREM)
    - `2`: N2 (Stage 2 Light NREM)
    - `3`: N3 (Stage 3 Slow-Wave / Deep Sleep)
    - `4`: REM (Rapid Eye Movement)
    - `5`: Unknown / recording artifact / unassigned

---

## 2. Preprocessing, Interpolation, and Feature Extraction

Raw data are transformed into a standardized 1-Hz multi-channel representation per 30-second epoch.

### 2.1 Heart Rate Resampling and Interpolation
- Apple Watch PPG pulse events are mapped to relative time offsets (in seconds) from verified recording onset.
- **1-Hz Temporal Grid**: An evenly spaced 1-second grid ($f_s = 1\text{ Hz}$, 30 samples per 30-second epoch) is constructed.
- **Gap Tolerance**: 1D linear interpolation is performed across heart-rate measurements up to a maximum continuous gap threshold of **120 seconds**.
- **Coverage Masking**: If a gap between valid PPG pulse readings exceeds 120 seconds, or if within a given 30-second epoch fewer than 50% of the seconds contain valid interpolated data, the epoch is flagged as invalid (`cov_hr < 0.5`).

### 2.2 Accelerometry Processing and Feature Engineering
- Triaxial accelerometer samples (50 Hz) are binned and aggregated into the 1-Hz grid.
- Derived motion channels include:
  1. Orthogonal linear accelerations $(a_x, a_y, a_z)$.
  2. Euclidean vector magnitude: $\text{mag}_t = \sqrt{a_{x,t}^2 + a_{y,t}^2 + a_{z,t}^2}$.
  3. Dynamic acceleration deviation: $\text{dyn}_t = |\text{mag}_t - \text{median}(\text{mag})|$.
  4. Jerk (first temporal difference of acceleration): $\text{jerk}_t = |\text{mag}_t - \text{mag}_{t-1}| \cdot f_s$.
  5. Rolling energy/variance: moving mean of squared dynamic acceleration over a 60-second window.
  6. Pulse and motion density counters: count of raw sensor samples registered within each second.
- Motion validity requires at least 50% sensor availability per epoch (`cov_motion >= 0.5`).

### 2.3 Feature Normalization Locality
- All continuous signal channels are scaled using a **strictly per-night** robust scaling algorithm:
  $$\text{scaled}(v) = \text{clip}\left(\frac{v - \text{median}(v)}{\max(\text{IQR}(v), \text{floor})}, -10, 10\right)$$
  where $\text{IQR} = Q_{0.75} - Q_{0.25}$.
- **Zero cross-subject normalization**: Medians and IQRs are computed independently on each recording night. No dataset-wide or split-wide statistics are ever shared.

### 2.4 Sleep Stage Label Consolidation
The 6 raw Dreem 2 classes are projected into 4 consolidated clinical stages:
- **Class 0 (Wake)**: Raw stage 0.
- **Class 1 (Light Sleep)**: Raw stages 1 (N1) and 2 (N2).
- **Class 2 (Deep Sleep)**: Raw stage 3 (N3 slow-wave sleep).
- **Class 3 (REM Sleep)**: Raw stage 4.
- **Unknown / Artifact (-100)**: Raw stage 5 and any epochs where sensor validity falls below 50% (`mask == 0`). These are encoded as target value `-100` and masked out of all loss functions and evaluation metrics.

### 2.5 Usable Night Inclusion Criteria
- A night is declared **usable** if overall signal coverage $\ge 60\%$ across both PPG and accelerometry.
- Out of 253 total recorded nights:
  - **7 nights are excluded** due to sub-threshold sensor coverage ($< 60\%$).
  - **246 nights are retained as usable** across the 47 subjects.

---

## 3. Dataset Splits and Partitioning

To prevent data leakage, all splits are strictly **subject-disjoint** (all nights from a single subject reside exclusively in one partition).

### 3.1 Official 3-Way Partition (EXP-012, EXP-013)
- **Training Set**: 31 subjects, 160 usable nights (164 raw nights, 4 excluded).
- **Validation Set**: 5 subjects, 28 usable nights (28 raw nights, 0 excluded).
- **Test Set**: 11 subjects, 58 usable nights (61 raw nights, 3 excluded).
- **Zero subject or night overlap** across partitions.

### 3.2 5-Fold Cross-Validation Partition (EXP-014)
- 47 subjects divided into 5 disjoint test folds (stratified by subject night counts):
  - **Fold 0**: 10 test subjects (53 usable nights), 5 validation subjects (27 usable nights), 32 train subjects (166 usable nights).
  - **Fold 1**: 10 test subjects (51 usable nights), 5 validation subjects (26 usable nights), 32 train subjects (169 usable nights).
  - **Fold 2**: 9 test subjects (47 usable nights), 5 validation subjects (27 usable nights), 33 train subjects (172 usable nights).
  - **Fold 3**: 9 test subjects (49 usable nights), 5 validation subjects (26 usable nights), 33 train subjects (171 usable nights).
  - **Fold 4**: 9 test subjects (46 usable nights), 5 validation subjects (26 usable nights), 33 train subjects (174 usable nights).
- Every one of the 47 subjects appears exactly once in a held-out test fold.

---

## 4. Sequence Truncation and Cropping Mechanics

- **Fixed Length $T = 600$**: Each night recording is cropped to the first 600 epochs ($T = 600 \times 30\text{ s} = 5.0\text{ hours}$) starting from index 0 (`rec["ihr"][:600]`).
- **Zero Padding**: Because all 246 usable nights contain $\ge 610$ epochs, zero-padding was never applied ($0\%$ padding artifacts).
- **Circadian Consequence**: Dropping epochs beyond hour 5 (an average of 28.55% of labeled data per night) enriches the sample for early-night slow-wave sleep (Deep sleep increases from 18.91% to 25.26%) and depletes late-night REM sleep (REM drops from 26.06% to 21.33%).

---

## 5. Model Architectures and OFAT Ablation Matrix

All models operate on 1-Hz inputs shaped $(B, T, 30, 1)$ for HR and ACC, alongside a $(B, T, 5)$ metadata tensor.

### 5.1 Common Encoders and Metadata Projection
- **HREncoder / ACCEncoder**: Parallel 1D convolutional feature extractors:
  - Conv1D(1, 32, kernel=7, stride=1, padding=3) $\to$ BatchNorm $\to$ GELU $\to$ MaxPool1D(2)
  - Conv1D(32, 64, kernel=5, stride=1, padding=2) $\to$ BatchNorm $\to$ GELU $\to$ AdaptiveAvgPool1D(1)
  - Produces 64-dimensional epoch-level modality embeddings $h_{\text{HR}, t}$ and $h_{\text{ACC}, t}$.
- **Meta Feature Projection**: Linear(5, 16) $\to$ GELU projects the 5 metadata features (`freq`, `clock`, `time_feat`, `hr_mean`, `hr_std`).

### 5.2 Five-Model Experimental Matrix
1. **Model A (Conventional CNN-BiGRU Baseline)**:
   - Modality fusion: Direct concatenation $[h_{\text{HR}} ; h_{\text{ACC}} ; h_{\text{meta}}]$ (dimension 144).
   - Temporal sequence model: 2-layer Bidirectional GRU (hidden size 48 per direction $\to$ total 96-d).
   - Parameters: **163,940**.
   - No adaptive gate, no transition matrix, no GRL, no boundary loss.
2. **Model B (Full Model minus Subject GRL)**:
   - Adaptive reliability fusion gate ON; Transition module ON; Subject GRL OFF.
   - Parameters: **153,974**.
3. **Model C (Full Model minus Transition Module)**:
   - Adaptive reliability fusion gate ON; Transition module OFF; Subject GRL ON.
   - Parameters: **162,325**.
4. **Model D (Full Model minus Adaptive Gating)**:
   - Static 50/50 fusion ($0.5 \cdot h_{\text{HR}} + 0.5 \cdot h_{\text{ACC}}$); Transition module ON; Subject GRL ON.
   - Parameters: **153,876**.
5. **Model E (Full Proposed SleepAdaptiveNet / Frozen EXP-012 Reference)**:
   - Adaptive reliability fusion gate ON; Transition module ON; Subject GRL ON.
   - Parameters: **162,325**.

---

## 6. Loss Function Formulation and Optimization

Training minimizes a multi-task composite objective:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{stage}} + \lambda_{\text{trans}} \mathcal{L}_{\text{trans}} + \lambda_{\text{adv}} \mathcal{L}_{\text{adv}}$$

### 6.1 Primary Stage Loss ($\mathcal{L}_{\text{stage}}$)
$$\mathcal{L}_{\text{stage}} = \mathcal{L}_{\text{CE}} + 0.01 \cdot \mathcal{L}_{\text{RCC}}$$
- $\mathcal{L}_{\text{CE}}$: Standard cross-entropy computed per epoch on valid targets ($0 \le y < 4$).
- $\mathcal{L}_{\text{RCC}}$: Relative class-cost loss penalizing clinically severe classification mistakes (e.g., misclassifying Wake as Deep sleep) using fixed author constants ($w=16433, l=63668, d=25685, r=32015$).

### 6.2 Transition Loss ($\mathcal{L}_{\text{trans}}$)
Learns a $4 \times 4$ Markovian transition compatibility matrix $M$:
$$\mathcal{L}_{\text{trans}} = - \frac{1}{\sum v_t} \sum_{t=1}^{T-1} v_t \log \left[\text{softmax}(M)_{y_t, y_{t+1}}\right]$$
Weighted by $\lambda_{\text{trans}} = 0.10$. Only transitions between two valid sleep epochs are evaluated.

### 6.3 Adversarial Subject Loss ($\mathcal{L}_{\text{adv}}$)
Encourages domain/subject-invariant temporal representations across training subjects:
$$\mathcal{L}_{\text{adv}} = \text{CrossEntropy}(\text{Discriminator}(\text{GRL}(h)), \text{subject\_id})$$
Operates via Gradient Reversal Layer with constant $\alpha_{\text{grl}} = 0.10$ and loss weight $\lambda_{\text{adv}} = 0.10$, yielding an effective gradient scale of $-0.010$ to the temporal GRU encoder.

---

## 7. Hyperparameters and Training Configuration

All models across all experiments are trained under the identical locked hyperparameter budget:
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999$, $\epsilon = 10^{-8}$)
- **Learning Rate**: $1 \times 10^{-4}$ (fixed; no learning rate warmup or decay schedule)
- **Weight Decay**: $0.0$
- **Dropout Rate**: $0.20$
- **Batch Size**: 4 nights per step
- **Epoch Budget**: Exactly 80 epochs
- **Early Stopping**: None (frozen epoch-80 checkpoint evaluated)
- **Random Seeds**: 42, 123, 2024
