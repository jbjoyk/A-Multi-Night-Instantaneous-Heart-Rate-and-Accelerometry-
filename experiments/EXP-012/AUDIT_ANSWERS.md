# AUDIT_ANSWERS.md: Comprehensive Methodological and Technical Audit

This audit document provides verified answers to the 9 critical questions raised during the external audit of the SLEEP research project (wearable sleep staging on the BIDSLab Apple Watch + Dreem 2 dataset). Every numerical value and code citation in this document was computed directly from saved source files and verified artifacts.

---

## Q1. Window Count Reconciliation (EXP-012 Test Benchmark)

### Question Summary
In the EXP-012 test benchmark, 58 usable nights $\times$ 600 epochs = 34,800 epoch slots. However, results report 34,241 evaluated windows. Where did the missing 559 windows go? What is the sum of the confusion matrix in `exp012_test_results.pkl`? What is the support sum in per-class metrics? Which count is the TRUE count of evaluated windows, 34,800 or 34,241? Were unknown labels (-100 or stage 5) counted in accuracy/kappa?

### Verified Audit Findings

1. **Confusion Matrix Sum**:
   In `exp012_test_results.pkl`, the confusion matrix sum across all three evaluated seeds is exactly **34,241**:
   - Seed 42: $\sum \text{CM} = 34,241$
   - Seed 123: $\sum \text{CM} = 34,241$
   - Seed 2024: $\sum \text{CM} = 34,241$

2. **Per-Class Support Sum**:
   The support sum in `exp012_test_results.pkl` and `review_package/exp012_per_seed_metrics.csv` is exactly **34,241**:
   - Wake (Class 0): $4,443$ epochs ($12.98\%$)
   - Light (Class 1): $13,878$ epochs ($40.53\%$)
   - Deep (Class 2): $8,639$ epochs ($25.23\%$)
   - REM (Class 3): $7,281$ epochs ($21.26\%$)
   - Total Support: $4,443 + 13,878 + 8,639 + 7,281 = \mathbf{34,241}$

3. **Breakdown of Missing 559 Windows ($34,800 - 34,241 = 559$)**:
   The 559 un-evaluated windows consist of:
   - **264 epochs** where the ground-truth Dreem 2 EEG stage was raw label 5 (Unknown stage, recording artifact, or unassigned).
   - **295 epochs** where signal coverage dropped below the 50% validity threshold (`cov_hr < 0.5` or `cov_motion < 0.5`), setting `mask[t] = 0`.
   - Total dropped windows: $264 + 295 = \mathbf{559}$.

4. **True Count of Evaluated Windows**:
   - **34,241** is the **TRUE count** of physiologically valid, fully scored sleep-stage epochs.
   - 34,800 is merely the unmasked rectangular tensor shape ($58 \text{ nights} \times 600 \text{ epochs}$).

5. **Filtering of Unknown Labels in Accuracy and Kappa**:
   - Unknown labels (-100 or stage 5) were **STRICTLY EXCLUDED** from accuracy, macro-F1, and Cohen's kappa.
   - Code verification in `run_exp012_final_benchmark.py`, lines 88–101:
     ```python
     valid = (targets >= 0) & (targets < 4) & mask
     # ...
     m_b = valid[b_i].cpu().numpy()
     p_b = preds[b_i][m_b].cpu().numpy()
     t_b = targets[b_i][m_b].cpu().numpy()
     all_preds.extend(p_b)
     all_targets.extend(t_b)
     ```
   - Only epochs satisfying `(targets >= 0) & (targets < 4) & mask` were accumulated. No unknown epochs entered any classification metric.

---

## Q2. Sequence Length and Cropping Mechanics (`ControlledSleepDataset`)

### Question Summary
In `ControlledSleepDataset`, how is the $T=600$ window selected? (start offset, random crop, fixed first 5 hours?). What is the distribution of total epochs per night across the 246 usable nights? (min, median, mean, max). What fraction of labeled epochs were dropped by taking only $T=600$? (per-night and overall). Was zero-padding used on short nights?

### Verified Audit Findings

1. **Window Selection Strategy**:
   - In `ControlledSleepDataset` (`run_controlled_experiment.py`, lines 75–80):
     ```python
     T = self.seq_len  # 600
     ihr = rec["ihr"][:T]
     acc = rec["accel"][:T]
     freq = rec["freq"][:T]
     time_feat = rec["time"][:T]
     y = rec["y"][:T]
     ```
   - The crop has **start offset = 0** (`[:600]`). It is a **fixed first-5-hour window** (first 600 epochs $\times$ 30 seconds = 18,000 seconds = 5.0 hours from recording onset).
   - It is NOT a random crop and NOT centered on sleep onset.

2. **Distribution of Total Epochs per Night (246 Usable Nights)**:
   - **Minimum**: 610 epochs (5.08 hours)
   - **Median**: 857.5 epochs (7.15 hours)
   - **Mean**: 861.14 epochs (7.18 hours)
   - **Maximum**: 1,282 epochs (10.68 hours)

3. **Fraction of Labeled Epochs Dropped by $T=600$ Truncation**:
   - **Overall across all 246 usable nights**:
     - Total labeled epochs in full usable nights: 203,338
     - Total labeled epochs retained in $T=600$ windows: 145,283
     - Total labeled epochs dropped: $203,338 - 145,283 = \mathbf{58,055}$
     - Overall dropped fraction: $\mathbf{28.55\%}$ ($58,055 / 203,338$)
   - **Per-night distribution of dropped fraction**:
     - Minimum: $1.48\%$ (night with 610 epochs, 9 epochs dropped)
     - 25th percentile: $22.77\%$
     - Median: $28.43\%$
     - Mean: $26.81\%$
     - 75th percentile: $31.81\%$
     - Maximum: $50.25\%$ (night with 1,208 epochs, 607 epochs dropped)

4. **Zero-Padding on Short Nights**:
   - **NO zero-padding was used**. Every single usable night in the dataset had at least 610 epochs ($\ge 610 > 600$). Zero padding was required for zero nights.

---

## Q3. Class Mix Shift from 5-Hour Cropping

### Question Summary
What are the class fractions (Wake, Light, Deep, REM) in the FULL 246 usable nights vs the CROPPED $T=600$ windows? Does taking the first 5 hours enrich for Deep sleep and deplete REM? By how many percentage points?

### Verified Audit Findings

The exact stage distributions computed across all 246 usable nights are:

| Sleep Stage | Full Nights Count | Full Nights % | Cropped $T=600$ Count | Cropped $T=600$ % | Absolute Shift (pp) | Relative Shift (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Wake** (0) | 22,019 | 10.83% | 18,487 | 12.72% | **+1.90 pp** | +17.5% |
| **Light** (1) | 89,881 | 44.20% | 59,103 | 40.68% | **-3.52 pp** | -8.0% |
| **Deep** (2) | 38,451 | 18.91% | 36,701 | 25.26% | **+6.35 pp** | **+33.6%** |
| **REM** (3) | 52,987 | 26.06% | 30,992 | 21.33% | **-4.73 pp** | **-18.1%** |
| **Total Valid** | **203,338** | 100.0% | **145,283** | 100.0% | — | — |

### Physiological Mechanism
- **Deep Sleep Enrichment (+6.35 pp, +33.6% relative)**: Non-REM slow-wave sleep (N3) is under strong homeostatic sleep pressure (Process S) and concentrates heavily in the first 2 sleep cycles (hours 1–3.5 of the night). Because $T=600$ captures the first 5 hours, almost all Deep sleep episodes ($36,701 / 38,451 = 95.45\%$) are retained.
- **REM Sleep Depletion (-4.73 pp, -18.1% relative)**: REM sleep episodes are circadian-governed and progressively lengthen across successive ultradian cycles, reaching maximum duration in hours 5–8 (the second half of the night). Truncating at hour 5 drops $21,995$ REM epochs ($41.51\%$ of all REM sleep in the dataset).

---

## Q4. Metadata Features Representation (`x_meta`) and Baseline Model Attribution

### Question Summary
`ControlledSleepDataset` constructs 5 features in `x_meta` (`freq`, `clock`, `time_feat`, `hr_mean`, `hr_std`). List exactly what each of these 5 features contains. Does `clock` use wall-clock time from the original recording? Does `hr_mean` / `hr_std` use cross-subject statistics or per-night / per-epoch? Does Model A (conventional CNN-BiGRU) receive `x_meta`? If so, where is it concatenated?

### Verified Audit Findings

1. **Exact Contents of the 5 `x_meta` Features**:
   - **Feature 0 (`freq`)**: Instantaneous heart rate sampling density for the epoch. Constructed in preprocessing as:
     `freq = np.log1p(ihr_count / 30.0)`
     where `ihr_count` is the number of raw, non-interpolated Apple Watch photoplethysmography (PPG) pulse timestamps falling within the 30-second epoch.
   - **Feature 1 (`clock`)**: **Constant zeros**. In `ControlledSleepDataset` line 84:
     `clock = np.zeros_like(time_feat)`
     `clock` does **NOT** contain wall-clock time, time-of-day, or circadian hour. It is a dummy zero placeholder.
   - **Feature 2 (`time_feat`)**: Elapsed recording time normalized to 10 hours:
     `time_feat[t] = t * 30.0 / 36000.0`
     Starts at $0.0$ at epoch 0 and increases linearly to $0.499167$ ($600 \times 30 / 36000$) at epoch 599.
   - **Feature 3 (`hr_mean`)**: Within-epoch mean heart rate. In line 82:
     `hr_mean = ihr.mean(axis=1, keepdims=True)`
     Computed over the 30 1-Hz interpolated heart rate samples of that specific 30-second epoch.
   - **Feature 4 (`hr_std`)**: Within-epoch heart rate standard deviation. In line 83:
     `hr_std = ihr.std(axis=1, keepdims=True)`
     Computed over the 30 1-Hz interpolated heart rate samples of that specific 30-second epoch.

2. **Cross-Subject Statistics**:
   - Neither `hr_mean` nor `hr_std` uses cross-subject or dataset-wide statistics. They are strictly **local, within-epoch** summary statistics.

3. **Model A Receipt and Concatenation**:
   - **YES**, Model A (`ConventionalCNNBiGRU` in `exp013_models.py` / `exp014_models.py`) receives `x_meta`.
   - In `exp013_models.py`, lines 69–84:
     ```python
     self.meta_proj = nn.Sequential(nn.Linear(5, 16), nn.GELU())
     in_temp = 2 * d_model + 16  # 64 (HR) + 64 (ACC) + 16 (Meta) = 144
     self.temporal_gru = nn.GRU(input_size=144, hidden_size=48, num_layers=2, bidirectional=True)
     ```
   - In lines 117–123:
     ```python
     h_hr = self.hr_encoder(x_ihr)       # (B, T, 64)
     h_acc = self.acc_encoder(x_accel)   # (B, T, 64)
     h_meta = self.meta_proj(x_meta)     # (B, T, 16)
     cat_features = torch.cat([h_hr, h_acc, h_meta], dim=-1)  # (B, T, 144)
     temp_out, _ = self.temporal_gru(cat_features)
     ```
   - `x_meta` is projected to a 16-dimensional embedding and concatenated with the CNN-extracted HR (64-d) and ACC (64-d) embeddings to form the 144-dimensional input to the 2-layer BiGRU.

---

## Q5. Preprocessing Cache and Feature Normalization Locality

### Question Summary
Does the preprocessing cache (`prep_57738d534b`) normalize features using statistics computed across all subjects before splitting? Or is normalization strictly per-night? Show the exact line of code that normalizes. Check RCCLoss class weights: are they computed from training fold, all data, or hardcoded?

### Verified Audit Findings

1. **Normalization Locality**:
   - Feature normalization is **STRICTLY PER-NIGHT**.
   - No feature uses cross-subject pooling, split-wide statistics, or dataset-wide z-scoring before or after splitting.
   - Code verification in `sleepchannelnet_protocol_patch_v2.ipynb`, Cell 16, lines 1–3:
     ```python
     def robust_scale(v, floor):
         med = np.nanmedian(v)
         iqr = np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
         return np.clip(np.nan_to_num((v - med) / max(float(iqr), floor)), -10, 10).astype(np.float32)
     ```
   - Inside `process_night(hr, mot, stage, rec_start, c: Config)`, `robust_scale` is called directly on the 1D arrays of each single night recording. The median and interquartile range (IQR) are calculated exclusively on that night's data.

2. **RCCLoss Class Weights**:
   - The RCCLoss class weights are **HARDCODED CONSTANTS** originating from the official SLAMSS-IFS author repository (`BIDSLabUMass/SLAMSS-IFS`).
   - Code verification in `test_rccloss_audit.py`, lines 37–51:
     ```python
     def get_official_weights():
         w, l, d, r = 16433.0, 63668.0, 25685.0, 32015.0
         fn_weights = [
             [1.0 / 0.0878, 0.00, 0.00, 0.00],
             [0.00, 1.0 / 0.4636, 0.00, 0.00],
             [0.00, 0.00, 1.0 / 0.1940, 0.00],
             [0.00, 0.00, 0.00, 1.0 / 0.2548]
         ]
         fp_weights4 = [
             [0.00, (w / l)**0.5, (w / d)**0.5, (w / r)**0.5],
             [(l / w)**0.5 * 0.5, 0.00, (l / d)**0.5 * 0.5, (l / r)**0.5],
             [(d / w)**0.5, (d / l)**0.5 * 0.5, 0.00, (r / l)**0.5 * 0.5],
             [(r / w)**0.5 * 0.5, (r / l)**0.5, (r / d)**0.5, 0.00]
         ]
         return fn_weights, fp_weights4
     ```
   - They were **NOT** computed from the training folds and were **NOT** fit dynamically on the current dataset. They are fixed external benchmark constants.

---

## Q6. Unknown Label Handling and Boundary Loss Protection

### Question Summary
How are unknown epochs (stage 5 or unlabelled) encoded in targets? In `compute_boundary_loss`, is there a check for valid labels? Could an unknown epoch produce a fake transition boundary? In `CrossEntropyLoss`, is `ignore_index` set?

### Verified Audit Findings

1. **Target Encoding**:
   - Unknown epochs and low-coverage dropout epochs are encoded as **`-100`**.
   - In `sleepchannelnet_protocol_patch_v2.ipynb`, Cell 16, line 111:
     `y = stage.astype(np.int64).copy(); y[~valid] = UNKNOWN` (where `UNKNOWN = -100`).

2. **Boundary Loss Protection**:
   - In `compute_boundary_loss` (`sleep_adaptive_net.py`, lines 439–444):
     ```python
     y_prev = targets[:, :-1]
     y_curr = targets[:, 1:]
     b_inner = (y_prev != y_curr) & (y_prev >= 0) & (y_curr >= 0)
     boundary_indicator = torch.zeros(B, T, dtype=torch.bool, device=targets.device)
     boundary_indicator[:, 1:] = b_inner
     ```
   - Notice `(y_prev >= 0) & (y_curr >= 0)`.
   - If either the preceding epoch $y_{t-1}$ or the current epoch $y_t$ is unknown ($-100$), `b_inner` is `False`.
   - An unknown epoch **CANNOT** produce a false positive transition boundary.

3. **Transition Module Loss Protection**:
   - In `TransitionAwareModule.forward` (`sleep_adaptive_net.py`, line 248):
     `valid = (y_prev >= 0) & (y_prev < K) & (y_curr >= 0) & (y_curr < K)`
   - Both states must be valid non-negative stage labels ($0 \le y < 4$).

4. **CrossEntropyLoss and Masking Mechanics**:
   - `self.ce_loss = nn.CrossEntropyLoss(reduction="none")` is initialized without `ignore_index`.
   - To prevent negative indexing faults on GPU, targets are clamped before loss computation, and subsequently masked:
     ```python
     valid = (flat_targets >= 0) & (flat_targets < self.n_classes) & mask.view(-1)
     per_epoch_ce = self.ce_loss(flat_logits, torch.clamp(flat_targets, 0, self.n_classes - 1))
     ce_term = per_epoch_ce.view(B, T)[valid.view(B, T)].mean()
     ```
   - Unknown epochs ($-100$) have `valid == False` and are completely excluded from the mean loss reduction.

---

## Q7. Subject-Invariant GRL Gradient Scaling and Scheduling

### Question Summary
In `SleepAdaptiveNet`, what is the value of `alpha` in `ReverseLayerF` during training? What is `lambda_adv`? What is the effective gradient scale from discriminator to the encoder? Is it scheduled or constant? Show the code.

### Verified Audit Findings

1. **GRL Scaling Hyperparameters**:
   - In `GradientReversalLayer` (`sleep_adaptive_net.py`, line 48): initialized with `alpha_grl = 0.1` (`sleep_adaptive_net.py:563`).
   - In `SleepAdaptiveNet` (`sleep_adaptive_net.py`, line 498): `lambda_adv = 0.1` (`lambda_adv: float = 0.1`).
   - In `total_loss` computation (`sleep_adaptive_net.py`, line 678):
     `total_loss = loss_stage + ... + self.lambda_adv * loss_adv`

2. **Effective Gradient Flow to Encoder**:
   - When backpropagating `loss_adv`:
     $$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \mathcal{L}_{\text{adv}}} = \lambda_{\text{adv}} = 0.10$$
   - In `GradientReversalFunction.backward` (`sleep_adaptive_net.py`, line 36):
     ```python
     output = grad_output.neg() * ctx.alpha  # -0.10 * grad_output
     ```
   - Consequently, the effective gradient scale arriving at the GRU temporal encoder is:
     $$\text{Effective Scale} = -\lambda_{\text{adv}} \times \alpha_{\text{grl}} = -0.10 \times 0.10 = \mathbf{-0.010}$$
   - The adversarial gradient is attenuated by **100-fold** (scaled down to 1% of the discriminator's raw gradient).

3. **Scheduling Status**:
   - **CONSTANT**.
   - Although `GradientReversalLayer.set_alpha(self, alpha: float)` was defined (`sleep_adaptive_net.py:50`), it was **never called** in any training loop (`grep -n "set_alpha" *.py` finds only the class definition).
   - The standard Ganin & Lempitsky (2015) progressive sigmoid schedule was **not implemented**; `alpha` remained fixed at $0.10$ for all 80 epochs.

---

## Q8. Official Test Set Touchpoint History (EXP-001 through EXP-014)

### Question Summary
List every experiment (EXP-001 through EXP-014) that has run evaluation on the official 11-subject test partition, with date and purpose.

### Verified Audit Findings

| Experiment | Execution Date | Evaluated on Official 11-Subject Test Set? | Purpose and Details |
| :--- | :---: | :---: | :--- |
| **EXP-001** | Prior | No | Historical run evaluated on old 33/7/7 split (fingerprint `459693836b56`, 7 test subjects). |
| **EXP-002** | Prior | No | Implementation only; no test evaluation. |
| **EXP-003** | Prior | No | Evaluated complete-graph model on old 7-subject test split. |
| **EXP-004** | Prior | No | Evaluated optimizer search on old 7-subject test split. |
| **EXP-005** | 2026-11-09 | No | Verified scientific equivalence and audited 31/5/11 split definitions; test set was not evaluated. |
| **EXP-006** | 2026-11-09 | No | Smoke tests and dummy forward passes; test set was not evaluated. |
| **EXP-007** | 2026-11-09 | No | Architecture search (A0–A5) on 5 validation subjects; test set strictly not loaded. |
| **EXP-008** | 2026-11-09 | No | Boundary weight sweep on 5 validation subjects; test set strictly excluded. |
| **EXP-009** | 2026-11-09 | No | Multi-seed stability on 5 validation subjects; test set strictly excluded. |
| **EXP-010** | 2026-11-10 | No | 500-epoch convergence study on validation set; test set strictly excluded. |
| **EXP-011** | 2026-11-10 | No | Multi-seed convergence confirmation on validation set; test set strictly excluded. |
| **EXP-012** | **2026-11-10** | **YES** | **First evaluation on official 11-subject test set**. Final frozen benchmark of SleepAdaptiveNet (`B0_GRL`) across seeds 42, 123, 2024 (58 usable nights, 34,241 valid windows). |
| **EXP-013** | **2026-11-10** | **YES** | **Second evaluation on official 11-subject test set**. Evaluated Models A, B, C, D across seeds 42, 123, 2024 to compute One-Factor-At-A-Time (OFAT) component attribution against Model E. |
| **EXP-014** | **2026-11-10 / 2026-11-11** | **YES (as part of 5-fold CV)** | Evaluated full 47-subject cohort in 5-fold cross-validation. The 11 official test subjects were evaluated as held-out test subjects in Fold 0 (10 test subjects) and Fold 1 (10 test subjects). |

---

## Q9. Validation vs Test Random-Seed Divergence in EXP-012

### Question Summary
In EXP-012, was validation evaluated on the validation split during the 80 epochs, or only at the end? Did running validation alter the RNG state for subsequent training epochs?

### Verified Audit Findings

1. **Validation Schedule**:
   - In EXP-012 (`run_exp012_final_benchmark.py`, line 338), validation was run **at every single epoch** ($ep \in [1, 80]$):
     ```python
     for ep in range(1, 81):
         model.train()
         for batch in train_loader:
             # train step ...
         val_m = evaluate_validation_sweep(model, val_loader, device=device)
     ```
   - This exact schedule matched EXP-011, ensuring identical epoch-by-epoch runtime execution.

2. **RNG State Invariance**:
   - Running validation **DID NOT** alter the random-number generator state for subsequent training epochs:
     - `val_loader` was instantiated with `shuffle=False` (`run_exp012_final_benchmark.py:294`).
     - `evaluate_validation_sweep` executes under `model.eval()` (disabling stochastic dropout) and `with torch.no_grad():` (`run_exp008_boundary_sweep.py:47, 59`).
     - Zero random sampling, data augmentation, or random number generator calls occur inside `evaluate_validation_sweep`.
   - The PyTorch CPU, CUDA, and NumPy random number generator streams progressed identically regardless of whether validation metrics were logged.
