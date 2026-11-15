# Handoff Report: Evaluation Pipeline, Fold 1 Manifest & Local Execution Harness

## 1. Observation

### 1.1 Fold 1 Partition Manifest (`experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`)
Direct inspection of `experiments/EXP-014/EXP014_MANIFESTS/fold_1.json` reveals the following exact subject breakdown:
```json
{
  "fold": 1,
  "outer_test_subjects": [
    "Bidslab08", "Bidslab30", "Bidslab36", "Bidslab39", "Bidslab41",
    "Bidslab45", "Bidslab47", "Bidslab64", "Bidslab65", "Bidslab68"
  ],
  "outer_train_subjects": [
    "Bidslab00", "Bidslab01", "Bidslab02", "Bidslab06", "Bidslab07",
    "Bidslab09", "Bidslab10", "Bidslab11", "Bidslab13", "Bidslab14",
    "Bidslab15", "Bidslab16", "Bidslab17", "Bidslab18", "Bidslab19",
    "Bidslab20", "Bidslab22", "Bidslab31", "Bidslab32", "Bidslab34",
    "Bidslab35", "Bidslab38", "Bidslab40", "Bidslab42", "Bidslab43",
    "Bidslab44", "Bidslab49", "Bidslab50", "Bidslab51", "Bidslab52",
    "Bidslab53", "Bidslab55", "Bidslab56", "Bidslab60", "Bidslab62",
    "Bidslab63", "Bidslab66"
  ],
  "inner_train_subjects": [
    "Bidslab00", "Bidslab01", "Bidslab02", "Bidslab06", "Bidslab07",
    "Bidslab10", "Bidslab11", "Bidslab13", "Bidslab14", "Bidslab15",
    "Bidslab18", "Bidslab19", "Bidslab20", "Bidslab22", "Bidslab31",
    "Bidslab32", "Bidslab34", "Bidslab35", "Bidslab40", "Bidslab42",
    "Bidslab43", "Bidslab44", "Bidslab49", "Bidslab50", "Bidslab51",
    "Bidslab52", "Bidslab53", "Bidslab56", "Bidslab60", "Bidslab62",
    "Bidslab63", "Bidslab66"
  ],
  "inner_val_subjects": [
    "Bidslab09", "Bidslab16", "Bidslab17", "Bidslab38", "Bidslab55"
  ],
  "n_outer_test_nights": 54,
  "n_outer_train_nights": 192,
  "n_inner_train_nights": 168,
  "n_inner_val_nights": 24
}
```
- **Subject counts**: 10 outer test subjects (54 usable nights), 32 inner train subjects (168 usable nights), 5 inner validation subjects (24 usable nights), 37 outer train subjects (192 usable nights).
- **Subject disjointness**: `outer_test_subjects ∩ outer_train_subjects == ∅`, `inner_train_subjects ∩ inner_val_subjects == ∅`.
- **Night count integrity**: 168 (train) + 24 (val) + 54 (test) = 246 usable nights across all 47 subjects.

### 1.2 Data Loading Pipeline
Inspection of `slamss_ifs_train.py` (lines 105–140) and `run_controlled_experiment.py` (lines 51–123):
- **Cache Path**: `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b` (499 precomputed `.npz` and `.json` files verified present).
- **Raw Dataset Path**: `a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0` (253 nights, 47 subjects verified).
- **Per-night arrays extracted** (`slamss_ifs_train.py:105-140`):
  - `ihr`: Channel 0 of preprocessed array `X[:, 0, :]` $\to$ shape `(T_orig, 30)` (1 Hz instantaneous heart rate over 30-second epoch).
  - `accel`: Channel 6 of preprocessed array `X[:, 6, :]` $\to$ shape `(T_orig, 30)` (accelerometer magnitude over 30-second epoch).
  - `meta`: 5 metadata features `[freq, clock, time_feat, hr_mean, hr_std]` $\to$ shape `(T_orig, 5)`.
  - `y`: Sleep stage labels mapped from raw 5 stages to 4 classes:
    - `0` $\to$ Wake (`0`)
    - `1, 2` $\to$ Light Sleep (`1`)
    - `3` $\to$ Deep Sleep (`2`)
    - `4` $\to$ REM Sleep (`3`)
    - Others $\to$ `-100` (masked/unknown).
- **Horizon padding & batch collation** (`ControlledSleepDataset` in `run_controlled_experiment.py:51-123`):
  - Sequence horizon: `seq_len = 600` epochs (5.0 hours).
  - If `T_orig < 600`, zero-padded to 600 with target `-100` and `mask = False`.
  - Batch size: `batch_size = 4` nights per batch.
  - Number of batches for Fold 1:
    - Train loader: 168 nights / 4 = 42 batches (`shuffle=True`).
    - Test loader: 54 nights / 4 = 14 batches (`shuffle=False`, 13 full batches of 4, 1 tail batch of 2).

### 1.3 Tensor Dimensions and Signatures
The DataLoader yields batch dictionaries with exact tensor dimensions:
| Key | Tensor Shape | Dtype | Semantic Meaning |
|---|---|---|---|
| `ihr` | `(batch_size, sequence_length, 30)` = `(4, 600, 30)` | `torch.float32` | 1 Hz instantaneous heart rate (30 samples/epoch) |
| `accel` | `(batch_size, sequence_length, 30)` = `(4, 600, 30)` | `torch.float32` | 1 Hz accelerometer magnitude (30 samples/epoch) |
| `meta` | `(batch_size, sequence_length, 5)` = `(4, 600, 5)` | `torch.float32` | Sampling frequency, clock, elapsed time, HR mean, HR std |
| `y` | `(batch_size, sequence_length)` = `(4, 600)` | `torch.int64` | Integer stage target $\in \{0, 1, 2, 3, -100\}$ |
| `mask` | `(batch_size, sequence_length)` = `(4, 600)` | `torch.bool` | Validity mask (`True` = valid epoch, `False` = pad) |
| `seq_len` | `(batch_size,)` = `(4,)` | `torch.int64` | Native unpadded length $L = \min(\text{native\_len}, 600)$ |
| `subject_idx` | `(batch_size,)` = `(4,)` | `torch.int64` | Integer subject ID mapping $\in [0, 31]$ |

Inside `ConventionalCNNBiGRU` (`exp013_models.py:114-140`):
- `h_hr = hr_encoder(x_ihr)`: flattens `(B, T, 30)` $\to$ `(B*T, 1, 30)`, runs 1D convs + max pooling $\to$ `(B, T, 64)`.
- `h_acc = acc_encoder(x_accel)`: flattens `(B, T, 30)` $\to$ `(B*T, 1, 30)` $\to$ `(B, T, 64)`.
- `h_meta = meta_proj(x_meta)`: `Linear(5, 16) + GELU` $\to$ `(B, T, 16)`.
- Concatenation: `torch.cat([h_hr, h_acc, h_meta], dim=-1)` $\to$ shape `(B, T, 144)`.
- Temporal BiGRU: `nn.GRU(input_size=144, hidden_size=48, num_layers=2, bidirectional=True)` $\to$ shape `(B, T, 96)`.
- Classifier Head: `Linear(96, 64) -> LayerNorm(64) -> GELU -> Dropout -> Linear(64, 4)` $\to$ logits shape `(B, T, 4)`.

### 1.4 Hyperparameters & Loss Formulation
Inspected in `run_exp014_cv.py` (lines 58–63) and `exp013_models.py` (lines 95–165):
- **Batch Size**: 4
- **Sequence Length**: 600 epochs
- **Optimizer**: `torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)`
- **Learning Rate**: `1e-4` (`0.0001`)
- **Loss Function**: Composite Stage Loss:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + 0.01 \cdot \mathcal{L}_{\text{RCC}}$$
  where:
  - $\mathcal{L}_{\text{CE}}$ is cross-entropy on valid positions: `valid = (targets >= 0) & (targets < 4) & mask`.
  - $\mathcal{L}_{\text{RCC}}$ is the official `RCCLoss` using official asymmetric false negative / false positive cost matrices obtained via `get_official_weights()`.

### 1.5 Execution Environment & Hardware
- **Python Binary**: `./.venv/bin/python` (`/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.venv/bin/python`)
- **Python Version**: 3.14.7 (Clang 22.1.3)
- **PyTorch**: 2.14.1+cu130
- **CUDA Hardware**:
  - `cuda:0`: NVIDIA GeForce RTX 3060 (12,288 MiB VRAM; currently 833 MiB used, 0% GPU load, 51°C).
  - Driver version: 595.99.02, CUDA runtime 13.2.
- **Dependencies Verified**: `torch`, `numpy`, `scipy`, `pandas`, `sklearn` (`scikit-learn`), `json`, `glob`, `pickle`.

### 1.6 Prototype 1-Epoch Evaluation Execution
A standalone harness prototype was constructed and executed locally:
- File: `.agents/teamwork/teamwork_explorer_r1_3/eval_harness_prototype.py`
- Command: `./.venv/bin/python .agents/teamwork/teamwork_explorer_r1_3/eval_harness_prototype.py`
- Output:
  ```
  Device: cuda:0
  Fold 1: 42 train batches (168 nights), 14 test batches (54 nights)
  Model A parameters: 163,940

  --- 1-Epoch Evaluation Results ---
  Train Loss: 1.3336 (took 1.20s)
  Eval Time: 0.11s, Windows: 32098
  Overall Accuracy: 45.27%
  Macro-F1: 0.3640
  Cohen's Kappa: 0.1631
  Per-class F1: {'Wake_f1': 0.2983, 'Light_f1': 0.5720, 'Deep_f1': 0.4331, 'REM_f1': 0.1524}
  ```
  Total execution time was 1.31 seconds!

---

## 2. Logic Chain

1. **Manifest Integrity (Step 1 $\to$ Observation 1.1)**:
   Inspection of `experiments/EXP-014/EXP014_MANIFESTS/fold_1.json` establishes the exact outer test split of 10 subjects (54 nights) and inner training split of 32 subjects (168 nights). Because subject IDs are disjoint between splits, Fold 1 provides zero data leakage for evaluating model generalizability.

2. **Input Dimensions and Tensor Contract (Step 2 $\to$ Observation 1.2 & 1.3)**:
   The dataset pipeline (`ControlledSleepDataset` + `load_night_features`) constructs 3D tensors:
   - `x_ihr`: `(B, T, 30)`
   - `x_accel`: `(B, T, 30)`
   - `x_meta`: `(B, T, 5)`
   - `targets`: `(B, T)`
   - `mask`: `(B, T)`
   Where $B=4$ and $T=600$. Any mutated architecture proposed by the Architect and implemented by the PyTorch Engineer MUST accept these exact dictionary keys and tensor dimensions in `forward(x_ihr, x_accel, x_meta, targets=targets, mask=mask, subject_idx=subject_idx)`.

3. **Hyperparameter Invariance (Step 3 $\to$ Observation 1.4 & Scope Constraints)**:
   The assignment and `ORIGINAL_REQUEST.md` mandate that batch size, optimizer, and learning rate must remain strictly unchanged:
   - Batch size = 4
   - Optimizer = `torch.optim.Adam`
   - Learning rate = `1e-4`
   - Loss function = CrossEntropy + 0.01 * RCCLoss
   Maintaining these constants isolates architectural mutations as the sole experimental variable.

4. **Macro-F1 Metric Standard (Step 4 $\to$ Observation 1.2 & 1.6)**:
   Sleep staging requires Macro-F1 across 4 stages (Wake=0, Light=1, Deep=2, REM=3).
   Because nights contain variable padding and unscored epochs (labeled `-100`), evaluations must filter using `valid = (targets >= 0) & (targets < 4) & mask` before calling `f1_score(y_true, y_pred, average="macro", zero_division=0)`. Per-class F1 scores verify that minority classes (Deep and REM) are tracked accurately alongside Wake and Light.

5. **Feasibility of Standalone Local Harness (Step 5 $\to$ Observation 1.5 & 1.6)**:
   Execution of `eval_harness_prototype.py` confirms that 1 epoch of training (42 batches, 168 nights) takes 1.20s and full outer test evaluation (14 batches, 54 nights, 32,098 windows) takes 0.11s on GPU `cuda:0`. Total execution is well within interactive budgets (~1.3s total) and uses ~850 MiB of VRAM, presenting zero risk of CUDA OOM or thermal saturation.

---

## 3. Caveats

1. **Epoch 1 Convergence**: A 1-epoch run produces an initial training state (Macro-F1 ~0.3640 on baseline Model A); standard full convergence in EXP-010/EXP-014 requires ~80 epochs (reaching Macro-F1 ~0.66–0.74). The 1-epoch harness is strictly designed for rapid functional, tensor-safety, gradient-flow, and directionality verification.
2. **Cache Dependency**: The harness requires the precomputed preprocessing cache at `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b`. This cache exists and is fully verified with 499 `.npz` files.
3. **Dual GPU Awareness**: While the host machine has two RTX 3060 GPUs (`cuda:0` and `cuda:1`), the task specifically mandates single GPU `cuda:0`. `torch.cuda.set_device(0)` or `to(torch.device("cuda:0"))` must be explicitly used.

---

## 4. Conclusion

The evaluation pipeline, Fold 1 manifest, data loading pipeline, and 1-epoch execution harness are fully mapped and empirically verified:
1. **Fold 1 Partitions**: Exactly 168 inner-train nights (32 subjects) and 54 outer-test nights (10 subjects).
2. **Tensor Interface**: Inputs must be shaped `(B, T, 30)` for `ihr` and `accel`, `(B, T, 5)` for `meta`, and `(B, T)` for `targets` and `mask`, with $B=4, T=600$.
3. **Hyperparameters**: Adam ($\text{lr}=10^{-4}$, weight_decay=0.0), batch size = 4, composite loss = CE + 0.01 * RCCLoss.
4. **Harness Delivery**: A clean, standalone, verified evaluation function `evaluate_1epoch_fold1(model, train_loader, test_loader, device)` is designed and tested. It takes any PyTorch `nn.Module` conforming to the Model A forward contract, runs 1 training epoch on Fold 1 train nights, runs outer-test evaluation on Fold 1 test nights, and reports Macro-F1 alongside per-class metrics in under 2 seconds on `cuda:0`.

---

## 5. Verification Method

To independently verify all findings and test the 1-epoch evaluation loop on single GPU `cuda:0`:

1. **Verify Python Environment and Hardware**:
   ```bash
   ./.venv/bin/python -c "
   import torch
   print('Device count:', torch.cuda.device_count())
   print('Device 0:', torch.cuda.get_device_name(0))
   assert torch.cuda.is_available()
   "
   ```
2. **Verify Fold 1 Manifest & Cache Integrity**:
   ```bash
   ./.venv/bin/python -c "
   import json, os, glob
   from slamss_ifs_train import load_night_features

   cache_dir = '/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b'
   with open('experiments/EXP-014/EXP014_MANIFESTS/fold_1.json') as f:
       m = json.load(f)
   train_subs, test_subs = m['inner_train_subjects'], m['outer_test_subjects']
   assert len(train_subs) == 32 and len(test_subs) == 10
   print('Fold 1 manifest verified: 32 train, 10 test subjects.')
   "
   ```
3. **Run Standalone 1-Epoch Evaluation Harness**:
   ```bash
   ./.venv/bin/python .agents/teamwork/teamwork_explorer_r1_3/eval_harness_prototype.py
   ```
   **Expected Result**:
   - Executes synchronously on `cuda:0` in <3 seconds.
   - Outputs: Train Loss ~1.33, Test Windows = 32,098, Overall Accuracy ~45.27%, Macro-F1 ~0.3640.
