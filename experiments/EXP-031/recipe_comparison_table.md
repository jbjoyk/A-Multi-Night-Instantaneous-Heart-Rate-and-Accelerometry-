# Recipe Comparison Table: Frozen Model A vs. Unified EXP-031/EXP-033 vs. Official SLAMSS-IFS

## Executive Summary & Script Provenance

This document tabulates every exact difference across:
1. **(a) Frozen Model A:** The training pipeline that produced the frozen checkpoints (`experiments/EXP-018/checkpoints/modelA_L1200_fold{fold}_seed{seed}.pt`), plus `exp013_models.py` for architecture.
   - **Training Script Provenance:** The 15 frozen `.pt` checkpoints were trained and serialized by `experiments/EXP-018/regenerate_modelA_checkpoints.py` (which mirrors the Phase 2 training loop in `experiments/EXP-018/run_exp018_execution.py`).
   - **Status of Standalone Stage 2 / Epoch Script:** If referring to a separate standalone training script in `experiments/STAGE2_FINAL_MODEL/` (e.g. `train_stage2_multitask.py`), or an epoch-based training script for Model A, that script is **NOT FOUND**. Stage 2 (`experiments/STAGE2_FINAL_MODEL/freeze_final_model.py`) is explicitly a read-only verification harness (zero retraining; lines 4-20). Model A was trained on a step-budget basis (exactly 3,200 optimizer steps via an infinite batch stream) rather than an epoch loop.
2. **(b) Unified EXP-031 and EXP-033 Recipe (`W100`, `p1_100k`, `P190`, `full_v5`):** 
   - `experiments/EXP-031/run_exp031_width_sweep.py`
   - `experiments/EXP-031/run_w100_retrain.py`
   - `experiments/EXP-030/run_exp030_cv.py`
   - `experiments/EXP-029/run_exp029_cv.py`
3. **(c) Official SLAMSS-IFS:**
   - EXP-021 Matched Benchmark: `experiments/EXP-021/run_tsk08_slamss_matched.py`
   - Official Repository Defaults: `experiments/EXP-005/official_slamss_ifs/SLAMSS_IFS_Training_Example.ipynb` & `SLAMSS_IFS_Testing.ipynb`

---

## 1. Hyperparameter Comparison Table

| Dimension | (a) Frozen Model A (`EXP-018` / Stage 2 Freeze) | (b) Unified `EXP-031` / `EXP-033` Recipe (`W100`, `p1_100k`, `P190`, `full_v5`) | (c) Official SLAMSS-IFS (`EXP-021` Matched Benchmark) | (c) Official SLAMSS-IFS (Official Repo Defaults) |
|---|---|---|---|---|
| **Optimizer** | `torch.optim.Adam(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)` (`regenerate_modelA_checkpoints.py:123-128`, `run_exp018_execution.py:439-444`) | `torch.optim.Adam(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)` (`run_exp031_width_sweep.py:147-149`, `run_w100_retrain.py:122-124`) | `torch.optim.Adam(lr=1e-4, betas=(0.9, 0.999))` (`run_tsk08_slamss_matched.py:306`) | Two separate Adam optimizers: `torch.optim.Adam(model.parameters(), lr=0.0001)` and `torch.optim.Adam(cnnmodel.parameters(), lr=0.0001)` (`SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 10-11`) |
| **Learning Rate** | `1e-4` (`0.0001`) (`regenerate_modelA_checkpoints.py:125`, `run_exp018_execution.py:441`) | `1e-4` (`0.0001`) (`run_exp031_width_sweep.py:375`, `run_w100_retrain.py:334`) | `1e-4` (`0.0001`) (`run_tsk08_slamss_matched.py:666`) | `0.0001` (`1e-4`) (`SLAMSS_IFS_Training_Example.ipynb:cell 15:line 10`) |
| **Learning Rate Schedule** | None (constant learning rate) (`regenerate_modelA_checkpoints.py:146-178`) | None (constant learning rate) (`run_exp031_width_sweep.py:162-229`, `run_w100_retrain.py:133-200`) | None (constant learning rate) (`run_tsk08_slamss_matched.py:373-490`) | None (constant learning rate reset each epoch) (`SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 10-11`) |
| **Weight Decay** | `0.0` (`regenerate_modelA_checkpoints.py:126`, `run_exp018_execution.py:442`) | `0.0` (`run_exp031_width_sweep.py:148`, `run_w100_retrain.py:123`) | `0.0` (PyTorch Adam default) (`run_tsk08_slamss_matched.py:306`) | `0.0` (PyTorch Adam default) (`SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 10-11`) |
| **Train Batch Size** | $B=2$ trajectories of $L=1200$ ($N_{\text{accum}}=1$, $2,400\text{ tokens/step}$) (`regenerate_modelA_checkpoints.py:62-65`, `exp018_dataset.py:21`) | micro-batch $B=4$ nights (effective $B=4$) (`run_exp031_width_sweep.py:135,373`, `run_w100_retrain.py:110,332`) | micro-batch $B=4$ (effective $B=4$, `accum_steps=1`) (`run_tsk08_slamss_matched.py:282,663-664`) | `BATCH_SIZE = 10` sequences (`SLAMSS_IFS_Training_Example.ipynb:cell 2:line 1`, `cell 7:line 3`) |
| **Validation Batch Size** | $B=1$ sequence (window-by-window in `SlidingWindowPredictor`) (`regenerate_modelA_checkpoints.py:85-91`, `exp018_inference.py:59-60`) | $B=16$ nights (batched sliding window) (`run_exp031_width_sweep.py:374`, `run_w100_retrain.py:333`) | $B=16$ nights (batched sliding window) (`run_tsk08_slamss_matched.py:665`) | `BATCH_SIZEV2 = 32` nights (val) / `BATCH_SIZEV = 13` (train eval) (`SLAMSS_IFS_Training_Example.ipynb:cell 2:lines 2-3`, `cell 7:lines 10,17`) |
| **Loss & Regularization** | $\text{CE} + 0.01 \times \text{RCCLoss}$ (`use_rcc_loss=True`) (`regenerate_modelA_checkpoints.py:79,117`, `exp013_models.py:270-278`) | $\text{CE} + 0.01 \times \text{RCCLoss}$ (`loss_type='rccloss'`) (`run_exp031_width_sweep.py:151,367`, `run_w100_retrain.py:151`) | $\text{CE} + 0.01 \times \text{RCCLoss}$ (`OfficialSLAMSSCompositeLoss`, `rcc_weight=0.01`) (`run_tsk08_slamss_matched.py:304`) | $\text{CE} + 0.01 \times \text{RCCLoss}$ (`total_loss_CE + 0.01 * total_loss_RWL`) (`SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 59-65`) |
| **Precision / Autocast** | FP32 standard precision (no AMP/autocast) (`regenerate_modelA_checkpoints.py:146-178`) | Mixed precision AMP (`torch.amp.autocast`, `GradScaler` enabled) (`run_exp031_width_sweep.py:150,177,380`, `run_w100_retrain.py:125,339`) | Mixed precision AMP (`torch.amp.autocast`, `GradScaler` enabled) (`run_tsk08_slamss_matched.py:307,389,670`) | FP32 standard precision (no AMP/autocast) (`SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 25-73`) |
| **Max Epochs** | Not epoch-based; exactly 3,200 optimizer steps (`regenerate_modelA_checkpoints.py:57`, `run_exp018_execution.py:403`) | 100 epochs (`--max-epochs 100`) (`run_exp031_width_sweep.py:377`, `run_w100_retrain.py:336`) | 100 epochs (`--max-epochs 100`) (`run_tsk08_slamss_matched.py:667`) | 1,000 epochs (`EPOCH = 1000`) (`SLAMSS_IFS_Training_Example.ipynb:cell 2:line 7`) |
| **Min Epochs** | None (not epoch-based) (`regenerate_modelA_checkpoints.py:146`) | 40 epochs (`--min-epochs 40`) (`run_exp031_width_sweep.py:378`, `run_w100_retrain.py:337`) | 40 epochs (`--min-epochs 40`) (`run_tsk08_slamss_matched.py:668`) | None (0) (`SLAMSS_IFS_Training_Example.ipynb:cell 15:line 5`) |
| **Patience** | None (no early stopping) (`regenerate_modelA_checkpoints.py:146`) | 25 epochs (`--patience 25`) (`run_exp031_width_sweep.py:379`, `run_w100_retrain.py:338`) | 25 epochs (`--patience 25`) (`run_tsk08_slamss_matched.py:669`) | None (no early stopping) (`SLAMSS_IFS_Training_Example.ipynb:cell 15`) |
| **Early Stopping Metric** | None (fixed step budget) (`regenerate_modelA_checkpoints.py:146`) | Inner-validation Macro-F1 (`val_macro_f1`) (`run_exp031_width_sweep.py:186,210`, `run_w100_retrain.py:181`) | Inner-validation Macro-F1 (`val_macro_f1`) (`run_tsk08_slamss_matched.py:442-452`) | None (tracks train/val accuracy, no early stopping) (`SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 147,222`) |
| **Checkpoint Selection Rule** | Final step (step 3,200) weights saved directly (`regenerate_modelA_checkpoints.py:197`, `run_exp018_execution.py:467-501`) | Best inner-validation epoch checkpoint reloaded for evaluation (`run_exp031_width_sweep.py:214-216,237`, `run_w100_retrain.py:209-216`) | Best inner-validation epoch checkpoint reloaded for evaluation (`run_tsk08_slamss_matched.py:460-466,504-508`) | Model saving commented out (`#torch.save`) or manual epoch loading (`SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 234-237`, `SLAMSS_IFS_Testing.ipynb:cell 10`) |
| **Completed Extent** | Exactly 3,200 steps across all 15 runs (mean 3200, range [3200, 3200]; epochs NOT RECORDED) (`run_L1200_fold*.json`, `final_model_spec.json:81`) | Cross-checked against saved run JSONs: `W100` mean 96.3 ep [65, 100]; `p1_100k` mean 97.8 ep [81, 100]; `P190` mean 98.1 ep [90, 100]; `full_v5` mean 89.2 ep [48, 100]; Pooled ($n=60$) mean 95.4 ep [48, 100] | Completed epochs (15 runs): mean 65.4 ep [64, 81]; best epoch: mean 33.9 [24, 56]; steps: mean 2812.2 [2688, 3483] (`slamss_ifs_fold*.json`) | 1,000 epochs (mean 1000, range [1000, 1000]; steps NOT RECORDED) (`SLAMSS_IFS_Training_Example.ipynb:cell 2:line 7`) |
| **Validation Subjects Selection** | `inner_val_subjects` from `fold_{fold}.json` / `fold_manifests.csv` (8–10 subjects/fold) (`regenerate_modelA_checkpoints.py:8-10`, `fold_manifests.csv:1-48`) | `inner_val_subjects` from `fold_manifests.csv` (8–10 subjects/fold) (`run_exp031_width_sweep.py:117`, `run_w100_retrain.py:92`) | `inner_val_subjects` from `fold_manifests.csv` (8–10 subjects/fold) (`run_tsk08_slamss_matched.py:270`) | Directory glob partition (`/media/godzilla/database/Data8/CVdata/V3/*.mat` for val/test, `T3` for train) (`SLAMSS_IFS_Training_Example.ipynb:cell 5:lines 14-20`) |
| **Gradient Clipping** | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` (`regenerate_modelA_checkpoints.py:170`, `run_exp018_execution.py:491`) | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` via `--grad-clip 1.0` (`run_exp031_width_sweep.py:185,376`, `run_w100_retrain.py:156,335`) | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` (`run_tsk08_slamss_matched.py:408`) | None (no gradient clipping) (`SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 67-73`) |
| **Augmentation** | Random temporal crop within night recording (`RandomCropSleepDataset`, start offset $s \sim \text{Uniform}(0, N - L)$) (`regenerate_modelA_checkpoints.py:130-136`, `exp018_dataset.py:41-83`) | Random temporal crop within night recording (`RandomCropSleepDataset`, start offset $s \sim \text{Uniform}(0, N - L)$) (`run_exp031_width_sweep.py:124-130`, `exp018_dataset.py:41-83`) | Random temporal crop within night recording (`SLAMSSMatchedDataset`, `random_crop=True`) (`run_tsk08_slamss_matched.py:313`) | Slices up to 1200 epochs from recording ($x[:, :sqe \times 30, :]$), no random temporal cropping jitter (`SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 25-27`) |
| **Input Normalization & Circadian Neutralization** | Circadian neutralized (`neutralize_circadian=True`, `time_feat` channel index 2 set to 0.0); per-epoch HR mean/std in meta channels 3/4 (`regenerate_modelA_checkpoints.py:134`, `exp018_dataset.py:49-50`) | Circadian neutralized (`neutralize_circadian=True`, `time_feat` channel index 2 set to 0.0); per-epoch HR mean/std in meta channels 3/4 (`run_exp031_width_sweep.py:128,195`, `run_w100_retrain.py:103`) | Circadian neutralized (`neutralize_circadian=True`, elapsed time set to 0.0) (`run_tsk08_slamss_matched.py:277,672-678`) | Raw elapsed time feature included (circadian NOT neutralized); standard scaling (`SLAMSS_IFS_Training_Example.ipynb:cell 15:line 26`) |
| **Evaluation Predictor** | `SlidingWindowPredictor` (strided sliding window $W=1200$, $S=600$, $m=300$, boundary window anchoring, probability averaging) (`regenerate_modelA_checkpoints.py:85-90`, `exp018_inference.py:23-145`) | `LogitStitchingSlidingWindowPredictor` (strided sliding window with logit stitching / probability averaging) (`run_exp031_width_sweep.py:192-197,219-224`, `run_w100_retrain.py:219-224`) | `SLAMSSSlidingWindowPredictor` (batched full-night sliding window with $S=600$, $m=300$, boundary window anchoring, probability averaging) (`run_tsk08_slamss_matched.py:276,506`) | Fixed-segment evaluation: validation evaluates first 1200 epochs without sliding window or margin discarding (`SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 177-220`, `SLAMSS_IFS_Testing.ipynb:cell 10`) |
| **Context Length** | $L = 1200$ epochs ($600\text{ min} / 10\text{ h}$) (`regenerate_modelA_checkpoints.py:61`, `run_exp018_execution.py:57`) | $L = 1200$ epochs ($600\text{ min} / 10\text{ h}$) (`run_exp031_width_sweep.py:372`, `run_w100_retrain.py:331`) | $L = 1200$ epochs ($600\text{ min} / 10\text{ h}$) (`run_tsk08_slamss_matched.py:662`) | $sqe = 1200$ epochs ($600\text{ min} / 10\text{ h}$; train eval $sqe800 = 600$ epochs) (`SLAMSS_IFS_Training_Example.ipynb:cell 2:line 9`, `cell 15:lines 25,98,177`) |
| **Seeds & Folds** | 3 seeds (`42, 123, 2024`) $\times$ 5 folds = 15 checkpoints (`regenerate_modelA_checkpoints.py:7,214-215`, `final_model_spec.json:72-73`) | 3 seeds (`42, 123, 2024`) $\times$ 5 folds = 15 checkpoints per model (`run_exp031_width_sweep.py:369-370`, `run_w100_retrain.py:328-329`) | 3 seeds (`42, 123, 2024`) $\times$ 5 folds = 15 checkpoints (`run_tsk08_slamss_matched.py:655-658`) | Single unseeded run on 1 split (`SLAMSS_IFS_Training_Example.ipynb:cells 1-15`) |
| **Parameter Count** | 163,940 parameters (`regenerate_modelA_checkpoints.py:121`, `exp013_models.py:120`, `claim_registry.yaml:45`) | `W100`: 163,940; `p1_100k`: 98,940; `P190`: 190,940; `full_v5`: 190,788 (`run_w100_retrain.py:120`, `exp031_models.py:31-45`) | 5,465,768 parameters (`run_tsk08_slamss_matched.py:303`, `claim_registry.yaml:49`) | 5,465,768 parameters (`SLAMSS_IFS_Model.py`, `official_slamss_ifs_hw_profile.json:4`) |

---

## 2. Quoted Code Lines by Dimension

### Dimension 1: Optimizer
- **Frozen Model A:** `experiments/EXP-018/regenerate_modelA_checkpoints.py:123-128`
  ```python
  optimizer = torch.optim.Adam(
      model.parameters(),
      lr=1e-4,
      weight_decay=0.0,
      betas=(0.9, 0.999)
  )
  ```
- **Unified EXP-031/EXP-033:** `experiments/EXP-031/run_exp031_width_sweep.py:147-149`
  ```python
  optimizer = torch.optim.Adam(
      model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0
  )
  ```
- **Official SLAMSS-IFS Matched:** `experiments/EXP-021/run_tsk08_slamss_matched.py:306`
  ```python
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
  ```
- **Official SLAMSS-IFS Repo Default:** `experiments/EXP-005/official_slamss_ifs/SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 10-11`
  ```python
  optimizer = torch.optim.Adam(model.parameters(), lr=LR)
  optimizer2 = torch.optim.Adam(cnnmodel.parameters(), lr=LR)
  ```

---

### Dimension 2: Learning Rate & Schedule
- **Frozen Model A:** `experiments/EXP-018/regenerate_modelA_checkpoints.py:125, 146-178`
  - LR: `lr=1e-4`
  - Schedule: Constant learning rate across all 3,200 steps; no `lr_scheduler` call.
- **Unified EXP-031/EXP-033:** `experiments/EXP-031/run_exp031_width_sweep.py:375, 162-229`
  - LR: `parser.add_argument("--lr", type=float, default=1e-4)`
  - Schedule: Constant learning rate across all epochs; no `lr_scheduler` call.
- **Official SLAMSS-IFS Matched:** `experiments/EXP-021/run_tsk08_slamss_matched.py:666, 373-490`
  - LR: `parser.add_argument("--lr", type=float, default=1e-4)`
  - Schedule: Constant learning rate across all epochs.
- **Official SLAMSS-IFS Repo Default:** `experiments/EXP-005/official_slamss_ifs/SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 10-11`
  - LR: `LR = 0.0001` (`1e-4`), re-instantiated at the top of each epoch loop with no decay schedule.

---

### Dimension 3: Batch Sizes & Gradient Accumulation
- **Frozen Model A:** `experiments/EXP-018/regenerate_modelA_checkpoints.py:62-65`, `exp018_dataset.py:21`
  ```python
  context_len = 1200
  sched = COMPUTE_SCHEDULE_LOOKUP[context_len]
  B = sched["B"]          # B = 2
  N_accum = sched["N_accum"] # N_accum = 1
  ```
  Validation evaluates 1 sequence at a time in `SlidingWindowPredictor` (`exp018_inference.py:59-60`).
- **Unified EXP-031/EXP-033:** `experiments/EXP-031/run_exp031_width_sweep.py:135, 373-374`
  ```python
  parser.add_argument("--micro-batch-size", type=int, default=4)
  parser.add_argument("--val-batch-size", type=int, default=16)
  ```
- **Official SLAMSS-IFS Matched:** `experiments/EXP-021/run_tsk08_slamss_matched.py:282, 663-665`
  ```python
  parser.add_argument("--micro-batch-size", type=int, default=4)
  parser.add_argument("--effective-batch-size", type=int, default=4)
  parser.add_argument("--val-batch-size", type=int, default=16)
  ```
- **Official SLAMSS-IFS Repo Default:** `experiments/EXP-005/official_slamss_ifs/SLAMSS_IFS_Training_Example.ipynb:cell 2:lines 1-3`
  ```python
  BATCH_SIZE = 10#100
  BATCH_SIZEV = 13#560
  BATCH_SIZEV2 = 32
  ```

---

### Dimension 4: Loss, Regularization & Precision (AMP)
- **Frozen Model A:** `experiments/EXP-018/regenerate_modelA_checkpoints.py:111-118`, `exp013_models.py:270-278`
  ```python
  model = ConventionalCNNBiGRU(
      d_model=64,
      d_temp=96,
      n_classes=4,
      n_meta=5,
      dropout=0.2,
      use_rcc_loss=True
  ).to(device)
  ```
  Precision: Pure FP32 standard precision (no autocast).
- **Unified EXP-031/EXP-033:** `experiments/EXP-031/run_exp031_width_sweep.py:150, 177-187, 367, 380`
  ```python
  parser.add_argument("--loss-type", type=str, default="rccloss", choices=["rccloss", "ce"])
  parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
  ...
  scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
  ...
  with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
      out = model(x_ihr=x_ihr, x_accel=x_acc, x_meta=x_meta, seq_lens=seq_len)
  loss = compute_stage_loss_fp32(model, out["logits"], y, mask)
  ```
- **Official SLAMSS-IFS Matched:** `experiments/EXP-021/run_tsk08_slamss_matched.py:304-307, 389`
  ```python
  loss_fn = OfficialSLAMSSCompositeLoss(rcc_weight=0.01, n_classes=4).to(device)
  scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
  ...
  with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
      logits = model(x_ihr, x_acc, x_meta, seq_lens=seq_lens)
  ```
- **Official SLAMSS-IFS Repo Default:** `experiments/EXP-005/official_slamss_ifs/SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 59-65`
  ```python
  loss = loss_func_CE(outputs.contiguous().view(-1,cls), t_y2)
  total_loss_CE = torch.sum(loss * t_mask)
  loss = loss_func(outputs.contiguous().view(-1,cls), t_y2)
  total_loss_RWL = -torch.sum(loss * t_mask2)
  total_loss = total_loss_CE + 0.01 * total_loss_RWL
  ```
  Precision: Pure FP32 standard precision (no AMP).

---

### Dimension 5: Epoch Limits, Patience & Checkpoint Selection
- **Frozen Model A:** `experiments/EXP-018/regenerate_modelA_checkpoints.py:57, 146, 197`
  - Not epoch-based. Fixed step budget: `max_steps = 3200`.
  - Min epochs: None.
  - Patience: None (no early stopping).
  - Selection: Final step weights saved directly via `torch.save(model.state_dict(), checkpoint_path)`.
- **Unified EXP-031/EXP-033:** `experiments/EXP-031/run_exp031_width_sweep.py:210-216, 226-228, 237, 377-379`
  ```python
  parser.add_argument("--max-epochs", type=int, default=100)
  parser.add_argument("--min-epochs", type=int, default=40)
  parser.add_argument("--patience", type=int, default=25)
  ...
  if val_f1 > best_val_f1 + 1e-6:
      best_val_f1 = val_f1
      best_epoch = epoch
      epochs_no_improve = 0
      torch.save(model.state_dict(), tmp_ckpt)
      os.replace(tmp_ckpt, ckpt_path)
  else:
      epochs_no_improve += 1
  ...
  if epoch >= min_epochs and epochs_no_improve >= patience:
      break
  ...
  model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
  ```
- **Official SLAMSS-IFS Matched:** `experiments/EXP-021/run_tsk08_slamss_matched.py:460-466, 480-485, 504-508, 667-669`
  - Max epochs = 100, min epochs = 40, patience = 25.
  - Selection: Best inner-validation Macro-F1 epoch reloaded before evaluation.
- **Official SLAMSS-IFS Repo Default:** `experiments/EXP-005/official_slamss_ifs/SLAMSS_IFS_Training_Example.ipynb:cell 2:line 7, cell 15:lines 234-237`
  - Max epochs = 1,000 (`EPOCH = 1000`).
  - Min epochs: None.
  - Patience: None (runs all 1,000 epochs).
  - Checkpoint saving is commented out in loop (`#torch.save(...)`).

---

### Dimension 6: Actual Completed Extent (Cross-Checked Against Logs)
Cross-checking across all 60 runs of the unified recipe in `experiments/EXP-031/runs/`, `experiments/EXP-030/run_results/`, and `experiments/EXP-029/run_results/`:
- **Unified Recipe (`W100`, 15 runs):**
  - Completed epochs: Mean **96.33** (range [65, 100]). 13 of 15 runs completed all 100 epochs; 2 early-stopped at epochs 65 and 80.
  - Best epoch: Mean **81.47** (range [23, 100]).
- **Unified Recipe (`p1_100k`, 15 runs):**
  - Completed epochs: Mean **97.80** (range [81, 100]). 13 of 15 runs completed 100 epochs; 2 early-stopped at epochs 81 and 86.
  - Best epoch: Mean **86.00** (range [56, 100]).
- **Unified Recipe (`P190`, 15 runs):**
  - Completed epochs: Mean **98.13** (range [90, 100]). 13 of 15 runs completed 100 epochs; 2 early-stopped at epochs 90 and 92.
  - Best epoch: Mean **85.33** (range [58, 100]).
- **Unified Recipe (`full_v5`, 15 runs):**
  - Completed epochs: Mean **89.20** (range [48, 100]). 9 of 15 runs completed 100 epochs; 6 early-stopped (epochs 48, 59, 75, 78, 88, 90).
  - Best epoch: Mean **70.33** (range [23, 100]).
- **Unified Recipe Pooled ($n=60$ runs):**
  - Completed epochs: Mean **95.37** (range [48, 100]).
  - Best epoch: Mean **80.78** (range [23, 100]).
- **Official SLAMSS-IFS Matched (`run_tsk08_slamss_matched.py`, 15 runs):**
  - Completed epochs: Mean **65.40** (range [64, 81]). All 15 runs early-stopped between epochs 64 and 81 (min-epochs 40 + patience 25).
  - Best epoch: Mean **33.93** (range [24, 56]).
  - Total optimizer steps: Mean **2,812.20** (range [2,688, 3,483]).
- **Frozen Model A (`regenerate_modelA_checkpoints.py` / `run_exp018_execution.py`, 15 runs):**
  - Completed steps: Exactly **3,200** optimizer steps in 15/15 runs (mean 3200.0, range [3200, 3200]; epochs NOT RECORDED; equivalent to ~37.6 passes through dataset).

---

### Dimension 7: Gradient Clipping, Augmentation & Input Normalization
- **Frozen Model A:**
  - Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` (`regenerate_modelA_checkpoints.py:170`).
  - Augmentation: `RandomCropSleepDataset(..., context_len=1200, seed=seed)` with start offset $s \sim \text{Uniform}(0, N - L)$ (`exp018_dataset.py:41-83`).
  - Normalization & Circadian: `neutralize_circadian=True` forces channel 2 (`time_feat`) to 0.0 (`exp018_dataset.py:49-50`).
- **Unified EXP-031/EXP-033:**
  - Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)` (`run_exp031_width_sweep.py:185, 376`).
  - Augmentation: `RandomCropSleepDataset(..., context_len=args.context_len)` (`run_exp031_width_sweep.py:124-130`).
  - Normalization & Circadian: `neutralize_circadian=True` (`run_exp031_width_sweep.py:128, 195`).
- **Official SLAMSS-IFS Matched:**
  - Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` (`run_tsk08_slamss_matched.py:408`).
  - Augmentation: `SLAMSSMatchedDataset(..., random_crop=True)` (`run_tsk08_slamss_matched.py:313`).
  - Normalization & Circadian: `neutralize_circadian=True` (`run_tsk08_slamss_matched.py:277`).
- **Official SLAMSS-IFS Repo Default:**
  - Gradient clipping: None (`SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 67-73`).
  - Augmentation: Slices fixed 1200-epoch window from start of recording, no random temporal jitter (`SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 25-27`).
  - Normalization & Circadian: Raw time feature included unneutralized (`SLAMSS_IFS_Training_Example.ipynb:cell 15:line 26`).

---

### Dimension 8: Evaluation Predictor, Context Length, Seeds & Parameters
- **Frozen Model A:**
  - Predictor: `SlidingWindowPredictor(context_len=1200, stride=600, margin=300)` (`exp018_inference.py:23-145`).
  - Context length: $L = 1200$ epochs ($600\text{ min} / 10\text{ h}$).
  - Seeds: `[42, 123, 2024]` across 5 folds = 15 runs.
  - Parameter count: `163,940` parameters (`exp013_models.py:120`).
- **Unified EXP-031/EXP-033:**
  - Predictor: `LogitStitchingSlidingWindowPredictor(context_len=1200, stride=600, margin=300)` (`run_exp031_width_sweep.py:192-197, 219-224`).
  - Context length: $L = 1200$ epochs ($600\text{ min} / 10\text{ h}$).
  - Seeds: `[42, 123, 2024]` across 5 folds = 15 runs per model.
  - Parameter count: `W100`: `163,940`; `p1_100k`: `98,940`; `P190`: `190,940`; `full_v5`: `190,788`.
- **Official SLAMSS-IFS Matched:**
  - Predictor: `SLAMSSSlidingWindowPredictor(context_len=1200, stride=600, margin=300)` (`run_tsk08_slamss_matched.py:276, 506`).
  - Context length: $L = 1200$ epochs ($600\text{ min} / 10\text{ h}$).
  - Seeds: `[42, 123, 2024]` across 5 folds = 15 runs.
  - Parameter count: `5,465,768` parameters (`run_tsk08_slamss_matched.py:303`).
- **Official SLAMSS-IFS Repo Default:**
  - Predictor: Truncated 1200-epoch evaluation without sliding window or margin discarding (`SLAMSS_IFS_Training_Example.ipynb:cell 15:lines 177-220`).
  - Context length: $sqe = 1200$ epochs (train eval $sqe800 = 600$ epochs).
  - Seeds: Single unseeded run.
  - Parameter count: `5,465,768` parameters (`SLAMSS_IFS_Model.py`).

---

## 3. Exhaustive List of Differences Found (Unranked)

1. **Training loop paradigm (step stream vs. epoch DataLoader):** Frozen Model A sampled batches from an infinite generator for a fixed 3,200 optimizer steps with no concept of epochs; unified EXP-031/EXP-033 and EXP-021 matched SLAMSS-IFS used epoch-based DataLoaders with a 40-epoch floor and 100 max epochs; official SLAMSS-IFS repo defaults trained for a fixed 1,000 epochs.
2. **Early stopping policy:** Frozen Model A and official SLAMSS-IFS repo defaults had no early stopping; unified EXP-031/EXP-033 and EXP-021 matched SLAMSS-IFS implemented early stopping with patience = 25 epochs on inner-validation Macro-F1 after the 40-epoch floor.
3. **Checkpoint selection rule:** Frozen Model A evaluated the model at its final step (step 3,200); official SLAMSS-IFS repo defaults had model saving commented out or manually loaded epoch weights; unified EXP-031/EXP-033 and EXP-021 matched SLAMSS-IFS saved and restored the checkpoint from the best inner-validation Macro-F1 epoch.
4. **Precision and mixed-precision execution:** Frozen Model A and official SLAMSS-IFS repo defaults ran strictly in FP32; unified EXP-031/EXP-033 and EXP-021 matched SLAMSS-IFS ran with PyTorch AMP (FP16 autocast forward pass, `GradScaler`, and FP32 loss computation).
5. **Training batch size:** Frozen Model A trained on $B=2$ trajectories of length 1200; unified EXP-031/EXP-033 and EXP-021 matched SLAMSS-IFS used micro-batch $B=4$ (effective $B=4$); official SLAMSS-IFS repo defaults used $B=10$.
6. **Validation batch size:** Frozen Model A evaluated window-by-window with batch size 1 in `SlidingWindowPredictor`; unified EXP-031/EXP-033 used batch size 16 in batched sliding-window evaluation; EXP-021 matched SLAMSS-IFS used batched sliding-window evaluation with batch size 16; official SLAMSS-IFS repo defaults used `BATCH_SIZEV2 = 32`.
7. **Completed training extent:** Frozen Model A completed exactly 3,200 steps (epochs NOT RECORDED); unified EXP-031/EXP-033 completed mean 95.37 epochs (range 48–100, steps NOT RECORDED); EXP-021 matched SLAMSS-IFS completed mean 65.40 epochs (range 64–81) and mean 2812.20 steps (range 2688–3483); official SLAMSS-IFS repo defaults ran for 1,000 epochs.
8. **Gradient clipping:** Frozen Model A, unified EXP-031/EXP-033, and EXP-021 matched SLAMSS-IFS used gradient clipping at `max_norm=1.0`; official SLAMSS-IFS repo defaults did not apply gradient clipping.
9. **Circadian neutralization:** Frozen Model A, unified EXP-031/EXP-033, and EXP-021 matched SLAMSS-IFS explicitly set `time_feat` (channel index 2) to 0.0 to prevent circular circadian leakage; official SLAMSS-IFS repo defaults retained the raw time feature unneutralized.
10. **Inference engine:** Frozen Model A used unstitched `SlidingWindowPredictor`; unified EXP-031/EXP-033 used `LogitStitchingSlidingWindowPredictor`; EXP-021 matched SLAMSS-IFS used `SLAMSSSlidingWindowPredictor`; official SLAMSS-IFS repo defaults performed a single 1200-epoch truncated pass without sliding-window stride or margin discarding.
11. **Data augmentation:** Frozen Model A, unified EXP-031/EXP-033, and EXP-021 matched SLAMSS-IFS used uniform random temporal cropping start offsets $s \sim \text{Uniform}(0, N - L)$; official SLAMSS-IFS repo defaults sliced the first 1200 epochs deterministically with no temporal jitter.
12. **Validation cohort selection:** Frozen Model A, unified EXP-031/EXP-033, and EXP-021 matched SLAMSS-IFS used stratified 5-fold cross-validation with inner validation sets (8–10 subjects) and strictly held-out outer test subjects; official SLAMSS-IFS repo defaults used an unstratified directory glob partition.
13. **Optimizer instance count:** Frozen Model A, unified EXP-031/EXP-033, and EXP-021 matched SLAMSS-IFS passed all parameters into a single Adam optimizer; official SLAMSS-IFS repo defaults instantiated two separate Adam optimizers for the sequence-to-sequence model and the CNN feature extractor.
14. **Deterministic seeding:** Frozen Model A, unified EXP-031/EXP-033, and EXP-021 matched SLAMSS-IFS fixed random seeds across PyTorch and NumPy for seeds `[42, 123, 2024]`; official SLAMSS-IFS repo defaults set no seeds.
15. **Parameter count:** Frozen Model A has 163,940 parameters; unified EXP-031/EXP-033 architectures range from 98,940 (`p1_100k`) to 190,940 (`P190`) parameters; official SLAMSS-IFS has 5,465,768 parameters.
