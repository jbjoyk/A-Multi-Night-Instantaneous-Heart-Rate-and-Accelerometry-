# PRE-REGISTRATION (PROPOSAL): EXP-034, 500-Epoch Convergence Audit on Compact CNN-BiGRU Widths

- **Status:** **EGM Review Resolved (Ready to Lock; NO-CODE / NO-TRAINING)** — Commit this file to lock the pre-registration hash before writing `experiments/EXP-034/` driver code or launching GPU runs.
- **Experiment ID:** `EXP-034` (`experiments/EXP-034/`).
- **Strict Isolation Rule:** All `EXP-018` through `EXP-033` artifacts, checkpoints, prediction caches, locked preregistrations, and summary CSVs/JSONs remain strictly **read-only**. The new `EXP-034` driver (`experiments/EXP-034/run_exp034_convergence.py`) shall hardcode `OUTPUT_DIR_DEFAULT = EXP034_DIR` (`experiments/EXP-034/`) so all new scripts, checkpoints, logs, and outputs live exclusively under `experiments/EXP-034/`.

---

## 1. Purpose & Rationale

In `EXP-030` and `EXP-031` (`--max-epochs 100`, `--min-epochs 40`, `--patience 25`), compact plain `ConventionalCNNBiGRU` models trained under the unified recipe exhibited a capacity plateau above `~94k` parameters:
- `W050` (`43,108` params): `65.61%` OOF Macro-F1 (`15/15` runs reached epoch 100)
- `W075` (`93,796` params): `66.26%` OOF Macro-F1 (mean completed epochs `98.5`)
- `p1_100k` (`98,940` params): `66.67%` OOF Macro-F1 (`13/15` runs hit epoch 100; best epoch mean `86.0`)
- `W100` (`163,940` params): `66.68%` OOF Macro-F1 (`13/15` runs hit epoch 100; best epoch mean `81.5`)

Because `13/15` runs for both `W100` and `p1_100k` terminated at the `--max-epochs 100` ceiling rather than via `patience=25` early stopping, the observed `~94k+` plateau is **conditional on the 100-epoch training cap**. `EXP-034` tests whether extending `--max-epochs` from `100` to `500` (with all other hyperparameters, manifests, and evaluation logic held identical) unlocks further Macro-F1 gains.

---

## 2. Experimental Design: Single Factor, Arms, & Driver Dispatch

- **Single Factor:** Maximum epoch budget (`--max-epochs 500` vs. locked `--max-epochs 100` baselines).
- **Design Matrix:** `4 arms × 5 folds × 3 seeds = 60 runs`.
- **Dedicated Driver:** Because `validate_trainable_arm()` in `experiments/EXP-031/run_exp031_width_sweep.py:72-79` blocks training `"A"` and rejects `"p1_100k"` / `*_500ep` arm names, `EXP-034` shall use a dedicated driver `experiments/EXP-034/run_exp034_convergence.py` importing read-only helpers from `EXP-018`/`EXP-029`/`EXP-030`/`EXP-031` and dispatching:
  - `W050_500ep`, `W075_500ep`, `W100_500ep` via `build_exp031_model(arm=..., loss_type="rccloss")` (`"W050"`, `"W075"`, `"A"`),
  - `p1_100k_500ep` via `build_exp030_model(grid_name="p1_100k", loss_type="rccloss")`.

| Arm ID | Source Factory & Key | Exact Parameter Count | `d_enc` / `d_temp` / `head_hidden` | Canonical 100-Epoch Baseline CSV |
|---|---|---:|---|---|
| `W050_500ep` | `EXP031_ARMS["W050"]` (`experiments/EXP-031/exp031_models.py:61-71`) | `43,108` | `32 / 48 / 32` | `experiments/EXP-031/W050_subject_metrics.csv` |
| `W075_500ep` | `EXP031_ARMS["W075"]` (`experiments/EXP-031/exp031_models.py:72-81`) | `93,796` | `48 / 72 / 48` | `experiments/EXP-031/W075_subject_metrics.csv` |
| `p1_100k_500ep` | `EXP030_GRID["p1_100k"]` (`experiments/EXP-030/exp030_models.py:42`) | `98,940` | `48 / 76 / 48` | `experiments/EXP-030/p1_100k_subject_metrics.csv` (`grid_tag` col) |
| `W100_500ep` | `EXP031_ARMS["A"]` (`experiments/EXP-031/exp031_models.py:83-93`) | `163,940` | `64 / 96 / 64` | `experiments/EXP-031/runs/W100/W100_subject_metrics.csv` |

- **`W100` Baseline Provenance Note:** In `exp031_models.py:92`, `EXP031_ARMS["A"]["reused_oof_csv"]` points to the older `experiments/EXP-030/p2_164k_subject_metrics.csv` (`64.79%`). The canonical 100-epoch baseline for `W100_500ep` is **`experiments/EXP-031/runs/W100/W100_subject_metrics.csv`** (`66.68%`, `EXP-031` Amendment A1 retrained under the identical `EXP-031` recipe).

---

## 3. Held Constants & Pre-Sweep Parity Gate

Every training, data, and evaluation constant matches [`experiments/EXP-031/run_exp031_width_sweep.py`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-031/run_exp031_width_sweep.py) except `--max-epochs 500`:
- **Optimizer:** `torch.optim.Adam(lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0)`, constant LR (no scheduler).
- **Batch Sizes & Inference:** `micro_batch_size=4` (`drop_last=False`). Validation/test uses `LogitStitchingSlidingWindowPredictor` evaluating window-by-window (`B=1` per sequence), matching `run_exp031_width_sweep.py:192-198` (even though `--val-batch-size 16` is declared in `argparse`).
- **Epoch & Early Stopping Schedule:** `max_epochs=500`, `min_epochs=40`, `patience=25` on inner-validation Macro-F1 (`val_f1 > best_val_f1 + 1e-6`), reloading the best inner-validation checkpoint before outer-test evaluation.
- **Gradient Clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`.
- **Loss & Precision:** PyTorch AMP (`torch.amp.autocast(..., dtype=torch.float16)` + `GradScaler`) for forward pass, with `CE + 0.01 * RCCLoss` computed in pure FP32 outside `autocast`.
- **Input & Context:** `context_len=1200`, `neutralize_circadian=True`, `RandomCropSleepDataset` random temporal crop.
- **Seeds & Determinism:** Seeds `[42, 123, 2024]` using `set_deterministic_seed` (`experiments/EXP-029/run_exp029_cv.py:74-79`), which seeds `random`, `np.random`, `torch`, `torch.cuda`, and `DataLoader` `Generator` with `seed` (without `cudnn.deterministic=True`), matching `EXP-031` identically.
- **Folds:** 5 subject-disjoint folds (`review_package/fold_manifests.csv`, SHA-256 `36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf`; 5 inner-validation subjects/fold, 9–10 outer-test subjects/fold, $N=47$ total).
- **Pre-Sweep Epoch `1..100` Reproducibility Check:** Before launching the full 60-run sweep, execute `W100_500ep` on `Fold 1, Seed 42` and verify that epochs `1..100` `val_macro_f1` and `best_epoch` match the locked `EXP-031` `W100_fold1_seed42.json` trajectory (within minor CUDA non-determinism if any).

---

## 4. Output Schema & Locked Endpoints / Decision Rules

### 4.1 Output Layout (`experiments/EXP-034/`)
- Per-run JSONs: `experiments/EXP-034/run_results/<arm_id>/<arm_id>_fold{fold}_seed{seed}.json`
- Best checkpoints: `experiments/EXP-034/checkpoints/<arm_id>/<arm_id>_L1200_fold{fold}_seed{seed}.pt`
- Subject metrics CSVs: `experiments/EXP-034/<arm_id>_subject_metrics.csv` (`N=47` rows keyed by `subject_id`)
- Aggregated summary & report: `experiments/EXP-034/exp034_summary.json` and `experiments/EXP-034/EXP034_FINAL_REPORT.md`
- **Mandatory Per-Run Convergence Keys:** Every run JSON must explicitly log top-level `best_epoch`, `completed_epochs` (`len(history)`), `early_stopped` (`bool(completed_epochs < max_epochs)`), `wall_sec`, and `peak_vram_mb`.

### 4.2 Primary Per-Arm Convergence Endpoint (`Δ_500_vs_100`)
- Join each `experiments/EXP-034/<arm_id>_subject_metrics.csv` with its canonical 100-epoch baseline CSV on `subject_id` ($N = 47$), scaling `macro_f1 * 100.0` to percentage points (`pp`):
  $$\Delta_{\text{500\_vs\_100}}^{(m)} = \text{F1}_{m\text{\_500ep}} - \text{F1}_{m\text{\_100ep}}$$
- **Statistical Computation Spec:**
  - **Bootstrap CI:** Percentile 95% CI (`[2.5, 97.5]`) over `10,000` participant-level resamples using `np.random.default_rng(42)` reset independently per arm.
  - **Wilcoxon Test:** `scipy.stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox", method="auto")`.
  - **Multiplicity Control:** Holm-Bonferroni adjustment across the $m = 4$ paired arm comparisons.
- **Decision Rule:** A 500-epoch convergence gain for arm $m$ is **SUPPORTED** only if:
  1. Mean $\Delta_{\text{500\_vs\_100}}^{(m)} \ge +0.75\text{ pp}$ (pre-registered practical threshold),
  2. 95% percentile bootstrap CI lower bound $> 0.0\text{ pp}$, and
  3. Holm-adjusted two-sided Wilcoxon $p_{\text{Holm}} < 0.05$.
  If $p_{\text{Holm}} < 0.05$ with CI lower bound $> 0$ but $\Delta < +0.75\text{ pp}$, it is **STATISTICALLY DETECTABLE BUT BELOW PRACTICAL THRESHOLD**. Otherwise **NOT SUPPORTED**.

### 4.3 Diagnostic `H-SCALE` Sensitivity Check & Gating Rule
- **Diagnostic Sensitivity Check (Mixed-Budget):** Recompute Page's $L$ test ($N = 47$) across the 9 ordered `EXP-031` widths substituting `W050_500ep` (`43,108`), `W075_500ep` (`93,796`), `W100_500ep` (`163,940`) alongside the 6 remaining 100-epoch CSVs:
  - `P190` (`190,940`): `experiments/EXP-031/P190_subject_metrics.csv`
  - `B` (`223,268`): `experiments/EXP-029/cnn_bigru_200k_subject_metrics.csv`
  - `W125` (`253,540`): `experiments/EXP-031/W125_subject_metrics.csv`
  - `W150` (`362,596`): `experiments/EXP-031/W150_subject_metrics.csv`
  - `W200` (`639,076`): `experiments/EXP-031/W200_subject_metrics.csv`
  - `W250` (`993,380`): `experiments/EXP-031/W250_subject_metrics.csv`
  Plus a 10-width sensitivity check inserting `p1_100k_500ep` (`98,940`) between `W075_500ep` and `W100_500ep`.
- **Page's $L$ Formula:** Standard normal approximation $z_L = \frac{L - \frac{N k (k+1)^2}{4}}{\sqrt{\frac{N k^2 (k+1)(k^2 - 1)}{144}}}$, with monotonically increasing one-sided alternative $p_{\text{one}} = 1 - \Phi(z_L)$ and two-sided $p_{\text{two}} = 2 \cdot \min(\Phi(z_L), 1 - \Phi(z_L))$.
- **Mandatory Epoch-Budget Gating Rule:** Because comparing 500-epoch compact arms against 100-epoch larger arms confounds capacity with epoch budget, this mixed-budget Page's $L$ test is strictly **diagnostic**. If any compact arm (`W050_500ep`, `W075_500ep`, `p1_100k_500ep`, `W100_500ep`) achieves `SUPPORTED` (or statistically detectable gain with $\text{CI}_{\text{lo}} > 0$) at 500 epochs, `P190`–`W250` must also be trained at 500 epochs before drawing any conclusion on `H-SCALE`.

---

## 5. Compute & Dual-GPU Resource Estimate

Measured from `EXP-030` / `EXP-031` run logs on dual `NVIDIA GeForce RTX 3060` (12,288 MiB VRAM each):
- **Peak VRAM:** `242.6 MB` (`W050`), `344.5 MB` (`W075` & `p1_100k`), `433.9 MB` (`W100`) — `< 0.5 GB` per run.
- **Throughput (includes per-epoch validation):** `3.11 s/ep` (`W050`), `3.33 s/ep` (`W075`), `3.39 s/ep` (`p1_100k`), `3.73 s/ep` (`W100`).
- **Dual-GPU Runtime Estimate:** Splitting 30 runs on GPU 0 (`W050_500ep` + `W100_500ep`, `~14.25 h` worst-case) and 30 runs on GPU 1 (`W075_500ep` + `p1_100k_500ep`, `~14.00 h` worst-case), worst-case wall time if all 60 runs reach epoch 500 is **~14.1 hours** (`~5–8 hours` expected under `patience=25` early stopping).
