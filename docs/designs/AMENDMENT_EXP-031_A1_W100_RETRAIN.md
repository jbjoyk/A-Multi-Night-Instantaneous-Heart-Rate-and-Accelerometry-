# AMENDMENT A1 to PREREG_EXP-031: Retrained W100 Control and Interleaved Latency Benchmark

**Status:** DRAFT until committed. Commit this file and record the hash in `EXPERIMENTS.md` **before** training any W100 run or running the latency benchmark. After the commit, sections 3 to 8 are immutable; corrections go in section 11.
**Parent:** `PREREG_EXP-031_WIDTH_SWEEP.md` (its decision rules, results and artifacts are unchanged).
**Untouched:** frozen Model A (`DEC-037`) checkpoints, all existing EXP-031 arm directories, and every EXP-018 to EXP-030 artifact.
**Pre-lock checks:** all repository verifications in Sections 3 and 9 have been resolved from actual code and logs before commit.

---

## 1. Why this amendment

In EXP-031, frozen Model A (163,940 parameters) scored 64.79% Macro-F1, 1.5 to 1.8 pp below every other plain width, including P190 (190,940 parameters, 66.60%). Architecturally, P190 differs from Model A only by encoder width 64 to 72 (×1.125) and GRU hidden size 48 to 50 (×1.04). A change that small cannot plausibly explain a 1.8 pp gap. The sweep arms also used early stopping with best-validation-epoch selection. Frozen Model A may have been trained with a different recipe.

**Question (RECIPE):** does Model A's architecture, retrained under the exact EXP-031 sweep recipe (W100), reach the level of its neighbors?

---

## 2. Disclosures

1. **Design-informed:** this control was designed after seeing the EXP-031 results, including the P190 and p1_100k gaps over frozen Model A.
2. **Blind rule:** the decision rule in section 6 is fixed before any W100 result exists.
3. **Anomaly source:** the frozen-Model A outlier was noticed from the EXP-031 figures, not predicted in advance.

---

## 3. Single factor and constants

**Single factor:** training recipe (frozen `DEC-037` recipe vs EXP-031 sweep recipe), with identical architecture.

**W100 architecture (must match frozen Model A):** `d_enc = 64`, `d_temp = 48` per direction (`d_temp = 96` bidirectional output dim in `ScaledConventionalCNNBiGRU`), `head_hidden = 64`, 163,940 parameters. Verified programmatically against reference checkpoint `experiments/EXP-018/checkpoints/modelA_L1200_fold1_seed42.pt`: `build_exp031_model("A")` yields 163,940 parameters, 62 state-dict keys, `list(w100.state_dict().keys()) == list(ckpt_sd.keys()) == True`, and 0 tensor shape mismatches.

**Training recipe (the exact locked EXP-031 recipe):**
- Adam, learning rate 1e-4, weight decay 0.0
- training batch 4, validation batch 16
- max 100 epochs, min 40 epochs, patience 25
- gradient clipping 1.0
- FP32 cross-entropy plus 0.01 × RCCLoss, computed outside autocast
- `LogitStitchingSlidingWindowPredictor` (same evaluation path as EXP-031; its parity with the frozen Model A evaluation was verified in EXP-030)
- fold manifests: `review_package/fold_manifests.csv` (SHA-256: `36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf`)
- seeds 42, 123, 2024; 5 folds; 15 runs

**Recipe comparison table (required before training):**

| Model / Experiment | Optimizer and Learning Rate | Weight Decay | Loss | Epoch and Patience Schedule | Batch Size | Checkpoint Selection Rule | Gradient Clipping |
|---|---|---|---|---|---|---|---|
| **Frozen Model A** (`DEC-037` / `EXP-018`: `exp013_models.py`, `regenerate_modelA_checkpoints.py`) | `Adam`, `lr=1e-4`, `betas=(0.9, 0.999)` | `0.0` | `CE + 0.01 * RCCLoss` (FP32, no autocast) | Fixed-step budget: `max_steps=3200` optimizer steps (`N_accum=1`), no early stopping | Train `B=2` (`RandomCropSleepDataset`, `L=1200`); Eval `SlidingWindowPredictor` (`B=1` per night) | Final step (`step=3200`) weights saved directly (no validation-epoch selection) | `clip_grad_norm_(max_norm=1.0)` |
| **`EXP-031` Sweep & `W100`** (`experiments/EXP-031/run_exp031_width_sweep.py`) | `Adam`, `lr=1e-4`, `betas=(0.9, 0.999)` | `0.0` | `CE + 0.01 * RCCLoss` computed in FP32 outside `torch.amp.autocast` (`float16` forward) | `max_epochs=100`, `min_epochs=40`, `patience=25` on inner-val Macro-F1 | Train `micro_batch_size=4`; Val/Test `val_batch_size=16` (`LogitStitchingSlidingWindowPredictor`) | Best inner-val Macro-F1 epoch (`val_f1 > best_val_f1 + 1e-6`) | `clip_grad_norm_(max_norm=1.0)` after `scaler.unscale_` |
| **Official SLAMSS-IFS** (`EXP-021`: `run_tsk08_slamss_matched.py`, `official_slamss_ifs_adapter.py`) | `Adam`, `lr=1e-4`, `betas=(0.9, 0.999)` | `0.0` (PyTorch default) | `CE + 0.01 * RCCLoss` (`OfficialSLAMSSCompositeLoss` in FP32 outside `torch.amp.autocast`) | `max_epochs=100`, `min_epochs=40`, `patience=25` on inner-val Macro-F1 | Train `micro_batch_size=4`, `effective_batch_size=4`; Val/Test `val_batch_size=16` (`SLAMSSSlidingWindowPredictor`) | Best inner-val Macro-F1 epoch | `clip_grad_norm_(max_norm=1.0)` after `scaler.unscale_` |

**Isolation:** write only to `experiments/EXP-031/runs/W100/` and `experiments/EXP-031/W100_retrained_summary.json`.

---

## 4. Outcomes and statistics

- **Primary outcome:** subject-level Macro-F1 (N = 47), averaged over the three seeds within subject.
- **Inference:** paired differences, 10,000-resample participant-level percentile bootstrap 95% CI (seed 42), two-sided Wilcoxon signed-rank, Cohen's d_z.
- **Reference values are read from the per-subject CSVs, not typed:** frozen Model A, p1_100k, P190, `full_v5` and official SLAMSS-IFS.
- **Multiplicity:** Δ_R below is the single primary comparison and is not adjusted. The three secondary comparisons are Holm-adjusted together.

**Power note:** EXP-029 gave a paired SD of about 2.3 pp for `full_v5` vs frozen Model A (derived from its CI). With N = 47, the minimum detectable effect at 80% power is about 0.9 pp, so the threshold below is set at +1.0 pp.

---

## 5. Comparisons

- **Primary (Δ_R):** W100 minus frozen Model A.
- **Secondary (Holm-adjusted together):** W100 minus P190; W100 minus p1_100k; W100 minus official SLAMSS-IFS.
- **Sensitivity (H-SCALE-R):** re-run Page's L across the ordered widths with W100 replacing frozen Model A. This is reported as a labelled sensitivity result. The original EXP-031 H-SCALE verdict (Page's L = 10,938, z = +2.49, p_Holm = 0.0254) is unchanged and stays in the record.

---

## 6. Decision rule (LOCKED at commit)

Let Δ_R = W100 minus frozen Model A (primary). Let Δ_P = W100 minus P190.

| Outcome | Condition | Interpretation |
|---|---|---|
| **R1: recipe effect** | Δ_R ≥ +1.0 pp and the CI lower bound > 0, **and** the CI lower bound of Δ_P > −1.0 pp | The gap between frozen Model A and its neighbors is largely a training-recipe effect. |
| **R2: unexplained gap** | the CI upper bound of Δ_R < +1.0 pp, **and** the CI upper bound of Δ_P < 0 | Recipe does not explain it, and W100 is meaningfully below P190. Something else (a sharp width threshold, or instability) is responsible. |
| **R3: inconclusive** | anything else | Neither explanation is established. |

**Actions per outcome:**
- **R1:** frozen Model A is reported as "fixed-budget baseline (older recipe)" and W100 as "Model A architecture under the sweep recipe." The SLAMSS-IFS comparison is re-stated against W100. Personalization, SleepAccel, Bland-Altman and INT8 results are re-run on the recipe-consistent backbone under a separate preregistration.
- **R2:** do not claim a capacity effect. Add intermediate widths (for example, `d_enc` 68 and `d_temp` 49) under a new preregistration, and report the 164k-to-191k discontinuity as an unexplained, possibly unstable result.
- **R3:** report all comparisons with CIs and make no recipe or capacity claim.

**Reporting rules (all outcomes):**
- Report frozen Model A, W100, P190, p1_100k and SLAMSS-IFS together. Never report only the best.
- No claim of "parity" with SLAMSS-IFS without a pre-specified equivalence margin. Report differences with CIs.
- State the recipe differences between all compared models wherever a comparison appears.

---

## 7. Interleaved CPU latency benchmark (also locked)

Run **after** all W100 training has finished, on an otherwise idle machine, so training does not distort timings.

- **Models:** p1_100k, W100 (retrained), `full_v5`, P190 and official SLAMSS-IFS, all at L = 1200, batch size 1.
- **Precision modes:** FP32 and dynamic INT8 for every model. Note that dynamic INT8 quantizes only recurrent and linear layers, so it affects SLAMSS-IFS little.
- **Path:** time the complete inference path, including SLAMSS-IFS's autoregressive decoder.
- **Threads:** 1 and 4 (`OMP_NUM_THREADS` and `torch.set_num_threads`); pin the process (`taskset`) if available and set the CPU governor to performance if permitted.
- **Order and repetitions:** interleaved round-robin across models; 20 to 50 warm-up passes, then 200 timed passes per model, per thread setting.
- **Report:** median, interquartile range and coefficient of variation; peak resident memory; MMACs. If any model's coefficient of variation exceeds 5%, repeat the run.
- **Interpretation rule:** latency differences under 10% are described as not distinguishable.
- **Record:** the exact CPU model string from `/proc/cpuinfo` (do not type it), core count, governor, torch and OS versions. Record the two earlier inconsistent Model A measurements (47.7 ms and 54.9 ms at four threads) as a cautionary note.
- **Output:** `experiments/EXP-031/interleaved_cpu_latency.json`.

---

## 8. Outputs and consistency

- Per-subject, per-seed metrics (CSV); `W100_retrained_summary.json`; the recipe comparison table; the verdict (R1, R2 or R3) written exactly as defined in section 6; H-SCALE-R result; latency JSON.
- **All figure annotations and manuscript numbers are generated from these artifacts by script.** No hand-typed numbers. This covers the saturation-knee wording and the H-SCALE callout, which depend on W100 and may change.
- Before any manuscript regeneration, run a consistency checker that extracts every number from the `.tex` and `.md` files and compares it with the JSON and CSV artifacts, and report mismatches.

---

## 9. Compute

About 41–47 minutes for 15 runs across two GPUs, verified from `experiments/EXP-031/run_results/{W050..W250}/*.json` (15-run single-GPU sum: 77.66–94.65 min, mean 84.53 min; dual-GPU ~38.8–47.3 min, specifically 41.00 min for `W075` and 47.32 min for `P190`, ~42.3 min average). The latency benchmark follows afterwards.

---

## 10. Not in scope

Personalization, SleepAccel and Bland-Altman re-runs on W100; any new widths; any change to the EXP-031 verdicts.

---

## 11. Amendments

| Date | Section | Change | Reason |
|---|---|---|---|
| | | | |
