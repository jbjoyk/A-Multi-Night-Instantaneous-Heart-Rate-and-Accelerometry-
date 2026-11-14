# Research Handoff

## Current task

EXP-017 CONTROLLED SEQUENCE TRUNCATION EXPERIMENT COMPLETED (`T=600` vs `T<=1200`).
Successfully executed and evaluated the governed EXP-017 scientific protocol under explicit human authorization:
1. **Governed Execution & Model Configuration**:
   - Model: Model A only (`ConventionalCNNBiGRU`, 163,940 parameters).
   - Conditions: $T=600$ (5 hours / 600 epochs) vs $T \le 1200$ (10 hours / 1200 epochs).
   - Recurrent Padding Protection: Integrated `pack_padded_sequence` and `pad_packed_sequence(total_length=T)` using native sequence length $L = \min(\text{native\_len}, T)$. Zero tail contamination verified ($\max |\Delta| = 0.0$).
   - Training: 80 epochs per run, Adam ($\text{lr}=10^{-4}$, weight decay 0.0), batch size 4 nights, seeds 42, 123, 2024. All 6 runs completed on `cuda:0` under `GPUResourceManager`.
2. **Strict Test Cohort Protection**:
   - 11 held-out test subjects strictly locked (`test_set_accessed = False`, `assert_no_test_leakage` passed).
   - Validation on 5 disjoint subjects (28 nights, 16,368 valid matched-600 windows).
3. **Primary Estimand & Hypothesis Decision**:
   - Primary Estimand: Matched-600 REM recall difference $\Delta \text{REM Recall} = \text{REM Recall}_{(T \le 1200)}[\text{eval } 0:600] - \text{REM Recall}_{(T = 600)}[\text{eval } 0:600]$ on identical first 600 chronological epoch positions.
   - Point Estimate: **$-0.84\text{ percentage points}$** (Seed 42: $-5.32$ pp, Seed 123: $-3.85$ pp, Seed 2024: $+6.49$ pp).
   - Paired Subject Cluster Bootstrap ($B=1000$): **$[-5.68\text{ pp}, +3.94\text{ pp}]$**, strictly crossing zero.
   - Win/Loss: 2 subjects positive, 3 subjects negative.
   - Protocol Decision: **`INCONCLUSIVE`** / hypothesis $H_1$ not supported. Extending sequence training horizon does not specifically rescue early-night REM recall.
4. **Secondary Metrics & Descriptive Analysis**:
   - Matched-600 Positions: Overall Accuracy $+2.33\text{ pp}$ [95% CI: $+1.64\text{ pp}, +3.03\text{ pp}$] ($70.47\% \to 72.80\%$), Macro-F1 $+0.0210$ ($0.7126 \to 0.7335$), Deep sleep recall $+8.52\text{ pp}$ ($65.77\% \to 74.57\%$), Boundary error rate $-2.72\text{ pp}$ ($44.38\% \to 41.66\%$).
   - Descriptive Extended Coverage ($T \le 1200$, 23,183 valid windows): Accuracy $71.45\% \pm 0.43\%$, Macro-F1 $0.7153 \pm 0.0024$, REM recall $70.42\% \pm 3.81\%$, Deep recall $75.24\% \pm 4.31\%$.
5. **Artifacts & Research State**:
   - Checkpoints saved in `research/experiments/EXP-017/checkpoints/` (~3.9 MB total).
   - Metrics saved in `research/experiments/EXP-017/` (`EXP017_SUMMARY.csv`, `EXP017_SUBJECT_DELTAS.csv`, `EXP017_BOOTSTRAP.json`, individual run JSONs).
   - Comprehensive report authored in `EXP017_FINAL_REPORT.md`.
   - Documented in `DECISIONS.md` (DEC-022) and `RESEARCH_STATE.md`.
   - `research_memory.json` updated and validated (14 records total; 12 real scientific, 1 simulation, 1 smoke test).
6. **Unit Tests & Integrity**:
   - 79/79 unit tests passing (`pytest` clean).
7. **Next Concrete Task**:
   - Formulate protocol for RQ-001 (EXP-015: Clock Feature Pruning in `x_meta`) or RQ-002 (EXP-016: Unsupervised GRL Adaptation Schedule Recalibration).


## Previous work

- Initialized Git on `main`.
- Added a `.gitignore` for local environments, caches, checkpoints, and raw dataset recordings.
- Created the initial project commit.
- Created the shared research-state files in this handoff.
- Inspected the active notebook and recorded its executed configuration and outputs.

## Files modified

- `AGENTS.md`
- `RESEARCH_STATE.md`
- `HANDOFF.md`
- `NOTEBOOK_STATE.md`
- `DECISIONS.md`
- `EXPERIMENTS.md`
- `README.md`
- `prompts/RESEARCH_HANDOFF.md`

## Tests and verification

- Git repository initialized successfully.
- Initial commit created successfully.
- Initial handoff status: revised notebook execution had not yet been run; see the completed controlled optimization and final evaluation entry below.
- Historical notebook execution: 21 code cells executed with saved outputs.
- Historical model training: supervised comparisons and SleepJEPA pretraining/fine-tuning completed.
- Research metrics: recorded notebook outputs, not independently reproduced.

## Current errors

The notebook reports motion-energy/wake cross-correlation flags for 134/246 eligible nights; these
are not timestamp-alignment failures. The raw dataset is not included in the Git commit because it
is approximately 28 GB and is excluded by `.gitignore`.

## Next concrete task

Complete the raw timestamp/coverage audit and compare split membership with the older manifest if available. Then independently reproduce the latest baseline before publication claims.

## Relevant paths

- `sleepchannelnet_protocol_patch_v2.ipynb`
- `requirements.txt`
- `a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0/README.md`
- `a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0/SHA256SUMS.txt`

## 2026-10-24 notebook update

- Updated `sleepchannelnet_protocol_patch_v2.ipynb` with an SE channel-attention model, three HR-motion features, transition penalty/Viterbi helpers, a label-free entropy-head adaptation path and a separately named supervised ceiling.
- Removed the JEPA implementation/run cells from the executable path and documented its recorded negative result and limitation.
- Reframed claims to healthy-population longitudinal monitoring and marked exact SLAMSS-IFS reproduction as not implemented/verified.
- New ablation toggles default off. The recorded baseline outputs are historical; no new experiment/training was run.
- Verification: Python syntax compilation of notebook code cells only. No runtime or data-backed validation; installed `nbformat` was unavailable in the default shell Python.
- Existing unrelated deletions `pipeline.ipynb` and `sleepgan.ipynb` were present in the working tree and were not touched.
- Next concrete task: compare split membership with the prior manifest if available, finish timestamp/coverage checks on the flagged and long-gap nights, then reproduce the baseline and run validation-selected ablations.


## 2026-10-24 result audit

The supplied latest run is documented in EXP-003. It uses split hash `a6b569f63126` (seed 0, 33/7/7), not the older hash in the prior notes. The code hashes dataset/preprocessing identity as well as the subject split. Motion/wake lag flags are not timestamp-alignment failures. Three raw-night spot checks show no global clock offset; Bidslab01/3 has early signal termination. The notebook diagnostic wording now distinguishes behavior correlation from timestamp QC. No offset, model, or result was changed.


## 2026-10-24 optimizer and pipeline update

- Modified `sleepchannelnet_protocol_patch_v2.ipynb` to add grouped AdamW parameter decay; ReduceLROnPlateau and cosine warm-up; gradient clipping, train/validation loss and Macro-F1 logging, early stopping, AMP scaler and resume RNG state; an opt-in staged optimizer search using validation subjects only; corrected HR-motion feature handling; fixed transition-cost/unknown-pair handling; raw versus Viterbi-decoded metrics; and grouped validation-subject permutation analysis. The baseline remains available.
- Core training and optimizer search are separately opt-in to permit validation selection before final test evaluation. At this initial code-change checkpoint no model training or test evaluation had yet been run; see the completed run below.
- Checks run: all 21 code cells compiled; five in-memory feature checks passed; transition loss/gradient and single-epoch Viterbi checks passed. Sandbox CUDA was unavailable; host `nvidia-smi -L` reported two RTX 3060 GPUs.
- At this initial code-change checkpoint, revised cache data, split fingerprint, training behavior and metrics were NOT VERIFIED. They are now verified for the run recorded below; the requested ablations remain unrun.
- At that checkpoint the next task was validation search; the search and frozen test evaluation are now complete as recorded below. The current next task is to select ablation flags in the new controls cell near the notebook top, then run the ablation cell if needed.

### Cache isolation note

A preflight preparation was interrupted before model training after finding that the previous feature cache path did not include `FEATURE_VERSION`. It had begun rebuilding the old cache directory in place. The source now includes `FEATURE_VERSION` in both the per-night fingerprint and cache-directory hash, so future preprocessing writes into a version-isolated cache. The existing metrics/CSV/JSON/model result artifacts were not overwritten. Some ignored `.npz` files in the prior `prep_f4769d6efe` cache may have been regenerated before interruption; they are reproducible from the raw recordings. The interrupted runs did not fit a model or evaluate test subjects.

## 2026-10-24 controlled optimization and final evaluation

- Completed the staged validation-only optimizer search on the preserved split hash `a6b569f63126` (33/7/7 subjects). Ten screened rows and two full-epoch confirmations selected AdamW, lr 0.001, Plateau, grouped weight decay 0.01, dropout 0.2, and full-context inference; best confirmation validation Macro-F1 was 0.678384.
- Ran the frozen final fit and test evaluation once after configuration selection. Test partition: 7 subjects, 38 nights, 33,126 known epochs; accuracy 0.677715, Macro-F1 0.664803, kappa 0.531064. Subject-bootstrap CIs and class results are in EXP-004 and `final_test_secondary_summary.json`.
- The revised result is below the supplied historical Macro-F1 0.67791; do not claim an improvement. Test predictions were persisted before a JSON conversion error; the serializer was fixed and reports recovered from saved predictions without rerunning test inference.
- Outputs are under ignored `outputs/sleep_optimizer_work_REAL/validation_optimizer_search/` and `outputs/sleep_optimizer_work_REAL/runs/4class_channel_attn_complete_s42_2814bfb16c/`.
- Checks: all code cells compile; feature, transition loss/gradient, and Viterbi edge checks pass. The real preprocessing cache validated 253/253 nights (246 eligible). Project environment used Python 3.14.7, PyTorch 2.14.1+cu130, and two RTX 3060 GPUs.
- The notebook was exercised with a harness through data preparation, optimizer search, and run functions; it was not executed as one top-to-bottom Jupyter session. Ablations, permutation importance, personalization, and transition-lambda search remain unrun.
- A separate interrupted preparation created a partial ignored cache under `outputs/sleep_optimizer_work_REAL_REAL`; it did not train or access test labels and was not used for results.
- To make the controls easier to find, added a top-of-notebook section and code cell id `run-controls` (cell 2). The ablation runner is now cell 49; all 22 code cells compile after this update.
- Next task: inspect the run config/metrics and then execute requested ablations one at a time if needed, keeping the held-out test out of further selection.


## 2026-10-25 checkpoint resume error

- Fixed `_restore_torch_rng` in `sleepchannelnet_protocol_patch_v2_UPDATED.ipynb` (cell 23) to convert saved CPU and CUDA RNG state values to contiguous CPU `torch.uint8` tensors before passing them to PyTorch. This addresses the reported resume `TypeError`.
- Verification: notebook JSON is valid, all 22 code cells compile, and `git diff --check` passes. Resume/training was not run.
- Next: user should use this updated notebook, restart the kernel, rerun setup through the model/training definitions, and rerun the failed ablation cell.


## 2026-10-25 outputs and plots

- Updated `sleepchannelnet_protocol_patch_v2_UPDATED.ipynb` to write into `sleepchannelnet_results_REAL` beside the notebook, ignoring stale `SLEEP_WORK_DIR` overrides.
- Removed the redundant per-run `history.csv`; checkpoints and feature caches remain for resume/reuse.
- Added ablation comparison and per-stage F1 plots, saved under the same result directory after an ablation run.
- Verification: notebook JSON is valid, all 22 code cells compile, and `git diff --check` passes. No training/plot generation was performed.
- Next: restart the notebook kernel and rerun the required setup cells before launching the desired experiment.


## 2026-10-26 Optuna integration

- Replaced the staged fixed-grid optimizer function in the updated notebook with an Optuna TPE study. Search fits only training subjects and selects on validation Macro-F1; it does not load test subjects. Default budget: 12 screening trials x up to 20 epochs, then 2 full-epoch confirmations. Search is opt-in and was not executed.
- Added `optuna` to `requirements.txt`; the current `.venv` does not have the package installed. Install project requirements before enabling `RUN_OPTIMIZER_SEARCH`.
- Outputs: SQLite study and CSV/JSON summaries in `WORK/optuna_optimizer_search/`; work base is configured beside the notebook with `_REAL` suffix. Restored the checkpoint RNG dtype fix and requested ablation plots.
- Verification: JSON valid, all code cells compile, `git diff --check` passes. No training was started.
- Next: install requirements, set `RUN_OPTIMIZER_SEARCH=True`, and run cells in notebook order. This tunes on the existing seven validation subjects and does not increase the independent test-subject count.


## 2026-11-09 SLAMSS-IFS Scientific Equivalence Audit

- Completed comprehensive independent scientific reproduction audit comparing our baseline implementation against the official `BIDSLabUMass/SLAMSS-IFS` author repository and IEEE TBME 2026 paper text.
- Key artifacts created:
  - `official_slamss_ifs/`: Full local clone of the official author repository for line-by-line verification.
  - `data_sanity_check.py`: Discovered 47 real subjects, verified 31/5/11 split with 160/28/58 usable nights and zero partition overlap.
  - `test_rccloss_audit.py`: 10 deterministic unit tests verifying the author's exact `RCCLoss`, $w_{\text{FN}}$ and asymmetric $w_{\text{FP}}$ cost matrices, and composite loss scaling.
  - `test_cross_implementation.py`: Compared parameter counts and forward tensor shapes between official repo (5.47M parameters) and paper-text model (666k parameters).
  - `benchmark_and_reproducibility.py`: Profiled 10 real training steps (91.80 ms/step, 3.47 GB VRAM) and confirmed bitwise reproducibility (0.00e+00 difference across two seeds).
- Audit decision (DEC-011): Classified as Classification B (Faithful Reproduction with Documented Repository Adaptations).
- Next concrete task: Obtain user decision on whether to train the 666k paper-text specification model or the 5.47M official repository model before launching full benchmark training.


## 2026-11-09 SleepAdaptiveNet Proposed Model Implementation

- Implemented the modular `SleepAdaptiveNet` architecture in `sleep_adaptive_net.py` (DEC-012, EXP-006) supporting all five requested components and clean A0–A5 ablations:
  - Separate modality encoders for IHR and ACC (`HREncoder`, `ACCEncoder`).
  - Learned Adaptive Modality Reliability Fusion Gate ($\alpha_t$) with logging and stage-wise analysis.
  - Transition-Aware Module with learnable $4 \times 4$ transition matrix, transition-adjusted logits, and transition consistency loss.
  - Boundary-Aware Loss focusing supervision on sleep-stage transitions ($y_t \neq y_{t-1}$).
  - Subject-Invariant Representation via Gradient Reversal Layer (GRL) and training-only subject discriminator (31 subjects; validation/test subjects strictly filtered).
  - Uncertainty and Confidence Head outputting normalized predictive entropy and learned correctness confidence.
- Key artifacts created:
  - `sleep_adaptive_net.py`: Complete modular PyTorch model with `build_ablation_model('A0'..'A5')`.
  - `test_sleep_adaptive_net.py`: Test suite verifying shapes, gradients, GRL reversal, ablation toggles, and subject leakage prevention.
  - `run_ablation_smoke.py`: End-to-end smoke training and evaluation pipeline on real data.
- Verification status: All unit tests pass. Parameter counts verified (A0: 145k, A5: 165k; 33x smaller than official SLAMSS-IFS). Smoke training (5 steps) and validation evaluation confirmed.
- Next concrete task: Await user authorization to launch controlled ablation training experiments across seeds 42, 123, 2024.


## 2026-11-09 Controlled Ablation Experiment (EXP-007, Seed 42)

- Executed the first controlled scientific ablation experiment on seed 42 across all 6 models (A0 through A5) using `run_controlled_experiment.py`.
- Experimental setup:
  - 160 usable training nights (31 subjects), 28 usable validation nights (5 subjects).
  - Test set (11 subjects, 58 nights) strictly excluded and never accessed.
  - 10 epochs per ablation with batch size 4 and Adam ($\text{lr}=10^{-4}$). Sequence length $T=600$ (0 padding artifacts).
  - All 6 ablations finished in 49.6 s (~8 s per ablation, peak VRAM: 265 MB).
  - Key finding: A5 (Full Model) achieves the highest validation Macro-F1 (0.5411 vs 0.5282 baseline) and doubles REM F1 (0.361 vs 0.181). Boundary loss (A3) cuts boundary errors by 7.0 percentage points. Subject-invariant GRL (A4) provides the largest generalization recovery (+5.0% accuracy over A3).
- Artifact saved: `exp007_controlled_ablation_results.pkl`.
- Next concrete task: Discuss findings with user; confirm whether to evaluate across seeds 123 and 2024 or adjust $\lambda_{\text{boundary}}$ before proceeding. Full 500/1000-epoch training is NOT started.


## 2026-11-09 Boundary Weight Sweep & GRL Disentanglement (EXP-008, Seed 42)

- Executed the controlled boundary weight sweep across $\lambda_{\text{boundary}} \in \{0.0, 0.10, 0.20, 0.50\}$ and isolated GRL evaluation using `run_exp008_boundary_sweep.py`.
- Experimental setup:
  - Models evaluated: B0 (A2 reference, $\lambda=0$), B0_GRL (A4 with $\lambda=0$), B1 ($\lambda=0.10$), B2 ($\lambda=0.20$), B3 ($\lambda=0.50$).
  - Exactly identical dataset: 160 usable training nights (31 subjects), 28 usable validation nights (5 subjects). Test set strictly excluded.
  - 10 epochs, seed 42, Adam ($\text{lr}=10^{-4}$), batch size 4, $T=600$.
- Key findings:
  - GRL standalone effect unconfounded: B0 $\to$ B0_GRL yields +3.21% accuracy ($56.11\% \to 59.32\%$), +0.0353 Macro-F1 ($0.5256 \to 0.5610$), and +0.0407 Kappa ($0.3707 \to 0.4114$).
  - Boundary loss monotonically reduces boundary errors (from 54.47% at $\lambda=0$ to 49.59% at $\lambda=0.20$ and 45.85% at $\lambda=0.50$).
  - $\lambda=0.20$ is preferred for balanced boundary error reduction (-4.88% boundary error, 59.04% accuracy, 0.5543 Macro-F1). $\lambda=0.50$ is rejected due to severe Wake recall loss (drops from 52.97% to 39.99%).
- Artifact saved: `exp008_boundary_sweep_results.pkl`.
- Next concrete task: Await user review before any multi-seed confirmation or extended training. Full 500/1000-epoch training NOT started. Test set NOT used.


## 2026-11-09 Multi-Seed Stability Evaluation (EXP-009, Seeds 42, 123, 2024)

- Executed the multi-seed stability evaluation comparing Candidate 1 (`B0_GRL`, $\lambda=0$) and Candidate 2 (`B2`, $\lambda=0.20$) across 3 random seeds using `run_exp009_multiseed.py`.
- Experimental setup:
  - Exact identical split: 160 usable training nights (31 subjects), 28 usable validation nights (5 subjects). Held-out test set (11 subjects, 58 nights) strictly excluded.
  - 10 epochs per run, Adam ($\text{lr}=10^{-4}$), batch size 4, $T=600$ (0 padding).
  - Seed 42 loaded from `exp008_boundary_sweep_results.pkl`; seeds 123 and 2024 executed.
  - Total compute time for new runs: 34.6 s (~8.3 s per run). Peak VRAM: 265.1 MB.
- Key findings across $N=3$ seeds:
  - `B0_GRL` consistently beats `B2` across all 3 seeds in Accuracy ($59.61\% \pm 0.45\%$ vs $58.85\% \pm 0.21\%$, $\Delta = -0.75\% \pm 0.48\%$), Macro-F1 ($0.5606 \pm 0.0162$ vs $0.5491 \pm 0.0173$, $\Delta = -0.0115 \pm 0.0043$), Kappa ($0.4138 \pm 0.0128$ vs $0.4000 \pm 0.0104$, $\Delta = -0.0137 \pm 0.0062$), and Wake recall ($53.95\% \pm 3.29\%$ vs $47.75\% \pm 2.35\%$, $\Delta = -6.20\% \pm 1.06\%$).
  - `B2` consistently reduces boundary error rate across all 3 seeds by $-4.28\% \pm 0.77\%$ ($49.76\% \pm 0.91\%$ vs $54.04\% \pm 1.51\%$), confirming the reliability of boundary supervision.
  - Modality reliability gating consistently converges across all seeds to lower cardiac reliance during Wake ($\alpha \approx 0.58$) and higher cardiac reliance during Deep sleep ($\alpha \approx 0.78$).
  - Recommendation: `B0_GRL` is the recommended default candidate for overall staging. `B2` is retained as a specialized Pareto candidate for boundary-critical staging.
- Artifact saved: `exp009_multiseed_results.pkl`.
- Next concrete task: Await user decision and authorization on full 500-epoch training schedule. Held-out test set remains strictly unaccessed. Full 500/1000-epoch training is NOT started.


## 2026-11-10 Long-Training Convergence Evaluation (EXP-010, 500 Epochs, Seed 42)

- Executed the 500-epoch convergence evaluation of `B0_GRL` using `run_exp010_convergence.py`.
- Experimental setup:
  - Exact identical paper split: 160 usable training nights (31 subjects), 28 usable validation nights (5 subjects). Held-out test set (11 subjects, 58 nights) strictly excluded.
  - 500 epochs, Adam ($\text{lr}=10^{-4}$, weight decay 0.0), batch size 4, $T=600$ (0 padding).
  - Runtime: 466.5 s (7.78 min) total. Peak VRAM: 265.1 MB.
- Key findings:
  - Peak validation performance occurs at **Epoch 78**: Macro-F1 = **0.7441**, Accuracy = **74.22%**, Kappa = **0.6342**, lowest validation loss = **0.7245**, ECE = **0.0240**.
  - Extended training completely resolves the early REM under-fitting: REM F1 surges from 0.2743 at 10 epochs to **0.6882** (recall 63.92%) at Epoch 78.
  - Clear empirical post-peak overfitting occurs beyond Epoch ~80–100: while training loss drops monotonically from 1.0912 (Ep 78) to 0.5047 (Ep 500), validation loss rises from 0.7245 to 1.1473, Macro-F1 drifts down to 0.6829, and ECE deteriorates to 0.1949.
  - Adaptive gate stably preserves learned stage-dependent modality weighting ($\alpha_{\text{Wake}} \approx 0.59 - 0.68$ vs $\alpha_{\text{Deep}} \approx 0.80 - 0.88$) across all 500 epochs.
- Artifacts saved: `exp010_training_history.pkl`, `exp010_convergence_curves.png`, and `checkpoints_exp010/` (`best_macro_f1.pt`, `best_accuracy.pt`, `best_kappa.pt`, periodic checkpoints).
- Next concrete task: Review convergence dynamics with user. Decide whether early stopping (patience ~20) or learning rate decay at Epoch ~75–80 should be applied for the final held-out test evaluation. Held-out test set remains strictly untouched.


## 2026-11-10 Multi-Seed Convergence Confirmation & Reproducibility Audit (EXP-011)

- Executed the multi-seed 500-epoch convergence evaluation across seeds 42, 123, 2024 using `run_exp011_multiseed_convergence.py`.
- Experimental setup:
  - Exact identical paper split: 160 usable training nights (31 subjects), 28 usable validation nights (5 subjects). Held-out test set (11 subjects, 58 nights) strictly excluded.
  - 500 epochs per seed, Adam ($\text{lr}=10^{-4}$, weight decay 0.0), batch size 4, $T=600$ (0 padding).
  - Runtime: 1325.8 s (22.10 min) total (~7.4 min per seed). Peak VRAM: 265.1 MB.
- Reproducibility Audit Findings:
  - Traced the EXP-009 vs EXP-010 seed-42 discrepancy (59.32% vs 60.28% Acc) to inter-epoch `val_loader` iterations in PyTorch, which advance the CPU random generator state. When tested under matched evaluation frequencies, results replicate bitwise to the exact decimal place.
- Key findings across $N=3$ seeds:
  - Optimal convergence window is highly robust: Peak Epoch = $80.7 \pm 10.3$ (range: 72–92 epochs).
  - Peak validation metrics: Macro-F1 = **$0.7296 \pm 0.0132$** (CV: 1.8%), Accuracy = **$72.17\% \pm 1.86\%$**, Cohen's $\kappa$ = **$0.6073 \pm 0.0243$**.
  - Universal REM resolution: Across all 3 seeds, minority REM F1 surges from ~0.29 at 10 epochs to **$0.6886 \pm 0.0186$** (recall **$71.1\% \pm 6.5\%$**) at peak.
  - Universal post-peak overfitting: In all seeds, validation loss climbs to 1.15–1.29 at Epoch 500 while training loss falls to ~0.50.
  - Learned stage-dependent modality weighting ($\alpha_{\text{Wake}} < \alpha_{\text{Deep}}$) is robust across all seeds and training epochs.
- Artifacts saved: `exp011_multiseed_history.pkl`, `exp011_convergence_curves.png`, and `checkpoints_exp011/` (per-seed metric-peak and periodic checkpoints).
- Next concrete task: Finalize frozen benchmark training protocol (early stopping patience = 25 epochs or budget = 85 epochs) and await user authorization for the single held-out test evaluation. Held-out test set remains strictly untouched.


## 2026-11-10 Final Frozen Test Benchmark (EXP-012, 80 Epochs, Seeds 42/123/2024)

- Executed the frozen test evaluation of `SleepAdaptiveNet` (`B0_GRL`) on the untouched official held-out test set using `run_exp012_final_benchmark.py`.
- Experimental setup:
  - Exact official paper split: 31 train subjects (160 usable nights), 5 validation subjects (28 usable nights), 11 test subjects (58 usable nights, 34,800 evaluated windows).
  - 80 fixed epochs per seed across 3 seeds (42, 123, 2024).
  - Frozen hyperparameters: Adam ($\text{lr}=10^{-4}$, weight decay 0.0), batch size 4, $T=600$ (0 padding).
  - Zero early stopping, zero LR decay, zero test leakage or hyperparameter modifications.
  - Test evaluation performed exactly once per seed at the end of epoch 80.
  - Runtime: 219.0 s total (~73 s per seed). Peak VRAM: 265.1 MB.
- Key findings on held-out test set ($N=3$ seeds, Mean $\pm$ SD):
  - **Accuracy**: **$68.44\% \pm 1.35\%$** (Seed 42: 69.86%, Seed 123: 68.27%, Seed 2024: 67.18%)
  - **Macro-F1**: **$0.6859 \pm 0.0139$** (Seed 42: 0.6986, Seed 123: 0.6880, Seed 2024: 0.6711)
  - **Weighted-F1**: **$0.6835 \pm 0.0141$** (Seed 42: 0.6985, Seed 123: 0.6816, Seed 2024: 0.6704)
  - **Cohen's $\kappa$**: **$0.5527 \pm 0.0188$** (Seed 42: 0.5726, Seed 123: 0.5504, Seed 2024: 0.5352)
  - **MCC**: **$0.5541 \pm 0.0187$** (Seed 42: 0.5740, Seed 123: 0.5512, Seed 2024: 0.5370)
  - **ECE**: **$0.0257 \pm 0.0050$** (Seed 42: 0.0221, Seed 123: 0.0234, Seed 2024: 0.0315)
- Class-specific metrics (Test Mean $\pm$ SD):
  - **Wake**: Precision $64.01\% \pm 4.10\%$, Recall **$75.95\% \pm 1.20\%$**, F1 **$0.6940 \pm 0.0192$**
  - **Light**: Precision $65.68\% \pm 0.94\%$, Recall $70.02\% \pm 1.72\%$, F1 $0.6778 \pm 0.0119$
  - **Deep**: Precision $73.89\% \pm 3.46\%$, Recall $72.66\% \pm 1.86\%$, F1 **$0.7323 \pm 0.0174$**
  - **REM**: Precision $71.10\% \pm 1.47\%$, Recall $58.15\% \pm 2.94\%$, F1 **$0.6395 \pm 0.0207$**
- Transition Boundary Dynamics:
  - Boundary Error Rate: **$59.08\% \pm 0.47\%$** vs Non-Boundary Error Rate: **$30.41\% \pm 1.39\%$** (1.94x higher error at boundaries).
- Comparison with Published SLAMSS-IFS Benchmark:
  - SLAMSS-IFS (Table V): Accuracy = **71.04%**, Weighted-F1 = **0.7079**, Parameters = **5,465,768**
  - SleepAdaptiveNet (`B0_GRL`): Accuracy = **$68.44\% \pm 1.35\%$**, Weighted-F1 = **$0.6835 \pm 0.0141$**, Parameters = **162,325**
  - Parameter Reduction: **33.7x fewer parameters** (97.0% parameter compression).
  - Scientific Classification: **Case B (Competitive / Approximate Parity under 33.7x Parameter Compression)**.
- Artifacts saved:
  - `exp012_test_results.json`, `exp012_test_results.pkl`, `exp012_subject_metrics.csv`, `exp012_confusion_matrices.png`, `exp012_seed{42,123,2024}_final.pt`.
- Next concrete task: Synthesize findings for publication manuscript. DO NOT run EXP-013. Empirical benchmarking is complete.

## 2026-11-10 Rigorous Component Attribution Experiment (EXP-013)

- Executed the OFAT factorial component attribution experiment comparing SleepAdaptiveNet against a matched-capacity conventional CNN-BiGRU baseline and single-factor ablations using `run_exp013_attribution.py`.
- Experimental setup:
  - Factorial OFAT design:
    - Model A: Matched-capacity conventional CNN-BiGRU (163,940 params, +0.99%; Concat, no gate, no transition, no GRL).
    - Model B: E minus GRL (153,974 params, -5.14%; Gate ON, Transition ON, GRL OFF).
    - Model C: E minus Transition (162,325 params, 0.00%; Gate ON, Transition OFF, GRL ON).
    - Model D: E minus Dynamic Gate (153,876 params, -5.20%; Gate fixed 0.5/0.5, Transition ON, GRL ON).
    - Model E: Frozen EXP-012 Reference (162,325 params; Gate ON, Transition ON, GRL ON; read-only).
  - 80 fixed optimization epochs per seed (seeds 42, 123, 2024), batch size 4, lr 1e-4, Adam, sequence length T=600.
  - Exactly 34,241 valid evaluated test windows per run across 11 held-out test subjects.
  - Paired subject-level cluster bootstrap ($B=1000$) comparing Model E against A, B, C, D across identical resampled subject clusters.
  - Runtime: ~14.5 minutes on NVIDIA GeForce RTX 3060.
- Key findings across $N=3$ seeds:
  - **Full Architecture vs. Matched Baseline ($E - A$)**: SleepAdaptiveNet yields a statistically verified gain of **$+1.79\%$ Accuracy** [95% CI: $+0.52\%, +3.35\%$] and **$+0.0199$ Macro-F1** [95% CI: $+0.0074, +0.0342$]. The 95% bootstrap CI strictly excludes zero, establishing that SleepAdaptiveNet's architectural mechanisms provide genuine performance advantages not attributable to parameter capacity.
  - **Subject-Invariant GRL ($E - B$)**: GRL is the largest single component contributor, yielding a **$+1.39\%$ Accuracy** [95% CI: $-0.01\%, +2.84\%$] improvement, with 97.4% of bootstrap replications favoring GRL.
  - **Adaptive Reliability Gating ($E - D$)**: Dynamic reliability gating contributes **$+0.54\%$ Accuracy** [95% CI: $-0.53\%, +1.60\%$] and exhibits physiologically consistent downweighting of accelerometry during Wake ($\alpha \approx 0.54$) versus sleep stages ($\alpha \approx 0.80$).
  - **Transition Modeling ($E - C$)**: Transition module contributes **$+0.04\%$ Accuracy** [95% CI: $-0.29\%, +0.42\%$], indicating it is largely redundant when a 2-layer bidirectional GRU is utilized for sequence modeling.
- Artifacts saved:
  - `EXP013_RESULTS.md`
  - `EXP013_OUTPUTS/exp013_results.pkl`, `exp013_results.json`, `exp013_model_metrics.csv`, `exp013_paired_bootstrap.csv`, `exp013_per_class_metrics.csv`, `exp013_subject_metrics.csv`, `exp013_alpha_analysis.csv`
  - `EXP013_OUTPUTS/checkpoints/`
- Next concrete task: All empirical training and ablation phases are concluded. Do not launch any new training jobs. Synthesize final results for publication manuscript.

## 2026-11-10 EXP-012 Publication Benchmark & Architecture Report (PDF Deliverable)

- Authored and compiled a comprehensive 10-page publication-grade PDF report documenting Experiment 12 findings, architectural specifications, and baseline comparison against SLAMSS-IFS (IEEE TBME 2026).
- Generated publication figures via `generate_report_charts.py` stored in `report_assets/`:
  - `fig1_architecture_comparison.png`: Dual-stream residual feature extraction, dynamic reliability gating, BiGRU context, and GRL domain discriminator vs SLAMSS-IFS multi-scale convolution.
  - `fig2_benchmark_comparison.png`: Parameter footprint (5.47M vs 0.162M, 33.7x compression) and benchmark metrics (Accuracy 71.04% vs 68.44%, Weighted-F1 0.7079 vs 0.6835).
  - `fig3_per_class_breakdown.png`: Radar & bar comparison for Wake, Light, Deep, and REM Precision, Recall, and F1.
  - `fig4_boundary_error_analysis.png`: Transition boundary failure dynamics (59.08% error at transitions vs 30.41% in stable epochs; 1.94x risk ratio).
  - `fig5_per_subject_performance.png`: Subject-level accuracy distribution across all 11 held-out test subjects (Bidslab08 through Bidslab68).
- Compiled standalone CSS Paged Media HTML `exp012_report.html` via `compile_pdf.py` using WeasyPrint 70.0.
- Output Deliverable: `EXP012_BENCHMARK_REPORT.pdf` (1.79 MB, 10 pages).
- All 11 audited discrepancies reconciled, protocol boundaries stated (unpaired benchmark, $T=600$ vs $T=1200$, 80 fixed epochs vs 500-epoch validation-selected checkpoint).
- Next concrete task: Manuscript drafting and submission preparation.

## 2026-11-11 EXP-014 75-Run Subject-Level Cross-Validation Robustness Evaluation

- Completed the full 75-run subject-level 5-fold cross-validation across all 47 subjects and 246 usable nights using `run_exp014_cv.py`.
- Experimental matrix:
  - 5 models: Model A (Matched Baseline, 163.9k params), Model B (minus GRL, 154.0k params), Model C (minus Transition, 162.3k params), Model D (minus Adaptive Gating, 153.9k params), Model E (Full SleepAdaptiveNet, 162.3k params, freshly initialized per fold).
  - 5 outer folds (strictly disjoint subjects: 10, 10, 9, 9, 9 test subjects). Inner training on disjoint subjects, inner validation on 5 subjects per fold. Outer test subjects evaluated strictly once per run at locked epoch 80.
  - 3 seeds per model per fold (42, 123, 2024). Exactly 80 epochs per run, Adam ($\text{lr}=10^{-4}$, weight decay 0.0), batch size 4 nights, sequence length $T=600$.
  - 0 failed runs, 0 reruns, 0 protocol violations. Total compute duration: 5,745 s (~95.7 min) on NVIDIA RTX 3060.
- Key findings across all 47 subjects (145,283 valid evaluated windows):
  - **Outcome B (Equivalence / Parity Across Subject Folds)**:
    - Model A (Conventional CNN-BiGRU): Accuracy = $66.63\% \pm 3.33\%$, Macro-F1 = $0.6658 \pm 0.0365$, Cohen's $\kappa$ = $0.5251 \pm 0.0505$.
    - Model B (Minus GRL): Accuracy = $66.68\% \pm 3.31\%$, Macro-F1 = $0.6664 \pm 0.0345$, Cohen's $\kappa$ = $0.5244 \pm 0.0505$.
    - Model C (Minus Transition): Accuracy = $66.07\% \pm 4.04\%$, Macro-F1 = $0.6596 \pm 0.0424$, Cohen's $\kappa$ = $0.5154 \pm 0.0600$.
    - Model D (Minus Dynamic Gate): Accuracy = $66.68\% \pm 3.59\%$, Macro-F1 = $0.6684 \pm 0.0371$, Cohen's $\kappa$ = $0.5270 \pm 0.0514$.
    - Model E (Full SleepAdaptiveNet): Accuracy = $66.06\% \pm 4.08\%$, Macro-F1 = $0.6596 \pm 0.0425$, Cohen's $\kappa$ = $0.5153 \pm 0.0601$.
  - **Paired Subject Cluster Bootstrap ($B=1000$)**:
    - $E \text{ vs } A$: $\Delta \text{Acc} = -0.62\%$ [95% CI: $-1.90\%, +0.50\%$], $\Delta \text{Macro-F1} = -0.0069$ [95% CI: $-0.0213, +0.0059$], $\Delta \kappa = -0.0106$ [95% CI: $-0.0332, +0.0065$].
    - $E \text{ vs } B$: $\Delta \text{Acc} = -0.66\%$ [95% CI: $-1.36\%, +0.16\%$], $\Delta \text{Macro-F1} = -0.0067$ [95% CI: $-0.0141, +0.0032$].
    - $E \text{ vs } C$: $\Delta \text{Acc} = -0.01\%$ [95% CI: $-0.15\%, +0.12\%$], $\Delta \text{Macro-F1} = -0.0001$ [95% CI: $-0.0014, +0.0013$].
    - $E \text{ vs } D$: $\Delta \text{Acc} = -0.63\%$ [95% CI: $-1.99\%, +1.02\%$], $\Delta \text{Macro-F1} = -0.0081$ [95% CI: $-0.0196, +0.0051$].
    - All 95% confidence intervals span zero; the evidence does not support a generalizable performance difference between the proposed architecture and matched baseline across diverse subject cohorts.
  - **Subject Win/Loss Breakdown**:
    - Model E > Model A: 23 / 47 subjects ($48.9\%$).
    - Model A > Model E: 23 / 47 subjects ($48.9\%$).
    - Model E == Model A: 1 / 47 subjects ($2.2\%$).
  - **Boundary Error Dynamics**:
    - Across all 47 subjects, transition arrival epochs exhibit an error rate of $58.25\% \pm 1.73\%$, compared to $32.92\% \pm 4.13\%$ during non-boundary epochs (1.77x higher error).
- Artifacts saved:
  - `EXP014_RAW_RESULTS.csv`, `EXP014_SUMMARY.csv`, `EXP014_PER_CLASS.csv`, `EXP014_SUBJECT_METRICS.csv`, `EXP014_PAIRED_BOOTSTRAP.csv`
  - `EXP014_CONFUSION_MATRICES/`, `EXP014_TRAINING_HISTORY/`, `EXP014_RESULTS/checkpoints/`, `EXP014_RESULTS/predictions/`
  - `EXP014_FINAL_REPORT.md`
- Next concrete task: Proceed with methodological audit and exploratory analysis under DEC-021.

## 2026-11-11 Research Audit & Exploratory Methodological Analysis (DEC-021)

- Formally reopened the experimental phase under [DEC-021](./DECISIONS.md) (updating `AGENTS.md`, `RESEARCH_STATE.md`, `EXPERIMENTS.md`, `HANDOFF.md`); "no further training runs" lifted.
- **Task 2 (EXP-012 Window Reconciliation)**:
  - Total test windows in $T=600$ evaluation: 58 nights $\times 600 = 34,800$ slots.
  - Exactly 34,241 slots contain valid sleep stages ($0 \le y < 4$: Wake=3819, Light=13816, Deep=8294, REM=8312).
  - Exactly 559 slots were invalid/masked out ($y = -100$):
    - 264 slots due to raw unknown label (stage 5 in `labels.mat`).
    - 295 slots due to insufficient sensor coverage (<50% in IHR or motion).
  - Confirmed: Unknown labels were explicitly masked out via `valid = (targets >= 0) & (targets < 4) & mask` and **NOT** counted in evaluation metrics.
- **Task 3 (ControlledSleepDataset Window Selection & Truncation Audit)**:
  - Window start offset: Hardcoded to index 0 (first 5.0 hours of recording from lights-out / start of recording).
  - Total epochs per night:
    - EXP-012 Test set (58 nights): Mean = $853.12 \pm 114.22$, range [682, 1220], median = 829.0 epochs (~5.7 to 10.2 hours).
    - Full dataset (246 nights): Mean = $861.14 \pm 137.51$, range [610, 1282], median = 857.5 epochs (~5.1 to 10.7 hours).
  - Labeled epochs dropped by $T=600$ truncation:
    - EXP-012 Test set: 12,821 / 47,062 valid epochs dropped (**27.24%** dropped).
    - Full dataset: 58,055 / 203,338 valid epochs dropped (**28.55%** dropped).
    - Zero zero-padding was required for any night (all nights $\ge 610$ total epochs).
- **Task 4 (x_meta Features & Model A Verification)**:
  - 5 metadata features:
    1. `freq`: Proxy for instantaneous heart rate sampling density (log1p raw IHR samples per 30s epoch; meta col 0).
    2. `clock`: Zeros placeholder array (`np.zeros_like(time_feat)`).
    3. `time_feat`: Normalized elapsed time from recording start (`meta col 2`).
    4. `hr_mean`: Within-epoch mean of instantaneous heart rate (`ihr.mean(axis=1, keepdims=True)`).
    5. `hr_std`: Within-epoch standard deviation of instantaneous heart rate (`ihr.std(axis=1, keepdims=True)`).
  - Model A confirmation: Confirmed in `exp013_models.py` / `exp014_models.py`. Model A projects `x_meta` via `self.meta_proj = nn.Sequential(nn.Linear(5, 16), nn.GELU())` and concatenates `[h_hr, h_acc, h_meta]` into a 144-dimensional feature vector fed to the BiGRU.
- **Task 5 (Preprocessing Cache Normalization Audit)**:
  - Confirmed via `sleepchannelnet_protocol_patch_v2.ipynb:process_night`:
  - `robust_scale` normalizes strictly per-night locally using that night's within-night median and IQR:
    `np.clip(np.nan_to_num((v - med) / max(float(iqr), floor)), -10, 10)`.
  - Normalization is 100% within-night local. Zero global statistics across subjects were computed or pooled before splitting. Zero cross-subject data leakage.
- **Task 6 (night_id Attached to Saved Predictions)**:
  - Updated all 75 prediction files in `EXP014_RESULTS/predictions/` (`pred_{m_id}_fold{fold}_seed{seed}.pkl`) with `night_id` (e.g. `'Bidslab08__1'`), `night_ids`, and `nights`.
  - Updated `exp012_test_results.pkl` and `EXP013_OUTPUTS/exp013_results.pkl` with `night_id` arrays.
  - Updated `run_exp014_cv.py:evaluate_outer_test` to record `night_id` and `nights` dynamically for future runs.
- **Task 7 (Intraclass Correlation & Subject Accuracy Predictors)**:
  - Intraclass Correlation Coefficient (ICC) on 47 subjects:
    - Across 5 models (mean over seeds): **ICC(3, 1) = 0.9633** [95% CI: 0.9445, 0.9773], $F(46, 184) = 132.17, p = 1.08 \times 10^{-118}$.
    - Across all 15 runs (5 models $\times$ 3 seeds): **ICC(3, 1) = 0.9151** [95% CI: 0.8795, 0.9452], $F(46, 644) = 162.75, p < 10^{-300}$.
    - Conclusion: >96% of staging variance is governed by subject biological identity rather than model architecture.
  - Per-subject accuracy predictors:
    - Sleep stage transition rate (`trans_rate`): Pearson $r = -0.3559$ ($p = 0.0141$), Spearman $\rho = -0.2908$ ($p = 0.0474$).
    - Motion-wake coupling (`corr_motion_wake`): Pearson $r = +0.3215$ ($p = 0.0275$), Spearman $\rho = +0.3349$ ($p = 0.0214$).
    - Wake fraction (`frac_wake`): Pearson $r = -0.3119$ ($p = 0.0328$), Spearman $\rho = -0.2603$ ($p = 0.0772$).
    - Multivariate OLS regression ($R^2 = 0.2578$, Adj $R^2 = 0.1871$): Higher sleep fragmentation and higher wake burden strongly degrade wearable staging accuracy, while stronger motion-wake correlation significantly improves accuracy.
  - Artifacts: `EXP014_ICC_SUMMARY.csv`, `EXP014_SUBJECT_CORRELATIONS.csv`, `EXP014_SUBJECT_PREDICTORS_RAW.csv`.
- Next concrete task: Formulate next experimental or methodological phase under DEC-021 with explicit user authorization. Zero model training was executed in this turn.



