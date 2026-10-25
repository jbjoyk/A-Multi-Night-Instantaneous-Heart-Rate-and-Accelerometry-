# Experiments

The following experiment is recorded from saved notebook outputs. It is not independently
reproduced yet.

## EXP-001 — Real-data SleepChannelNet and SleepJEPA comparison

- Date: 2026-10-23 (recorded notebook execution)
- Research question: Whether real Apple Watch heart rate and accelerometry support four-class
  sleep staging, and whether SleepJEPA pretraining improves SleepChannelNet.
- Hypothesis: SleepJEPA pretraining may improve downstream performance; the notebook treats this
  as experimental and does not guarantee improvement.
- Dataset and split: 253 nights from 47 subjects; one deterministic subject-level split with
  33 train subjects / 7 validation subjects / 7 test subjects; split fingerprint `459693836b56`.
- Model architecture: Primary SleepChannelNet complete graph with channel attention; comparisons
  include identity/no-graph SleepChannelNet, early-fusion CNN-GRU, GBM, and SleepJEPA
  transformer pretraining followed by fine-tuning.
- Configuration: Four classes Wake/Light/Deep/REM, expert labels, seed 42, 10 registered input
  nodes, 30-second epochs at 1 Hz, 240-step sequences, batch size 16, maximum 40 epochs,
  patience 10, learning rate 1e-3. SleepJEPA pretraining used 20 epochs, mask ratio 0.25,
  transformer temporal encoder, and learning rate 1e-4.
- Random seed: Split seed 0; training seed 42.
- Hardware: CUDA, 2 GPUs detected.
- Training duration: Not recorded in the notebook state summary.
- Checkpoint path: Work/artifact directory
  `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL`.
- Evaluation metrics:
  - SleepChannelNet complete graph: accuracy 0.6932, macro-F1 0.6851, kappa 0.5529.
  - SleepChannelNet identity/no graph: accuracy 0.6658, macro-F1 0.6599, kappa 0.5173.
  - CNN-GRU early fusion: accuracy 0.6683, macro-F1 0.6462, kappa 0.5069.
  - GBM epoch features plus context: accuracy 0.5990, macro-F1 0.5952, kappa 0.4229.
  - SleepJEPA plus fine-tuning: accuracy 0.6411, macro-F1 0.6325, kappa 0.4844.
  - Frozen JEPA linear probe on validation: accuracy 0.4632, macro-F1 0.4508.
- Results: On this split, the complete-graph supervised model was stronger than the recorded
  SleepJEPA fine-tuning result. The graph-vs-no-graph difference was +0.0210 macro-F1 with
  95% subject-bootstrap CI [-0.0062, +0.0503] and Wilcoxon p=0.156.
- Failures: The notebook reports a SleepJEPA collapse detection and retry with stabilized
  settings before the completed run. 134/246 eligible nights exceeded the alignment diagnostic
  threshold or had an undefined lag check.
- Interpretation: SleepJEPA improvement was not demonstrated on this split. Personalization was
  also not demonstrated for k=1 or k=2. Do not generalize from this single split/seed.
- Verified: YES as saved notebook outputs; NO as an independent reproduction.

## Experiment recording template

For every experiment, add a section using this structure:

### EXP-XXX — Short description

- Date:
- Research question:
- Hypothesis:
- Dataset and split:
- Model architecture:
- Configuration:
- Random seed:
- Hardware:
- Python / PyTorch / CUDA versions:
- Training duration:
- Checkpoint path:
- Evaluation metrics:
- Results:
- Failures:
- Interpretation:
- Verified: YES / NO

Do not fill fields from stale notebook outputs without confirming their provenance and execution
state.

## EXP-002 — Requested SE, transitions, HR-motion and unsupervised-personalization implementation

- Date: 2026-10-24 (implementation only)
- Research question: Whether SE channel attention, HR-motion features, transition regularization/decoding, and label-free personalization improve four-class wearable sleep staging.
- Hypothesis: NOT TESTED.
- Dataset and split: The same real-data subject-level 70/15/15 split; split seed remains 0. No new data run was performed.
- Model architecture: Added `SleepChannelNet_SE` alongside the existing `SleepChannelNet`; baseline default remains the original ten features. Three interaction channels are registered for the opt-in 13-node variant.
- Configuration: New feature/transition/Viterbi flags added. Lambda candidates 0.05/0.10/0.20 are selected using training and validation only when the transition ablation is run. New ablation toggles default off.
- Random seed: No training run. Existing split seed 0 was not changed.
- Hardware: No experiment run.
- Python / PyTorch / CUDA versions: No experiment run; project-recorded environment remains Python 3.14.7 / PyTorch 2.14.1+cu130 / CUDA with 2 GPUs detected, not re-verified here.
- Training duration: Not applicable; NOT RUN.
- Checkpoint path: None created.
- Evaluation metrics: None generated for this implementation.
- Results: NOT RUN; no performance claim.
- Failures: Notebook code syntax was checked with Python's built-in compiler. Runtime behavior, cached data compatibility, and metric calculations are NOT VERIFIED.
- Interpretation: Implementation scaffold only. Exact SLAMSS-IFS reimplementation remains NOT IMPLEMENTED/VERIFIED. Existing architecture uses GroupNorm, so the unsupervised path cannot adapt BatchNorm running statistics; it optionally entropy-minimizes the prediction head. Feature engineering uses retrospective centered/forward windows and is not causal.
- Verified: NO (syntax only; no notebook execution or training).

## EXP-003 — Latest supplied complete-graph baseline artifacts and QC audit

- Date: 2026-10-24 (artifacts saved in `sleepchannelnet_v8_preprocessing_audit_work_REAL`)
- Research question: Recorded four-class supervised baseline performance and whether timestamp evidence supports a preprocessing shift.
- Hypothesis: NOT TESTED; this records existing results and a limited read-only QC audit.
- Dataset and split: Real data, 253 discovered nights, 246 eligible nights, 47 subjects. Split seed 0, 33/7/7 subjects, current split hash `a6b569f63126`. An older memory entry has hash `459693836b56`; the older manifest was not found in the searched project tree, so prior subject assignments cannot be directly compared. Split hashing includes dataset and preprocessing identity as well as subject assignments; differing hashes alone do not establish different subject membership.
- Model architecture: `channel_attn`, complete graph, original ten input nodes; no SE, HR-motion interaction channels, transition penalty, Viterbi decoding, or personalization. Config and output are in `summary.json` and `model_comparison.csv`.
- Configuration: seed 42; fixed subject split seed 0; 70/15/15; expert labels; `transition_lambda=0`; `use_viterbi_decoding=false`; one training seed.
- Random seed: training 42; split 0.
- Hardware / software: Recorded in reproducibility artifacts; not independently verified in this audit.
- Evaluation metrics: Pooled test accuracy 0.68659, Macro-F1 0.67791, Cohen's κ 0.54722. Subject-bootstrap 95% CIs: accuracy [0.63760, 0.72408], Macro-F1 [0.63032, 0.71882], κ [0.48339, 0.59387]; seven test subjects.
- Per-class results from the saved confusion matrix: Wake F1 0.6517 / recall 0.6379; Light F1 0.7048; Deep F1 0.6728 / recall 0.6114; REM F1 0.6823.
- Results: The complete-graph model exceeded the recorded identity/no-graph control by 0.02225 Macro-F1, but the paired subject-bootstrap CI [-0.06957, 0.01752] and Wilcoxon p=0.8125 do not show a reliable graph benefit on this split. The permutation-importance image is test-set descriptive evidence and must not be used for feature selection.
- QC findings: 134/246 eligible nights have absolute motion-energy/wake cross-correlation lag >4 epochs or an undefined lag (133 numeric, one undefined). This is not a clock-alignment test. Preliminary raw timestamp checks on Bidslab00/4, Bidslab01/3 and Bidslab07/5 show HR and motion streams start close to each other and close to `recStart`; Bidslab01/3 has both streams ending about 9,235 seconds before the label interval ends, consistent with incomplete coverage rather than a demonstrated uniform time shift. This is a three-night spot check, not an exhaustive audit.
- Failures: No new training or metric calculation was run. Seven nights were excluded by the run for low usable signal fraction.
- Interpretation: Do not shift timestamps based on the motion/wake correlation lag. The documented `recStart` epoch mapping remains in place; no global correction is supported by this spot check. Confirm old versus current split membership if the earlier hash is to be used for comparison.
- Verified: YES as saved artifacts; NO as an independent reproduction. Timestamp conclusion is preliminary and limited to the documented parser, QC summary and three raw-night spot checks.

## EXP-004 — Controlled optimizer search and frozen final evaluation

- Date: 2026-10-24
- Research question: Which AdamW learning rate, scheduler, weight decay, and dropout setting performs best on the fixed validation subjects?
- Hypothesis: NOT TESTED.
- Dataset and split: Configured for the same real dataset and existing seed-0 subject split; no split was regenerated by this code edit.
- Model architecture: Existing SleepChannelNet; no architecture change in this optimizer study.
- Configuration: Staged validation-only search: legacy AdamW reference; four learning rates; plateau versus cosine warm-up; three weight-decay values; dropout 0.2 versus 0.3; top two configurations confirmed at the 40-epoch maximum. Biases and normalization parameters were excluded from grouped decay. The original subject split was preserved and asserted before search and final evaluation.
- Random seed: Training seed 42; split seed 0. Epoch-specific window generation and shuffling were deterministic.
- Hardware: NVIDIA RTX 3060 x2, CUDA available to the project environment.
- Python / PyTorch / CUDA versions: Python 3.14.7, PyTorch 2.14.1+cu130 (CUDA 13.0 build).
- Training duration: Validation search candidates reported 2,298.1 cumulative candidate-training seconds (sum of per-candidate recorded durations); final fit 265.9 seconds (4.43 min). Final test inference 12.09 seconds.
- Checkpoint and result path: `outputs/sleep_optimizer_work_REAL/validation_optimizer_search/` and `outputs/sleep_optimizer_work_REAL/runs/4class_channel_attn_complete_s42_2814bfb16c/` (ignored local outputs).
- Search results: 12 candidate rows (10 screens and 2 confirmations). Selected by validation subjects only: AdamW, LR 0.001, Plateau, grouped weight decay 0.01, dropout 0.2, full-context inference. Confirmed best validation Macro-F1 = 0.678384 (best epoch 19). Split hash `a6b569f63126`.
- Frozen final test evaluation: Performed once after selection, on 7 test subjects / 38 nights / 33,126 known epochs. Accuracy 0.677715, Macro-F1 0.664803, kappa 0.531064. Subject-bootstrap 95% CIs: accuracy [0.6145, 0.7222], Macro-F1 [0.5974, 0.7151], kappa [0.4474, 0.5909]. Subject-mean accuracy 0.6740, Macro-F1 0.6601, kappa 0.5312. Per-class F1: Wake 0.6194, Light 0.7013, Deep 0.6678, REM 0.6707. Test metrics are lower than the supplied historical result (Macro-F1 0.67791); this is not evidence of improvement.
- Clinical agreement (post-processing of saved predictions, 38 test nights): TST bias +15.45 min (LoA -56.49 to +87.39); SE bias +3.64 percentage points (LoA -12.73 to +20.01); SOL bias -12.82 min (LoA -84.79 to +59.16); REM share bias +1.05 points (LoA -20.04 to +22.13); Deep share bias -4.67 points (LoA -20.83 to +11.49). These descriptive night-level agreements do not establish clinical reliability.
- Failures and recovery: The first optimizer harness reached its report display and exited because headless `display()` was undefined; the CSV/JSON summary had already been written. The final test pass persisted `results.pkl` and then exposed a JSON conversion error for NumPy arrays. `_plain` now converts arrays to lists, and the metrics/config/history files were recovered from the saved result without rerunning test inference. A prior separate preflight also hit an unavailable external cache path; no outputs from that attempt were used.
- Interpretation: This is an executed validation search and one frozen final evaluation, not proof the revised optimizer improves the model. Ablation suite, transition lambda sweep, personalization, and permutation importance were not run. Exact SLAMSS-IFS remains a proxy, not a reproduction.
- Post-run notebook refinement: source cell id `requested-ablation-suite` (now cell 49) exposes `RUN_FROZEN_FINAL_EVALUATION`, reuses a saved validation-selected configuration for the ablation suite, and includes the original complete-graph baseline row. These orchestration changes were syntax-checked but not runtime-executed after this experiment; the already-recorded search/test results are unchanged. A top-of-notebook code cell id `run-controls` (cell 2) now defines the toggles in one easy-to-find place.
- Verified: Candidate fits, saved validation search rows, final fit/test inference, metrics, and post-hoc summaries are verified. The search harness initially exited at its display statement after saving the CSV/JSON; a headless fallback was added afterward. The notebook was not run top-to-bottom as a Jupyter session. Post-run cell 49 orchestration and controls-cell refinements are syntax-checked only.
