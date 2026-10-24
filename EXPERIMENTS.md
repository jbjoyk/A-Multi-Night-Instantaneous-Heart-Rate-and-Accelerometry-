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
