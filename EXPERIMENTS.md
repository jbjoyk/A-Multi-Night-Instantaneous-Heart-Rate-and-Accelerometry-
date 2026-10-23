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
