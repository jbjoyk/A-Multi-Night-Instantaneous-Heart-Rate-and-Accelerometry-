# Notebook State

## Active notebook

### `sleepchannelnet_protocol_patch_v2.ipynb`

- Purpose: Real-data-only comparison of supervised SleepChannelNet baselines and SleepJEPA
  self-supervised pretraining for four-class sleep staging.
- Notebook structure: 46 total cells, 21 code cells, 25 markdown cells.
- Execution counts: all 21 code cells have execution counts 1 through 21; 13 code cells have
  saved outputs.
- Cells modified since last verified execution: UNKNOWN from the attached snapshot.
- Cells requiring execution: None for the recorded run; reproduce before publication claims.
- Kernel: Python 3.14.7, kernel display name
  `02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels`.
- Runtime: PyTorch 2.14.1+cu130, CUDA selected, 2 GPUs detected.
- Dataset configuration: 253 nights, 47 subjects, one deterministic 33/7/7 subject split,
  70/15/15 train/validation/test, split fingerprint `459693836b56`.
- Model configuration: four classes (Wake, Light, Deep, REM), expert labels, seed 42, complete
  graph primary model, 10 input nodes, 240-step sequences, batch size 16, 40 epoch maximum,
  patience 10, learning rate 1e-3.
- Training status: supervised comparisons, SleepJEPA pretraining, and SleepJEPA fine-tuning
  completed in the recorded run.
- Checkpoint/artifact directory:
  `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL`
- Latest verified metrics: primary SleepChannelNet test accuracy 0.6932, macro-F1 0.6851,
  kappa 0.5529; SleepJEPA fine-tuning accuracy 0.6411, macro-F1 0.6325, kappa 0.4844.
- Known execution issues: 134/246 eligible nights triggered the alignment diagnostic threshold
  or had an undefined lag check. This requires review against the official documentation.
- Next action: review timestamp alignment and reproduce the primary comparison before publication.

## Notebook handling rules

- Preserve cell IDs, metadata, and useful outputs.
- Record exact changed cell IDs or execution counts after edits.
- Never assume saved outputs represent the current code.
- Mark a cell as verified only after it executes successfully in the documented environment.
