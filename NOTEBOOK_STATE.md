# Notebook State

## Active notebook

### `sleepchannelnet_protocol_patch_v2.ipynb`

- Purpose: Real-data-only comparison of recorded supervised baselines plus opt-in SE attention,
  HR-motion, transition-aware, and unsupervised-personalization experiments for four-class staging.
- Notebook structure after revision: 48 total cells, 21 code cells, 27 markdown cells.
- Historical execution counts: all 21 code cells in the earlier recorded revision had execution
  counts 1 through 21; 13 had saved outputs. The revised code cells have not been executed, and
  changed cells have had stale outputs cleared.
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
- Historical training status: supervised comparisons and SleepJEPA pretraining/fine-tuning completed
  in the older recorded run. Revised code has not been executed.
- Checkpoint/artifact directory:
  `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL`
- Latest historical metrics: primary SleepChannelNet test accuracy 0.6932, macro-F1 0.6851,
  kappa 0.5529; SleepJEPA fine-tuning accuracy 0.6411, macro-F1 0.6325, kappa 0.4844. These
  outputs belong to the older code and do not verify the revision.
- Known execution issues: 134/246 eligible nights triggered the alignment diagnostic threshold
  or had an undefined lag check. This requires review against the official documentation.
- Next action: review timestamp alignment and reproduce the primary comparison before publication.

## Notebook handling rules

- Preserve cell IDs, metadata, and useful outputs.
- Record exact changed cell IDs or execution counts after edits.
- Never assume saved outputs represent the current code.
- Mark a cell as verified only after it executes successfully in the documented environment.

## 2026-10-24 revision status

- Notebook now has 48 cells (21 code, 27 markdown). The previous JEPA implementation cell was removed from execution and replaced with a limitation statement.
- Modified cells: 0, 1, 5, 7, 14, 19, 21, 27, 28-31, 34-36, 38, 44, 46, 47; appended cell IDs are `requested-ablation-notes` and `requested-ablation-suite`.
- The fixed subject split remains seed 0 and 70/15/15. Default baseline nodes remain the original 10. New 13-feature inputs require selecting the interaction ablation.
- New experiment toggles default off. No training or fresh data processing was run. Existing outputs in untouched cells are historical and do not verify the changed implementation; changed code outputs were cleared.
- Syntax check: all code cells compile with Python built-in `compile`; runtime, data/cache, model forward, and reporting are NOT VERIFIED. `nbformat` is not installed in the shell's default Python.
- Key limitations: GroupNorm has no BatchNorm running stats; unsupervised adaptation only entropy-minimizes the classifier head. HR-reactivity/coupling use look-ahead/centered windows and are retrospective. Exact SLAMSS-IFS reproduction is not implemented/verified.
