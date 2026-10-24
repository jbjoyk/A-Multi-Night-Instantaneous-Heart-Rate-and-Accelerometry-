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
  70/15/15 train/validation/test. Latest supplied artifact split fingerprint `a6b569f63126`; the older `459693836b56` manifest is unavailable here for direct partition comparison.
- Model configuration: four classes (Wake, Light, Deep, REM), expert labels, seed 42, complete
  graph primary model, 10 input nodes, 240-step sequences, batch size 16, 40 epoch maximum,
  patience 10, learning rate 1e-3.
- Historical training status: supervised comparisons and SleepJEPA pretraining/fine-tuning completed
  in the older recorded run. Revised code has not been executed.
- Checkpoint/artifact directory:
  `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL`
- Earlier historical metrics: primary SleepChannelNet accuracy 0.6932, Macro-F1 0.6851, κ 0.5529;
  JEPA fine-tuning accuracy 0.6411, Macro-F1 0.6325, κ 0.4844.
- Latest supplied run artifacts: accuracy 0.68659, Macro-F1 0.67791, κ 0.54722; Macro-F1
  subject-bootstrap 95% CI [0.63032, 0.71882], seven test subjects. See EXP-003; not independently reproduced.
- QC note: 134/246 eligible nights trigger the motion-energy/wake cross-correlation threshold or
  have undefined lag. This is not a timestamp-alignment test. Three raw-night spot checks showed no
  support for a global time shift; full audit remains incomplete.
- Next action: compare current and prior split membership if the older manifest is recovered; complete raw timestamp/coverage audit; reproduce the current baseline before publication.

## Notebook handling rules

- Preserve cell IDs, metadata, and useful outputs.
- Record exact changed cell IDs or execution counts after edits.
- Never assume saved outputs represent the current code.
- Mark a cell as verified only after it executes successfully in the documented environment.

## 2026-10-24 revision status

- Notebook now has 48 cells (21 code, 27 markdown). The previous JEPA implementation cell was removed from execution and replaced with a limitation statement.
- Modified cells: 0, 1, 5, 7, 13, 14, 19, 21, 27, 28-31, 34-36, 38, 44, 46, 47; appended cell IDs are `requested-ablation-notes` and `requested-ablation-suite`.
- The fixed subject split remains seed 0 and 70/15/15. Default baseline nodes remain the original 10. New 13-feature inputs require selecting the interaction ablation.
- New experiment toggles default off. No training or fresh data processing was run. Existing outputs in untouched cells are historical and do not verify the changed implementation; changed code outputs were cleared.
- Syntax check: all code cells compile with Python built-in `compile`; runtime, data/cache, model forward, and reporting are NOT VERIFIED. `nbformat` is not installed in the shell's default Python.
- Key limitations: GroupNorm has no BatchNorm running stats; unsupervised adaptation only entropy-minimizes the classifier head. HR-reactivity/coupling use look-ahead/centered windows and are retrospective. Exact SLAMSS-IFS reproduction is not implemented/verified.


## Preliminary audit update — 2026-10-24

- The dataset guide mapping is implemented: Unix signal timestamps are indexed against `recStart` in 30-second epochs, and naive `recStart` strings are localized to America/New_York using the documented standard-time ambiguity handling.
- The reported 134/246 flags are from the motion-energy/wake cross-correlation, not a timestamp alignment metric. The notebook now labels this distinction explicitly.
- Spot checks of Bidslab00/4, Bidslab01/3 and Bidslab07/5 found HR/motion starts near one another and near `recStart`; Bidslab01/3 has both streams ending early. No global timestamp shift is justified from this evidence.
- Current split hash is `a6b569f63126`; current manifest lists train/validation/test subjects 33/7/7. Hashing also includes dataset/preprocessing identity, so the older `459693836b56` hash alone does not prove the subject assignment changed.
- Full audit and independent reproduction remain outstanding.

- Latest artifact audit: split hash `a6b569f63126`, seed 0, partitions 33/7/7. The 134/246 flags are motion-wake correlation alerts, not confirmed timestamp shifts.
