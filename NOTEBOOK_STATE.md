# Notebook State

## Active notebook

### `sleepchannelnet_protocol_patch_v2.ipynb`

- Purpose: Real-data-only comparison of recorded supervised baselines plus opt-in SE attention,
  HR-motion, transition-aware, and unsupervised-personalization experiments for four-class staging.
- Notebook structure after latest update: 50 total cells, 22 code cells, 28 markdown cells.
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
  in the older recorded run. This revision has since received a controlled validation search and
  one frozen final test evaluation; see the dated status below and EXP-004.
- Checkpoint/artifact directory:
  `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL`
- Earlier historical metrics: primary SleepChannelNet accuracy 0.6932, Macro-F1 0.6851, κ 0.5529;
  JEPA fine-tuning accuracy 0.6411, Macro-F1 0.6325, κ 0.4844.
- Latest supplied run artifacts: accuracy 0.68659, Macro-F1 0.67791, κ 0.54722; Macro-F1
  subject-bootstrap 95% CI [0.63032, 0.71882], seven test subjects. See EXP-003; not independently reproduced.
- QC note: 134/246 eligible nights trigger the motion-energy/wake cross-correlation threshold or
  have undefined lag. This is not a timestamp-alignment test. Three raw-night spot checks showed no
  support for a global time shift; full audit remains incomplete.
- Next action: compare current and prior split membership if the older manifest is recovered; complete raw timestamp/coverage audit; then run individual requested ablations if needed.

## Notebook handling rules

- Preserve cell IDs, metadata, and useful outputs.
- Record exact changed cell IDs or execution counts after edits.
- Never assume saved outputs represent the current code.
- Mark a cell as verified only after it executes successfully in the documented environment.

## 2026-10-24 initial revision status (before controlled run)

- Before the run-controls addition, the notebook had 48 cells (21 code, 27 markdown). The previous JEPA implementation cell was removed from execution and replaced with a limitation statement.
- Modified cells: 0, 1, 5, 7, 13, 14, 19, 21, 27, 28-31, 34-36, 38, 44, 46, 47; appended cell IDs are `requested-ablation-notes` and `requested-ablation-suite`.
- The fixed subject split remains seed 0 and 70/15/15. Default baseline nodes remain the original 10. New 13-feature inputs require selecting the interaction ablation.
- New experiment toggles default off. No training or fresh data processing was run. Existing outputs in untouched cells are historical and do not verify the changed implementation; changed code outputs were cleared.
- Syntax check: all code cells compile with Python built-in `compile`; runtime, data/cache, model forward, and reporting are NOT VERIFIED. `nbformat` is not installed in the shell's default Python.
- Key limitations: GroupNorm has no BatchNorm running stats; unsupervised adaptation only entropy-minimizes the classifier head. HR-reactivity/coupling use look-ahead/centered windows and are retrospective. Exact SLAMSS-IFS reproduction is not implemented/verified.

## 2026-10-24 validation and final-evaluation status

- Modified and recompiled source cell 21 (`_plain` now converts NumPy arrays for JSON reports), source cell id `requested-ablation-suite` (now cell 49; headless display and frozen-config orchestration), and markdown cells 20 and 46.
- Validation-only optimizer search: completed on split hash `a6b569f63126`, 12 rows including 10 screens and two full-epoch confirmations. Selected AdamW, lr 0.001, Plateau, grouped weight decay 0.01, dropout 0.2, full-context inference; confirmation validation Macro-F1 0.678384.
- Frozen final fit and test evaluation: one test inference pass, 7 subjects / 38 nights / 33,126 known epochs; accuracy 0.677715, Macro-F1 0.664803, kappa 0.531064. Subject-bootstrap CIs and per-class, per-night, clinical agreement summaries are in EXP-004 and the ignored run output directory.
- Runtime: Python 3.14.7, PyTorch 2.14.1+cu130, two RTX 3060 GPUs visible to the configured project environment. Feature cache validated 253/253 nights; 246 eligible.
- The first reporting attempt exposed a NumPy-array JSON serialization issue after `results.pkl` was persisted. After fixing the notebook serializer, reports were regenerated from the saved result without touching test inference a second time.
- Source cell id `requested-ablation-suite` (now cell 49) provides `RUN_FROZEN_FINAL_EVALUATION` and `USE_VALIDATION_SELECTED_OPTIMIZER`; it can load the saved validation selection and reuse it for frozen evaluation or ablations. The existing baseline row is included in the ablation suite.
- To make the controls discoverable, added markdown id `run-controls-heading` and code id `run-controls` near the top (cell 2). The controls are defined once there and consumed by the ablation runner. Current notebook structure: 50 cells, 22 code cells, 28 markdown cells. All 22 code cells pass syntax compilation; the new controls cell and final orchestration were not runtime-executed after this discoverability edit.
- Execution used a harness for source cells through data prep and model run functions; no single top-to-bottom Jupyter session was run. Ablations, permutation importance, personalization, and transition-lambda search remain unrun. The model did not improve over the supplied historical pooled Macro-F1; do not claim a gain.


## Preliminary audit update — 2026-10-24

- The dataset guide mapping is implemented: Unix signal timestamps are indexed against `recStart` in 30-second epochs, and naive `recStart` strings are localized to America/New_York using the documented standard-time ambiguity handling.
- The reported 134/246 flags are from the motion-energy/wake cross-correlation, not a timestamp alignment metric. The notebook now labels this distinction explicitly.
- Spot checks of Bidslab00/4, Bidslab01/3 and Bidslab07/5 found HR/motion starts near one another and near `recStart`; Bidslab01/3 has both streams ending early. No global timestamp shift is justified from this evidence.
- Current split hash is `a6b569f63126`; current manifest lists train/validation/test subjects 33/7/7. Hashing also includes dataset/preprocessing identity, so the older `459693836b56` hash alone does not prove the subject assignment changed.
- Full audit and independent reproduction remain outstanding.

- Latest artifact audit: split hash `a6b569f63126`, seed 0, partitions 33/7/7. The 134/246 flags are motion-wake correlation alerts, not confirmed timestamp shifts.


## 2026-10-24 optimizer and pipeline audit

- Source cells changed: 7 (optimizer controls), 14 (HR-motion feature derivation), 17 (feature cache fingerprint), 19 (interaction masking), 21 (optimizer/training/checkpoints, transition loss/Viterbi checks, raw-vs-decoded metrics and run identity), 27 (fixed split preparation; core training now opt-in), 33/37 (guarded core reports), 39 (validation-only grouped permutation), 43 (in-memory feature unit checks), and 47 (validation-only optimizer search and opt-in ablations). Markdown cells 20 and 46 record audit status and limitations. Cell metadata/cell order are retained. Outputs in source-modified cells were cleared so old result output cannot be mistaken for the new source.
- Local checks run: all 21 code cells compile; feature helper checks pass for dimensions/alignment, missing values, constants, numeric stability and per-night boundaries; transition loss and gradient are finite with unknown labels; Viterbi handles a one-epoch sequence.
- Training/runtime status: no notebook execution or model training was run. Sandbox Python cannot access CUDA; `nvidia-smi -L` outside the sandbox detected two RTX 3060 GPUs.
- Core comparison and optimizer search are separate opt-ins. `RUN_CORE_COMPARISON=False` and `RUN_OPTIMIZER_SEARCH=False` by default, so validation tuning can be run before any final test evaluation.
- Subject split seed and split-construction code were not modified. Feature cache version is separate from `PREP_VERSION`; no new split hash or result is asserted.

- Cache safety follow-up: `FEATURE_VERSION` is now part of the cache-directory identity as well as each per-night fingerprint. A preflight rebuild was stopped before candidate training after discovering the old path could be replaced. The ignored cache at `sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_f4769d6efe` may contain a mixture of regenerated and prior cache files from that interrupted attempt; saved result artifacts were not changed. Raw data remains the source of truth.
