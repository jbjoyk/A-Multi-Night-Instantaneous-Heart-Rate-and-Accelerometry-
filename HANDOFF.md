# Research Handoff

## Current task

Update and validate the revised wearable sleep-staging notebook and its opt-in ablations. No new
training results have been established.

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
