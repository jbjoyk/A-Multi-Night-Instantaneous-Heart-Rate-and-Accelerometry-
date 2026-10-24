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
- Revised notebook execution: NOT RUN.
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
