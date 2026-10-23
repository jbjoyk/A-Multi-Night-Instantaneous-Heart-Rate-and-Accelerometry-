# Research Handoff

## Current task

Establish a persistent, provider-neutral research memory system for the sleep-staging project.

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
- Notebook execution: NOT RUN.
- Notebook execution: 21 code cells executed with saved outputs.
- Model training: supervised comparisons and SleepJEPA pretraining/fine-tuning completed.
- Research metrics: recorded notebook outputs, not independently reproduced.

## Current errors

The notebook reports an alignment diagnostic warning for 134/246 eligible nights. The raw dataset
is not included in the Git commit because it is approximately 28 GB and is excluded by
`.gitignore`.

## Next concrete task

Review the alignment warning against the official dataset documentation, then reproduce the
primary complete-graph SleepChannelNet comparison using the recorded split before making
publication claims.

## Relevant paths

- `sleepchannelnet_protocol_patch_v2.ipynb`
- `requirements.txt`
- `a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0/README.md`
- `a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0/SHA256SUMS.txt`
