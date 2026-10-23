# Research Handoff

## Current task

Establish a persistent, provider-neutral research memory system for the sleep-staging project.

## Previous work

- Initialized Git on `main`.
- Added a `.gitignore` for local environments, caches, checkpoints, and raw dataset recordings.
- Created the initial project commit.
- Created the shared research-state files in this handoff.

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
- Model training: NOT RUN.
- Research metrics: NOT VERIFIED.

## Current errors

None recorded. The raw dataset is not included in the Git commit because it is approximately
28 GB and is excluded by `.gitignore`.

## Next concrete task

Inspect `sleepchannelnet_protocol_patch_v2.ipynb` with a notebook-aware tool, record its cells,
execution counts, important variables, dataset paths, and any verified outputs in
`NOTEBOOK_STATE.md`. Do not start expensive training.

## Relevant paths

- `sleepchannelnet_protocol_patch_v2.ipynb`
- `requirements.txt`
- `a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0/README.md`
- `a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0/SHA256SUMS.txt`

