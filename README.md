# Sleep Staging Research Project

This repository contains research materials for investigating sleep staging using instantaneous
heart rate, accelerometry, and EEG sleep-stage labels.

The research scope is healthy-population longitudinal sleep monitoring using wearable heart rate and
accelerometry against EEG-derived stage labels. The active notebook retains the recorded
SleepChannelNet baseline and adds opt-in SE channel attention, HR-motion features, transition
constraints, and unsupervised personalization experiments. The prior SleepJEPA run did not improve
over supervised learning and is documented as a negative result, not an active contribution.

The revised experiment code has not been run. The previously recorded metrics belong to the older
notebook execution and are not results for the revised implementation. Exact SLAMSS-IFS reproduction
also remains unimplemented and unverified; the notebook labels its GAT-free baseline as a proxy.

## Repository contents

- `sleepchannelnet_protocol_patch_v2.ipynb` — active real-data-only notebook with opt-in ablations;
  execution and verification state is tracked in [NOTEBOOK_STATE.md](./NOTEBOOK_STATE.md).
- `requirements.txt` — Python dependencies.
- `a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0/`
  — dataset documentation, license, and checksums. Raw recordings are excluded from Git.
- Research background documents in DOCX and PDF formats.

## Dataset

The project uses the A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG
Sleep Stage Labels. The raw data includes physiological readings synchronized with EEG-derived
sleep-stage labels.

Raw `Bidslab*` recordings are approximately 28 GB and are not stored in this Git repository.
Provision them separately and verify their checksums using the supplied `SHA256SUMS.txt` when
applicable. Do not commit private or restricted data.

## Shared agent memory

All coding agents should start with [AGENTS.md](./AGENTS.md), then read:

- [RESEARCH_STATE.md](./RESEARCH_STATE.md)
- [HANDOFF.md](./HANDOFF.md)
- [NOTEBOOK_STATE.md](./NOTEBOOK_STATE.md)
- [DECISIONS.md](./DECISIONS.md)
- [EXPERIMENTS.md](./EXPERIMENTS.md)

## Environment and notebook workflow

Create or activate the configured project environment, install `requirements.txt`, and inspect the
notebook before running it. New ablation toggles default off. Do not treat saved outputs in unchanged
cells as results for the revised code or start expensive training automatically. Record execution
counts, changed cells, configuration, and verification status in `NOTEBOOK_STATE.md` and
`EXPERIMENTS.md`.

## Git workflow

Use focused commits and do not commit `.venv`, caches, checkpoints, generated outputs, or raw
dataset recordings. Update `HANDOFF.md` before switching agents.
