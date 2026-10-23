# Sleep Staging Research Project

This repository contains research materials for investigating sleep staging using instantaneous
heart rate, accelerometry, and EEG sleep-stage labels.

The research objective is to evaluate wearable-based sleep staging and compare hybrid temporal
and attention-based models, including SleepChannelNet and experimental SleepJEPA pretraining.

## Repository contents

- `sleepchannelnet_protocol_patch_v2.ipynb` — active real-data-only research notebook; execution
  state is tracked in [NOTEBOOK_STATE.md](./NOTEBOOK_STATE.md).
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

Create or activate a project environment, install `requirements.txt`, and inspect the notebook
before running it. Do not start expensive training automatically. Record execution counts,
changed cells, configuration, and verified results in `NOTEBOOK_STATE.md` and `EXPERIMENTS.md`.

## Git workflow

Use focused commits and do not commit `.venv`, caches, checkpoints, generated outputs, or raw
dataset recordings. Update `HANDOFF.md` before switching agents.
