# EXP-001: Baseline SleepChannelNet & Dataset Ingestion

## Overview
Foundational experiment establishing the baseline SleepChannelNet graph neural network pipeline on instantaneous heart rate and accelerometry, including initial data ingestion notebooks, protocol patches, and problem formulation documentation.

## Interface
- `sleepchannelnet_protocol_patch_v2.ipynb`: Jupyter notebook evaluating initial SleepChannelNet implementation and SleepJEPA pretraining.
- `sleepchannelnet_protocol_patch_v2_UPDATED.ipynb`: Updated protocol notebook auditing hyperparameter search and baseline evaluation.
- `Research Gaps.docx`: Problem definition and scientific gap analysis for wearable sleep staging.

## Invariants
- 4-class classification targets: Wake (0), Light (1), Deep (2), REM (3).
- Subject-level split isolation maintained across all notebook cells.

## Side Effects
- Loads raw dataset recordings from PhysioNet Bidslab directory.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_dataset_gate.py`
