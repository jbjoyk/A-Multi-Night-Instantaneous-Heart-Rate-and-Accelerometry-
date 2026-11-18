# EXP-002: SleepJEPA Pretraining, Feature Engineering, & SE Ablation Scaffold

> [!NOTE]
> **Governance & Naming Note:** In `EXPERIMENTS.md` and `DECISIONS.md` (DEC-004), EXP-002 formally designates the SE-Net channel attention, HR-motion interaction features, transition penalty, and unsupervised personalization implementation scaffold (authored Oct 24, 2026; unexecuted on Oct 24; post-hoc executed against Cohort Alpha on Oct 28). The SleepJEPA pretraining comparison described below was evaluated as part of the initial multi-architecture benchmark in EXP-001 (cells 17–21 of `experiments/EXP-001/sleepchannelnet_protocol_patch_v2.ipynb`).

## Overview
Self-supervised masked representation learning (SleepJEPA) pretraining on instantaneous heart rate and accelerometry, evaluating whether masked modeling improves downstream 4-class sleep staging over supervised learning. Negative result documented in `EXPERIMENTS.md` (EXP-001 / DEC-004), where supervised graph architectures outperformed SleepJEPA on this dataset.

## Interface
- Historical execution cells 17–21 in `experiments/EXP-001/sleepchannelnet_protocol_patch_v2.ipynb`.
- Verification metrics: Accuracy: 0.6411, Macro-F1: 0.6325, Cohen's Kappa: 0.4844.

## Invariants
- Pretrained encoder weights frozen during initial downstream probing, followed by full fine-tuning.
- 4-class stage targets: Wake (0), Light (1), Deep (2), REM (3).

## Side Effects
- Loads raw Bidslab recording data during pretraining.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_dataset_gate.py`
