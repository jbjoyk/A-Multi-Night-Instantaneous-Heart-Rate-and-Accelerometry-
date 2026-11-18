# EXP-005: SLAMSS-IFS Baseline Reproduction & Audit

## Overview
Independent scientific reproduction and audit of the published IEEE TBME 2026 SLAMSS-IFS architecture (`BIDSLabUMass/SLAMSS-IFS`). Re-implements and benchmarks intra-epoch multi-scale CNN feature extraction, intra-epoch BiLSTM representations, Bahdanau additive attention, inter-epoch Seq2Seq temporal context, and class-weighted loss (RCCLoss/RWLoss).

## Interface
- `SLAMSS_IFS` (`slamss_ifs_baseline.py`): Full dual-modality Seq2Seq model with Bahdanau attention and recurrent encoder-decoder.
- `IntraEpochCNN` (`slamss_ifs_baseline.py`): Multi-scale 1D convolutional feature extractor for heart rate and accelerometry signals.
- `IntraEpochBiLSTM` (`slamss_ifs_baseline.py`): Bidirectional recurrent context layer over intra-epoch features.
- `BahdanauAttention` (`slamss_ifs_baseline.py`): Attention alignment between decoder queries and encoder annotations.
- `InterEpochSeq2Seq` (`slamss_ifs_baseline.py`): Inter-epoch recurrent sequence-to-sequence model.
- `RWLoss` (`slamss_ifs_baseline.py`): Relative cross-entropy loss with inverse frequency class weighting.
- `make_reproducible_split` (`slamss_ifs_baseline.py`): Deterministic patient-level train/val/test data splitting.
- `SLAMSSDataset` (`slamss_ifs_train.py`): PyTorch Dataset collating 30-second multi-night sensor epochs.
- `train_slamss_ifs` (`slamss_ifs_train.py`): Training loop with early stopping on validation Macro-F1.

## Invariants
- Input signal layout strictly formatted as `[batch_size, sequence_length, channels, epoch_length]` (30s at 1Hz or 4Hz).
- 4-class target domain: Wake (0), Light (1), Deep (2), REM (3).
- Data splits are strictly partitioned by Subject ID to prevent inter-night cross-contamination.

## Side Effects
- Loads sensor recordings from PhysioNet dataset files.
- Writes training checkpoints (`slamss_ifs_short_run.pt`, `slamss_smoke_test.pt`).

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_evaluator.py tests/test_dataset_gate.py`
