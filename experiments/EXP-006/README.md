# EXP-006: SleepAdaptiveNet Modular Implementation

## Overview
Defines the `SleepAdaptiveNet` architecture (~162k parameters, 33.7x smaller than official SLAMSS-IFS). Combines dual-branch CNN encoders for instantaneous heart rate and accelerometry, adaptive reliability gating, bidirectional GRU temporal context, gradient reversal domain adaptation (GRL) for inter-subject alignment, and transition-aware boundary loss.

## Interface
- `SleepAdaptiveNet` (`sleep_adaptive_net.py`): Full modular multi-branch sleep staging architecture.
- `HREncoder` (`sleep_adaptive_net.py`): Multi-scale 1D CNN extracting intra-epoch instantaneous heart-rate features.
- `ACCEncoder` (`sleep_adaptive_net.py`): Multi-scale 1D CNN extracting intra-epoch accelerometry motion dynamics.
- `AdaptiveReliabilityFusion` (`sleep_adaptive_net.py`): Gated fusion dynamically weighting modality confidence.
- `GradientReversalLayer` (`sleep_adaptive_net.py`): Domain-adversarial reversal layer for subject invariant representations.
- `TransitionAwareModule` (`sleep_adaptive_net.py`): Detects and regularizes sleep stage boundary transitions.
- `SubjectDiscriminator` (`sleep_adaptive_net.py`): Adversarial classifier predicting subject ID from shared latents.
- `build_ablation_model` (`sleep_adaptive_net.py`): Parameterized factory instantiating ablation configurations (A0–A5).
- `run_smoke_training` (`run_ablation_smoke.py`): Quick execution validation across simulated epochs.

## Invariants
- Parameter budget strictly capped at ~162k parameters.
- Reversal scaling $\lambda$ scheduled monotonically from 0 to 1 over training epochs.
- Cross-modal attention weights sum to 1.0 across heart rate and accelerometry modalities.

## Side Effects
- Allocates CUDA device memory on designated GPU (`cuda:0` or `cuda:1`).

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_evaluator.py tests/test_dataset_gate.py`
