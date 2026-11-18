# experiments

## Overview
The `experiments` subsystem encapsulates all isolated research experiments (EXP-001 through EXP-019), tracking baseline reproductions, modular network designs, hyperparameter optimization, factorial ablations, multi-night adaptation protocols, and autonomous Neural Architecture Search (NAS) continuous loops.

## Subdirectories & Boundaries
- [EXP-001](./EXP-001/README.md) - Baseline SleepChannelNet graph neural network and data ingestion notebooks.
- [EXP-002](./EXP-002/README.md) - SleepJEPA self-supervised representation learning pretraining and fine-tuning.
- [EXP-003](./EXP-003/README.md) - Hyperparameter and optimization search on the validation cohort.
- [EXP-004](./EXP-004/README.md) - Frozen test set baseline evaluation for SleepChannelNet.
- [EXP-005](./EXP-005/README.md) - Independent scientific reproduction and audit of official IEEE TBME 2026 SLAMSS-IFS.
- [EXP-006](./EXP-006/README.md) - SleepAdaptiveNet compact dual-branch modular architecture with dynamic reliability gating.
- [EXP-007](./EXP-007/README.md) - Controlled ablation study (A0–A5) isolating individual architectural components on Seed 42.
- [EXP-008](./EXP-008/README.md) - Boundary loss weight sweep and GRL disentanglement on validation data.
- [EXP-009](./EXP-009/README.md) - Multi-seed stability comparison (B0_GRL vs B2) across seeds 42, 123, 2024.
- [EXP-010](./EXP-010/README.md) - Extended 500-epoch convergence benchmark and overfitting analysis.
- [EXP-011](./EXP-011/README.md) - Multi-seed 500-epoch convergence audit across random weight initializations.
- [EXP-012](./EXP-012/README.md) - Frozen test set benchmark across 7 subjects, confusion matrices, and audit dossiers.
- [EXP-013](./EXP-013/README.md) - Factorial component attribution study across 34,241 held-out test windows (Models A–E).
- [EXP-014](./EXP-014/README.md) - Definitive 75-run 5-fold cross-validation suite and autonomous NAS continuous iteration loop.
- [EXP-015](./EXP-015/README.md) - Circadian/clock feature pruning simulation and protocol specification.
- [EXP-016](./EXP-016/README.md) - Governed live execution smoke test under automated safety controller.
- [EXP-017](./EXP-017/README.md) - Sequence horizon evaluation comparing standard ($T=600$) vs extended ($T \le 1200$) sequences.
- [EXP-018](./EXP-018/README.md) - Multi-night within-person adaptation protocol (AUTORESEARCH) with invariant token exposure.
- [EXP-019](./EXP-019/README.md) - 500-epoch extended training benchmark comparing Baseline Model A against MultiScale Model A.
