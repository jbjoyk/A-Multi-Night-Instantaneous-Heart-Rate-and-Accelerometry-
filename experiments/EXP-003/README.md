# EXP-003: Hyperparameter and Optimization Search

- **Status**: Completed (Documented in historical notebook and EXPERIMENTS.md)
- **Provenance**: Executed in `experiments/EXP-001/sleepchannelnet_protocol_patch_v2_UPDATED.ipynb` (Cell 23).
- **Split Hash**: `a6b569f63126` (Seed 0, 33 train / 7 val / 7 test).
- **Metrics**: Accuracy: 0.68659, Macro-F1: 0.67791, Cohen's Kappa: 0.54722.
- **Outcome**: Tuned AdamW learning rate, grouped weight decay, dropout, and learning rate scheduler on validation cohort.
