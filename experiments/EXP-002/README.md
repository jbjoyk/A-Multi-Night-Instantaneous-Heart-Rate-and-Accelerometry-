# EXP-002: SleepJEPA Pretraining and Fine-Tuning

- **Status**: Completed (Documented in historical notebook and EXPERIMENTS.md)
- **Provenance**: Executed in `experiments/EXP-001/sleepchannelnet_protocol_patch_v2.ipynb` (Cells 17-21).
- **Hypothesis**: Self-supervised masked representation learning (SleepJEPA) pretraining on instantaneous heart rate and accelerometry will improve downstream 4-class sleep staging accuracy and Macro-F1 over supervised learning.
- **Results**: Accuracy: 0.6411, Macro-F1: 0.6325, Cohen's Kappa: 0.4844.
- **Outcome**: Negative result documented in EXPERIMENTS.md (DEC-002). Supervised graph network outperformed SleepJEPA on this dataset.
