# Research State

Last updated: 2026-10-23

## Research question

Can real Apple Watch instantaneous heart rate and accelerometry signals support four-class sleep
staging against EEG-derived labels, and does SleepJEPA self-supervised latent-prediction
pretraining improve SleepChannelNet performance over a supervised baseline?

## Research objectives

- Investigate wearable-based sleep staging using heart-rate and accelerometry signals.
- Preserve and document reproducible research progress.
- Maintain a reliable handoff context for different coding agents.

## Current research direction

The active notebook runs a real-data-only pipeline comparing supervised SleepChannelNet variants,
an early-fusion CNN-GRU baseline, a gradient-boosted tree baseline, and SleepJEPA pretraining
followed by SleepChannelNet fine-tuning. The primary task is four-class classification:
Wake, Light, Deep, and REM.

## Dataset information

The notebook used the real dataset at the configured local path and discovered 253 nights from
47 subjects. Raw label counts were `{0: 23850, 1: 15414, 2: 80881, 3: 39908, 4: 55861,
5: 2363}`, with 1.08% unknown labels. The raw `Bidslab*` recordings are intentionally excluded
from Git because they are approximately 28 GB. See the dataset README and `SHA256SUMS.txt` for
the available documentation and integrity information.

## Model architecture

The primary model is SleepChannelNet with channel attention and a complete graph configuration,
using 10 registered nodes/features, a bidirectional GRU temporal encoder, frequency/time
features, embedding dimension 64, 4 attention heads, 2 graph-attention layers, hidden size 96,
dropout 0.3, and 240-step sequences. The notebook also evaluates identity/no-graph,
early-fusion CNN-GRU, and GBM baselines. SleepJEPA uses a transformer temporal encoder and
20 pretraining epochs.

## Current implementation

- Notebook: `sleepchannelnet_protocol_patch_v2.ipynb`
- Python dependencies: `requirements.txt`
- Notebook kernel: Python 3.14.7; PyTorch 2.14.1+cu130; CUDA device detected with 2 GPUs.
- Additional research documents: `Problem Statement.docx`, `Research Gaps.docx`, and
  `Datasets Informations.docx`

## Completed tasks

- Git repository initialized on the `main` branch.
- Initial project materials committed locally.
- Shared research-memory documentation created.
- Raw dataset recordings excluded from the normal Git commit.
- Notebook executed through its final banner with all 21 code cells executed.
- Seven real-data integrity checks passed; no synthetic data was generated.
- SleepJEPA pretraining and fine-tuning completed.

## Current problems

- Results are from one deterministic subject-level split and seed 42; they are not cross-validation
  results.
- The notebook reports a substantial label/motion alignment diagnostic warning: 134 of 246
  eligible nights exceeded the lag threshold or had an undefined lag check. This diagnostic was
  not used for model selection and requires documentation review.
- Raw recordings are not included in the GitHub repository.

## Unverified assumptions

- Saved notebook outputs are evidence of the recorded run, but should be reproduced before making
  publication claims.
- The configured work directory is
  `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL`.

## Current experiment and training status

Completed for the recorded run. The primary supervised model achieved test accuracy 0.6932,
macro-F1 0.6851, and kappa 0.5529. SleepJEPA fine-tuning achieved accuracy 0.6411,
macro-F1 0.6325, and kappa 0.4844. The frozen JEPA validation probe achieved macro-F1 0.4508.
See [EXPERIMENTS.md](./EXPERIMENTS.md) for the full comparison and caveats.

## Latest verified results

Recorded run results:

- Complete-graph SleepChannelNet: accuracy 0.6932, macro-F1 0.6851, kappa 0.5529.
- Identity/no-graph SleepChannelNet: accuracy 0.6658, macro-F1 0.6599, kappa 0.5173.
- CNN-GRU early fusion: accuracy 0.6683, macro-F1 0.6462, kappa 0.5069.
- GBM epoch features plus context: accuracy 0.5990, macro-F1 0.5952, kappa 0.4229.
- SleepJEPA pretraining plus fine-tuning: accuracy 0.6411, macro-F1 0.6325, kappa 0.4844.

These results are verified as notebook outputs for the recorded run, not independently reproduced.

## Important constraints

- Do not change the methodology, architecture, dataset, training configuration, or experimental
  results without explicit approval.
- Do not fabricate metrics or claim that a notebook cell executed when it has not been verified.
- Do not commit the raw dataset recordings or local environment files.

## Next research milestone

Review the 134-night alignment diagnostic against the official dataset documentation, preserve
the recorded split and artifacts, and independently reproduce the primary comparison before
using the metrics in a paper or claim.
