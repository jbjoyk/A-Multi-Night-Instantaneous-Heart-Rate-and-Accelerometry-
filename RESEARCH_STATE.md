# Research State

Last updated: 2026-10-23

## Research question

Can real Apple Watch instantaneous heart rate and accelerometry signals support four-class sleep
staging against EEG-derived labels for healthy-population longitudinal sleep monitoring, and do
opt-in channel-attention, HR-motion, transition-aware, and unsupervised-personalization experiments
improve on the recorded supervised baseline? The previous SleepJEPA experiment did not improve over
supervised learning and is a documented negative result, not an active contribution.

## Research objectives

- Investigate wearable-based sleep staging using heart-rate and accelerometry signals.
- Preserve and document reproducible research progress.
- Maintain a reliable handoff context for different coding agents.

## Current research direction

The active notebook retains the real-data supervised SleepChannelNet comparisons and adds opt-in
SE attention, HR-motion interactions, transition regularization/Viterbi, and unsupervised head
entropy adaptation. The primary task is four-class classification: Wake, Light, Deep, and REM.
Revised experiments have not been run. Exact SLAMSS-IFS reproduction remains unimplemented and
unverified; the notebook's GAT-free baseline is explicitly an architectural proxy.

## Dataset information

The notebook used the real dataset at the configured local path and discovered 253 nights from
47 subjects. Raw label counts were `{0: 23850, 1: 15414, 2: 80881, 3: 39908, 4: 55861,
5: 2363}`, with 1.08% unknown labels. The raw `Bidslab*` recordings are intentionally excluded
from Git because they are approximately 28 GB. See the dataset README and `SHA256SUMS.txt` for
the available documentation and integrity information.

## Model architecture

The recorded baseline is SleepChannelNet with channel attention and a complete graph configuration,
using 10 registered nodes/features, a bidirectional GRU temporal encoder, frequency/time
features, embedding dimension 64, 4 attention heads, 2 graph-attention layers, hidden size 96,
dropout 0.3, and 240-step sequences. The notebook also retains identity/no-graph, early-fusion
CNN-GRU, and GBM baselines. New model/configuration paths are unverified and opt-in. Existing
architectures use GroupNorm, so the unsupervised adaptation path entropy-minimizes the head and
cannot update BatchNorm running statistics.

## Current implementation

- Notebook: `sleepchannelnet_protocol_patch_v2.ipynb`
- Python dependencies: `requirements.txt`
- Notebook kernel: Python 3.14.7; PyTorch 2.14.1+cu130; CUDA device detected with 2 GPUs.
- Additional research documents: `Problem Statement.docx`, `Research Gaps.docx`, and
  `Datasets Informations.docx`

## Historical completed tasks (recorded 2026-10-23)

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
- QC reports 134/246 eligible nights with absolute motion-energy/wake cross-correlation lag >4
  epochs or undefined lag. This is a behavioral correlation diagnostic, not evidence of timestamp
  misalignment. A three-night raw timestamp spot check found no basis for a global shift; it is not
  an exhaustive audit. Seven nights were excluded for low usable signal fraction.
- Raw recordings are not included in the GitHub repository.

## Unverified assumptions

- Saved notebook outputs are evidence of the recorded run, but should be reproduced before making
  publication claims.
- The configured work directory is
  `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL`.

## Current experiment and training status

The older recorded run completed, with historical metrics listed below and in
[EXPERIMENTS.md](./EXPERIMENTS.md). At the time of the initial implementation note, revised-source
experiments had not run. The latest revised-source validation search and one frozen test evaluation
are now recorded below and in EXP-004. Treat the historical values as results from their own saved
run, not as reproduced results from the revised source.

## Latest historical recorded results

The latest supplied run artifacts (split hash `a6b569f63126`) report accuracy 0.68659, Macro-F1
0.67791 and κ 0.54722; see EXP-003. The earlier run reported below remains EXP-001. These are
separate saved runs, not independent reproductions.

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

Preserve the current split manifest and compare its 33/7/7 subject membership with the earlier
manifest if available. Complete a raw timestamp/coverage audit for flagged and long-gap nights;
do not shift timestamps based on motion/wake correlation. Then independently reproduce this exact
baseline and only afterward run validation-selected ablations.

## Notebook revision note (2026-10-24)

The notebook has an implementation revision adding an SE model, optional HR-motion features, transition regularization/Viterbi decoding, unsupervised head-entropy adaptation, and revised training controls. A validation-only optimizer search and one frozen final test evaluation are now recorded in EXP-004. The previously recorded results above remain historical results from a separate run. Exact SLAMSS-IFS reproduction remains unimplemented/unverified; the project scope is healthy-population longitudinal sleep monitoring.


The supplied latest baseline run and preliminary timestamp QC audit are recorded in EXP-003. The 134-night value is a motion-wake correlation flag, not a confirmed alignment failure.

## Latest revised-source run (2026-10-24)

- Preserved subject split: seed 0, 33/7/7 subjects, split hash `a6b569f63126`.
- Validation-selected training configuration: AdamW, lr 0.001, Plateau scheduler, grouped weight decay 0.01, dropout 0.2; confirmation validation Macro-F1 0.678384.
- Frozen final test, evaluated once: accuracy 0.677715, Macro-F1 0.664803, kappa 0.531064; 7 test subjects, 38 nights, 33,126 known epochs. Subject-bootstrap CIs and detailed reporting are in EXP-004.
- This revised result is below the supplied historical pooled Macro-F1 of 0.67791. No improvement is claimed. Requested ablations and personalization remain unrun.
