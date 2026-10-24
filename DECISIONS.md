# Decisions

## DEC-001 — Use repository files as the shared research memory

- Date: 2026-10-23
- Decision: Store agent instructions, research state, handoffs, notebook state, decisions, and
  experiment history in version-controlled Markdown files.
- Reason: The researcher switches between multiple coding agents and needs a provider-neutral
  source of truth.
- Alternatives considered: Relying on agent conversation history or provider-specific memory.
- Expected consequences: Agents can recover project context from the repository.
- Reversible: Yes.

## DEC-002 — Exclude raw dataset recordings from normal Git commits

- Date: 2026-10-23
- Decision: Exclude the `Bidslab*` raw recordings from Git using `.gitignore`.
- Reason: The raw recordings occupy approximately 28 GB and are unsuitable for a normal GitHub
  repository.
- Alternatives considered: Committing the recordings directly or using Git LFS.
- Expected consequences: The repository contains documentation and code, while raw data must be
  provisioned separately.
- Reversible: Yes, subject to storage and repository limits.

## DEC-003 — Do not infer unverified research facts

- Date: 2026-10-23
- Decision: Mark unknown research questions, architecture, execution state, and metrics as
  `UNKNOWN` or `NOT VERIFIED`.
- Reason: Research integrity requires distinguishing saved artifacts from reproduced results.
- Alternatives considered: Inferring facts from filenames or notebook outputs.
- Expected consequences: The memory files may initially contain placeholders that are filled after
  inspection and execution.
- Reversible: Yes.


## DEC-004 — Keep requested ablations separate from the recorded baseline

- Date: 2026-10-24
- Decision: Preserve the original SleepChannelNet and its 10-feature default inputs, add the SE-Net and HR-motion variants alongside it, and make the new experiments opt-in.
- Reason: The baseline's saved outputs are historical and should remain distinguishable from new, unexecuted implementations.
- Expected consequences: New model comparisons require a fresh run; no new metrics are implied by the notebook edits.
- Reversible: Yes.

## DEC-005 — Do not describe current no-graph control as exact SLAMSS-IFS

- Date: 2026-10-24
- Decision: Mark exact SLAMSS-IFS reproduction as not implemented/verified because the current pipeline lacks the paper's intra-epoch LSTM and a confirmed matching preprocessing protocol.
- Reason: Avoid attributing preprocessing or architecture differences to the requested improvements.
- Reversible: Yes, after the paper protocol is implemented and verified.

## DEC-006 — Use head entropy adaptation where BatchNorm adaptation is unavailable

- Date: 2026-10-24
- Decision: The unsupervised personalization implementation freezes the backbone and optionally entropy-minimizes the prediction head. It does not adapt BatchNorm running statistics.
- Reason: Existing models use GroupNorm, which has no running statistics.
- Reversible: Yes, if a BatchNorm-based architecture is deliberately added.
