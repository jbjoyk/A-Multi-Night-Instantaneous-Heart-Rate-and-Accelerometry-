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

