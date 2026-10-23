---
description: "Prepare a concise, evidence-based handoff for the next research coding agent"
name: "Research Handoff"
argument-hint: "Summarize the current research work and update the shared handoff files"
agent: "agent"
---
Read [AGENTS.md](../../AGENTS.md), [RESEARCH_STATE.md](../../RESEARCH_STATE.md),
[HANDOFF.md](../../HANDOFF.md), [NOTEBOOK_STATE.md](../../NOTEBOOK_STATE.md),
[DECISIONS.md](../../DECISIONS.md), and [EXPERIMENTS.md](../../EXPERIMENTS.md).

Inspect the relevant source files and notebook without starting expensive training. Update the
shared files with only verified facts:

- Record completed work and modified files in `HANDOFF.md`.
- Record notebook cell IDs, execution counts, and whether cells actually ran in `NOTEBOOK_STATE.md`.
- Record verified research facts and clearly mark unknowns in `RESEARCH_STATE.md`.
- Record durable implementation or research choices in `DECISIONS.md`.
- Record every experiment, including failures and unverified runs, in `EXPERIMENTS.md`.

Finish with the next concrete task, validation performed, remaining errors, and any assumptions
that the next agent must verify.
