# Research Project Guidelines

## Research scope

This workspace contains research materials for sleep staging using instantaneous heart rate,
accelerometry, and EEG sleep-stage labels. The exact research question, model architecture,
training status, and verified metrics are recorded in [RESEARCH_STATE.md](./RESEARCH_STATE.md).
Under [DEC-021](./DECISIONS.md), the experimental phase has been reopened; the prior restriction
specifying "no further training runs" no longer applies (though launching training jobs remains
strictly subject to explicit user approval).
Do not invent missing research facts.

## Required context before work

Before changing code or notebooks, read:

1. [RESEARCH_STATE.md](./RESEARCH_STATE.md)
2. [HANDOFF.md](./HANDOFF.md)
3. [NOTEBOOK_STATE.md](./NOTEBOOK_STATE.md)
4. [DECISIONS.md](./DECISIONS.md)
5. [EXPERIMENTS.md](./EXPERIMENTS.md)

Then inspect the relevant source files and notebook cells.

## Research integrity

- Preserve the existing methodology, model architecture, dataset, configuration, and results unless
  the user explicitly requests a change.
- Distinguish clearly between code that exists and code that has executed successfully.
- Never treat stale notebook outputs as current results.
- Record experiment configuration, random seeds, hardware, failures, and verification status.
- Do not commit credentials, private data, or generated model checkpoints.

## Notebook workflow

- Inspect notebooks with `nbformat` or an equivalent notebook-aware tool.
- Preserve metadata, cell IDs, useful outputs, and execution counts unless a change requires otherwise.
- Record changed cell numbers or IDs in [NOTEBOOK_STATE.md](./NOTEBOOK_STATE.md).
- Do not run expensive training automatically.
- A notebook kernel state is never assumed to be available to another agent.

## GPU and environment rules

- Use the configured project environment and document the Python, PyTorch, CUDA, and GPU assumptions
  used for an experiment.
- Do not assume a GPU is available; verify it before GPU experiments.
- Avoid launching long-running training jobs without explicit user approval.
- Keep raw dataset recordings outside normal Git commits; see [.gitignore](./.gitignore).

## Verification and handoff

- Run the smallest relevant validation command after a change.
- Update [HANDOFF.md](./HANDOFF.md) with modified files, tests, failures, and the next concrete task.
- Update [RESEARCH_STATE.md](./RESEARCH_STATE.md) only with verified or explicitly marked unknown facts.
- Update [DECISIONS.md](./DECISIONS.md) when an implementation or research decision changes.
- Update [EXPERIMENTS.md](./EXPERIMENTS.md) for every experiment, including failed or unverified runs.

## Continuous Git synchronization (Claude & collaborator access)

- **MANDATORY PUSH POLICY**: Immediately after every completed experiment, evaluation, or research documentation change, all experiment artifacts (markdown reports, CSV summaries, metrics JSON, scripts, and documentation) **MUST be committed and pushed to GitHub (`origin main`)**.
- This ensures Claude and any external collaborator or agent can immediately inspect the latest findings and propose next research actions.
- Never let findings sit unpushed locally.
- Heavy raw dataset files (`Bidslab*`, ~28 GB) and binary checkpoints/pickles (`*.pt`, `*.pkl`) are strictly ignored via `.gitignore` to keep syncs fast, reliable, and lightweight.

