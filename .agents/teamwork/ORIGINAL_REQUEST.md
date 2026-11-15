# Original User Request

## Initial Request — 2026-11-15T03:28:56Z

Act as an autonomous Neural Architecture Search (NAS) syndicate to iteratively mutate the attached PyTorch baseline sleep-staging model. Self-organize into the necessary sub-roles to accomplish this (e.g., an Architect to propose the changes, a PyTorch Engineer to write the tensor-safe nn.Module, and an Evaluator to run the code and verify the Macro-F1).

Working directory: /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels
Integrity mode: development

## Requirements

### R1. Structural Mutation Proposal and Tensor-Safe Module Implementation (Iteration 1)
Analyze the baseline sleep-staging model (Model A, `ConventionalCNNBiGRU`, 163,940 parameters in `exp013_models.py`) and propose an isolated structural mutation (such as multi-scale feature fusion or depthwise convolutions) that directly supports the pre-registered "H-A" hypothesis for multi-night within-person adaptation detailed in `RESEARCH_DIRECTION_v3_FULL.md`. Implement the mutated architecture as a PyTorch `nn.Module`, ensuring complete tensor dimension compatibility with necessary projection layers if channel dimensions change. This run is scoped strictly to Iteration 1.

### R2. Architectural Budget and Mutation Constraints
The total parameter count of the mutated architecture must not exceed a 20% increase over the baseline (maximum 196,728 parameters). The modifications must focus strictly on structural mutations; batch sizes, optimizers, and learning rates must remain unchanged.

### R3. Local Forward Pass and Evaluation Loop Execution
Write the evaluation script and execute the forward pass and a 1-epoch evaluation loop locally on the host Ubuntu dual-RTX 3060 environment using a single GPU (`cuda:0`) and `./.venv/bin/python`. The 1-epoch evaluation loop must run against Fold 1 (`experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`) of the BIDSleep dataset and report the resulting Macro-F1 metric.

## Acceptance Criteria

### Architectural Integrity
- [ ] Mutated architecture implements an isolated structural mutation grounded in hypothesis H-A from `RESEARCH_DIRECTION_v3_FULL.md`.
- [ ] Total parameter count does not exceed 196,728 parameters (<= 20% increase over the baseline 163,940 parameters).
- [ ] All tensor operations and dimension transitions are dimension-compatible, with projection layers implemented for any channel changes.

### Local Execution & Verification
- [ ] Forward pass executes without shape or runtime errors on expected BIDSleep input tensor dimensions.
- [ ] 1-epoch evaluation loop completes successfully on Fold 1 (`experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`) on single GPU `cuda:0` using `./.venv/bin/python`.
- [ ] Evaluation script logs the Macro-F1 score without altering batch size, optimizer, or learning rate.
