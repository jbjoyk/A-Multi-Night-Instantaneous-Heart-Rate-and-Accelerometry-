# Task Assignment: Worker 1 (Mutated Module Implementation & 1-Epoch Fold 1 Evaluation)

You are teamwork_worker_r1_1.
Working directory: /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r1_1/

## Mandatory Reading
Before starting any work, you MUST read:
- Original Request: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md`
- Scope: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator_1/SCOPE.md`
- Explorer 1 Report: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r1_1/handoff.md`
- Explorer 2 Report: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r1_2/handoff.md`
- Explorer 3 Report: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r1_3/handoff.md`

## File Ownership
You exclusively own and may create/modify:
- `experiments/EXP-014/nas_iteration1_model.py`
- `experiments/EXP-014/evaluate_nas_iteration1.py`
- Any temporary verification scripts or artifacts inside your working directory `.agents/teamwork/teamwork_worker_r1_1/`.

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Requirements & Implementation Instructions

### 1. Implement Mutated PyTorch `nn.Module` (`experiments/EXP-014/nas_iteration1_model.py`)
- Implement a genuine PyTorch `nn.Module` (e.g. `MultiScaleConventionalCNNBiGRU` or `MutatedModelA_MSCNNBiGRU`) that mutates baseline Model A (`ConventionalCNNBiGRU` from `exp013_models.py`).
- Implement the isolated structural mutation grounded in Hypothesis H-A from Explorer 2 / Explorer 1:
  Specifically, incorporate multi-scale temporal convolutional processing (parallel receptive fields $k \in \{1, 3, 5\}$ with projection layers, group normalization, and residual shortcut).
- Ensure total parameter count is strictly $\le 196,728$ parameters (baseline is 163,940, allowing max $+20\%$).
- Ensure complete tensor dimension compatibility: forward pass must accept input dictionaries or tensors `(x_ihr, x_accel, x_meta, targets, mask, seq_lens, subject_idx)` with identical interfaces to `ConventionalCNNBiGRU`.

### 2. Implement and Execute 1-Epoch Evaluation Loop (`experiments/EXP-014/evaluate_nas_iteration1.py`)
- Use the evaluation harness pattern verified by Explorer 3 (`eval_harness_prototype.py`).
- Run using `./.venv/bin/python` on single GPU `cuda:0`.
- Evaluate on Fold 1 (`experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`):
  - 1 epoch of training on inner-train nights (42 batches, 168 nights).
  - Evaluation on outer-test nights (14 batches, 54 nights).
- Hyperparameters must remain strictly unchanged:
  - Batch size = 4
  - Optimizer = `torch.optim.Adam`
  - Learning rate = `1e-4` (weight_decay = 0.0)
  - Loss = CrossEntropy + 0.01 * RCCLoss
- Compute and log:
  - Total parameter count and delta over baseline.
  - Forward pass verification output.
  - 1-epoch training loss.
  - Test set Macro-F1 across 4 stages, plus per-class F1 (Wake, Light, Deep, REM), Overall Accuracy, and Cohen's Kappa.

### 3. Deliverable
Execute the evaluation, verify passing execution on `cuda:0`, and write your comprehensive handoff report to:
`/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r1_1/handoff.md`
Follow the Handoff Protocol: Observation, Logic Chain, Caveats, Conclusion, Verification Method.
Send a message back when complete.


## 2026-11-15T03:41:21Z
Received dispatch from parent `a06aefdd-00a4-4ecc-9d80-120250d9d52f`:
Follow all instructions in DISPATCH.md:
1. Implement the mutated PyTorch nn.Module in experiments/EXP-014/nas_iteration1_model.py.
2. Implement the evaluation script in experiments/EXP-014/evaluate_nas_iteration1.py and execute the 1-epoch evaluation loop on Fold 1 on cuda:0 using ./.venv/bin/python.
3. Write your handoff report to:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r1_1/handoff.md
Send a completion message back when done.
