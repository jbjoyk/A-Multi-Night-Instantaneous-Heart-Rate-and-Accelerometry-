# Task Assignment: Explorer 3 (Evaluation Pipeline, Fold 1 Manifest & Local Execution Harness)

You are teamwork_explorer_r1_3.
Working directory: /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r1_3/

## Mandatory Reading
Before starting any work, you MUST read:
- Original Request: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md`
- Scope: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator_1/SCOPE.md`

## Mission & Scope
Investigate the evaluation and training codebase:
1. Examine `experiments/EXP-014/EXP014_MANIFESTS/fold_1.json` and any existing EXP-013/EXP-014 training and evaluation scripts (e.g. `exp013_train.py`, `train_exp014.py`, dataset loaders, feature extractors).
2. Document the exact data loading pipeline, expected input tensor dimensions (e.g., `(batch_size, sequence_length, channels)` or `(batch_size, channels, sequence_length)`), batch size, optimizer type, learning rate, and loss function used for Fold 1.
3. Determine how to construct a lightweight, standalone 1-epoch evaluation script that runs on single GPU `cuda:0` using `./.venv/bin/python`.
4. Ensure the evaluation script computes and logs the Macro-F1 score across all sleep stage classes accurately (e.g. via sklearn or torchmetrics) without altering batch size, optimizer, or learning rate.
5. Identify the exact Python binary, dependencies, and execution command.

## Deliverable
Write your findings to:
`/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r1_3/handoff.md`
Follow the Handoff Protocol: Observation, Logic Chain, Caveats, Conclusion, Verification Method.
Send a message when complete.


## 2026-11-15T03:32:23Z
[Message from a06aefdd-00a4-4ecc-9d80-120250d9d52f]
You are assigned as teamwork_explorer_r1_3.
Your working directory is:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r1_3/

Read your assignment details in:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r1_3/DISPATCH.md

Also read the original user request at:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md

Follow all instructions in DISPATCH.md and write your handoff report to:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r1_3/handoff.md
Send a completion message back with send_message when done.
