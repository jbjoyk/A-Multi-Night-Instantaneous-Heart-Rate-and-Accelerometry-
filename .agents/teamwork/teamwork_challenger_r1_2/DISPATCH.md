# Task Assignment: Challenger 2 (Evaluation Loop, Metric Oracles & Memory Stress-Testing)

You are teamwork_challenger_r1_2.
Working directory: /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_challenger_r1_2/

## Mandatory Reading
Before starting any work, you MUST read:
- Original Request: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md`
- Scope: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator_1/SCOPE.md`
- Worker 1 Report: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r1_1/handoff.md`

## Code to Stress-Test
- `experiments/EXP-014/evaluate_nas_iteration1.py`
- `experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`
- `experiments/EXP-014/nas_iteration1_eval_results.json`

## Adversarial Verification Mission
Write and execute an adversarial test harness in your working directory on single GPU `cuda:0` using `./.venv/bin/python`:
1. **Metric Oracle Stress Testing**:
   - Construct adversarial prediction arrays (all zeros, uniform random predictions, perfectly inverted predictions, biased predictions) and test the evaluation script's metric computation logic to verify that Macro-F1, Cohen's Kappa, and Accuracy compute exact mathematical values without edge-case crashes (e.g. division by zero, empty classes).
2. **GPU Memory Leak & Resource Stress Test**:
   - Run multiple consecutive iterations of the 1-epoch evaluation loop on `cuda:0` to ensure peak VRAM does not grow (zero PyTorch memory leaks, proper detachment of loss/tensors).
3. **Partition & Manifest Invariance Test**:
   - Verify that test data is never shuffled or contaminated by training targets, and that subject disjointness holds across all folds.
4. Deliver your verdict: `APPROVE` or `REJECT` (with detailed empirical evidence).

## Deliverable
Write your report to:
`/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_challenger_r1_2/handoff.md`
Follow the Handoff Protocol: Observation, Logic Chain, Caveats, Conclusion, Verification Method.
Send a completion message back when done.


## 2026-11-15T03:56:02Z
[Message] timestamp=2026-11-15T03:56:02Z sender=a06aefdd-00a4-4ecc-9d80-120250d9d52f priority=MESSAGE_PRIORITY_HIGH content=You are assigned as teamwork_challenger_r1_2.
Your working directory is:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_challenger_r1_2/

Read your assignment details in:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_challenger_r1_2/DISPATCH.md

Also read the original user request at:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md

Follow all instructions in DISPATCH.md:
Stress test the evaluation loop and metrics in experiments/EXP-014/evaluate_nas_iteration1.py on cuda:0 using ./.venv/bin/python.
Write your handoff report to:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_challenger_r1_2/handoff.md
Send a completion message back when done.
