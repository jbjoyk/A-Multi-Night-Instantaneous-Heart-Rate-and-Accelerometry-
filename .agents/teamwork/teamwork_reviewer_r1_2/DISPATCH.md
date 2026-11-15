# Task Assignment: Reviewer 2 (Evaluation Loop, Hyperparameters & Metrics Review)

You are teamwork_reviewer_r1_2.
Working directory: /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_reviewer_r1_2/

## Mandatory Reading
Before starting any work, you MUST read:
- Original Request: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md`
- Scope: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator_1/SCOPE.md`
- Worker 1 Report: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r1_1/handoff.md`

## Code to Review
- `experiments/EXP-014/evaluate_nas_iteration1.py`
- `experiments/EXP-014/nas_iteration1_model.py`
- `experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`
- `experiments/EXP-014/nas_iteration1_eval_results.json`

## Review Scope & Responsibilities
1. Review the evaluation script `experiments/EXP-014/evaluate_nas_iteration1.py`:
   - Verify hyperparameter invariance: batch size = 4, optimizer = Adam, learning rate = 1e-4, weight_decay = 0.0, sequence length = 600, CE + 0.01 * RCCLoss.
   - Verify that Fold 1 manifest is strictly followed: 168 inner-train nights (32 subjects), 54 outer-test nights (10 subjects), zero subject leakage.
   - Verify Macro-F1 metric computation: proper masking of invalid/padded epochs, 4-stage calculation, per-class F1 reporting.
2. Run the evaluation script on single GPU `cuda:0` using `./.venv/bin/python`:
   `./.venv/bin/python experiments/EXP-014/evaluate_nas_iteration1.py`
3. Verify that results match the claims in `handoff.md` and `nas_iteration1_eval_results.json`.
4. Deliver your review report with an explicit verdict: `APPROVE` or `REQUEST_CHANGES`.

## Deliverable
Write your report to:
`/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_reviewer_r1_2/handoff.md`
Follow the Handoff Protocol: Observation, Logic Chain, Caveats, Conclusion, Verification Method.
State your verdict prominently (e.g. `Verdict: APPROVE`).

## 2026-11-15T03:48:35Z
[Message] timestamp=2026-11-15T03:48:35Z sender=a06aefdd-00a4-4ecc-9d80-120250d9d52f priority=MESSAGE_PRIORITY_HIGH content=You are assigned as teamwork_reviewer_r1_2.
Your working directory is:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_reviewer_r1_2/

Read your assignment details in:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_reviewer_r1_2/DISPATCH.md

Also read the original user request at:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md

Follow all instructions in DISPATCH.md:
1. Review experiments/EXP-014/evaluate_nas_iteration1.py, experiments/EXP-014/EXP014_MANIFESTS/fold_1.json, and nas_iteration1_eval_results.json.
2. Execute the evaluation script on cuda:0 using ./.venv/bin/python.
3. Write your handoff report with an explicit verdict (APPROVE or REQUEST_CHANGES) to:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_reviewer_r1_2/handoff.md
Send a completion message back when done.
