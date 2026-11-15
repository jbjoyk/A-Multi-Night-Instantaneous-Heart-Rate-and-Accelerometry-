# Task Assignment: Challenger 1 (Architectural & Tensor Empirical Adversarial Verification)

You are teamwork_challenger_r1_1.
Working directory: /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_challenger_r1_1/

## Mandatory Reading
Before starting any work, you MUST read:
- Original Request: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md`
- Scope: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator_1/SCOPE.md`
- Worker 1 Report: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r1_1/handoff.md`

## Code to Stress-Test
- `experiments/EXP-014/nas_iteration1_model.py`
- `exp013_models.py`

## Adversarial Verification Mission
Write and execute an adversarial test harness in your working directory to empirically probe `MultiScaleConventionalCNNBiGRU` on single GPU `cuda:0` using `./.venv/bin/python`:
1. **Parameter Budget Edge-Case Verification**:
   - Verify parameter count formulas under various channel budgets; confirm that the mutated model is strictly 192,692 parameters (<= 196,728 cap).
2. **Tensor Shape & Boundary Stress Testing**:
   - Test extreme batch sizes ($B=1, B=2, B=16$).
   - Test extreme sequence lengths ($T=1, T=5, T=600, T=1000$).
   - Test extreme input values (all zeros, $\pm 1000.0$, NaNs/Infs handling, random noise).
3. **Gradient Flow & Numerical Stability**:
   - Verify non-zero gradient propagation through all parallel multi-scale branches ($k=1, 3, 5$), projection layers, and residual shortcuts.
   - Verify that train/eval mode switches (`model.train()`, `model.eval()`) maintain deterministic forward outputs when dropout is disabled or tested.
4. Deliver your verdict: `APPROVE` or `REJECT` (with detailed empirical evidence).

## Deliverable
Write your report to:
`/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_challenger_r1_1/handoff.md`
Follow the Handoff Protocol: Observation, Logic Chain, Caveats, Conclusion, Verification Method.
Send a completion message back when done.

## 2026-11-15T03:55:57Z
Received dispatch message:
You are assigned as teamwork_challenger_r1_1.
Your working directory is:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_challenger_r1_1/
Follow all instructions in DISPATCH.md:
Stress test the architecture in experiments/EXP-014/nas_iteration1_model.py on cuda:0 using ./.venv/bin/python.
Write your handoff report to:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_challenger_r1_1/handoff.md
Send a completion message back when done.
