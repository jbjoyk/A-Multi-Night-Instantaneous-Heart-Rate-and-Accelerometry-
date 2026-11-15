# DISPATCH: NAS Iteration 17 — Candidate Omicron Architecture Exploration

## Mission
You are dispatched as `teamwork_explorer_r17_1` to formulate, implement, and verify the structural mutation for **Candidate Omicron** (`PriorCalibratedBoundedMarginPyramidCNNBiGRU`) targeting Hypotheses H-F and H-A on the BIDSleep dataset.

## Required Reading
1. `ORIGINAL_REQUEST.md` (project root)
2. `.agents/teamwork/orchestrator_gen2/SCOPE.md`
3. `.agents/teamwork/orchestrator_gen2/DEAD_ENDS.md` (critical: review failures of Iterations 2-16)
4. `.agents/teamwork/orchestrator_gen2/GATE_STATUS.md`
5. `experiments/EXP-014/nas_iteration12_model.py` (Candidate Kappa: the project's high-water mark, 73.00% inner-val, 67.04% outer-test)
6. `experiments/EXP-014/nas_iteration16_model.py` (Candidate Xi)

## Core Architectural Motivation: Candidate Omicron
- **Historical Grounding**:
  * Candidate Kappa (Iteration 12) achieved project-record inner-validation of **73.00% Accuracy and 0.7395 Macro-F1**, proving its feature representations are fully capable of beating the 71.04% SLAMSS-IFS benchmark.
  * On outer test, accuracy dropped to 67.04% because Light sleep constitutes 43.19% of training windows ($p = [0.0999, 0.4319, 0.2467, 0.2215]$). The classifier head learns a large positive bias $b_{\text{light}}$. On held-out test subjects where $W h(t)$ amplitude fluctuates, the classifier defaults to Light sleep, misclassifying 2,461 REM and 2,052 Deep sleep windows as Light sleep.
  * Modifying representation geometry (ACAM, SAN, TBR) failed:
    - ACAM (Iteration 13): 61.85% (angular margin over-carved Light sleep down to 47.15%).
    - SAN (Iteration 15): 64.52% (whole-sequence z-scoring destroyed macro architecture).
    - TBR (Iteration 16): 62.93% (causal convolution smeared transition boundaries, 2,976 Light->REM errors).
- **The Candidate Omicron Mutation**:
  * Keep the exact, proven BM-CMS (Bounded-Margin Cross-Modal Synergy) and Multi-Scale Ultradian Pyramid BiGRU backbone of Candidate Kappa (196,381 parameters).
  * Ensure margin stability: enforce bounded margin cutoffs ($\tau_{\text{acc}} \ge 0.05, \tau_{\text{ihr}} \le -0.05$) to prevent negative margin drift.
  * Apply **Prior-Calibrated Logit Balancing (PLB)** directly to the final logits:
    $$\text{logits}_c(t) = \text{emission\_logits}_c(t) + \text{synergy\_logits}_c(t) - \tau_{\text{prior}} \cdot \log(p_c)$$
    where:
    - $p_c = [0.0999, 0.4319, 0.2467, 0.2215]$ are the empirical training stage priors (Wake, Light, Deep, REM).
    - $\tau_{\text{prior}} = \tau_{\max} \cdot \sigma(\alpha_{\text{prior}})$, with $\tau_{\max} = 0.5$ and $\alpha_{\text{prior}}$ initialized to $-0.4055$ (giving $\tau_{\text{prior}} = 0.20$).
    - Bounding $\tau_{\text{prior}} \in [0, 0.5]$ guarantees that the correction provides controlled, gentle counterbalance to the majority bias without over-penalizing Light sleep.
  * Parameter Count: Exactly 196,382 parameters (well within the hard cap $\le 196,728$).

## Specific Deliverables
1. Implement Candidate Omicron in `experiments/EXP-014/nas_iteration17_model.py`.
2. Implement and run a rigorous verification script `experiments/EXP-014/verify_candidate_omicron.py` using `./.venv/bin/python` on `cuda:0` verifying:
   - Parameter budget compliance ($\le 196,728$).
   - Forward pass on `(4, 4, 600)`.
   - Boundedness of $\tau_{\text{prior}} \in [0, 0.5]$.
   - Gradient flow to all trainable parameters.
   - RCCLoss compatibility.
3. Write your final report to `.agents/teamwork/teamwork_explorer_r17_1/handoff.md`.
4. Message the orchestrator upon completion.
