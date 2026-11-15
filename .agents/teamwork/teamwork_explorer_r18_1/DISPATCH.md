# DISPATCH: NAS Iteration 18 — Candidate Pi Architecture Exploration

## Mission
You are dispatched as `teamwork_explorer_r18_1` to formulate, implement, and verify the structural mutation for **Candidate Pi** (`UnbiasedHypersphericalBoundedMarginPyramidCNNBiGRU`) targeting Hypotheses H-F and H-A on the BIDSleep dataset.

## Required Reading
1. `ORIGINAL_REQUEST.md` (project root)
2. `.agents/teamwork/orchestrator_gen2/SCOPE.md`
3. `.agents/teamwork/orchestrator_gen2/DEAD_ENDS.md` (critical: review failures of Iterations 2-17)
4. `.agents/teamwork/orchestrator_gen2/GATE_STATUS.md`
5. `experiments/EXP-014/nas_iteration12_model.py` (Candidate Kappa: baseline 73.00% inner-val, 67.04% outer-test)
6. `experiments/EXP-014/nas_iteration13_model.py` (Candidate Lambda: proved cosine representations achieve 69.74% REM and 78.39% Deep recall, but failed due to asymmetric margins m_rem=0.15)
7. `experiments/EXP-014/nas_iteration17_model.py` (Candidate Omicron: verified BoundedMarginSTE)

## Core Architectural Motivation: Candidate Pi
- **The Empirical Problem**:
  * Inner-validation across Candidate Kappa (73.00%) and Candidate Omicron (73.04%) proves that the Dual-Scale ACC DSLBS + BM-CMS + Ultradian Temporal Pyramid BiGRU representations already possess the capacity to beat the 71.04% SLAMSS-IFS benchmark.
  * However, on held-out test subjects, outer-test accuracy drops to ~66.8% because of a massive collapse of minority classes into Light sleep: 2,657 REM windows (37.4%) and 2,225 Deep windows (28.1%) are misclassified as Light sleep.
  * In linear classifier heads ($\text{logits} = W h + b$), because Light sleep is 43.19% of training windows, $b_{\text{light}}$ is large and positive. On held-out test subjects where feature norms $\|h\|$ attenuate due to physiological distribution shifts, $W h$ shrinks and the classifier defaults to $\arg\max_c(b_c) = \text{Light Sleep}$.
  * In Iteration 13 (Candidate Lambda), an angular cosine classifier was used, but it included aggressive asymmetric margins ($m_{\text{rem}}=0.15, m_{\text{deep}}=0.12$), which over-carved Light sleep (Light recall collapsed to 47.15%).
  * In Iteration 17 (Candidate Omicron), additive prior correction ($-\tau \log p$) was absorbed by the unconstrained linear bias parameters $b_c$ during training.
- **The Candidate Pi Mutation**:
  1. **Unbiased Hyperspherical Cosine Head (UH-CH)**:
     - Activate the hyperspherical cosine classifier:
       $$\text{logits}_c(t) = s \cdot \frac{W_c^\top h(t)}{\|W_c\|_2 \|h(t)\|_2}$$
       where $h(t)$ is L2-normalized onto the unit hypersphere $S^{63}$ and class prototypes $W_c$ are L2-normalized onto $S^{63}$.
     - **Symmetric Zero Margins ($m = 0.0$)**: All 4 stages are treated with exact geometric symmetry on the hypersphere. No class is penalized or distorted during training, completely avoiding the over-carving that harmed Light sleep in Candidate Lambda.
     - **Completely Bias-Free**: There is NO additive bias vector $b_c$. The classifier CANNOT collapse into Light sleep simply because of feature norm attenuation on unseen test subjects!
     - **Learnable Temperature Scale**: $s = \exp(\log s)$ or bounded parameter initialized at $s = 16.0$.
  2. **Zero-Bias Synergy Projection**:
     - `fc_logit` in BM-CASCI uses `bias=False` ($3 \times 4 = 12$ params), preventing any static bias injection into the final logits.
  3. **Bounded Margin Stability Cutoffs (`BoundedMarginSTE`)**:
     - Retain `BoundedMarginSTE` from Candidate Omicron to clamp $\tau_{\text{acc}} \ge 0.05$ and $\tau_{\text{ihr}} \ge -0.05$, permanently preventing negative margin drift while ensuring 100% gradient flow.
  4. **Strict Parameter Budget**:
     - Candidate Kappa: 196,381 params.
     - Cosine head ($18,881$ vs $18,884$ params) + zero-bias `fc_logit` ($12$ vs $16$ params) $\implies$ **196,374 parameters**.
     - Strictly $\le 196,728$ hard cap (354 parameters headroom).
     - 96.41% compression ratio vs 5.47M benchmark.

## Specific Deliverables
1. Implement Candidate Pi in `experiments/EXP-014/nas_iteration18_model.py`.
2. Implement and run a rigorous verification script `experiments/EXP-014/verify_candidate_pi.py` using `./.venv/bin/python` on `cuda:0` verifying:
   - Parameter budget compliance ($\le 196,728$, expected 196,374).
   - Forward pass on `(B=4, C=4, T=600)` on `cuda:0`.
   - Complete absence of additive linear bias terms in classifier head and synergy logit projection.
   - Cosine normalization on $S^{63}$ and scale initialization ($s = 16.0$).
   - BoundedMarginSTE stability cutoffs ($\tau_{\text{ihr\_rem}} \ge -0.05, \tau_{\text{acc\_rem}} \ge 0.05$).
   - 100% finite, non-zero gradient flow to all trainable parameters during backprop.
   - RCCLoss compatibility.
3. Write your final report to `.agents/teamwork/teamwork_explorer_r18_1/handoff.md`.
4. Message the orchestrator upon completion.

## 2026-11-15T11:44:55Z
You are teamwork_explorer_r18_1. Your working directory is `.agents/teamwork/teamwork_explorer_r18_1/`.
Read your dispatch instructions in `.agents/teamwork/teamwork_explorer_r18_1/DISPATCH.md` and read `ORIGINAL_REQUEST.md`.

Your mission is to formulate, implement, and verify the structural mutation for Candidate Pi (`UnbiasedHypersphericalBoundedMarginPyramidCNNBiGRU`) targeting Hypotheses H-F and H-A on the BIDSleep dataset.

Follow the instructions in `DISPATCH.md` carefully:
1. Ground the architecture in Candidate Kappa (`experiments/EXP-014/nas_iteration12_model.py`), Candidate Lambda (`experiments/EXP-014/nas_iteration13_model.py`), and Candidate Omicron (`experiments/EXP-014/nas_iteration17_model.py`).
2. Implement Candidate Pi in `experiments/EXP-014/nas_iteration18_model.py`.
3. Implement Unbiased Hyperspherical Cosine Head (UH-CH) with:
   - Symmetrically normalized features h_norm on S^63 and class prototypes w_norm on S^63.
   - S = exp(log_s) or parameter initialized at s = 16.0.
   - Symmetric zero-margin (m = 0.0) so no class is penalized during training.
   - Zero bias: complete elimination of additive linear bias terms in classifier head.
   - Zero-bias synergy projection: fc_logit with bias=False.
   - BoundedMarginSTE stability cutoffs (tau_acc >= 0.05, tau_ihr >= -0.05).
4. Parameter budget cap: <= 196,728 parameters (expected: exactly 196,374 parameters).
5. Write and execute a rigorous verification script `experiments/EXP-014/verify_candidate_pi.py` using `./.venv/bin/python` on `cuda:0` testing:
   - Parameter budget compliance (<= 196,728).
   - Forward pass on (B=4, C=4, T=600) on cuda:0.
   - Absence of linear bias in classifier and synergy logits.
   - BoundedMarginSTE enforcement.
   - Gradient flow to all trainable parameters during backprop.
   - RCCLoss compatibility.
6. Write your comprehensive handoff report to `.agents/teamwork/teamwork_explorer_r18_1/handoff.md`.
7. Use `send_message` to communicate completion back to me.
