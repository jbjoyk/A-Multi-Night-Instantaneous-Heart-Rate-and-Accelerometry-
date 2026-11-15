# BRIEFING — 2026-11-15T08:44:00Z

## Mission
Formulate Candidate Iota (`SmoothSynergyUltradianPyramidCNNBiGRU`) for NAS Iteration 11: Replace rigid CASCI ReLU multiplication with Soft Continuous Bilinear Synergy Manifold (SCBSM) and Causal Temporal Synergy Smoothing, strictly within <= 196,728 parameters (<= 200 parameter addition over Theta 196,373).

## 🔒 My Identity
- Archetype: explorer
- Roles: neural architecture exploration, mathematical derivation, synthesis
- Working directory: /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r11_1
- Original parent: 5bbd46d9-608b-43f0-a1c2-9eb9cfad1c91
- Milestone: NAS Iteration 11 Candidate Iota formulation

## 🔒 Key Constraints
- Read-only investigation — do NOT implement training runs
- Strictly parameter budget: total model <= 196,728 params (target <= 196,550 params, Candidate Theta = 196,373 params; max addition <= 200 params)
- Retain all validated winning components of Candidate Theta (Dual-Scale IHR DSLBS, Dual-Scale ACC DSLBS, MUP, Latent MUP LUAN, DCPG, 4-head IESA, direct 144-dim intra-epoch skip highway)
- Remediate rigid ReLU multiplication in CASCI with Soft Continuous Bilinear Synergy Manifold (SCBSM) and Causal Temporal Synergy Smoothing (W_syn=5)
- Guarantee step-0 identity preservation and strict causal temporal non-leakage

## Current Parent
- Conversation ID: 5bbd46d9-608b-43f0-a1c2-9eb9cfad1c91
- Updated: 2026-11-15T08:35:00Z

## Investigation State
- **Explored paths**:
  * `experiments/EXP-014/nas_iteration10_model.py` (Candidate Theta architecture, 196,373 params)
  * `experiments/EXP-014/nas_iteration10_proxy_results.json` & `.agents/teamwork/teamwork_worker_r10_1/handoff.md` (Fold 1 50-epoch results: 66.84% Acc / 0.6677 WF1 / 67.01% Ep 50 Acc; REM->Light false negatives 2,459; Deep->Light 2,054)
  * `.agents/teamwork/orchestrator/DEAD_ENDS.md` & `ORIGINAL_REQUEST.md`
- **Key findings**:
  * CASCI's rigid ReLU multiplication $s_{\text{rem}} = \text{ReLU}(\Delta z_{\text{ihr}}) \cdot \text{ReLU}(-\Delta z_{\text{acc}})$ zeroed out during phasic twitches or minor decelerations with zero gradients.
  * Formulated SCBSM replacing ReLU with softplus continuous manifold, retaining $s_{\text{rem}} = 1.0177$ during twitches with active gradients ($\sigma(u) > 0$).
  * Added Causal Temporal Synergy Smoothing ($W_{\text{syn}} = 5$ epochs, groups=4 depthwise conv1d, 24 params) retaining 80.5% of tonic REM energy ($2.3317$).
  * Designed 12-dimensional synergy context vector $[u_{\text{ihr}}, u_{\text{acc}}, z_{\text{macro\_ihr}}, z_{\text{macro\_acc}}, s_{\text{inst}}, s_{\text{smooth}}] \in \mathbb{R}^{B \times T \times 12}$.
  * Verified exact parameter count of 196,409 trainable parameters (+36 params over Theta; 141 headroom under 196,550 target, 319 under 196,728 cap).
  * Passed all 12 automated verification tests on CUDA with zero logit difference vs Zeta/Theta at step 0, zero future leakage, and 3.43 ms inference latency.
- **Unexplored areas**: Full proxy execution on Fold 1 by worker r11_1.

## Key Decisions Made
- Selected `hidden_dim = 3` for SCBSM MLP with `in_features = 12` to provide comprehensive instantaneous + smoothed synergy while keeping parameter addition to exactly +36 parameters (total: 196,409 params).
- Zero-initialized `fc2.weight` and `fc_logit` to guarantee exact Step-0 equivalence to Candidate Zeta and Candidate Theta.

## Artifact Index
- DISPATCH.md — Incoming parent dispatch message
- BRIEFING.md — Persistent working memory and state
- progress.md — Liveness heartbeat and milestone tracker
- proposed_nas_iteration11_model.py — Complete implementation of Candidate Iota (`SmoothSynergyUltradianPyramidCNNBiGRU`)
- test_candidate_iota.py — Comprehensive 12-test automated verification suite on CUDA
- report.md — Full mathematical derivation, tensor progression, parameter accounting, and diagnostic analysis
- handoff.md — 5-component handoff report for teamwork_orchestrator and worker r11_1
