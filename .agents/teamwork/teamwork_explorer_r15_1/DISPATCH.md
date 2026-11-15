# DISPATCH: NAS Iteration 15 — Architecture Exploration (Candidate Nu)

**To**: `teamwork_explorer_r15_1`  
**From**: Project Orchestrator Gen 2 (`5a45d55d-cdec-4610-8dcf-b95f2285b08e`)  
**Working Directory**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r15_1`  
**Timestamp**: 2026-11-15T16:41:30Z  

## Objective
Formulate Candidate Nu for NAS Iteration 15, directly targeting Hypothesis H-A (Subject Adaptation / Inter-Subject Invariance) and Hypothesis H-F (Conjunctive Cross-Modal Interaction) to eliminate the generalization drop between inner-validation (73.00% in Candidate Kappa) and outer-test (66.84%–67.04%) and beat the 71.04% paper benchmark.

## Mandatory Context Files (Read Before Formulating)
1. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md`
2. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator_gen2/SCOPE.md`
3. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator_gen2/DEAD_ENDS.md`
4. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r14_1/handoff.md`
5. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r12_1/handoff.md`
6. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/experiments/EXP-014/nas_iteration12_model.py`
7. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/experiments/EXP-014/nas_iteration14_model.py`

## Historical Grounding & Synthesis for Iteration 15
- **Candidate Kappa (Iter 12)**: Reached 73.00% Acc / 0.7395 MF1 on inner-val and 67.04% on outer-test. Protected Light sleep (70.61% recall, only 1,880 false alarms) via strict conjunctive gating ($s_{\text{rem}} = \text{ReLU}(u_{\text{ihr}} - \tau_{\text{ihr}}) \cdot \text{ReLU}(\tau_{\text{acc}} - u_{\text{acc}})$) + Multi-Stream Linear Head.
- **Candidate Lambda (Iter 13)**: Proved autonomic dispersion $v_{\text{disp}}$ drives REM recall to 69.74% and Deep recall to 78.39%, but ACAM asymmetric angular margins crushed Light sleep.
- **Candidate Mu (Iter 14)**: Proved that eliminating ACAM recovered Light recall (+17.86 pp, +2,476 rescued windows). However, additive OR-gating ($u_{\text{ihr}} \lor v_{\text{disp}}$) produced 2,453 false alarms because transitional Light sleep triggered the dispersion branch independently.
- **Iteration 15 Synthesis (Candidate Nu)**:
  1. **Conjunctive Autonomic Dispersion Modulation (CADM) [H-F]**: Replace additive OR-gating with multiplicative/conjunctive modulation:
     $s_{\text{rem}} = \text{ReLU}(u_{\text{ihr}} - \tau_{\text{ihr\_rem}}) \cdot \text{ReLU}(\tau_{\text{acc\_rem}} - u_{\text{acc}}) \cdot (1 + \alpha_{\text{disp}} \tanh(v_{\text{disp}}))$
     This guarantees strict zero whenever $u_{\text{ihr}} \le \tau_{\text{ihr}}$ or $u_{\text{acc}} > \tau_{\text{acc}}$, preventing Light sleep false alarms, while amplifying genuine REM storms when dispersion surges!
  2. **Sequence-Level Subject Adaptive Normalization (SAN) [H-A]**:
     To eliminate the 6.16 pp generalization gap between inner-val and outer-test, apply causal/sequence-level subject-adaptive standardization on the temporal BiGRU representations or attention context before the linear classifier head, centering each unseen subject's feature distribution onto a canonical sphere.
  3. **Strict Budget Cap**: Keep total parameters <= 196,728 (target ~196,400).
  4. **Verification**: Full 14+ test suite in `verify_candidate_nu.py` on `cuda:0` testing parameter budget, causality, scale invariance, and gradient flow.

## Deliverables
- Exploration report in `.agents/teamwork/teamwork_explorer_r15_1/report.md`
- Complete verification script with Candidate Nu PyTorch definition in `.agents/teamwork/teamwork_explorer_r15_1/verify_candidate_nu.py`
- Self-contained handoff report in `.agents/teamwork/teamwork_explorer_r15_1/handoff.md`
- Message parent orchestrator upon completion.
