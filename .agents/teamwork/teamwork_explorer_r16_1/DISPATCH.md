# DISPATCH: NAS Iteration 16 — Architecture Exploration (Candidate Xi)

**To**: `teamwork_explorer_r16_1`  
**From**: Project Orchestrator Gen 2 (`5a45d55d-cdec-4610-8dcf-b95f2285b08e`)  
**Working Directory**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r16_1`  
**Timestamp**: 2026-11-15T17:02:00Z  

## Objective
Formulate Candidate Xi for NAS Iteration 16, synthesizing the proven strengths of Candidate Kappa (67.04% outer-test Acc, 73.00% inner-val Acc) and CADM (Conjunctive Autonomic Dispersion Modulation) while discarding the failed sequence-level z-scoring (SAN), and introducing a lightweight Temporal Bout Refinement (TBR) block to enforce physiological sleep bout continuity and eliminate transitional classification noise.

## Mandatory Context Files (Read Before Formulating)
1. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md`
2. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator_gen2/SCOPE.md`
3. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator_gen2/DEAD_ENDS.md`
4. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r15_1/handoff.md`
5. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r12_1/handoff.md`
6. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/experiments/EXP-014/nas_iteration12_model.py`
7. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/experiments/EXP-014/nas_iteration15_model.py`

## Critical Architectural Insights Across Iterations 10–15
1. **Candidate Kappa (Iter 12)** remains the gold-standard baseline: 73.00% inner-val Acc, 67.04% final outer-test Acc, 70.61% Light recall, 70.21% Deep recall, only 1,880 Light->REM false alarms.
2. **Failure of SAN (Iter 15)**: Whole-sequence z-scoring ($T=600$) is a proven dead end because subtracting the whole-night temporal mean destroys subject macro-structural sleep stage composition (making Deep-sleep dominant subjects look like zero-mean Light sleep). SAN MUST BE REMOVED.
3. **Success of CADM (Iter 15)**: Multiplicative dispersion modulation ($s_{\text{rem}} = \text{ReLU}(u_{\text{ihr}} - \tau_{\text{ihr}}) \cdot \text{ReLU}(\tau_{\text{acc}} - u_{\text{acc}}) \cdot \text{ReLU}(1 + \alpha_{\text{disp}} \tanh(v_{\text{disp}}))$) successfully eliminated additive leakage, reducing Light->REM false alarms by -278 windows.
4. **Candidate Xi Proposal**:
   - Keep Candidate Kappa's foundation: Dual-Scale IHR & ACC DSLBS ($W_{\text{macro}}=91, W_{\text{micro}}=15$), MUP temporal pyramids, LUAN latent normalization, DCPG circadian gating, 2-layer BiGRU, 4-head IESA, 144-dim intra-epoch highway, and Calibrated Multi-Stream Linear Head.
   - Integrate CADM (Conjunctive Autonomic Dispersion Modulation) without SAN.
   - Introduce **Temporal Bout Refinement (TBR)**: a lightweight depthwise causal 1D convolution block (kernel size 5 or 7, dilation 1, ~320 params) applied to the combined pre-classifier feature stream or logits, enforcing physiological multi-epoch sleep bout continuity and eliminating single-epoch transition noise.
   - Enforce parameter budget <= 196,728 (target ~196,400–196,650).
   - Write full verification script `verify_candidate_xi.py` testing parameter budget, causality, scale invariance, and gradient flow on `cuda:0`.

## Deliverables
- Exploration report in `.agents/teamwork/teamwork_explorer_r16_1/report.md`
- Complete verification script with Candidate Xi PyTorch definition in `.agents/teamwork/teamwork_explorer_r16_1/verify_candidate_xi.py`
- Self-contained handoff report in `.agents/teamwork/teamwork_explorer_r16_1/handoff.md`
- Message parent orchestrator upon completion.
