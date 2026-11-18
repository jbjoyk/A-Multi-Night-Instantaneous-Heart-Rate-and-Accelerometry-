# DISPATCH: NAS Iteration 13 — Architecture Exploration (Candidate Lambda)

**To**: `teamwork_explorer_r13_1`  
**From**: Project Orchestrator Gen 2 (`5a45d55d-cdec-4610-8dcf-b95f2285b08e`)  
**Working Directory**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r13_1`  
**Timestamp**: 2026-11-15T16:08:00Z  

## Objective
Investigate the failure and pathology of Iteration 12 (Candidate Kappa: 66.84% best / 67.04% final outer-test Acc, inner-val record 73.00% Acc / 0.7395 MF1) and formulate Candidate Lambda for NAS Iteration 13 targeting Hypothesis H-F (cross-modal IHR-accelerometry interaction) and Hypothesis H-A (subject adaptation / inter-subject distribution shift).

## Mandatory Context Files (Read Before Formulating)
1. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md`
2. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator_gen2/SCOPE.md`
3. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator_gen2/DEAD_ENDS.md`
4. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r12_1/handoff.md`
5. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/experiments/EXP-014/nas_iteration12_model.py`
6. `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/experiments/EXP-014/nas_iteration12_proxy_results.json`

## Dead Ends to Avoid
- Do NOT use affine FiLM modulation without causal basal anchoring (Iteration 2 failed, 62.40%).
- Do NOT use static MLP fusion (Iteration 3 failed, 63.09%).
- Do NOT use global whole-night z-score normalization (Iteration 6 failed, 62.11%, collapsed non-stationarity).
- Do NOT use unstandardized raw accelerometry standard deviation (Iteration 9 failed, 64.57%, sensor gain shift).
- Do NOT use smooth softplus without sharp zero boundary (Iteration 11 failed, 65.12%, synergy leaked into Light sleep).
- Note from Candidate Kappa: Candidate Kappa achieved 73.00% inner-val Acc and 67.04% final outer-test Acc. It proved that learned asymmetric margins ($\tau$) with sharp ReLU cutoffs successfully prevent Light sleep leakage (1,880 false alarms). However, outer-test still suffered from 2,461 REM->Light and 2,052 Deep->Light errors due to inter-subject baseline shifts (Hypothesis H-A).

## Exploration & Design Scope for Candidate Lambda
Target both H-F and H-A:
1. How to solve the inter-subject shift that causes REM and Deep to collapse into Light on unseen subjects?
   - Consider Test-time / Sequence-level Subject Adaptive Centering / Normalization (e.g. causal rolling z-score / adaptive affine calibration on the sequence representations or classifier logits).
   - Consider Adaptive Angular Margin / Cosine Classifier Head with learnable class centers that enforce a spherical margin $m$ between Light, Deep, and REM representations, preventing majority-class Light collapse.
   - Consider Cross-Modal Autonomic Dispersion / Entropy Gating: REM is characterized not just by mean IHR surges, but by high autonomic heart rate variability (HRV / local dispersion) coupled with somatosensory stillness.
2. Parameter Budget Cap: Trainable parameters must remain strictly <= 196,728 (target ~196,400).
3. Tensor Safety: Complete mathematical verification script (`verify_candidate_lambda.py`) with unit tests demonstrating parameter budget, step-0 equivalence or provable initialization stability, causality, and gradient flow.

## Deliverables
- Exploration report in `.agents/teamwork/teamwork_explorer_r13_1/report.md`
- Complete verification script with Candidate Lambda PyTorch definition in `.agents/teamwork/teamwork_explorer_r13_1/verify_candidate_lambda.py`
- Self-contained handoff report in `.agents/teamwork/teamwork_explorer_r13_1/handoff.md`
- Message parent orchestrator upon completion.
