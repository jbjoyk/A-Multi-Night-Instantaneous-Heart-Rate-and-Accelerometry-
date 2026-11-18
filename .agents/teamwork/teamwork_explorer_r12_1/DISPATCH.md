## 2026-11-15T08:51:57Z
You are teamwork_explorer_r12_1, a neural architecture exploration agent for sleep staging NAS Iteration 12.
Working directory: /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r12_1/

MANDATORY CONTEXT: Read:
- /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md
- /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator/BRIEFING.md
- /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/orchestrator/DEAD_ENDS.md
- /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/experiments/EXP-014/nas_iteration10_model.py (Candidate Theta: All-time project best 66.84% outer test Acc / 67.01% Ep 50 Acc, 0.6677 WF1 / 0.6684 Ep 50 WF1, Kappa 0.5153, 196,373 params)
- /home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/experiments/EXP-014/nas_iteration11_model.py & .agents/teamwork/teamwork_worker_r11_1/handoff.md (Candidate Iota analysis: why softplus without zero-boundary caused synergy leakage into Light sleep)

MISSION:
Formulate Candidate Kappa (`BoundedMarginSynergyPyramidCNNBiGRU` or similar) for NAS Iteration 12:
1. Synthesize the core breakthroughs from Candidates Theta and Iota:
   - Candidate Theta (CASCI with Dual-Scale ACC DSLBS) established the project all-time record of 66.84% Acc / 67.01% Ep 50 Acc / 0.6684 WF1 / 0.5153 Kappa by using sharp ReLU gating, which reduced Light->REM false alarms by 33.6%. However, strict zero thresholds killed REM synergy during small twitches.
   - Candidate Iota proved that removing the sharp zero boundary with softplus causes synergy leakage into Light sleep, blowing Light->REM false alarms back up to 2,412.
   - Key Insight: We MUST retain the sharp zero-cutoff of ReLU to protect Light sleep, but replace zero thresholds with **learned asymmetric margins / tolerances**:
     * REM synergy: s_rem = ReLU(u_ihr - tau_ihr_rem) * ReLU(tau_acc_rem - u_acc)
       where tau_acc_rem is initialized to e.g. +0.25 (allowing physiological twitches up to +0.25 without killing REM synergy, but strictly zeroing once real body movement u_acc > tau_acc_rem occurs!).
     * Deep sleep synergy: s_deep = ReLU(tau_ihr_deep - u_ihr) * ReLU(tau_acc_deep - u_acc)
     * Wake synergy: s_wake = ReLU(u_ihr - tau_ihr_wake) * ReLU(u_acc - tau_acc_wake)
     * Light synergy: s_light = ReLU(tau_ihr_light - u_ihr) * ReLU(u_acc - tau_acc_light)
2. Retain all proven foundational components:
   - Dual-Scale Local Basal Standardization for IHR (DSLBS, W_micro=15, W_macro=91)
   - Dual-Scale Accelerometry Standardization (Dual-Scale ACC DSLBS, W_micro=15, W_macro=91)
   - Multi-Scale Ultradian Temporal Pyramids (MUP, depthwise causal convs W=5, 15, 31)
   - Latent MUP LUAN
   - Dynamic Circadian Phase Gating (DCPG)
   - 4-head Inter-Epoch Sequence Attention (IESA)
   - Direct 144-dim intra-epoch skip highway bypassing BiGRU into multi-stream classifier
3. Explore Cosine / Angular Margin Classification in Head:
   - Evaluate or integrate normalized linear projection / angular margin to enforce angular separation between Light, Deep, and REM classes in the 288-dim feature representation.
4. Strictly budget parameters:
   - Total model parameters must be <= 196,728 (Candidate Theta has 196,373 params; 355 headroom).
   - Keep total parameters strictly <= 196,500 parameters (e.g. around 196,380 - 196,450 params).
5. Mathematical Guarantees:
   - Step-0 mathematical equivalence to Candidate Theta (logit difference < 1e-5).
   - Strict causal non-leakage (diff_past = 0.0).
   - Invariance to sensor gain and DC gravity offsets.
6. Design a comprehensive 12-test automated verification suite on CUDA.
7. Write your complete design to .agents/teamwork/teamwork_explorer_r12_1/report.md, and send a completion message with the file path.
