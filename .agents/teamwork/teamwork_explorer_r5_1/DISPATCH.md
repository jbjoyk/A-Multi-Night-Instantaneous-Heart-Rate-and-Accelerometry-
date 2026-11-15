# Dispatch: NAS Iteration 5 Architecture Proposal (Candidate Gamma)

**Target Agent**: `teamwork_explorer_r5_1` (Neural Architecture Explorer)  
**Date**: 2026-11-15T06:26:00Z  
**Working Directory**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r5_1`  

---

## 1. Mandatory First Step & Context
1. **Authoritative Request**: Read `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md`
2. **Project Scope & Dead Ends**:
   - Master Project Scope: `.agents/teamwork/PROJECT.md`
   - Scope Document: `.agents/teamwork/orchestrator/SCOPE.md`
   - Dead Ends Log: `.agents/teamwork/orchestrator/DEAD_ENDS.md`
   - Gate Status Log: `.agents/teamwork/orchestrator/GATE_STATUS.md`
   - Iteration 4 Worker Handoff: `.agents/teamwork/teamwork_worker_r4_1/handoff.md`
   - Iteration 4 Proxy Results: `experiments/EXP-014/nas_iteration4_proxy_results.json`
   - Published SLAMSS-IFS Code: `slamss_ifs_baseline.py`
   - Baseline Model A & EXP-014 Report: `exp013_models.py` and `experiments/EXP-014/EXP014_FINAL_REPORT.md`

---

## 2. Retrospective Diagnosis from Iterations 1 to 4
- **Iteration 1**: Multi-Scale depthwise smoothing across concatenated multimodal channels collapsed REM F1 to 0.1287. *Lesson*: Modality-specific convolutions must remain isolated prior to fusion.
- **Iteration 2 (Candidate B)**: Unconstrained affine FiLM scaling ($\gamma \odot x + \beta$) overfit to training subjects' baseline resting HR on outer test (62.40% Acc). *Lesson*: Subject resting HR offsets require nocturnal normalization.
- **Iteration 3 (Candidate Alpha)**: Masked TIN + Bounded Tanh Gating reached record inner-val (72.53% Acc / 0.7256 WF1), but outer test was 63.09% Acc.
- **Iteration 4 (Candidate Beta)**: Localized Cosine Cross-Attention ($W=5$, bounded $[-1, 1]$) + inner-val peak checkpoint restore peaked at Epoch 45 with 71.26% Acc / 0.7105 WF1 on inner-val, but outer test remained at 62.59% Acc / 0.6236 WF1.
- **CRITICAL ARCHITECTURAL INSIGHT**:
  In ALL 4 iterations, the temporal sequence model was kept IDENTICAL to baseline Model A: a simple 2-layer BiGRU (hidden 48 -> 96 bidirectional) with NO skip connections and NO attention across the 600-epoch night! Over $T=600$ epochs (5 hours), this tiny 48-dim GRU suffers from severe information bottleneck and exponential gradient/information decay. It cannot preserve ultradian sleep cycles (90-min = 180 epochs) or direct epoch features.
  In contrast, the published SLAMSS-IFS paper model (Song et al., IEEE TBME 2026, Table V: 71.04% Acc, 0.7079 WF1 in `slamss_ifs_baseline.py`):
  1. Features an Inter-Epoch Attention mechanism aligning states across the entire sequence.
  2. Features a DIRECT SKIP CONNECTION bypassing the recurrent model directly into the classification head (`Linear(hidden + context + skip_features -> 4)`).

---

## 3. Mission & Requirements for Iteration 5 (Candidate Gamma)
Propose **Candidate Gamma** targeting Hypothesis H-F and H-A:
1. **Architectural Budget**:
   - Must remain strictly within or as close to the 162k–196k parameter range as possible (>90% compression vs 5.47M SLAMSS-IFS). Hard cap: 196,728 parameters (unless performance exceeds target).
2. **Structural Mutation Proposal**:
   - Preserve genuine modality isolation: `hr_encoder` (CNN 64 dims), `acc_encoder` (CNN 64 dims), and `meta_proj` (16 dims).
   - Preserve Masked Temporal Instance Normalization (`MaskedTemporalInstanceNorm`).
   - **Inter-Epoch Architectural Innovations**:
     * **Feature Skip Connection to Classifier**: Directly bypass the 144-dim intra-epoch fused representations around the temporal model directly into the classifier head.
     * **Inter-Epoch Temporal Attention or Multi-Scale Context**: Implement an efficient Sequence Self-Attention / Multi-Scale Temporal Convolution / Seq2Seq Attention layer across the sequence length $T$ that enables direct receptive field over ultradian sleep dynamics without suffering GRU information bottleneck.
     * Ensure dimension alignment, causality/masking awareness (using the boolean mask for valid epochs), and zero-initialization on residual branches so the initial model at step 0 behaves stably.
3. **Artifact Deliverables**:
   - Write candidate PyTorch module to `.agents/teamwork/teamwork_explorer_r5_1/proposed_nas_iteration5_model.py`.
   - Write comprehensive unit test script `.agents/teamwork/teamwork_explorer_r5_1/test_candidate_gamma.py` verifying exact parameter count ($\le 196,728$), forward shape with masking, backward gradient flow, and step-0 stability. Run and verify 100% pass via `./.venv/bin/python`.
   - Write full analysis report `.agents/teamwork/teamwork_explorer_r5_1/analysis.md`.
   - Write handoff report `.agents/teamwork/teamwork_explorer_r5_1/handoff.md`.
4. **Completion Signal**:
   - Send completion message to parent orchestrator via `send_message`.


## 2026-11-15T06:26:39Z
[Message from Parent (5bbd46d9-608b-43f0-a1c2-9eb9cfad1c91)]
You are teamwork_explorer_r5_1. Read your dispatch instructions at:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r5_1/DISPATCH.md
and the authoritative request at:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md.

Formulate Candidate Gamma for NAS Iteration 5 targeting Hypotheses H-F and H-A within the 162k–196k parameter budget, incorporating inter-epoch feature skip connections and sequence-level temporal context to overcome the 2-layer BiGRU information bottleneck.
Implement proposed module and unit tests in your working directory (.agents/teamwork/teamwork_explorer_r5_1/), verify unit test pass with ./.venv/bin/python, write analysis.md and handoff.md, and send completion message via send_message.
