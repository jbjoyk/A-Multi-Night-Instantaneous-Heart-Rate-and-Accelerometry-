# Dispatch: NAS Iteration 6 Architecture Proposal (Candidate Delta)

**Target Agent**: `teamwork_explorer_r6_1` (Neural Architecture Explorer)  
**Date**: 2026-11-15T06:50:00Z  
**Working Directory**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r6_1`  

---

## 1. Mandatory First Step & Context
1. **Authoritative Request**: Read `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md`
2. **Project Scope & Dead Ends**:
   - Master Project Scope: `.agents/teamwork/PROJECT.md`
   - Scope Document: `.agents/teamwork/orchestrator/SCOPE.md`
   - Dead Ends Log: `.agents/teamwork/orchestrator/DEAD_ENDS.md`
   - Gate Status Log: `.agents/teamwork/orchestrator/GATE_STATUS.md`
   - Iteration 5 Worker Handoff: `.agents/teamwork/teamwork_worker_r5_1/handoff.md`
   - Iteration 5 Proxy Results: `experiments/EXP-014/nas_iteration5_proxy_results.json`
   - Published SLAMSS-IFS Code: `slamss_ifs_baseline.py`
   - EXP-019 500-Epoch Benchmark (DEC-027): `experiments/EXP-019/run_exp019_500epochs.py` and `DECISIONS.md:380-400`
   - Iteration 1 MultiScale Model: `experiments/EXP-014/nas_iteration1_model.py`

---

## 2. Retrospective Diagnosis & Critical Breakthrough Opportunity
1. **The 63.2% Ceiling Phenomenon**:
   Across Iterations 1 to 5, all candidates consistently achieve 70%–73% Accuracy on the 5 inner-validation subjects, but on the 10 locked outer-test subjects in Fold 1, performance has been capped at 62.4%–63.16%:
   - Iteration 2 (FiLM): 62.40%
   - Iteration 3 (Candidate Alpha): 63.09%
   - Iteration 4 (Candidate Beta): 62.59%
   - Iteration 5 (Candidate Gamma): 63.16%
2. **The Unnormalized Heart Rate Flaw (Grounding Hypothesis H-A)**:
   - In `exp013_models.py` and all previous candidate models, `x_meta[:, :, 3]` (30s HR mean in bpm) and `x_meta[:, :, 4]` (30s HR std in bpm) are passed unnormalized into `self.meta_proj = nn.Linear(5, 16)`.
   - Resting nocturnal heart rate varies enormously across human subjects (e.g. 48 bpm in athletic subjects vs 82 bpm in sedentary subjects).
   - Because `meta_proj` receives absolute heart rate in bpm, a held-out test subject whose basal sleeping heart rate is 75 bpm has their entire feature manifold shifted into what the model learned as "Wake / REM" states on training subjects with a 60 bpm baseline! This directly explains the thousands of Light-REM and Deep-Light boundary confusions on held-out subjects.
3. **Physiological Sleep Staging Ground Truth**:
   - Sleep stages are governed not by absolute heart rate, but by **relative parasympathetic/sympathetic heart rate modulation relative to the individual's nocturnal basal heart rate** ($\Delta \text{HR}(t) = \text{HR}(t) - \text{HR}_{\text{basal}}$) and standardized nocturnal z-scores:
     $$z_{\text{HR}}(t) = \frac{\text{HR}(t) - \mu_{\text{night}}}{\sigma_{\text{night}} + \epsilon}$$
4. **Proven Sequence Dynamics from EXP-019 (DEC-027)**:
   - In EXP-019, `MultiScaleConventionalCNNBiGRU` with multi-scale temporal convolutions ($k \in \{1, 3, 5\}$) achieved **65.72% Accuracy** on Fold 1 outer test at Epoch 53.
   - Candidate Gamma demonstrated that a direct skip connection to the classification head provides a strong gradient highway that raised outer-test accuracy to 63.16%.

---

## 3. Mission & Requirements for Iteration 6 (Candidate Delta)
Formulate **Candidate Delta** targeting Hypotheses H-F and H-A:
1. **Structural Mutation Proposal**:
   - **Physiological Basal Standardization Module**:
     * In the forward pass, dynamically compute each recording's nocturnal basal/mean and standard deviation using the valid sequence mask.
     * Standardize both the raw IHR signals and the `x_meta` HR columns (columns 3 & 4) to nocturnal relative deviations before projection.
   - **Multi-Scale Sequence Context**:
     * Integrate sequence-level multi-scale temporal convolutions ($k \in \{1, 3, 5\}$ along sequence dimension $T$) with depthwise bottleneck projection.
   - **Direct Feature Skip Connection**:
     * Retain direct intra-epoch feature skip connection to the classifier head.
   - **Strict Parameter Budget**:
     * Exactly within or as close to the 162k–196k parameter range as possible (>90% compression vs 5.47M SLAMSS-IFS). Budget cap: 196,728 parameters.
2. **Artifact Deliverables**:
   - Module: `.agents/teamwork/teamwork_explorer_r6_1/proposed_nas_iteration6_model.py`
   - Unit Test Suite: `.agents/teamwork/teamwork_explorer_r6_1/test_candidate_delta.py` verifying exact parameter count ($\le 196,728$), forward/backward execution, nocturnal normalization invariants, and step-0 stability. Run and verify 100% pass via `./.venv/bin/python`.
   - Comprehensive Analysis: `.agents/teamwork/teamwork_explorer_r6_1/analysis.md`
   - Handoff Report: `.agents/teamwork/teamwork_explorer_r6_1/handoff.md`
3. **Completion Signal**:
   - Send completion message to parent orchestrator via `send_message`.


## 2026-11-15T06:50:59Z
[Message] timestamp=2026-11-15T06:50:59Z sender=5bbd46d9-608b-43f0-a1c2-9eb9cfad1c91 priority=MESSAGE_PRIORITY_HIGH content=You are teamwork_explorer_r6_1. Read your dispatch instructions at:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_explorer_r6_1/DISPATCH.md
and the authoritative request at:
/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/ORIGINAL_REQUEST.md.

Formulate Candidate Delta for NAS Iteration 6 targeting Hypotheses H-F and H-A within the 162k–196k parameter budget, incorporating physiological basal standardization of raw HR/meta features, multi-scale sequence temporal fusion, and direct feature skip connections to eliminate the inter-subject resting HR shift barrier.
Implement proposed module and unit tests in your working directory (.agents/teamwork/teamwork_explorer_r6_1/), verify unit test pass with ./.venv/bin/python, write analysis.md and handoff.md, and send completion message via send_message.
