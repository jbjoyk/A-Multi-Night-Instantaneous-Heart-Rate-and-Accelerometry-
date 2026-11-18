# Soft Handoff: Project Orchestrator (Generation 0 -> Generation 1)

**Agent**: `teamwork_orchestrator` (Gen 0)  
**Successor**: `teamwork_orchestrator` (Gen 1)  
**Parent Conversation ID**: `fb288bfc-537f-41c9-8fa6-e473a32ff9a1`  
**Timestamp**: 2026-11-15T07:51:00Z  
**Workspace**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels`  
**Trigger**: Cumulative subagent spawn count reached 16 (16/16 spawns completed).

---

## 1. Observation

### 1.1 Mission & Authoritative Objective
Continuous Neural Architecture Search (NAS) syndicate targeting hypotheses H-F (cross-modal IHR-ACC interaction) and H-A (cross-modal/adaptive architecture with statistically significant gain across all 47 subjects in BIDSleep) until definitively beating the published SLAMSS-IFS paper model (Table V in Song et al., IEEE TBME 2026: **71.04% Accuracy, 0.7079 Weighted-F1**, 5.47M parameters).
- **Fast Proxy Filter (Fold 1, 50 epochs, seed 42 on cuda:0)**: Threshold: Proxy Accuracy $\ge 70.0\%$ AND Proxy Weighted-F1 $\ge 0.6900$. Revert if below.
- **Definitive Cohort Evaluation**: 5-fold cross-validation across all 47 subjects (80 epochs, 3 seeds: 42, 123, 2024) parallelized on `cuda:0` and `cuda:1`. Target: Aggregated Accuracy > 71.04% AND Weighted-F1 > 0.7079.
- **Parameter Budget**: 162k–196k parameters (>90% compression vs 5.47M). Invariant batch size (4), optimizer (Adam, lr=1e-4, weight_decay=0.0), sequence length (600), RCCLoss ($\lambda=0.01$).

### 1.2 Iteration History & Empirical Progression (Iterations 1 through 8)
Across Generation 0, eight architectural generations were systematically investigated, implemented, and benchmarked on the locked Fold 1 outer-test partition (10 subjects, 54 nights, 32,098 windows):

| Iteration | Model / Candidate | Params | Inner-Val (Peak) | Outer-Test Acc | Outer-Test WF1 | Cohen's Kappa | Gate Verdict |
|---|---|---|---|---|---|---|---|
| Model A | `ConventionalCNNBiGRU` | 163,940 | ~69.0% Acc | 61.9%–63.3% | ~0.620 | ~0.44 | Baseline |
| Iteration 1 | `MultiScaleConventionalCNNBiGRU` | 192,692 | 67.4% Acc | 61.6% | 0.601 | 0.42 | REVERT (Blurred cross-modal timing, REM F1=0.1287) |
| Iteration 2 | Candidate B: Cross-Modal FiLM | 193,380 | 69.56% Acc | 62.40% | 0.6238 | 0.451 | REVERT (Affine FiLM overfit training resting HR) |
| Iteration 3 | Candidate Alpha: CACMF | 190,502 | 72.53% Acc | 63.09% | 0.6314 | 0.468 | REVERT (Static gating MLP Light-REM confusion) |
| Iteration 4 | Candidate Beta: Localized Cosine CACA | 186,870 | 71.26% Acc | 62.59% | 0.6236 | 0.458 | REVERT (W=5 attention did not solve inter-subject shift) |
| Iteration 5 | Candidate Gamma: IESA + Skip Highway | 193,156 | 69.34% Acc | 63.16% | 0.6312 | 0.472 | REVERT (Direct skip boosted REM recall to 54.76%) |
| Iteration 6 | Candidate Delta: Physiological Basal PBMF | 195,346 | 70.60% Acc | 62.11% (Ep 40) / 63.09% (Ep 50) | 0.6231 | 0.459 | REVERT (Global nightly z-score collapsed non-stationarity) |
| Iteration 7 | Candidate Epsilon: Local Ultradian LUBS | 195,708 | 70.42% Acc | 64.27% | 0.6435 | 0.4806 | REVERT (Running W=91 LUBS boosted REM recall to 59.64%) |
| **Iteration 8** | **Candidate Zeta: Dual-Scale MUP + DSLBS** | **195,892** | **70.66% Acc** | **65.54%** | **0.6556** | **0.5051** | **REVERT (All-Time Project Best; Deep recall 70.88%, REM recall 67.98%)** |

---

## 2. Logic Chain & Core Physiological Findings

1. **Resolution of Non-Stationarity**:
   - In Candidate Delta (Iteration 6), global whole-night z-scoring failed because early sleep is slow-wave (low HR ~50 bpm) and late sleep is REM (autonomic surges ~68 bpm). Global normalization caused massive Light-REM confusion.
   - Candidate Epsilon (Iteration 7) resolved this by introducing **Local Ultradian Basal Standardization (LUBS)** across a running 45.5-minute window ($W=91$). This immediately pushed outer-test accuracy to 64.27% and reduced Light->REM false positives by 34.6%.
2. **Resolution of Deep-Light Autonomic Confusion**:
   - Analysis of Candidate Epsilon revealed that 3,758 windows were confused across the Deep-Light boundary (2,290 Deep windows misclassified as Light, 1,468 Light windows misclassified as Deep).
   - Candidate Zeta (Iteration 8) introduced **Dual-Scale Local Basal Standardization (DSLBS)** and **Multi-Scale Ultradian Temporal Pyramids (MUP)** combining micro timescale ($W_{\text{micro}} = 15$ epochs = 7.5 min) and macro timescale ($W_{\text{macro}} = 91$ epochs = 45.5 min).
   - The **Autonomic Contrast Differential** $\Delta z(t) = \frac{\mu_{\text{micro}}(t) - \mu_{\text{macro}}(t)}{\sigma_{\text{macro}}(t)}$ cleanly separated sustained parasympathetic troughs (Deep sleep: $\Delta z \approx 0.0, z_{\text{macro}} < -0.5$) from transient micro-arousals (Light sleep: $\Delta z > 0.0$).
   - This drove outer-test accuracy to an all-time record of **65.54% Accuracy and 0.6556 Weighted-F1**, with Cohen's Kappa breaking above 0.50 (0.5051) for the first time, Deep recall reaching 70.88%, and REM recall reaching 67.98%.
3. **The Remaining Frontier for Iteration 9 (Candidate Eta)**:
   In Candidate Zeta's outer-test confusion matrix:
   ```
   [[1901,  903,  149,  253],   # Wake  (support: 3,206)
    [ 575, 8689, 1771, 2829],   # Light (support: 13,864)
    [  88, 1809, 5613,  409],   # Deep  (support: 7,919)
    [  78, 1685,  513, 4833]]   # REM   (support: 7,109)
   ```
   The largest remaining error cluster is:
   - **Light sleep vs REM**: 2,829 Light windows classified as REM (20.4% of Light sleep) and 1,685 REM windows classified as Light sleep (23.7% of REM).
   - **Somatomotor Gating**: Both Light sleep and REM exhibit zero gross movement, but REM exhibits motor atonia with autonomic irregularity, while Light sleep exhibits variable autonomic tone with micro-movements. Successor Gen 1 should target **Somatomotor Stillness Gating** (cross-modal ACC variance thresholding modulating HR features) and/or **Adaptive Class-Margin Loss** to conquer the 70.0% proxy threshold.

---

## 3. Milestone State & Search Status

| Milestone | Description | Status |
|---|---|---|
| M1 | Survey & Evaluation Pipeline Verification | DONE |
| M2 | NAS Iterations 2 through 8 (Candidates B through Zeta) | DONE (All proxy-evaluated; high-water mark: 65.54% Acc / 0.6556 WF1) |
| M3 | Continuous Search Loop (Iteration 9+) -> 5-Fold Cross-Validation | IN_PROGRESS (Gen 1 to dispatch Iteration 9) |
| M4 | Victory Condition (Aggregated 5-fold Acc > 71.04% & WF1 > 0.7079) | PLANNED |

---

## 4. Key Constraints for Successor (Gen 1)

1. **DISPATCH-ONLY PERSONA**: NEVER directly write, modify, or create source code files. NEVER run training scripts directly. Strictly dispatch subagents (`teamwork_explorer`, `teamwork_worker`).
2. **Metadata Files Only**: Orchestrator edits only metadata files (`.md`) in `.agents/teamwork/orchestrator/`.
3. **Budget Constraint**: Model parameters must remain close to 162k–196k parameters (>90% compression vs 5.47M paper model, cap $\le 196,728$).
4. **Invariant Hyperparameters**: Batch size 4, Adam(lr=1e-4, weight_decay=0.0), seq_len 600, RCCLoss ($\lambda=0.01$).
5. **Proxy Gating**: Filter threshold: Proxy Accuracy $\ge 70.0\%$ AND Proxy Weighted-F1 $\ge 0.6900$ on Fold 1 (50 epochs, seed 42 on `cuda:0`).
6. **Continuous Git Sync**: Commit and push all experimental code and JSON results to `origin main`.
7. **Parent Passthrough**: Parent Conversation ID is `fb288bfc-537f-41c9-8fa6-e473a32ff9a1`.

---

## 5. Concrete Next Steps for Successor

1. Initialize Gen 1 Briefing and reset spawn count to 0 / 16.
2. Start heartbeat cron `schedule(CronExpression="*/10 * * * *", Prompt="Heartbeat check: inspect subagent progress and update progress.md")`.
3. Dispatch `teamwork_explorer` for Iteration 9 (Candidate Eta) targeting:
   - Somatomotor stillness gating (coupling ACC energy with autonomic contrast).
   - Multi-Scale Ultradian Temporal Pyramids (MUP) + Dual-Scale Local Basal Standardization (DSLBS).
   - Retaining 4-head IESA and 144-dim intra-epoch skip highway.
4. Dispatch `teamwork_worker` to deploy Candidate Eta, run unit tests, and evaluate 50-epoch Fold 1 proxy.
5. If proxy passes ($\ge 70.0\%$ Acc, $\ge 0.6900$ WF1), launch full 5-fold CV across `cuda:0` and `cuda:1`. If not, revert and continue loop.
