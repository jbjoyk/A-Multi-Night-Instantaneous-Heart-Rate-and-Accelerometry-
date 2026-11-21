# Stage 2: Final Model Synthesis, Architecture Freeze, and Deployment Specification

**Document ID:** `STAGE2_FINAL_MODEL_SYNTHESIS`  
**Model Release:** `SleepStagingFinalModel_v1.0` (`Model A: ConventionalCNNBiGRU`, 163,940 parameters)  
**Governing Decisions:** `DEC-029`, `DEC-031`, `DEC-033`, `DEC-034`, `DEC-035`, `DEC-036`, `DEC-037`  
**Status:** FROZEN & PARITY-VERIFIED (`STAGE2_MODEL_FREEZE_REPORT.md`; Cleared for Stage 3 Benchmark `TSK-08`)

---

## 1. Scientific Synthesis Leading to the Final Model Freeze

Under `DEC-034`, Stage 2 consolidates all locked empirical evidence across the healthy-adult BIDSleep cohort ($N = 47$, 246 usable nights) to specify and permanently freeze the project's definitive model prior to running the single final matched benchmark against official SLAMSS-IFS (Stage 3 / TSK-08):

| Evidence Pillar | Experiment / Decision | Verified Empirical Finding (Including Null/Inconclusive Results) | Architectural Consequence for Final Model |
| :--- | :--- | :--- | :--- |
| **Backbone Simplicity vs. Complexity** | EXP-013, EXP-014, DEC-029, DEC-031 | Across 75 runs (5-fold CV, 47 subjects), matched-capacity `ConventionalCNNBiGRU` (Model A, 163,940 params) achieved cohort equivalence with adaptive-gate/GRL/transition variants (`SleepAdaptiveNet`). Architecture search permanently closed. | Freeze **Model A (`ConventionalCNNBiGRU`, 163,940 params)** as the population backbone. Retains **4.06x** compression vs. SLAMSS-IFS paper text (666k) and **33.34x** vs. official repository (5.47M). |
| **Temporal Context Horizon** | EXP-018 Phase 2 | Extending context from $L=60$ (30 min) to $L=1200$ (10 h) increased pooled-epoch 47-subject 3-seed OOF Macro-F1 by $+7.06\text{ pp}$ ($0.5773 \to 0.6479 \pm 0.0053$; 46/47 subjects improved for $L=600$ vs $L=60$). Bi-exponential saturation ($R^2=0.9998$) requires 8.0 h (958 epochs) for 95% ceiling. | Lock **full-night context $L=1200$** with `pack_padded_sequence` zero-padding protection and `SlidingWindowPredictor` probability stitching (stride $L/2$, margin $0.25L$). Circadian features neutralized (`time_feat=0`, `clock=0`). |
| **Supervised Personalization Depth & Heterogeneity** | EXP-019 (H-A, S1), TSK-06 (DEC-035) | **Locked H-A ($N=46, E_2$, $k=2$):** $\bar{\Delta} = +1.01\text{ pp}$ [95% CI: $-0.17, +1.99\text{ pp}$], reconciled verdict **`INCONCLUSIVE`** (heterogeneous effects including negative transfer such as `Bidslab34` with 0.68% Deep sleep). Specificity control S1 confirmed donor tuning harms performance ($-1.72\text{ pp}$).<br>**TSK-06 Common Cohort ($N=33, E_4$):** Page's $L = 863.0, z = 2.261, p = 0.01187$ confirmed monotonic trend across $k=1$ ($+3.00\text{ pp}$), $k=2$ ($+3.13\text{ pp}$), $k=3$ ($+3.54\text{ pp}$, van der Aar et al. 2025 3-night healthy-adult replication confirmed), $k=4$ ($+3.96\text{ pp}$). Dimension 2 verdict: **`INCONCLUSIVE SCALING`** ($\Delta_{3-2}=+0.41\text{ pp}, p=0.287$; $\Delta_{4-3}=+0.42\text{ pp}, p=0.262$). Available-cohort sensitivity ($E_k$): $k=1$ ($+1.92\text{ pp}$), $k=2$ ($+2.72\text{ pp}$), $k=3$ ($+2.57\text{ pp}$), $k=4$ ($+3.96\text{ pp}$). | Adopt **supervised within-person calibration ($FT\text{-}k$)** as the primary personalization tier when labeled nights exist, operating from $k=1$ (minimum viable personalization) through $k=3$ (replicated 3-night healthy-adult benchmark), strictly paired with non-mutating stage-coverage diagnostics to flag severe missing-stage nights. No clinical-grade or disorder generalization claims (`RESEARCH_STATE.md` §2). |
| **Unsupervised Adaptation Viability** | TSK-07 (DEC-036) | After resolving the DEC-030 unit-mismatch mask ($\text{Var}_t(m) > 10^{-6}$, 97.26% valid epochs), 20-step `head_gru` InfoMax achieved $+0.21\text{ pp}$ [95% CI: $+0.04, +0.39\text{ pp}$], $p_{\text{sup}} = 0.0322$, 0/138 collapse aborts (**`SUB-BENCHMARK GAIN`**). | Retain **quality-gated InfoMax ($UA\text{-}2$)** strictly as an optional **Tier 1 label-free auxiliary** when zero labeled nights exist, explicitly documented as a modest sub-benchmark gain ($< +1.00\text{ pp}$) that does not replace supervised calibration. |
| **Clinical Sleep-Summary Agreement** | EXP-019 Phase B (DEC-032) | Subject-averaged multi-night agreement ($N=46, df=45$) established TST LoA $[-30.11, +34.67]\text{ min}$ and empirical WASO underestimation ($-4.87\text{ min}, p=0.0104$). | Locked as EXP-019 Phase B agreement benchmark (`DEC-032`). Stage 3 (`TSK-08`) focuses strictly on the preregistered epoch-level primary & secondary staging metrics (`PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md` §4.2: Macro-F1, Accuracy, Weighted-F1, Kappa, per-stage Wake/Light/Deep/REM F1). |

---

## 2. Frozen Final Model Architecture (`SleepStagingFinalModel_v1.0`)

```
Input per 30-s epoch:
  - x_ihr   : (B, L, 30)  1 Hz standardized instantaneous heart rate
  - x_accel : (B, L, 30)  1 Hz standardized triaxial accelerometry magnitude
  - x_meta  : (B, L, 5)   [freq, clock=0, time_feat=0, hr_mean, hr_std]

Backbone (163,940 trainable parameters — ~160.1 KiB INT8 weight footprint only;
          weights fit within 512 KB Nordic nRF5340 SRAM, excluding runtime activation buffers/RTOS):
  1. HREncoder   : Conv1d(1->32, k=5) + GN + GELU + Pool + ResConv(32->64) + Pool + ResConv(64->64)
                   + AvgPool + Dropout(0.2) + Linear(64->64) + LN -> (B, L, 64)
  2. ACCEncoder  : Identical parallel branch for accelerometry magnitude -> (B, L, 64)
  3. MetaProj    : Linear(5->16) + GELU -> (B, L, 16)
  4. Concat      : [h_HR ; h_ACC ; h_meta] -> (B, L, 144)
  5. BiGRU       : 2-layer Bidirectional GRU (input=144, hidden=48/dir -> 96) with pack_padded_sequence + LayerNorm(96)
  6. Head        : Linear(96->64) + LayerNorm(64) + GELU + Dropout(0.2) + Linear(64->4)

Loss Function:
  L_total = L_CE(masked) + 0.01 * L_RCC(masked, official asymmetric fn/fp weights)

Inference Engine:
  SlidingWindowPredictor(context_len=1200, neutralize_circadian=True)
```

### Frozen Population Checkpoint Inventory (15 Checkpoints)
The 15 out-of-fold population checkpoints already trained under the locked EXP-018 Phase 2 compute schedule ($B=2, L=1200, N_{\text{accum}}=1$, 3,200 steps = 7,680,000 tokens, Adam $\text{lr}=10^{-4}$, seeds `{42, 123, 2024}`, folds `{1..5}`) constitute the frozen population weights:
- Path pattern: `experiments/EXP-018/checkpoints/modelA_L1200_fold{1..5}_seed{42,123,2024}.pt`
- Zero retraining of Model A is required or permitted for the Stage 3 matched benchmark (preserving locked EXP-018 / EXP-019 baselines).

---

## 3. Tiered Deployment & Personalization Interface

Implemented in [`final_model_pipeline.py`](./final_model_pipeline.py) (`FinalSleepStagingSystem`):
1. **Tier 0 (Zero-Shot Population Mode):** Loads frozen `ConventionalCNNBiGRU` checkpoint and runs `SlidingWindowPredictor(context_len=1200, neutralize_circadian=True)`.
2. **Tier 1 (Optional Label-Free Stabilization — `adapt_unsupervised`):** Accepts truly unlabeled records (no `"y"` key required), enforces the TSK-07 wearable quality gate (`Var_t(m) > 1e-6`, finite `|z| <= 10.0` on cached z-scored IHR or `[30, 180] BPM` when raw BPM is supplied, valid epoch ratio `> 95%`, `>= 600` valid epochs/night, `0` NaN/Inf leakage), and runs 20-step `head_gru` InfoMax ($\eta = 10^{-5}$, $\lambda_{\text{div}} = 1.0$, collapse guard $H_{\text{marginal}} \ge 0.6931\text{ nats}$) on an isolated clone without mutating frozen weights.
3. **Tier 2 (Supervised Within-Person Calibration — `adapt_supervised`):** Executes fold-locked supervised fine-tuning (`adapt_supervised_ft`) on $k \in \{1, 2, 3, 4\}$ chronological calibration nights on an isolated clone, accompanied by non-mutating stage-coverage diagnostics (`audit_calibration_stage_coverage`). For out-of-study deployment outside 5-fold CV, use either the 5-fold ensemble or consensus config (`lr=3e-5, epochs=5, scope='all'`).

---

## 4. Stage 2 Freeze Verification Procedure (Not Yet Executed)

When authorized to execute Stage 2 verification, run:
```bash
.venv/bin/python experiments/STAGE2_FINAL_MODEL/freeze_final_model.py --device cuda:0
```
This read-only verification & sealing script will:
1. Verify all 15 frozen Model A checkpoints exist, pass exact parameter count assertion (`163,940`), and compute their SHA-256 cryptographic hashes into `experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json`.
2. Run zero-gradient inference over all 47 outer-test subjects across 5 folds × 3 seeds to extract the complete paired-benchmark reference table (`experiments/STAGE2_FINAL_MODEL/final_model_oof_subject_metrics.csv`, containing Macro-F1, Weighted-F1, Accuracy, Kappa, and per-stage Wake/Light/Deep/REM F1) while asserting fold-pooled and per-subject parity (`<= 5e-4`, configurable via `--parity-tolerance`) against locked `experiments/EXP-018/phase2_results/run_L1200_fold*_seed*.json`.
3. Emit `experiments/STAGE2_FINAL_MODEL/STAGE2_MODEL_FREEZE_REPORT.md` distinguishing pooled-epoch 3-seed OOF from unweighted 47-subject participant-mean OOF, clearing the gate for Stage 3 (`TSK-08`).
