# PRE-REGISTRATION: EXP-019, H-A Multi-Night Within-Person Adaptation

**Status:** LOCKED (Committed before Stage C Confirmatory Execution).
**Authoritative Hash / Date:** 2026-11-16
**Experiment ID:** EXP-019

---

## 1. Purpose and scientific question

> Does adapting a subject-independent sleep-staging model with 2 of a held-out person's earlier labeled nights improve staging on that person's later nights, compared with the same model without adaptation?

This is hypothesis **H-A** in `RESEARCH_DIRECTION_v3_FULL.md`. Motivation from existing evidence: EXP-014 (75 runs, 47 subjects) found no detectable gain from architectural mechanisms, and fold-level accuracy ranged from about 62.5% to 72.3%, so between-subject variability dominates. This experiment tests whether using a person's own data reduces that variability. It is **not** an architecture experiment.

---

## 2. Hypotheses (LOCKED at commit)

**H-A (primary, confirmatory).** Fine-tuning the zero-shot (ZS) model on a held-out subject's first 2 labeled nights (FT-2) increases Macro-F1 on that subject's remaining nights relative to ZS by at least +1.50 percentage points, with the 95% bootstrap confidence interval strictly excluding zero.

**Secondary confirmatory hypotheses:**

- **S1 (personalization-specific effect).** FT-2 on the subject's **own** 2 nights outperforms the control FT-other-2, which fine-tunes with identical hyperparameters on 2 nights from a single other training subject of the same fold. This separates "personal data helps" from "any extra optimization helps."
- **S2.** FT-2 outperforms PS-2 (a per-subject model trained from scratch on the same 2 nights).
- **S3.** Label-free adaptation UA-2 (InfoMax entropy minimization with marginal diversity penalty on the 2 calibration nights' unlabeled inputs) outperforms ZS.

**Descriptive (no confirmatory claim):** learning curve over k = 1, 2, 3, 4; heterogeneity of gain vs ZS performance; effect by night index; per-stage F1; clinical sleep-summary agreement (TST, SOL, WASO, sleep efficiency, REM%, deep%).

---

## 3. Single factor and constants

**Single factor:** whether, and how, the held-out subject's own earlier nights are used.

**Held constant across all arms:**

- Architecture: Model A (`ConventionalCNNBiGRU`, exactly 163,940 parameters). No architectural change in any arm.
- ZS backbone per outer fold and seed: The EXP-018 Phase 2 L=1200 Model A checkpoints, trained on training subjects only, circadian feature neutralized (`time_feat = 0`). All 15 checkpoints ($5\text{ folds} \times 3\text{ seeds}$ [42, 123, 2024]) exist on disk at `experiments/EXP-018/checkpoints/modelA_L1200_fold{fold}_seed{seed}.pt`. Every checkpoint was verified against historical EXP-018 Phase 2 results within $\le 0.00010$ Macro-F1 difference. Backbones are **never** modified after training.
- Inference: `SlidingWindowPredictor` at L=1200, whole-night forward pass with sliding-window probability stitching when N > L.
- Task: 4-class (Wake/Light/Deep/REM). Valid epochs: `(target >= 0) & (target < 4)` and mask (as in EXP-017). The mapping of raw stages: Wake = 0, Light = 1 (N1 and N2 combined), Deep = 2 (N3), REM = 3. Unknown stages and padded positions are mapped to `-100` (`ignore_index`).
- Labels: `expert_label` from `labels.mat`.
- Seeds: 42, 123, 2024. Adaptation seed = backbone seed.
- Fold manifests: EXP-014/EXP-018 Phase 2 subject-level folds (47 subjects, 5 outer folds) from `experiments/EXP-014/EXP014_MANIFESTS/` and `review_package/fold_manifests.csv`. Manifest SHA256 hash: `36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf`.

---

## 4. Eligibility, night ordering and arms

**Eligible subjects:** Exactly 46 subjects having at least 3 usable nights in the manifest ($n_{\text{usable}} \ge 3$), providing a total of 152 outer evaluation nights across the cohort.
- **Excluded subject:** `Bidslab49` ($n_{\text{usable}} = 2$), excluded because reserving $k=2$ calibration nights leaves zero evaluation nights.

**Night ordering:** by `recStart` / night index (strictly chronological). Calibration nights C = first $k=2$ usable nights. Evaluation nights E = all later usable nights ($k \ge 1$). C and E never overlap. Every arm is evaluated on the **same** E. ZS metrics on E are computed from ZS predictions on those nights only.

**Arms** (each test subject starts every arm from the same ZS checkpoint via in-memory deep copy, with zero information carried between subjects):

| Arm | Description |
|---|---|
| ZS | Backbone as trained, no subject data |
| FT-2 | Fine-tune ZS on the subject's first 2 labeled nights |
| FT-other-2 | Fine-tune ZS with identical hyperparameters on 2 nights from a single other training subject of the same fold (deterministic donor selection) |
| PS-2 | Same architecture, trained from scratch on the subject's first 2 labeled nights only |
| UA-2 | ZS with label-free InfoMax adaptation (entropy minimization + marginal diversity penalty) on the subject's first 2 nights' wearable inputs |

---

## 5. Hyperparameter selection (nested, no evaluation-night access)

Adaptation hyperparameters are chosen **per outer fold**, using only that fold's 5 inner-validation subjects (25 subjects across 5 folds; calibration nights = first 2, evaluation nights = remaining nights). Outer test subjects are never used for selection.

**Pre-declared grids (final; evaluated in Stage D):**
- FT-2 and FT-other-2 (shared config): learning rate {3e-5, 1e-4, 3e-4} × epochs {5, 15} × trainable scope {prediction head + last recurrent layer (`head_gru`), all parameters (`all`)} = 12 configs.
- PS-2: learning rate {1e-4, 1e-3} × steps {300, 500} = 4 configs.
- UA-2: learning rate {1e-5, 1e-4} × steps {20, 100} = 4 configs.

**Locked Winning Configurations Selected in Stage D:**

| Outer Fold | FT-2 / FT-other-2 Selected Config | PS-2 Selected Config | UA-2 Selected Config | Inner-Val FT Macro-F1 |
|---|---|---|---|---|
| **Fold 1** | lr = 1e-4, epochs = 5, scope = all | lr = 1e-3, steps = 300 | lr = 1e-5, steps = 20 | 0.7133 |
| **Fold 2** | lr = 3e-5, epochs = 15, scope = all | lr = 1e-3, steps = 500 | lr = 1e-5, steps = 20 | 0.7025 |
| **Fold 3** | lr = 3e-5, epochs = 15, scope = all | lr = 1e-3, steps = 300 | lr = 1e-5, steps = 20 | 0.6587 |
| **Fold 4** | lr = 3e-5, epochs = 5, scope = head_gru | lr = 1e-3, steps = 300 | lr = 1e-5, steps = 20 | 0.6841 |
| **Fold 5** | lr = 3e-5, epochs = 5, scope = all | lr = 1e-3, steps = 300 | lr = 1e-5, steps = 20 | 0.6160 |

---

## 6. Primary estimand and decision rule (LOCKED)

**Estimand:** mean over the 46 eligible subjects of the paired difference in subject-level Macro-F1 (FT-2 minus ZS) on evaluation nights. Subject-level Macro-F1 pools all valid epochs of that subject's evaluation nights, then averages across the 3 seeds **within subject** before bootstrapping. Subjects are the unit of analysis.

**Inference:** 10,000-resample percentile bootstrap over subjects, 95% CI. Wilcoxon signed-rank on subject-level differences as a confirmatory non-parametric test.

**Verdict Rules:**

- **SUPPORTED:** $\bar{\Delta} \ge +1.50\text{ pp}$ AND the bootstrap 95% CI lower bound > 0.
- **NOT SUPPORTED:** the bootstrap 95% CI upper bound < +1.50 pp (the data rule out an effect of practical size).
- **INCONCLUSIVE:** anything else.

**Stage D Variance & Practical Threshold Confirmation:**
- Empirical standard deviation of paired differences ($FT\text{-}2 - ZS$) across the 25 inner-validation subjects in Stage D: $s_{\Delta} = 2.2432\text{ pp}$.
- Continuous 80%-power threshold ($0.4131 \times s_{\Delta}$): $0.9267\text{ pp}$.
- Since $s_{\Delta} \le 5.0\text{ pp}$, no threshold escalation is triggered.
- **Confirmed Practical Threshold:** $\max(1.50, 0.9267) = \mathbf{+1.50\text{ pp}}$ (locked for Stage C).

**Secondary tests:** S1, S2, S3 use the same estimand, inference and verdict logic:
- S1 supported if $FT\text{-}2 > FT\text{-}other\text{-}2$ with 95% CI lower bound > 0 and Wilcoxon $p < 0.05$.
- S2 supported if $FT\text{-}2 > PS\text{-}2$ with 95% CI lower bound > 0 and Wilcoxon $p < 0.05$.
- S3 supported if $UA\text{-}2 > ZS$ with 95% CI lower bound > 0 and Wilcoxon $p < 0.05$.

---

## 7. Safeguards against leakage and selection bias

- Backbones are trained on training subjects only; inner-validation subjects are used for Stage D selection only; outer test subjects are used strictly for Stage C confirmatory evaluation.
- Calibration nights of a test subject never appear in that subject's evaluation set ($C \cap E = \emptyset$). Chronological split only.
- Evaluation nights are never used to choose hyperparameters, epochs, stopping points or arms.
- Zero per-subject fine-tuning state is shared across subjects. Each subject starts from the unmodified ZS checkpoint via in-memory deep copy.
- No test-subject information enters normalization beyond what the ZS pipeline already does per night.
- Confirmatory execution script: `experiments/EXP-019/run_exp019_stage_c.py`.
- Report every subject, fold, seed and arm. No post-hoc subject exclusion.

---

## 8. Required outputs

Per-subject, per-seed, per-arm metrics table (CSV and JSON in `experiments/EXP-019/exp019_stage_c_results.json`); selected configs per fold (`exp019_stage_d_selected_configs.json`); paired bootstrap table; hypothesis verdict summaries in `EXP019_FINAL_REPORT.md`.

---

## 9. Not claimed

- No claim about architecture. Architecture search is closed for this project (EXP-014).
- No claim of clinical-grade accuracy, disorder detection or real-time use.
- No SLAMSS-IFS superiority claim from this experiment.
- Labeled calibration nights require EEG-based labels; the deployment story is "short headband calibration, then watch-only." UA-2 is the label-free variant.

---

## 10. Known limitations to state in the paper

- Healthy adults, single site, headband-derived labels with single-scorer review.
- First-night and night-order effects may confound chronological adaptation.
- Few nights per subject (up to 7), so learning curves at k ≥ 3 rest on fewer subjects.

---

## 11. Amendments

| Date | Section | Change | Reason |
|---|---|---|---|
| 2026-11-16 | All | Resolved all TO FILL / TO VERIFY placeholders; locked winning Stage D configurations; confirmed practical threshold at +1.50 pp based on empirical s_delta = 2.2432 pp; locked status to LOCKED. | Phase-0 pre-registration lock prior to Stage C execution. |
