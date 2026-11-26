# PRE-REGISTRATION (PROPOSAL): EXP-033, Personalization and External Evaluation on the Recipe-Consistent W100 Backbone

- **Status:** DRAFT until committed. Commit this file, record the hash in `EXPERIMENTS.md`, freeze the analysis script (record its commit hash), and only then run any fine-tuning. After the lock commit, Sections 3 to 9 are immutable; corrections go in Section 13 as dated amendments.
- **Experiment ID:** `EXP-033` (`experiments/EXP-033/`, `EXP033_FINAL_REPORT.md`).
- **Untouched:** `EXP-018` to `EXP-032` artifacts, frozen `Model A` checkpoints, `W100` checkpoints (`experiments/EXP-031/runs/W100/`), and the locked results of `EXP-019` and `EXP-020`.

---

## 1. Purpose

Every adaptation result in the current manuscripts (`FT-2`, the `k = 1` to `4` curve, `GATE`, `S1` to `S3`, Bland-Altman by arm) was computed on frozen `Model A`, whose training recipe differs from the recipe used for all later arms (`EXP-031` A1: `W100` outperforms frozen `Model A` by `+1.89 pp`). Earlier recipe-consistent backbones showed small or negative `FT-2` effects on the pipelines then used (plain 223k: `-1.19 pp`; `p1_100k`: `-0.57 pp`; `full_v5`: `+0.92 pp`). Whether two-night adaptation helps a recipe-consistent backbone is therefore unknown. This experiment answers it with rules fixed in advance.

**Questions:**
- **Q-ADAPT:** Does fine-tuning `W100` on a held-out person's first two labeled nights improve Macro-F1 on later nights?
- **Q-SPECIFIC:** Is any benefit specific to that person's own nights?
- **Q-DEPTH:** How does the effect change with 1 to 4 calibration nights?
- **Q-EXTERNAL:** How does `W100` perform zero-shot on SleepAccel relative to `Model A` and `SLAMSS-IFS`?

---

## 2. Disclosures

1. **Design-informed:** This design follows `EXP-019` (inconclusive), the `k`-curve, `EXP-020`, and the `EXP-031` results. Decision rules in Section 7 are fixed before any `W100` adaptation result exists.
2. **Backbone choice:** `W100` is the primary backbone because the `EXP-031` selection rule (smallest width within one paired standard error of the best eligible width), applied with retrained `W100` in place of frozen `Model A`, selects it. Applying the rule this way is not stated in `EXP-031` or its A1 amendment, so it is declared here. Other backbones are reported descriptively (Section 5), never chosen by outcome.
3. **Previous bug:** The multi-depth driver once let zero-shot evaluations inherit weights from earlier subjects' fine-tuning. Section 6 specifies safeguards.

---

## 3. Single Factor and Constants

- **Single factor:** Whether, and how, a held-out subject's own earlier nights are used to adapt a frozen, recipe-consistent backbone.
- **Backbone (primary):** `W100`, i.e., `Model A`'s architecture (`163,940` parameters) trained under the `EXP-031` recipe, 15 checkpoints (5 folds × 3 seeds), `experiments/EXP-031/runs/W100/checkpoints/`.
  - **Checkpoint SHA-256 digests:**
    - `W100_L1200_fold1_seed42.pt`: `9cbf86756d8f8cb037f7a8b8afd69ba2d125bd4b32abb4bcc7f870b6ea0e124b`
    - `W100_L1200_fold1_seed123.pt`: `d2a1b68448d32f5dc32a20e00e301c11f382ba13fa8e61df8eefbe160d20fee1`
    - `W100_L1200_fold1_seed2024.pt`: `b24597ecc7aafcfddc628e6f980a97bbc8d08ffdb4715876932272194e58bef1`
    - `W100_L1200_fold2_seed42.pt`: `f080601580dbb1f2783804a82e472faaa0810050506cb4fd6cbe55b7971d8c09`
    - `W100_L1200_fold2_seed123.pt`: `7b3568e6e88155275581e063829d134a52402c07c6dfcd59fda60ffd7b80b76e`
    - `W100_L1200_fold2_seed2024.pt`: `06ec27de92fadfa46e001c98d9e379e4bfbc1a508c6ebfbc331d1383ff688a1c`
    - `W100_L1200_fold3_seed42.pt`: `70363741a45e28d545d530de891ca350eba2f1c0c71d05b6f980cafbc05b870b`
    - `W100_L1200_fold3_seed123.pt`: `c74d5527050783b91fdadc2d195d5da84b84d712f2ba7896be3588f764a2af92`
    - `W100_L1200_fold3_seed2024.pt`: `a11a17c4a3b8de3e1eaab56179c60955f5ebe34e11d4a8a7fdee693324fd7add`
    - `W100_L1200_fold4_seed42.pt`: `fdac65033e373fae7ad4895bf1e7c42f4709a466200b626da2a4925827f2253f`
    - `W100_L1200_fold4_seed123.pt`: `6881ce59c9a03f7c2e6bb97943a7310a94d91c3e81716c630a8728d4a16e65a9`
    - `W100_L1200_fold4_seed2024.pt`: `88627e07f496c7a980371bcee97887cb4433bc3116d8922b92b6cb42319d3837`
    - `W100_L1200_fold5_seed42.pt`: `8cb2e46da46df11b4c56094ab31b29f8076822823723095dea56baea194a490f`
    - `W100_L1200_fold5_seed123.pt`: `08323ad53fe82cdba60c56998dd3df8b40377eb53bf08f241f9c54eaf7c030ec`
    - `W100_L1200_fold5_seed2024.pt`: `dd87d1275a5fe5f5fce1d1999348df9af174751055382b4b9ab4eb204804deda`
- **Held constant:**
  - Subject-disjoint fold manifests: `review_package/fold_manifests.csv` (SHA-256: `36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf`)
  - Full-night context: `L = 1200`, `LogitStitchingSlidingWindowPredictor`
  - 4-class task with training label mapping: `{0: Wake, 1: Light, 2: Deep, 3: REM}`, `-100: ignore`
  - Seeds: `42`, `123`, `2024` (adaptation seed equals backbone seed)
- **Nights:** Chronological by `recStart`. Calibration nights are the first `k` usable nights; evaluation nights are all later usable nights; the two never overlap. A subject is eligible for depth `k` if it has at least `k + 1` usable nights.

---

## 4. Arms

| Arm | Description |
|---|---|
| `ZS` | `W100` without adaptation (frozen checkpoint) |
| `FT-k` | `ZS` fine-tuned on the subject's first `k` labeled nights (`k = 1, 2, 3, 4`; primary `k = 2`) |
| `FT-other-2` | `ZS` fine-tuned with identical hyperparameters on 2 nights from another subject of the same fold's training set, averaged over 5 independent donor draws per subject and seed (fixed RNG seed) |
| `GATE-τ` | Use `ZS` predictions for a subject if its minimum class proportion in the calibration labels is below `τ`; otherwise use `FT-2` predictions. `τ ∈ {1%, 2%, 3%, 5%}`, all reported, none selected |

**Not repeated:** `PS-2` (from scratch; result known and uninformative), `UA-2` (label-free adaptation), and the KL-regularized arm (declared out of scope here).

---

## 5. Backbones Reported

- **Primary (confirmatory):** `W100` (`experiments/EXP-031/runs/W100/checkpoints/`, 15 checkpoints: 5 folds × 3 seeds).
- **Descriptive (same analysis, same rules, no verdict labels):**
  - `p1_100k`: `experiments/EXP-030/checkpoints/p1_100k/` (15 checkpoints: 5 folds × 3 seeds)
  - `P190`: `experiments/EXP-031/checkpoints/P190/` (15 checkpoints: 5 folds × 3 seeds)
  - `full_v5`: `experiments/EXP-029/checkpoints/full_v5/` (15 checkpoints: 5 folds × 3 seeds)
- **Legacy reference:** `EXP-019` and the `k`-curve results on frozen `Model A`, shown beside `W100`'s, labelled `"older recipe."`

No backbone is promoted or dropped by outcome. All four are reported in one table and one forest plot.

---

## 6. Protocol and Safeguards

- **Hyperparameter selection (nested, per backbone):** For each outer fold, select from the pre-declared grid using only that fold's 5 inner-validation subjects (calibration = first 2 nights, evaluation = remaining nights):
  - Learning rate: `{3e-5, 1e-4, 3e-4}`
  - Epochs: `{5, 15}`
  - Trainable scope: `{head plus last recurrent layer (head_gru), all parameters (all)}`
  - Total: `12` configurations.
  - Selection metric: mean subject-level Macro-F1 over seeds. Ties go to the smaller learning rate, then fewer epochs (then `head_gru` before `all` if still tied).
  - **Training details (matching `experiments/EXP-019/exp019_adaptation_core.py` `adapt_supervised_ft` and `exp013_models.py`):**
    - Optimizer: `torch.optim.Adam` (`betas=(0.9, 0.999)`, `weight_decay=0.0`)
    - Batch size: all `k` calibration nights for the subject in a single physical mini-batch (`batch_size = k`, i.e., `2` nights for `FT-2` / `FT-other-2` and `k ∈ {1, 2, 3, 4}` nights for `FT-k`; `1 epoch = 1 optimizer step`, `context_len = 1200`)
    - Gradient clipping: `torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)`
    - Loss: `CE + 0.01 * RCCLoss` (`out["total_loss"]`, masked cross-entropy plus `0.01 * RCCLoss` over valid epochs)
  - The same selected configuration is used for `FT-k` at every depth and for `FT-other-2`.

- **Isolation and reset guard (mandatory):**
  - Every `(subject, seed, arm, depth)` fine-tuning starts from a fresh deep copy of the frozen state dict.
  - Log a hash of the state dict before and after, per subject.
  - Unit test: `ZS` predictions for a subject must be bitwise identical before and after any other subject's fine-tuning. Run it before and after the full run.
  - `ZS` Macro-F1 over all nights must equal the `W100` out-of-fold per-subject value, and evaluation-night `ZS` values must come from these frozen predictions.

- **No outer-test information in selection:** Outer-test subjects' evaluation nights are never used to choose any hyperparameter, depth, or arm.

---

## 7. Hypotheses and Decision Rules (LOCKED at commit)

- **Outcome (all tests):** Subject-level Macro-F1 on evaluation nights, averaged over three seeds within subject.
- **Inference:** Paired differences, 10,000-resample participant-level percentile bootstrap 95% CI (seed `42`), two-sided Wilcoxon signed-rank, Cohen's `d_z`.

- **H-A2 (primary, confirmatory):** `W100` `FT-2` improves Macro-F1 over `W100` `ZS` on evaluation nights, `N = 46` (subjects with at least 3 usable nights). `Δ = FT-2 minus ZS`. Verdict (identical to the locked `H-A` rule):
  - **SUPPORTED:** `Δ ≥ +1.5 pp` and the CI lower bound `> 0`.
  - **NOT SUPPORTED (practically):** The CI upper bound `< +1.5 pp`.
  - **INCONCLUSIVE:** Otherwise.

- **S-SPEC (confirmatory, secondary):** `FT-2 minus FT-other-2` (averaged over 5 donor draws), same verdict rule. Also report `FT-other-2 minus ZS` (donor harm check) descriptively.

- **Descriptive (no verdict labels):**
  - **Depth curve:** `Δ` (`FT-k minus ZS`) for `k = 1` to `4`, on the fixed common cohort `E4` (`N = 33`, nights after night 4) and on available cohorts `E_k` (`N = 46, 46, 42, 33`). Page's `L` trend test with the raw two-sided `p` as primary and the raw one-sided `p` as secondary; every `p`-value labelled raw or Holm-adjusted, and one-sided or two-sided.
  - **GATE sweep:** For each `τ`: number of participants abstained, mean `Δ` vs `ZS`, minimum `Δ`, 10th-percentile `Δ`, number with `Δ ≤ -5 pp`.
  - **Backbone table:** The same statistics for all four backbones.
  - Per-stage F1, per-subject `Δ` distribution, and `Δ` by night index.

- **Claim rules (locked):**
  - A personalization benefit may be claimed only if `H-A2` is **SUPPORTED**.
  - A calibration-depth effect may be claimed only if Page's raw two-sided `p < 0.05` and the `E4` `k = 4` `Δ` has a CI lower bound `> 0`.
  - No gating benefit may be claimed. `GATE` results are descriptive, and the rule was formed after seeing earlier results.
  - Frozen `Model A` results are labelled `"older recipe"` wherever shown.

- **Power note:** The paired SD of `FT-2 minus ZS` was about `3.7 pp` in `EXP-019`. With `N = 46`, the minimum detectable effect at 80% power is about `1.5 pp`, so a true effect near `+1 pp` is expected to yield `INCONCLUSIVE`.

---

## 8. Part B: W100 Zero-Shot on SleepAccel (Descriptive)

- Evaluate the 15 `W100` checkpoints on SleepAccel exactly as in the earlier external evaluation (isolated cache under `experiments/EXP-033/sleepaccel_cache/`, no retraining, no threshold tuning, `em_off`; the unsupervised prior calibration is not used).
- Cohorts: `qc_clean` `n = 25` (primary) and all `N = 31` (sensitivity); primary mode single-model mean, secondary mode 15-checkpoint ensemble.
- Paired comparisons of `W100` against frozen `Model A`, `SLAMSS-IFS`, `p1_100k`, and `P190`, with 95% bootstrap CIs. Descriptive only.
- State the power limit: the paired SD observed earlier (about `4 pp`) gives a minimum detectable effect of about `2.3 pp` at `n = 25`.
- **Clinical agreement:** Bland-Altman for TST, SOL, WASO, SE, Deep%, and REM% for `W100`. Clinical margins: `none; descriptive only` (matching `DEC-041`–`DEC-044`).
- Do not describe the outcome as `"parity."`

---

## 9. Reporting Rules

- Report every backbone, depth, `τ`, and arm in this plan, including nulls and negative effects.
- All numbers in the manuscripts are generated from the artifacts by script; extend `scripts/verify_manuscript_numbers.py` to check each reported statistic and each `p`-value label (raw or Holm, one-sided or two-sided).
- If `H-A2` is not **SUPPORTED**, personalization is reported as a negative or inconclusive finding, and the paper's framing follows the recipe and benchmark result.

---

## 10. Outputs

- Per-subject, per-seed, per-arm, per-depth metrics (CSV) under `experiments/EXP-033/`
- Selected configurations per fold and backbone
- State-dict hash log
- Paired statistics tables
- Depth curves
- `GATE` sweep table
- Backbone forest plot
- SleepAccel and Bland-Altman tables for `W100`
- `experiments/EXP-033/EXP033_FINAL_REPORT.md` stating verdicts exactly as defined in Section 7

---

## 11. Compute

Fine-tuning is small-model and fast. Estimated work per backbone:
- Nested inner-validation selection: `5 folds × 5 inner-val subjects × 3 seeds × 12 configs = 900` `FT-2` runs.
- Outer `FT-k` (`k = 1..4`): `46 + 46 + 42 + 33 = 167` subject-depths × `3` seeds = `501` runs.
- Outer `FT-other-2`: `46 subjects × 3 seeds × 5 donor draws = 690` `FT-2` runs.
- Total per backbone: `2,091` fine-tuning runs.

**Wall-clock estimate (from `EXP-019` / `EXP-030` logs):**
- In `EXP-019` (`tsk06_k_curve_20261119_090250.log`), `501` `FT-k` runs completed in `342 s` (`5.7 min`, `~0.68 s/run`) on a single RTX 3060 (`cuda:1`).
- Scaling to `2,091` runs per backbone gives `~24 min` on a single RTX 3060, or **`~12–15 min` per backbone** on dual RTX 3060 (`cuda:0` + `cuda:1`).
- Across all 4 backbones (`W100`, `p1_100k`, `P190`, `full_v5`): **`~50–65 min` total** on dual RTX 3060 (`cuda:0` + `cuda:1`), plus **`~1–2 min`** for `W100` SleepAccel zero-shot evaluation (`em_off`, matching `EXP-030` `p1_100k` SleepAccel eval).

---

## 12. Declared Deviations from the EXP-019 Design

1. **Backbone:** `W100`, with three further backbones (`p1_100k`, `P190`, `full_v5`) reported descriptively.
2. **`FT-other-2`:** Averaged over 5 donor draws (`EXP-019` used one).
3. **Omitted arms:** `PS-2`, `UA-2`, and the KL arm are not repeated.
4. **`GATE`:** Derived from existing `FT-2` and `ZS` predictions, with the `τ` sweep reported and nothing selected.
5. **Page's `L`:** Raw two-sided `p` is primary; one-sided is secondary and labelled.
6. **Safeguards:** Reset guard and unit test added.

---

## 13. Amendments

| Date | Section | Change | Reason |
|---|---|---|---|
