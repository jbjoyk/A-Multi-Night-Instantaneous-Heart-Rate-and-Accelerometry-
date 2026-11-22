# Stage D Post-Execution EGM Drift Report

* **Date:** 2026-11-23
* **Locked Specs Audited:** `docs/designs/exp019b_exp020_closure_spec.md`, `docs/designs/exp027_exp028_sleepaccel_external_validation.md`
* **Executed Targets:** `experiments/EXP-019B/`, `experiments/EXP-020/`, `experiments/EXP-027/`, `experiments/EXP-028/`
* **Final Verdict:** **`CLEARED FOR DEC-040/041 & MANUSCRIPT v0.2`** (0 `BLOCKING`, 0 `MAJOR`, 3 `NIT` items)

---

## Post-Execution Drift Summary Table

| Axis | Target | Check | Drift | Observed Result |
|---|---|---|---|---|
| **1. Artifact & Schema** | `EXP-019B` | `exp019b_subject_features.csv`, `exp019b_correlation_summary.csv`, `exp019b_diagnostics.json`, `EXP019B_CLOSURE_REPORT.md` | `NONE` | `(47, 14)` features (0 NaNs on `all_*`); `(24, 12)` corr rows (`12` primary + `6` `N=46` all-nights + `6` `N=46` cal-only); `N=34` vs `N=12` subgroup Mann-Whitney in `exp019b_diagnostics.json`. |
| **1. Artifact & Schema** | `EXP-020` | `exp020_subject_metrics.csv`, `exp020_inner_val_selection.json`, `exp020_summary_metrics.json`, `EXP020_CLOSURE_REPORT.md` | `NONE` | `(46, 16)` subject rows (`N=46` $\times$ 3 seeds, 0 NaNs); all 5 folds in `exp020_inner_val_selection.json`; `Bidslab34` audit in JSON. |
| **1. Artifact & Schema** | `EXP-027` & `EXP-028` | `exp027_summary.json`, `exp027_per_subject_metrics.csv`, `sleepaccel_qc_summary.csv`, `preflight_distribution_gate.json`, `exp028_sleepaccel_raw_paired_tuples.csv`, `exp028_sleepaccel_bland_altman_summary.csv` | `NIT` | `EXP-027`: `(124, 12)` rows (`qc_clean=25`, `all=31`, 0 NaNs). `EXP-028`: `744` raw tuples (`31*2*6*2`, 0 NaNs), `(48, 30)` BA rows. Benign path nit: `run_exp028_sleepaccel_bland_altman.py` resides in `experiments/EXP-027/` (per Phase 4A prompt) and writes to `experiments/EXP-028/`. |
| **2. Runtime & Stats** | `EXP-019B` | BH-FDR NaNs & `N=46` vs `N=47` | `NONE` | `0` NaNs across all `primary` BH-FDR columns; exact `N=46` (Family 1) and `N=47` (Family 2). |
| **2. Runtime & Stats** | `EXP-020` | `EXP-019` parity, `Bidslab34`, abstain counts | `NIT` | `ZS` max diff vs `EXP-019` = `4.65e-05`. `FT-2` max subject diff = `0.0146` (mean `0.6560` vs `0.6559`) due to deterministic per-call seeding (`exp019b_exp020_closure_spec.md:L117`). Abstentions: `2/46` (Fold 1: `1`, Fold 2: `1` [`Bidslab34`]). `Bidslab34` (Fold 2, `min_stage_frac_cal=0.0068`): `GATE_tau` abstained to `ZS=0.6006` (cutting cohort min delta to `-6.40 pp`), while Arm c fell back to `FT-2-KL` (`0.4492`, `-15.14 pp`) via `FOLD_2_COMBINED_SKIPPED_BY_INNER_VAL_GATE`. |
| **2. Runtime & Stats** | `EXP-027` & `EXP-028` | `evaluator.py:129` `RuntimeWarning` & Shapiro-Wilk fallback | `NIT` | `96` `RuntimeWarning`s in `logs/exp027.log` traced to **3 zero-support PSG subjects** (`1455390` [`qc_pass=True`, Deep=`0`], `5132496` [`qc_pass=False`, REM=`0`], `7749105` [`qc_pass=False`, REM=`0`]; $3 \times 2 \times 16 = 96$). `np.nan_to_num(..., 0.0)` and `zero_division=0` prevented any NaNs (`0` NaNs in `exp027_per_subject_metrics.csv`). `EXP-028` flagged `10/48` non-normal BA rows (`nonparam_nans=0`) with valid 2.5th–97.5th percentile LoA fallback. |
| **3. Governance & Git** | Repo | Locked artifacts & `.gitignore` (`DEC-002`) | `NONE` | `git diff --name-only 945c7c1..HEAD` confirms `0` edits to `EXP-018`, `EXP-019/`, or `EXP-021/`. `data/external/` and `experiments/EXP-027/cache/` remain gitignored. |
| **4. Claim Boundaries** | `DEC-040/041` & `v0.2` | Mandatory wording guardrails | `NONE` | See Section 4 below; all guardrails enforced in `DEC-040` and `DEC-041`. |

---

## Section 4: Mandatory Claim-Boundary Guardrails (`DEC-040`, `DEC-041`, `MANUSCRIPT_DRAFT_v0.2.md`)

1. **`EXP-019B` & `EXP-020` (`DEC-040`):** Must be framed strictly as **design-informed exploratory closures** on BIDSleep (`N=46`/`N=47`) following prior inspection of `EXP-019` and `EXP-021` results. Confirmatory verdict labels (`CONFIRMED`, `SUPPORTED`, `VALIDATED`) are strictly prohibited.
2. **`EXP-027` (`DEC-041`):** Non-significant paired differences vs. Official SLAMSS-IFS (`prob_ensemble` $\bar{\Delta} = +0.15\text{ pp}, p = 0.9789$; `mean_single_model` $\bar{\Delta} = -0.57\text{ pp}, p = 0.5782$) must be reported as **no statistically significant difference / empirical parity** under paired testing—do **not** claim formal TOST equivalence without an explicit pre-registered equivalence margin.
3. **SleepAccel Denominators (`EXP-027` & `EXP-028`):** Report both `qc_clean` (`n_obs=25`, primary) and `all_n31_sensitivity` (`n_obs=31`) transparently, noting the 6 exclusions (`5` short PSG `< 600` epochs, `1` Apple Watch ACC dropout `7749105`).
