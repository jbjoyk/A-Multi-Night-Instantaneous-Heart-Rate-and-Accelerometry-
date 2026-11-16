# Implementation Readiness Test Report (Stage 9)

**Target:** `experiments/EXP-019/audit_heterogeneity_mechanisms.py`  
**Standard:** `implementation-readiness-standard`  
**Reviewers:** `egm-implementation-readiness` (Pragmatic Tech Lead), `egm-codebase-details` (Codebase Technical Quality Auditor), `egm-style` (Technical Copy Editor)  
**Context:** `experiments/EXP-019/HETEROGENEITY_AUDIT_REPORT.md`, `experiments/EXP-019/heterogeneity_predictors_table.csv`

---

## Reviewer Status

| Reviewer | Lens | Status |
|---|---|---|
| `egm-implementation-readiness` | Pragmatic Tech Lead (Actionability, tensor math, leakage safety) | Completed |
| `egm-codebase-details` | Codebase Technical Quality Auditor (Code-doc parity, imports, reproducibility) | Completed |
| `egm-style` | Technical Copy Editor (Docstrings, CLI help, comment hygiene) | Completed |

---

## Executive Invariant & Verification Summary

1. **Zero-Leakage Invariant**: **VERIFIED (PASS)**
   - Pre-adaptation predictors (`P_cal`, `hr_volatility_cal`, `loss_zs_cal`, `param_displacement_norm`, `n_valid_tokens_cal`, `transition_rate_cal`) are derived strictly from calibration nights 2–3 and pre-adaptation base model checkpoints.
   - Retrospective evaluation divergence (`D_JS_eval_DIAGNOSTIC`) is cleanly segregated and quarantined from deployable hypothesis testing.
2. **Mathematical & Statistical Parity**: **VERIFIED (PASS — 100% PARITY)**
   - Every empirical value reported in `HETEROGENEITY_AUDIT_REPORT.md` matches `audit_heterogeneity_mechanisms.py` and `heterogeneity_predictors_table.csv` with zero discrepancy.
   - Monotone step-up Benjamini-Hochberg FDR at $q=0.05$ and Oldham's mean calculation are algebraically correct.
3. **Architectural & Cross-Experiment Reuse**: **VERIFIED (PASS)**
   - Cleanly imports `ConventionalCNNBiGRU` from `exp013_models.py`, `SlidingWindowPredictor` from `exp018_inference.py`, and `adapt_supervised_ft` from `exp019_adaptation_core.py`.
4. **Overall Verdict**: **READY WITH OBSERVATIONS** (Core logic and invariants are sound; requires path anchoring, RNG seeding, and docstring cleanups).

---

## Prioritized Findings (Blockers -> Major -> Nits)

### 1. Non-Deterministic Parameter Displacement Due to Unseeded PyTorch Dropout — Severity: 🟡 Major
* **Reviewer Lens:** `egm-implementation-readiness`
* **Affected Scope:** `audit_heterogeneity_mechanisms.py`, Lines 232–246
* **Discrepancy Description**: `adapt_supervised_ft()` puts the model in training mode (`model.train()`). Because `ConventionalCNNBiGRU` employs `dropout=0.2`, PyTorch's RNG is consumed during the forward pass over the calibration nights. However, `audit_heterogeneity_mechanisms.py` does not call `torch.manual_seed(...)` or `np.random.seed(...)`. Re-running the script produces stochastic fluctuations in `adapted_state` and `param_displacement_norm`, compromising bitwise exact table reproduction.
* **Rationale & Recommendation**: Set deterministic seeds at the entry point of the script and before each subject's adaptation loop:
  ```python
  torch.manual_seed(42)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(42)
  np.random.seed(42)
  ```

### 2. Working-Directory Dependency and Brittle sys.path — Severity: 🟡 Major
* **Reviewer Lens:** `egm-codebase-details`, `egm-implementation-readiness`
* **Affected Scope:** `audit_heterogeneity_mechanisms.py`, Lines 30–35 & Lines 426–433
* **Discrepancy Description**: 
  1. Lines 30–35 insert `repo_root` and `exp018_dir` into `sys.path`, but omit `exp019_dir` (`os.path.dirname(__file__)`). If invoked from outside `experiments/EXP-019/`, importing `exp019_adaptation_core` can fail.
  2. CLI default arguments (`--checkpoint-dir`, `--subject-metrics`, etc.) are defined as relative paths from CWD. If run from `experiments/EXP-019/` instead of repository root, all defaults fail with `FileNotFoundError`.
* **Rationale & Recommendation**: Anchor all default file paths to `repo_root` using `os.path.join(repo_root, ...)` and add `exp019_dir` to `sys.path`.

### 3. Missing Function Docstring and CLI Help Strings — Severity: 🟡 Major
* **Reviewer Lens:** `egm-style`
* **Affected Scope:** `audit_heterogeneity_mechanisms.py`, Lines 80–90 & Lines 423–435
* **Discrepancy Description**: 
  1. The primary entrypoint function `run_heterogeneity_audit` takes 9 arguments but has zero docstring.
  2. All 9 `parser.add_argument(...)` calls lack `help=` strings, producing an undocumented `--help` screen.
  3. `--cache-dir` hardcodes an absolute user path (`/home/rj2826/ai-research/...`), while all other arguments specify workspace-relative paths.
* **Rationale & Recommendation**: Add standard Google-style docstrings and clear CLI `--help` strings; harmonize `--cache-dir` with an environment variable or workspace-relative fallback.

### 4. Dead Code and In-Loop Import in `f1_zs_cal` — Severity: 🟢 Nit
* **Reviewer Lens:** `egm-implementation-readiness`, `egm-codebase-details`, `egm-style`
* **Affected Scope:** `audit_heterogeneity_mechanisms.py`, Lines 223–227
* **Discrepancy Description**: Lines 223–225 compute an unused fallback correlation into `f1_zs_cal` that is immediately overwritten on line 227 by `f1_score`. In addition, `from sklearn.metrics import f1_score` is re-imported 46 times inside the subject loop, and lacks a guard if a night has zero valid epochs.
* **Rationale & Recommendation**: Move `f1_score` to top of module, delete dead lines 223–225, and use a clean defensive ternary: `f1_zs_cal = float(f1_score(cal_y_t[cal_v], cal_y_p[cal_v], average="macro")) if np.any(cal_v) else 0.0`.

### 5. Verbatim Double Curly-Brace Markdown Escape Artifact — Severity: 🟢 Nit
* **Reviewer Lens:** `egm-codebase-details`
* **Affected Scope:** `audit_heterogeneity_mechanisms.py`, Line 408
* **Discrepancy Description**: Line 408 is a plain string (not f-string), so double braces `$D_{{JS}}(P_C \parallel P_{{\text{{pop}}}})$` are emitted verbatim into `HETEROGENEITY_AUDIT_REPORT.md`.
* **Rationale & Recommendation**: Replace with single curly braces: `$D_{JS}(P_C \parallel P_{\text{pop}})$`.

### 6. Population Prior Includes Unusable Nights — Severity: 🟢 Nit
* **Reviewer Lens:** `egm-implementation-readiness`, `egm-codebase-details`
* **Affected Scope:** `audit_heterogeneity_mechanisms.py`, Lines 103–106
* **Discrepancy Description**: `df_inv["wake_frac"].mean()` averages across all records in `night_inventory.csv` without filtering `df_inv["usable"] == True`. Unusable nights have 0.0 values.
* **Rationale & Recommendation**: Filter `df_inv[df_inv["usable"] == True]` before computing demographic priors.
