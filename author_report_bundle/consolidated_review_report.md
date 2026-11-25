# Consolidated EGM Review Report — `author_report_bundle/TECHNICAL_EXPERIMENTAL_REPORT.md`

**Review Mode:** Uninterrupted Consolidated Review (`egm-zoom-out`, `egm-comprehension`, `egm-design`, `egm-structural`, `egm-codebase-details`)
**Target:** `author_report_bundle/TECHNICAL_EXPERIMENTAL_REPORT.md`, `author_report_bundle/tables/*.csv`, `fig1`–`fig4`
**GoldfishContext:** 15 specified ground-truth code, CSV, JSON, and PNG files (excluding full `DECISIONS.md` and `EXPERIMENTS.md`).

---

## 1. Verification of the 4 Requested Fixes (Pre-Remediation)

| Fix | Scope | Reviewer Verdict |
|---|---|---|
| 1 | `fig4_latency_memory_pareto` Panel A sorted x-axis (`W050` → `W250`) | **PASS** (`egm-design`, `egm-codebase-details`) |
| 2 | `fig1` & `fig1a/d/e` BiGRU per-direction `hidden_size` (`48/38/50` → `96/76/100`) & Classifier Head `Linear` dims | **PASS** (`egm-design`, `egm-codebase-details`) |
| 3 | Table 2 `full_v5` `hr_encoder` / `acc_encoder` shapes (`(1, 1200, 30)` → `(1200, 64, 7)`) | **PASS** (`egm-design`, `egm-codebase-details`) |
| 4 | Table 6 `P190` / `p1_100k` Bland–Altman rows & `309.54` vs `309.91 KiB` note | **PASS** (`egm-design`, `egm-codebase-details`) |

---

## 2. Consolidated Findings & Remediation Traceability Matrix

All Blockers and Majors identified across the 5 Goldfish reviewers were remediated in `scripts/build_report_architectures.py` and `scripts/build_report_tables_and_md.py` and re-verified.

| ID | Severity | Source Reviewer(s) | Finding Summary | Resolution Status |
|---|---|---|---|---|
| **F-01** | Blocker | `egm-comprehension`, `egm-structural` | Unescaped `\|` pipe in Table 2 GRU `out_shape` split 5 rows into 10 columns | **FIXED** (`; ` separator in `shape_str()`; all tables verified equal column count) |
| **F-02** | Blocker | `egm-structural` | Unescaped spaces in 17 `file://` links | **FIXED** (`%20` URL encoding via `file_url()`; 0 unescaped spaces) |
| **F-03** | Blocker / Major | `egm-zoom-out`, `egm-design` | Table 8 vs `exp031_cpu_profile.json` CPU latency session difference (`Model A` `69.39/47.73` vs `77.24/56.58 ms`; `P190` `73.30/53.07` vs `71.46/54.56 ms`) & `P190` serialized `402.91 KiB` | **FIXED** (Reconciliation Notes 1–3 added in Section 4) |
| **F-04** | Major | `egm-structural`, `egm-codebase-details` | Table 4 `p < 5e-5` collapsed to `0.0` under 4-decimal rounding | **FIXED** (Scientific notation `:.2e` for `p < 1e-3`) |
| **F-05** | Major | `egm-comprehension`, `egm-structural`, `egm-zoom-out` | Table 7 16-digit proportions vs `pp` units, `nan` strings, missing `Table 7b` heading | **FIXED** (`Mean_ZS_Clean`/`Mean_FT` formatted as `%`, `—` for `NaN`, added `### Table 7b:`) |
| **F-06** | Major | `egm-zoom-out`, `egm-design`, `egm-comprehension` | Legacy `31`-subject split (`137,801` epochs) vs 5-fold CV `N=47` (`203,338` epochs), non-uniform `P190`/`B`, undefined acronyms, and `"statistical parity"` without TOST | **FIXED** (Clarified legacy `EXP-005` split vs `N=47` CV, added Abbreviations & Conventions block, framed `p1_100k` vs `SLAMSS-IFS` without TOST claim) |

---

## 3. Final Post-Remediation Status

- **Stage 9 Numerical & Codebase Audit (`egm-codebase-details`):** **PASS (0 Blockers, 0 Majors)**.
- **Post-Remediation Markdown & Link Verification:** **PASS (0 Blockers, 0 Majors)** — all Markdown table row column counts match headers, all `file://` URLs are `%20`-encoded, and zero raw `nan` or `p = 0.0` strings remain.
