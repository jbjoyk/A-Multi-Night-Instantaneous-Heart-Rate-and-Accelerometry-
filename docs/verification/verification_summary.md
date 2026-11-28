# Manuscript Verification Gate Summary (Checks A–J)

- **Primary TeX target:** `paper_orchestra_out/manuscript_jbhi.tex`
- **Primary Markdown target:** `research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md`
- **Technical Report target:** `author_report_bundle/TECHNICAL_EXPERIMENTAL_REPORT.md`
- **Total checks executed:** `83` (`PASS`: `83`, `FAIL`: `0`, `EXPECTED_FAIL`: `0`, `WARN`: `0`)

## Promoted Hard-Fail Gates (`EF-01`–`EF-10`)

| Gate ID | Check | Status | Target | Observed Finding |
|---|---|---|---|---|
| `EF-01` | `J` | **`PASS`** | `All manuscripts` | Found in 0 document(s): [] () |
| `EF-02` | `J` | **`PASS`** | `All manuscripts` | 0 issue(s) () |
| `EF-03` | `C` | **`PASS`** | `All manuscripts` | 0 suspicious pair(s), 0 stale mix(es) () |
| `EF-04` | `F` | **`PASS`** | `All manuscripts` | 0 unlabeled, 0 mislabeled () |
| `EF-05` | `H` | **`PASS`** | `experiments/EXP-031/interleaved_cpu_latency.json & paper_orchestra_out/manuscript_jbhi.tex` | 0 CV violation(s), 0 4T-FP32 issue(s), 0 table issue(s) () |
| `EF-06` | `J` | **`PASS`** | `All manuscripts` | 0 hit group(s) () |
| `EF-07` | `J` | **`PASS`** | `paper_orchestra_out/manuscript_jbhi.tex, research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md` | 0 document(s) with internal IDs () |
| `EF-08` | `G` | **`PASS`** | `experiments/EXP-033/reconciled_references.json` | 0 unjustified collisions, 2 verified justified (ModelA prob_ensemble (62.94%, delta=0.001890 pp); p1_100k prob_ensemble (62.51%, delta=0.004784 pp)) |
| `EF-09` | `E` | **`PASS`** | `experiments/EXP-019/exp019_stage_c_results.json` | Calc Delta=+1.0054 pp, 1s p=0.00131, 2s p=0.00261, EXP-033 summary ok=True (match_delta=True, match_p=True, exp033_legacy_ok=True) |
| `EF-10` | `F` | **`PASS`** | `All manuscripts` | 0 mislabeled reported p-values () |
| `EF-11` | `J` | **`PASS`** | `docs/verification/claim_registry.yaml` | Verified (Phan 2023 (~6.3e5 params, IEEE J-BHI) & Heremans 2022 (J. Neural Eng.) + quarantine status verified) |

## Results by Check Group (A–J)

| Group | PASS | FAIL | EXPECTED_FAIL | WARN |
|---|---:|---:|---:|---:|
| `A` | 18 | 0 | 0 | 0 |
| `B` | 4 | 0 | 0 | 0 |
| `C` | 4 | 0 | 0 | 0 |
| `D` | 22 | 0 | 0 | 0 |
| `E` | 9 | 0 | 0 | 0 |
| `F` | 5 | 0 | 0 | 0 |
| `G` | 7 | 0 | 0 | 0 |
| `H` | 7 | 0 | 0 | 0 |
| `I` | 2 | 0 | 0 | 0 |
| `J` | 5 | 0 | 0 | 0 |

## Flagged Items Requiring Remediation (`FAIL`, `EXPECTED_FAIL`, `WARN`)

None — all checks passed.

## Canonical Ground-Truth Summary (Recomputed Directly from Raw Artifacts)

### 1. Parameters & Compression Ratios (Check A)
| Model | Exact Params (`sum(p.numel())`) | Compression vs SLAMSS-IFS |
|---|---:|---:|
| `p1_100k` | 98,940 | 55.24× |
| `Model A` | 163,940 | 33.34× |
| `W100` | 163,940 | 33.34× |
| `full_v5` | 190,788 | 28.65× |
| `P190` | 190,940 | 28.63× |
| `Official SLAMSS-IFS` | 5,465,768 | 1.00× |

### 2. Participant-Mean Macro-F1 (`ddof=1`, `N=47`, Check D)
| Model | Mean Macro-F1 (%) | SD (`ddof=1`, pp) |
|---|---:|---:|
| `p1_100k` | 66.67% | 7.57 |
| `Model_A` | 64.79% | 7.45 |
| `W100` | 66.68% | 7.11 |
| `full_v5` | 66.74% | 6.81 |
| `P190` | 66.60% | 6.85 |
| `SLAMSS_IFS` | 67.37% | 6.73 |

### 3. Paired Statistics (`seed=42`, 10,000 Bootstrap Resamples, Check E)
| Comparison | $\Delta$ (pp) | 95% Bootstrap CI (pp) | Wilcoxon $p_{\text{raw}}$ | Cohen's $d_z$ | $p_{\text{Holm}}$ |
|---|---:|---|---:|---:|---:|
| `W100_minus_Model_A` | +1.89 | [+1.23, +2.56] | 2.121e-06 | +0.811 | — |
| `W100_minus_P190` | +0.08 | [-0.29, +0.46] | 0.6634 | +0.061 | 1.0000 |
| `W100_minus_p1_100k` | +0.01 | [-0.45, +0.45] | 0.8709 | +0.003 | 1.0000 |
| `W100_minus_Official_SLAMSS_IFS` | -0.69 | [-1.44, +0.09] | 0.04777 | -0.255 | 0.1433 |
| `full_v5_minus_P190` | +0.14 | [-0.34, +0.61] | 0.4037 | +0.086 | — |

### 4. Raw Label Class Mixes (Check C)
- **BIDSleep masked usable (`N=47`, `246` nights, `203,338` epochs):** Wake `10.83%`, Light `44.20%`, Deep `18.91%`, REM `26.06%`
- **BIDSleep raw unmasked `labels.mat` pooled (`209,482` epochs):** Wake `11.06%`, Light `44.60%`, Deep `18.54%`, REM `25.79%`
- **BIDSleep raw unmasked night-averaged:** Wake `11.11%`, Light `44.35%`, Deep `18.70%`, REM `25.83%`
- **BIDSleep raw unmasked participant-averaged:** Wake `10.96%`, Light `44.39%`, Deep `18.63%`, REM `26.02%`
- **SleepAccel `qc_clean` (`n=25`, `22,633` masked epochs):** Wake `8.24%`, Light `57.74%`, Deep `11.08%`, REM `22.94%`
