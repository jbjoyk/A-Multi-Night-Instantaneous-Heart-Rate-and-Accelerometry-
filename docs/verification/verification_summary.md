# Manuscript Verification Gate Summary (Checks A–J)

- **Primary TeX target:** `paper_orchestra_out/manuscript_jbhi.tex`
- **Primary Markdown target:** `research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md`
- **Technical Report target:** `author_report_bundle/TECHNICAL_EXPERIMENTAL_REPORT.md`
- **Total checks executed:** `79` (`PASS`: `71`, `FAIL`: `0`, `EXPECTED_FAIL`: `7`, `WARN`: `1`)

## Seeded Known Open Issues (7 `EXPECTED_FAIL` Test Cases)

| Case ID | Check | Status | Target | Observed Finding |
|---|---|---|---|---|
| `EF-01` | `J` | **`EXPECTED_FAIL`** | `paper_orchestra_out/manuscript_jbhi.tex, research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md` | Found in 2 document(s): ['paper_orchestra_out/manuscript_jbhi.tex', 'research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md'] () |
| `EF-02` | `J` | **`EXPECTED_FAIL`** | `paper_orchestra_out/manuscript_jbhi.tex, research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md, author_report_bundle/TECHNICAL_EXPERIMENTAL_REPORT.md` | Stale v1.0.0 found in 3 document(s): ['paper_orchestra_out/manuscript_jbhi.tex', 'research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md', 'author_report_bundle/TECHNICAL_EXPERIMENTAL_REPORT.md'] () |
| `EF-03` | `C` | **`EXPECTED_FAIL`** | `All manuscripts` | 1 suspicious pair(s) found (research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md: (10.8, 44.2, 18.9, 26.1) vs (12.8, 44.2, 18.9, 24.1) (+/-2.0%)) |
| `EF-04` | `F` | **`EXPECTED_FAIL`** | `All manuscripts` | 3 unlabeled occurrence(s) (paper_orchestra_out/manuscript_jbhi.tex: Page's $L$ $p=0.01187; paper_orchestra_out/manuscript_jbhi.tex: Page's $L$ $p=0.01187; research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md: Page's $L$ $p=0.01187) |
| `EF-05` | `H` | **`EXPECTED_FAIL`** | `experiments/EXP-031/interleaved_cpu_latency.json & paper_orchestra_out/manuscript_jbhi.tex` | 4 CV > 5% violation(s), 2 table mixing issue(s) (threads_4/p1_100k/fp32 CV=8.82%; threads_4/W100/fp32 CV=7.28%; threads_4/full_v5/fp32 CV=14.60%; threads_4/P190/fp32 CV=5.96%; SLAMSS-IFS FP32 20.85 MiB placed in INT8 (KiB) column; tab:scaling uses non-interleaved W0-B 4T ms (46.35) while prose uses interleaved (41.02)) |
| `EF-06` | `J` | **`EXPECTED_FAIL`** | `All manuscripts` | 8 hit group(s) (paper_orchestra_out/manuscript_jbhi.tex:UMass_Amherst(1); paper_orchestra_out/manuscript_jbhi.tex:parity_without_CI; research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md:proves(1); research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md:UMass_Amherst(1); research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md:four_canonical(1); research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md:k_ge_3_night_index(2); author_report_bundle/TECHNICAL_EXPERIMENTAL_REPORT.md:Ultradian(10); author_report_bundle/TECHNICAL_EXPERIMENTAL_REPORT.md:parity_without_CI) |
| `EF-07` | `J` | **`EXPECTED_FAIL`** | `paper_orchestra_out/manuscript_jbhi.tex, research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md` | 2 document(s) with internal IDs (paper_orchestra_out/manuscript_jbhi.tex: 8 unique IDs (DEC-043, DEC-044, EXP-005, EXP-019B, EXP-020, EXP-029); research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md: 16 unique IDs (DEC-018, DEC-028, DEC-041, DEC-043, DEC-044, EXP-005)) |

## Results by Check Group (A–J)

| Group | PASS | FAIL | EXPECTED_FAIL | WARN |
|---|---:|---:|---:|---:|
| `A` | 18 | 0 | 0 | 0 |
| `B` | 4 | 0 | 0 | 0 |
| `C` | 3 | 0 | 1 | 0 |
| `D` | 22 | 0 | 0 | 0 |
| `E` | 8 | 0 | 0 | 0 |
| `F` | 3 | 0 | 1 | 0 |
| `G` | 5 | 0 | 0 | 1 |
| `H` | 6 | 0 | 1 | 0 |
| `I` | 2 | 0 | 0 | 0 |
| `J` | 0 | 0 | 4 | 0 |

## Flagged Items Requiring Remediation (`FAIL`, `EXPECTED_FAIL`, `WARN`)

| Check ID | Group | Status | Target | Expected | Observed | Detail |
|---|---|---|---|---|---|---|
| `C_SUSPICIOUS_MIX_PAIR_EF03` | `C` | **`EXPECTED_FAIL`** | `All manuscripts` | 0 suspicious pairs | 1 suspicious pair(s) found | research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md: (10.8, 44.2, 18.9, 26.1) vs (12.8, 44.2, 18.9, 24.1) (+/-2.0%) |
| `F_PAGES_L_EXPLICIT_LABEL_EF04` | `F` | **`EXPECTED_FAIL`** | `All manuscripts` | 0 unlabeled Page's L p-values | 3 unlabeled occurrence(s) | paper_orchestra_out/manuscript_jbhi.tex: Page's $L$ $p=0.01187; paper_orchestra_out/manuscript_jbhi.tex: Page's $L$ $p=0.01187; research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md: Page's $L$ $p=0.01187 |
| `G_IDENTICAL_P_4DEC` | `G` | **`WARN`** | `SleepAccel tables & exp027_summary.json` | 0 suspicious duplicate 4-decimal p-values | 1 collision(s) | exp027_summary/qc_clean/mean_single_model: p=0.1336 shared by ['accuracy', 'wake_f1'] |
| `H_LATENCY_CV_AND_TABLE_EF05` | `H` | **`EXPECTED_FAIL`** | `experiments/EXP-031/interleaved_cpu_latency.json & paper_orchestra_out/manuscript_jbhi.tex` | All CV <= 5.0% and no FP32/INT8 or W0-B/interleaved mixing | 4 CV > 5% violation(s), 2 table mixing issue(s) | threads_4/p1_100k/fp32 CV=8.82%; threads_4/W100/fp32 CV=7.28%; threads_4/full_v5/fp32 CV=14.60%; threads_4/P190/fp32 CV=5.96%; SLAMSS-IFS FP32 20.85 MiB placed in INT8 (KiB) column; tab:scaling uses non-interleaved W0-B 4T ms (46.35) while prose uses interleaved (41.02) |
| `J_HARDWARE_AW_SERIES6_EF01` | `J` | **`EXPECTED_FAIL`** | `paper_orchestra_out/manuscript_jbhi.tex, research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md` | Absent unless explicitly in README.md (in_readme=False) | Found in 2 document(s): ['paper_orchestra_out/manuscript_jbhi.tex', 'research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md'] |  |
| `J_SLEEPACCEL_VERSION_EF02` | `J` | **`EXPECTED_FAIL`** | `paper_orchestra_out/manuscript_jbhi.tex, research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md, author_report_bundle/TECHNICAL_EXPERIMENTAL_REPORT.md` | SleepAccel v1.0.1 (0 stale v1.0.0 strings) | Stale v1.0.0 found in 3 document(s): ['paper_orchestra_out/manuscript_jbhi.tex', 'research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md', 'author_report_bundle/TECHNICAL_EXPERIMENTAL_REPORT.md'] |  |
| `J_BANNED_PHRASES_EF06` | `J` | **`EXPECTED_FAIL`** | `All manuscripts` | 0 banned phrase hits | 8 hit group(s) | paper_orchestra_out/manuscript_jbhi.tex:UMass_Amherst(1); paper_orchestra_out/manuscript_jbhi.tex:parity_without_CI; research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md:proves(1); research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md:UMass_Amherst(1); research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md:four_canonical(1); research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md:k_ge_3_night_index(2); author_report_bundle/TECHNICAL_EXPERIMENTAL_REPORT.md:Ultradian(10); author_report_bundle/TECHNICAL_EXPERIMENTAL_REPORT.md:parity_without_CI |
| `J_INTERNAL_IDS_PROSE_EF07` | `J` | **`EXPECTED_FAIL`** | `paper_orchestra_out/manuscript_jbhi.tex, research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md` | 0 internal IDs in manuscript prose | 2 document(s) with internal IDs | paper_orchestra_out/manuscript_jbhi.tex: 8 unique IDs (DEC-043, DEC-044, EXP-005, EXP-019B, EXP-020, EXP-029); research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md: 16 unique IDs (DEC-018, DEC-028, DEC-041, DEC-043, DEC-044, EXP-005) |

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
