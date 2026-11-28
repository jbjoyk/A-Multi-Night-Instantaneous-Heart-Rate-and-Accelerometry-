# Completed TRIPOD+AI Reporting Checklist

**Manuscript:** *Training-Recipe Sensitivity, Edge Capacity Scaling, and Personalization Boundaries in Consumer-Smartwatch Sleep Staging: A Pre-Registered, Leakage-Free Benchmark*
**Target Journal:** *IEEE Journal of Biomedical and Health Informatics* (*IEEE J-BHI*)
**Reference:** Collins GS et al., *BMJ* 2024;385:e078378 (TRIPOD+AI Statement).

| Item | Section / Topic | Checklist Requirement | Reported Location | Status |
|---|---|---|---|---|
| **1** | **Title** | Identify study as developing/validating a prediction model, target population, and outcome | Title & Abstract (`manuscript_jbhi.tex` L20) | **Complete** |
| **2** | **Abstract** | Structured summary of objectives, data, methods, key results, and limitations | Abstract (Structured Objective/Methods/Results/Conclusion) | **Complete** |
| **3a** | **Background** | Clinical context, existing wearable/EEG staging models (`L-SeqSleepNet`, SLAMSS-IFS), rationale | Sec. I (Introduction) | **Complete** |
| **3b** | **Objectives** | Specify pre-registered hypotheses (`H-ARCH`, `H-SCALE`, `H-A2`, `S-SPEC`, 500-ep audit) | Sec. I & Sec. II-C | **Complete** |
| **4a** | **Data Sources** | Describe datasets (`BIDSleep v1.0.1` doi:`10.13026/rees-1092`; `SleepAccel v1.0.0` doi:`10.13026/hmhs-py35`) | Sec. II-A & Supp. Sec. S2 | **Complete** |
| **4b** | **Study Dates / IRB** | Data collection periods, IRB approvals (UMass Lowell, U. Michigan), ODC-By v1.0 exemptions | Sec. II-A | **Complete** |
| **5a** | **Eligibility & QC** | Inclusion/exclusion criteria (`>= 60%` sync coverage in BIDSleep; `>= 600` epochs & `>= 50%` ACC in SleepAccel) | Sec. II-A & Supp. Sec. S3 | **Complete** |
| **5b** | **Participant Characteristics** | Sample sizes ($N=47$ / 246 nights; $N=46$ adaptation; $n=25$ / $N=31$ SleepAccel), IHR sampling rate (`0.197 Hz`) | Sec. II-A | **Complete** |
| **6** | **Outcome Definition** | 4-class AASM sleep stage consolidation (Wake, Light=N1+N2, Deep=N3, REM) from Dreem 2 EEG & in-lab PSG | Sec. I & Sec. II-A | **Complete** |
| **7** | **Predictors** | Apple Watch HealthKit IHR + 1 Hz 3-axis ACC, circadian-neutralized metadata (`time_feat=0.0`), $L=1200$ context | Sec. II-B | **Complete** |
| **8** | **Sample Size** | Fixed public cohort sizes ($N=47$ internal; $n=25$/$N=31$ external, 80% MDE `2.36 pp` reported) | Sec. II-A & Sec. III-D | **Complete** |
| **9** | **Missing Data** | Masked ignore labels (`-100`), exclusion of `< 60%` coverage nights (`7/253`), `all N=31` sensitivity cohort | Sec. II-A & Supp. Sec. S3 | **Complete** |
| **10a** | **Model Building** | Plain CNN-BiGRU widths (`W050`–`W250`, `p1_100k`, `P190`), `full_v5`, `M1`–`M3`, `SLAMSS-IFS`, Dynamic INT8 | Sec. II-B & Table II/III | **Complete** |
| **10b** | **Training Recipe** | Exact optimizer, batch size, AMP/FP32 loss, `max_epochs=100/500`, `patience=25`, `LogitStitching` | Sec. II-B & Table I | **Complete** |
| **11** | **Validation Scheme** | Subject-disjoint 5-fold × 3-seed CV (`5` inner-val / `9–10` outer-test/fold); strictly zero-shot external PSG | Sec. II-C & Supp. Sec. S2 | **Complete** |
| **12** | **Performance Metrics** | Macro-F1, Accuracy, Bland-Altman summary metrics (TST, SOL, WASO, SE, Deep\%, REM\%), INT8 latency/MMACs | Sec. III-A–D & Tables II–VI | **Complete** |
| **13** | **Statistical Inference** | 10,000-resample participant bootstrap CIs (`seed=42`), Wilcoxon signed-rank, Page's $L$, Holm-Bonferroni | Sec. II-C | **Complete** |
| **14** | **Model Updating** | Pre-registered two-night personalization (`FT-2`), 5-draw donor control (`FT-other-2`), $k=1$--4 depth, `GATE_tau` | Sec. II-C & Sec. III-C | **Complete** |
| **15** | **Open Science / Prereg** | Locked git commits (`4dff5b7`, `945c7c1`, `24ca187`, `c1b7843`, `c7d2d19`) & read-only artifact isolation | Sec. II-C & Supp. Sec. S1/S5 | **Complete** |
| **16** | **Participant Flow** | Exact night/subject exclusions (`253 -> 246`, `47 -> 46 -> 33`, `31 -> 25`) & class distributions | Sec. II-A & Supp. Sec. S3 | **Complete** |
| **17** | **Model Performance** | Full OOF scaling, 500-epoch convergence, architectural ablation, personalization, and SleepAccel tables | Sec. III-A–D & Figs. 1–5 | **Complete** |
| **18** | **Model Calibration / Shift** | Unsupervised EM prior-shift negative check (`em_on`: `63.51% -> 61.30%`) and Bland-Altman bias audit | Sec. III-D & Fig. 5 | **Complete** |
| **19** | **Edge Deployment** | Weights-only INT8 payload, `MMACs`, desktop-CPU latency + `L=1200` activation-buffering caveat | Sec. III-B & Table IV | **Complete** |
| **20** | **Limitations** | Healthy-adult scope, Dreem 2 vs.\ PSG shift, small personalization effect, offline buffering vs.\ 512 KB MCU | Sec. I & Sec. IV | **Complete** |
| **21** | **Interpretation & Claims** | Explicit claim boundaries (no parity/equivalence claim, no clinical-agreement claim, negative ablation) | Sec. I, Sec. III, Sec. IV | **Complete** |
| **22** | **Funding / Ethics / COI** | De-identified public PhysioNet data, anonymous authorship for double-blind review | Title Page footnote | **Complete** |
