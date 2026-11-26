# Internal Tracking ID to Manuscript Protocol Phrase Map (Supplement)

This supplement maps every descriptive protocol phrase used in the public manuscripts (`paper_orchestra_out/manuscript_jbhi.tex` and `research_knowledge/updates/MANUSCRIPT_DRAFT_v2.0.md`) back to its internal experiment (`EXP-*`) and governance decision (`DEC-*`) identifiers in the research repository.

| Manuscript Descriptive Protocol Phrase | Internal Experiment ID(s) | Governance Decision ID(s) | Primary Verification Artifact |
| :--- | :--- | :--- | :--- |
| Legacy 31-subject single training split | `EXP-005` | `DEC-011` | `experiments/EXP-005/` |
| Matched 5-fold cross-validation baseline (`Model A`, $L=1200$) | `EXP-018`, `STAGE2_FINAL_MODEL` | `DEC-018`, `DEC-037` | `experiments/STAGE2_FINAL_MODEL/final_model_oof_subject_metrics.csv` |
| Exploratory multi-night personalization & calibration-depth protocol ($k=1\text{–}4$, `FT-2`, `S1`–`S3`) | `EXP-019`, `EXP-019-K` (`TSK-06`), `EXP-019-S3` (`TSK-07`) | `DEC-028`, `DEC-035`, `DEC-036`, `DEC-039` | `experiments/EXP-019/exp019_k_learning_curve_CLEAN_RECONCILED.csv` |
| Exploratory covariate heterogeneity audit (BH-FDR & Oldham check) | `EXP-019B` | `DEC-040` | `experiments/EXP-019B/exp019b_correlation_summary.csv` |
| Exploratory regularized & stage-coverage-gated adaptation (`GATE_tau`, `FT-2-KL`) | `EXP-020` | `DEC-040` | `experiments/EXP-020/exp020_summary_metrics.json` |
| Protocol-matched 5-fold CV Official `SLAMSS-IFS` server baseline | `EXP-021` (`TSK-08`) | `DEC-038` | `experiments/EXP-021/slamss_ifs_paired_comparison_summary.json` |
| Zero-shot external SleepAccel PSG validation | `EXP-027` | `DEC-041` | `experiments/EXP-027/exp027_summary.json` |
| External SleepAccel PSG Bland–Altman clinical agreement | `EXP-028` | `DEC-041` | `experiments/EXP-028/exp028_sleepaccel_bland_altman_summary.csv` |
| 2-scale multi-resolution FiLM + temporal pooling ablation & `em_on` prior-shift check | `EXP-029` | `DEC-042` | `experiments/EXP-029/full_v5_summary.json` |
| 6-point capacity–edge Pareto grid & deterministic elbow selection (`p1_100k`) | `EXP-030` | `DEC-043` | `experiments/EXP-030/elbow_selection.json` |
| 9-width single-factor sweep (`W050`–`W250`), `H-SCALE`, `H-ARCH`, and `<400` KiB `1-SE` selection (`P190`) | `EXP-031` | `DEC-044` | `experiments/EXP-031/exp031_confirmatory_summary.json` |
| Recipe-consistent `W100` retrained control & interleaved round-robin CPU latency benchmark | `EXP-031` Amendment A1 | Amendment A1 (`16e70ef`) | `experiments/EXP-031/W100_retrained_summary.json`, `experiments/EXP-031/interleaved_cpu_latency.json` |
| Dynamic INT8 edge quantization & latency profiling protocol | `W0-B` | `DEC-039` | `experiments/STAGE2_FINAL_MODEL/int8_quantization_profile.json` |
