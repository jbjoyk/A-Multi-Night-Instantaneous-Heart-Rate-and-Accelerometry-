# Literature Search Log

| Date | Search Batch | Query String / Source | Engine | Purpose | Screened / Included |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 2026-11-17 | Phase1-Internal | `Sleep_research` NotebookLM (`notebooks/b1fb0d70-179b-4147-847d-b4585edaacd9`) | NotebookLM | Inspect existing project artifacts, Song et al. 2026, EXP-019, heterogeneity audit | 9 sources / 1 primary (`P001`) |
| 2026-11-17 | Phase1-Seed | 11 targeted seed queries (wearable staging, personalization, TTA pitfalls, negative transfer, leakage, datasets) | Google Search | Balanced 12-paper seed set (`P001`–`P012`) | ~130 hits / 11 new (`P002`–`P012`) |
| 2026-11-17 | Phase2-Expand | 4 targeted queries (selective prediction/gating, continual learning, Bland-Altman clinical validation, MetaSleepLearner) | Google Search | Phase 2 depth (`P013`–`P016`) + `P003` full-text browse | ~45 hits / 4 new (`P013`–`P016`) |
| 2026-11-17 | Phase3-Scale-1 | Wearable PPG, IHR, HRV, accel, foundation models, smartwatch/ring validation (4 queries) | Google Search | Expand wearable & benchmark coverage | ~48 hits / 34 new (`P017`–`P052` subset) |
| 2026-11-17 | Phase3-Scale-2 | Personalized, multi-night, few-shot, & domain-adaptive sleep staging (4 queries) | Google Search | Expand personalization & DA/DG sleep literature | ~48 hits / 19 new (`P053`–`P071`) |
| 2026-11-17 | Phase3-Scale-3 | TTA collapse, label shift, time-series DA, negative transfer, continual learning (4 queries) | Google Search | Expand failure modes & safe adaptation | ~48 hits / 18 new (`P072`–`P089`) |
| 2026-11-17 | Phase3-Scale-4 | Selective prediction, abstention, uncertainty, calibration shift, gating (4 queries) | Google Search | Expand selective gating & uncertainty | ~48 hits / 11 new (`P090`–`P100`) |
| 2026-11-17 | Phase3-Scale-5 | External validation, Bland-Altman, data leakage, reproducibility, classic anchors (12 queries) | Google Search | Expand leakage/clinical validation & classic anchors | ~140 hits / 7 new (`P101`–`P105`, `P035`–`P038`, `P023`–`P024`) |
| 2026-11-17 | Errata-v2-ForwardCit | Forward-citation sweeps (`"Cited by"` on `P001`, `P006`, `P056`, `P106`) + sleep-gating/abstention search (`"sleep staging" "selective prediction" OR "abstention" OR "gating"`) + metadata/author resolution across 49 records | Google Search + PubMed/PMC/arXiv/MDPI/Frontiers/Preprints browse | Apply KB v2 Errata (`P106` ingestion, `P006`/`P007`/`P101`/`P102` fixes, author resolution, sleep-gating precedent check) | 1 new (`P106`); 48 metadata/author updates |
