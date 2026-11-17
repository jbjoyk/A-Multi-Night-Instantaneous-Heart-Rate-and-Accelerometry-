# Research Landscape Map: Multi-Night Wearable Sleep Staging

**Last Updated:** 2026-11-17 (KB v2 Errata Applied)  
**Maturity Legend:** `Mature` (established baselines, multiple benchmarks) | `Active` (rapidly evolving, unresolved trade-offs) | `Underexplored` (sparse rigorous evidence) | `Saturated` (diminishing returns)

```
Multi-Night Wearable Sleep Staging
├── Subject-independent staging      [Mature / Saturated on single-split]
├── Subject-adaptive staging         [Active — EEG precedent + wearable PPG+Accel (van der Aar et al. 2025); IHR+Accel safety open]
├── Multi-night learning             [Active — longitudinal home PPG+EEG (van der Aar et al. 2025, n=59, <=7 nights) & BIDSleep (n=47, 3-7 nights)]
├── Personalization                  [Active — supervised FT works on avg, fails on outliers]
├── Transfer learning                [Mature in EEG, Active in wearable PPG/IHR]
├── Domain adaptation                [Active — UDA outcomes mixed; implementation/regime-dependent]
├── Test-time adaptation             [Active — brittle to label shift & batch dependency]
├── Continual learning               [Underexplored in wearable sleep staging]
├── Distribution shift               [Active — prior/label shift & inter-subject shift]
├── Calibration                      [Underexplored — representativeness vs quantity trade-off]
├── Uncertainty                      [Underexplored in wearable sleep adaptation]
├── Selective prediction / abstention [Underexplored in physiological adaptation]
├── Adaptation failure detection     [Underexplored — emerging in TTA risk monitoring]
├── Adaptive gating                  [Underexplored — high-potential safe adaptation]
├── External validation              [Active — cross-dataset drops are common]
└── Clinical validation              [Active — Bland-Altman & summary metrics required]
```

| Sub-Area | Maturity | Strongest Seed Papers | Common Methods & Datasets | Weaknesses / Contradictions / Unresolved Questions | Relevance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Subject-independent staging** | Mature | [Song et al. (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12931632/), [Walch et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6930135/), [Sridhar et al. (2020)](https://pmc.ncbi.nlm.nih.gov/articles/PMC7441407/) | CNN-BiGRU/LSTM, U-Sleep, SleepPPG-Net; BIDSleep, MESA, SleepAccel | Single-split vs strict OOF CV gap (~3–4 pp); wake specificity remains ~60% on Apple Watch; N1/light overlap. | HIGH |
| **Subject-adaptive & Personalized staging** | Active | [van der Aar et al. (2025)](https://pubmed.ncbi.nlm.nih.gov/40882681/), [Phan et al. (2020)](https://lirias.kuleuven.be/retrieve/616969), [Lorenzen et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/38833404/), [Banluesombatkul et al. (2020)](https://arxiv.org/abs/2004.04157) | Partial retraining / Supervised FT + KL-reg, MAML; Wrist PPG+Accel (n=59), Sleep-EDF, wearable EEG | **Contradiction / Nuance:** [van der Aar et al. (2025)](https://pubmed.ncbi.nlm.nih.gov/40882681/) showed wearable PPG+Accel personalization improved 82.5% of subjects (significant at >=3 nights); EXP-019 on 2-night wearable IHR+Accel found +1.01 pp (below +1.50 pp) due to outlier harm (17.5% non-responders in P106; 12/46 non-responders in EXP-019). | HIGH |
| **Multi-night learning** | Active | [van der Aar et al. (2025)](https://pubmed.ncbi.nlm.nih.gov/40882681/), [Song et al. (2026)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12931632/), [Davidson et al. (2026)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2026.1693860/full) | Wrist PPG+Accel + wearable EEG (<=7 nights, n=59), BIDSleep (3–7 nights/subj), longitudinal home ECG | Multi-night labeled wearable cohorts exist ([van der Aar et al. 2025](https://pubmed.ncbi.nlm.nih.gov/40882681/); BIDSleep), though most public benchmarks (MESA, SleepAccel) still have only 1 overlapping PSG night. | HIGH |
| **Unsupervised Domain Adaptation & TTA** | Active (Unresolved) | [Lorenzen et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/38833404/), [Heremans et al. (2022)](https://pubmed.ncbi.nlm.nih.gov/35877770/), [Zhao et al. (2023)](https://arxiv.org/html/2306.03536) | Adversarial DA, Statistics Alignment, TENT, SHOT, InfoMax | **Mixed / Unresolved:** [Lorenzen et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/38833404/) observed no UDA benefit in wearable EEG, whereas [Heremans et al. (2022)](https://pubmed.ncbi.nlm.nih.gov/35877770/) (`P054`) reported ~1–2% unsupervised adversarial gains; EXP-019 `UA-2` (+0.00 pp exact tie) requires parameter-update verification (possible no-op). | HIGH |
| **Distribution Shift & Negative Transfer** | Active | [Zhao et al. (2023)](https://arxiv.org/html/2306.03536), [Zhao et al. (2025)](https://arxiv.org/html/2501.03764v1), [Li et al. (2026)](https://arxiv.org/html/2604.27033v1) | EMD domain alignment, Multi-source DA, Dirichlet label shift | Single-source pooling causes negative transfer; calibration-night label shift is a candidate mechanism for classifier-head collapse. | HIGH |
| **Safe Adaptation, Gating & Risk Monitoring** | Underexplored | [Schirmer et al. (2025)](https://proceedings.neurips.cc/paper_files/paper/2025/file/746960ad49ddb47248970a0e1404230c-Paper-Conference.pdf) | Time-uniform confidence sequences, selective prediction | Unexplored for wearable IHR sleep staging; pre-adaptation gating using calibration-only signals has no established benchmark. | HIGH |
| **External & Clinical Validation** | Active | [Walch et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6930135/), [Kotzen et al. (2024)](https://arxiv.org/html/2404.06869v1), [Ostadsharif Memar et al. (2026)](https://www.preprints.org/manuscript/202607.1006) | Apple Watch -> MESA, Bland-Altman (TST, SOL, WASO, SE) | Zero-shot transfer across devices/demographics remains fragile; Bland-Altman heteroscedasticity persists. | HIGH |
