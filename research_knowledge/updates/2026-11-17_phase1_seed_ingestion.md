# Update Log: 2026-11-17 (Phase 1 — Architecture & Seed Ingestion)

- **Date:** 2026-11-17
- **Search Scope:** Balanced Phase 1 seed set across wearable IHR+Accel sleep staging, EEG/wearable personalization, TTA pitfalls/label shift, negative transfer, risk monitoring, and leakage-free evaluation protocols.
- **New Papers Ingested:** 12 (`P001`–`P012` in `evidence_matrix.csv`, with full 32-field records in `papers/`).
- **Changed Conclusions:** None (`LOCKED` research direction and EXP-019 verdicts remain immutable).
- **Newly Discovered Contradictions & Negative Results:**
  1. [Lorenzen et al. (2023)](https://www.techrxiv.org/doi/10.36227/techrxiv.24500104) found unsupervised Adversarial Domain Adaptation and statistics alignment yielded **no beneficial personalization** in wearable EEG—initially hypothesized to corroborate EXP-019 `S3` (`UA-2` = 0.00 pp), though Phase A2 forensic audit (`S3_INFOMAX_NOOP_AUDIT_REPORT.md`, DEC-030) subsequently proved that EXP-019 `S3` was an implementation no-op (signal-unit mismatch), rendering S3 scientifically uninformative.
  2. [Zhao et al. (2023)](https://arxiv.org/html/2306.03536) demonstrated that TTA methods collapse under strong **label shift** ($\alpha = 0.01$) and suffer from online batch dependency—directly paralleling EXP-019's `Bidslab34` label-shift collapse.
  3. [Phan et al. (2020)](https://lirias.kuleuven.be/retrieve/616969) found subject-specific normalization caused poor personalization in subjects with large inter-night signal magnitude differences, requiring night-specific normalization.
  4. [Li et al. (2026)](https://arxiv.org/html/2604.27033v1) highlighted that pooling heterogeneous subjects into single-source domain adaptation risks negative transfer.
- **Newly Discovered Gaps:** `G001` (Safe/Selective adaptation under calibration label shift), `G002` (KL-regularized adaptation on low-SNR wearable IHR+Accel), `G003` (External validation on SleepAccel/MESA), `G004` (Clinical Bland-Altman agreement).
- **Reason for Change:** Initial Phase 1 establishment of the Research Knowledge & Literature Intelligence System.
