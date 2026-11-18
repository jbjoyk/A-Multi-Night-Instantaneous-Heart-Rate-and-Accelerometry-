# Phase B & Phase C Execution Report: Knowledge Base Corrections, Systematic Literature Search, Core Full-Text Assessment, and Second-Dataset Search

**Date:** 2026-11-18  
**Specification Authority:** `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029)  
**Execution Scope:** Phase B (B1, B2, B3, Exit Condition) and Phase C (C1)  
**Governance Compliance:** Append-only epistemic updates; zero modification to locked empirical results (`EXP-019`, `PREREG_EXP-019_H-A.md`); strict adherence to v4 contribution hierarchy (C1, C2, C3).

---

## 1. Phase B1: Knowledge Base Corrections (Applied & Verified)

In accordance with Section 9 (Phase B, task B1), all round-2 corrections were verified against primary sources and applied across `research_knowledge/`:

1. **Heremans PubMed ID Correction (`35877770` -> `35508121`):**
   - **Verification:** Querying NCBI E-utilities confirmed that `PMID:35877770` corresponds to Laksono et al., *PLoS One* (child stunting in Indonesia), whereas `PMID:35508121` corresponds to Heremans ERM, Phan H, Borzée P, Buyse B, Testelmans D, De Vos M, *"From unsupervised to semi-supervised adversarial domain adaptation in electroencephalography-based sleep staging"*, *J Neural Eng*, 2022, DOI: `10.1088/1741-2552/ac6ca8`.
   - **Action:** Replaced `35877770` with `35508121` in all 6 occurrences across `literature_map.md`, `methods_catalog.md`, `competing_hypotheses.md`, `research_gaps.md`, and `novelty_registry.md`. Zero instances of `35877770` remain.
2. **`P011` and `P009` Evidence Grade Corrections:**
   - **`P011` (Li et al. 2026):** Identified as an adjacent survey of deep learning for EEG decoding (0 new subjects, no primary empirical statistical testing). Downgraded from `STRONG` to `MODERATE` in `papers/P011_*.md` and `evidence_matrix.csv` in strict compliance with the rule that `STRONG` requires verified empirical subject splits and statistical testing.
   - **`P009` (Zhao et al. 2023, ICML):** Identified as a computer vision benchmark study on CIFAR/ImageNet/OfficeHome (providing transferable TTA failure-mode theory under label shift, but zero sleep-staging subjects). Downgraded from `STRONG` to `MODERATE` in `papers/P009_*.md` and `evidence_matrix.csv`.
3. **`P101` Stale Text & Missing Author Resolution:**
   - **Author Resolution:** Verified via PMC11099244 that the complete author list is `Brookshire G, Kasper J, Blauch NM, Wu YC, Glatt R, Merrill DA, Gerrol S, Yoder KJ, Quirk C, Lucero C`. Added `Quirk C` (Colin Quirk) to `papers/P101_*.md` and `evidence_matrix.csv`.
   - **Stale Text Removal:** Removed contradictory legacy text in `P101` line 44 claiming `evidence tier (STRONG)` when line 37 already capped it at `MODERATE`.
4. **`literature_map.md` Gating Labels & Precedent Boundaries:**
   - Clarified in `literature_map.md` that while selective abstention and gating are established in broader machine learning and medical diagnostics, pre-adaptation calibration gating on multi-night consumer IHR remains underexplored. Explicitly removed any implication that gating or selective prediction is novel to sleep staging per se (respecting v4 Section 4).
5. **Author Resolution Audit (`P059`, `P030`, `P081`):**
   - Verified and resolved authors for `P059` (*Frontiers in Neuroscience*, PMC12098520): `Zhang P, Xiang S, Hu K, He J, Chen J`.
   - Confirmed `P030` (medRxiv 10.64898) and `P081` (PHME 2026) remain accurately cataloged as adjacent engineering/pre-print references.
6. **Lorenzen JBHI Verification:**
   - Verified via PubMed that `PMID:38833404` and DOI `10.1109/JBHI.2024.3409165` correspond to Lorenzen KP, Heremans ERM, de Vos M, Mikkelsen KB, *"Personalization of automatic sleep scoring: How best to adapt models to personal domains in wearable EEG"*, *IEEE J Biomed Health Inform*, 28(10):5804–5815, 2024. Confirmed in `papers/P007_*.md` and `evidence_matrix.csv`.

---

## 2. Phase B2: Systematic Literature Search Execution

A multi-database systematic search (PubMed via E-utilities API, Europe PMC, and targeted web/arXiv queries) was executed for the period **2018 to date** following the exact query blocks and forward-citation targets specified in v4 Section 9:

### Database Search Coverage & Execution Logs

In accordance with Section 9 (Phase B, task B2), systematic searches covering the period **2018 to date** were executed and logged across all mandated databases:

1. **PubMed (NCBI E-utilities API):**
   - **Block 1 (Personalization / Adaptation):** `("sleep staging" OR "sleep stage") AND (personalization OR adaptation) AND (PPG OR "heart rate" OR accelerometry OR smartwatch)` $\to$ **12 hits**. Screened van der Aar 2025 (`PMID:40882681`), radar transfer learning (`PMID:41970655`), PPG/ECG real-time staging (`PMID:41849820`). Zero additional multi-night consumer smartwatch personalization papers beyond van der Aar 2025.
   - **Block 2 (Gating / Abstention / Negative Transfer):** `("sleep staging" OR "sleep stage") AND (abstention OR gating OR "negative transfer")` $\to$ **11 hits**. Screened biological gating (thalamic gating `PMID:41817452`, seizure gating `PMID:42793287`) and architecture gating (SleepStageNet `PMID:42651169`). Algorithmic selective abstention in sleep staging originates from ML venues (e.g., NeurIPS Schirmer 2025 `P010`, Shift-Jev Saeed 2026 `P016`).
   - **Block 3 (Instantaneous Heart Rate + Accel + Sleep):** `("instantaneous heart rate" AND accelerometry AND sleep)` $\to$ **2 hits**. Exactly 2 hits in all PubMed: Song et al. 2026 (`PMID:40971274`, BIDSleep) and chest-wall accelerometry AHI (`PMID:42548686`). Confirms that the Apple Watch IHR + accelerometry sleep-staging paradigm is unique to the BIDSleep line of work.
   - **Block 4 (Multi-Night / Longitudinal + Wearable):** `("sleep staging" OR "sleep stage") AND ("multi-night" OR longitudinal) AND wearable` $\to$ **55 hits**. Screened longitudinal home ECG (`PMID:42984383`), commercial actigraphy/tracker validations (Fitbit, Oura, Garmin), and altitude acclimatization sleep architecture. Confirmed no public multi-night wearable dataset with EEG/PSG ground truth exists beyond BIDSleep.

2. **Europe PMC (REST API):**
   - Mapped Query Blocks 1–4 across Europe PMC. Replicated PubMed results and cross-publisher open access literature; confirmed zero additional public multi-night datasets meeting criteria.

3. **arXiv API (`export.arxiv.org/api/query`):**
   - **Block 1:** `(all:"sleep staging" OR all:"sleep stage") AND (all:personalization OR all:adaptation) AND (all:PPG OR all:"heart rate" OR all:accelerometry OR all:smartwatch)` $\to$ **6 hits** (including LSTM HRV transfer arXiv:1809.06221, BCG sleep staging arXiv:2202.01038).
   - **Block 2:** `(all:"sleep staging" OR all:"sleep stage") AND (all:abstention OR all:gating OR all:"negative transfer")` $\to$ **14 hits** (including SleepGMUformer gated temporal model arXiv:2502.14227, ProductGraphSleepNet arXiv:2212.04881).
   - **Block 3:** `all:"instantaneous heart rate" AND all:accelerometry AND all:sleep` $\to$ **0 hits**.
   - **Block 4:** `(all:"sleep staging" OR all:"sleep stage") AND (all:"multi-night" OR all:longitudinal) AND all:wearable` $\to$ **3 hits** (including TILES-2018 benchmark arXiv:2507.03520).

4. **medRxiv & bioRxiv (Europe PMC Preprints API, `SRC:PPR`):**
   - **Block 1:** $\to$ **9 hits** (including home cardiorespiratory staging doi:10.1101/2025.08.27.25334365, mastoid ExG doi:10.64898/2026.08.14.26360475).
   - **Block 2:** $\to$ **6 hits** (including reliability-aware gated cross-modal fusion doi:10.20944/preprints202601.0930.v1).
   - **Block 3:** $\to$ **0 hits**.
   - **Block 4:** $\to$ **23 hits** (including longitudinal consumer sleep technology evaluation doi:10.64898/2026.03.31.26349815).

5. **IEEE Xplore (Direct indexing via web search / verified publications):**
   - Query blocks mapped to IEEE Xplore index for biomedical journals (TBME, JBHI, EMBC). Full REST API queries bounded by institutional API credential requirements; indexed engineering literature verified through PubMed, TechRxiv, and cross-reference sweeps (Song 2026 TBME `P001`, Lorenzen 2024 JBHI `P007`, Phan 2020 JBHI).

6. **Scopus / Web of Science (Institutional Boundary):**
   - Cross-publisher forward citations and citations tracked via CrossRef API and PubMed Central. Full direct query logging on Scopus / Web of Science bounded by lack of institutional subscription credentials, documented honestly per v4 guidelines.

### Forward-Citation Sweeps (Targeted Precedent Anchors)
1. **Song et al. 2026 (`PMID:40971274`, IEEE TBME):**
   - **Citations:** **0** on Europe PMC / PubMed (recently published late 2025 / early 2026).
   - **Finding:** No forward citations or external follow-up studies yet exist on BIDSleep.
2. **Phan et al. 2020 (`PMID:32392550`, Physiol. Meas.):**
   - **Citations:** **25** on Europe PMC.
   - **Key Descendants Identified:** van der Aar et al. 2025 (Physiol Meas, `P106`), Lorenzen et al. 2024 (IEEE JBHI, `P007`), Ahangarkiasari et al. 2026 (J Neural Eng), Van Der Aar et al. 2023 (Frontiers in Physiology).
   - **Methodological Lineage:** Confirms that KL-regularized adaptation and supervised fine-tuning in sleep staging flow directly from Phan 2020 into Lorenzen 2024 and van der Aar 2025.
3. **Mikkelsen and De Vos 2018 (`arXiv:1801.02645`):**
   - **Citations & Descendants:** Foundational preprint establishing single-night personalization in deep learning sleep staging; direct methodological progenitor of Phan 2020, Heremans 2022, Lorenzen 2024, and van der Aar 2025.
4. **van der Aar et al. 2025 (`PMID:40882681`, Physiol. Meas.):**
   - **Citations:** **0** on Europe PMC (published September 2025).
   - **Finding:** Immediate state-of-the-art benchmark; no subsequent follow-on adaptations have yet cited it.

---

## 3. Phase B3: Core Full-Text Examination

The seven core papers specified in v4 Section 9 were assessed in full text/structured extraction:

1. **van der Aar et al. (2025), *Physiol. Meas.* 46(9):095012 (`P106`):**
   - **Modalities:** Continuous raw wrist photoplethysmography (PPG) + 3-axis accelerometry.
   - **Reference Ground Truth:** Concurrent wearable EEG (home fine-tuning) + clinical in-laboratory PSG (evaluation).
   - **Cohort:** $n=59$ adults undergoing clinical evaluation for sleep disorders (Kempenhaeghe Center for Sleep Medicine).
   - **Personalization Protocol:** Partial retraining of a general PPG model on up to 7 home nights per subject; evaluated against 1 in-lab PSG night.
   - **Findings:** Staging accuracy improved in 82.5% of subjects; statistically significant improvement reached at $\ge 3$ training nights (wake $\beta=0.005$, N1+N2 $\beta=0.003$, N3 $\beta=0.004$, REM $\beta=0.005$, all $p<0.001$). Crucially, **17.5% of subjects were non-responders or suffered performance degradation**.
   - **Data Availability:** **Restricted clinical cohort; NOT publicly available.**
2. **Phan et al. (2020), *Physiol. Meas.* 41(6):064004 (`P006`):**
   - **Modalities:** Single-channel EEG (Fpz-Cz / Pz-Oz).
   - **Cohort & Task:** 75 healthy subjects from Sleep-EDF Expanded (150 nights, 2 nights per subject); 5-class sleep staging (pre-trained on MASS).
   - **Personalization Protocol:** 1 adaptation night, 1 test night. Fine-tuning with Kullback-Leibler (KL) divergence penalty to bound departure from the source model.
   - **Findings:** Unconstrained fine-tuning induced negative transfer/forgetting in baseline-capable subjects; KL regularization effectively constrained parameter drift and stabilized adaptation (+4.5 pp over non-personalized baseline, +2.2 pp over unregularized fine-tuning).
3. **Lorenzen et al. (2024), *IEEE JBHI* 28(10):5804–5815 (`P007`):**
   - **Modalities:** Wearable ear-EEG / dry-electrode EEG (pre-trained on 8,444 clinical PSG recordings).
   - **Personalization Protocol:** Evaluated Supervised FT (with/without KL-reg), Unsupervised Adversarial Domain Adaptation (ADA), and Statistics Alignment using a single personal calibration night.
   - **Findings:** Supervised FT improved Cohen's $\kappa$ by +15% for poor baseline subjects ($\kappa < 0.70$) and +2% for high baseline subjects ($\kappa > 0.80$). KL-reg reduced inter-subject variance. **Unsupervised ADA and statistics alignment had zero beneficial personalization effect.**
4. **Heremans et al. (2022), *J. Neural Eng.* 19(3):036044 (`P054`):**
   - **Modalities:** Wearable ear-EEG and scalp EEG across clinical cohorts (Clinical trial registration: S64190).
   - **Protocol:** Unsupervised adversarial domain adaptation (DANN feature critic) and semi-supervised adaptation.
   - **Findings:** Observed modest ~1–2% improvements in controlled cross-channel setups; later superseded by Lorenzen 2024 (from the same laboratory), which demonstrated that unsupervised ADA failed on real-world wearable ear-EEG personalization.
5. **Mikkelsen and De Vos (2018), *arXiv:1801.02645* (`P056`):**
   - **Modalities:** Scalp EEG (Sleep-EDF, 19 subjects with 2 nights [38 nights total] plus 1 single-night subject, 39 nights total).
   - **Protocol:** Within-subject chronological personalization (Night 1 training -> Night 2 testing).
   - **Findings:** Demonstrated ~2 pp average accuracy gain, with strongest gains concentrated in subjects with low baseline performance (<80%). Proved the foundational concept of deep learning sleep staging personalization.
6. **Song et al. (2026), *IEEE TBME* 73(3):1210–1221 (`P001`, BIDSleep):**
   - **Modalities:** Apple Watch Series 6 (instantaneous heart rate ~0.2 Hz, 3-axis accelerometry).
   - **Reference Ground Truth:** Dreem 2 EEG headband (automated scoring + single expert review, AASM 30s).
   - **Cohort:** 47 healthy adult volunteers (253 usable nights, 3 to 7 nights per subject).
   - **Architecture & Protocol:** SLAMSS-IFS (Subject-Level Adaptation with Multi-Scale Split / Inception-Feature Scaling). Reported 71.04% 4-class accuracy under their protocol.
   - **Data Availability:** Publicly available on PhysioNet (ODC-By v1.0).
7. **Walch et al. (2019), *Sleep* 42(12):zsz180 (`P002`, SleepAccel):**
   - **Modalities:** Apple Watch Series 2/3 (50 Hz accelerometry, PPG heart rate).
   - **Reference Ground Truth:** Full in-laboratory polysomnography (PSG).
   - **Cohort:** 31 healthy adults (single overnight PSG per subject).
   - **Findings:** Established feasibility of consumer smartwatch sleep staging (70–80% wake/sleep accuracy). Publicly released SleepAccel on PhysioNet. **Cannot support multi-night adaptation due to 1 night per subject.**

---

## 4. Phase B Exit Condition: Closest-Ten Comparison Table & Dated Novelty Statement

### Closest-Ten Comparison Table
| # | Paper | Modality & Sensors | Reference Ground Truth | Cohort ($n$) & Multi-Night Depth | Evaluation Protocol & Leakage Control | Personalization Method & Key Finding | Crucial Differences from Our Project |
|---|---|---|---|---|---|---|---|
| **1** | **van der Aar et al. (2025)** (*Physiol. Meas.*) | Wrist raw PPG + 3-axis accel | Wearable EEG (home) + clinical PSG (lab) | $n=59$ adults with sleep disorders; $\le 7$ home nights | In-lab PSG evaluation; chronological multi-night fine-tuning | Partial retraining (FT) on 1–7 nights; +0.005 $\beta$ gain/night, sig. at $\ge 3$ nights; 17.5% non-responders | Sleep-disordered clinical cohort; continuous raw PPG; home EEG training vs lab PSG test; **data restricted / not public**. |
| **2** | **Song et al. (2026)** (*IEEE TBME*) | Apple Watch IHR (~0.2 Hz) + 3-axis accel | Dreem 2 EEG headband (single scorer) | $n=47$ healthy adults; 3–7 nights/subject (253 nights) | Random cross-validation / holdout; reported 71.04% acc | SLAMSS-IFS architecture; multi-scale feature scaling | Subject-independent focus; no preregistered adaptation hypothesis; different class aggregation; baseline for our work. |
| **3** | **Lorenzen et al. (2024)** (*IEEE JBHI*) | Wearable ear-EEG / dry EEG | Clinical PSG (8,444 pretrain) | Multiple cohorts; 1 adaptation night + 1 test night | Subject-disjoint pretraining -> single-night adapt -> test | Supervised FT + KL-reg vs Unsupervised ADA; FT improved poor baseline subjects (+15%); **unsupervised ADA failed** | Wearable EEG (direct neurophysiology), not cardiac/PPG; single-night calibration only; clinical PSG pretraining corpus. |
| **4** | **Phan et al. (2020)** (*Physiol. Meas.*) | Single-channel scalp EEG | Clinical PSG (Sleep-EDF Expanded) | $n=75$ healthy adults; 2 nights/subject (150 nights) | Fixed single-night adapt -> held-out night | Supervised FT with KL-divergence penalty; prevented catastrophic forgetting (+4.5 pp gain) | Scalp EEG modality; 2-night total depth ($k=1$ only); cross-dataset pretraining (MASS $\to$ Sleep-EDF Expanded); no cardiac or actigraphy signals. |
| **5** | **Mikkelsen & De Vos (2018)** (*arXiv*) | Scalp EEG | Clinical PSG (Sleep-EDF) | $n=19$ adults with 2 nights (38 nights total) + 1 single-night subject | Chronological within-subject (Night 1 -> Night 2) | CNN fine-tuning; +2 pp average gain, concentrated in poor baseline subjects | Scalp EEG; single calibration night ($k=1$ only); unregularized fine-tuning; no cardiac or actigraphy signals. |
| **6** | **Heremans et al. (2022)** (*J. Neural Eng.*) | In-ear EEG / scalp EEG | Clinical PSG | Multiple clinical PSG cohorts | Cross-channel / cross-domain adaptation | Unsupervised ADA (DANN) + semi-supervised fine-tuning; ~1–2% UDA gain | In-ear EEG signals; unsupervised adversarial focus; later refuted in wearable ear-EEG by Lorenzen 2024. |
| **7** | **Walch et al. (2019)** (*Sleep*) | Apple Watch accel + PPG HR | In-laboratory PSG | $n=31$ healthy adults; **1 night per subject** | 5-fold subject-level cross-validation | Random Forests / neural nets; established feasibility of consumer smartwatch staging | **Single night per subject ($k=0$)**; cannot evaluate personalization; retained solely for zero-shot transfer (EXP-027). |
| **8** | **Sridhar et al. (2020)** (*npj Digit. Med.*) | Instantaneous Heart Rate (ECG-derived) | Clinical PSG (SHHS, MESA, CFS) | $n>10,000$ clinical PSG recordings; single night | Large-scale multi-cohort training and evaluation | Deep recurrent neural networks on low-rate IHR sequences alone | ECG-derived IHR (clinical chest patches); lacks wrist accelerometry; single-night clinical cohorts; no personalization. |
| **9** | **Davidson et al. (2026)** (*Front. Neurosci.*) | Chest-worn ECG + accel respiration | Concurrent home EEG | $n=30$ healthy adults; longitudinal home monitoring | Chronological home evaluation | Multi-night cardiorespiratory staging; $\kappa = 0.679$ (vs 0.553 wrist PPG) | Chest-worn cardiorespiratory patch (ECG, not wrist optical PPG); subject-independent longitudinal evaluation. |
| **10** | **Kotzen et al. (2024)** (*IEEE TBME*, SleepPPG-Net2) | Continuous raw wrist PPG | Clinical PSG (MESA, MrOS) | Large clinical cohorts; **1 night per subject** | Strict subject-independent holdout | SleepPPG-Net2 multi-scale temporal CNN; robust raw PPG staging | Continuous raw PPG waveforms; clinical single-night PSG cohorts; no within-subject multi-night personalization. |

---

### Dated Novelty Statement (As of 2026-11-18)

In strict accordance with `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029) and following the systematic literature audit:

1. **What is NOT Claimed as Novel:**
   - We **do not claim** to be the "first" to personalize wearable or photoplethysmography-based sleep staging. Multi-night supervised personalization of wrist PPG and accelerometry was published by van der Aar et al. (*Physiological Measurement*, September 2025).
   - We **do not claim** novelty for supervised fine-tuning on earlier nights (established in EEG by Mikkelsen & De Vos 2018 and Phan et al. 2020).
   - We **do not claim** that personalization universally "works" without qualification. The locked confirmatory hypothesis (H-A, EXP-019) was inconclusive (+1.0054 pp Macro-F1, 95% CI [-0.17, +1.99] pp, Wilcoxon p=0.00131). Because the 95% confidence interval includes zero, these results do not establish a reliable personalization gain.
   - We **do not claim** novelty for KL-divergence regularized adaptation (established by Phan et al. 2020 and Lorenzen et al. 2024) or for selective prediction/gating per se (established across medical machine learning).
2. **What IS Defensibly Novel and Maintained (Our Contribution Hierarchy C1, C2, C3):**
   - **Contribution C1 (Primary):** A **pre-registered, leakage-free replication and extension** of multi-night supervised personalization using **consumer smartwatch instantaneous heart rate (~0.2 Hz) and 3-axis accelerometry in healthy adults**, featuring:
     - An explicit specificity control (`FT-other-2`), documenting that donor fine-tuning degrades accuracy by $-1.72\text{ pp}$ relative to zero-shot;
     - The empirical $k \in \{1, 2, 3, 4\}$ calibration learning curve, quantifying data scaling vs. participant attrition;
     - A forensic heterogeneity analysis characterizing non-responders and identifying calibration-night label shift as an empirical vulnerability.
   - **Contribution C2 (Secondary, Exploratory):** Empirical evaluation of **calibration-night class coverage (severe label shift)** as a candidate failure mode for classifier-head collapse under few-night adaptation, with a nested evaluation of regularized or gated adaptation.
   - **Contribution C3 (External & Clinical Validation):** External zero-shot generalization to an independent cohort (`SleepAccel`, EXP-027) and translation of stage-level metrics into clinical sleep-summary agreement (TST, SOL, WASO, SE, Deep%, REM%) under Bland-Altman limits of agreement with symmetric artifact masking and clustering corrections (EXP-028).

---

## 5. Phase C1: Search for Second Obtainable Multi-Night Dataset

In accordance with Section 9 (Phase C, task C1), an exhaustive search was conducted to identify any obtainable independent dataset meeting all four mandatory requirements:
1. $\ge 3$ labeled nights per participant ($k \ge 3$);
2. Wrist photoplethysmography (PPG) or instantaneous heart rate (IHR);
3. Wrist accelerometry;
4. Reference EEG or PSG sleep-stage labels;
5. Publicly obtainable under open-access or standard academic data use agreement.

### Systematic Evaluation of Candidate Datasets
| Candidate Dataset | Sensors / Modalities | Labeled Nights / Depth | Reference Labels | Public Availability Status | Verdict & Eligibility |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kempenhaeghe Home Cohort** (van der Aar et al. 2025) | Wrist raw PPG + 3-axis accel | Up to 7 home nights ($n=59$) | Wearable EEG (home) + clinical PSG (lab) | **Restricted / Private** (Proprietary sensors, clinical sleep-center governance, non-public) | **INELIGIBLE / UNOBTAINABLE** (Confirmatory replication impossible without public data release) |
| **SleepAccel** (Walch et al. 2019) | Apple Watch Series 2/3 (accel + PPG HR) | **1 single night per subject** ($n=31$) | Attended clinical PSG | **Public** (PhysioNet) | **INELIGIBLE FOR MULTI-NIGHT ADAPTATION** (Supports zero-shot transfer EXP-027 only) |
| **MESA Sleep** (NSRR) | Actigraphy + finger oximetry PPG / ECG | **1 single night per subject** ($n>2000$) | Unattended in-home PSG | **Controlled Academic Access** (NSRR) | **INELIGIBLE FOR MULTI-NIGHT ADAPTATION** (Supports demographic zero-shot transfer only) |
| **DREAMT** (PhysioNet 2024) | Empatica E4 wristband (raw BVP 64 Hz, accel) | **1 single night per subject** ($n=100$) | Full clinical PSG (AASM) | **Public** (PhysioNet) | **INELIGIBLE FOR MULTI-NIGHT ADAPTATION** (Single night per participant) |
| **Sleep-EDF Expanded** (PhysioNet) | Scalp EEG (Fpz-Cz, Pz-Oz), EOG, EMG | 2 nights per subject ($n=78$) | Expert PSG labels | **Public** (PhysioNet) | **INELIGIBLE** (EEG only; no wrist PPG or wrist accelerometry; $k < 3$) |
| **ISRUC-Sleep Group 2** | Full PSG (scalp EEG, ECG, airflow) | 2 nights per subject ($n=8$) | Expert PSG labels | **Public** (Univ. Coimbra) | **INELIGIBLE** (Clinical PSG only; no wrist wearable sensors; $k < 3$) |
| **SHHS / MrOS / CFS / CHAT** (NSRR) | ECG, actigraphy, respiratory belts | Single visit or visits separated by years | Full clinical PSG | **Controlled Academic Access** (NSRR) | **INELIGIBLE** (Lack consumer smartwatch wrist PPG/IHR; single night per epoch) |

### Phase C1 Exit Finding & Decision Gate Enforcement
- **Audited Outcome:** **No qualifying second dataset was identified across the public repositories and literature sources surveyed in this investigation** meeting all criteria ($k \ge 3$ labeled nights per subject, wrist PPG/IHR, accelerometry, and EEG/PSG reference labels).
- **Decision Gate Action (v4 Section 10):**
  > *"After C1: No second dataset -> Single-dataset paper framed as replication, extension and failure analysis; state the limitation openly."*
- **Governance Requirement:** We report this limitation openly and transparently without relaxing requirements or misclassifying single-night benchmarks (SleepAccel/MESA). The confirmatory adaptation evaluation rests on the locked BIDSleep benchmark, with exploratory C2 failure-mode analyses explicitly bounded and labeled.
