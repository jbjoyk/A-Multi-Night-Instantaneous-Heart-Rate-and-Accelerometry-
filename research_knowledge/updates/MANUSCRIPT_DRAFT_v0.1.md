# Manuscript Draft: Sections 1–5 (Version 0.1)

**Working Title:** *Does a few labeled nights of personal data help wearable sleep staging? A pre-registered, leakage-free study with consumer-smartwatch heart rate and accelerometry.*  
**Document Status:** Working Draft v0.1 (Sections 1–5: Introduction, Related Work, Dataset & Task Definition, Methods, Evaluation Protocol)  
**Governance Authority:** `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029), Section 15; DEC-021  
**Target Venue Plausibility:** Biomedical engineering or sleep informatics (e.g., *IEEE Journal of Biomedical and Health Informatics*, *Computers in Biology and Medicine*, *Biomedical Signal Processing and Control*, or *Journal of Sleep Research*)  
**Draft Date:** 2026-11-18  

---

## 1. Introduction

### 1.1 Clinical and Wearable Context
Sleep is a fundamental physiological process governing metabolic recovery, immune competence, neural plasticity, and cognitive performance. In contemporary sleep medicine, sleep architecture is quantified according to the American Academy of Sleep Medicine (AASM) manual into four canonical states: Wakefulness ($W$), Light sleep (combining Non-Rapid Eye Movement stages N1 and N2), Deep slow-wave sleep (N3), and Rapid Eye Movement (REM) sleep ($R$). The gold standard for assessing sleep architecture is laboratory polysomnography (PSG), which records electroencephalography (EEG), electrooculography (EOG), electromyography (EMG), and electrocardiography (ECG). Although PSG provides high diagnostic fidelity, it suffers from severe practical limitations: it requires specialized facilities, encumbering contact montages, manual scoring by credentialed technologists, and high financial cost. Crucially, in-lab PSG captures only one or two isolated nights, rendering it susceptible to the "first-night effect" and inadequate for characterizing longitudinal night-to-night sleep dynamics across weeks or months.

Recent advances in consumer wearables—most notably optical photoplethysmography (PPG) smartwatches and tri-axial accelerometers—have democratized longitudinal physiological tracking in naturalistic home environments. Rather than recording raw electrical brain activity, commercial smartwatches infer sleep architecture indirectly through autonomic nervous system (ANS) tone, cardiac autonomic modulation, and peripheral biomechanical immobility. Specifically, during Non-REM sleep, parasympathetic vagal activity predominates, resulting in decreased heart rate and heightened respiratory sinus arrhythmia; conversely, during REM sleep, sympathetic bursts elicit pronounced heart rate fluctuations and transient tachypnea against a background of muscle atonia.

### 1.2 The Promise and Peril of Personalization
Despite these physiological correlations, mapping surrogate peripheral vitals to central EEG-defined sleep stages remains an inherently ill-posed problem. Standard machine learning models trained on large population cohorts exhibit noticeable performance degradation when evaluated on unseen individuals. This inter-individual variance stems from substantial biological heterogeneity: basal heart rates, autonomic responsivity, vascular compliance, resting movement patterns, and circadian phase shifts vary dramatically across persons due to age, biological sex, cardiorespiratory fitness, and psychological stress.

A natural and appealing solution to this domain shift is **model personalization**: fine-tuning a general population model on a small number of labeled nights acquired from a specific target individual. In clinical practice, this corresponds to a scenario where an individual wears a convenient home-EEG headband (such as Dreem or similar clinical dry-sensor devices) for their initial two or three nights to generate ground-truth training labels, after which the fine-tuned wearable model operates autonomously on smartwatch vitals alone. 

However, transferring machine learning models via parameter fine-tuning on extremely small personal sample sizes ($k = 1$ to $4$ nights, comprising roughly 800 to 4,000 thirty-second epochs per subject) introduces severe empirical vulnerabilities. When unconstrained gradient descent is applied to small, class-imbalanced calibration nights, deep neural networks risk catastrophic forgetting, calibration overfitting, and negative transfer—in which personalized fine-tuning paradoxically impairs classification performance on subsequent nights relative to the unmodified zero-shot population model.

### 1.3 Addressed Research Gaps
While personalization has garnered considerable enthusiasm across digital health, the published literature suffers from three foundational gaps:

- **Gap G1: Unquantified Personalization Efficacy in Healthy Cohorts Under Leakage-Free Evaluation.** Within-person multi-night adaptation has been demonstrated for raw optical PPG in sleep-disordered clinical populations (e.g., van der Aar et al., 2025). However, its behavior when using low-rate instantaneous heart rate (IHR; $\sim 0.2\text{ Hz}$, typical of Apple HealthKit) and accelerometry in healthy adult cohorts under strictly chronological, leakage-free, pre-registered evaluation—including rigorous quantification of who is helped, who is harmed, and the underlying failure mechanisms—remains unestablished.
- **Gap G2: Protocol Heterogeneity and Incomparable Subject-Independent Benchmarks.** Published subject-independent sleep staging baselines using low-rate cardiac vitals report widely disparate performance figures (e.g., overall accuracies spanning 62% to 75%). As our forensic audits demonstrate, these discrepancies arise primarily from severe protocol heterogeneity: arbitrary subject-versus-night data splitting, epoch cropping (e.g., truncating to 5-hour fixed windows), disparate sleep stage class mappings, and selective reporting of weighted F1 rather than balanced Macro-F1. A rigorous, leakage-free benchmark under uniform full-night evaluation is critically lacking.
- **Gap G3: Disconnect Between Stage-Level Metrics and Clinical Sleep Indices.** Wearable staging studies overwhelmingly report aggregate epoch-level accuracy or F1-scores, rarely translating these numbers into clinically actionable nocturnal sleep metrics such as Total Sleep Time (TST), Sleep Onset Latency (SOL), Wake After Sleep Onset (WASO), Sleep Efficiency (SE), and Deep or REM sleep percentages. Furthermore, the clinical agreement and robustness of personalized models against missing or irregular heart rate sampling intervals remain largely uncharacterized.

### 1.4 Formal Contributions and Claim Boundaries
To resolve these gaps, this study presents a rigorous, pre-registered, and leakage-free empirical investigation into wearable sleep staging personalization using the public multi-night BIDSleep dataset (Song et al., 2026; PhysioNet, DOI: 10.13026/rees-1092). We circumscribe our contributions strictly within verifiable empirical boundaries:

1. **Contribution C1 (Primary): Rigorous Quantification of Multi-Night Adaptation.** We conduct a pre-registered, chronologically ordered, subject-disjoint investigation into whether fine-tuning on a held-out individual's first two labeled nights improves 4-class sleep staging on subsequent nights. Our protocol incorporates negative specificity controls (fine-tuning on arbitrary donor subjects, FT-other), per-subject training from scratch (PS-2), unsupervised adaptation (UA-2), and empirical evaluation of the calibration trajectory across $k = 1, 2, 3, 4$ nights.
2. **Contribution C2 (Diagnostic & Methodological): Failure Mode Dissection and Constrained Adaptation.** Through forensic audit of subject-level heterogeneity, we explore calibration-night class coverage (such as severe scarcity of Deep slow-wave sleep observed in outlier cases) as a candidate failure mechanism associated with negative transfer. We evaluate regularized fine-tuning (Kullback-Leibler divergence constraints, FT-2-KL) and data-quality gating ($\text{GATE}_\tau$) as principled exploratory mechanisms to mitigate personalization failure.
3. **Contribution C3 (Benchmarking & Clinical Translation): Leakage-Free Baseline and Agreement Analysis.** We deliver an audited, leakage-free 5-fold cross-validation benchmark comparing our parameter-efficient sequence architecture against handcrafted gradient-boosted trees and prior published architectures under identical full-night temporal contexts ($L = 1200$ epochs), supplemented by Bland-Altman agreement analysis of clinical sleep-summary parameters.

**Explicit Claim Boundaries:** We do *not* claim to present the first multi-night personalization study, nor do we claim that fine-tuning universally succeeds. We explicitly disclose that our primary confirmatory hypothesis (H-A, EXP-019) was statistically inconclusive under our pre-registered decision rules. We make no claims of algorithmic superiority over published benchmarks evaluated under disparate single-split protocols, nor do we claim generalization to clinical sleep-disordered cohorts.

---

## 2. Related Work

### 2.1 Wearable Sleep Staging from Autonomic and Biomechanical Signals
The estimation of sleep stages from non-invasive peripheral sensors has a long history in actigraphy and ambulatory cardiology. Early commercial systems relied almost exclusively on actigraphy—measuring wrist movement via piezoelectric or MEMS accelerometers—to classify epochs as binary Wake or Sleep based on activity counts (Cole-Kripke and Sadeh algorithms). While actigraphy achieves high sensitivity for sleep detection, it exhibits notoriously poor specificity for detecting quiet wakefulness and is intrinsically incapable of resolving sleep architecture (Light, Deep, and REM stages).

The integration of optical photoplethysmography (PPG) into commercial wristbands enabled the continuous measurement of heart rate, pulse rate variability (PRV), and pulse wave morphology. Walch et al. (2019) demonstrated that combining Apple Watch instantaneous heart rate (IHR) and accelerometer movement data with circadian time proxies allowed a multi-layer neural network to identify Sleep vs. Wake and distinguish coarse sleep states in a cohort of 31 subjects. Beattie et al. (2017) utilized raw PPG-derived inter-beat intervals (IBI) and spectral HRV features to differentiate Wake, Light, Deep, and REM sleep, establishing that autonomic fluctuations mirror sleep architecture transitions.

Subsequent investigations focused on advanced neural backbones. Song et al. (2026) introduced SLAMSS-IFS, a sequence-to-sequence neural network combining multi-scale convolutions, bidirectional recurrent units, and an asymmetric real-world cost loss (RCCLoss) to address class imbalance on the multi-night BIDSleep dataset. Concurrently, lightweight sequence models such as temporal convolutional networks (TCNs) and bidirectional GRUs (Phan et al., 2020) have emerged as preferred architectures for resource-constrained wearable edge processing.

### 2.2 Protocol Heterogeneity and Data Leakage Pitfalls
A critical shortcoming in the wearable sleep staging literature is pervasive protocol heterogeneity and data leakage, which severely undermines cross-study comparability. In an exhaustive survey, Quirk et al. (2024 / P101) identified data leakage as a dominant source of inflated accuracy in deep learning sleep research. 

Leakage in multi-night studies primarily manifests in three modalities:
1. **Epoch-Level Random Splitting:** Randomly allocating individual 30-second epochs into train and test sets, which allows the network to interpolate between adjacent epochs of the identical night and subject, violating temporal independence.
2. **Night-Level Subject Splitting:** Partitioning dataset nights randomly across folds without subject grouping. Because physiological signals exhibit strong within-person autocorrelation (e.g., idiosyncratic basal heart rates and circadian patterns), the presence of a subject's Night 1 in training and Night 2 in testing constitutes severe identity leakage.
3. **Arbitrary Temporal Cropping:** Multiple published studies crop nocturnal recordings to a fixed 5-hour duration (e.g., $T = 600$ epochs) centered around sleep onset. As documented in our preliminary investigations (DEC-021), this arbitrary truncation discards over 28% of total labeled nocturnal epochs and disproportionately depletes REM sleep, which naturally clusters during the final third of the night.

Furthermore, published studies frequently diverge in sleep stage taxonomy (3-class Wake/NREM/REM vs. 4-class Wake/Light/Deep/REM vs. 5-class AASM staging) and selectively report weighted F1 or overall accuracy, which are heavily dominated by the majority Light sleep class ($>50\%$ of epochs) and mask catastrophic misclassification of Wake or REM states.

### 2.3 Personalization and Domain Adaptation in Physiological Signals
Personalization has been investigated across biomedical signal processing to combat inter-individual domain shift:

- **Wearable EEG Staging:** In ambulatory EEG, Mikkelsen et al. (2018) and Lorenzen et al. (2023) explored individual calibration using ear-EEG and forehead headbands. Lorenzen et al. reported that calibrating models with a user's initial sleep nights yielded measurable gains in Cohen's kappa, but emphasized that calibration was highly sensitive to electrode impedance stability and night-to-night stage distribution.
- **Wrist PPG Personalization:** Most relevant to our investigation is the groundbreaking study by van der Aar et al. (2025, *Physiological Measurement*, DOI: 10.1088/1361-6579/ae0119). Investigating a clinical cohort of $n = 59$ adults with suspected sleep disorders, the authors evaluated supervised personalization of wrist-worn PPG and accelerometry over up to seven home nights against laboratory PSG. They observed that supervised fine-tuning improved overall accuracy in 82.5% of subjects, with performance gains reaching statistical significance when three or more calibration nights were available ($k \ge 3$). However, their study evaluated high-resolution raw optical PPG in a clinically disordered population, relied on unconstrained full-network fine-tuning, and did not evaluate pre-registered directional stopping rules or explicit safeguards against catastrophic negative transfer.
- **Unsupervised Test-Time Adaptation (TTA):** Unsupervised adaptation strategies (such as Tent or entropy minimization; Zhao et al., 2023) have been proposed to adapt neural networks without ground-truth labels. However, as demonstrated in our empirical experiments, applying unregularized unsupervised adaptation to low-rate cardiac vitals without structural domain alignment risks trivial divergence or operational failure (no-op collapse).

---

## 3. Dataset and Task Definition

### 3.1 Dataset Description and Provenance
We conduct our investigation on the publicly available **BIDSleep** dataset (Song et al., 2026; PhysioNet, DOI: 10.13026/rees-1092; *IEEE Transactions on Biomedical Engineering*, DOI: 10.1109/TBME.2025.3612158). BIDSleep is an open-access multi-night multimodal dataset collected by the Biomedical Informatics and Data Science Laboratory (BIDS Lab) at the University of Massachusetts Amherst.

The cohort consists of **47 healthy adult participants** (denoted `Bidslab00` through `Bidslab68`), who wore consumer smartwatches and ambulatory EEG headbands concurrently during sleep in home living environments for up to seven consecutive nights. Data collection received appropriate institutional review board (IRB) approval, and the public release is licensed under the Open Data Commons Attribution License (ODC-By v1.0).

### 3.2 Sensor Modalities and Signal Quality Screening
Participants wore an **Apple Watch Series 6** on their non-dominant wrist and a **Dreem 2 EEG headband** (Beacon Biosignals) on the forehead:

1. **Instantaneous Heart Rate (IHR):** Heart rate is recorded via Apple HealthKit from the watch's optical photoplethysmography sensor. Unlike high-frequency research PPG (typically 25–100 Hz), Apple HealthKit delivers non-uniformly sampled instantaneous heart rate readings. Across the 246 usable recording nights, the median effective sampling rate is **$0.197\text{ Hz}$** ($\text{IQR} = 0.073\text{ Hz}$; approximately one reading every 5.08 seconds), throttling down to $\sim 0.064\text{ Hz}$ during episodes of vigorous nocturnal movement. The cohort-wide nightly median heart rate is $64.0\text{ BPM}$ ($\text{IQR} = 9.8\text{ BPM}$, range $49.0 - 87.0\text{ BPM}$).
2. **Tri-Axial Accelerometry:** Linear acceleration is continuously sampled along three orthogonal axes ($x, y, z$) by the onboard smartwatch accelerometer. We extract acceleration magnitude ($m = \sqrt{x^2 + y^2 + z^2}$), movement intensity, and jerk to quantify somatic motor activity.
3. **Reference Sleep Stage Labels:** Ground truth sleep stages were recorded simultaneously by the 5-channel dry-sensor Dreem 2 headband, which automatically scored 30-second epochs according to AASM standards. Every automated sleep recording was subsequently reviewed and validated by a credentialed sleep technologist (`expert_label`).

**Quality Screening and Usable Night Inventory:** The dataset contains 253 total recorded night sessions. We apply a strict pre-registered signal completeness threshold: a night is excluded if usable, artifact-free synchronized sensor data covers less than $60\%$ of total sleep recording duration. Exactly **7 nights across 6 participants** failed this criterion (due to sensor dislodgement, early battery depletion, or hardware interruption):
- `Bidslab08` Night 7 (19.3% coverage)
- `Bidslab16` Night 2 (24.0% coverage)
- `Bidslab34` Night 1 (58.7% coverage)
- `Bidslab42` Night 1 (30.1% coverage) and Night 3 (1.1% coverage)
- `Bidslab49` Night 1 (12.4% coverage)
- `Bidslab60` Night 1 (33.8% coverage)

The resulting signal-quality screened cohort comprises **47 participants and 246 usable nights** (average $5.23 \pm 1.48$ nights per participant).

### 3.3 Cohort Reconciliation: 47 Baseline Subjects vs. 46 Adaptation Subjects
A critical distinction in our experimental governance is the precise denominator governing different analysis phases:
- **Population Baseline Cohort ($N = 47$ subjects, 246 nights):** All 47 subjects possess at least two usable nights and are included in the subject-independent 5-fold cross-validation benchmarking (EXP-014, EXP-018).
- **Adaptation Analysis Cohort ($N = 46$ subjects, 198 nights evaluated):** Under our pre-registered adaptation protocol (EXP-019), a participant is eligible for supervised fine-tuning evaluation under $k$ calibration nights if and only if they possess **at least $k+1$ usable nights** ($n_{\text{usable}} \ge k + 1$). This requirement ensures that reserving the first $k$ chronological nights for personal calibration leaves at least one independent, unobserved subsequent night for evaluation ($n_{\text{eval}} \ge 1$). Across calibration depths $k \in \{1, 2, 3, 4\}$, the evaluable cohort sizes are:
  - $k=1$: $N=47$ evaluable participants (199 evaluation nights; 198 under the 46-subject EXP-019 cohort that excluded `Bidslab49` a priori).
  - $k=2$: $N=46$ evaluable participants (152 evaluation nights; exactly one participant, **`Bidslab49`**, had only two usable nights, leaving zero evaluation nights, $n_{\text{eval}} = 0$, and was excluded a priori from the adaptation cohort).
  - $k=3$: $N=42$ evaluable participants (106 evaluation nights).
  - $k=4$: $N=33$ evaluable participants (64 evaluation nights).

*(Evaluation nights represent usable nights minus calibration nights per subject; night inventories and cohort manifests are derived from `review_package/night_inventory.csv` and `experiments/EXP-019/exp019_k_learning_curve.csv`.)*

### 3.4 Classification Granularity and Class Imbalance
In alignment with standard clinical sleep staging manuals and wearable benchmark literature, we formulate sleep staging as a **4-class sequential classification task**:
- **Class 0: Wakefulness ($W$)** — 16,433 epochs (11.9% of cohort)
- **Class 1: Light Sleep ($L$)** — Combining Non-REM Stage 1 (N1) and Stage 2 (N2); 63,668 epochs (46.2% of cohort)
- **Class 2: Deep Sleep ($D$)** — Non-REM Stage 3 (N3, slow-wave sleep); 25,685 epochs (18.6% of cohort)
- **Class 3: REM Sleep ($R$)** — Rapid Eye Movement sleep; 32,015 epochs (23.2% of cohort)

AASM Stage 4 (historical N4) is subsumed into Stage 3 per current guidelines. Unscored, indeterminate, or movement artifact epochs (encoded as class 5 in Dreem outputs) are masked out during loss calculation and evaluation (mapped to label `-100`).

---

## 4. Methods

### 4.1 Input Representation and Temporal Framing
Rather than restricting sequences to arbitrary 5-hour cropped windows ($T = 600$), our model processes **full-night nocturnal sequences** with an effective context window of $L = 1200$ thirty-second epochs (corresponding to 10.0 hours). Shorter recordings are zero-padded; longer recordings are evaluated via sliding-window inference with 50% overlap and logit averaging.

For each 30-second epoch $t$, we construct a multi-channel feature vector derived from synchronized IHR and accelerometry:
1. **Cardiac Autonomic Features:** Instantaneous heart rate resampled to 1 Hz, epoch mean heart rate, epoch linear slope, standard deviation, and inter-beat variability proxy.
2. **Biomechanical Motion Features:** Tri-axial accelerometer signals ($x, y, z$), composite vector magnitude ($m_t = \sqrt{x_t^2 + y_t^2 + z_t^2}$), movement jerk ($\Delta m_t$), signal energy, and epoch immobility density.
3. **Per-Night Normalization:** To prevent day-to-day baseline drift from distorting feature representations, each nocturnal stream is standardized using robust statistics:
   $$\tilde{x}_t = \frac{x_t - \text{median}(x_{1:T})}{\text{IQR}(x_{1:T}) + \epsilon}$$
   where the median and interquartile range (IQR) are computed across the unpadded nocturnal sequence.

### 4.2 Benchmark Model Architecture: Model A (ConventionalCNNBiGRU)
To ensure computational efficiency and avoid overparameterized overfitting on small personal calibration sets, our primary backbone is **Model A (`ConventionalCNNBiGRU`)**. Model A is an audited, lightweight deep neural sequence architecture comprising exactly **163,940 trainable parameters**. This design achieves a **33.3-fold parameter compression** ($5,465,768 / 163,940 = 33.34\times$) relative to the 5.47-million parameter official SLAMSS-IFS author repository codebase, and a **4.06-fold parameter compression** ($666,060 / 163,940 = 4.06\times$) relative to the architecture specification in the published paper text.

The architecture consists of two tightly coupled stages:
1. **Epoch-Level Feature Encoder:** A 1D convolutional neural network operating across the 10-channel input streams. The encoder incorporates three sequential convolutional blocks with kernel sizes $\{3, 5, 5\}$, feature channels $\{32, 64, 128\}$, batch normalization, ReLU non-linearities, and spatial dropout ($p = 0.20$), compressing the raw 30-second epoch into a 128-dimensional latent vector $z_t$.
2. **Sequence-Level Temporal Modeling:** A 2-layer Bidirectional Gated Recurrent Unit (BiGRU) with hidden dimension $h = 64$ (128-dimensional bidirectional output), which captures long-range temporal sleep stage transition dynamics across the entire night.
3. **Classification Head:** A shared linear projection layer mapping bidirectional recurrent representations to 4-class unnormalized logits:
   $$\hat{y}_t = \mathbf{W}_c h_t + b_c, \quad \hat{y}_t \in \mathbb{R}^4$$

### 4.3 Personalization Arms and Experimental Controls
Under our pre-registered adaptation protocol (EXP-019), personalization is evaluated on held-out test subjects using the following strictly delineated experimental arms:

```
                            [Zero-Shot Population Model (ZS)]
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
   [Supervised FT]                  [Negative Control]              [Unsupervised TTA]
   Fine-tune on Subject's           Fine-tune on Donor              Self-entropy loss on
   Nights 1–2 (FT-2)                Nights 1–2 (FT-other-2)         Unlabeled Nights (UA-2)
         │                                 │                                 │
         └─────────────────────────────────┼─────────────────────────────────┘
                                           ▼
                                [Target Subject Nights 3+]
                                 Evaluate Macro-F1 vs. ZS
```

- **Arm 1: Zero-Shot Baseline (ZS).** The frozen population model trained across the training folds, evaluated directly on all evaluation nights ($k \ge 3$) of the held-out subject without any parameter updates.
- **Arm 2: Supervised Personal Fine-Tuning (FT-2).** The population model is fine-tuned on the held-out subject's first $k = 2$ chronological calibration nights using supervised cross-entropy loss:
  $$\mathcal{L}_{\text{CE}}(\theta) = -\frac{1}{M} \sum_{t=1}^M \sum_{c=0}^3 y_{t,c} \log \sigma(\hat{y}_{t,c}(\theta))$$
  Fine-tuning is conducted for a fixed budget of 15 epochs using AdamW ($\eta = 10^{-4}$, weight decay $10^{-2}$, batch size 4).
- **Arm 3: Specificity Control (FT-other-2).** To isolate whether performance changes arise from true individual personalization versus general gradient fine-tuning or learning rate decay, the model is fine-tuned on two calibration nights randomly sampled from a *different, unrelated donor subject*.
- **Arm 4: From-Scratch Personal Baseline (PS-2).** Model A is initialized from random weights and trained exclusively on the target subject's two calibration nights, testing whether a small personal sample can support de novo representation learning without population pre-training.
- **Arm 5: Unsupervised Adaptation (UA-2).** The model adapts to the target subject's calibration nights without labels by minimizing Shannon prediction entropy:
  $$\mathcal{L}_{\text{Ent}}(\theta) = -\frac{1}{M} \sum_{t=1}^M \sum_{c=0}^3 \sigma(\hat{y}_{t,c}) \log \sigma(\hat{y}_{t,c})$$

### 4.4 Diagnostic and Regularized Adaptation Formulations (EXP-020 Draft)
To combat catastrophic negative transfer identified during forensic error analysis, we formulate two exploratory adaptation variants:

1. **Kullback-Leibler Regularized Fine-Tuning (FT-2-KL):** To prevent personal fine-tuning from distorting population representations, we penalize the divergence between personalized output probabilities $\hat{p}_\theta(y_t)$ and frozen zero-shot probabilities $\hat{p}_{\text{ZS}}(y_t)$:
   $$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{CE}}(\theta) + \lambda_{\text{KL}} \cdot \mathcal{D}_{\text{KL}}\left(\hat{p}_{\text{ZS}} \,||\, \hat{p}_\theta\right)$$
   where $\lambda_{\text{KL}} \in \{0.1, 1.0, 10.0\}$ is selected strictly on inner-validation subjects.
2. **Quality-Gated Adaptation ($\text{GATE}_\tau$):** A deterministic safeguard evaluated as a candidate mechanism to mitigate severe individual collapse (such as observed in outlier participant `Bidslab34`, where calibration-night Deep sleep scarcity reached 0.68%, representing exactly 15 Deep-sleep epochs out of 2,194 labeled epochs across Nights 2 and 3). Gating inspects the empirical sleep stage distribution across calibration nights: if any sleep stage $c \in \{W, L, D, R\}$ comprises less than a critical prevalence threshold $\tau$ (e.g., $\tau \in \{1\%, 2\%, 3\%, 5\%\}$), the system detects calibration insufficiency, abstains from adaptation, and reverts automatically to the frozen zero-shot model ($ZS$):
   $$\theta_{\text{adapted}} = \begin{cases} \theta_{\text{FT-2}}, & \text{if } \min_{c} P_{\text{cal}}(Y = c) \ge \tau \\ \theta_{\text{ZS}}, & \text{otherwise} \end{cases}$$
   Because cohort-wide pre-adaptation stage scarcity does not show a statistically significant monotonic correlation with personalization delta ($q = 0.9711$), this formulation is evaluated strictly as an exploratory, design-informed heuristic.

---

## 5. Evaluation Protocol and Pre-Registration Statement

### 5.1 Strict Subject-Disjoint Partitioning and Splitting Rules
To eliminate data leakage, we enforce strict partitioning principles:
- **Zero Subject-Level Overlap:** Splitting is performed strictly at the subject level. No subject appearing in the test partition contributes any nocturnal epoch to the training or validation sets.
- **Chronological Calibration Ordering:** Within each held-out test subject, calibration nights are partitioned strictly chronologically: earlier nights ($k = 1, 2$) serve exclusively as calibration data, and all subsequent nights ($k \ge 3$) serve as the unobserved evaluation test set. Shuffling or random night selection within a subject is strictly prohibited.
- **Formal Disclosure on Historical Test Partitions:** We explicitly disclose that the 11-subject holdout partition (`Cohort Beta`, evaluated in EXP-004 and EXP-012) is considered scientifically spent. In EXP-012, our lightweight model evaluated on Cohort Beta achieved an empirical 3-seed mean accuracy of **$68.44\% \pm 1.35\%$** (Seed 42: 69.86%, Seed 123: 68.27%, Seed 2024: 67.18%; Macro-F1 $0.6859 \pm 0.0139$; DEC-018), which is higher than cohort-wide cross-validation results, indicating that single-split holdouts exhibit considerable variance. To maintain uncompromised rigor, all final benchmark claims in this study rest on **5-fold outer cross-validation across all 47 subjects**.

### 5.2 Primary Metrics and Statistical Inference
Because sleep stage classes are inherently imbalanced (Light sleep comprises $46.2\%$ of epochs, while Wake comprises $11.9\%$), overall accuracy and weighted F1 are inadequate primary metrics. 
- **Primary Metric:** **Unweighted Macro-F1**, defined as the unweighted arithmetic mean of stage-specific F1-scores:
  $$\text{Macro-F1} = \frac{1}{4} \left( F1_{\text{Wake}} + F1_{\text{Light}} + F1_{\text{Deep}} + F1_{\text{REM}} \right)$$
- **Secondary Metrics:** Cohen's kappa ($\kappa$), Matthews Correlation Coefficient (MCC), per-stage sensitivity and specificity, overall accuracy, and confusion matrices.
- **Unit of Statistical Analysis:** The individual participant ($N = 46$ for adaptation; $N = 47$ for baseline cross-validation). All metrics are computed per subject across their evaluation nights before calculating cohort-level statistics.
- **Confidence Intervals and Significance Testing:** We construct two-sided $95\%$ confidence intervals using a **10,000-resample non-parametric subject-level percentile bootstrap**. Null hypothesis testing for paired differences is performed via the two-sided Wilcoxon signed-rank test. Where multiple secondary comparisons are conducted (S1 to S3), family-wise error rate is controlled via the Holm-Bonferroni step-down procedure.

### 5.3 Clinical Sleep-Summary Agreement
In addition to epoch-by-epoch classification metrics, we evaluate clinical sleep architecture parameters derived from model hypnograms against EEG expert labels:
- **Total Sleep Time (TST):** Total minutes scored as Non-REM (L, D) or REM (R).
- **Wake After Sleep Onset (WASO):** Minutes of wakefulness occurring between sleep onset and final awakening.
- **Sleep Efficiency (SE):** Ratio of TST to total time in bed (TIB), expressed as a percentage.
- **Stage Distribution Proportions:** Percentage of TST spent in Deep sleep ($\text{Deep}\%$) and REM sleep ($\text{REM}\%$).

Agreement is evaluated using Bland-Altman 95% limits of agreement (mean bias $\pm 1.96 \cdot \text{SD}$), mean absolute error (MAE), and Pearson correlation coefficients.

### 5.4 Formal Pre-Registration and Governance Statement
This study was executed under strict computational research governance designed to prevent p-hacking, HARKing (Hypothesizing After Results are Known), and selective reporting:
1. **Confirmatory Adaptation Pre-Registration (EXP-019):** The primary hypothesis H-A, decision rules, statistical thresholds ($\Delta \ge +1.5\text{ pp}$ Macro-F1 with lower CI $> 0$), and evaluation manifests were formally committed in repository commit `4dff5b7` on November 13, 2026, prior to model training.
2. **Prior-Access Disclosure for Exploratory Studies (EXP-019B and EXP-020):** In accordance with open science standards, we explicitly disclose that exploratory protocols for heterogeneity analysis (EXP-019B, SHA-256: `30ca80f1f5191a9f76484d3d71256ab12c9f08f5404ee48009202666cd06cd77`) and regularized adaptation (EXP-020, SHA-256: `c226cc41873fbe93e61bfa23aa8ff8a407dd55e237fd82eef2a37562ee9da039`) were drafted post-hoc following inspection of EXP-019 results. They are designated as *design-informed exploratory investigations* and must not be conflated with blind prospective confirmatory trials.
3. **Open Reproducibility:** All subject manifests, model configurations, training code, and evaluation scripts are publicly archived at [repository URL pending publication].
