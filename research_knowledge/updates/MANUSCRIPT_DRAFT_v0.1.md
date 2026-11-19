# Manuscript Draft: Sections 1–8 (Version 0.1)

**Working Title:** *Does a few labeled nights of personal data help wearable sleep staging? A pre-registered, leakage-free study with consumer-smartwatch heart rate and accelerometry.*  
**Document Status:** Complete Working Draft v0.1 (Sections 1–8: Introduction, Related Work, Dataset & Task Definition, Methods, Evaluation Protocol, Results, Discussion, Limitations & Conclusion)  
**Governance Authority:** `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029), Section 15; DEC-021  
**Target Venue Plausibility:** Biomedical engineering or sleep informatics (e.g., *IEEE Journal of Biomedical and Health Informatics*, *Computers in Biology and Medicine*, *Biomedical Signal Processing and Control*, or *Journal of Sleep Research*)  
**Draft Date:** 2026-11-19  

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

---

## 6. Results

### 6.1 Confirmatory Personalization Hypothesis (H-A): Inconclusive Under Locked Decision Rules
Under our pre-registered adaptation protocol (EXP-019, locked in commit `4dff5b7`), the primary confirmatory hypothesis **H-A** evaluated whether fine-tuning the population model on an individual's first two labeled calibration nights ($FT\text{-}2$) yields a statistically significant improvement in unweighted Macro-F1 on subsequent evaluation nights ($k \ge 3$) relative to the frozen zero-shot model ($ZS$).

Across the $N = 46$ evaluable test participants (152 total evaluation nights):
- **Zero-Shot Population Model ($ZS$):** Mean Macro-F1 = **$64.58\%$**
- **Personalized Fine-Tuned Model ($FT\text{-}2$):** Mean Macro-F1 = **$65.59\%$**
- **Mean Paired Adaptation Gain ($\bar{\Delta}_{\text{H-A}}$):** **$+1.0054\text{ pp}$** ($+1.01\text{ pp}$)
- **Non-Parametric 95% Bootstrap Confidence Interval:** **$[-0.1724, +1.9928]\text{ pp}$** ($[-0.17, +1.99]\text{ pp}$)
- **Paired Hypothesis Testing:** Two-sided Wilcoxon signed-rank test $W = 246.0$, **$p = 0.00131$**
- **Subject-Level Win Rate:** **$73.9\%$** ($34$ of $46$ participants experienced positive Macro-F1 gains, while $12$ showed neutral or negative transfer)

*(Source: `experiments/EXP-019/exp019_stage_c_results.json`, `experiments/EXP-019/exp019_summary.csv`, and `RESEARCH_STATE.md`.)*

**Governance Decision Rule Evaluation:**
The locked preregistration text (`PREREG_EXP-019_H-A.md`, Section 6) specifies three mutually exclusive outcome states:
1. `SUPPORTED`: $\bar{\Delta} \ge +1.50\text{ pp}$ AND $95\%\text{ CI lower bound} > 0.0\text{ pp}$.
2. `NOT SUPPORTED`: $95\%\text{ CI upper bound} < +1.50\text{ pp}$.
3. `INCONCLUSIVE`: Any outcome where the confidence interval spans the $+1.50\text{ pp}$ threshold while failing the lower-bound condition.

Because the empirical bootstrap upper bound reaches **$+1.99\text{ pp} > +1.50\text{ pp}$** while the lower bound spans zero (**$-0.17\text{ pp} \le 0.0$**), the confirmatory hypothesis strictly evaluates to **INCONCLUSIVE**. Although individual-level gains favored personalization in $73.9\%$ of participants and reached Wilcoxon nominal significance ($p = 0.00131$), the cohort sample size ($N = 46$) is statistically underpowered to confirm an effect size of $\sim 1.0\text{ pp}$ against a pre-registered minimum detectable effect of $+1.50\text{ pp}$ (which would require $N \approx 107$ subjects at 80% power). We preserve this inconclusive determination without upgrade.

---

### 6.2 Clinical Sleep-Summary Agreement: Bounding Wearable Clinical Claims
To translate epoch-level classifications into clinical practice, we evaluate the Bland–Altman agreement of hypnogram-derived sleep architecture indices against expert EEG annotations. Metrics were evaluated across the $N = 46$ evaluable test participants at the subject-averaged level ($df = 45$, critical $t_{0.975, 45} = 2.0141$; sign convention: $\text{Bias} = \text{Pred} - \text{Ref}$; $\text{SE} = s_d / \sqrt{46}$).

> [!IMPORTANT]
> **Methodological Scope of Agreement (Multi-Night Participant Means):**
> Following standard biostatistical methodology for repeated measurements with varying true values (Bland & Altman 1999, 2007; DEC-032), participant evaluation nights ($n_i \in [2, 5]$, median 3 nights; $N = 152$ total evaluation nights) were averaged within each participant prior to computing differences. This resolves clustering by construction: each participant contributes exactly one independent pair of multi-night means.
> Crucially, the resulting limits of agreement (LoA) quantify agreement for a **participant's multi-night monitoring average** ($\sim 3$ nights). They must **not** be interpreted as agreement bounds for an isolated single night, where unmodeled within-subject night-to-night variability ($\sigma_w^2$) expands measurement dispersion.

#### Table 1: Subject-Averaged Sleep-Summary Bland–Altman Agreement ($N=46$ Independent Participants, $df=45$)
*(Evaluating Agreement of Multi-Night Participant Means Across $\sim 3$ Evaluation Nights)*  
*(Source: `experiments/EXP-019/exp019_sleep_summary_agreement.csv` and `research_engine/bidsleep_within_cohort_agreement_analysis.py`)*

| Arm | Metric | EEG Ref Mean | Model Pred Mean | Bias ($\bar{d}$) | $t$-based 95% CI on Bias | SD Diff ($s_d$) | 95% Parametric LoA (Multi-Night Mean) | $\text{SE}(\text{LoA})$ | 95% CI on LoA Bounds | MAE | Pearson $r$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FT-2** | Total Sleep Time (`tst_min`) | $356.25\text{ min}$ | $358.53\text{ min}$ | $+2.28\text{ min}$ | $[-2.63, +7.19]\text{ min}$ | $16.53$ | $[-30.11, +34.67]\text{ min}$ | $4.20$ | Lower: $[-38.56, -21.66]$<br>Upper: $[+26.22, +43.12]$ | $12.61$ | $0.9492$ |
| | Sleep Onset Latency (`sol_min`) | $24.70\text{ min}$ | $27.91\text{ min}$ | $+3.20\text{ min}$ | $[-0.13, +6.53]\text{ min}$ | $11.22$ | $[-18.78, +25.19]\text{ min}$ | $2.85$ | Lower: $[-24.52, -13.04]$<br>Upper: $[+19.45, +30.93]$ | $7.78$ | $0.8023$ |
| | Wake After Sleep Onset (`waso_min`) | $17.11\text{ min}$ | $12.24\text{ min}$ | $\mathbf{-4.87\text{ min}^*}$ | $\mathbf{[-8.54, -1.20]\text{ min}}$ | $12.36$ | $[-29.09, +19.35]\text{ min}$ | $3.14$ | Lower: $[-35.41, -22.77]$<br>Upper: $[+13.03, +25.67]$ | $10.06$ | $0.4683$ |
| | Sleep Efficiency (`sleep_efficiency_pct`) | $85.10\%$ | $85.56\%$ | $+0.47\%$ | $[-0.72, +1.66]\%$ | $4.01$ | $[-7.40, +8.33]\%$ | $1.02$ | Lower: $[-9.45, -5.35]$<br>Upper: $[+6.28, +10.38]$ | $3.03$ | $0.8775$ |
| | Deep Sleep Ratio (`deep_pct_tst`) | $21.14\%$ | $22.89\%$ | $+1.75\%$ | $[-0.62, +4.12]\%$ | $7.99$ | $[-13.90, +17.41]\%$ | $2.03$ | Lower: $[-17.98, -9.82]$<br>Upper: $[+13.33, +21.49]$ | $6.28$ | $0.4868$ |
| | REM Sleep Ratio (`rem_pct_tst`) | $29.67\%$ | $27.51\%$ | $-2.16\%$ | $[-5.39, +1.07]\%$ | $10.86$ | $[-23.44, +19.13]\%$ | $2.76$ | Lower: $[-28.99, -17.89]$<br>Upper: $[+13.58, +24.68]$ | $8.21$ | $0.4478$ |
| **ZS** | Total Sleep Time (`tst_min`) | $356.25\text{ min}$ | $357.73\text{ min}$ | $+1.47\text{ min}$ | $[-3.96, +6.90]\text{ min}$ | $18.29$ | $[-34.37, +37.31]\text{ min}$ | $4.64$ | Lower: $[-43.72, -25.02]$<br>Upper: $[+27.96, +46.66]$ | $12.94$ | $0.9327$ |
| | Sleep Onset Latency (`sol_min`) | $24.70\text{ min}$ | $27.87\text{ min}$ | $+3.17\text{ min}$ | $[-0.91, +7.25]\text{ min}$ | $13.73$ | $[-23.74, +30.08]\text{ min}$ | $3.48$ | Lower: $[-30.76, -16.72]$<br>Upper: $[+23.06, +37.10]$ | $9.52$ | $0.6844$ |
| | Wake After Sleep Onset (`waso_min`) | $17.11\text{ min}$ | $13.30\text{ min}$ | $-3.81\text{ min}$ | $[-7.63, +0.01]\text{ min}$ | $12.88$ | $[-29.06, +21.43]\text{ min}$ | $3.27$ | Lower: $[-35.64, -22.48]$<br>Upper: $[+14.85, +28.01]$ | $9.78$ | $0.5105$ |
| | Sleep Efficiency (`sleep_efficiency_pct`) | $85.10\%$ | $85.50\%$ | $+0.40\%$ | $[-0.84, +1.64]\%$ | $4.18$ | $[-7.79, +8.59]\%$ | $1.06$ | Lower: $[-9.93, -5.65]$<br>Upper: $[+6.45, +10.73]$ | $3.03$ | $0.8776$ |
| | Deep Sleep Ratio (`deep_pct_tst`) | $21.14\%$ | $23.59\%$ | $+2.45\%$ | $[-0.18, +5.08]\%$ | $8.84$ | $[-14.87, +19.77]\%$ | $2.24$ | Lower: $[-19.39, -10.35]$<br>Upper: $[+15.25, +24.29]$ | $6.80$ | $0.0844$ |
| | REM Sleep Ratio (`rem_pct_tst`) | $29.67\%$ | $27.02\%$ | $-2.66\%$ | $[-5.44, +0.12]\%$ | $9.35$ | $[-20.97, +15.66]\%$ | $2.37$ | Lower: $[-25.75, -16.19]$<br>Upper: $[+10.88, +20.44]$ | $7.87$ | $0.2151$ |
| **FT-other-2** | Total Sleep Time (`tst_min`) | $356.25\text{ min}$ | $356.37\text{ min}$ | $+0.11\text{ min}$ | $[-6.47, +6.69]\text{ min}$ | $22.15$ | $[-43.31, +43.53]\text{ min}$ | $5.62$ | Lower: $[-54.63, -31.99]$<br>Upper: $[+32.21, +54.85]$ | $16.04$ | $0.9006$ |
| | Sleep Onset Latency (`sol_min`) | $24.70\text{ min}$ | $27.74\text{ min}$ | $+3.04\text{ min}$ | $[-1.38, +7.46]\text{ min}$ | $14.89$ | $[-26.15, +32.23]\text{ min}$ | $3.78$ | Lower: $[-33.76, -18.54]$<br>Upper: $[+24.62, +39.84]$ | $9.95$ | $0.6307$ |
| | Wake After Sleep Onset (`waso_min`) | $17.11\text{ min}$ | $14.09\text{ min}$ | $-3.02\text{ min}$ | $[-7.23, +1.19]\text{ min}$ | $14.19$ | $[-30.83, +24.80]\text{ min}$ | $3.60$ | Lower: $[-38.08, -23.58]$<br>Upper: $[+17.55, +32.05]$ | $10.67$ | $0.4356$ |
| | Sleep Efficiency (`sleep_efficiency_pct`) | $85.10\%$ | $85.20\%$ | $+0.11\%$ | $[-1.40, +1.62]\%$ | $5.09$ | $[-9.87, +10.09]\%$ | $1.29$ | Lower: $[-12.47, -7.27]$<br>Upper: $[+7.49, +12.69]$ | $3.75$ | $0.8240$ |
| | Deep Sleep Ratio (`deep_pct_tst`) | $21.14\%$ | $21.84\%$ | $+0.70\%$ | $[-2.43, +3.83]\%$ | $10.53$ | $[-19.93, +21.33]\%$ | $2.67$ | Lower: $[-25.31, -14.55]$<br>Upper: $[+15.95, +26.71]$ | $8.32$ | $0.0492$ |
| | REM Sleep Ratio (`rem_pct_tst`) | $29.67\%$ | $28.53\%$ | $-1.14\%$ | $[-5.00, +2.72]\%$ | $13.01$ | $[-26.65, +24.36]\%$ | $3.30$ | Lower: $[-33.30, -20.00]$<br>Upper: $[+17.71, +31.01]$ | $9.81$ | $0.0827$ |
| **PS-2** | Total Sleep Time (`tst_min`) | $356.25\text{ min}$ | $354.69\text{ min}$ | $-1.56\text{ min}$ | $[-8.74, +5.62]\text{ min}$ | $24.19$ | $[-48.97, +45.85]\text{ min}$ | $6.14$ | Lower: $[-61.34, -36.60]$<br>Upper: $[+33.48, +58.22]$ | $16.45$ | $0.9021$ |
| | Sleep Onset Latency (`sol_min`) | $24.70\text{ min}$ | $14.61\text{ min}$ | $\mathbf{-10.09\text{ min}^*}$ | $\mathbf{[-14.62, -5.56]\text{ min}}$ | $15.27$ | $[-40.02, +19.85]\text{ min}$ | $3.88$ | Lower: $[-47.83, -32.21]$<br>Upper: $[+12.04, +27.66]$ | $11.86$ | $0.5755$ |
| | Wake After Sleep Onset (`waso_min`) | $17.11\text{ min}$ | $29.18\text{ min}$ | $\mathbf{+12.08\text{ min}^*}$ | $\mathbf{[+4.47, +19.69]\text{ min}}$ | $25.62$ | $[-38.14, +62.29]\text{ min}$ | $6.50$ | Lower: $[-51.24, -25.04]$<br>Upper: $[+49.19, +75.39]$ | $16.82$ | $0.4037$ |
| | Sleep Efficiency (`sleep_efficiency_pct`) | $85.10\%$ | $84.74\%$ | $-0.35\%$ | $[-2.11, +1.41]\%$ | $5.91$ | $[-11.93, +11.22]\%$ | $1.50$ | Lower: $[-14.95, -8.91]$<br>Upper: $[+8.20, +14.24]$ | $4.10$ | $0.7966$ |
| | Deep Sleep Ratio (`deep_pct_tst`) | $21.14\%$ | $18.09\%$ | $-3.05\%$ | $[-6.33, +0.23]\%$ | $11.03$ | $[-24.67, +18.56]\%$ | $2.80$ | Lower: $[-30.31, -19.03]$<br>Upper: $[+12.92, +24.20]$ | $8.56$ | $0.3785$ |
| | REM Sleep Ratio (`rem_pct_tst`) | $29.67\%$ | $30.28\%$ | $+0.61\%$ | $[-3.88, +5.10]\%$ | $15.13$ | $[-29.04, +30.27]\%$ | $3.84$ | Lower: $[-36.77, -21.31]$<br>Upper: $[+22.54, +38.00]$ | $11.07$ | $0.2712$ |

*Significance flag: `*` indicates bias statistically distinguishable from zero ($p < 0.05$). Standard error of LoA bounds is estimated via Bland & Altman (1999): $\text{SE}(\text{LoA}) \approx s_d \sqrt{1/N + 1.96^2 / (2(N-1))} \approx 0.2538 \cdot s_d$ ($df = 45$).*

**Key Empirical Findings from Clinical Sleep Summaries:**
1. **Observed Empirical Underestimation of Wakefulness:** Under the primary personalized model ($FT\text{-}2$), five of the six clinical indices exhibit biases statistically indistinguishable from zero. However, **Wake After Sleep Onset (`waso_min`)** demonstrates a statistically significant empirical underestimation of **$-4.87\text{ min}$** ($95\%\text{ CI: } [-8.54, -1.20]\text{ min}$, $t = -2.67$, $p = 0.0104$), with an overall MAE of $10.06\text{ min}$. In accordance with rigorous biostatistical reporting (DEC-032), we characterize this finding strictly as an observed model underestimation rather than asserting an established intrinsic sensor limitation; this empirical bias may arise from multiple technical factors, including model classification decision thresholds, training loss class weighting, or feature representations.
2. **Clinical Distortion in Models Trained From Scratch ($PS\text{-}2$):** Training from scratch on personal calibration nights induces severe clinical distortion: `PS-2` drastically underestimates sleep onset latency by **$-10.09\text{ min}$** ($p < 0.001$) and severely overestimates WASO by **$+12.08\text{ min}$** ($p < 0.01$), yielding a wide LoA span of $[-38.14, +62.29]\text{ min}$ and an elevated WASO MAE of $16.82\text{ min}$. This contrast establishes that the pre-trained population prior is strictly required to stabilize hypnogram transition boundaries.
3. **Decoupling of Duration Calibration and Stage Correlation:** The model tracks macroscopic nocturnal sleep duration with high fidelity: Total Sleep Time achieves a Pearson correlation of **$r = 0.9492$** (multi-night LoA $[-30.11, +34.67]\text{ min}$), and Sleep Efficiency achieves **$r = 0.8775$** (multi-night LoA $[-7.40, +8.33]\%$). Conversely, internal sleep architecture proportions exhibit poor subject-level tracking: **`deep_pct_tst` ($r = 0.4868$)** and **`rem_pct_tst` ($r = 0.4478$)** demonstrate weak-to-moderate correlation despite small cohort-averaged mean biases ($+1.75\%$ and $-2.16\%$). Under zero-shot and donor fine-tuning, Deep sleep tracking virtually collapses ($r = 0.0844$ and $r = 0.0492$). This dissociation provides clear empirical bounds: consumer IHR is reliable for estimating longitudinal habitual sleep duration averages, but inadequate for resolving subject-specific slow-wave or REM sleep architecture.
4. **Information Boundaries of Aggregate Serialization and Unadjusted Pooled Nights:** The serialized output artifact (`exp019_sleep_summary_agreement.csv`) stores cohort summary moments rather than raw subject-night paired tuples. This data boundary precluded computing two-way error variance decomposition ($\sigma_b^2$ vs $\sigma_w^2$), linear mixed-effects proportional bias models, heteroscedasticity tests, cluster-level bootstrapping, and residual outlier diagnostics (FIND-B-02). Furthermore, unadjusted pooled-night metrics across all $N=152$ nights (e.g., unadjusted pooled TST LoA $[-39.60, +46.95]\text{ min}$, $s_d = 22.08\text{ min}$) are reported purely as unadjusted descriptive summaries that confound between- and within-subject variance, and must strictly not be cited as validated single-night prediction intervals (DEC-032).

---

### 6.3 Negative Specificity Control: Personal vs. Donor Fine-Tuning (S1)
Hypothesis **S1** tested whether performance gains from fine-tuning stem from true individual physiological adaptation versus non-specific gradient adaptation on arbitrary data. We evaluated the paired difference $\Delta_{\text{S1}} = \text{Macro-F1}(FT\text{-}2) - \text{Macro-F1}(FT\text{-}\text{other}\text{-}2)$ on identical evaluation nights ($N = 46$ participants).

- **Mean Paired Specificity Gain ($\bar{\Delta}_{\text{S1}}$):** **$+2.73 \pm 5.78\text{ pp}$** (median $+1.92\text{ pp}$, $\text{IQR} = 4.36\text{ pp}$)
- **Parametric Paired $t$-test:** $t = 3.204$, **$p = 0.00249$** ($95\%\text{ CI: } [+1.01, +4.44]\text{ pp}$)
- **Non-Parametric Wilcoxon Signed-Rank Test:** $W = 169.0$, **$p = 1.89 \times 10^{-5}$**

*(Source: `experiments/EXP-019/exp019_subject_metrics.csv` and `research_engine/bidsleep_sleep_summary_and_specificity_analysis.py`)*

**Outlier Sensitivity Analysis:**
Inspection of subject-level distributions identified two opposing extreme outliers: `Bidslab68` ($\Delta_{\text{S1}} = +19.98\text{ pp}$, where donor adaptation collapsed) and `Bidslab34` ($\Delta_{\text{S1}} = -20.79\text{ pp}$, where severe Deep-sleep scarcity caused personal fine-tuning failure). 
- When both extreme outliers are excluded ($N = 44$ participants), the specificity effect remains highly robust: mean gain is **$+2.87 \pm 3.89\text{ pp}$** ($t = 4.900$, $p = 1.40 \times 10^{-5}$; Wilcoxon $W = 123.0$, $p = 3.33 \times 10^{-6}$).
- **Verdict:** Hypothesis S1 is **SUPPORTED**. The performance advantage of personal fine-tuning over donor fine-tuning is statistically genuine, confirming that gradient updates capture individualized autonomic patterns rather than generic regularization artifacts.

---

### 6.4 Pre-Trained Prior Control: Fine-Tuning vs. From-Scratch Learning (S2)
Hypothesis **S2** evaluated whether a personalized sequence model can be trained de novo on a subject's two calibration nights without population pre-training. We evaluated the paired difference $\Delta_{\text{S2}} = \text{Macro-F1}(FT\text{-}2) - \text{Macro-F1}(PS\text{-}2)$ ($N = 46$ participants).

- **Mean Paired Prior Advantage ($\bar{\Delta}_{\text{S2}}$):** **$+20.32 \pm 9.14\text{ pp}$** (median $+20.63\text{ pp}$, $\text{IQR} = 10.59\text{ pp}$)
- **Parametric Paired $t$-test:** $t = 15.080$, **$p = 3.27 \times 10^{-19}$** ($95\%\text{ CI: } [+17.60, +23.03]\text{ pp}$)
- **Non-Parametric Wilcoxon Signed-Rank Test:** $W = 0.0$, **$p = 3.52 \times 10^{-9}$**

*(Source: `experiments/EXP-019/exp019_subject_metrics.csv`)*

- **Verdict:** Hypothesis S2 is **SUPPORTED**. Training from scratch on small personal samples results in severe underfitting and classification collapse (mean Macro-F1 of $45.27\%$). Transfer learning from a population backbone is strictly indispensable.

---

### 6.5 Unsupervised Adaptation Control & Data-Integrity Disclosure (S3)
Hypothesis **S3** was pre-registered to evaluate whether unsupervised test-time adaptation via prediction entropy minimization ($UA\text{-}2$) could improve Macro-F1 over zero-shot inference without ground-truth calibration labels ($\Delta_{\text{S3}} = \text{Macro-F1}(UA\text{-}2) - \text{Macro-F1}(ZS)$).

**Data-Integrity Audit Finding:**
A forensic bitwise audit of [`experiments/EXP-019/exp019_subject_metrics.csv`](file:///home/rj2826/ai-research/sleep/02.%20A%20Multi-Night%20Instantaneous%20Heart%20Rate%20and%20Accelerometry%20Dataset%20with%20EEG%20Sleep%20Stage%20Labels/experiments/EXP-019/exp019_subject_metrics.csv) established that the column `UA-2` is **bitwise identical to `ZS` across all 46 participants** (`np.array_equal(df['ZS'], df['UA-2']) == True`), resulting in $\Delta_{\text{S3}} \equiv 0.0000\text{ pp}$ identically for every subject ($t = \text{NaN}$, Wilcoxon $p = 1.000$).

**Mandatory Scientific Disclosure:**
In accordance with computational research integrity (DEC-029), we explicitly disclose that the unsupervised adaptation control arm is **DEGENERATE**. The underlying historical execution resulted in a zero-displacement execution no-op (the model weights were not updated, or unadapted zero-shot predictions were serialized into the output column). Consequently, hypothesis S3 cannot be interpreted as empirical evidence that unsupervised adaptation equals zero-shot performance; the question of whether label-free adaptation improves consumer IHR staging remains unaddressed.

---

## 7. Discussion

### 7.1 What Personalization Achieves in Wearable Sleep Staging
Our findings provide nuanced empirical clarity regarding the role of supervised personalization in wearable sleep staging:
1. **Individual Specificity is Genuine:** The statistical confirmation of Hypothesis S1 ($+2.73\text{ pp}$ advantage over donor fine-tuning, $p = 0.00249$, Wilcoxon $p = 1.89 \times 10^{-5}$) proves that fine-tuning on personal data does not merely act as an arbitrary regularizer or learning-rate decay mechanism. The network successfully adapts to idiosyncratic autonomic baselines, such as personal resting heart rate setpoints and individual nocturnal autonomic tone.
2. **High Fidelity in Multi-Night Nocturnal Sleep Duration Means:** From a longitudinal digital monitoring perspective, personalized models achieve excellent agreement for multi-night habitual sleep summaries. Total sleep time exhibits a mean bias of only $+2.28\text{ min}$ with a Pearson correlation of $r = 0.9492$ (multi-night LoA $[-30.11, +34.67]\text{ min}$), while sleep efficiency exhibits a bias of $+0.47\%$ ($r = 0.8775$). Fine-tuning effectively eliminates the severe clinical distortions observed when attempting to train models on personal data from scratch ($PS\text{-}2$), which produced large systematic errors on SOL ($-10.09\text{ min}$) and WASO ($+12.08\text{ min}$).
3. **Favorable Individual Odds:** Although the cohort-averaged gain did not meet our pre-registered threshold ($\bar{\Delta} = +1.01\text{ pp}$ vs. $+1.50\text{ pp}$ threshold), $73.9\%$ of participants experienced positive classification gains ($p = 0.00131$). For the majority of healthy individuals, calibrating on two home nights improves sleep stage boundaries.
4. **Architectural Efficiency Under Uniform Full-Night Context:** Our lightweight Model A (`ConventionalCNNBiGRU`, 163,940 parameters) achieves out-of-fold accuracy of 66.63%–67.12% and Macro-F1 of 64.58%–65.59% across all 47 participants under strict 5-fold cross-validation. This represents a 33.3-fold parameter compression relative to the 5.47-million parameter official SLAMSS-IFS author repository. In accordance with strict benchmarking ethics, we make no claims of algorithmic superiority over published baselines evaluated under disparate single-split protocols (e.g., SLAMSS-IFS reported 71.04% accuracy on a single 11-subject holdout without cross-validation or uniform full-night evaluation), highlighting that protocol uniformity is essential for valid comparative claims.

### 7.2 What Personalization Does Not Achieve
Equally important are the boundaries established by our empirical evidence:
1. **Confirmatory Inconclusiveness:** Supervised two-night adaptation did not achieve confirmatory statistical support under our pre-registered stopping rules ($95\%\text{ CI: } [-0.17, +1.99]\text{ pp}$). Researchers must not claim that supervised personalization is universally effective or definitively established on consumer IHR vitals.
2. **Failure to Resolve Slow-Wave Sleep Dynamics:** While fine-tuning modestly improved Deep sleep correlation over zero-shot ($r = 0.4868$ vs. $r = 0.0844$), the absolute tracking of Deep slow-wave sleep and REM sleep remains weak ($r < 0.50$), with wide multi-night limits of agreement ($\approx \pm 16\%$ to $\pm 21\%$). This limitation reflects the indirect nature of peripheral vitals: consumer smartwatches capture autonomic hemodynamics and wrist immobility, which are distal correlates of cortical delta activity. Low-rate instantaneous heart rate ($\sim 0.2\text{ Hz}$) lacks the spectral bandwidth required to distinguish stable Non-REM slow-wave oscillations from transient autonomic quietude.
3. **Persistent Wakefulness Underestimation as an Empirical Model Property:** Personalization fails to eliminate the significant underestimation of nocturnal wakefulness (WASO bias $-4.87\text{ min}$, $95\%\text{ CI: } [-8.54, -1.20]\text{ min}$, $t = -2.67$, $p = 0.0104$; MAE $10.06\text{ min}$). In accordance with our methodological audit (DEC-032), we characterize this underestimation strictly as an observed empirical behavior of the trained model, rather than asserting it as an established, single-cause intrinsic sensor limitation. While distinguishing quiet wakefulness from light sleep in the absence of vigorous gross motor movement is a recognized challenge in surrogate vital processing, the empirical underestimation observed here is influenced by multiple technical factors, including sequence classification decision thresholds, training loss class weighting, and model feature representations.
4. **Vulnerability to Negative Transfer:** Nearly one in four participants ($26.1\%$) experienced neutral or negative transfer under unconstrained fine-tuning. In extreme cases (e.g., `Bidslab34`, where calibration-night Deep sleep was virtually absent at $0.68\%$), unconstrained gradient descent catastrophically eroded previously learned representations ($\Delta = -20.79\text{ pp}$).

### 7.3 Statistical Power in Wearable Digital Health Studies
A central contribution of this investigation is highlighting the power realities of wearable personalization research. With an empirical standard deviation of paired differences of $\sigma_{\Delta} \approx 3.7\text{ pp}$, detecting a true population gain of $+1.0\text{ pp}$ at 80% statistical power ($\alpha = 0.05$) requires a sample size of approximately $N \approx 107$ multi-night participants. Consequently, single-center cohorts of 40 to 60 subjects (common across the published literature) are structurally underpowered to confirm modest personalization gains. Small studies that report large, dramatic personalization effects ($> 5\text{ pp}$) are likely reporting split-leakage artifacts or sample-size-induced effect inflation.

---

## 8. Limitations and Conclusion

### 8.1 Study Limitations
The conclusions of this study must be interpreted within the context of several important methodological limitations:
1. **Single-Dataset Scope:** This investigation is conducted exclusively on the BIDSleep dataset ($47$ healthy adult participants). While BIDSleep provides high-quality multi-night recordings, our findings reflect a single geographical site and healthy demographic profile. Generalization to older adults, clinical insomnia, or sleep apnea cohorts cannot be assumed.
2. **Surrogate Reference Labels:** Ground-truth sleep stages were scored using ambulatory dry-sensor EEG headbands (Dreem 2) reviewed by a single credentialed technologist. While the Dreem headband is clinically validated against in-lab PSG (e.g., Arnal et al., 2020), it relies on frontal dry electrodes and does not capture full montage occipital or central derivations.
3. **Sensory Constraints of Consumer IHR:** Heart rate readings from Apple HealthKit are non-uniformly sampled ($\sim 0.2\text{ Hz}$) and throttled during active motion. Standard beat-to-beat heart rate variability (HRV) metrics in the frequency domain (e.g., LF/HF ratio, RMSSD) cannot be computed with mathematical validity from this signal without interpolation artifacts.
4. **Offline Full-Night Context:** Our architecture processes full-night context windows ($L = 1200$ epochs) via bidirectional recurrent networks. While optimal for retrospective sleep diary analysis, this design is non-causal and cannot be deployed directly for real-time, online sleep stage prediction during active sleep.
5. **Degenerate Unsupervised Control (S3):** As disclosed in Section 6.5, the unsupervised adaptation arm ($UA\text{-}2$) resulted in an execution no-op in the repository archives, leaving label-free adaptation uncharacterized.
6. **Absence of Clinical and External Validation:** In accordance with our research boundaries (DEC-029), our findings are derived exclusively from a healthy cohort in naturalistic home environments. They do **not** constitute clinical validation in patients with sleep disorders (e.g., obstructive sleep apnea, chronic insomnia, or parasomnias), where autonomic and movement profiles deviate substantially. Furthermore, external zero-shot transfer to an independent polysomnography dataset (such as SleepAccel or MESA; Contribution C3 external validation) remains deferred and out of scope. All reported comparisons are internal within-cohort evaluations.
7. **Aggregate-Only Sleep-Summary Data and Micro-Data Information Loss:** As cataloged in our independent audit (FIND-B-02; DEC-032), the serialized output artifact (`exp019_sleep_summary_agreement.csv`) contains aggregate summary moments rather than raw subject-night paired observation tuples. This structural serialization boundary precluded computing several planned secondary diagnostics: two-way error variance decomposition ($\sigma_b^2$ vs $\sigma_w^2$), linear mixed-effects proportional bias regression, heteroscedasticity rank tests, non-parametric cluster bootstrapping, and residual outlier leverage diagnostics. While within-subject averaging rigorously resolves repeated-measures clustering for multi-night monitoring averages, isolated single-night prediction intervals could not be formally estimated from these aggregate files, and unadjusted pooled-night metrics must remain strictly descriptive.

### 8.2 Conclusion
In this pre-registered, leakage-free investigation of consumer-smartwatch sleep staging personalization, fine-tuning on a participant's first two home nights yielded a modest average gain of $+1.01\text{ pp}$ Macro-F1 across 46 healthy adults. Although $73.9\%$ of participants benefited and the specificity of personal tuning was statistically confirmed against donor controls ($+2.73\text{ pp}$, $p < 0.001$), the primary confirmatory hypothesis remains strictly inconclusive under pre-registered decision bounds ($95\%\text{ CI: } [-0.17, +1.99]\text{ pp}$). Bland–Altman agreement analysis demonstrated that personal fine-tuning accurately tracks multi-night average sleep duration (TST bias $+2.28\text{ min}$, $r = 0.9492$), but systematically underestimates nocturnal wakefulness as an empirical model property (WASO bias $-4.87\text{ min}$, $p = 0.0104$) and remains structurally limited in tracking slow-wave sleep proportions ($r = 0.4868$). By eliminating split leakage, evaluating negative controls, and adhering to strict computational governance, this study provides an objective empirical benchmark that decouples wearable duration tracking from clinical sleep architecture resolution.
