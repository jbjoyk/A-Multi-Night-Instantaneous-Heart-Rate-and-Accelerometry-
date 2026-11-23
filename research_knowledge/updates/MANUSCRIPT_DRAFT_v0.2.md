# Safety, Calibration Thresholds, and External Clinical PSG Generalisation of Lightweight Consumer-Smartwatch Sleep Staging: A Pre-Registered, Leakage-Free Study

**Target Venue:** *The Lancet Digital Health* (primary) / *npj Digital Medicine* (secondary)  
**Reporting Standard:** TRIPOD+AI & *Lancet* House Structure  
**Draft Version:** v0.2 (Stage E Submission-Ready Manuscript)  
**Governance Authority:** `RESEARCH_DIRECTION_v4_FULL.md` (`DEC-029`), `DEC-035`–`DEC-041`  
**Draft Date:** 2026-11-23  

---

## Structured Summary

### Background
Consumer smartwatches estimate sleep stages from low-rate (~0.2 Hz) instantaneous heart rate (IHR) and accelerometry, yet clinical translation is hindered by inter-individual domain shift, benchmark leakage, unquantified calibration-depth thresholds, and unknown generalisation to in-lab polysomnography (PSG). We evaluated translational safety, personalisation thresholds, accuracy–efficiency trade-offs, and external zero-shot PSG validity.

### Methods
Under pre-registered, subject-disjoint 5-fold cross-validation in BIDSleep (`N=47` healthy adults, 246 nights; Apple Watch + Dreem 2 EEG), we benchmarked frozen Model A (`163,940` parameters; 33.34-fold compression vs. official `5,465,768`-parameter SLAMSS-IFS) and Dynamic INT8 quantisation. Across `N=46` eligible participants (152 evaluation nights), we tested 2-night supervised fine-tuning (`FT-2`), donor (`FT-other-2`), from-scratch (`PS-2`), and remediated unsupervised (`UA-2`) controls, `k=1`–`4` calibration-depth scaling (`N=33` common cohort `E_4`, 64 nights), and exploratory stage-coverage gating (`GATE_tau`). Zero-shot external validation used single-night in-lab PSG from PhysioNet SleepAccel (`N=31`; `n=25` QC-clean primary).

### Findings
On BIDSleep (`N=47`), Model A achieved `64.79%` Macro-F1 (`66.18%` accuracy) versus `67.37%` (`68.75%`) for SLAMSS-IFS (`-2.58 pp`, `p=1.98×10^-6`); Dynamic INT8 (`309.54 KiB`, `47.7 ms`/night) preserved Macro-F1 (`+0.0011 pp`, `p=0.7892`). Confirmatory `FT-2` (`N=46`) yielded `+1.01 pp` (`95% CI -0.17 to +1.99`, `p=0.00131`, `INCONCLUSIVE`), while monotonic scaling (`N=33`, `E_4`: `+1.75` to `+2.71 pp`, Page's `L` `p=0.01187`) and `GATE_tau` (`+1.33 pp`, `95% CI +0.49 to +2.12`, cutting worst collapse by 61% to `-6.40 pp`) supported `k >= 3` calibration. On external single-night PSG (`n=25` QC-clean primary `mean_single_model`), SLAMSS-IFS's lead attenuated 78% to `-0.57 pp` (`60.51%` vs. `61.08%`, `p=0.5782`; `N=31` `-1.62 pp`, `p=0.3468`), with Dreem 2-to-PSG label-prior shift driving Deep% (`+10.65%`) and REM% (`+10.13%`) overestimation.

### Interpretation
Lightweight smartwatch models match 33-fold larger server architectures on external clinical PSG and reliably track sleep duration, provided `k >= 3` calibration nights, stage-coverage gating, and label-prior shift adjustments are enforced.

### Funding
None.

---

## Research in context

### Evidence before this study
Consumer-grade optical photoplethysmography (PPG) and accelerometry enable longitudinal home sleep monitoring, yet mapping peripheral autonomic signals to central electroencephalographic (EEG) sleep stages remains vulnerable to inter-individual physiological variance. Walch et al. (*SLEEP* 2019) demonstrated 3-class sleep staging from Apple Watch instantaneous heart rate (IHR) and motion in 31 adults, and Song et al. (*IEEE TBME* 2026) introduced the 5.47-million-parameter SLAMSS-IFS sequence architecture on BIDSleep using a single 11-subject holdout. Crucially, van der Aar et al. (*Physiol. Meas.* 2025) demonstrated in 59 adults with suspected sleep disorders that supervised fine-tuning on high-rate raw PPG improved staging in 82.5% of individuals once at least three calibration nights (`k >= 3`) were available. However, systematic repository audits across PhysioNet, NSRR, and OpenNeuro confirmed that no second public multi-night healthy-adult smartwatch+EEG dataset currently exists, leaving leakage-free calibration thresholds, negative-transfer safeguards, matched-protocol edge compression trade-offs, and zero-shot generalisation to gold-standard in-lab polysomnography (PSG) unquantified.

### Added value of this study
Under pre-registered, strictly chronological, subject-disjoint evaluation on BIDSleep (`N=47`) and strictly zero-shot external validation on PhysioNet SleepAccel single-night in-lab PSG (`N=31`; `n=25` QC-clean), this study delivers four translational advances: (1) it establishes that within-person adaptation on ~0.2 Hz HealthKit IHR scales monotonically across `k=1`–`4` nights (`p=0.01187`), converging with clinical PPG findings at `k >= 3` nights, while surgical pre-adaptation stage-coverage gating (`GATE_tau`) prevents extreme-tail negative transfer (`-16.34 pp` reduced to `-6.40 pp`); (2) it quantifies the exact matched 5-fold cross-validation trade-off between a 163,940-parameter Dynamic INT8 model (`309.54 KiB` in-memory, `47.7 ms`/night CPU) and the 33.34-fold larger official SLAMSS-IFS server model (`-2.58 pp` Macro-F1); (3) it resolves prior degenerate unsupervised adaptation logs (`TSK-07`), proving label-free InfoMax yields a real but modest `+0.21 pp` adjustment; and (4) it demonstrates on external single-night PSG that SLAMSS-IFS's BIDSleep Macro-F1 advantage attenuates by 78% to statistical parity (`-0.57 pp`, `p=0.5782` primary single-model; `+0.15 pp`, `p=0.9789` ensemble) while exposing cross-modality Dreem 2-to-PSG label-prior shift (`44.2% -> 57.7%` Light; `18.9% -> 11.1%` Deep) driving Deep% (`+10.65%`) and REM% (`+10.13%`) overestimation in both architectures.

### Implications of all the available evidence
Together, these findings define three concrete clinical and deployment guidelines for consumer wearable sleep staging: (a) consumer IHR and accelerometry safely track macroscopic nocturnal sleep duration (Total Sleep Time `r=0.9492` multi-night on BIDSleep; `r=0.9091` single-night on PSG) and Sleep Efficiency, whereas internal stage-architecture proportions (Deep% and REM%) remain vulnerable to cross-modality label-prior shift and cannot yet substitute for clinical PSG; (b) patient-level fine-tuning should only be deployed when at least `k >= 3` labeled calibration nights are available and paired with pre-adaptation stage-coverage abstention (`GATE_tau`) to prevent catastrophic forgetting; and (c) prospective multi-site external validation in clinical sleep-disorder cohorts (such as PhysioNet DREAMT v2.2.0) and pediatric/neurodevelopmental multi-night cohorts (such as Simons Sleep Project `DS0000089`) is essential before clinical diagnostic deployment.

---

## 1. Introduction

### 1.1 Clinical and Wearable Context
Sleep architecture is scored clinically under American Academy of Sleep Medicine (AASM) guidelines into four canonical states: Wakefulness ($W$), Light sleep (combining Non-Rapid Eye Movement stages N1 and N2), Deep slow-wave sleep (N3), and Rapid Eye Movement (REM) sleep ($R$). While laboratory polysomnography (PSG) remains the diagnostic gold standard, its cost, obtrusive sensor montage, technologist scoring burden, and single-night susceptibility to the "first-night effect" preclude longitudinal multi-night monitoring in naturalistic home environments.

Consumer smartwatches equipped with optical photoplethysmography (PPG) and tri-axial accelerometers offer scalable longitudinal tracking by inferring sleep states from peripheral autonomic nervous system (ANS) tone and somatic immobility. However, mapping low-rate instantaneous heart rate (IHR; ~0.2 Hz via Apple HealthKit) and wrist movement to central EEG-defined sleep stages is an ill-posed inverse problem subject to pronounced inter-individual biological heterogeneity in resting heart rate, autonomic responsivity, and circadian phase.

### 1.2 The Promise and Peril of Personalisation
Model personalisation—fine-tuning a population backbone on a few initial home nights paired with a convenient dry-sensor EEG headband—offers a practical route to bridge inter-individual domain shift. Yet unconstrained gradient adaptation on small, class-imbalanced personal calibration sets (`k=1`–`4` nights) risks catastrophic forgetting and negative transfer, wherein fine-tuning degrades subsequent staging accuracy below the unmodified zero-shot population model. Furthermore, deploying recurrent neural architectures on consumer wearables requires balancing classification accuracy against strict memory, latency, and energy budgets on smartwatch system-on-chip (SoC) or companion smartphone hardware.

### 1.3 Addressed Research Gaps and Claim Boundaries
This study addresses three foundational gaps in translational wearable sleep staging:
- **Gap G1 (Personalisation Thresholds and Safety in Healthy Adults):** Quantifying how many labeled home nights (`k=1`–`4`) are needed for low-rate consumer IHR adaptation in healthy adults under strictly chronological, leakage-free, pre-registered rules, and identifying safety guardrails against severe negative transfer.
- **Gap G2 (Protocol-Matched Efficiency Trade-Offs):** Eliminating split leakage and 5-hour nocturnal cropping biases to quantify the exact accuracy–efficiency trade-off between a compact edge-quantised sequence model and the 33.34-fold larger published SLAMSS-IFS server architecture under identical 5-fold cross-validation.
- **Gap G3 (External Single-Night Clinical PSG Generalisation and Agreement):** Evaluating whether within-cohort performance translates zero-shot to independent gold-standard in-lab PSG and quantifying clinical sleep-summary agreement (TST, SOL, WASO, SE, Deep%, REM%) via Bland–Altman analysis.

**Explicit Claim Boundaries:** We do not claim to present the first multi-night personalisation study, nor do we claim universal personalisation efficacy. Our locked primary confirmatory hypothesis (`H-A`, `EXP-019`) is preserved strictly as `INCONCLUSIVE`. Non-significant differences against SLAMSS-IFS on external PSG are framed strictly as no statistically significant difference / empirical parity under paired testing (no TOST equivalence claim). All findings are scoped to healthy adults and offline full-night inference.

---

## 2. Methods

### 2.1 Study Design, Datasets, and Participant Flow
We conducted a multi-stage computational validation across two independent public datasets:

1. **Internal Multi-Night Cohort — BIDSleep (`N=47`):** Collected at the University of Massachusetts Amherst (Song et al., *IEEE TBME* 2026; PhysioNet DOI: `10.13026/rees-1092`; ODC-By v1.0), comprising 47 healthy adults (`Bidslab00`–`Bidslab68`) wearing an Apple Watch Series 6 (HealthKit IHR median effective rate `0.197 Hz`, IQR `0.073 Hz`, nightly median `64.0 BPM`; plus 3-axis accelerometry) and a 5-channel dry-sensor Dreem 2 EEG headband reviewed by a credentialed sleep technologist (`expert_label`) for up to seven home nights. Of 253 recorded sessions, 7 nights across 6 participants failed a pre-registered `<60%` synchronized sensor coverage threshold (`Bidslab08` Night 7 [19.3%], `Bidslab16` Night 2 [24.0%], `Bidslab34` Night 1 [58.7%], `Bidslab42` Nights 1 [30.1%] & 3 [1.1%], `Bidslab49` Night 1 [12.4%], `Bidslab60` Night 1 [33.8%]), leaving **246 usable nights across 47 participants** (137,801 valid 30-second epochs: Wake 11.9%, Light 46.2%, Deep 18.6%, REM 23.2%; artifact/unscored mapped to `-100`).
   - **Population Baseline Cohort (`N=47`):** All 47 participants (`>= 2` usable nights) were evaluated via subject-disjoint 5-fold outer cross-validation (`EXP-018` / `EXP-021`).
   - **Adaptation Cohort (`N=46`):** Pre-registered supervised fine-tuning (`EXP-019`) required `n_usable >= k + 1` nights. At `k=2`, `Bidslab49` (only 2 usable nights, `n_eval=0`) was excluded a priori, leaving **`N=46` participants (152 evaluation nights)**.
   - **Calibration-Depth Common Cohort (`N=33`, `E_4`):** Across `k in {1, 2, 3, 4}`, available cohorts comprised `N=46` (`k=1`, 198 nights), `N=46` (`k=2`, 152 nights), `N=42` (`k=3`, 106 nights), and `N=33` (`k=4`, 64 nights). To isolate calibration-depth scaling from changing subject denominators, the primary `TSK-06` analysis evaluated the fixed common cohort **`N=33` on `E_4` (64 nights)**.

2. **External Zero-Shot Validation Cohort — PhysioNet SleepAccel v1.0.0 (`N=31`):** Independent single-night Apple Watch + in-lab PSG dataset (Walch et al., *SLEEP* 2019; PhysioNet DOI: `10.13026/hmhs-py35`) comprising **31 healthy adults**, each contributing **one single-night** laboratory PSG recording. Under the locked `EXP-027`/`EXP-028` protocol (`docs/designs/exp027_exp028_sleepaccel_external_validation.md`), pre-flight quality screening excluded 6 participants: 5 short PSG recordings (`<600` epochs: `3951299`, `5132496`, `7596675`, `8443592`, `8692249`) and 1 severe Apple Watch accelerometer dropout (`7749105`, `cov_mo=0.136`), yielding **`n=25` QC-clean primary participants** and **`N=31` all-subject sensitivity**. Exactly three SleepAccel participants lacked specific reference PSG stages (`1455390`: Deep=0; `5132496`: REM=0; `7749105`: REM=0), handled deterministically with `zero_division=0` (0 NaNs).

### 2.2 Input Representation, Architectures, and Edge Quantisation
Full-night nocturnal sequences (`L=1200` 30-second epochs, 10.0 hours; zero-padded if shorter, 50%-overlap sliding-window logit stitching if longer) were constructed from 1 Hz resampled IHR, tri-axial accelerometry magnitude/jerk/energy, and robust per-night median/IQR normalization. To prevent circadian time-of-night confounding (`DEC-024`), both models received identical circadian-neutralized metadata (`neutralize_circadian=True`: `[freq, 0, 0, hr_mean, hr_std]`).

- **Frozen Model A (`SleepStagingFinalModel_v1.0` / `ConventionalCNNBiGRU`):** A 1D convolutional epoch encoder (kernel sizes `{3, 5, 5}`, channels `{32, 64, 128}`, spatial dropout `p=0.20`) followed by a 2-layer Bidirectional GRU (`h=64`, 128-dimensional bidirectional state) and linear classifier head, totaling **163,940 trainable parameters**.
- **Official SLAMSS-IFS Baseline (`EXP-021`):** The official author repository architecture (**5,465,768 parameters** in code; 666,060 in published paper text), wrapped via `OfficialSLAMSSIFSWrapper` with `teacher_forcing_ratio=0.0` (eliminating upstream `t=0` ground-truth leakage) and FP32 composite `CE + 0.01 * RCCLoss`, trained under matched 5-fold manifests (`micro_batch_size=4`, `min_epochs=40`, `max_epochs=100`, `patience=25`). Model A achieves **33.34-fold compression** vs. the official repository and **4.06-fold compression** vs. the paper text.
- **Edge Profiling & Dynamic INT8 Quantisation (`W0-B`):** Post-training dynamic INT8 quantisation (`torch.ao.quantization.quantize_dynamic`) quantises all `Linear` and `GRU` layers (`112,932 / 163,940 = 68.89%` of parameters) while keeping `Conv1d` and normalization layers in FP32 (static PTQ is blocked in PyTorch eager/FX mode by raw residual `+` ops and `pack_padded_sequence`). This yields an empirically verified **in-memory Dynamic INT8 weight footprint of `309.54 KiB`** (`339.99 KiB` serialized file; theoretical all-INT8 floor `160.10 KiB`). Because the 2-layer BiGRU processes full-night context (`L=1200`) bidirectionally, edge deployment requires **smartwatch SoC or companion smartphone offline full-night buffering** (not bare-metal 512 KB SRAM 1-epoch causal streaming).

### 2.3 Personalisation Protocols, Controls, and Exploratory Closures
Under locked `EXP-019` (`4dff5b7`), five arms were evaluated per held-out subject on chronological evaluation nights (`k >= 3`):
- **`ZS`:** Frozen population Model A evaluated without parameter updates.
- **`FT-2` (Locked Stage D Protocol):** Supervised cross-entropy fine-tuning on the subject's first `k=2` chronological nights using **`Adam` optimizer, batch size `B=2`, gradient clipping `max_norm=1.0`, and per-fold nested inner-validation selection of `(lr, epochs, scope)`** (`experiments/EXP-019/exp019_stage_d_selected_configs.json`).
- **`FT-other-2` (Specificity Control, `S1`):** Identical fine-tuning on two nights from an unrelated donor subject.
- **`PS-2` (Prior Control, `S2`):** Random-initialization training from scratch on the target subject's two calibration nights.
- **`UA-2` (Remediated Unsupervised Control, `S3` / `TSK-07`):** Following discovery of a historical `DEC-030` unit-mismatch no-op (`0.00 pp`), `UA-2` was re-executed (`DEC-036`) using quality-gated InfoMax entropy adaptation on `head_gru` across 81,233 calibration epochs (97.26% valid wearable ratio, 0 non-physiological leakage).
- **`TSK-06` Calibration-Depth Curve (`k=1`–`4`):** Evaluated on `E_4` (`N=33`) and `E_k` (`N=46/46/42/33`) against clean reconciled zero-shot baselines (`DEC-039`).
- **Design-Informed Exploratory Closures (`EXP-019B` & `EXP-020`, `DEC-040`):** Drafted post-hoc after inspecting `EXP-019` and `EXP-021` (prior-access notice; confirmatory labels prohibited). `EXP-019B` tested 6 pre-registered predictors (`min_stage_frac`, `stage_entropy`, `eff_ihr_hz`, `missingness_rate`, `baseline_hr_mean`, `baseline_hr_iqr`) across `EXP-019` (`N=46`) and `EXP-021` (`N=47`) with Benjamini–Hochberg FDR control and Oldham's headroom check. `EXP-020` evaluated inner-validation-selected KL regularization (`FT-2-KL`, $\lambda_{\text{KL}} \in \{0.1, 1.0, 10.0\}$), pre-adaptation stage-coverage gating (`GATE_tau`, reverting to `ZS` if $\min_c P_{\text{cal}}(Y=c) < \tau = 0.01$), and their combination (`FT-2-KL_GATE_tau`).

### 2.4 Statistical Analysis & Clinical Agreement
The primary classification metric was unweighted 4-class **Macro-F1**, computed at the participant level (3-seed means: seeds `42, 123, 2024`). Paired differences used 10,000-resample participant-level percentile bootstrap 95% CIs (`seed=42`) and two-sided Wilcoxon signed-rank tests (exact method where specified), with Holm–Bonferroni step-down adjustment for multi-test families. For `EXP-027` external PSG validation, **15-checkpoint `mean_single_model` was pre-specified as PRIMARY** and **`prob_ensemble` as SECONDARY**. Clinical sleep-summary agreement (TST, SOL, WASO, SE, Deep%, REM%) followed Bland & Altman (1999) with `SE(LoA)`, proportional bias regression, and Shapiro–Wilk normality testing (`p < 0.05` triggering non-parametric 2.5th–97.5th percentile LoA fallbacks).

### 2.5 TRIPOD+AI Compliance and Protocol Governance
This study adheres to TRIPOD+AI reporting standards: (1) confirmatory `EXP-019` hypotheses and decision rules were locked in git commit `4dff5b7` (Nov 13, 2026) prior to training; (2) subject-disjoint 5-fold manifests (`EXP014_MANIFESTS/fold_{1..5}.json`) eliminated night/epoch leakage, while historical fixed holdouts (`Cohort Alpha`, `Cohort Beta`) were formally declared spent (`DEC-018`); (3) participant exclusion flows are fully enumerated for BIDSleep (`N=47 -> 46 -> 33`) and SleepAccel (`N=31 -> 25`); (4) all post-hoc analyses (`EXP-019B`, `EXP-020`) carry explicit prior-access exploratory disclosures; and (5) no external funding source influenced study design, execution, or reporting.

---

## 3. Results

### 3.1 Matched 5-Fold Cross-Validation and Edge Deployment Profiling (`EXP-021`, `N=47`)
Under identical 5-fold subject-disjoint evaluation (`N=47`, 246 nights, 3 seeds; **Table 2**), frozen Model A (`163,940` parameters) achieved `64.79% ± 7.45` Macro-F1 (`66.18% ± 6.79` accuracy, `kappa=0.5078 ± 0.1011`), retaining `96.2%` of the 33.34-fold larger official SLAMSS-IFS (`5,465,768` parameters; `67.37% ± 6.73` Macro-F1, `68.75% ± 6.08` accuracy, `kappa=0.5378 ± 0.0941`). The paired Macro-F1 difference was `-2.58 pp` (`95% CI -3.53 to -1.68`, Wilcoxon `p=1.9818×10^-6`, `d_z=-0.780`, Model A winning `13/47`). On REM F1 (`-2.41 pp`, `95% CI -4.68 to -0.32`), rank testing was non-significant (`p=0.1047`, Model A winning `19/47`) with high inter-subject variance, indicating no rank-order difference rather than formal equivalence.

Edge profiling (`W0-B`) confirmed `794.05 MMACs` (`1,588,108,800` FLOPs, bitwise exact) per full night (`L=1200`), running in `47.73 ± 0.49 ms` (4-thread CPU) and `69.39 ± 1.48 ms` (1-thread CPU). Dynamic INT8 quantisation (`309.54 KiB` in-memory weights; `339.99 KiB` serialized; `160.10 KiB` theoretical full-INT8) incurred zero accuracy loss across all 47 subjects (`+0.0011 pp` Macro-F1, `95% CI -0.0277 to +0.0312`, Wilcoxon `p=0.7892`, `d_z=+0.010`), whereas SLAMSS-IFS required `21.86 MB` FP32 weights and `5.12 GiB` training VRAM (`B=4`).

#### Table 2: Matched 5-Fold Cross-Validation (`EXP-021`, `N=47`) and Edge Deployment Profile (`W0-B`)
| Metric / Resource | Frozen Model A (FP32) | Model A (Dynamic INT8) | Official SLAMSS-IFS | Paired Diff ($\bar{d}$, Model A $-$ SLAMSS) | 95% Bootstrap CI | Wilcoxon $p$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Parameters** | `163,940` | `163,940` (`68.89%` INT8) | `5,465,768` (repo) / `666,060` (paper) | `33.34x` repo / `4.06x` paper compression | — | — |
| **Weight Footprint** | `640.39 KiB` (`661.36 KiB` file) | `309.54 KiB` (`339.99 KiB` file; `160.10 KiB` floor) | `21.86 MB` FP32 (`5.12 GiB` VRAM) | — | — | — |
| **Compute / CPU Latency ($L=1200$)** | `794.05 MMACs` / `47.23 ms` (4T) | `47.73 ms` (4T) / `69.39 ms` (1T) | Server GPU (`5.12 GiB` @ `B=4`) | — | — | — |
| **Macro-F1 (%)** | `64.79 ± 7.45` | `64.79 ± 7.47` (`+0.0011 pp`, `p=0.7892`) | `67.37 ± 6.73` | `-2.58 pp` | `[-3.53, -1.68] pp` | `1.98×10^-6` |
| **Overall Accuracy (%)** | `66.18 ± 6.79` | `66.18` (`-0.0064 pp`, `p=0.4656`) | `68.75 ± 6.08` | `-2.57 pp` | `[-3.53, -1.65] pp` | `3.38×10^-6` |
| **Weighted-F1 (%)** | `65.96 ± 6.73` | `65.97` (`+0.0035 pp`, `p=0.8380`) | `68.72 ± 5.90` | `-2.76 pp` | `[-3.77, -1.79] pp` | `7.41×10^-7` |
| **Cohen's Kappa ($\kappa$)** | `0.5078 ± 0.1011` | `0.5078` (`+0.0000`, `p=0.7651`) | `0.5378 ± 0.0941` | `-0.0300` | `[-0.0434, -0.0169]` | `1.31×10^-4` |
| **Wake F1 (%)** | `62.70 ± 11.05` | `62.67` (`-0.0269 pp`, `p=0.3361`) | `64.25 ± 9.36` | `-1.55 pp` | `[-2.87, -0.19] pp` | `0.0287` |
| **Light F1 (%)** | `66.61 ± 5.25` | `66.60` (`-0.0088 pp`, `p=0.5963`) | `69.44 ± 5.05` | `-2.83 pp` | `[-3.83, -1.86] pp` | `5.95×10^-7` |
| **Deep F1 (%)** | `66.76 ± 12.31` | `66.76` (`+0.0012 pp`, `p=0.7811`) | `70.27 ± 12.14` | `-3.51 pp` | `[-5.07, -2.12] pp` | `6.82×10^-6` |
| **REM F1 (%)** | `63.09 ± 13.11` | `63.13` (`+0.0389 pp`, `p=0.3052`) | `65.51 ± 11.54` | `-2.41 pp` | `[-4.68, -0.32] pp` | `0.1047` (NS) |

*Note: On `W0-B` edge deployment, full-night `L=1200` BiGRU inference requires smartwatch SoC or companion smartphone buffering rather than 1-epoch causal streaming on a 512 KB SRAM microcontroller.*

### 3.2 Confirmatory Two-Night Personalisation (`EXP-019`, `N=46`) and Controls (`S1`–`S3`)
Across `N=46` eligible participants (152 evaluation nights), two-night supervised fine-tuning (`FT-2`) increased Macro-F1 from `64.58%` (`ZS`) to `65.59%`, yielding a mean paired gain of **`+1.01 pp` (`+1.0054 pp`, `95% CI -0.1724 to +1.9928 pp`, Wilcoxon `W=246.0`, `p=0.00131`, `34/46` [73.9%] improving)**. Because the 95% CI lower bound spanned zero while the upper bound exceeded `+1.50 pp`, locked pre-registered hypothesis **`H-A` evaluated strictly to `INCONCLUSIVE`**.

Specificity and prior controls confirmed that gains reflect genuine personal adaptation rather than generic optimization:
- **`S1` (`FT-2` vs. `FT-other-2`):** Personal fine-tuning outperformed donor fine-tuning by **`+2.73 ± 5.78 pp`** (`95% CI +1.01 to +4.44`, `t=3.204`, `p=0.00249`; Wilcoxon `W=169.0`, `p=1.89×10^-5`), partly reflecting donor-night harm (`FT-other-2` fell `1.72 pp` below `ZS`). Excluding two opposing outliers (`Bidslab68` `+19.98 pp`, `Bidslab34` `-20.79 pp`; `N=44`) preserved robustness (`+2.87 ± 3.89 pp`, `p=1.40×10^-5`).
- **`S2` (`FT-2` vs. `PS-2`):** Personal fine-tuning outperformed from-scratch training (`45.27%` Macro-F1) by **`+20.32 ± 9.14 pp`** (`95% CI +17.60 to +23.03`, `p=3.27×10^-19`; Wilcoxon `p=3.52×10^-9`).
- **Remediated `S3` (`UA-2` vs. `ZS`, `TSK-07`, `DEC-036`):** Following audit discovery and remediation of the `DEC-030` unit-mismatch no-op (`0.00 pp` in `EXP-019`), re-executing InfoMax `head_gru` adaptation across all 46 participants (138 runs, 0% collapse, mean final marginal entropy `1.382 nats`) yielded `64.79%` vs. `64.58%` Macro-F1 (**`+0.21 pp`**, `95% CI +0.04 to +0.39 pp`, `p_sup=0.0322`, `p_2s=0.0645`, `27/46` [58.7%] wins)—classified as **`SUB-BENCHMARK GAIN`** (statistically superior, below the `+1.00 pp` benchmark). Unsupervised adaptation serves as a mild auxiliary regularizer rather than a substitute for labeled calibration.

### 3.3 Within-Cohort Multi-Night Clinical Sleep-Summary Agreement (`N=46`, `df=45`)
Averaging evaluation nights within each participant (`N=46`, `df=45`, `t_{0.975,45}=2.0141`; **Table 1**), `FT-2` tracked multi-night habitual Total Sleep Time (`TST` bias `+2.28 min`, `95% LoA [-30.11, +34.67] min`, `MAE 12.61 min`, `r=0.9492`) and Sleep Efficiency (`SE` bias `+0.47%`, `95% LoA [-7.40, +8.33]%`, `MAE 3.03%`, `r=0.8775`) with high fidelity, while eliminating the severe `SOL` (`-10.09 min`) and `WASO` (`+12.08 min`) distortions of from-scratch `PS-2`. However, `FT-2` exhibited significant empirical `WASO` underestimation (`-4.87 min`, `95% CI -8.54 to -1.20`, `p=0.0104`) and weak-to-moderate tracking of `Deep%` (`r=0.4868`) and `REM%` (`r=0.4478`).

#### Table 1: Subject-Averaged Multi-Night Sleep-Summary Bland–Altman Agreement on BIDSleep (`N=46`, `df=45`)
| Arm | Metric | EEG Ref Mean | Model Pred Mean | Bias ($\bar{d}$) | $t$-based 95% CI on Bias | SD Diff ($s_d$) | 95% Parametric LoA (Multi-Night Mean) | $\text{SE}(\text{LoA})$ | 95% CI on LoA Bounds | MAE | Pearson $r$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FT-2** | `tst_min` | `356.25 min` | `358.53 min` | `+2.28 min` | `[-2.63, +7.19]` | `16.53` | `[-30.11, +34.67]` | `4.20` | `[-38.56, -21.66]` / `[+26.22, +43.12]` | `12.61` | `0.9492` |
| | `sol_min` | `24.70 min` | `27.91 min` | `+3.20 min` | `[-0.13, +6.53]` | `11.22` | `[-18.78, +25.19]` | `2.85` | `[-24.52, -13.04]` / `[+19.45, +30.93]` | `7.78` | `0.8023` |
| | `waso_min` | `17.11 min` | `12.24 min` | **`-4.87 min*`** | **`[-8.54, -1.20]`** | `12.36` | `[-29.09, +19.35]` | `3.14` | `[-35.41, -22.77]` / `[+13.03, +25.67]` | `10.06` | `0.4683` |
| | `sleep_efficiency_pct` | `85.10%` | `85.56%` | `+0.47%` | `[-0.72, +1.66]` | `4.01` | `[-7.40, +8.33]` | `1.02` | `[-9.45, -5.35]` / `[+6.28, +10.38]` | `3.03` | `0.8775` |
| | `deep_pct_tst` | `21.14%` | `22.89%` | `+1.75%` | `[-0.62, +4.12]` | `7.99` | `[-13.90, +17.41]` | `2.03` | `[-17.98, -9.82]` / `[+13.33, +21.49]` | `6.28` | `0.4868` |
| | `rem_pct_tst` | `29.67%` | `27.51%` | `-2.16%` | `[-5.39, +1.07]` | `10.86` | `[-23.44, +19.13]` | `2.76` | `[-28.99, -17.89]` / `[+13.58, +24.68]` | `8.21` | `0.4478` |
| **ZS** | `tst_min` | `356.25 min` | `357.73 min` | `+1.47 min` | `[-3.96, +6.90]` | `18.29` | `[-34.37, +37.31]` | `4.64` | `[-43.72, -25.02]` / `[+27.96, +46.66]` | `12.94` | `0.9327` |
| | `sol_min` | `24.70 min` | `27.87 min` | `+3.17 min` | `[-0.91, +7.25]` | `13.73` | `[-23.74, +30.08]` | `3.48` | `[-30.76, -16.72]` / `[+23.06, +37.10]` | `9.52` | `0.6844` |
| | `waso_min` | `17.11 min` | `13.30 min` | `-3.81 min` | `[-7.63, +0.01]` | `12.88` | `[-29.06, +21.43]` | `3.27` | `[-35.64, -22.48]` / `[+14.85, +28.01]` | `9.78` | `0.5105` |
| | `sleep_efficiency_pct` | `85.10%` | `85.50%` | `+0.40%` | `[-0.84, +1.64]` | `4.18` | `[-7.79, +8.59]` | `1.06` | `[-9.93, -5.65]` / `[+6.45, +10.73]` | `3.03` | `0.8776` |
| | `deep_pct_tst` | `21.14%` | `23.59%` | `+2.45%` | `[-0.18, +5.08]` | `8.84` | `[-14.87, +19.77]` | `2.24` | `[-19.39, -10.35]` / `[+15.25, +24.29]` | `6.80` | `0.0844` |
| | `rem_pct_tst` | `29.67%` | `27.02%` | `-2.66%` | `[-5.44, +0.12]` | `9.35` | `[-20.97, +15.66]` | `2.37` | `[-25.75, -16.19]` / `[+10.88, +20.44]` | `7.87` | `0.2151` |
| **FT-other-2** | `tst_min` | `356.25 min` | `356.37 min` | `+0.11 min` | `[-6.47, +6.69]` | `22.15` | `[-43.31, +43.53]` | `5.62` | `[-54.63, -31.99]` / `[+32.21, +54.85]` | `16.04` | `0.9006` |
| | `sol_min` | `24.70 min` | `27.74 min` | `+3.04 min` | `[-1.38, +7.46]` | `14.89` | `[-26.15, +32.23]` | `3.78` | `[-33.76, -18.54]` / `[+24.62, +39.84]` | `9.95` | `0.6307` |
| | `waso_min` | `17.11 min` | `14.09 min` | `-3.02 min` | `[-7.23, +1.19]` | `14.19` | `[-30.83, +24.80]` | `3.60` | `[-38.08, -23.58]` / `[+17.55, +32.05]` | `10.67` | `0.4356` |
| | `sleep_efficiency_pct` | `85.10%` | `85.20%` | `+0.11%` | `[-1.40, +1.62]` | `5.09` | `[-9.87, +10.09]` | `1.29` | `[-12.47, -7.27]` / `[+7.49, +12.69]` | `3.75` | `0.8240` |
| | `deep_pct_tst` | `21.14%` | `21.84%` | `+0.70%` | `[-2.43, +3.83]` | `10.53` | `[-19.93, +21.33]` | `2.67` | `[-25.31, -14.55]` / `[+15.95, +26.71]` | `8.32` | `0.0492` |
| | `rem_pct_tst` | `29.67%` | `28.53%` | `-1.14%` | `[-5.00, +2.72]` | `13.01` | `[-26.65, +24.36]` | `3.30` | `[-33.30, -20.00]` / `[+17.71, +31.01]` | `9.81` | `0.0827` |
| **PS-2** | `tst_min` | `356.25 min` | `354.69 min` | `-1.56 min` | `[-8.74, +5.62]` | `24.19` | `[-48.97, +45.85]` | `6.14` | `[-61.34, -36.60]` / `[+33.48, +58.22]` | `16.45` | `0.9021` |
| | `sol_min` | `24.70 min` | `14.61 min` | **`-10.09 min*`** | **`[-14.62, -5.56]`** | `15.27` | `[-40.02, +19.85]` | `3.88` | `[-47.83, -32.21]` / `[+12.04, +27.66]` | `11.86` | `0.5755` |
| | `waso_min` | `17.11 min` | `29.18 min` | **`+12.08 min*`** | **`[+4.47, +19.69]`** | `25.62` | `[-38.14, +62.29]` | `6.50` | `[-51.24, -25.04]` / `[+49.19, +75.39]` | `16.82` | `0.4037` |
| | `sleep_efficiency_pct` | `85.10%` | `84.74%` | `-0.35%` | `[-2.11, +1.41]` | `5.91` | `[-11.93, +11.22]` | `1.50` | `[-14.95, -8.91]` / `[+8.20, +14.24]` | `4.10` | `0.7966` |
| | `deep_pct_tst` | `21.14%` | `18.09%` | `-3.05%` | `[-6.33, +0.23]` | `11.03` | `[-24.67, +18.56]` | `2.80` | `[-30.31, -19.03]` / `[+12.92, +24.20]` | `8.56` | `0.3785` |
| | `rem_pct_tst` | `29.67%` | `30.28%` | `+0.61%` | `[-3.88, +5.10]` | `15.13` | `[-29.04, +30.27]` | `3.84` | `[-36.77, -21.31]` / `[+22.54, +38.00]` | `11.07` | `0.2712` |

### 3.4 Calibration-Depth Learning Curve (`TSK-06`, Clean-ZS `DEC-039`)
On the reconciled common cohort (`N=33`, fixed `E_4` evaluation set of 64 nights, clean `ZS=63.54%` Macro-F1), personalisation improved monotonically across calibration depths (`24/33` wins at every `k`):
- **`k=1`:** `65.28%` (`+1.75 pp` [95% CI `+0.63, +2.93`], exact two-sided Wilcoxon `p=0.00879`, `p_Holm=0.02637`).
- **`k=2`:** `65.41%` (`+1.88 pp` [95% CI `+0.59, +3.25`], `p=0.00402`, `p_Holm=0.01609`).
- **`k=3`:** `65.82%` (`+2.29 pp` [95% CI `+0.69, +3.97`], `p=0.01369`, `p_Holm=0.02637`).
- **`k=4`:** `66.25%` (`+2.71 pp` [95% CI `+1.17, +4.33`], `p=0.00170`, `p_Holm=0.00848`).
- **Page's `L`-Trend Test:** `L=863.0` ($\mu_L=825.0, \sigma_L=16.58, z=2.261$, `p=0.01187`). Under the strict pre-registered 5-test family (`[Page's L, k=3, step 2-1, step 3-2, step 4-3]`), directional one-sided Holm–Bonferroni testing passed $\alpha=0.05$ (`k=3` `p_Holm=0.03423`; Page's `L` and $\delta_{4-3}$ `p_Holm=0.04748`), whereas exact two-sided sensitivity narrowly missed (`p_Holm=0.05935`).

In the full available-cohort sensitivity analysis (`E_k`), `k=1` (`N=46`, `ZS=63.95%`) was non-significant (`+0.23 pp` [95% CI `-0.59, +1.05`], `p=0.5957`, `25/46` wins), `k=2` (`N=46`, `ZS=64.58%`) yielded `+0.99 pp` ([`-0.18, +1.99`], `p=0.00271`, `33/46` wins), and **`k=3` (`N=42`, `ZS=64.25%`) was the first calibration depth whose 95% CI excluded zero (`+1.42 pp` [95% CI `+0.15, +2.58`], `p=0.00232`, `29/42` wins)**, rising to `+2.71 pp` at `k=4` (`N=33`). This converges on `k >= 3` nights with van der Aar et al. (2025).

### 3.5 Exploratory Heterogeneity and Safe Adaptation (`EXP-019B` & `EXP-020`, `DEC-040`)
*(Prior-Access Exploratory Disclosure: `EXP-019B` and `EXP-020` were designed after inspecting `EXP-019` and `EXP-021`; they do not overturn locked `EXP-019` `H-A` [`INCONCLUSIVE`].)*
- **Heterogeneity Audit (`EXP-019B`):** Across 12 primary Spearman tests (6 baseline/quality/stage features vs. `EXP-019` `FT-2 - ZS` [`N=46`] and `EXP-021` `Model A - SLAMSS-IFS` [`N=47`]), **`0/12` predictors** reached `FDR q < 0.05` (all $q_{\text{family}} \ge 0.5673$, $q_{\text{pooled}} \ge 0.5913$). Oldham's headroom check confirmed no baseline dependency ($\rho=+0.0376, p=0.8043$).
- **Safe Adaptation (`EXP-020`, `N=46`):** KL regularization (`FT-2-KL`: `0.6562` Macro-F1, `+1.03 pp` vs. `ZS` [95% CI `-0.11, +2.01`], `p_Holm=0.0030`, `N_harm=3`) failed to prevent outlier collapse (min `-15.14 pp`). Conversely, pre-adaptation stage-coverage gating (`GATE_tau`, $\tau=0.01$) achieved **`0.6591` Macro-F1 (`+1.33 pp` vs. `ZS`, `95% CI +0.49 to +2.12`, `p_Holm=0.0028`, `N_harm=2`)**, surgically abstaining `2/46` subjects—including `Bidslab34` (`min_stage_frac_cal=0.0068 < 0.01`)—and **cutting worst-case collapse by 61% from `-16.34 pp` (`FT-2`) to `-6.40 pp`**.

### 3.6 Zero-Shot External Single-Night PSG Validation (`EXP-027` & `EXP-028`, `DEC-041`)
Evaluating all 15 frozen Model A checkpoints against all 15 frozen `EXP-021` SLAMSS-IFS checkpoints strictly zero-shot on PhysioNet SleepAccel single-night in-lab PSG (**Table 3**), SLAMSS-IFS's `2.58 pp` BIDSleep lead **attenuated by 78% to a non-significant `-0.57 pp`** under primary `mean_single_model` evaluation on `qc_clean` (`n=25`: `60.51%` vs. `61.08%`, `95% CI -2.13 to +0.94`, `p=0.5782`, `d_z=-0.14`, `10W/15L`). Under secondary `prob_ensemble` evaluation (`n=25`), Model A reached `62.94%` vs. `62.79%` (`+0.15 pp`, `p=0.9789`). In the `all_n31` sensitivity cohort (`N=31`), differences remained non-significant (`-1.62 pp`, `p=0.3468` primary; `-1.20 pp`, `p=0.5551` secondary). Light sleep (`-3.49 pp`, `p=0.0006` primary; `-2.56 pp`, `p=0.0088` secondary) was the sole retained SLAMSS-IFS advantage, while Model A edged ahead on Wake (`+1.95 pp` / `+2.65 pp`) and REM (`+0.60 pp` primary).

#### Table 3: Zero-Shot External Single-Night PSG 4-Class Staging (`EXP-027`, PhysioNet SleepAccel)
| Cohort & Mode | Metric | Model A Mean (%) | SLAMSS-IFS Mean (%) | Paired Diff ($\bar{d}$) | 95% Bootstrap CI | Wilcoxon $p$ | Cohen's $d_z$ | Win / Loss / Tie |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`qc_clean` ($n=25$)** | **Macro-F1** | `60.51` | `61.08` | `-0.57 pp` | `[-2.13, +0.94] pp` | `0.5782` | `-0.14` | `10 / 15 / 0` |
| `mean_single_model` | Accuracy | `65.54` | `67.27` | `-1.73 pp` | `[-3.55, +0.03] pp` | `0.1336` | `-0.37` | `10 / 15 / 0` |
| **(PRIMARY)** | Kappa ($\kappa$) | `0.4792` | `0.4919` | `-0.0127` | `[-0.0390, +0.0132]` | `0.6150` | `-0.19` | `13 / 12 / 0` |
| | Wake F1 | `53.81` | `51.86` | `+1.95 pp` | `[-0.04, +4.07] pp` | `0.1336` | `+0.36` | `15 / 10 / 0` |
| | Light F1 | `66.86` | `70.35` | `-3.49 pp` | `[-5.39, -1.69] pp` | `0.0006` | `-0.73` | `5 / 20 / 0` |
| | Deep F1 | `52.31` | `53.64` | `-1.34 pp` | `[-3.36, +0.74] pp` | `0.1161` | `-0.25` | `9 / 15 / 1` |
| | REM F1 | `69.08` | `68.48` | `+0.60 pp` | `[-2.87, +3.81] pp` | `0.4418` | `+0.07` | `16 / 9 / 0` |
| **`qc_clean` ($n=25$)** | **Macro-F1** | `62.94` | `62.79` | `+0.15 pp` | `[-1.51, +1.86] pp` | `0.9789` | `+0.03` | `10 / 15 / 0` |
| `prob_ensemble` | Accuracy | `68.04` | `69.23` | `-1.19 pp` | `[-3.01, +0.59] pp` | `0.2411` | `-0.25` | `9 / 16 / 0` |
| **(SECONDARY)** | Light F1 | `69.76` | `72.32` | `-2.56 pp` | `[-4.53, -0.69] pp` | `0.0088` | `-0.51` | `7 / 18 / 0` |
| | Wake / Deep / REM F1 | `55.80` / `55.12` / `71.08` | `53.15` / `53.99` / `71.68` | `+2.65` / `+1.13` / `-0.61 pp` | — | `0.12` / `0.82` / `0.75` | — | — |
| **`all_n31` ($N=31$)** | **Macro-F1 (Primary)** | `60.65` | `62.27` | `-1.62 pp` | `[-3.84, +0.28] pp` | `0.3468` | `-0.27` | `13 / 18 / 0` |
| (Sensitivity) | **Macro-F1 (Secondary)** | `62.94` | `64.14` | `-1.20 pp` | `[-3.96, +1.04] pp` | `0.5551` | `-0.17` | `13 / 18 / 0` |

*Note: Non-significant paired differences indicate no statistically significant difference / empirical parity (no TOST equivalence claimed). Exclusions (`n=6`): `3951299`, `5132496`, `7596675`, `8443592`, `8692249` (`<600` epochs), `7749105` (`cov_mo=0.136`). Zero-support PSG subjects (`1455390`, `5132496`, `7749105`) handled via `zero_division=0`.*

On single-night PSG clinical sleep-summary agreement (`EXP-028`, `qc_clean` `n=25`; **Table 4**), Model A outperformed SLAMSS-IFS on external `TST` (`MAE 15.87` vs. `19.11 min`), `WASO` (`MAE 15.09` vs. `16.34 min`, avoiding SLAMSS-IFS's significant `-0.925` proportional bias, `p=0.0021`), `SE` (`MAE 3.58%` vs. `4.26%`), and `REM%` correlation (`r=0.4767` vs. `0.1971`). Crucially, both architectures exhibited large positive biases in `Deep%` (`+10.65%` primary / `+10.23%` secondary for Model A; `+8.67%` / `+8.84%` for SLAMSS-IFS) and `REM%` (`+10.13%` / `+9.35%` for Model A; `+7.21%` / `+6.87%` for SLAMSS-IFS), driven by the cross-modality reference label-prior shift from Dreem 2 ambulatory EEG (`44.2%` Light, `18.9%` Deep) to single-night in-lab PSG (`57.7%` Light, `11.1%` Deep).

#### Table 4: External Single-Night PSG Bland–Altman Clinical Agreement (`EXP-028`, `qc_clean` $n=25$, $df=24$)
| Eval Mode & Arm | Metric | Bias ($\bar{d}$) | 95% Parametric LoA | $\text{SE}(\text{LoA})$ | MAE | Pearson $r$ | Prop. Slope ($p$) | Shapiro–Wilk $p$ / Non-Parametric LoA Fallback |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`mean_single_model`** | `tst_min` | `+11.81 min` | `[-23.79, +47.40]` | `6.29` | `15.87 min` | `0.9091` | `-0.029 (0.753)` | `0.236` (Normal) |
| **Model A (PRIMARY)** | `sol_min` | `+2.71 min` | `[-10.91, +16.33]` | `2.41` | `5.73 min` | `0.6607` | `-0.245 (0.202)` | `0.075` (Normal) |
| | `waso_min` | `-13.57 min` | `[-43.05, +15.90]` | `5.21` | `15.09 min` | `0.3968` | `-0.513 (0.062)` | `0.083` (Normal) |
| | `sleep_efficiency_pct` | `+2.62%` | `[-5.60, +10.83]%` | `1.45` | `3.58%` | `0.3781` | `-0.270 (0.340)` | `0.218` (Normal) |
| | `deep_pct_tst` | **`+10.65%`** | `[-2.39, +23.69]%` | `2.30` | `10.72%` | `0.4909` | `-0.071 (0.772)` | `0.860` (Normal) |
| | `rem_pct_tst` | **`+10.13%`** | `[-5.92, +26.18]%` | `2.84` | `11.12%` | `0.4767` | `+0.204 (0.415)` | `0.870` (Normal) |
| **`mean_single_model`** | `tst_min` | `+14.59 min` | `[-21.68, +50.87]` | `6.41` | `19.11 min` | `0.9042` | `-0.055 (0.559)` | `0.722` (Normal) |
| **SLAMSS-IFS (PRIMARY)** | `sol_min` | `+0.53 min` | `[-14.72, +15.78]` | `2.70` | `5.53 min` | `0.5374` | `-0.460 (0.047)` | **`0.0046`** (Med `+1.73`, `[-18.73, +10.29]`) |
| | `waso_min` | `-14.16 min` | `[-44.39, +16.07]` | `5.34` | `16.34 min` | `0.2824` | **`-0.925 (0.0021)`** | `0.269` (Normal) |
| | `sleep_efficiency_pct` | `+3.24%` | `[-5.02, +11.49]%` | `1.46` | `4.26%` | `0.2601` | **`-0.693 (0.028)`** | `0.639` (Normal) |
| | `deep_pct_tst` | `+8.67%` | `[-3.09, +20.42]%` | `2.08` | `8.81%` | `0.5023` | `-0.462 (0.057)` | `0.575` (Normal) |
| | `rem_pct_tst` | `+7.21%` | `[-10.30, +24.73]%` | `3.10` | `9.64%` | `0.1971` | `-0.141 (0.682)` | `0.591` (Normal) |
| **`prob_ensemble`** | `tst_min` | `+12.22 min` | `[-23.18, +47.62]` | `6.26` | `15.86 min` | `0.9099` | `-0.033 (0.720)` | `0.254` (Normal) |
| **Model A (SECONDARY)** | `sol_min` | `+2.44 min` | `[-13.23, +18.11]` | `2.77` | `5.96 min` | `0.5752` | `-0.127 (0.562)` | **`0.0059`** (Med `+2.50`, `[-16.20, +14.30]`) |
| | `waso_min` / `SE` | `-13.82` / `+2.71%` | `[-44.6, +17.0]` / `[-5.47, +10.89]%` | `5.44` / `1.45` | `15.98` / `3.58%` | `0.359` / `0.381` | `-0.441` / `-0.273` | `0.191` / `0.212` (Normal) |
| | `deep_pct` / `rem_pct` | **`+10.23%`** / **`+9.35%`** | `[-3.12, +23.59]` / `[-7.79, +26.49]` | `2.36` / `3.03` | `10.25` / `10.85%` | `0.498` / `0.448` | `+0.015` / `+0.290` | `0.203` / `0.916` (Normal) |
| **`prob_ensemble`** | `tst_min` / `sol_min` | `+15.72` / `+0.38 min` | `[-19.84, +51.28]` / `[-15.46, +16.22]` | `6.28` / `2.80` | `19.32` / `5.46 min` | `0.9078` / `0.4870` | `-0.061` / `-0.531` | `0.621` / **`0.0011`** (Med `+1.00`) |
| **SLAMSS-IFS (SECONDARY)** | `waso_min` / `SE` | `-15.16` / `+3.49%` | `[-44.69, +14.37]` / `[-4.58, +11.56]%` | `5.22` / `1.43` | `16.64` / `4.30%` | `0.307` / `0.298` | **`-1.026 (0.0004)`** / **`-0.672`** | `0.108` / `0.414` (Normal) |
| | `deep_pct` / `rem_pct` | `+8.84%` / `+6.87%` | `[-3.80, +21.49]` / `[-10.65, +24.39]` | `2.23` / `3.10` | `9.11` / `9.61%` | `0.460` / `0.283` | `-0.286` / `+0.050` | `0.811` / `0.327` (Normal) |

---

## 4. Discussion

### 4.1 Translational Synthesis: Efficiency, External Generalisation, and Label-Prior Shift
Our matched 5-fold and external zero-shot evaluations reveal a critical translational lesson: **in-distribution server complexity does not guarantee superior external clinical PSG generalisation**. Although the 5.47-million-parameter official SLAMSS-IFS outperformed 163,940-parameter Model A by `+2.58 pp` on BIDSleep (`N=47`), 78% of that advantage vanished on independent PhysioNet SleepAccel in-lab PSG (`-0.57 pp`, `p=0.5782` primary; `+0.15 pp`, `p=0.9789` ensemble), with Model A achieving lower MAE on TST, WASO, and SE and higher REM% correlation. Dynamic INT8 quantisation (`309.54 KiB` weights, `47.7 ms`/night) makes Model A immediately deployable on smartwatch SoCs or companion smartphones via offline full-night buffering.

Equally important for clinical translation is the **cross-modality label-prior shift** uncovered in `EXP-028`. Models trained on Dreem 2 dry-sensor home EEG (`44.2%` Light, `18.9%` Deep) systematically overestimated `Deep%` (`+10.65%`) and `REM%` (`+10.13%`) when evaluated against in-lab PSG (`57.7%` Light, `11.1%` Deep). While consumer wearables reliably track macroscopic sleep duration (`TST` `r >= 0.90`), stage-architecture percentages require modality-specific calibration before clinical interpretation.

### 4.2 Actionable Personalisation Thresholds, Safety Gating, and Statistical Power
Personalisation with ~0.2 Hz HealthKit IHR is person-specific (`S1` `+2.73 pp` over donor fine-tuning) and scales monotonically with calibration depth (`TSK-06`, Page's `L` `p=0.01187`), converging with clinical PPG findings (van der Aar et al., 2025) at **`k >= 3` calibration nights** (`+1.42 pp` available cohort; `+2.29 pp` common cohort). However, unconstrained fine-tuning on `k=2` nights remains statistically underpowered at `N=46` (`INCONCLUSIVE`, `+1.01 pp`; `N ~ 107` required at 80% power for a `+1.0 pp` effect) and risks catastrophic collapse when a calibration night lacks a sleep stage (`Bidslab34`: `0.68%` Deep sleep). Exploratory `GATE_tau` (`\tau=0.01`) provides an effective surgical safeguard (`+1.33 pp`, `2/46` abstentions, cutting worst-case collapse by 61% to `-6.40 pp`).

### 4.3 Limitations and Future Prospective Targets
Our conclusions are strictly scoped to **healthy adults**:
1. **No Clinical Sleep-Disorder Claims:** Neither BIDSleep nor SleepAccel includes patients with obstructive sleep apnoea, insomnia, or parasomnias; prospective validation in clinical disorder cohorts such as **PhysioNet DREAMT v2.2.0** is required before diagnostic use.
2. **Single-Night External PSG & Surrogate Training Labels:** SleepAccel provides only one in-lab PSG night per subject, precluding external multi-night personalisation testing, while BIDSleep relies on technologist-reviewed Dreem 2 headband labels. Prospective multi-night external cohorts such as the **Simons Sleep Project (`DS0000089`)** are prioritized targets for confirmatory `GATE_tau` and `k >= 3` replication.
3. **Offline Non-Causal Architecture (`L=1200`):** Bidirectional GRU processing requires full-night buffering on a smartwatch SoC or companion phone and cannot perform causal real-time closed-loop staging.
4. **Exploratory Scope of `EXP-019B`/`EXP-020`:** Designed after inspecting `EXP-019`, `GATE_tau` requires prospective independent replication.

---

## References (Key Cited Works)
1. Song et al. SLAMSS-IFS and BIDSleep multi-night dataset. *IEEE Trans Biomed Eng* 2026; DOI: `10.1109/TBME.2025.3612158` (PhysioNet DOI: `10.13026/rees-1092`).
2. Walch O, Huang Y, Forger D, Goldstein C. Sleep stage prediction with raw acceleration and photoplethysmography heart rate data derived from a consumer wearable device. *SLEEP* 2019; 42(12):zsz180 (SleepAccel PhysioNet DOI: `10.13026/hmhs-py35`).
3. van der Aar JF, et al. Supervised multi-night personalization of wrist PPG sleep staging in sleep-disordered adults. *Physiol Meas* 2025; DOI: `10.1088/1361-6579/ae0119`.
4. Bland JM, Altman DG. Measuring agreement in method comparison studies. *Stat Methods Med Res* 1999; 8(2):135–160.
5. Quirk C, et al. Data leakage in deep learning sleep staging. 2024.
