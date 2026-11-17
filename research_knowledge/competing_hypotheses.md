# Competing Hypotheses Database: Heterogeneous Adaptation Response

**Epistemic Rule:** Every mechanism is strictly classified as `Observed phenomenon`, `Proposed explanation`, `Tested explanation`, or `Confirmed mechanism`. Exploratory EXP-019 findings are **never** promoted to confirmed mechanisms.

---

### H-MECH-01: Calibration-Night Label / Prior Probability Shift
- **Status:** `Proposed explanation with weak exploratory support` (Exploratory in EXP-019; TTA label-shift theory in vision [Zhao et al. 2023](https://arxiv.org/html/2306.03536), not supervised fine-tuning on physiological time series)
- **Observed Phenomenon (EXP-019 Exploratory):** Subject `Bidslab34` degraded by $-16.34\text{ pp}$ under `FT-2`. Calibration nights 2–3 had $0.00\%$ and $1.34\%$ Deep sleep, whereas evaluation nights 4–5 had $17.65\%$ and $18.91\%$ Deep sleep ($D_{\text{JS}}(P_C \parallel P_E) = 0.3319$).
- **Proposed Explanation:** Fine-tuning on calibration nights missing a sleep stage suppresses emission logits/biases for that stage, causing systematic failure when the stage reappears on evaluation nights.
- **Empirical Evidence:**
  - *EXP-019 Exploratory:* Post-hoc $D_{\text{JS}}(P_C \parallel P_E)$ correlated with $\Delta$ ($\rho = -0.2933, p = 0.0479$). Pre-adaptation $D_{\text{JS}}(P_C \parallel P_{\text{pop}})$ was non-significant as a continuous linear predictor ($q = 0.9711$), suggesting a possible categorical threshold effect rather than linear trend, though this remains unconfirmed.
  - *External Literature (Adjacent):* [Zhao et al. (2023)](https://arxiv.org/html/2306.03536) demonstrated that unsupervised TTA methods degrade under strong label shift ($\alpha = 0.01$) in vision benchmarks, which is an indirect analogy rather than direct proof for supervised physiological fine-tuning.
- **Falsification / Next Test:** Preregistered test of KL-regularized fine-tuning ([Phan et al. 2020](https://lirias.kuleuven.be/retrieve/616969)) or class-prior-preserving adaptation on `Bidslab34` and high-$D_{\text{JS}}$ subjects.

### H-MECH-02: Regression to the Mean (RTM) / Baseline Ceiling Effect
- **Status:** `Tested explanation` (**Refuted** in EXP-019 Exploratory Audit)
- **Observed Phenomenon:** Naive correlation between $ZS$ accuracy and $\Delta$ appeared negative ($\rho = -0.1710, p = 0.2559$).
- **Empirical Evidence:** Under Oldham's correction ($\text{Corr}(\frac{ZS+FT}{2}, \Delta)$), correlation collapsed to $\rho = +0.0376, p = 0.8043$.
- **Contradictory Literature Nuance:** [Lorenzen et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/38833404/) reported larger supervised personalization gains for poor baseline subjects ($\kappa < 0.70$: $+15\%$) vs high baseline subjects ($\kappa > 0.80$: $+2\%$) in wearable EEG. Thus, true headroom effects may exist in EEG, but were **not** the driver of EXP-019's heterogeneity on BIDSleep.

### H-MECH-03: Parameter Displacement / Optimization Instability
- **Status:** `Tested explanation` (**Null** in EXP-019 Exploratory Audit once scope-stratified)
- **Empirical Evidence:** Pooled correlation ($\rho = +0.0441$) was an artifact of Fold 4 using `scope=head_gru` vs `scope=all` in Folds 1, 2, 3, 5. Stratified within `scope=all` ($n=37$), parameter displacement norm had $\rho = -0.0062, p = 0.9711$.

### H-MECH-04: Physiological / Autonomic Drift (HR Volatility)
- **Status:** `Tested explanation` (Exploratory, non-significant in EXP-019)
- **Empirical Evidence:** Intra-calibration HR volatility ($|\text{HR}_{C_1} - \text{HR}_{C_2}|$) had Spearman $\rho = -0.2094, p = 0.1625, q = 0.9711$ across $n=46$ subjects. Insufficient power or non-linear effect; remains unresolved.

### H-MECH-05: Inter-Night Signal Magnitude / Normalization Mismatch
- **Status:** `Proposed explanation` (Literature-grounded, **not yet tested** in EXP-019)
- **Literature Precedent:** [Phan et al. (2020)](https://lirias.kuleuven.be/retrieve/616969) explicitly found that subject-specific normalization caused poor personalization in subjects with large signal magnitude differences across nights, requiring **night-specific normalization**.

### H-MECH-06: Unsupervised Adaptation Failure under Label Imbalance / Small Batch
- **Status:** `Observed phenomenon` (`LOCKED` in EXP-019 S3; `Unresolved: implementation check pending`)
- **Empirical Evidence:** `UA-2` (InfoMax) yielded $+0.00\text{ pp}$ (`CI [0.00, 0.00]`, all 46 subjects tied with `ZS`) in EXP-019, which resembles a no-op rather than label-shift degradation; verification of whether `UA-2` parameters or logits actually changed is pending.
- **Contradictory / Nuanced Literature:** [Lorenzen et al. (2024)](https://pubmed.ncbi.nlm.nih.gov/38833404/) (`P007`) reported no beneficial personalization from unsupervised ADA and statistics alignment in wearable EEG (full-text re-extraction pending), whereas [Heremans et al. (2022)](https://pubmed.ncbi.nlm.nih.gov/35877770/) (`P054`) reported ~1–2% unsupervised adversarial personalization gains.
