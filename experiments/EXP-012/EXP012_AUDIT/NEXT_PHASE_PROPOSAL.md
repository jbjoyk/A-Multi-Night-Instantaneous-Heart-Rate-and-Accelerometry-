# Next-Phase Research Proposal: Controlled Ablation & Baseline Attribution

> [!IMPORTANT]
> **PROPOSAL ONLY — DO NOT EXECUTE.**
> This document specifies the design for the next experimental phase. In accordance with audit rules, **zero models from this proposal may be trained, tuned, or evaluated until supervisor approval is granted.**

---

## 1. Scientific Objective

In EXP-012, `SleepAdaptiveNet` (`B0_GRL`, 162k parameters) achieved **$68.44\% \pm 1.35\%$ Accuracy** and **$0.6859 \pm 0.0139$ Macro-F1** on the official 11-subject held-out test set, trailing the 5.47M parameter SLAMSS-IFS benchmark by 2.6 pp while reducing parameter count by 33.7x.

To determine whether `SleepAdaptiveNet`'s specific architectural innovations (dynamic reliability gating, transition regularization, subject-invariant GRL) genuinely drive this performance—or whether any generic ~160k parameter network achieves equivalent results—the next phase will execute a rigorous, controlled attribution study.

---

## 2. Experimental Model Matrix

| Model Identifier | Architectural Description | Parameters | Fusion Mechanism | Transition Loss ($\lambda_{\text{trans}}$) | Subject GRL ($\lambda_{\text{adv}}$) | Specific Scientific Purpose |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Model A** | **Matched-Size Plain Baseline** | $\sim 160\text{k} \pm 10\%$ | Simple Early Concatenation | 0.0 (OFF) | 0.0 (OFF) | Isolates capacity from architecture: tests whether a standard CNN-BiGRU of identical parameter budget matches SleepAdaptiveNet. |
| **Model B** | **SleepAdaptiveNet w/o GRL** | $\sim 162\text{k}$ | Learned Adaptive Gating ($\alpha_t$) | 0.10 (ON) | 0.0 (OFF) | Isolates the contribution of Gradient Reversal Layer (subject invariance) on held-out test subjects. |
| **Model C** | **SleepAdaptiveNet w/o Transition** | $\sim 162\text{k}$ | Learned Adaptive Gating ($\alpha_t$) | 0.0 (OFF) | 0.10 (ON) | Isolates the contribution of the transition probability regularization module on boundary and sequence staging. |
| **Model D** | **SleepAdaptiveNet w/o Adaptive Fusion** | $\sim 161\text{k}$ | Fixed Static Equal Fusion ($0.5 / 0.5$) | 0.10 (ON) | 0.10 (ON) | Isolates the contribution of state-dependent dynamic modality reliability gating vs. static feature combination. |
| **Model E** | **SleepAdaptiveNet (`B0_GRL`)** | **162,325** | Learned Adaptive Gating ($\alpha_t$) | 0.10 (ON) | 0.10 (ON) | **Audited EXP-012 benchmark model.** *(FROZEN: Reuses existing predictions; WILL NOT BE RETRAINED).* |

---

## 3. Strict Controlled Execution Protocol (Locked)

When authorized, the execution must strictly adhere to the following constraints:
1. **Identical Cohort & Split**: Exact same 31 train subjects (160 nights), 5 validation subjects (28 nights), 11 test subjects (58 nights).
2. **Identical Sequence Preprocessing**: Sequence length $T=600$, identical IHR/ACC feature normalization, identical padding/masking.
3. **Fixed Training Budget**: Exactly 80 optimization epochs per seed under Adam ($\text{lr} = 10^{-4}$, weight decay 0.0, batch size 4).
4. **Matched Seeds**: Seeds 42, 123, 2024 evaluated for all comparator models ($N=3$).
5. **Zero Test Set Tuning**: Test set strictly isolated during training; evaluated exactly once per seed at locked epoch 80.
6. **No Hyperparameter Search**: Zero per-model learning rate or architecture searching.

---

## 4. Paired Subject-Level Bootstrap Comparison Protocol

To eliminate inter-subject variance confounders when comparing Model E against comparators (A, B, C, D):

1. **Paired Cluster Bootstrap**:
   - Resampling unit: Subject ($N=11$ test subjects).
   - Iterations: $B = 1000$.
   - **Critical Constraint**: The *identical resampled subject indices* must be used simultaneously for Model E and Comparator $M$.
2. **Paired Metric Delta**:
   $$\Delta_{\text{metric}}^{(b)} = \text{Metric}(E, \mathcal{S}_b) - \text{Metric}(M, \mathcal{S}_b)$$
   where $\mathcal{S}_b$ is the $b$-th resampled set of test subjects.
3. **95% Confidence Interval**:
   $$[\Delta_{2.5\%}, \Delta_{97.5\%}]$$
4. **Statistical Decision Rule**:
   - If the 95% CI strictly excludes zero ($\Delta_{2.5\%} > 0$): **Statistically significant evidence supporting component benefit**.
   - If the 95% CI includes zero: **Component benefit is not distinguishable from zero under this bootstrap analysis**.
   - No p-values will be reported unless a parametric or permutation procedure is explicitly implemented and justified.

---

## 5. Explicit Stop Notice

**DO NOT RUN ANY TRAINING OR TESTING FOR THIS PROPOSAL.**  
This document is prepared solely for supervisor review. No script from this phase has been executed.
