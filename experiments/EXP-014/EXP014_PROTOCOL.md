# EXP-014 Protocol Specification: 5-Fold Subject Cross-Validation

**Date**: 2026-11-10  
**Status**: SPECIFICATION LOCKED (AWAITING USER AUTHORIZATION BEFORE ANY TRAINING)  
**Research Objective**: Perform a rigorous, subject-level 5-fold cross-validation across all 47 subjects in the BIDSLab Apple Watch + Dreem 2 dataset to test the generalization robustness of SleepAdaptiveNet, determine the marginal attribution of its individual mechanisms (Adaptive Reliability Fusion, Transition Modeling, and Subject-Invariant GRL), and evaluate whether it provides a real performance benefit over an approximately matched-capacity conventional CNN-BiGRU baseline.

---

## 1. Experimental Architecture Matrix (5 Models)

EXP-014 includes all five models across all five outer folds and three random seeds:

| Model ID | Model Architecture | Parameters (31 subs) | Adaptive Fusion | Transition Module | Subject GRL | Boundary Loss | Scientific Purpose |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model A** | Matched Conventional CNN-BiGRU | 163,940 | OFF (Concat) | OFF | OFF | OFF | Matched-capacity conventional baseline |
| **Model B** | SleepAdaptiveNet minus GRL | 153,974 | ON (Dynamic) | ON | OFF | OFF | Domain-adversarial GRL attribution |
| **Model C** | SleepAdaptiveNet minus Transition | 162,325 | ON (Dynamic) | OFF | ON | OFF | Transition-modeling attribution |
| **Model D** | SleepAdaptiveNet minus Dynamic Gate | 153,876 | OFF (Fixed 0.5) | ON | ON | OFF | Adaptive-reliability-fusion attribution |
| **Model E** | Full SleepAdaptiveNet | 162,325 | ON (Dynamic) | ON | ON | OFF | Full proposed architecture |

### Critical Separation Between Historical EXP-012 and EXP-014 Model E:
- **EXP-012 Frozen Benchmark**: Refers exclusively to the historical single-split experiment (31 train / 5 val / 11 test subjects), evaluated at epoch 80 with frozen weights saved in `exp012_seed{42,123,2024}_final.pt`.
- **EXP-014 Model E / Full SleepAdaptiveNet**: Refers to a freshly instantiated model trained from scratch inside each EXP-014 fold using that fold's inner training subjects. **The historical EXP-012 checkpoint is NEVER loaded for EXP-014 training.**

---

## 2. Experimental Run Count & Computational Budget

The complete experimental matrix is:
$$\mathbf{5\text{ Models}} \times \mathbf{5\text{ Outer Folds}} \times \mathbf{3\text{ Seeds}} = \mathbf{75\text{ Training Runs}}$$

- **Seeds**: 42, 123, 2024 (evaluated for every model in every fold).
- **Optimization Budget**: Exactly 80 epochs per run (fixed budget, no early stopping, no LR scheduler, Adam, $\text{lr}=10^{-4}$, $\text{weight\_decay}=0.0$, batch size 4, sequence length $T=600$ epochs / 5.0 hours).
- **Estimated Runtime**: Each 80-epoch run executes in ~70 seconds on the NVIDIA GeForce RTX 3060.
  $$\text{Total Compute Time} \approx 75 \times 70\text{ s} \approx 5,250\text{ s} \approx \mathbf{87.5\text{ minutes}} \; (\approx 1.45\text{ hours})$$

---

## 3. Subject-Level 5-Fold Partitioning (All 47 Subjects)

Partitioning is deterministic (`seed=42`) across all 47 subjects and 246 usable nights. Zero subject or night overlap exists across outer test partitions:

$$\bigcup_{k=1}^5 \text{OuterTest}_k = \text{All 47 Subjects}, \quad \text{OuterTest}_j \cap \text{OuterTest}_k = \emptyset \; (\forall j \neq k)$$
$$\text{InnerTrain}_k \cap \text{InnerVal}_k = \emptyset, \quad (\text{InnerTrain}_k \cup \text{InnerVal}_k) \cap \text{OuterTest}_k = \emptyset$$

Fold manifests are preserved in `EXP014_MANIFESTS/fold_{1..5}.json`:

| Fold Manifest | Outer Test Subjects ($N_{\text{sub}}$) | Outer Test Nights | Outer Train Subjects ($N_{\text{sub}}$) | Outer Train Nights | Inner Train Subjects ($N_{\text{sub}}$) | Inner Val Subjects ($N_{\text{sub}}$) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`fold_1.json`** | 10 | 54 | 37 | 192 | 32 (168 nights) | 5 (24 nights) |
| **`fold_2.json`** | 10 | 53 | 37 | 193 | 32 (168 nights) | 5 (25 nights) |
| **`fold_3.json`** | 9 | 45 | 38 | 201 | 33 (175 nights) | 5 (26 nights) |
| **`fold_4.json`** | 9 | 47 | 38 | 199 | 33 (174 nights) | 5 (25 nights) |
| **`fold_5.json`** | 9 | 47 | 38 | 199 | 33 (171 nights) | 5 (28 nights) |
| **Total Cohort** | **47 Subjects** | **246 Nights** | — | — | — | — |

---

## 4. Training, Model Selection & Outer-Test Isolation

1. **Inner Training**:
   - For fold $k$, the model is trained exclusively on the `inner_train_subjects` partition.
   - Modality feature normalization (IHR, accelerometry, frequency, time) is fit strictly on inner-train nights.
2. **Inner Validation Monitoring**:
   - The `inner_val_subjects` partition (5 subjects per fold) is evaluated after each epoch 1..80 to record validation trajectories and RNG execution flow.
   - Primary pre-registered validation criterion: Validation Macro-F1.
   - Secondary criterion: Validation Cohen's $\kappa$.
3. **Outer Test Isolation**:
   - Outer test subjects are completely isolated from the training process.
   - Outer test DataLoader is instantiated and evaluated strictly ONCE per run at locked epoch 80.
   - Zero outer-test metrics are used for early stopping, model selection, or hyperparameter adjustment.

---

## 5. Pre-Registered Metrics & Statistical Plan

### Primary Generalization Metrics:
1. **Macro-F1** (unweighted mean across 4 classes)
2. **Cohen's $\kappa$**

### Secondary Metrics:
- Overall Accuracy (%)
- Matthews Correlation Coefficient (MCC)
- Weighted-F1
- Expected Calibration Error (ECE)
- Per-class F1, Precision, and Recall (Wake, Light, Deep, REM)
- Boundary vs. Non-boundary Error Rates (using verified 1-epoch arrival definition)

### Planned Statistical Comparisons across 47 Subjects:
1. **Model E vs. Model A**: Full proposed architecture vs. matched-capacity baseline.
2. **Model E vs. Model B**: Domain-adversarial subject-invariant training attribution.
3. **Model E vs. Model C**: Transition-modeling attribution across subject folds.
4. **Model E vs. Model D**: Adaptive reliability gating attribution across subject folds.

*All headline comparisons will report paired subject-level differences ($\Delta$) and paired cluster bootstrap 95% confidence intervals across all 47 subjects.*

---

## 6. Deliverables & Artifact Manifest

Upon completion of authorized EXP-014 training:
1. `EXP014_RESULTS.csv`: Seed-level and fold-level summary metrics for all 75 runs.
2. `EXP014_SUBJECT_RESULTS.csv`: Granular performance across all 47 subjects.
3. `EXP014_PREDICTIONS/`: Serialized prediction arrays for all outer test windows.
4. `EXP014_CV_REPORT.md`: Comprehensive cross-validation findings and attribution report.
5. `EXP014_ERROR_ANALYSIS.md`: Confusion matrices, transition error analysis, and boundary error rates.
6. `EXP014_EFFICIENCY.md`: Benchmarked GPU/CPU latency, VRAM, and parameter counts.
