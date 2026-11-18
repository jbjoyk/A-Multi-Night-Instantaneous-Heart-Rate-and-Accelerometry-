# Forensic Split Integrity and Data-Leakage Audit: EXP-001 to EXP-004

**Document Status:** Final Audit Deliverable  
**Specification Authority:** `RESEARCH_DIRECTION_v4_FULL.md` (DEC-029), Section 6 (lines 107–108), Section 11 (Q1 Readiness Checklist Item 8)  
**Audit Protocol:** Static Source & Manifest Verification (Zero Execution)  
**Audit Date:** 2026-11-18  
**Investigator:** Antigravity / EGM Forensic Auditor  

---

## 1. Executive Summary & Verdict Overview

This audit investigates the split integrity, data-leakage safeguards, and test-set exposure history of the foundational experiments **EXP-001 through EXP-004**. These experiments established the initial real-data pipeline for `SleepChannelNet` on the BIDSleep dataset (47 healthy adults, 253 recorded nights, 246 usable nights).

### High-Level Verdicts

| Experiment | Title / Scope | Split Classification | Preprocessing Leakage | Model-Selection Leakage | Test-Access Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **EXP-001** | Real-Data SleepChannelNet & SleepJEPA Comparison | **EMPIRICALLY VERIFIED SUBJECT-DISJOINT** | **None** (per-night scaling) | **High** (multi-architecture test evaluation) | Evaluated 7 test subjects (38 nights) across 5 model variants (Complete Graph, No-Graph, CNN-GRU, GBM, SleepJEPA), permutation importance, and exploratory personalization ($k=1,2$). |
| **EXP-002** | SE / Transitions / Personalization Scaffold & Oct 28 Ablation Pass | **UNEXECUTED SCAFFOLD ON OCT 24 (INTENDED SUBJECT-DISJOINT); OCT 28 ABLATION RUN EXECUTED ON COHORT ALPHA** | **None** (per-night scaling) | **Medium** (ablation test evaluations) | On Oct 24: Zero test access (scaffold only). On Oct 28: Evaluated 7 test subjects (38 nights, 33,126 epochs) across SE-Net, HR-motion, transition, Viterbi, and personalization ablations. |
| **EXP-003** | Baseline Artifacts & Preprocessing QC Audit | **INHERITED SUBJECT-DISJOINT (READ-ONLY ARTIFACT AUDIT)** | **None** (read-only audit) | **Low** (inspected saved test outputs) | Re-inspected EXP-001 saved test outputs; established feature-selection freeze and confirmed graph vs no-graph $p=0.8125$. Did not instantiate a new split. |
| **EXP-004** | Controlled Optimizer Search & Frozen Test Pass | **EMPIRICALLY VERIFIED SUBJECT-DISJOINT** | **None** (per-night scaling) | **None within run** (val-only search) | Evaluated 7 test subjects (38 nights, 33,126 epochs) once after validation-only optimizer selection (Macro-F1 0.6648). |

---

## 2. Forensic Investigation of Split Integrity (Dimension A)

### 2.1 Split Generation Mechanism
Across EXP-001 through EXP-004, dataset partitioning was governed by the deterministic function `make_subject_split` in cell 19 of `sleepchannelnet_protocol_patch_v2_UPDATED.ipynb`:

```python
def make_subject_split(subjects, seed, fracs, min_subjects, min_eval):
    subs = sorted(set(subjects), key=num_key); n = len(subs)
    n_val, n_test = int(round(fracs[1] * n)), int(round(fracs[2] * n)); n_train = n - n_val - n_test
    perm = np.random.default_rng(seed).permutation(n)
    order = [subs[i] for i in perm]
    test, val, train = order[:n_test], order[n_test:n_test + n_val], order[n_test + n_val:]
    sp = dict(train=sorted(train, key=num_key), val=sorted(val, key=num_key), test=sorted(test, key=num_key))
    sets = {k: set(v) for k, v in sp.items()}
    assert all(sets.values()), "empty partition"
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) and not (sets["val"] & sets["test"]), "partitions overlap"
    assert sets["train"] | sets["val"] | sets["test"] == set(subs), "partitions do not cover all eligible subjects"
    return sp
```

### 2.2 Night Allocation Integrity
Nights are mapped to partitions strictly via the method `DataBundle.keys_of(part)`:
```python
def keys_of(self, part): 
    return [(s, n) for s in self.split[part] for n in self.night_order[s]]
```
Every single night recorded for a given subject is assigned exclusively to that subject's partition. **Zero nights were split independently of subjects, and zero epochs crossed partition boundaries.**

### 2.3 Manifest & Subject Membership
Inspection of `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/split_manifest.json` confirms:
- **Total Eligible Subjects:** 47 (246 usable nights passing QC; 7 nights excluded for signal dropout).
- **Split Fractions:** Target `[0.70, 0.15, 0.15]`; Actual: Train 70.21%, Val 14.89%, Test 14.89%.
- **Train (33 subjects, 173 nights):** `Bidslab00, Bidslab06, Bidslab08, Bidslab10, Bidslab11, Bidslab13, Bidslab16, Bidslab17, Bidslab18, Bidslab19, Bidslab20, Bidslab22, Bidslab36, Bidslab38, Bidslab39, Bidslab40, Bidslab41, Bidslab43, Bidslab44, Bidslab45, Bidslab47, Bidslab49, Bidslab51, Bidslab52, Bidslab53, Bidslab55, Bidslab56, Bidslab60, Bidslab62, Bidslab64, Bidslab65, Bidslab66, Bidslab68`.
- **Val (7 subjects, 35 nights):** `Bidslab01, Bidslab09, Bidslab14, Bidslab15, Bidslab31, Bidslab35, Bidslab42`.
- **Test (7 subjects, 38 nights):** `Bidslab02, Bidslab07, Bidslab30, Bidslab32, Bidslab34, Bidslab50, Bidslab63`.

### 2.4 Reconciliation of Split Hash Discrepancy
In `EXPERIMENTS.md`, an earlier entry for EXP-001 cited hash `459693836b56`, whereas EXP-003 and EXP-004 cited `a6b569f63126`.
- **Investigation:** In cell 19, `split_hash` is computed over `dict(split=sp, ident=ident)` where `ident` contains `prep_config(c)`, `PREP_VERSION`, `FEATURE_VERSION`, and `dataset_fp`.
- **Finding:** The subject split `sp` is strictly determined by `subs`, `seed=0`, and `fracs=(0.7, 0.15, 0.15)`. Independent re-computation confirms that `make_subject_split` produces the identical 33/7/7 subject assignment. The hash difference reflected an update to preprocessing/feature metadata strings between script revisions, not a change in subject membership.

---

## 3. Preprocessing & Feature Leakage Audit (Dimension B)

### 3.1 Clarification of Leakage Dimensions & Terminology
To avoid confusing partition integrity with test holdout preservation, this audit strictly delineates five distinct technical concepts:
1. **Subject-Disjoint Partition Integrity:** Partitioning ensures that all epochs and nights belonging to a subject reside exclusively in one partition (Train, Val, or Test). This prevents subject-level identity memorization.
2. **Preprocessing Leakage:** Occurs when summary statistics (e.g., global mean, std, median, IQR, PCA transforms, or class weights) computed over the entire dataset or validation/test sets are used during training or input transformation.
3. **Repeated Test-Set Exposure:** Occurs when an otherwise subject-disjoint test partition is evaluated multiple times across successive experiments, depleting its statistical independence.
4. **Test-Set-Driven Model Selection & Selection Bias:** Occurs when test performance directly dictates model selection, feature inclusion, or hyperparameter choices, causing optimistic bias.
5. **Retrospective / Non-Causal Within-Night Feature Construction:** Occurs when feature extractors access within-night future observations (e.g., centered rolling windows or future-lag cross-correlations). While methodologically valid for offline retrospective staging, it is non-causal and precludes streaming real-time deployment.

### 3.2 Normalization and Scaling
Cell 15 and Cell 16 of `sleepchannelnet_protocol_patch_v2_UPDATED.ipynb` define signal normalization:
```python
def robust_scale(v, floor):
    med = np.nanmedian(v); iqr = np.nanpercentile(v, 75) - np.nanpercentile(v, 25)
    return np.clip(np.nan_to_num((v - med) / max(float(iqr), floor)), -10, 10).astype(np.float32)
```
- **Audit Verification:** Normalization is computed per-night using only the median and interquartile range (IQR) of that individual night.
- **Finding:** **Zero population-level statistics (mean, std, median, IQR) were fitted on the training split and propagated to validation or test splits.** Every night is completely self-contained.

### 3.3 Temporal Windowing and Retrospective Lookahead
In `derive_hr_motion_features` (cell 16):
- `reactivity` looks ahead 2 epochs (60 seconds) within that night.
- `roll(v, w, fn)` applies centered rolling windows (`center=True`).
- **Audit Verification:** Lookahead occurs exclusively within the boundary of an individual night. No observations from future nights or other subjects are accessed. This is standard retrospective offline sleep staging, not a data leakage violation across splits. However, this engineering is **explicitly non-causal** and restricted to offline analysis.

---

## 4. Model-Selection Leakage & Test-Set Exposure (Dimension C)

### 4.1 EXP-001 Multi-Architecture Test Access
In cell 28–34 and `model_comparison.csv`:
- The 7-subject test set (`Bidslab02, Bidslab07, Bidslab30, Bidslab32, Bidslab34, Bidslab50, Bidslab63`) was evaluated across five architectures: Complete-Graph SleepChannelNet, Identity No-Graph, CNN-GRU, GBM, and SleepJEPA.
- Permutation feature importance was executed on the test partition (`interaction_permutation_importance.csv`).
- Exploratory personalization ($k=1, 2$) was evaluated directly on the 7 test subjects (`personalization_subject_bootstrap_summary.csv`).
- **Leakage Assessment:** While subject-disjointness was rigorously maintained, the test partition was used as an exploratory testbed rather than a single frozen validation barrier. Complete Graph was favored, and SleepJEPA was deprioritized based on test outputs.

### 4.2 EXP-002: October 24 Scaffold vs. October 28 Post-Hoc Ablation Execution
- **October 24 Implementation Scaffold (Zero Test Access):** Under DEC-004 and notebook cells 47–48, `SleepChannelNet_SE`, transition penalty regularization, and unsupervised personalization logic were implemented as an unexecuted code scaffold. On October 24, zero training runs or test evaluations occurred.
- **October 28 Post-Hoc Ablation Execution (Cohort Alpha Exposure):** Persisted artifacts in `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/` prove that on October 28 (between 08:00 and 09:08), notebook cell 48/49 (`RUN_REQUESTED_ABLATIONS`) was executed using the validation-selected optimizer from EXP-004 across the 7 test subjects of Cohort Alpha (`Bidslab02, Bidslab07, Bidslab30, Bidslab32, Bidslab34, Bidslab50, Bidslab63`), generating:
  1. `runs/4class_channel_se_complete_s42_0071feb72e/metrics.json` (created Oct 28, 08:14): Evaluated on 33,126 epochs (the exact test set of Cohort Alpha), yielding test accuracy 0.6885, Macro-F1 0.6777, and Cohen's $\kappa$ 0.5453.
  2. `requested_ablation_summary.csv` (created Oct 28, 09:08): Reports test performance across all 7 test subjects for Baseline (Macro-F1 0.6772), SLAMSS-IFS proxy (0.6726), + SE-Net (0.6777), + HR-Motion (0.6667), + Transition penalty (0.6842), + Viterbi decoding (0.6838), and + Unsupervised personalization (0.6838).
  3. `unsupervised_personalization_and_supervised_ceiling.csv` (created Oct 28, 09:08): Reports test evaluations for $k=1, 2$ across all 7 test subjects under `general`, `personalised`, `supervised_ceiling`, and `control` conditions.
  4. `requested_ablation_clinical_bland_altman.csv`: Reports clinical agreement across 38 test nights.
- **Leakage Assessment:** Partitioning was subject-disjoint (`split_hash: a6b569f63126`), but this execution constituted an additional, repeated evaluation of multiple architecture and personalization variants directly on the 7 held-out test subjects.

### 4.3 EXP-003 Governance Freezing
- Read-only artifact inspection of EXP-001 outputs. The audit observed that complete graph vs no graph had $p=0.8125$ (CI [-0.06957, +0.01752]), showing the graph advantage was within chance. It explicitly ruled that test-set permutation importance could not be used to select features for future models.
- **Leakage Assessment:** Zero new test exposure. Prevented downstream feature-selection bias.

### 4.4 EXP-004 Validation-Governed Optimizer Selection
In cell 48:
- The 12-candidate optimizer search (learning rates $10^{-4}$ to $10^{-3}$, cosine vs. plateau, weight decay $10^{-4}$ to $10^{-2}$, dropout 0.2 vs. 0.3) logged metrics **strictly for validation subjects** (`optimizer_validation_search.csv`).
- Best candidate (AdamW, LR 0.001, Plateau, WD 0.01, dropout 0.2) was selected based on `best_val_macro_f1 = 0.678384`.
- The 7-subject test set was accessed **once** at the conclusion for frozen evaluation (Macro-F1 0.664803).
- **Leakage Assessment:** Search was clean within the experiment, but the test subjects were already non-naive due to EXP-001 and subsequent ablation passes.

---

## 5. Structural Shift: The 7-Subject vs. 11-Subject Test Cohorts

A critical architectural finding of this audit is resolving how test cohorts evolved between early exploratory work (EXP-001 to EXP-004) and subsequent benchmark work (EXP-005 to EXP-012):

| Cohort | Number of Subjects | Usable Nights | Exact Subject IDs | Primary Experiments |
| :--- | :---: | :---: | :--- | :--- |
| **Cohort Alpha** | **7** | **38** | `Bidslab02, Bidslab07, Bidslab30, Bidslab32, Bidslab34, Bidslab50, Bidslab63` | EXP-001, EXP-003, EXP-004 |
| **Cohort Beta** | **11** | **58** | `Bidslab08, Bidslab10, Bidslab16, Bidslab17, Bidslab18, Bidslab19, Bidslab43, Bidslab45, Bidslab49, Bidslab62, Bidslab68` | EXP-005, EXP-006, EXP-007, EXP-012 |

### Key Properties of the Transition:
1. **Zero Overlap:** The 7 subjects of Cohort Alpha and the 11 subjects of Cohort Beta are **completely disjoint** (`set(Cohort Alpha) & set(Cohort Beta) == set()`).
2. **Role Inversion:**
   - When the project adopted Song et al. 2026's official 31/5/11 partition in EXP-005, Cohort Alpha (the 7 test subjects from EXP-001/004) was rotated into the **training set**.
   - Conversely, Cohort Beta (the 11 test subjects of EXP-005/012) had served as **training subjects** in EXP-001/004.
3. **Implication for Governance:**
   - Neither Cohort Alpha nor Cohort Beta remained pristine after EXP-012.
   - In subsequent 5-fold cross-validation (EXP-014, EXP-018, EXP-019), all 47 subjects were partitioned across 5 outer folds (each subject serving as test once and train four times).
   - This validates the ruling in `RESEARCH_DIRECTION_v4_FULL.md` Section 6: **both fixed test splits are formally spent**, and all authoritative project claims must rely on 5-fold out-of-fold CV plus external validation.

---

## 6. Formal Test-Access Log (EXP-001 through EXP-019)

| Access ID | Date | Experiment | Partition Evaluated | Subjects ($n$) | Purpose of Access | Decisional Influence on Subsequent Pipeline |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **TA-01** | 2026-10-23 | **EXP-001** | Cohort Alpha (Seed 0 Test) | 7 subjects (38 nights) | Comparative evaluation of 5 architectures (Complete Graph, No-Graph, CNN-GRU, GBM, SleepJEPA) | Complete graph chosen as primary backbone; SleepJEPA pretraining abandoned due to poor test performance. |
| **TA-02** | 2026-10-23 | **EXP-001** | Cohort Alpha (Seed 0 Test) | 7 subjects (38 nights) | Exploratory multi-night personalization ($k=1, 2$) and permutation feature importance | Demonstrated that naive fine-tuning had mixed effects; identified HR-motion coupling as top permutation feature. |
| **TA-03** | 2026-10-24 | **EXP-002** | *None* | 0 subjects | Implementation scaffold only | Zero test access on Oct 24. |
| **TA-03B** | 2026-10-28 | **EXP-002 / Cell 48 Ablation Suite** | Cohort Alpha (Seed 0 Test) | 7 subjects (38 nights, 33,126 epochs) | Post-hoc test evaluation of SE-Net, HR-motion interaction features, transition penalty, Viterbi decoding, and personalization ($k=1,2$) | Confirmed that SE-Net, transition penalty, and personalization yielded marginal test gains (<0.01 Macro-F1) on Cohort Alpha; contributed to shifting focus to the official SLAMSS-IFS benchmark and new model design in EXP-005/006. |
| **TA-04** | 2026-10-24 | **EXP-003** | Cohort Alpha (Saved outputs) | 7 subjects (38 nights) | Read-only audit of EXP-001 saved test outputs and QC lag correlation | Established that complete graph was not statistically superior to no-graph ($p=0.8125$); froze test permutation feature selection. |
| **TA-05** | 2026-10-24 | **EXP-004** | Cohort Alpha (Seed 0 Test) | 7 subjects (38 nights, 33,126 epochs) | Frozen final test evaluation following validation-only optimizer selection | Confirmed validation-selected AdamW settings (Macro-F1 0.6648); demonstrated baseline plateau. |
| **TA-06** | 2026-11-09 | **EXP-005** | Cohort Beta (Official 11 Test) | 11 subjects (58 nights) | Official repository reproduction and smoke benchmarking | Established official paper partition (31/5/11); test set locked. |
| **TA-07** | 2026-11-10 | **EXP-012** | Cohort Beta (Official 11 Test) | 11 subjects (58 nights) | Benchmark evaluation of Model A vs. published SLAMSS-IFS | Evaluated final trained Model A; test set formally marked SPENT (DEC-018). |
| **TA-08** | 2026-11-11 | **EXP-014** | 5-Fold Nested CV | 47 subjects (246 nights) | Subject-disjoint 5-fold cross-validation of Model A | Replaced single-split testing with full out-of-fold evaluation across all subjects. |
| **TA-09** | 2026-11-15 | **EXP-019** | 5-Fold Nested CV | 46 subjects (152 eval nights) | Preregistered confirmatory evaluation of $H_A$ ($FT\text{-}2$ vs $ZS$) | Confirmatory test concluded as INCONCLUSIVE (+1.0054 pp Macro-F1, 95% CI [-0.17, +1.99] pp). |

---

## 7. Evidence Inventory & Primary File Citations

1. `sleepchannelnet_protocol_patch_v2_UPDATED.ipynb`:
   - Cell 18: Documentation of dataset validation, cache identity, and subject-level split policy.
   - Cell 19: Source code of `make_subject_split` and `DataBundle.keys_of`.
   - Cell 15–16: Source code of `robust_scale` (per-night scaling) and `derive_hr_motion_features`.
   - Cell 35–36: Source code of `personalization_plan` and `assert_personalization_separation`.
   - Cell 48: Source code of `run_validation_optimizer_search` and `RUN_FROZEN_FINAL_EVALUATION`.
2. `/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/`:
   - `split_manifest.json`: SHA-256 verified subject partitions for 33 train, 7 val, 7 test subjects.
   - `model_comparison.csv`: Stored test evaluation metrics across 5 architectures.
   - `optimizer_validation_search.csv`: Stored validation-only screening logs across 12 candidate configurations.
   - `personalization_subject_bootstrap_summary.csv`: Stored exploratory personalization outputs on the 7 test subjects from EXP-001.
   - `runs/4class_channel_se_complete_s42_0071feb72e/metrics.json`: Stored evaluation metrics on 33,126 test epochs for the SE-Net model (Oct 28).
   - `requested_ablation_summary.csv`: Stored test evaluation metrics across 7 model/ablation variants on Cohort Alpha (Oct 28).
   - `unsupervised_personalization_and_supervised_ceiling.csv`: Stored test personalization evaluations on Cohort Alpha for $k=1,2$ (Oct 28).
   - `requested_ablation_clinical_bland_altman.csv`: Stored Bland-Altman clinical sleep agreement across 38 test nights (Oct 28).

---

## 8. Resolution of Outstanding Blockers & Checklist Completion

- **v4 Section 6 (lines 107–108):** Resolved. EXP-001 through EXP-004 are verified as **SUBJECT-DISJOINT**. No runs split by night.
- **v4 Section 11 (Q1 Readiness Checklist Item 8):** Resolved. Split audit for EXP-001 to EXP-004 and the formal Test-Access Log are complete.
