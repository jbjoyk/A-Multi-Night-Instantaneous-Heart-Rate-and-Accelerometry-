# Handoff Report: Independent Review & Adversarial Stress-Test of Evaluation Pipeline (NAS Iteration 1)

**Agent**: `teamwork_reviewer_r1_2`  
**Roles**: reviewer, critic  
**Target Path**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_reviewer_r1_2/handoff.md`  
**Date**: 2026-11-15T03:55:00Z  
**Verdict**: **`APPROVE`**  

---

## Review Summary

**Verdict**: **`APPROVE`**  

The evaluation pipeline implemented in `experiments/EXP-014/evaluate_nas_iteration1.py`, the mutated model architecture `MultiScaleConventionalCNNBiGRU` (`MutatedModelA_MSCNNBiGRU`) in `experiments/EXP-014/nas_iteration1_model.py`, and the Fold 1 manifest in `experiments/EXP-014/EXP014_MANIFESTS/fold_1.json` satisfy all acceptance criteria defined in `ORIGINAL_REQUEST.md` and `SCOPE.md`.

Key audit findings:
1. **Hyperparameter Invariance**: Batch size = 4, optimizer = Adam, learning rate = 1e-4, weight decay = 0.0, sequence length = 600, loss = CrossEntropy + 0.01 * RCCLoss are strictly preserved without alteration.
2. **Subject Isolation & Dataset Compliance**: Fold 1 manifest strictly partitions 32 inner-train subjects (168 nights) and 10 outer-test subjects (54 nights, 32,098 evaluated 30-second windows). Verified zero subject leakage ($\text{outer\_test} \cap \text{inner\_train} = \emptyset$, $\text{outer\_test} \cap \text{outer\_train} = \emptyset$).
3. **Metric Integrity**: 4-class Macro-F1 (Wake, Light, Deep, REM) is correctly computed over masked valid chronological epochs (`valid = (targets >= 0) & (targets < 4) & mask`).
4. **Parameter Cap Compliance**: Baseline Model A has 163,940 parameters; mutated model has 192,692 parameters ($+28,752$, $+17.54\% \le 20.00\%$), leaving a 4,036-parameter margin under the 196,728 ceiling.
5. **Execution & Integrity Audit**: Clean execution on single GPU `cuda:0` via `./.venv/bin/python`. Zero integrity violations (no hardcoded outputs, genuine forward/backward gradient flow across all 192,692 parameters, real GPU evaluation).
6. **Findings**: One Major finding regarding unseeded model weight initialization in `evaluate_nas_iteration1.py` causing 1-epoch metric variance, and two Minor architectural observations.

---

## 1. Observation

### 1.1 Hyperparameter Invariance Audit
Direct inspection of `experiments/EXP-014/evaluate_nas_iteration1.py`:
- **Batch size**: Fixed at $4$ (`batch_size=4` at lines 52, 83, 84, 203).
- **Optimizer**: `torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)` (line 106).
- **Learning rate**: $10^{-4}$ (`lr=1e-4` at line 94).
- **Weight decay**: $0.0$ (line 106).
- **Sequence horizon**: $600$ epochs ($300$ minutes / $5$ hours at 30s epochs; `seq_len=600` at lines 52, 80, 81, 203).
- **Loss function**: Composite CrossEntropy + $0.01 \times \text{RCCLoss}$ (`nas_iteration1_model.py` lines 301–310 and `exp013_models.py` lines 153–162). Matches `run_exp013_attribution.py` and `sleep_adaptive_net.py`.

### 1.2 Fold 1 Partition and Subject Isolation Audit
Direct inspection of `experiments/EXP-014/EXP014_MANIFESTS/fold_1.json` and verification via Python set operations:
- **Inner-train partition**: 32 subjects (`Bidslab00`, `01`, `02`, `06`, `07`, `10`, `11`, `13`, `14`, `15`, `18`, `19`, `20`, `22`, `31`, `32`, `34`, `35`, `40`, `42`, `43`, `44`, `49`, `50`, `51`, `52`, `53`, `56`, `60`, `62`, `63`, `66`), yielding 168 recording nights (42 batches of size 4).
- **Inner-validation partition**: 5 subjects (`Bidslab09`, `16`, `17`, `38`, `55`), yielding 24 recording nights.
- **Outer-train partition**: 37 subjects (192 recording nights, exactly inner-train + inner-val).
- **Outer-test partition**: 10 subjects (`Bidslab08`, `30`, `36`, `39`, `41`, `45`, `47`, `64`, `65`, `68`), yielding 54 recording nights (14 batches: 13 batches of 4 + 1 batch of 2).
- **Leakage check**:
  ```python
  set(outer_test_subjects).intersection(set(inner_train_subjects)) == set() # True
  set(outer_test_subjects).intersection(set(outer_train_subjects)) == set() # True
  ```
  Result: **0 subject overlap (leakage-free)**.

### 1.3 Metric Calculation Verification
In `evaluate_nas_iteration1.py` (lines 155–175):
```python
valid = (targets >= 0) & (targets < 4) & mask
all_preds.extend(preds[valid].cpu().numpy())
all_targets.extend(targets[valid].cpu().numpy())
...
acc = float(accuracy_score(y_true, y_pred))
macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
kappa = float(cohen_kappa_score(y_true, y_pred))
prec, rec, f1_cls, supp = precision_recall_fscore_support(
    y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0
)
```
- Invalid targets (such as padding tokens $-100$ and unlabeled epochs) and padded sequence tails (`mask == False`) are strictly masked out.
- Exactly 32,098 valid 30-second windows are evaluated across the 54 test nights.
- Four sleep stages (Wake=0, Light=1, Deep=2, REM=3) are evaluated with per-class F1, and `macro_f1` is the exact unweighted mean of the four classes.

### 1.4 Independent Single GPU (`cuda:0`) Execution
Ran `./.venv/bin/python experiments/EXP-014/evaluate_nas_iteration1.py`:
- Device: `cuda:0` (NVIDIA GeForce RTX 3060, memory consumption peak ~850 MiB).
- Verbatim terminal output (clean run):
  ```
  Forward Pass Verification on Expected BIDSleep Shapes (B=4, T=600, L=30):
    Logits shape: torch.Size([4, 600, 4]) (expected: (4, 600, 4))
    Total loss: 1.5347
    Forward pass verification: PASSED

  Executing 1-Epoch Training and Fold 1 Outer Test Evaluation (Mutated Model)...
    1-Epoch Train Loss: 1.3431 (took 1.89s)
    Outer Test Windows: 32,098 (eval took 0.24s)
    Overall Accuracy:   46.97%
    Macro-F1 (4-class): 0.3285
    Cohen's Kappa:      0.1512
    Per-class F1:
      - Wake_f1: 0.2320
      - Light_f1: 0.6020
      - Deep_f1: 0.4303
      - REM_f1: 0.0496

  Executing 1-Epoch Training and Fold 1 Outer Test Evaluation (Baseline Model A)...
    1-Epoch Train Loss: 1.3637 (took 1.76s)
    Outer Test Windows: 32,098 (eval took 0.23s)
    Overall Accuracy:   43.19%
    Macro-F1 (4-class): 0.2882
    Cohen's Kappa:      0.1146
    Per-class F1:
      - Wake_f1: 0.0167
      - Light_f1: 0.5909
      - Deep_f1: 0.3595
      - REM_f1: 0.1860

  NAS Iteration 1 Summary Comparison (1 Epoch on Fold 1):
    Parameters: Baseline = 163,940 | Mutated = 192,692 (Delta: +28,752, +17.54%)
    Train Loss: Baseline = 1.3637 | Mutated = 1.3431 (Delta: -0.0206)
    Accuracy:   Baseline = 43.19% | Mutated = 46.97% (Delta: +3.78 pp)
    Macro-F1:   Baseline = 0.2882 | Mutated = 0.3285 (Delta: +0.0402)
    Kappa:      Baseline = 0.1146 | Mutated = 0.1512 (Delta: +0.0366)
  ```

### 1.5 Adversarial Multi-Seed Stress Testing
To stress-test reproducibility and evaluate whether the 1-epoch evaluation loop reliably favors the mutated model, we executed a controlled experiment with synchronized seeding across 4 seeds ($s \in \{42, 100, 2024, 777\}$), setting seeds BEFORE model weight initialization:
- **Seed 42**: Baseline F1 = 0.3443 (Loss = 1.3341) | Mutated F1 = 0.3676 (Loss = 1.3317) | $\Delta \text{F1} = \mathbf{+0.0233}$
- **Seed 100**: Baseline F1 = 0.3340 (Loss = 1.2931) | Mutated F1 = 0.3274 (Loss = 1.3081) | $\Delta \text{F1} = \mathbf{-0.0066}$
- **Seed 2024**: Baseline F1 = 0.3643 (Loss = 1.3154) | Mutated F1 = 0.3571 (Loss = 1.3058) | $\Delta \text{F1} = \mathbf{-0.0072}$
- **Seed 777**: Baseline F1 = 0.3546 (Loss = 1.3173) | Mutated F1 = 0.3373 (Loss = 1.3243) | $\Delta \text{F1} = \mathbf{-0.0173}$
- **4-Seed Mean**: Baseline F1 = $0.3493$ | Mutated F1 = $0.3474$ | Mean $\Delta \text{F1} = -0.0020$ ($-0.20$ pp).

---

## 2. Logic Chain

1. **Baseline and Mutated Parameter Budget (Step 1 $\to$ Observation 1.1)**:
   Baseline Model A (`ConventionalCNNBiGRU`) has 163,940 parameters. The $+20\%$ ceiling is $\lfloor 163,940 \times 1.20 \rfloor = 196,728$ parameters.
   The mutated model (`MultiScaleConventionalCNNBiGRU` with pre-recurrent fusion) contains exactly 192,692 parameters ($+28,752$, $+17.54\%$), adhering to the cap with 4,036 parameters of headroom.

2. **Hyperparameter Integrity (Step 2 $\to$ Observation 1.1)**:
   Verification confirms that batch size ($4$), optimizer (`torch.optim.Adam`), learning rate ($10^{-4}$), weight decay ($0.0$), sequence length ($600$), and loss formulation ($\text{CE} + 0.01 \times \text{RCCLoss}$) match the established baseline without variation.

3. **Partition Integrity & Zero Leakage (Step 3 $\to$ Observation 1.2)**:
   Fold 1 has 168 inner-train nights and 54 outer-test nights. Cross-referencing subject IDs confirmed 0 subject overlap between outer test and inner train, validating that evaluation is strictly out-of-subject.

4. **Metric Integrity (Step 4 $\to$ Observation 1.3)**:
   Macro-F1 is evaluated across the 4 stages using standard scikit-learn metrics with invalid classes and padding excluded. The metric calculation logic is correct.

5. **Local Single GPU Execution (Step 5 $\to$ Observation 1.4)**:
   The evaluation script runs successfully on `cuda:0` using `./.venv/bin/python`, verifying forward tensor safety, loss backpropagation, evaluation loop execution, and JSON persistence.

6. **Integrity Check (Step 6 $\to$ Observations 1.1–1.5)**:
   Active checking confirmed:
   - No hardcoded test results in source code or evaluation loop.
   - Genuine `nn.Module` implementation with full backward gradient flow.
   - Real dataset cache loaded from `cache/prep_57738d534b`.
   - Real GPU training and evaluation passes.
   - Zero evidence of cheating or integrity violations.

7. **Root Cause Analysis of 1-Epoch Metric Variance (Step 7 $\to$ Observation 1.5)**:
   In `evaluate_nas_iteration1.py`, models are initialized before `torch.manual_seed()` is set inside `evaluate_1epoch_fold1()`. Because 1 epoch consists of only 42 gradient steps, metric scores after 1 epoch are sensitive to the initial random weights. This explains the variance observed between worker 1's report (+0.0501) and our independent runs (+0.0402, +0.0233, -0.0072). This is a methodological finding, not an integrity violation.

---

## 3. Findings

### [Major] Finding 1: Unseeded Model Instantiation in `evaluate_nas_iteration1.py`
- **What**: In `evaluate_nas_iteration1.py` (lines 209–215), `baseline_model` and `mutated_model` are instantiated prior to calling `evaluate_1epoch_fold1()`. The `torch.manual_seed(seed)` call occurs inside `evaluate_1epoch_fold1()`, after weights have already been drawn from an unseeded process state.
- **Where**: `experiments/EXP-014/evaluate_nas_iteration1.py`, lines 209, 215, 248.
- **Why**: PyTorch parameters are randomized at instantiation. In a 42-step training loop, unseeded initial weights cause run-to-run Macro-F1 delta fluctuations (ranging from $-0.0173$ to $+0.0402$).
- **Remedy / Recommendation**: In future evaluation scripts and NAS iterations, call `torch.manual_seed(seed)`, `np.random.seed(seed)`, and `torch.cuda.manual_seed_all(seed)` at the very beginning of the evaluation script before initializing any models.

### [Minor] Finding 2: Receptive Field Zero-Padding Boundary Handling
- **What**: Parallel depthwise convolutions in `MultiScaleTemporalFusion` use symmetric zero-padding (`padding = k // 2`).
- **Where**: `experiments/EXP-014/nas_iteration1_model.py`, line 84.
- **Why**: At sequence boundaries or padded night tails, symmetric padding introduces zero values into the convolution window.
- **Impact**: Minimal, because night sequences are long (~600 epochs), residual identity connections provide signal stability, and loss computation masks padded tokens.
- **Suggestion**: Consider causal padding or reflection padding in later iterations.

### [Minor] Finding 3: Pointwise Projection Parameter Dominance
- **What**: 27,888 out of the 28,752 added parameters ($96.9\%$) are consumed by the input ($144 \to 96$) and output ($96 \to 144$) pointwise $1 \times 1$ projection layers, while the multi-scale depthwise convolutions use only 576 parameters.
- **Where**: `experiments/EXP-014/nas_iteration1_model.py`, lines 75, 94.
- **Suggestion**: For subsequent iterations, exploring grouped channel projections or intra-encoder mutations (such as `MultiScaleTemporalConvBlock`, which adds only 9,344 parameters) will afford greater parameter budget flexibility for deeper temporal receptive fields.

---

## 4. Adversarial Challenge & Stress-Testing

### Challenge 1: 1-Epoch Macro-F1 Superiority Assumption
- **Assumption challenged**: That the mutated model definitively outperforms baseline Model A by $+5.01$ pp in 1 epoch on Fold 1.
- **Attack scenario**: Evaluated 4 different random seeds with synchronized initialization and DataLoader shuffling.
- **Result**: Under seeds 100, 2024, and 777, baseline Model A achieved comparable or slightly higher 1-epoch Macro-F1 (average delta across 4 seeds: $-0.0020$).
- **Mitigation**: Acknowledge that a 1-epoch evaluation loop evaluates model viability, gradient propagation, and initial inductive bias rather than asymptotic performance. Asymptotic convergence across 80 epochs should be used for definitive ranking in later milestones.

### Challenge 2: Fold 1 Subject Leakage
- **Assumption challenged**: That subjects in Fold 1 test partition are strictly absent from training.
- **Attack scenario**: Cross-referenced all 10 test subjects against all 32 inner-train and 5 inner-val subjects.
- **Result**: Exactly zero overlap. Subject isolation is 100% verified.

### Challenge 3: Hyperparameter Drift
- **Assumption challenged**: That batch size, optimizer, learning rate, or loss weighting were quietly adjusted to favor the mutation.
- **Attack scenario**: Line-by-line verification against `exp013_models.py` and `run_exp013_attribution.py`.
- **Result**: No drift. All hyperparameters are identical.

---

## 5. Caveats

1. **1-Epoch Convergence Limitation**: A 1-epoch loop on 168 nights (42 optimization steps) cannot produce asymptotic convergence (~0.66–0.74 Macro-F1). The results reflect initialization behavior and immediate gradient response.
2. **Single Fold Scope**: Scoped strictly to Fold 1 on single GPU `cuda:0` per milestone requirements. Full 5-fold cross-validation remains for downstream phases.
3. **No Code Modification Constraint**: In accordance with the Reviewer role constraints, all code changes remain within worker scope; findings and remedies are documented here.

---

## 6. Conclusion

1. **Verdict**: **`APPROVE`**
2. **Acceptance Criteria Verification**:
   - [x] Mutated architecture implements an isolated structural mutation grounded in hypothesis H-A from `RESEARCH_DIRECTION_v3_FULL.md`.
   - [x] Total parameter count does not exceed 196,728 parameters (mutated: 192,692, $+17.54\% \le 20.00\%$).
   - [x] All tensor operations and dimension transitions are dimension-compatible, with projection layers implemented.
   - [x] Forward pass executes without shape or runtime errors on expected BIDSleep input tensor dimensions.
   - [x] 1-epoch evaluation loop completes successfully on Fold 1 (`experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`) on single GPU `cuda:0` using `./.venv/bin/python`.
   - [x] Evaluation script logs the Macro-F1 score without altering batch size, optimizer, or learning rate.
   - [x] Zero integrity violations detected (no hardcoding, no mock implementations, real GPU training/eval).

---

## 7. Verification Method

To independently verify this evaluation:

1. **Execute Unit Test Suite**:
   ```bash
   ./.venv/bin/python -m pytest .agents/teamwork/teamwork_worker_r1_1/test_nas_iteration1.py -v
   ```
   *Expected*: 7 passed in ~1.5s.

2. **Execute Evaluation Script on Single GPU `cuda:0`**:
   ```bash
   ./.venv/bin/python experiments/EXP-014/evaluate_nas_iteration1.py
   ```
   *Expected*:
   - Validates baseline parameters: 163,940.
   - Validates mutated parameters: 192,692 ($\le 196,728$).
   - Forward pass verification: PASSED.
   - 1-epoch training and evaluation on Fold 1 completes in ~4.5s.
   - Exports JSON to `experiments/EXP-014/nas_iteration1_eval_results.json`.

3. **Verify Fold 1 Subject Isolation**:
   ```bash
   ./.venv/bin/python -c '
   import json
   with open("experiments/EXP-014/EXP014_MANIFESTS/fold_1.json") as f:
       m = json.load(f)
   assert set(m["outer_test_subjects"]).isdisjoint(m["inner_train_subjects"])
   assert set(m["outer_test_subjects"]).isdisjoint(m["outer_train_subjects"])
   print("Subject isolation verified: 0 leakage.")
   '
   ```
