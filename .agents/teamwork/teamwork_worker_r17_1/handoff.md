# Handoff Report: NAS Iteration 17 — Candidate Omicron Deployment & Fast Proxy Evaluation

- **Agent**: `teamwork_worker_r17_1` (Implementer / QA / Proxy Evaluation)
- **To**: Project Orchestrator Gen 2 (`5a45d55d-cdec-4610-8dcf-b95f2285b08e`)
- **Timestamp**: 2026-11-15T17:40:00Z
- **Working Directory**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r17_1`
- **Candidate Architecture**: `PriorCalibratedBoundedMarginPyramidCNNBiGRU` (Candidate Omicron, 196,382 parameters)
- **Deployment Script**: `experiments/EXP-014/evaluate_nas_iteration17_proxy.py`
- **Results Output**: `experiments/EXP-014/nas_iteration17_proxy_results.json`
- **Model Checkpoint**: `experiments/EXP-014/nas_iteration17_best.pt`
- **Verdict**: `PROXY_FAIL (REVERT)` (Outer-Test Acc: 66.54% < 70.00%, Weighted-F1: 0.6644 < 0.6900)

---

## 1. Observation

### 1.1 Execution Environment & Parameters
- **Script**: `experiments/EXP-014/evaluate_nas_iteration17_proxy.py` (454 lines, compiled and verified clean).
- **Architecture**: `PriorCalibratedBoundedMarginPyramidCNNBiGRU` imported from `nas_iteration17_model.py`.
- **Parameter Count**: Exactly 196,382 trainable parameters (strictly compliant with hard cap $\le 196,728$, 346 params headroom, 96.41% compression vs 5.47M benchmark).
- **Execution Command**: `./.venv/bin/python experiments/EXP-014/evaluate_nas_iteration17_proxy.py`
- **Hardware**: `cuda:0` (NVIDIA GeForce RTX 3060 12GB).
- **Runtime**: 102.91 seconds (~1.72 minutes) across 50 full epochs.
- **Data Partitions (Fold 1)**:
  - Inner-Train: 32 subjects, 168 nights (42 batches / epoch, batch size 4, seq_len 600).
  - Inner-Validation: 5 subjects, 24 nights (6 batches / epoch, 14,089 evaluated 30-s windows).
  - Outer-Test (Locked): 10 subjects, 54 nights (14 batches, 32,098 evaluated 30-s windows).
- **Optimizer & Loss**: Adam lr=1e-4, weight_decay=0.0, grad norm clipping max=1.0, multi-task CE + RCCLoss ($\lambda=0.01$).

### 1.2 Training Dynamics and Peak Inner-Validation Results
- **Peak Inner-Validation Epoch**: Epoch 35 / 50.
- **Peak Inner-Validation Accuracy**: **73.04%** (0.730428) — *All-time project record for inner-validation accuracy, surpassing Candidate Kappa's 73.00% at Epoch 45.*
- **Peak Inner-Validation Macro-F1**: **0.7391** (0.739073) (comparable to Candidate Kappa's 0.7395).
- **Peak Inner-Validation Weighted-F1**: 0.7304 (0.730380).
- **Peak Inner-Validation Cohen's Kappa**: 0.6109 (0.610886).
- **Inner-Validation Per-Class Metrics at Epoch 35**:
  - Wake:  F1 = 0.8250 | Recall = 78.46% | Precision = 86.99% (support: 1,866)
  - Light: F1 = 0.7353 | Recall = 77.71% | Precision = 69.78% (support: 5,968)
  - Deep:  F1 = 0.7046 | Recall = 66.70% | Precision = 74.66% (support: 2,871)
  - REM:   F1 = 0.6914 | Recall = 67.20% | Precision = 71.20% (support: 3,384)

### 1.3 Outer-Test Evaluation Metrics (Best Checkpoint, Epoch 35)
- **Evaluated Windows**: 32,098 windows (10 held-out test subjects, 54 nights).
- **Accuracy**: **66.54%** (0.665369) — Threshold $\ge 70.00\%$ $\to$ **FAILED (-3.46%)**
- **Weighted-F1**: **0.6644** (0.664362) — Threshold $\ge 0.6900$ $\to$ **FAILED (-0.0256)**
- **Macro-F1**: 0.6577 (0.657688).
- **Cohen's Kappa**: 0.5067 (0.506734).
- **Test Loss**: 0.8318.
- **Per-Class Metrics on Outer-Test (Best Checkpoint)**:
  - Wake:  F1 = 0.6493 | Recall = 60.11% (1,927 / 3,206) | Precision = 70.59%
  - Light: F1 = 0.6788 | Recall = 73.09% (10,133 / 13,864) | Precision = 63.36%
  - Deep:  F1 = 0.7056 | Recall = 67.09% (5,313 / 7,919) | Precision = 74.40%
  - REM:   F1 = 0.5971 | Recall = 56.04% (3,984 / 7,109) | Precision = 63.90%
- **Confusion Matrix on Outer-Test (Wake, Light, Deep, REM)**:
  ```
  [[ 1927,   977,    96,   206],   # Wake  (recall: 60.11%)
   [  602, 10133,  1377,  1752],   # Light (recall: 73.09%)
   [   88,  2225,  5313,   293],   # Deep  (recall: 67.09%)
   [  113,  2657,   355,  3984]]   # REM   (recall: 56.04%)
  ```

### 1.4 Final Checkpoint Evaluation Metrics (Epoch 50, Ablation Comparison)
- **Accuracy**: 66.59% (0.665898).
- **Weighted-F1**: 0.6646 (0.664630).
- **Macro-F1**: 0.6594 (0.659438).
- **Cohen's Kappa**: 0.5024 (0.502382).
- **Test Loss**: 0.8207.
- **Confusion Matrix at Epoch 50**:
  ```
  [[ 1836,  1057,   108,   205],
   [  416, 10482,  1234,  1732],
   [   35,  2539,  5092,   253],
   [   45,  2800,   300,  3964]]
  ```

### 1.5 Learned Calibration and Margin Values
- **$\tau_{\text{prior}}$ Dynamics**:
  - Initialized: $\alpha_{\text{prior}} = -0.4055 \implies \tau_{\text{prior}} = 0.2000$.
  - Best Checkpoint (Epoch 35): $\alpha_{\text{prior}} = -0.4257 \implies \tau_{\text{prior}} = 0.1976$.
  - Final Checkpoint (Epoch 50): $\alpha_{\text{prior}} = -0.4317 \implies \tau_{\text{prior}} = 0.1969$.
- **Learned Margins at Best Checkpoint (Epoch 35)**:
  - `tau_ihr_rem`: raw = -0.0747 | clamped effective = -0.0500 (**anti-drift cutoff strictly enforced via STE**)
  - `tau_acc_rem`: raw = 0.2733 | clamped effective = 0.2733
  - `tau_ihr_deep`: raw = 0.0297 | clamped effective = 0.0297
  - `tau_acc_deep`: raw = 0.0612 | clamped effective = 0.0612
  - `tau_ihr_wake`: raw = 0.0212 | clamped effective = 0.0212
  - `tau_acc_wake`: raw = -0.0062 | clamped effective = 0.0000
  - `tau_ihr_light`: raw = 0.1126 | clamped effective = 0.1126
  - `tau_acc_light`: raw = -0.0493 | clamped effective = 0.0000

---

## 2. Logic Chain

1. **Step 1: Hypothesis Grounding and Formulation Verification**:
   - Upstream hypothesis: Candidate Kappa's outer-test bottleneck was caused by majority Light sleep (43.19% of training windows) dominating the classifier, coupled with negative margin drift ($\tau_{\text{ihr\_rem}} = -0.0839$).
   - Candidate Omicron introduced two targeted mechanisms: Prior-Calibrated Logit Balancing (PLB) ($\tau_{\text{prior}} \in [0, 0.5]$ initialized at 0.20) and `BoundedMarginSTE` stability cutoffs ($\tau_{\text{acc}} \ge 0.05, \tau_{\text{ihr}} \ge -0.05$).
   - Total parameter count: 196,382 parameters (exactly $+1$ scalar $\alpha_{\text{prior}}$).

2. **Step 2: Margin Stability Verification**:
   - As observed in Section 1.5, `tau_ihr_rem` drifted to raw = -0.0747 at Epoch 35 and -0.0868 at Epoch 50.
   - Without `BoundedMarginSTE`, this would have allowed cardiac deceleration to falsely trigger REM synergy during non-REM epochs (as occurred in Candidate Kappa).
   - With `BoundedMarginSTE`, the effective margin was clamped at exactly -0.0500 throughout forward passes while backpropagation gradient flow was 100% maintained.

3. **Step 3: Inner-Validation Surge**:
   - On inner-validation (5 subjects, 24 nights), Candidate Omicron achieved **73.04% Accuracy** and **0.7391 Macro-F1** at Epoch 35.
   - This established a new project-wide inner-validation accuracy record (surpassing Candidate Kappa's 73.00% at Epoch 45).
   - All 4 sleep stages exhibited balanced performance on inner-validation: Wake F1 0.8250, Light F1 0.7353, Deep F1 0.7046, REM F1 0.6914.

4. **Step 4: Outer-Test Discrepancy & Root Cause Analysis**:
   - Despite setting an inner-validation record, outer-test accuracy remained at 66.54% (Ep 35) and 66.59% (Ep 50), failing both the 70.0% accuracy and 0.6900 Weighted-F1 thresholds.
   - Inspection of the confusion matrix reveals that Light sleep recall actually increased to 73.09% (vs Kappa's 70.61%), while Deep sleep recall dropped to 67.09% (vs Kappa's 70.21%) and REM recall dropped to 56.04% (vs Kappa's 57.38%).
   - Severe misclassifications into Light sleep persisted:
     - REM $\to$ Light: 2,657 windows misclassified (37.38% of all REM windows!)
     - Deep $\to$ Light: 2,225 windows misclassified (28.10% of all Deep windows!)
     - Wake $\to$ Light: 977 windows misclassified (30.47% of all Wake windows!)
   - *Why did PLB fail to stop test-time Light sleep absorption?*
     1. **Gradient Attenuation of Prior Correction**: Training under cross-entropy directly rewards minimizing training error on the empirical class distribution ($p_{\text{light}} = 43.19\%$). Consequently, gradient descent on $\alpha_{\text{prior}}$ steadily drove it downwards from -0.4055 to -0.4257 at Ep 35 and -0.4317 at Ep 50, actively reducing $\tau_{\text{prior}}$ to weaken the prior penalty!
     2. **Linear Head Bias Absorption**: In a standard linear classification head $\text{logits} = W x + b$, adding a static prior correction $-\tau \log p$ is mathematically equivalent to shifting the bias vector $b \gets b - \tau \log p$. During 50 epochs of training, the unconstrained bias parameters $b_c$ in the linear head can adapt and offset the prior adjustment, effectively neutralizing the calibration.

5. **Step 5: Gating Conclusion**:
   - Proxy Accuracy = 66.54% < 70.00% (Requirement failed).
   - Proxy Weighted-F1 = 0.6644 < 0.6900 (Requirement failed).
   - Strict gating rules require: `PROXY_FAIL (REVERT)`.

---

## 3. Caveats

1. **Proxy Evaluation Protocol**:
   - Evaluation followed the official single-seed (seed 42), Fold 1, 50-epoch fast proxy protocol. No 5-fold CV was executed per protocol instructions.
2. **Fixed vs Learnable $\tau_{\text{prior}}$**:
   - In Candidate Omicron, $\tau_{\text{prior}}$ was learnable via gradient descent. Making it a fixed hyperparameter or applying test-time temperature scaling without training feedback might yield different calibration behavior, but within the current trainable formulation, the network naturally optimizes away from penalizing the majority class.
3. **Candidate Kappa Remains Pareto-Optimal**:
   - Candidate Kappa (Iteration 12) remains the highest performing model on outer-test final epoch accuracy (67.04% vs Omicron's 66.59%).

---

## 4. Conclusion

Candidate Omicron (`PriorCalibratedBoundedMarginPyramidCNNBiGRU`, 196,382 parameters) has been fully deployed and genuinely evaluated on Fold 1 `cuda:0` under the 50-epoch proxy protocol.

- **Peak Inner-Validation**: **73.04% Accuracy** and **0.7391 Macro-F1** (Epoch 35) — new project-record inner-validation accuracy.
- **Outer-Test Performance**: **66.54% Accuracy** and **0.6644 Weighted-F1** (Epoch 35) / **66.59% Accuracy** and **0.6646 Weighted-F1** (Epoch 50).
- **Gate Verdict**: **`PROXY_FAIL (REVERT)`**. Both the 70.0% accuracy and 0.6900 Weighted-F1 thresholds were not reached.
- **Key Empirical Discovery**: Additive logit prior balancing with learnable $\tau_{\text{prior}}$ is absorbed by the linear classifier bias and attenuated by training loss gradients, failing to resolve the outer-test generalization gap between Light sleep and minority stages.

---

## 5. Verification Method

To independently verify all findings and reproduce the proxy evaluation:

1. **Verify JSON Output Metrics**:
   ```bash
   ./.venv/bin/python -c "
   import json
   with open('experiments/EXP-014/nas_iteration17_proxy_results.json') as f:
       data = json.load(f)
   print('Candidate:', data['candidate'])
   print('Best Epoch:', data['best_epoch'])
   print('Inner-Val Acc:', f\"{data['val_metrics_at_best_epoch']['accuracy']*100:.2f}%\")
   print('Inner-Val Macro-F1:', f\"{data['val_metrics_at_best_epoch']['macro_f1']:.4f}\")
   print('Outer-Test Acc:', f\"{data['outer_test_metrics_best_checkpoint']['accuracy']*100:.2f}%\")
   print('Outer-Test Weighted-F1:', f\"{data['outer_test_metrics_best_checkpoint']['weighted_f1']:.4f}\")
   print('Proxy Passed:', data['gating']['proxy_passed'])
   print('Action:', data['gating']['action'])
   "
   ```
   *Expected Output*:
   `Candidate: Candidate Omicron (PriorCalibratedBoundedMarginPyramidCNNBiGRU)`
   `Best Epoch: 35`
   `Inner-Val Acc: 73.04%`
   `Inner-Val Macro-F1: 0.7391`
   `Outer-Test Acc: 66.54%`
   `Outer-Test Weighted-F1: 0.6644`
   `Proxy Passed: False`
   `Action: PROXY_FAIL (REVERT)`

2. **Inspect Saved Checkpoint**:
   ```bash
   ./.venv/bin/python -c "
   import torch
   ckpt = torch.load('experiments/EXP-014/nas_iteration17_best.pt', map_location='cpu', weights_only=False)
   print('Epoch:', ckpt['epoch'])
   print('tau_prior:', ckpt['tau_prior'])
   print('Val Macro-F1:', ckpt['val_metrics']['macro_f1'])
   "
   ```

3. **Re-run Evaluation Script from Scratch**:
   ```bash
   ./.venv/bin/python experiments/EXP-014/evaluate_nas_iteration17_proxy.py
   ```

4. **Invalidation Conditions**:
   - If `nas_iteration17_proxy_results.json` shows `proxy_passed == True` while outer-test accuracy is 66.54%.
   - If parameter count deviates from 196,382.
   - If `nas_iteration17_best.pt` cannot be loaded.
