# Handoff Report: NAS Iteration 11 Candidate Iota Fast Proxy Evaluation

**From**: `teamwork_worker_r11_1` (PyTorch Engineer & Evaluation Agent)  
**To**: `teamwork_orchestrator` (`5bbd46d9-608b-43f0-a1c2-9eb9cfad1c91`)  
**Date**: 2026-11-15T08:50:00Z  
**Type**: Hard Handoff (Evaluation Complete, Results Verified, Verdict Determined)  

---

## 1. Observation

1. **Model Deployment and Parameter Verification**:
   - Deployed Candidate Iota (`SmoothSynergyUltradianPyramidCNNBiGRU`) to `experiments/EXP-014/nas_iteration11_model.py`.
   - Verified trainable parameter count via PyTorch:
     ```python
     from nas_iteration11_model import SmoothSynergyUltradianPyramidCNNBiGRU
     model = SmoothSynergyUltradianPyramidCNNBiGRU()
     sum(p.numel() for p in model.parameters() if p.requires_grad)
     # Output: Exactly 196,409 parameters
     ```
   - Target parameter budget compliance:
     * Model parameters: **196,409**
     * Target budget: $\le 196,550$ (141 parameters headroom)
     * Hard budget cap: $\le 196,728$ (319 parameters headroom)
     * Compression ratio: **96.41%** reduction vs 5.47M published SLAMSS-IFS benchmark.

2. **Proxy Evaluation Execution**:
   - Created `experiments/EXP-014/evaluate_nas_iteration11_proxy.py` matching the standard proxy protocol.
   - Run command: `./.venv/bin/python experiments/EXP-014/evaluate_nas_iteration11_proxy.py`
   - Device: `cuda:0` (NVIDIA GeForce RTX 3060 12GB).
   - Training dataset: BIDSleep Fold 1 manifest (`experiments/EXP-014/EXP014_MANIFESTS/fold_1.json`, 32 inner-train subjects, 168 nights).
   - Validation dataset: Fold 1 inner-val (5 subjects, 24 nights, 14,089 windows).
   - Outer-test dataset: Fold 1 locked outer-test (10 subjects, 54 nights, 32,098 windows).
   - Training duration: 50 epochs completed in **102.48s** (~1.71 min).

3. **Empirical Convergence and Validation Metrics**:
   - Model selection was monitored via inner-validation Macro-F1 at every epoch.
   - Peak validation performance occurred at **Epoch 50**:
     * Validation Macro-F1: **0.7182**
     * Validation Weighted-F1: **0.7105**
     * Validation Accuracy: **71.06%**
     * Validation Loss: **0.7757**
     * Validation Kappa: **0.5831**
     * Per-class Validation F1: Wake = 0.7958, Light = 0.7161, Deep = 0.6891, REM = 0.6718.

4. **Outer-Test Partition Evaluation Results (32,098 windows)**:
   - Primary evaluation on selected best checkpoint (Epoch 50):
     * **Proxy Accuracy**: **65.12%** (20,901 / 32,098 correct windows)
     * **Proxy Weighted-F1**: **0.6513**
     * **Proxy Macro-F1**: **0.6497**
     * **Cohen's Kappa**: **0.4940**
     * Per-Class Metrics:
       - Wake (Support: 3,206): Precision = 71.76%, Recall = 60.23%, F1 = **0.6549**
       - Light (Support: 13,864): Precision = 64.65%, Recall = 66.15%, F1 = **0.6539**
       - Deep (Support: 7,919): Precision = 70.27%, Recall = 70.39%, F1 = **0.7033**
       - REM (Support: 7,109): Precision = 57.96%, Recall = 59.43%, F1 = **0.5869**
     * Confusion Matrix (rows = true [Wake, Light, Deep, REM], cols = pred [Wake, Light, Deep, REM]):
       ```
       [[1931,  848,  167,  260],   # Wake:  Recall = 60.23%
        [ 571, 9171, 1710, 2412],   # Light: Recall = 66.15%
        [  93, 1860, 5574,  392],   # Deep:  Recall = 70.39%
        [  96, 2307,  481, 4225]]   # REM:   Recall = 59.43%
       ```
   - Final Epoch 50 weights: Identical to best checkpoint (Epoch 50 was the peak epoch).

5. **Proxy Gating Threshold Comparison**:
   - Filter Condition 1: Proxy Accuracy $\ge 70.0\%$ $\to$ **65.12%** (**FAILED**, -4.88 pp below threshold).
   - Filter Condition 2: Proxy Weighted-F1 $\ge 0.6900$ $\to$ **0.6513** (**FAILED**, -0.0387 below threshold).
   - **Verdict**: **PROXY_FAIL (REVERT)**.

---

## 2. Logic Chain

1. **Mechanism of Intended Remediation**:
   - In Candidate Theta (CASCI with rigid ReLU multiplication), REM $\to$ Light false negatives surged to 2,459 because any transient accelerometry twitch ($u_{\text{acc}} > 0$) zeroed out $s_{\text{rem}} = \text{ReLU}(u_{\text{ihr}}) \cdot \text{ReLU}(-u_{\text{acc}})$, killing gradients and collapsing predictions into the dominant Light prior.
   - Candidate Iota replaced rigid ReLU clipping with a smooth softplus manifold ($s_{\text{rem}} = \text{softplus}(u_{\text{ihr}}) \cdot \text{softplus}(-u_{\text{acc}})$) and causal temporal smoothing ($W_{\text{syn}} = 5$).
   - This successfully resolved the intended defect: REM recall increased from 57.41% (4,081 windows in Theta) to 59.43% (4,225 windows in Iota, +144 correct windows), and REM $\to$ Light errors dropped from 2,459 to 2,307 (-152 windows).

2. **Mechanism of Failure (Unintended Side Effect)**:
   - While softplus eliminated zero-gradient dead zones, it has a strictly positive lower bound: $\text{softplus}(u) > 0$ for all $u \in \mathbb{R}$.
   - Consequently, during genuine Light sleep epochs (where heart rate fluctuates moderately and accelerometry is low), $s_{\text{rem}}(t)$ remained non-trivially active.
   - Furthermore, the causal depthwise smoothing convolution ($W_{\text{syn}} = 5$ epochs = 2.5 minutes) smeared these elevated synergy values forward across time.
   - As a result, Light $\to$ REM false alarms surged dramatically from 1,879 in Theta to **2,412** in Iota (+533 false alarms, a +28.4% increase).
   - This degraded Light precision to 64.65% and REM precision to 57.96%, dragging overall test accuracy down to 65.12% (vs 66.84% in Theta and 66.70% in Zeta).

3. **Conclusion from Evidence**:
   - Continuous softplus gating without a sharp sparsity boundary causes cross-modal synergy leakage from true Light sleep into false REM detections.
   - Candidate Iota fails both proxy gates ($65.12\% < 70.0\%$ and $0.6513 < 0.6900$). Per Requirement R1.3 of `ORIGINAL_REQUEST.md`, Candidate Iota must be reverted, and the next iteration must explore alternative interaction designs (e.g., bounded margin softplus, learned thresholding, or asymmetric gating).

---

## 3. Caveats

1. **Proxy Protocol Scope**:
   - The evaluation was conducted under the single-fold proxy protocol (Fold 1, seed 42, 50 epochs). The locked outer test set contains 10 subjects and 54 nights (32,098 windows).
2. **Inner Validation vs Outer Test Discrepancy**:
   - Peak inner-validation Macro-F1 reached 0.7182 and validation accuracy reached 71.06% on inner-val subjects, indicating strong training fit. The lower generalization on outer-test subjects (65.12% accuracy) reflects inter-subject variability that requires stronger regularization or sharper contrast boundaries.

---

## 4. Conclusion

- **Candidate Iota Verdict**: **PROXY_FAIL (REVERT)**.
- **Outer Test Accuracy**: **65.12%** (Threshold: $\ge 70.0\%$, FAILED).
- **Outer Test Weighted-F1**: **0.6513** (Threshold: $\ge 0.6900$, FAILED).
- **Outer Test Macro-F1**: **0.6497**.
- **Cohen's Kappa**: **0.4940**.
- **Model Parameters**: Exactly **196,409** (Budget compliant: 141 params under 196,550 target).
- **Next Concrete Action**: Revert Candidate Iota mutation and proceed to NAS Iteration 12 proposal.

---

## 5. Verification Method

To independently reproduce and verify this proxy evaluation:

```bash
# 1. Run the proxy evaluation script on cuda:0
./.venv/bin/python experiments/EXP-014/evaluate_nas_iteration11_proxy.py

# 2. Inspect the saved JSON report
python3 -c "
import json
with open('experiments/EXP-014/nas_iteration11_proxy_results.json') as f:
    d = json.load(f)
print('Candidate:', d['candidate'])
print('Parameters:', d['parameters'])
print('Best Epoch:', d['best_epoch'])
print('Outer Test Accuracy:', f\"{d['outer_test_metrics_best_checkpoint']['accuracy']*100:.2f}%\")
print('Outer Test Weighted-F1:', f\"{d['outer_test_metrics_best_checkpoint']['weighted_f1']:.4f}\")
print('Gating Action:', d['gating']['action'])
assert d['gating']['action'] == 'REVERT'
"
```

### Invalidation Conditions:
- If `nas_iteration11_proxy_results.json` does not exist or has incomplete fields.
- If outer test accuracy $\ge 70.0\%$ or weighted-F1 $\ge 0.6900$ under the documented protocol.
- If model parameter count differs from 196,409.
