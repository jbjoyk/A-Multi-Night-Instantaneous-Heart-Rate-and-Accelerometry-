# NAS Iteration 18 Handoff Report: Candidate Pi Fast Proxy Evaluation

## 1. Observation
1. **Execution Configuration & Environment**:
   - Host Machine: Linux (`x86_64`), Python 3.12 via `./.venv/bin/python`.
   - Hardware: NVIDIA GeForce RTX 3060 (`cuda:0`), 12 GB VRAM.
   - Script Deployed: `experiments/EXP-014/evaluate_nas_iteration18_proxy.py`.
   - Model Instantiated: `UnbiasedHypersphericalBoundedMarginPyramidCNNBiGRU` via `build_nas_iteration18_model` in `experiments/EXP-014/nas_iteration18_model.py`.
   - Parameter Count: Exactly 196,374 trainable parameters (strictly compliant with hard cap $\le 196,728$, 354 parameters headroom).
   - Evaluation Protocol: Fold 1 inner-train (32 subjects, 168 nights), inner-validation (5 subjects, 24 nights), locked outer-test (10 subjects, 54 nights, 32,098 windows). 50 epochs, batch size 4, lr 1e-4, Adam optimizer, seq_len 600, RCCLoss($\lambda=0.01$), seed 42 on `cuda:0`.

2. **Training Dynamics & Epoch-by-Epoch Progress**:
   - Total Training Duration: 102.59 seconds (~1.71 minutes, ~2.05s/epoch).
   - Epoch Progress Sample:
     * Epoch 05/50: Train Loss: 1.2511 | Val Loss: 1.1384 | Val Acc: 56.41% | Val Macro-F1: 0.5327 | Scale s: 15.9862
     * Epoch 10/50: Train Loss: 1.0776 | Val Loss: 1.0437 | Val Acc: 61.21% | Val Macro-F1: 0.6152 | Scale s: 15.9814
     * Epoch 20/50: Train Loss: 0.8848 | Val Loss: 0.9718 | Val Acc: 66.07% | Val Macro-F1: 0.6697 | Scale s: 15.9745
     * Epoch 30/50: Train Loss: 0.8232 | Val Loss: 0.9208 | Val Acc: 69.14% | Val Macro-F1: 0.6954 | Scale s: 15.9686
     * Epoch 40/50: Train Loss: 0.7531 | Val Loss: 0.9593 | Val Acc: 70.08% | Val Macro-F1: 0.7079 | Scale s: 15.9649
     * Epoch 50/50: Train Loss: 0.7097 | Val Loss: 0.9169 | Val Acc: 71.68% | Val Macro-F1: 0.7248 | Scale s: 15.9616
   - Model Selection: Peak inner-validation Macro-F1 was achieved at **Epoch 50** with:
     * Val Accuracy: 71.68%
     * Val Weighted-F1: 0.7163
     * Val Macro-F1: 0.7248
     * Val Cohen's Kappa: 0.5986
     * Val Loss: 0.9169

3. **Outer-Test Partition Evaluation (32,098 Windows, Best Checkpoint @ Epoch 50)**:
   - Evaluated Windows: 32,098
   - Accuracy: **65.24%** (20,941 / 32,098 correct; Gating Threshold: $\ge 70.0\%$ $\rightarrow$ **FAILED**, gap: -4.76%)
   - Weighted-F1: **0.6517** (Gating Threshold: $\ge 0.6900$ $\rightarrow$ **FAILED**, gap: -0.0383)
   - Macro-F1: **0.6452**
   - Cohen's Kappa: **0.4958**
   - Final Epoch 50 Outer-Test Metrics: Exactly identical (Best checkpoint was selected at Epoch 50).
   - Per-Class Metrics:
     * Wake (Class 0): Recall = 62.35% (1,999 / 3,206), Precision = 63.08% (1,999 / 3,169), F1 = 0.6271
     * Light (Class 1): Recall = 67.35% (9,337 / 13,864), Precision = 64.60% (9,337 / 14,454), F1 = 0.6594
     * Deep (Class 2): Recall = 70.02% (5,545 / 7,919), Precision = 69.17% (5,545 / 8,016), F1 = 0.6960
     * REM (Class 3): Recall = 57.11% (4,060 / 7,109), Precision = 62.86% (4,060 / 6,459), F1 = 0.5985
   - Confusion Matrix (Wake, Light, Deep, REM):
     ```
     [[ 1999,   852,   164,   191],
      [  813,  9337,  1857,  1857],
      [  133,  1890,  5545,   351],
      [  224,  2375,   450,  4060]]
     ```

4. **Learned Geometry, Scale, and Margins at Best Checkpoint (Epoch 50)**:
   - Temperature scale $s$: Initialized at $16.0$, converged smoothly to $15.9616$ (total drift: $-0.0384$, indicating stable optimization without gradient divergence).
   - Bounded-Margin CASCI Parameters:
     * `tau_ihr_rem`: raw = -0.0629 $\rightarrow$ effective clamped = -0.0500 (STE lower bound enforced)
     * `tau_acc_rem`: raw = 0.2497 $\rightarrow$ effective clamped = 0.2497
     * `tau_ihr_deep`: raw = 0.0252 $\rightarrow$ effective clamped = 0.0252
     * `tau_acc_deep`: raw = 0.0503 $\rightarrow$ effective clamped = 0.0503
     * `tau_ihr_wake`: raw = 0.0287 $\rightarrow$ effective clamped = 0.0287
     * `tau_acc_wake`: raw = -0.0164 $\rightarrow$ effective clamped = 0.0000 (STE lower bound enforced)
     * `tau_ihr_light`: raw = 0.1286 $\rightarrow$ effective clamped = 0.1286
     * `tau_acc_light`: raw = -0.0675 $\rightarrow$ effective clamped = 0.0000 (STE lower bound enforced)

5. **Saved Artifacts**:
   - `experiments/EXP-014/evaluate_nas_iteration18_proxy.py` (486 lines)
   - `experiments/EXP-014/nas_iteration18_proxy_results.json` (69,820 bytes)
   - `experiments/EXP-014/nas_iteration18_best.pt` (2,572,403 bytes)
   - `experiments/EXP-014/nas_iteration18_best_proxy_ckpt.pt` (2,572,403 bytes)

## 2. Logic Chain
1. **Hypothesis Under Test**:
   - Candidate Pi investigated whether removing additive linear classification biases $b_c$ and normalizing features and prototypes symmetrically onto the unit hypersphere $S^{63}$ with zero margins ($m=0.0$) and a learnable scale $s \approx 16.0$ would resolve the majority-class collapse into Light sleep without triggering the catastrophic over-carving observed in Candidate Lambda.
2. **Analysis of Empirical Results vs Hypotheses**:
   - **Hypothesis Confirmed: Minority stage collapse into Light sleep was mitigated**:
     * In Candidate Kappa (linear head), REM recall was only 52.88% (2,657 windows collapsed into Light). In Candidate Pi, REM recall rose to 57.11% (2,375 collapsed into Light, an improvement of +4.23% absolute recall).
     * Deep recall rose from 68.39% (Kappa) to 70.02% (Pi), an improvement of +1.63% absolute recall.
     * Light recall decreased from 73.18% (Kappa) to 67.35% (Pi), confirming that without positive bias $b_{\text{light}}$, the model no longer blindly defaults to Light sleep.
   - **Hypothesis Refuted: Hyperspherical Cosine Head without prior prevalence weighting hurts overall accuracy**:
     * In physiological ground truth, sleep stages have an inherently non-uniform prior distribution: Light (43.19%), Deep (24.67%), REM (22.15%), Wake (9.99%).
     * Under standard linear classification heads, linear feature magnitude $\|h\|$ carries physiological confidence information (e.g. active wakefulness or deep slow-wave delta rhythms produce large-amplitude physiological dynamics that scale $\|h\|$).
     * Discarding $\|h\|$ via L2 normalization on $S^{63}$ removes this physiological amplitude channel.
     * Furthermore, under pure spherical symmetry with zero bias, the classifier treats all 4 stages as equally probable a priori. As a direct consequence, 1,857 true Light sleep windows were misclassified as Deep and another 1,857 true Light sleep windows were misclassified as REM (a total of 3,714 Light windows misclassified into minority stages, vs only 2,166 in Candidate Kappa).
     * This massive dispersion of Light sleep drove outer-test accuracy down from 67.04% (Kappa) to 65.24% (Pi) and Weighted-F1 down from 0.6698 (Kappa) to 0.6517 (Pi).
3. **Comparison Across Generations**:
   - Candidate Kappa (Iter 12, Linear Head): 67.04% Acc, 0.6698 WF1, 0.6558 MF1, Kappa 0.5262
   - Candidate Lambda (Iter 13, ACAM Asymmetric Margins Cosine): 62.77% Acc, 0.6277 WF1, 0.6402 MF1
   - Candidate Omicron (Iter 17, Prior-Calibrated Linear Head): 66.54% Acc, 0.6651 WF1, 0.6517 MF1, Kappa 0.5186
   - Candidate Pi (Iter 18, Unbiased Hyperspherical Cosine Head): 65.24% Acc, 0.6517 WF1, 0.6452 MF1, Kappa 0.4958
4. **Proxy Gating Action**:
   - Both proxy gating conditions failed:
     * Accuracy: $65.24\% < 70.00\%$
     * Weighted-F1: $0.6517 < 0.6900$
   - Verdict: **PROXY_FAIL (REVERT)**. Candidate Pi is rejected and cannot proceed to 5-fold cross-validation.

## 3. Caveats
- No full 5-fold cross-validation was run because Candidate Pi failed the fast proxy gate on Fold 1.
- Only a single random seed (seed 42) was evaluated per standard proxy protocol.
- The temperature scale $s$ was initialized to 16.0 and barely moved (15.9616). A higher initial scale (e.g. $s=32.0$ or $s=64.0$) might sharpen the softmax distribution, but would not fundamentally resolve the loss of feature norm $\|h\|$ or prior distribution imbalance.

## 4. Conclusion
Candidate Pi (`UnbiasedHypersphericalBoundedMarginPyramidCNNBiGRU`) was successfully deployed and evaluated on Fold 1 fast proxy on `cuda:0` across 50 epochs in 102.59 seconds.
While the Unbiased Hyperspherical Cosine Head successfully increased REM recall from 52.88% to 57.11% and Deep recall from 68.39% to 70.02% by removing linear bias dominance, discarding feature norms $\|h\|$ and ignoring class prior imbalance caused 3,714 Light sleep windows to be misclassified into Deep and REM, dropping outer-test Accuracy to 65.24% and Weighted-F1 to 0.6517.
Candidate Pi decisively fails the fast proxy gating criteria ($\text{Acc} \ge 70.0\%, \text{WF1} \ge 0.6900$).
The proxy verdict is **PROXY_FAIL (REVERT)**. The search loop should revert to the Candidate Kappa architectural lineage.

## 5. Verification Method
1. **Re-evaluate Saved Checkpoint on Outer Test**:
   ```bash
   ./.venv/bin/python -c "
   import torch, json
   from nas_iteration18_model import build_nas_iteration18_model
   from evaluate_nas_iteration18_proxy import load_fold1_data, evaluate_dataset
   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
   _, _, test_loader, _ = load_fold1_data(batch_size=4, seq_len=600)
   model = build_nas_iteration18_model().to(device)
   ckpt = torch.load('experiments/EXP-014/nas_iteration18_best.pt', map_location=device)
   model.load_state_dict(ckpt['model_state'])
   res = evaluate_dataset(model, test_loader, device)
   print('Accuracy:', res['accuracy'])
   print('Weighted-F1:', res['weighted_f1'])
   print('Macro-F1:', res['macro_f1'])
   assert abs(res['accuracy'] - 0.6524) < 0.001
   "
   ```
2. **Inspect Results JSON**:
   - Path: `experiments/EXP-014/nas_iteration18_proxy_results.json`
3. **Invalidation Conditions**:
   - If evaluated accuracy differs from $65.24\%$ or weighted-F1 differs from $0.6517$.
   - If parameter count is not exactly 196,374.
