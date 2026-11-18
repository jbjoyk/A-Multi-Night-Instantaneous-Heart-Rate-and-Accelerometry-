# Handoff Report: NAS Iteration 16 — Candidate Xi Deployment and Fast Proxy Evaluation

- **Agent**: `teamwork_worker_r16_1`
- **Role**: Implementer / Specialist / QA
- **Target Architecture**: Candidate Xi (`TemporalBoutRefinedPyramidCNNBiGRU`)
- **Evaluation Phase**: Fast Proxy Evaluation on Fold 1 (50 Epochs, Seed 42, `cuda:0`)
- **Timestamp**: 2026-11-15T11:18:30Z
- **Working Directory**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r16_1`

---

## 1. Observation

### 1.1 Architectural Implementation and Parameter Audit
The Candidate Xi architecture was deployed into `experiments/EXP-014/nas_iteration16_model.py` and evaluated via `experiments/EXP-014/evaluate_nas_iteration16_proxy.py`.

Direct parameter count audit executed on `cuda:0` via `./.venv/bin/python experiments/EXP-014/nas_iteration16_model.py`:
- **Total Trainable Parameters**: Exactly **196,706**
- **Hard Cap Requirement**: $\le 196,728$ parameters (headroom: 22 parameters)
- **Compression Ratio**: 96.40% compression relative to the 5.47M parameter SLAMSS-IFS published benchmark
- **Submodule Breakdown**:
  * `hr_encoder`: 29,568 params
  * `acc_encoder`: 29,568 params
  * `meta_proj`: 96 params
  * `basal_dslbs`: 16 params
  * `acc_dslbs`: 0 params (scale-invariant running box filter buffers)
  * `luan_hr`: 320 params
  * `luan_acc`: 320 params
  * `cadm`: 494 params (10 learned margins/scales + MLP)
  * `circadian_gate`: 2,592 params
  * `temporal_gru`: 97,920 params
  * `temp_norm`: 192 params
  * `inter_epoch_attn`: 16,416 params
  * `classifier_head.in_proj`: 18,624 params
  * `classifier_head.tbr`: 320 params (`nn.Conv1d(64, 64, kernel_size=5, groups=64, bias=False)`)
  * `classifier_head.linear`: 260 params (`nn.Linear(64, 4, bias=True)`)

### 1.2 Proxy Training Runtime and Validation Trajectory
The 50-epoch proxy training completed on single GPU `cuda:0` in **102.73 seconds (~1.71 minutes)**:
- Epoch 05/50: Train Loss 1.0920 | Val Loss 0.9915 | Val Acc 58.36% | Val Macro-F1 0.5797
- Epoch 15/50: Train Loss 0.8366 | Val Loss 0.8788 | Val Acc 62.79% | Val Macro-F1 0.6372
- Epoch 25/50: Train Loss 0.7536 | Val Loss 0.8958 | Val Acc 63.65% | Val Macro-F1 0.6351
- Epoch 35/50: Train Loss 0.7016 | Val Loss 0.8517 | Val Acc 66.04% | Val Macro-F1 0.6727
- Epoch 45/50: Train Loss 0.6513 | Val Loss 0.8056 | Val Acc 69.54% | Val Macro-F1 0.7036
- Epoch 49/50: Train Loss 0.6385 | Val Loss 0.7986 | Val Acc 70.25% | Val Macro-F1 **0.7125** (PEAK)
- Epoch 50/50: Train Loss 0.6366 | Val Loss 0.8163 | Val Acc 69.18% | Val Macro-F1 0.7071

Best inner-validation checkpoint achieved at **Epoch 49**:
- **Inner-Val Macro-F1**: 0.7125
- **Inner-Val Accuracy**: 70.25%
- **Inner-Val Weighted-F1**: 0.7015
- **Inner-Val Cohen's Kappa**: 0.5822

### 1.3 Outer-Test Evaluation Metrics (Locked Partition: 10 Subjects, 54 Nights, 32,098 Windows)
Evaluated on peak inner-validation checkpoint (Epoch 49) per protocol:
- **Proxy Outer-Test Accuracy**: **62.93%** (Threshold: $\ge 70.0\%$ $\to$ **FAILED**)
- **Proxy Outer-Test Weighted-F1**: **0.6300** (Threshold: $\ge 0.6900$ $\to$ **FAILED**)
- **Proxy Outer-Test Macro-F1**: **0.6293**
- **Cohen's Kappa**: **0.4770**
- **Per-Class Breakdown**:
  * **Wake** (3,206 windows): F1 = 0.6108 | Recall = 68.93% | Precision = 54.83%
  * **Light** (13,864 windows): F1 = 0.6145 | Recall = 57.47% | Precision = 66.03%
  * **Deep** (7,919 windows): F1 = 0.6897 | Recall = 67.70% | Precision = 70.28%
  * **REM** (7,109 windows): F1 = 0.6022 | Recall = 65.58% | Precision = 55.67%
- **4x4 Confusion Matrix (Rows=True [Wake, Light, Deep, REM], Columns=Pred [Wake, Light, Deep, REM])**:
  ```python
  [[2210,  602,  120,  274],   # Wake
   [1231, 7967, 1690, 2976],   # Light
   [ 251, 1845, 5361,  462],   # Deep
   [ 339, 1651,  457, 4662]]   # REM
  ```

Final Epoch 50 weights comparison:
- **Outer-Test Accuracy**: 62.44%
- **Outer-Test Weighted-F1**: 0.6243
- **Outer-Test Macro-F1**: 0.6294
- **Cohen's Kappa**: 0.4692

### 1.4 Learned Parameters State at Best Checkpoint (Epoch 49)
- **TBR Depthwise Weights**: norm = 0.9693, mean = +0.0501, std = 0.0206 (shape `[64, 1, 5]`)
- **CADM Learned Margins**:
  * $\tau_{\text{ihr\_rem}} = -0.0672$ (drifted negative from 0.0)
  * $\tau_{\text{acc\_rem}} = +0.3113$ (expanded outward from +0.25)
  * $\alpha_{\text{disp\_rem}} = +0.4443$
  * $\tau_{\text{ihr\_deep}} = +0.0360$
  * $\tau_{\text{acc\_deep}} = +0.0943$
  * $\alpha_{\text{disp\_deep}} = +0.5210$
  * $\tau_{\text{ihr\_wake}} = -0.0055, \quad \tau_{\text{acc\_wake}} = -0.0099$
  * $\tau_{\text{ihr\_light}} = +0.0936, \quad \tau_{\text{acc\_light}} = -0.0153$

---

## 2. Logic Chain

1. **Parameter Compliance & Clean Gradient Flow**:
   Candidate Xi instantiates with exactly 196,706 trainable parameters, precisely satisfying the $< 196,728$ cap (22 parameters headroom). The depthwise causal convolution TBR block (`Conv1d(64, 64, kernel_size=5, groups=64, bias=False)`, 320 params) initialized at 0.0 and actively learned throughout training, reaching a weight norm of 0.9693 and mean weight of +0.0501 across channels at Epoch 49.

2. **Inner-Validation Generalization**:
   On the inner-validation partition (5 subjects, 24 nights), Candidate Xi demonstrated solid optimization, reaching peak inner-validation Macro-F1 of 0.7125 and Accuracy of 70.25% at Epoch 49.

3. **Outer-Test Generalization Gap & Failure Mechanism**:
   Despite strong inner-validation metrics, outer-test performance dropped to 62.93% Accuracy and 0.6300 Weighted-F1 (a -7.32 pp drop from inner validation). Detailed inspection of the confusion matrix and learned margins reveals two interacting failure mechanisms:
   - **CADM Threshold Relaxation Leaks Light Sleep into REM**:
     In Candidate Xi, $\tau_{\text{ihr\_rem}}$ drifted negative to $-0.0672$ and $\tau_{\text{acc\_rem}}$ opened outward to $+0.3113$. Consequently, transitional Light sleep windows with mild autonomic variance or small twitches triggered REM synergy. This caused **2,976 Light $\to$ REM false alarms** (accounting for 21.5% of all genuine Light sleep windows). Compared to Candidate Kappa (1,880 false alarms) and Candidate Theta (1,879 false alarms), Light $\to$ REM false alarms surged by over +1,090 windows.
   - **Symmetric Pre-Classifier Causal Smoothing Dilutes Sleep Stage Transitions**:
     The depthwise causal convolution TBR was applied across all 64 pre-classifier channels uniformly without stage-dependent gating. Because sleep stages have asymmetric durations (majority Light sleep vs brief transition bouts), uniform 5-epoch causal temporal smoothing smeared dominant sleep representations across transition boundaries, lowering Light sleep precision to 66.03% and REM precision to 55.67%.

4. **Comparison across Iteration Milestones**:
   - Candidate Kappa (Iter 12): 66.84% (67.04% Ep 50) Acc | 0.6678 (0.6686) WF1 (All-time high water mark)
   - Candidate Zeta (Iter 8): 65.54% Acc | 0.6556 WF1
   - Candidate Theta (Iter 10): 66.84% Acc | 0.6677 WF1
   - Candidate Nu (Iter 15): 64.52% Acc | 0.6452 WF1
   - Candidate Xi (Iter 16): 62.93% Acc | 0.6300 WF1

Candidate Xi performs significantly below the benchmark high water mark established by Candidate Kappa (Iter 12) and Candidate Theta (Iter 10).

---

## 3. Caveats

- **Hyperparameter Invariance**:
  Per Requirement R2 of `ORIGINAL_REQUEST.md`, training hyperparameters (batch size 4, lr=1e-4, Adam, seq_len 600, RCCLoss weight 0.01) were strictly held constant. No learning rate warm-up, weight decay on margins, or margin regularization was introduced.
- **TBR Placement**:
  TBR was evaluated strictly at the pre-classifier feature level ($h \in \mathbb{R}^{64}$). Alternative placements (such as emission logit-level Markovian transition smoothing or hidden GRU state smoothing) were not tested in this run.

---

## 4. Conclusion & Gate Verdict

- **Gating Assessment**:
  * Proxy Accuracy: **62.93%** (Required: $\ge 70.0\%$) $\to$ **FAIL**
  * Proxy Weighted-F1: **0.6300** (Required: $\ge 0.6900$) $\to$ **FAIL**
- **Action**: **`PROXY_FAIL (REVERT)`**
- Candidate Xi fails the fast proxy filter. Per Requirement R1.3 of `ORIGINAL_REQUEST.md` and the Continuous Search Protocol, Candidate Xi is formally rejected and reverted. The search loop proceeds to propose a new mutation for Iteration 17.

---

## 5. Verification Method

To independently verify all findings and reproduce metrics:

1. **Verify Exact Parameter Count**:
   ```bash
   ./.venv/bin/python experiments/EXP-014/nas_iteration16_model.py
   ```
   *Expected output*: `Candidate Xi Trainable Parameters: 196,706` and `Parameter verification SUCCESS`.

2. **Inspect Saved Results Artifacts**:
   ```bash
   ./.venv/bin/python -c '
   import json
   with open("experiments/EXP-014/nas_iteration16_proxy_results.json") as f:
       d = json.load(f)
   print("Gating verdict:", d["gating"]["action"])
   print("Outer-test Accuracy:", d["outer_test_metrics_best_checkpoint"]["accuracy"])
   print("Outer-test Weighted-F1:", d["outer_test_metrics_best_checkpoint"]["weighted_f1"])
   print("Confusion matrix:", d["outer_test_metrics_best_checkpoint"]["confusion_matrix"])
   '
   ```

3. **Inspect Checkpoint Integrity**:
   ```bash
   ./.venv/bin/python -c '
   import torch
   ckpt = torch.load("experiments/EXP-014/nas_iteration16_best_proxy_ckpt.pt", map_location="cpu")
   print("Checkpoint epoch:", ckpt["epoch"])
   print("TBR weight shape:", ckpt["model_state"]["classifier_head.tbr.conv.weight"].shape)
   print("Validation Macro-F1:", ckpt["val_metrics"]["macro_f1"])
   '
   ```
