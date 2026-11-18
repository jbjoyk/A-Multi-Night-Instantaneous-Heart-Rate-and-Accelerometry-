# Handoff Report: NAS Iteration 14 (Candidate Mu) Deployment & Fast Proxy Evaluation
**Candidate Mu: Dual-Autonomic Dispersion Synergy with Calibrated Multi-Stream Linear Head**  
`CalibratedAutonomicDispersionPyramidCNNBiGRU`

- **Agent**: `teamwork_worker_r14_1` (Implementer & QA)
- **Recipient**: Project Orchestrator Gen 2 (`5a45d55d-cdec-4610-8dcf-b95f2285b08e`)
- **Timestamp**: 2026-11-15T16:40:00Z
- **Working Directory**: `/home/rj2826/ai-research/sleep/02. A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels/.agents/teamwork/teamwork_worker_r14_1`
- **Execution Target**: Single GPU `cuda:0` (NVIDIA GeForce RTX 3060, Driver 595.99.02, CUDA 13.2)
- **Virtualenv**: `./.venv/bin/python`

---

## 1. Observation

### 1.1 Artifacts Created & Verified
- **Production Architecture**: `experiments/EXP-014/nas_iteration14_model.py`
  - Model Class: `CalibratedAutonomicDispersionPyramidCNNBiGRU`
  - Factory Function: `build_nas_iteration14_model`
  - Aliases: `CandidateMu`, `CandidateMuPyramidCNNBiGRU`, `SubjectAdaptiveCalibratedDispersionPyramidCNNBiGRU`
- **Fast Proxy Evaluation Pipeline**: `experiments/EXP-014/evaluate_nas_iteration14_proxy.py`
- **Metric Bundle & Logs**: `experiments/EXP-014/nas_iteration14_proxy_results.json`
- **Model Checkpoint**: `experiments/EXP-014/nas_iteration14_best_proxy_ckpt.pt` (Epoch 37, 2.5 MB)

### 1.2 Exact Trainable Parameter Count Verification
- **Candidate Mu (Default Linear Head)**: **Exactly 196,387 parameters**
- **Hard Parameter Cap ($\le 196,728$)**: **PASS** (+341 parameters headroom)
- **Target Budget Cap ($\le 196,500$)**: **PASS** (+113 parameters headroom)
- **Target Search Proximity ($\sim 196,400$)**: **PASS** (13 parameters delta)
- **Compression Ratio vs 5.47M SLAMSS-IFS Benchmark**: **96.41% parameter compression**
- **14-Test Unit Suite**: `./.venv/bin/python .agents/teamwork/teamwork_explorer_r14_1/verify_candidate_mu.py` passed 14/14 tests on `cuda:0` with 0 errors.

### 1.3 Training & Inner-Validation Dynamics (50 Epochs, Seed 42, cuda:0)
- **Dataset Partition (Fold 1 Manifest)**:
  - Inner-Train: 32 subjects, 168 nights, 42 batches per epoch (batch size 4, seq_len 600)
  - Inner-Validation: 5 subjects, 24 nights, 14,089 windows, 6 batches per epoch
  - Outer-Test: 10 subjects, 54 nights, 32,098 windows, 14 batches
- **Total Training Duration**: 102.11 seconds (~1.70 minutes, ~2.04 seconds per epoch)
- **Peak Inner-Validation Checkpoint**: **Epoch 37**
  - Validation Macro-F1: **0.7205** (0.72049)
  - Validation Accuracy: **70.85%** (0.70850)
  - Validation Weighted-F1: **0.7085** (0.70851)
  - Validation Cohen's Kappa: **0.5855**
  - Validation Loss: 0.7655 (Train Loss: 0.6860)
  - Per-class Validation F1: Wake 0.8013, Light 0.6984, Deep 0.6994, REM 0.6828
- **Final Epoch 50 State**:
  - Validation Macro-F1: 0.7140, Accuracy: 70.15%, Weighted-F1: 0.7013, Loss: 0.7850

### 1.4 Outer-Test Evaluation Metrics (Locked Partition: 10 Subjects, 32,098 Windows)
Evaluated at peak inner-validation checkpoint (**Epoch 37**):
- **Outer-Test Accuracy**: **64.55%** (0.64549) — Threshold: $\ge 70.0\%$ [**FAILED**]
- **Outer-Test Weighted-F1**: **0.6458** (0.64575) — Threshold: $\ge 0.6900$ [**FAILED**]
- **Outer-Test Macro-F1**: **0.6423** (0.64226)
- **Outer-Test Cohen's Kappa**: **0.4871**
- **Test Loss**: 0.8799

#### Per-Class Classification Performance
| Sleep Stage | F1 Score | Recall | Precision | Class Support (Windows) |
|---|---|---|---|---|
| **Wake** | 0.6338 | 59.17% (1,897 / 3,206) | 68.24% (1,897 / 2,780) | 3,206 |
| **Light** | 0.6473 | 65.01% (9,013 / 13,864) | 64.45% (9,013 / 13,985) | 13,864 |
| **Deep** | 0.6983 | 69.63% (5,514 / 7,919) | 70.03% (5,514 / 7,874) | 7,919 |
| **REM** | 0.5896 | 60.42% (4,295 / 7,109) | 57.58% (4,295 / 7,459) | 7,109 |
| **Overall** | **0.6458** | **64.55%** | **64.81%** | **32,098** |

#### 4x4 Confusion Matrix (Rows: Ground Truth, Columns: Predictions)
```
                Pred Wake    Pred Light    Pred Deep    Pred REM    Total Ground Truth
True Wake:        1,897         850          182          277             3,206
True Light:         687       9,013        1,711        2,453            13,864
True Deep:           96       1,875        5,514          434             7,919
True REM:           100       2,247          467        4,295             7,109
Total Pred:       2,780      13,985        7,874        7,459            32,098
```

#### Final Epoch 50 Metrics (Ablation Comparison)
- Accuracy: **64.11%** (0.64110)
- Weighted-F1: **0.6408** (0.64076)
- Macro-F1: **0.6387** (0.63868)
- Cohen's Kappa: **0.4851**

### 1.5 Learned Asymmetric Margins & Scales at Peak Checkpoint (Epoch 37)
- `tau_ihr_rem`: `-0.0466` (initialized to `0.00`)
- `tau_acc_rem`: `+0.2790` (initialized to `+0.25`, preserving phasic twitch tolerance)
- `tau_disp_rem`: `+0.0196` (initialized to `+0.05`, actively constrained at `0.02` floor)
- `gamma_disp_rem`: `+0.5370` (initialized to `+0.50`)
- `tau_ihr_deep`: `+0.0402` (initialized to `0.00`)
- `tau_acc_deep`: `+0.0952` (initialized to `+0.05`)
- `tau_disp_deep`: `+0.0869` (initialized to `+0.05`)
- `tau_ihr_wake`: `-0.0154` (initialized to `0.00`)
- `tau_acc_wake`: `-0.0219` (initialized to `0.00`)
- `tau_ihr_light`: `+0.0816` (initialized to `0.00`)
- `tau_acc_light`: `-0.0012` (initialized to `0.00`)

---

## 2. Logic Chain

1. **Resolution of Candidate Lambda's ACAM Distortion**:
   - In Iteration 13 (Candidate Lambda), the Anti-Collapse Angular Margin (ACAM) cosine head applied asymmetric angular penalties ($m_{\text{rem}}=0.15, m_{\text{deep}}=0.12$ vs $m_{\text{light}}=0.05$). This severely compressed the decision boundary for Light sleep, causing Light recall to collapse to 47.15% (3,464 Light$\rightarrow$REM false alarms) and overall accuracy to fall to 61.85%.
   - In Candidate Mu, eliminating ACAM and restoring the Calibrated Multi-Stream Linear Head produced an immediate, decisive recovery of Light sleep:
     - Light recall surged from 47.15% to **65.01%** (+17.86 percentage points).
     - Correctly classified Light sleep windows increased from 6,537 to **9,013** (+2,476 rescued windows).
     - Light$\rightarrow$REM false alarms were reduced from 3,464 down to **2,453** (-1,011 false alarms).
     - Light$\rightarrow$Deep false alarms were reduced from 2,519 down to **1,711** (-808 false alarms).
     - Overall Outer-Test Accuracy increased by **+2.70 percentage points** (from 61.85% to 64.55%), and Weighted-F1 increased by **+0.0328** (from 0.6130 to 0.6458).

2. **Why Candidate Mu Falls Short of Candidate Kappa (Iteration 12)**:
   - Candidate Kappa reached 66.84% accuracy with 70.61% Light sleep recall (9,789 windows) and 1,880 Light$\rightarrow$REM false alarms.
   - Candidate Mu achieved 64.55% accuracy with 65.01% Light sleep recall (9,013 windows) and 2,453 Light$\rightarrow$REM false alarms.
   - **Root Cause Analysis**:
     In Candidate Mu, REM autonomic activation is dual-channel and additive:
     $$a_{\text{rem}} = \text{ReLU}(u_{\text{ihr}} - \tau_{\text{ihr\_rem}}) + \text{ReLU}(\gamma_{\text{disp\_rem}}) \cdot \text{ReLU}(v_{\text{disp}} - \tau_{\text{disp\_rem\_eff}})$$
     Because the two channels are combined additively ($+$), the condition for triggering REM synergy is a logical OR ($\text{mean HR surge} \lor \text{autonomic dispersion surge}$).
     During training, $\tau_{\text{ihr\_rem}}$ drifted negative to $-0.0466$, while $\tau_{\text{disp\_rem}}$ drifted to $+0.0196$ (held at the $+0.02$ floor). In held-out test subjects, transitional Light sleep windows (which physiologically border REM and have transient autonomic instability or slight vagal withdrawal) frequently exhibit $u_{\text{ihr}} > -0.0466$ or $v_{\text{disp}} > 0.02$. Because either condition suffices to activate the additive synergy term, Candidate Mu generated 2,453 Light$\rightarrow$REM false alarms (573 more than Candidate Kappa's 1,880).
     Similarly, Deep sleep synergy thresholds drifted upwards ($\tau_{\text{ihr\_deep}} = +0.0402, \tau_{\text{disp\_deep}} = +0.0869$), capturing 1,711 Light$\rightarrow$Deep false alarms (239 more than Candidate Kappa's 1,472).
     Together, this residual cross-modal leakage across Light sleep borders capped outer-test accuracy at 64.55%.

3. **Gating Assessment & Verdict**:
   - Fast proxy filter thresholds (Requirement R1.3):
     - Proxy Accuracy $\ge 70.0\%$ (Achieved: **64.55%** $\rightarrow$ **FAILED**, -5.45 pp delta)
     - Proxy Weighted-F1 $\ge 0.6900$ (Achieved: **0.6458** $\rightarrow$ **FAILED**, -0.0442 delta)
   - Both metrics fail to meet the proxy thresholds.
   - Per Requirement R1.3 of `ORIGINAL_REQUEST.md`, Candidate Mu must be reverted and logged as a search failure.

---

## 3. Caveats

1. **Fast Proxy Scope**: Evaluation was performed on Fold 1 single-seed (seed 42, 50 epochs) on `cuda:0` per the standard proxy protocol. Full 5-fold cross-validation across all 47 subjects was not executed because the candidate failed the proxy gating filter.
2. **Dispersion Floor Tuning**: The margin floor $\min \tau_{\text{disp\_rem}}$ was set to $+0.02$. While this prevented catastrophic negative drift (which caused Lambda to drop to $-0.0014$), it did not eliminate false triggers from transitional Light sleep autonomic fluctuations. A higher threshold or non-additive (e.g. multiplicative / soft-max gated) interaction was not explored in this iteration.
3. **Hyperparameter Invariance**: Training hyperparameters (batch size 4, lr 1e-4, Adam, seq_len 600, RCC loss $\lambda=0.01$) remained strictly locked to preserve comparative validity.

---

## 4. Conclusion

- **Structural Outcome**: Candidate Mu demonstrated that eliminating asymmetric ACAM angular margins directly repairs Light sleep collapse (+17.86 pp Light recall, +2,476 rescued Light windows, +2.70 pp accuracy over Candidate Lambda).
- **Failure Mode Identified**: Additive OR-gating in dual-autonomic synergy ($u_{\text{ihr}} \lor v_{\text{disp}}$) permits minor physiological autonomic variations in transitional Light sleep to trigger false REM and Deep synergy terms, resulting in 2,453 Light$\rightarrow$REM and 1,711 Light$\rightarrow$Deep misclassifications.
- **Proxy Gate Verdict**: **`PROXY_FAIL (REVERT)`** (Accuracy 64.55% < 70.0%, Weighted-F1 0.6458 < 0.6900).
- **Search Recommendation for Iteration 15**:
  Retain the Calibrated Multi-Stream Linear Head and investigate a **Conjunctive or Soft-Gated Interaction** for autonomic dispersion (e.g. soft gating where dispersion modulates the mean contrast rather than acting as an independent additive trigger), or apply a sharper, higher dispersion floor to fully protect Light sleep.

---

## 5. Verification Method

To independently reproduce the evaluation results:

```bash
# 1. Run 14-test verification suite
./.venv/bin/python .agents/teamwork/teamwork_explorer_r14_1/verify_candidate_mu.py

# 2. Inspect generated proxy results JSON
cat experiments/EXP-014/nas_iteration14_proxy_results.json | grep -A 15 '"gating"'

# 3. Verify checkpoint integrity and model parameters
./.venv/bin/python -c "
import torch
ckpt = torch.load('experiments/EXP-014/nas_iteration14_best_proxy_ckpt.pt', map_location='cpu')
print('Candidate:', ckpt['candidate'])
print('Peak Epoch:', ckpt['epoch'])
print('Peak Val Macro-F1:', ckpt['val_metrics']['macro_f1'])
print('Peak Val Acc:', ckpt['val_metrics']['accuracy'])
"
```
