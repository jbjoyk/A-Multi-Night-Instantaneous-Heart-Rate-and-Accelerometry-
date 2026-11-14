# EXP-014 Preflight Audit 0A: Model A Validation Convergence Curve

**Date**: 2026-11-10  
**Auditor**: ML Research Engineer / Experimental Auditor  
**Artifacts Inspected**:
- EXP-013 Training Execution Logs (`task-1058.log`)
- Checkpoints: `EXP013_OUTPUTS/checkpoints/model_A_seed{42,123,2024}_final.pt`
- Reference Convergence Curves: `exp011_multiseed_history.pkl` (Model E, 500 epochs)

---

## 1. Audit Objective

Determine whether the fixed 80-epoch protocol creates an unfair training-budget difference between Model A (conventional matched-capacity CNN-BiGRU, 163,940 params) and Model E (full SleepAdaptiveNet, 162,325 params). Specifically investigate:
1. Validation Accuracy, Macro-F1, and loss by epoch across seeds for Model A.
2. The convergence trajectory of Model A: is it still improving, overfitting, or approximately stable around epoch 80?
3. Comparison against Model E's validation convergence dynamics.

---

## 2. Model A Validation Metrics Across Optimization Epochs

In EXP-013, inter-epoch validation evaluation was executed deterministically after every epoch 1..80 on the 28 usable validation nights (5 subjects, completely isolated from test subjects). The periodic log checkpoints recorded at epochs 20, 40, 60, and 80:

| Seed | Epoch | Training Loss | Validation Loss | Validation Accuracy (%) | Validation Macro-F1 | Convergence State at Epoch 80 |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | 20 | 0.9043 | 0.8240 | 65.55% | 0.6430 | Initial convergence |
| **42** | 40 | 0.8174 | 0.7407 | 69.79% | 0.6952 | Active improvement |
| **42** | 60 | 0.7497 | **0.7257** | **71.58%** | 0.7112 | Minimum validation loss |
| **42** | 80 | **0.6959** | 0.7374 | 71.55% | **0.7239** | Plateau / Near-optimal |
| **123** | 20 | 0.9137 | 0.8489 | 64.49% | 0.6275 | Initial convergence |
| **123** | 40 | 0.8076 | 0.7626 | 68.63% | 0.6939 | Active improvement |
| **123** | 60 | 0.7476 | **0.7328** | 70.65% | 0.7149 | Minimum validation loss |
| **123** | 80 | **0.7015** | 0.7526 | **70.76%** | **0.7165** | Plateau / Near-optimal |
| **2024** | 20 | 0.9046 | 0.8492 | 64.77% | 0.6363 | Initial convergence |
| **2024** | 40 | 0.8129 | 0.7500 | 69.92% | 0.7035 | Active improvement |
| **2024** | 60 | 0.7557 | **0.7362** | **70.02%** | **0.7122** | Minimum validation loss |
| **2024** | 80 | **0.6975** | 0.7391 | 69.09% | 0.6972 | Slight post-peak drift |

---

## 3. Comparative Convergence Dynamics: Model A vs. Model E

Comparing Model A against the 500-epoch multi-seed convergence audit of Model E (`exp011_multiseed_history.pkl`):

| Model & Seed | Peak Val Macro-F1 | Epoch of Peak Val F1 | Val Loss at Peak | Val Macro-F1 at Epoch 80 | Val Acc at Epoch 80 | Overfitting Post-Epoch 80? |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model A (Seed 42)** | 0.7239 | 80 | 0.7374 | 0.7239 | 71.55% | Plateau around ep 60-80 |
| **Model A (Seed 123)** | 0.7165 | 80 | 0.7526 | 0.7165 | 70.76% | Plateau around ep 60-80 |
| **Model A (Seed 2024)** | 0.7122 | 60 | 0.7362 | 0.6972 | 69.09% | Plateau around ep 60-80 |
| **Model E (Seed 42)** | 0.7441 | 78 | 0.7245 | 0.7386 | 73.27% | Universal post-peak overfitting after ep 90 |
| **Model E (Seed 123)** | 0.7266 | 92 | 0.7565 | 0.7190 | 71.00% | Universal post-peak overfitting after ep 92 |
| **Model E (Seed 2024)** | 0.7182 | 72 | 0.7817 | 0.7028 | 69.42% | Universal post-peak overfitting after ep 85 |

### Observations:
1. **Convergence Timing**: Both Model A and Model E enter their empirical peak validation window between epochs 60 and 85.
   - For Model A, validation loss hits a minimum at Epoch ~60 ($\approx 0.726\text{--}0.736$), and validation Macro-F1 plateaus between epochs 60 and 80 ($\approx 0.712\text{--}0.724$).
   - For Model E, validation loss hits a minimum at Epoch ~72–85 ($\approx 0.724\text{--}0.756$), and validation Macro-F1 peaks at Epoch $80.7 \pm 10.3$ ($\approx 0.718\text{--}0.744$).
2. **Trajectory Characterization at Epoch 80**:
   - Model A is **approximately stable and plateaus around epoch 80**. It is neither severely under-trained nor suffering from catastrophic overfitting.
   - Model A is **NOT** still rapidly improving at epoch 80 (training loss continues to fall from 0.75 to 0.69, but validation loss has already begun a slight rise from 0.73 to 0.75, indicating standard regularization saturation).
3. **Budget Fairness**:
   - The fixed 80-epoch protocol evaluated both Model A and Model E within their respective peak/plateau validation regimes.
   - Therefore, the observed performance advantage of Model E over Model A (+1.79 pp accuracy, +0.0199 Macro-F1 on test) **cannot be dismissed as a mere artifact of premature stopping of Model A**. Both models were evaluated in their mature plateau regimes.

---

## 4. Recommendation for EXP-014 Cross-Validation

While 80 epochs was empirically fair for the historical 31/5/11 split, in a 5-fold cross-validation across all 47 subjects:
- Outer fold dataset sizes will differ (folds will have ~37–38 training subjects instead of 31).
- Training convergence rates may shift across different fold compositions.
- **Protocol Decision**: As required in Phase 2, EXP-014 should evaluate inner validation tracking within each training fold to verify optimal stopping, or confirm that an 80-epoch fixed budget remains appropriate if nested validation introduces excessive computational variance.
