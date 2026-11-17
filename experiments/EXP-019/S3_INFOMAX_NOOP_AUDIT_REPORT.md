# EXP-019 S3 Diagnostic Audit: Implementation No-Op Forensic Report

**Date:** 2026-11-17  
**Auditor:** EGM Review System (Phase A2 Implementation Forensic Audit)  
**Target Subject:** Secondary Hypothesis S3 ($UA\text{-}2$ vs $ZS = 0.00\text{ pp}$, 95% CI: $[0.00, 0.00]\text{ pp}$, $N=46$)  
**Governing Documents:**
- `RESEARCH_DIRECTION_v4_FULL.md` (Phase A2 Mandate)
- `PREREG_EXP-019_H-A.md` (Locked at `4dff5b7`)
- `DECISIONS.md` (`DEC-029`)
- `experiments/EXP-019/run_exp019_stage_c.py`
- `experiments/EXP-019/run_exp019_stage_d.py`
- `experiments/EXP-019/exp019_adaptation_core.py`
- `experiments/EXP-019/exp019_stage_d_selected_configs.json`
- `experiments/EXP-019/exp019_stage_c_results.json`

---

## 1. Executive Summary & Verdict Classification

### Verdict: **C. Adaptation did not actually occur because of an implementation/configuration no-op.**

The observed $+0.00\text{ pp}$ gain ($[0.00, 0.00]\text{ pp}$ across all 46 subjects) was **NOT** an empirical failure of unsupervised InfoMax adaptation. It was an implementation no-op caused by a physiological unit mismatch in the wearable data filtering logic:

1. `exp019_adaptation_core.py::compute_wearable_quality_mask` applied a physiological plausibility check expecting raw heart rate values in beats-per-minute:
   $$\text{valid\_hr} = (\mu_{\text{ihr}} \ge 30.0) \land (\mu_{\text{ihr}} \le 180.0)$$
2. However, the cached preprocessed dataset ingested via `prepare_dataset_records` provides **z-score standardized IHR** ($\mu \approx 0, \sigma \approx 1$), whose empirical values across the entire dataset lie strictly within $[-3.95, +9.94]$.
3. Consequently, `valid_hr` evaluated to `False` for **100% of all 82,762 epochs across all 94 calibration nights for all 47 subjects in the dataset** (0 valid epochs).
4. Inside `adapt_unsupervised_infomax`, the adaptation loop encountered:
   ```python
   valid_probs = probs[wearable_mask]  # shape: (0, 4)
   if valid_probs.shape[0] == 0:
       break
   ```
   This condition triggered on **step 0** for every subject and every random seed.
5. As a result:
   - **Zero** loss was computed;
   - **Zero** backward passes were run;
   - **Zero** optimizer steps were executed;
   - **Zero** parameters were changed (displacement $\equiv 0.00000000$);
   - The returned model object was bitwise identical to the unadapted zero-shot base model;
   - Sliding-window evaluation evaluated the exact zero-shot base model twice, generating bitwise identical stage predictions and class probabilities for every epoch of all 152 evaluation nights across all 46 subjects.

---

## 2. Complete S3 Execution Path Audit (Stages 1–14)

| Stage | Inspected File & Location | Expected Behavior | Observed Execution | Forensic Finding |
|---|---|---|---|---|
| **1. Preregistration / Configuration** | `PREREG_EXP-019_H-A.md` (Sec. 2)<br>`exp019_stage_d_selected_configs.json` | Pre-registered S3 as InfoMax entropy minimization on 2 unlabeled nights. | Stage D grid: `lr in [1e-5, 1e-4]`, `steps in [20, 100]`. All 4 grid options produced tied zero-shot scores due to the no-op; tie-breaker selected `(lr=1e-5, steps=20)` for all 5 folds. | Config selected was a byproduct of the no-op tie-breaker. |
| **2. Adaptation Dataset Construction** | `exp019_adaptation_core.py` (L220–222, L34–60) | Construct batch with valid calibration data and boolean wearable quality mask. | Right-padded $B=2, L=1200$ correctly. However, `compute_wearable_quality_mask` checked `30 <= hr_mean <= 180` on standardized z-score data. Mask had 0 valid epochs out of 82,762. | **CRITICAL FAILURE**: `wearable_mask` was 100% `False`. |
| **3. Adaptation Loss Formulation** | `exp019_adaptation_core.py` (L245–260) | Compute Shannon entropy $H(p)$ and marginal diversity $H(\bar{p})$. | Mathematically coded, but line 243 `break` prevented execution. | Loss was never computed. |
| **4. Optimizer Creation** | `exp019_adaptation_core.py` (L212–218) | Adam optimizer targeting trainable parameters. | Adam instantiated targeting 16 parameters (48,836 params). | Instantiated correctly, but never stepped. |
| **5. optimizer.step()** | `exp019_adaptation_core.py` (L263) | Execute gradient update step. | Line 263 was never reached. | **0 optimizer steps executed**. |
| **6. Parameter requires_grad / Freezing** | `exp019_adaptation_core.py` (L207–211) | Freeze CNN backbone, unfreeze GRU layer 1 + head. | Pattern matched 16 parameters (48,836 params); 46 parameters remained frozen. | Parameter selection regex functioned as intended. |
| **7. model.train() / eval() State** | `exp019_adaptation_core.py` (L224, L265) | Set `model.train()` during loop, `model.eval()` after. | Set `train()` at L224, immediately broke at L243, set `eval()` at L265. | Returned model was in `eval()` mode. |
| **8. Checkpoint Loading** | `run_exp019_stage_c.py` (L258) | Restore clean backbone weights prior to adaptation. | `reset_model_weights(base_model, backbone_state)` correctly loaded the clean fold checkpoint. | Checkpoint load succeeded. |
| **9. Adaptation Loop Count** | `exp019_adaptation_core.py` (L229–263) | Execute 20 optimization iterations. | Broke on step 0. Loop ran 0 steps. | **0 iterations executed**. |
| **10. Adapted State Persistence** | `run_exp019_stage_c.py` (L259–265) | Return adapted model in-memory. | Returned unmodified `model` object. | Model returned, but weights unchanged. |
| **11. Inference Model Selection** | `run_exp019_stage_c.py` (L266) | Pass `adapted_ua` to `SlidingWindowPredictor`. | `adapted_ua` was passed to `SlidingWindowPredictor`. | Correct model instance passed, but model had zero updates. |
| **12. Evaluation Preprocessing** | `exp018_inference.py` (L15–125) | Sliding-window inference over full evaluation nights. | Ran identical sliding-window logic for ZS and UA-2. | Identical inputs processed. |
| **13. RNG / Seeding** | `run_exp019_stage_c.py` (L188–268) | Seeds 42, 123, 2024. | Seeds looped over correctly, but no stochastic operations took place. | Zero effect from seeding. |
| **14. ZS vs UA-2 Output Comparison** | `run_exp019_stage_c.py` (L267, L282) | Compare evaluation predictions and compute F1 delta. | Logits and stage predictions were identical down to floating point precision. | Metric delta was mathematically $\equiv 0.000000$. |

---

## 3. Quantitative Forensic Metrics

Empirical measurements computed using the actual Stage C checkpoints (`modelA_L1200_fold{1..5}_seed42.pt`) and cached evaluation records:

| Diagnostic Metric | Empirical Value | Audit Significance |
|---|---|---|
| **Total Calibration Nights Checked** | 94 nights (47 subjects $\times$ 2 nights) | 100% of calibration dataset audited. |
| **Total Calibration Epochs Checked** | 82,762 epochs | All epochs across calibration nights. |
| **Epochs Passing `compute_wearable_quality_mask`** | **0** (0.00%) | Filter rejected all epochs due to standardized z-score range. |
| **Dataset IHR Minimum Value** | $-3.9531$ | Global minimum in cached dataset. |
| **Dataset IHR Maximum Value** | $+9.9375$ | Global maximum in cached dataset (never reaches 30.0). |
| **Subjects with $>0$ Valid Calibration Epochs** | **0 / 47** | Universal failure across all subjects. |
| **Relative Parameter Displacement** $\frac{\|\theta_{\text{UA}} - \theta_{\text{ZS}}\|}{\|\theta_{\text{ZS}}\|}$ | **`0.00000000`** | Not a single weight parameter shifted. |
| **Maximum Absolute Parameter Difference** | **`0.0`** | Bitwise identity across all 163,940 parameters. |
| **Parameters Changed** | **`0 / 163,940` (0.00%)** | Zero parameters updated. |
| **Maximum Logit Difference on Eval Nights** | **`0.000000`** | Evaluation logits identical on identical inputs. |
| **Prediction Differences on Eval Nights** | **`0` epochs** | Zero label discrepancies across all 152 evaluation nights. |
| **Actual Optimizer Steps Taken** | **`0`** | Optimization was aborted prior to step 1. |
| **Adaptation Loss Computed** | **`None`** (never reached) | Loop broke before loss calculation. |
| **Gradients Nonzero** | **`No`** (`loss.backward()` never called) | No gradient flow occurred. |

---

## 4. Root Cause Analysis

### Code Discrepancy in `experiments/EXP-019/exp019_adaptation_core.py`

#### 1. The Wearable Quality Filter
```python
# exp019_adaptation_core.py, lines 56-58
valid_hr = (hr_mean >= 30.0) & (hr_mean <= 180.0)
valid_acc = (acc_var > 0.0)
return (valid_hr & valid_acc).astype(bool)
```
- **Flaw:** The function checked whether instantaneous heart rate fell between 30 and 180 beats per minute.
- **Data Reality:** The cached dataset ingested via `prepare_dataset_records` provides **standardized features** (z-scored with zero mean and unit variance). The maximum instantaneous heart rate value in the entire preprocessed dataset is $+9.94$ standard deviations.
- **Result:** `hr_mean >= 30.0` was never `True`. The returned mask was all-`False`.

#### 2. The Silent Loop Termination
```python
# exp019_adaptation_core.py, lines 241-243
valid_probs = probs[wearable_mask]  # (M, 4)
if valid_probs.shape[0] == 0:
    break
```
- **Flaw:** Because `wearable_mask` was completely `False`, `valid_probs.shape[0]` was 0.
- **Result:** The loop broke immediately on step 0. It did not log an error, raise an exception, or alert the pipeline; it simply exited silently and returned the unchanged model.
- **Reporting Mask:** The function's return metadata dictionary set:
  ```python
  "completed_steps": abort_step if aborted else steps
  ```
  Because `aborted` was only set to `True` if the collapse threshold triggered (which occurred *after* line 243), `aborted` remained `False`. The function erroneously reported `"completed_steps": 20` even though 0 steps were executed.

#### 3. Impact on Stage D Hyperparameter Tuning
In `run_exp019_stage_d.py`:
- All 4 grid configurations in `UA_GRID` (`lr: [1e-5, 1e-4]`, `steps: [20, 100]`) encountered the exact same step-0 break.
- Every candidate returned the exact zero-shot validation Macro-F1.
- In `select_best_config`, tie-breaking sorted by `(-score, lr, steps, scope_penalty)`. Because all scores were tied at zero-shot performance, the lowest `lr` (`1e-5`) and fewest `steps` (`20`) were selected automatically for all 5 folds.

---

## 5. Minimum Corrective Actions Required by `RESEARCH_DIRECTION_v4`

Under the Phase A2 mandate of `RESEARCH_DIRECTION_v4_FULL.md`:

1. **Maintain Confirmatory Immuntability:**
   - The historical EXP-019 Stage C result files (`exp019_stage_c_results.json`, `exp019_subject_metrics.csv`, `exp019_summary.csv`) and locked preregistration `PREREG_EXP-019_H-A.md` remain **byte-for-byte unchanged**.
   - No experimental rerun or post-hoc alteration of EXP-019 is authorized.

2. **Formal Documentation Classification:**
   - In all research governance documents, the status of Secondary Hypothesis S3 must be updated to:
     > **`UNRESOLVED / INVALIDATED (Implementation No-Op)`**
   - All claims in research summaries, literature notes, and drafts asserting that "unsupervised InfoMax adaptation fails" or that "unsupervised alignment on 2 nights is ineffective without supervisory signal" must be retracted or qualified as uninformative.

3. **Codebase Remediation (for future exploratory or replication runs):**
   - **Quality Mask Fix:** In `exp019_adaptation_core.py::compute_wearable_quality_mask`, replace the raw BPM threshold with a valid standardized signal check (e.g., physiological z-score boundaries $|z| \le 5.0$, or relying strictly on signal validity and accelerometry variance `acc_var > 0.0`).
   - **No-Op Guard Assertion:** Add a strict precondition in `adapt_unsupervised_infomax`:
     ```python
     if wearable_mask.sum() == 0:
         raise RuntimeError("Adaptation batch contains 0 valid wearable epochs. Cannot adapt.")
     ```
   - **Displacement Assertion:** Assert that at least one optimizer step took place and that $\|\theta_{\text{adapted}} - \theta_{\text{orig}}\| > 0$.
   - **Accurate Step Reporting:** Fix `completed_steps` in the returned info dictionary so that early exits report actual steps taken.

4. **Future Scientific Re-testing:**
   - Any legitimate evaluation of unsupervised adaptation (whether test-time adaptation, InfoMax, or entropy minimization) must be executed under a new, dedicated Phase D protocol (e.g., exploratory EXP-020 or a formal replication protocol), with the corrected quality mask and explicit gradient verification.
