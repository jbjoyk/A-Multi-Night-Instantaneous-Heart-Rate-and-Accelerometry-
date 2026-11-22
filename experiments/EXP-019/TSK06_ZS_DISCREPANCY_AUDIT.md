# Forensic Audit: TSK-06 Calibration-Depth Zero-Shot Baseline Discrepancy

**Audit Date:** 2026-11-22  
**Auditor:** Strict Research Integrity Auditor  
**Governance Authority:** `RESEARCH_STATE.md`, `DECISIONS.md` (DEC-035, DEC-037, DEC-038)  
**Target Investigation:** Discrepancy between locked EXP-019 $k=2$ baseline ($\text{ZS} = 64.58\%$, $\text{FT-2} = 65.59\%$, $\Delta = +1.01\text{ pp}$) and TSK-06 / EXP-019-K executed sensitivity record ($\text{ZS} = 62.86\%$, $\text{FT} = 65.58\%$, $\Delta = +2.72\text{ pp}$).

---

## 1. Executive Summary & Root Cause Finding

A rigorous forensic audit was conducted across `experiments/EXP-019/run_tsk06_k_curve.py`, `experiments/EXP-019/exp019_k_learning_curve_executed.csv`, `experiments/EXP-019/exp019_k_curve_subject_metrics.csv`, and `experiments/EXP-019/exp019_subject_metrics.csv`.

### Key Findings:
1. **Root Cause Identified (Model Weight Leakage Across Subjects in Zero-Shot Predictor):**
   In `run_tsk06_k_curve.py`, `predictor_zs = SlidingWindowPredictor(base_model, ...)` was instantiated **once per fold and seed at line 211**, before the loop over outer test subjects (`for sub in fold_subjects:` at line 213).
   While `base_model` was properly reset before fine-tuning in the inner $k$-loop (`reset_model_weights(base_model, backbone_state)` at line 246), **`base_model` was NOT reset before the Zero-Shot evaluation block (lines 217–231)** when iterating to subsequent subjects.
2. **Empirical Consequence (Prior-Subject Contamination = De Facto Donor Control):**
   - For the **first subject in each fold** (e.g. `Bidslab08` in Fold 1, `Bidslab07` in Fold 2, `Bidslab06` in Fold 3, `Bidslab00` in Fold 4, `Bidslab01` in Fold 5), `base_model` remained in its pristine zero-shot state. For these 5 subjects, evaluated Zero-Shot Macro-F1 matched the locked EXP-019 ZS value down to 4 decimal places ($\Delta = 0.0000$).
   - For all **subsequent 41 subjects**, `base_model` was mutated in-place by the fine-tuning steps of the preceding subject (specifically, fine-tuned on the preceding subject's $k=4$ nights).
   - In sleep staging, fine-tuning on another subject is functionally identical to the donor specificity control (`FT-other-2`). Because inter-subject donor fine-tuning degrades performance by $-1.72\text{ pp}$ below true zero-shot, the mean of this contaminated baseline across all 46 subjects evaluated to **$62.8601\%$**, which is **virtually identical to the locked `FT-other-2` mean of $62.8598\%$**.
3. **Fine-Tuning Arm (`FT_Ek_MacroF1`) Was Uncontaminated:**
   Because line 246 explicitly called `reset_model_weights(base_model, backbone_state)` immediately prior to each adaptation pass, all fine-tuned models were adapted from the clean, uncorrupted population checkpoint. Consequently, the $k=2$ fine-tuning mean of **$65.58\%$** matched locked EXP-019's **$65.59\%$** ($+0.01\text{ pp}$ difference attributable to single-float rounding).
4. **Apparent Inflation of $\Delta_{k=2}$:**
   The apparent delta reported in `exp019_k_learning_curve_executed.csv` and `EXPERIMENTS.md` §8.1 for available cohort $k=2$:
   $$\Delta_{\text{reported}} = 65.58\% - 62.86\% = \mathbf{+2.72\text{ pp}}$$
   was effectively computing $\text{FT} - \text{FT-other}$ (which was $+2.73\text{ pp}$ in locked EXP-019 Hypothesis S1), rather than $\text{FT} - \text{ZS}$ ($+1.01\text{ pp}$).

---

## 2. Evaluation of Candidate Hypotheses

| Candidate Hypothesis | Verdict | Forensic Evidence & Code Analysis |
|---|---|---|
| **(a) Wrong column lookup** | **REFUTED** | `run_tsk06_k_curve.py` did not read from `exp019_subject_metrics.csv` at runtime; it executed complete forward passes for all models from raw checkpoints and cache. The value $62.86\%$ was computed de novo, not copied from a column. |
| **(b) Epoch-pooled vs. night- vs. subject-averaged aggregation** | **REFUTED** | Both `run_exp019_stage_c.py` (lines 55–70) and `run_tsk06_k_curve.py` (lines 108–122) pooled valid epochs $[0..3]$ across all evaluation nights within each subject to compute per-subject Macro-F1, followed by unweighted subject-level arithmetic averaging across participants. |
| **(c) Circadian feature flags (`time_feat`)** | **REFUTED** | Both scripts explicitly set `neutralize_circadian=True` in `SlidingWindowPredictor` (line 218 in `run_exp019_stage_c.py`; lines 211 and 262 in `run_tsk06_k_curve.py`). Both zeroed out the time channel identically. |
| **(d) Cross-subject in-place model mutation (Actual Root Cause)** | **CONFIRMED** | Line 211 initialized `predictor_zs(base_model)` outside the subject loop. `adapt_supervised_ft` modified `base_model` in-place. `reset_model_weights` was placed inside the inner $k$-loop (line 246) but omitted before line 217. 41 of 46 subjects ran ZS inference on weights adapted to the prior subject. |

---

## 3. Detailed Code Line Citations (`run_tsk06_k_curve.py`)

### The Defect:
```python
# experiments/EXP-019/run_tsk06_k_curve.py
204: # Instantiate Model A backbone
205: base_model = ConventionalCNNBiGRU().to(args.device)
206: ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=True)
207: base_model.load_state_dict(ckpt)
208: base_model.eval()
209: backbone_state = copy.deepcopy(base_model.state_dict())
210: 
211: predictor_zs = SlidingWindowPredictor(base_model, context_len=1200, neutralize_circadian=True, device=torch.device(args.device))
212: 
213: for sub in fold_subjects:
214:     nights = records_by_sub[sub]
215:     n_nights = len(nights)
216: 
217:     # 1. Zero-Shot Evaluation
218:     # E_4 evaluation (Nights 5+)
219:     e4_records = nights[4:] if n_nights >= 5 else []
220:     zs_e4_f1, zs_e4_acc = (evaluate_records(predictor_zs, e4_records) if e4_records else (np.nan, np.nan))
221: 
222:     # E_k evaluations
223:     zs_ek_metrics = {}
224:     for k in K_VALUES_DEFAULT:
225:         ek_recs = nights[k:] if n_nights > k else []
226:         if ek_recs:
227:             f1, acc = evaluate_records(predictor_zs, ek_recs)
...
233:     # 2. Adaptation for each k in {1, 2, 3, 4}
234:     for k in K_VALUES_DEFAULT:
...
245:         # Restore clean backbone state
246:         reset_model_weights(base_model, backbone_state)
...
250:         adapted_model = adapt_supervised_ft(...)
```

Notice:
1. At line 211, `predictor_zs` wraps `base_model`.
2. When `sub` completes $k=4$, `base_model` has been mutated in-place by `adapt_supervised_ft`.
3. When the next `sub` begins at line 213, `base_model` is **NOT** reset to `backbone_state`.
4. Therefore, lines 220 and 227 run inference using the preceding subject's adapted weights.

---

## 4. Empirical Verification & Impact Analysis

### A. Spot-Check of Fold First Subjects vs. Subsequent Subjects
Auditing the difference $\Delta_{\text{diff}} = \text{ZS\_Ek\_MacroF1} - \text{Locked\_ZS}$:

| Fold | Subject Index | Subject ID | Locked ZS | TSK-06 ZS ($k=2$) | $\Delta_{\text{diff}}$ | Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| 1 | 1st (clean) | `Bidslab08` | 0.5984 | 0.5984 | **-0.0000** | Pristine Backbone |
| 1 | 2nd (polluted) | `Bidslab30` | 0.6550 | 0.6540 | -0.0010 | Mutated by Bidslab08 |
| 1 | 4th (polluted) | `Bidslab39` | 0.7587 | 0.6878 | **-0.0709** | Mutated by Bidslab36 |
| 1 | 8th (polluted) | `Bidslab64` | 0.6423 | 0.5340 | **-0.1083** | Mutated by Bidslab47 |
| 2 | 1st (clean) | `Bidslab07` | 0.6475 | 0.6475 | **-0.0000** | Pristine Backbone |
| 2 | 2nd (polluted) | `Bidslab10` | 0.7153 | 0.6155 | **-0.0998** | Mutated by Bidslab07 |
| 3 | 1st (clean) | `Bidslab06` | 0.7616 | 0.7616 | **-0.0000** | Pristine Backbone |
| 4 | 1st (clean) | `Bidslab00` | 0.7090 | 0.7090 | **+0.0000** | Pristine Backbone |
| 5 | 1st (clean) | `Bidslab01` | 0.6224 | 0.6224 | **+0.0000** | Pristine Backbone |

### B. Clean Recalculation (Zero-Shot Reset Invariant Enforced)
When `reset_model_weights(base_model, backbone_state)` is strictly executed before `predictor_zs` evaluation:
1. **Available Cohort $k=2$ ($N=46$, 152 nights):**
   - Clean Zero-Shot Macro-F1 = **$64.58\%$**
   - Fine-Tuned Macro-F1 = **$65.58\%$**
   - True $\bar{\Delta}_{k=2} = \mathbf{+1.00\text{ pp}}$ (reconciling with locked EXP-019's $+1.01\text{ pp}$ within $0.01\text{ pp}$).
2. **Common Cohort ($N=33$, $E_4$, 64 nights):**
   - Clean Zero-Shot Macro-F1 on $E_4$ = **$63.54\%$** (vs. contaminated $62.29\%$).
   - Fine-Tuned Progression on $E_4$:
     * $k=1$: $65.28\%$ ($\Delta_{\text{clean}} = \mathbf{+1.74\text{ pp}}$)
     * $k=2$: $65.41\%$ ($\Delta_{\text{clean}} = \mathbf{+1.87\text{ pp}}$)
     * $k=3$: $65.82\%$ ($\Delta_{\text{clean}} = \mathbf{+2.28\text{ pp}}$)
     * $k=4$: $66.25\%$ ($\Delta_{\text{clean}} = \mathbf{+2.71\text{ pp}}$)
3. **Integrity of Core Hypothesis Tests (Page's $L$ & Replication):**
   - **Page's $L$-trend test:** In `run_tsk06_k_curve.py` line 343, Page's $L$ test was computed strictly on the adapted trajectories (`ft_matrix`), which were fine-tuned from clean backbones. The monotonic progression ($65.28\% \to 65.41\% \to 65.82\% \to 66.25\%$) remains strictly monotonic and statistically significant ($p = 0.01187$).
   - **van der Aar Replication ($k=3$):** Even against the clean baseline ($63.54\%$), the 3-night adaptation gain is $\Delta = +2.28\text{ pp}$, which easily satisfies the pre-registered replication threshold ($\ge +1.00\text{ pp}$, $p < 0.05$).

---

## 5. Recommended Erratum & Reconciliation Text for `DECISIONS.md`

The following text is recommended to be ratified as **DEC-039**:

```markdown
### DEC-039 (2026-11-22): Forensic Erratum on TSK-06 Calibration-Depth Baseline & Reconciliation of Available Cohort k=2

- Date: 2026-11-22
- Context: Forensic audit of TSK-06 execution script (`experiments/EXP-019/run_tsk06_k_curve.py`) resolving the discrepancy between locked EXP-019 k=2 (+1.01 pp, ZS = 64.58%, FT = 65.59%) and TSK-06 executed summary (+2.72 pp, ZS = 62.86%, FT = 65.58%).
- Finding: An in-place weight mutation defect in `run_tsk06_k_curve.py` caused `predictor_zs` on subjects 2..N of each fold to evaluate a model fine-tuned on the prior subject (de facto donor control), yielding an average baseline of 62.86% (matching `FT-other-2`). All fine-tuned models (`FT_Ek_MacroF1`, `FT_E4_MacroF1`) were correctly reset from the clean backbone and are completely valid.
- Reconciliation & Impact:
  1. The true available-cohort k=2 gain over clean Zero-Shot is +1.00 pp (64.58% -> 65.58%), confirming perfect harmony with locked EXP-019 (+1.01 pp).
  2. The true common-cohort (N=33, E4) Zero-Shot baseline is 63.54%, yielding clean adaptation gains of +1.74 pp (k=1), +1.87 pp (k=2), +2.28 pp (k=3), and +2.71 pp (k=4).
  3. All primary hypotheses ratified under DEC-035 remain valid: Page's L trend test (p = 0.01187) and van der Aar replication (k=3 gain +2.28 pp >= +1.00 pp) remain robustly supported.
- Governance Action: Preserve historical executed CSVs under the immutable data rule; record this erratum in DECISIONS.md and EXPERIMENTS.md.
```
