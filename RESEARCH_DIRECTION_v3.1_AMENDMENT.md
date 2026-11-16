# Amendment v3.1 to `RESEARCH_DIRECTION_v3_FULL.md`

**Date:** 2026-11-15
**How to apply:** Do not rewrite v3. Add the rows in section A to v3 section 15 (Amendments) and append the errata in section B to `DECISIONS.md` and `EXPERIMENTS.md` (append-only, dated). Commit together with `PREREG_EXP-019_H-A.md`.
**Source of facts:** `HANDOFF.md`, `EXPERIMENTS.md`, `DECISIONS.md` as uploaded on 2026-11-15. Items marked `TO VERIFY` are not confirmed by those files.

---

## A. Amendments to v3

| # | v3 section | Change | Reason (evidence) |
|---|---|---|---|
| A1 | 6 (task definition) | Primary task is **4-class** (Wake/Light/Deep/REM). Valid labels are 0 to 3; unknown and out-of-range epochs are masked. The 5-class statement in v3 is withdrawn. The definition of "Light" in terms of AASM stages: `TO VERIFY`. Whether the published 71.04% (Table V) uses the same 4-class definition: `TO VERIFY`. | Pipeline in EXP-012, EXP-014, EXP-017 and EXP-018 uses `targets < 4`. |
| A2 | 7 (frozen test set) | The EXP-012 test set (11 subjects, 58 usable nights) is **no longer untouched**. Test-access log: EXP-004 one-time test pass (DEC-008, test Macro-F1 0.6648); EXP-012 final evaluation (3 seeds, DEC-018); EXP-013 comparison on the same split (`TO VERIFY` whether it used the test set); EXP-001 to EXP-003 notebook runs (`TO VERIFY`). The same 11 subjects were then used inside EXP-014 and EXP-018 Phase 2 cross-validation folds. Final claims therefore rest on **out-of-fold nested subject-level CV across all 47 subjects** plus external validation. The EXP-012 split is retained only as a historical reference. | DEC-018, DEC-020, DEC-025. |
| A3 | 7 (split audit) | Audit result so far: EXP-012, EXP-014 and EXP-018 Phase 2 use subject-disjoint splits. EXP-017 and EXP-018 Phase 1 use the 31/5/11 split (hash `bf33573623f47f3f`) with test locked. EXP-001 to EXP-004 (notebook era): `TO VERIFY`; any run that split by night is marked invalid. | DEC-018, DEC-020; EXP-017/018 entries. |
| A4 | 6, 4 (context) | The H-A backbone and any SLAMSS-IFS comparison use **full-night** context (L = 1200, sliding-window inference), not the 5-hour T = 600 crop. Report all cropped results as cropped. | DEC-021: the crop dropped about 28% of labeled epochs (selective REM depletion, Deep enrichment). EXP-018 Phase 2 out-of-fold: L600 Macro-F1 0.6306 (accuracy 65.45%), L1200 0.6479 (66.18%). |
| A5 | 4 (H-C) | The planned H-C replication (old EXP-022) is **dropped**. EXP-018 Phase 2 carried its own pre-registered decision rule (Δ ≥ +1.0 pp with a strictly positive CI lower bound) and passed it (+5.33 pp [4.28, 6.39], 46/47 subjects). Classify it as confirmatory for L600 vs L60 **if** `git log` shows the decision rule was committed before the Phase 2 runs: `TO VERIFY`. Otherwise it stays exploratory and the replication is reinstated. | `HANDOFF.md` and EXP-018 entries. |
| A6 | 4 (H-F), 11 | Architecture search is **closed**. EXP-014 (75 runs, 47 subjects) found no detectable benefit from gating, transition or GRL (Model E vs A: Macro-F1 -0.0069, 95% CI [-0.0213, +0.0059]; 23 vs 23 subjects). H-F as written is demoted to optional. Any future architecture test must be single-factor, parameter-matched within ±2% (DEC-019 principle), and pre-registered with a decision rule before running. NAS Iteration 1 is reclassified as an exploratory smoke test (see B). | DEC-019, DEC-020. |
| A7 | 8 (baselines, wording) | Retract the "Case B parity" framing in DEC-018 as a general claim. It rests on one 11-subject split (68.44% accuracy). Out-of-fold CV gives about 66.2% to 66.6% accuracy for matched Model A, which is about 4.4 to 4.9 pp below the published 71.04%, under mismatched protocols (crop, class definition and splits not verified as matched). Fold accuracies range about 62.5% to 72.3%, and 5-subject validation splits swung to 70.5% to 72.8% in EXP-017, so **no comparison to a published number may be made from a single small split**. Allowed wording until a protocol-matched re-run exists: "achieved lower/higher accuracy than the reported SLAMSS-IFS result under a different protocol." The official-repository reproduction (DEC-011, Classification B) must be independently re-run before any publication claim (as `HANDOFF.md` already states). | DEC-011, DEC-018, DEC-020, EXP-017. |
| A8 | 9 (power) | The +1.5 pp practical threshold is conditional on the paired SD (about 3.6 pp, from EXP-018 Phase 2). EXP-019 re-estimates the SD in Stage D and raises the threshold in the lock commit if the SD exceeds 5 pp. | `PREREG_EXP-019_H-A.md` section 6. |
| A9 | 11 (roadmap) | Replace the v3 Phase 2/3 rows: **Phase 2** = cite EXP-018 Phase 2 (A5). **Phase 3** = EXP-019 (H-A, `PREREG_EXP-019_H-A.md`), then the learning-curve and heterogeneity analyses (same experiment, descriptive). External validation, clinical metrics and a protocol-matched SLAMSS-IFS comparison remain as in v3. | Section A rows above. |

---

## B. Append-only errata (copy as written, then adjust the bracketed items after checking the JSON)

### B1. Append to `DECISIONS.md` under DEC-026

```
### ERRATUM to DEC-026 (2026-11-15)

1. Mismatch with H-A: the MSTF-Pre block is a within-night multi-scale temporal
   convolution over epochs. It uses no other nights and is not an operationalization of
   H-A (multi-night within-person adaptation). H-A is tested in PREREG_EXP-019_H-A.md.
2. Not pre-registered: no pre-registration block or decision rule existed for this change.
   It must not be described as pre-registered.
3. Evidence is insufficient: single fold, single seed, 1 training epoch, evaluated on the
   outer test fold; models at about 44 to 47% accuracy. The primary-run Macro-F1 gain
   (+5.01 pp, 0.3056 -> 0.3556) was not reproduced in the independent re-run, which
   reported Macro-F1 0.3764 vs 0.3882 [confirm order in nas_iteration1_eval_results.json:
   if mutated is listed first, as in the accuracy and loss rows, the re-run difference is
   -1.18 pp] and a baseline value that differs from the primary run [0.3056 vs 0.3882 if
   the second figure is the baseline], so run-to-run noise exceeds the claimed effect.
   Only an accuracy difference (+0.33 pp) and a loss difference (-0.0009) were confirmed.
4. Capacity confound: +17.54% parameters violates the matched-capacity principle of DEC-019.
5. Selection on test data: outer-test-fold results must not guide architecture selection.
6. Status: exploratory smoke test with no performance claim. NAS Iteration 2 and later are
   not authorized (see EXP-014, DEC-020). "VICTORY CONFIRMED" refers to code, test-suite
   and provenance checks only, not to scientific validity.
```

### B2. Append to the "NAS Iteration 1" entry in `EXPERIMENTS.md`

```
ERRATUM (2026-11-15): Reclassified as an exploratory smoke test. The reported "+5.01 pp"
Macro-F1 gain was not reproduced by the independent re-run (see ERRATUM to DEC-026).
Class F1 gains for REM and Wake in the primary run are likewise unreplicated (REM F1 0.1287
in the re-run vs 0.1872 in the primary run). The entry above must not be cited as evidence
for H-A or for any architecture benefit.
```

---

## C. Pre-lock checklist (resolve before committing v3.1 and EXP-019)

1. `git log` for EXP-018's decision rule vs the Phase 2 run timestamps (A5).
2. Confirm EXP-018 Phase 2 L = 1200 checkpoints exist for 5 folds × 3 seeds (EXP-019 section 3).
3. Fill: manifest hash, number of eligible subjects and evaluation nights per subject, label variable, definition of "Light".
4. Fill the test-access log completely (A2) and the EXP-001 to EXP-004 split audit (A3).
5. Check the published Table V class definition and protocol (A1, A7).
6. Run Stage D, compute the SD, and finalize the practical threshold.
7. Freeze the analysis script, record its commit hash, then run Stage C once.
