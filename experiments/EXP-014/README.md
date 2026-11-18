# EXP-014: 75-Run 5-Fold Cross-Validation & Autonomous NAS Loop

## Overview
Hosts the 75-run 5-fold cross-validation benchmark across all 47 subjects (5 model variants x 5 folds x 3 seeds = 75 runs) and the autonomous Neural Architecture Search (NAS) continuous iteration framework. Evaluates structural mutations (Iterations 1 through 19) targeting cross-modal autonomic synergy (H-F) and within-person adaptation (H-A) under strict parameter constraints ($\le 196,728$ parameters).

## Interface
- `run_exp014_cv` (`run_exp014_cv.py`): Orchestrates full 5-fold cross-validation across subjects.
- `MultiScaleConventionalCNNBiGRU` (`nas_iteration1_model.py`): Multi-scale temporal dilation mutation.
- `DualScaleLocalBasalStandardization` (`nas_iteration8_model.py`): Running 7.5m / 45.5m dual-scale basal pyramid.
- `BoundedMarginCrossModalSynergy` (`nas_iteration12_model.py`): Learned asymmetric synergy margins with straight-through estimator.
- `PriorCalibratedBoundedMarginPyramidCNNBiGRU` (`nas_iteration17_model.py`): Prior-calibrated logit balancing (PLB).
- `UnbiasedHypersphericalBoundedMarginPyramidCNNBiGRU` (`nas_iteration18_model.py`): Zero-bias hyperspherical cosine classification head on $\mathbb{S}^{63}$.
- `AutonomicHomeostasisBoundedMarginPyramidCNNBiGRU` (`nas_iteration19_model.py`): Autonomic & Somatomotor Homeostasis Contrast (AH-CASCI).
- `evaluate_nas_iteration*_proxy.py`: 50-epoch Fold 1 proxy evaluation harnesses.
- `verify_candidate_*.py`: Pre-flight unit verification suites for candidate models.

## Invariants
- Parameter ceiling capped at $\le 196,728$ parameters ($+20\%$ over baseline, maintaining $>96\%$ compression vs 5.47M SLAMSS-IFS).
- Proxy evaluation gate: Proxy Accuracy $\ge 70.0\%$ and Weighted-F1 $\ge 0.6900$ required for 5-fold escalation.
- Fold partitions strictly disjoint by Subject ID; zero data leakage across folds.

## Side Effects
- Saves candidate checkpoints (`nas_iteration*_best.pt`, `nas_iteration*_best_proxy_ckpt.pt`).
- Records evaluation metrics in `nas_iteration*_proxy_results.json` and `EXP014_RESULTS.csv`.

## Verification
- Run: `PYTHONPATH=. ./.venv/bin/pytest tests/test_nas_iteration1_eval_stress.py tests/test_dataset_gate.py`
