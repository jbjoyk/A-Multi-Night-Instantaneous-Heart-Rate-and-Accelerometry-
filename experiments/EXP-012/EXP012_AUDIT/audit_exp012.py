"""
EXP-012 Comprehensive Independent Audit Script
Script: EXP012_AUDIT/audit_exp012.py

This script independently audits the frozen EXP-012 results:
1. Verifies saved artifacts and discovers pickle structure.
2. Audits real vs. padded/invalid windows (resolves 34,800 vs 34,241 issue).
3. Executes Assertions A through I.
4. Regenerates all test metrics from raw predictions and targets across all 3 seeds.
5. Generates confusion matrices (raw and normalized) and figures.
6. Audits per-subject metrics and calculates pooled vs. mean-of-subject accuracy.
7. Executes subject-level bootstrap (B=1000) for 95% CIs.
8. Audits boundary vs. non-boundary error dynamics.
9. Audits ECE methodology and parameters.
10. Generates detailed old vs. recomputed comparison CSV and audit JSON.
"""

import os
import sys
import glob
import json
import pickle
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, precision_recall_fscore_support
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIT_DIR = os.path.join(BASE_DIR, "EXP012_AUDIT")
RESULTS_DIR = os.path.join(AUDIT_DIR, "results")
FIGURES_DIR = os.path.join(AUDIT_DIR, "figures")
PICKLE_PATH = os.path.join(BASE_DIR, "exp012_test_results.pkl")
JSON_PATH = os.path.join(BASE_DIR, "exp012_test_results.json")
SUBJECT_CSV_PATH = os.path.join(BASE_DIR, "exp012_subject_metrics.csv")
DATA_DIR = os.path.join(BASE_DIR, "a-multi-night-instantaneous-heart-rate-and-accelerometry-dataset-with-eeg-sleep-stage-labels-1.0.0")
CACHE_DIR = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    print("=" * 80)
    print("STARTING INDEPENDENT AUDIT OF EXP-012")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Inspect Pickle Structure
    # -------------------------------------------------------------
    assert os.path.exists(PICKLE_PATH), f"Missing artifact: {PICKLE_PATH}"
    with open(PICKLE_PATH, "rb") as f:
        pickle_data = pickle.load(f)

    print("\n--- Discovered Pickle Structure ---")
    print(f"Top-level keys (seeds): {list(pickle_data.keys())}")
    seeds = sorted(list(pickle_data.keys()))
    assert seeds == [42, 123, 2024], f"Unexpected seeds: {seeds}"

    for s in seeds:
        print(f"\nSeed {s} Keys: {list(pickle_data[s].keys())}")
        print(f"  y_true shape: {pickle_data[s]['y_true'].shape}, dtype: {pickle_data[s]['y_true'].dtype}")
        print(f"  y_pred shape: {pickle_data[s]['y_pred'].shape}, dtype: {pickle_data[s]['y_pred'].dtype}")
        print(f"  subjects shape: {pickle_data[s]['subjects'].shape}, dtype: {pickle_data[s]['subjects'].dtype}")

    # -------------------------------------------------------------
    # 2. Real vs Padded Windows Audit (34,800 vs 34,241 Investigation)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 1: REAL VS PADDED WINDOWS AUDIT")
    print("=" * 80)

    # Inspect test split records to trace where 34,800 came from
    sys.path.insert(0, BASE_DIR)
    from slamss_ifs_baseline import make_reproducible_split
    from slamss_ifs_train import load_night_features

    mat_files = sorted(glob.glob(os.path.join(DATA_DIR, "Bidslab*", "*", "labels.mat")))
    all_nights = [(f.split(os.sep)[-3], f.split(os.sep)[-2]) for f in mat_files]
    all_subjects = sorted(list(set(s for s, n in all_nights)))
    split = make_reproducible_split(all_subjects, split_type="paper", seed=0)
    test_subs = split["test"]

    test_records = []
    for s, n in all_nights:
        if s in test_subs:
            rec = load_night_features(CACHE_DIR, s, n)
            if rec is not None:
                test_records.append(rec)

    n_test_nights = len(test_records)
    total_slots_600 = n_test_nights * 600
    print(f"Test Nights count: {n_test_nights}")
    print(f"Total slots when padded/truncated to T=600: {n_test_nights} * 600 = {total_slots_600}")

    all_y_600 = np.concatenate([r["y"][:600] for r in test_records])
    unique_targets, target_counts = np.unique(all_y_600, return_counts=True)
    target_dict = dict(zip([int(u) for u in unique_targets], [int(c) for c in target_counts]))

    print(f"Raw Target Distribution across all 34,800 slots:")
    for k, v in target_dict.items():
        print(f"  Label {k:4d}: {v:6d} windows")

    padded_invalid_count = target_dict.get(-100, 0)
    real_window_count = sum(v for k, v in target_dict.items() if k in [0, 1, 2, 3])

    print(f"\nAudit Findings on Window Counts:")
    print(f"  Total Padded/Truncated Slots: {total_slots_600}")
    print(f"  Invalid / Padded Labels (-100): {padded_invalid_count}")
    print(f"  Valid / Real Staging Windows:   {real_window_count}")
    print(f"  Identity check: {real_window_count} + {padded_invalid_count} = {real_window_count + padded_invalid_count} (== {total_slots_600})")

    # Real windows per class
    real_per_class = [target_dict[c] for c in [0, 1, 2, 3]]
    print(f"  Real windows per class [Wake, Light, Deep, REM]: {real_per_class}")
    print(f"  Old reported support in table: [3819, 13816, 8294, 8871]")
    print(f"  Old reported REM support - Real REM support: 8871 - {target_dict[3]} = {8871 - target_dict[3]}")
    print(f"  NOTE: 8871 was derived by subtracting Wake, Light, Deep from 34,800! (34800 - 3819 - 13816 - 8294 = 8871).")
    print(f"  The 559 padded/invalid windows were falsely absorbed into REM class support in the old text table!")

    # Real windows per subject
    real_per_subject = {}
    for r in test_records:
        s = r["subject"]
        y_s = r["y"][:600]
        valid_s = int(np.sum((y_s >= 0) & (y_s < 4)))
        real_per_subject[s] = real_per_subject.get(s, 0) + valid_s

    print("\nReal windows per test subject:")
    for s_id in sorted(real_per_subject.keys()):
        print(f"  {s_id}: {real_per_subject[s_id]} windows")

    # -------------------------------------------------------------
    # 3. Required Assertions A through I
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 2: EXECUTING REQUIRED ASSERTIONS A - I")
    print("=" * 80)

    # Assertion A: Real-window count is computed from the mask/valid filtering
    assert real_window_count == 34241, f"Assertion A Failed: Expected 34241 real windows, got {real_window_count}"
    print("Assertion A PASSED: Real-window count is exactly 34,241 (excluding 559 invalid slots).")

    for s in seeds:
        yt = pickle_data[s]["y_true"]
        yp = pickle_data[s]["y_pred"]
        cm_raw = np.array(pickle_data[s]["confusion_matrix_raw"])

        # Assertion B: Confusion matrix sum equals real windows
        assert cm_raw.sum() == real_window_count, f"Assertion B Failed for seed {s}: {cm_raw.sum()} != {real_window_count}"

        # Assertion C: Row totals equal actual target support
        row_totals = cm_raw.sum(axis=1).tolist()
        assert row_totals == real_per_class, f"Assertion C Failed for seed {s}: {row_totals} != {real_per_class}"

        # Assertion D: Accuracy computed from CM diagonal equals accuracy computed from predictions
        acc_cm = float(np.diag(cm_raw).sum() / cm_raw.sum())
        acc_direct = float(accuracy_score(yt, yp))
        assert abs(acc_cm - acc_direct) < 1e-12, f"Assertion D Failed for seed {s}: {acc_cm} != {acc_direct}"

        # Assertion F: Per-seed real-window class supports are identical
        assert len(yt) == real_window_count, f"Assertion F Failed for seed {s}: len(yt) = {len(yt)}"
        assert list(np.bincount(yt, minlength=4)) == real_per_class, f"Assertion F Failed for seed {s}"

        # Assertion G: No padded window contributes to metrics (yt contains only valid [0..3])
        assert np.all((yt >= 0) & (yt < 4)), f"Assertion G Failed for seed {s}: Invalid labels found in yt"

    print("Assertion B PASSED: Confusion matrix sum equals 34,241 across all seeds.")
    print("Assertion C PASSED: Row totals equal ground-truth class supports [3819, 13816, 8294, 8312].")
    print("Assertion D PASSED: Accuracy from CM diagonal equals accuracy from predictions.")
    print("Assertion F PASSED: Class supports are bitwise identical across all seeds.")
    print("Assertion G PASSED: Zero padded/invalid windows exist in evaluation arrays.")

    # Assertion E: Check whether old reported accuracy matches recomputed accuracy
    old_reported_accs = {42: 0.6986, 123: 0.6827, 2024: 0.6718}
    for s in seeds:
        recomputed_acc = accuracy_score(pickle_data[s]["y_true"], pickle_data[s]["y_pred"])
        diff = abs(recomputed_acc - old_reported_accs[s])
        print(f"Assertion E (Seed {s}): Old reported = {old_reported_accs[s]}, Recomputed = {recomputed_acc:.6f}, Diff = {diff:.6f}")
        # Note: Old reported was rounded to 4 decimals (e.g. 69.86% = 0.6986)
        assert diff < 0.0005, f"Assertion E Discrepancy too large: {diff}"
    print("Assertion E PASSED: Old reported accuracy matched recomputed accuracy to 4 decimal rounding.")

    # Assertion H: Boundary + non-boundary real windows = total real windows
    # Reconstruct boundary array: boundary is defined where y_t != y_{t-1} within each night
    # We trace exact night boundaries
    night_boundaries = []
    running_idx = 0
    all_is_boundary = []

    for r in test_records:
        y_slice = r["y"][:600]
        valid_mask = (y_slice >= 0) & (y_slice < 4)
        valid_y = y_slice[valid_mask]
        n_valid = len(valid_y)

        # Boundary within this night: y_t != y_{t-1}
        # In run_exp012:
        # y_prev = targets[:, :-1]
        # y_curr = targets[:, 1:]
        # is_bound[:, 1:] = (y_prev != y_curr) & (y_prev >= 0) & (y_curr >= 0)
        # then is_bound was masked by valid
        y_prev = y_slice[:-1]
        y_curr = y_slice[1:]
        bound_raw = np.zeros(len(y_slice), dtype=bool)
        bound_raw[1:] = (y_prev != y_curr) & (y_prev >= 0) & (y_prev < 4) & (y_curr >= 0) & (y_curr < 4)
        bound_valid = bound_raw[valid_mask]
        all_is_boundary.extend(bound_valid)

    is_boundary_arr = np.array(all_is_boundary, dtype=bool)
    n_boundary = int(np.sum(is_boundary_arr))
    n_stable = int(np.sum(~is_boundary_arr))

    assert n_boundary + n_stable == real_window_count, f"Assertion H Failed: {n_boundary} + {n_stable} != {real_window_count}"
    assert n_boundary == 1379, f"Assertion H Failed: Expected 1379 boundary windows, got {n_boundary}"
    assert n_stable == 32862, f"Assertion H Failed: Expected 32862 stable windows, got {n_stable}"
    print(f"Assertion H PASSED: Boundary ({n_boundary}) + Stable ({n_stable}) = Total Real Windows ({real_window_count}).")

    # Assertion I: Window-weighted error rate == 1 - accuracy
    for s in seeds:
        yt = pickle_data[s]["y_true"]
        yp = pickle_data[s]["y_pred"]
        correct = (yt == yp)
        err_b = 1.0 - correct[is_boundary_arr].mean()
        err_s = 1.0 - correct[~is_boundary_arr].mean()
        weighted_err = (err_b * n_boundary + err_s * n_stable) / real_window_count
        overall_err = 1.0 - accuracy_score(yt, yp)
        assert abs(weighted_err - overall_err) < 1e-12, f"Assertion I Failed for seed {s}: {weighted_err} != {overall_err}"
    print("Assertion I PASSED: Window-weighted boundary/stable error rate exactly equals 1 - accuracy across all seeds.")

    # -------------------------------------------------------------
    # 4. Regenerate All EXP-012 Metrics
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 3: REGENERATING ALL EXP-012 METRICS FROM SCRATCH")
    print("=" * 80)

    per_seed_metrics = []
    per_class_metrics = []

    stage_names = ["Wake", "Light", "Deep", "REM"]

    for s in seeds:
        yt = pickle_data[s]["y_true"]
        yp = pickle_data[s]["y_pred"]

        acc = float(accuracy_score(yt, yp))
        macro_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(yt, yp, average="weighted", zero_division=0))
        kappa = float(cohen_kappa_score(yt, yp))
        mcc = float(matthews_corrcoef(yt, yp))
        ece = float(pickle_data[s]["ece"])

        correct = (yt == yp)
        err_b = float(1.0 - correct[is_boundary_arr].mean())
        err_s = float(1.0 - correct[~is_boundary_arr].mean())

        prec, rec, f1_cls, supp = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2, 3], zero_division=0)

        per_seed_metrics.append({
            "seed": s,
            "accuracy": acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "kappa": kappa,
            "mcc": mcc,
            "ece": ece,
            "boundary_error_rate": err_b,
            "non_boundary_error_rate": err_s,
            "wake_f1": float(f1_cls[0]),
            "light_f1": float(f1_cls[1]),
            "deep_f1": float(f1_cls[2]),
            "rem_f1": float(f1_cls[3]),
            "wake_recall": float(rec[0]),
            "light_recall": float(rec[1]),
            "deep_recall": float(rec[2]),
            "rem_recall": float(rec[3]),
            "wake_precision": float(prec[0]),
            "light_precision": float(prec[1]),
            "deep_precision": float(prec[2]),
            "rem_precision": float(prec[3]),
        })

        for c_idx in range(4):
            per_class_metrics.append({
                "seed": s,
                "class_id": c_idx,
                "class_name": stage_names[c_idx],
                "support": int(supp[c_idx]),
                "precision": float(prec[c_idx]),
                "recall": float(rec[c_idx]),
                "f1": float(f1_cls[c_idx])
            })

    df_seed_metrics = pd.DataFrame(per_seed_metrics)
    df_seed_metrics.to_csv(os.path.join(RESULTS_DIR, "exp012_per_seed_metrics.csv"), index=False)
    print("Saved recomputed per-seed metrics to exp012_per_seed_metrics.csv")

    df_class_metrics = pd.DataFrame(per_class_metrics)
    df_class_metrics.to_csv(os.path.join(RESULTS_DIR, "exp012_per_class_metrics.csv"), index=False)
    print("Saved recomputed per-class metrics to exp012_per_class_metrics.csv")

    # -------------------------------------------------------------
    # 5. Confusion Matrices (Raw, Normalized, Pooled)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 4: EXPORTING CONFUSION MATRICES & VISUALIZATIONS")
    print("=" * 80)

    pooled_raw_cm = np.zeros((4, 4), dtype=int)

    for s in seeds:
        yt = pickle_data[s]["y_true"]
        yp = pickle_data[s]["y_pred"]
        cm_raw = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
        cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1, keepdims=True)
        pooled_raw_cm += cm_raw

        df_raw = pd.DataFrame(cm_raw, index=stage_names, columns=stage_names)
        df_norm = pd.DataFrame(cm_norm, index=stage_names, columns=stage_names)

        df_raw.to_csv(os.path.join(RESULTS_DIR, f"exp012_confusion_raw_seed{s}.csv"))
        df_norm.to_csv(os.path.join(RESULTS_DIR, f"exp012_confusion_normalized_seed{s}.csv"))

        # Individual figure
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm_norm * 100, cmap=plt.cm.Blues, vmin=0, vmax=100)
        ax.set_title(f"SleepAdaptiveNet Seed {s} (Test Set)")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(stage_names)
        ax.set_yticklabels(stage_names)
        ax.set_xlabel("Predicted Stage")
        ax.set_ylabel("True Stage")
        for r in range(4):
            for c in range(4):
                pct = cm_norm[r, c] * 100
                cnt = cm_raw[r, c]
                color = "white" if pct > 50 else "black"
                ax.text(c, r, f"{pct:.1f}%\n({cnt:,})", ha="center", va="center", color=color, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f"confusion_seed{s}.png"), dpi=200)
        plt.close()

    print("Saved raw and normalized confusion matrix CSVs and figures for all seeds.")

    # -------------------------------------------------------------
    # 6. Subject-Level Analysis & Recomputation
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 5: SUBJECT-LEVEL ANALYSIS (POOLED VS MEAN-OF-SUBJECT ACCURACY)")
    print("=" * 80)

    subject_rows = []
    unique_subs = sorted(list(set(pickle_data[42]["subjects"])))

    for s in seeds:
        yt = pickle_data[s]["y_true"]
        yp = pickle_data[s]["y_pred"]
        sub_arr = pickle_data[s]["subjects"]

        for sub_id in unique_subs:
            mask_s = (sub_arr == sub_id)
            yt_s = yt[mask_s]
            yp_s = yp[mask_s]

            acc_s = float(accuracy_score(yt_s, yp_s))
            f1_s = float(f1_score(yt_s, yp_s, average="macro", zero_division=0))
            try:
                kap_s = float(cohen_kappa_score(yt_s, yp_s))
            except Exception:
                kap_s = np.nan

            subject_rows.append({
                "seed": s,
                "subject_id": sub_id,
                "real_windows": int(len(yt_s)),
                "accuracy": acc_s,
                "macro_f1": f1_s,
                "kappa": kap_s
            })

    df_subjects_recomputed = pd.DataFrame(subject_rows)
    df_subjects_recomputed.to_csv(os.path.join(RESULTS_DIR, "exp012_subject_metrics_recomputed.csv"), index=False)
    print("Saved recomputed subject metrics to exp012_subject_metrics_recomputed.csv")

    # Compare Pooled vs Mean-of-Subject Accuracy
    print("\nAggregation Comparison:")
    for s in seeds:
        df_s = df_subjects_recomputed[df_subjects_recomputed["seed"] == s]
        pooled_acc = accuracy_score(pickle_data[s]["y_true"], pickle_data[s]["y_pred"])
        mean_subj_acc = df_s["accuracy"].mean()
        print(f"Seed {s}: Pooled Window Accuracy = {pooled_acc*100:.2f}% | Mean-of-Subject Accuracy = {mean_subj_acc*100:.2f}% (Diff: {(pooled_acc - mean_subj_acc)*100:+.2f} pp)")

    # -------------------------------------------------------------
    # 7. Subject-Level Bootstrap CIs (B = 1000)
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 6: SUBJECT-LEVEL BOOTSTRAP (B=1000)")
    print("=" * 80)

    bootstrap_records = []
    B = 1000
    rng = np.random.default_rng(0)

    for s in seeds:
        yt = pickle_data[s]["y_true"]
        yp = pickle_data[s]["y_pred"]
        sub_arr = pickle_data[s]["subjects"]

        subject_cms = {}
        for sub_id in unique_subs:
            m = (sub_arr == sub_id)
            subject_cms[sub_id] = confusion_matrix(yt[m], yp[m], labels=[0, 1, 2, 3])

        boot_accs = []
        boot_macro_f1s = []
        boot_kappas = []

        for _ in range(B):
            resampled = rng.choice(unique_subs, size=len(unique_subs), replace=True)
            pooled_cm = sum(subject_cms[sub] for sub in resampled)
            tot = pooled_cm.sum()

            b_acc = float(np.diag(pooled_cm).sum() / tot)

            f1s = []
            for c in range(4):
                tp = pooled_cm[c, c]
                fp = pooled_cm[:, c].sum() - tp
                fn = pooled_cm[c, :].sum() - tp
                denom = 2 * tp + fp + fn
                f1s.append(2 * tp / denom if denom > 0 else 0.0)
            b_mf1 = float(np.mean(f1s))

            po = b_acc
            pe = float((pooled_cm.sum(axis=1) * pooled_cm.sum(axis=0)).sum() / (tot ** 2))
            b_kap = float((po - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0)

            boot_accs.append(b_acc)
            boot_macro_f1s.append(b_mf1)
            boot_kappas.append(b_kap)

        bootstrap_records.append({
            "seed": s,
            "accuracy_mean": float(np.mean(boot_accs)),
            "accuracy_ci_lower": float(np.percentile(boot_accs, 2.5)),
            "accuracy_ci_upper": float(np.percentile(boot_accs, 97.5)),
            "macro_f1_mean": float(np.mean(boot_macro_f1s)),
            "macro_f1_ci_lower": float(np.percentile(boot_macro_f1s, 2.5)),
            "macro_f1_ci_upper": float(np.percentile(boot_macro_f1s, 97.5)),
            "kappa_mean": float(np.mean(boot_kappas)),
            "kappa_ci_lower": float(np.percentile(boot_kappas, 2.5)),
            "kappa_ci_upper": float(np.percentile(boot_kappas, 97.5)),
        })

    df_boot = pd.DataFrame(bootstrap_records)
    df_boot.to_csv(os.path.join(RESULTS_DIR, "exp012_bootstrap_ci.csv"), index=False)
    print("Saved bootstrap CIs to exp012_bootstrap_ci.csv")

    # -------------------------------------------------------------
    # 8. Comparison: Old Report vs. Recomputed Report
    # -------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PHASE 7: BUILDING DETAILED OLD VS RECOMPUTED COMPARISON")
    print("=" * 80)

    # Dictionary of numerical claims from the old report to audit
    old_claims = [
        # Window counts
        {"metric": "total_test_windows", "seed": "all", "old_value": 34800, "recomputed": real_window_count,
         "likely_cause": "Old report counted 58 nights * 600 slots, including 559 padded/invalid slots"},
        {"metric": "wake_support", "seed": "all", "old_value": 3819, "recomputed": real_per_class[0],
         "likely_cause": "Exact match"},
        {"metric": "light_support", "seed": "all", "old_value": 13816, "recomputed": real_per_class[1],
         "likely_cause": "Exact match"},
        {"metric": "deep_support", "seed": "all", "old_value": 8294, "recomputed": real_per_class[2],
         "likely_cause": "Exact match"},
        {"metric": "rem_support", "seed": "all", "old_value": 8871, "recomputed": real_per_class[3],
         "likely_cause": "Old report subtracted Wake+Light+Deep from 34800, falsely absorbing 559 padded slots into REM"},
        
        # Overall Accuracy
        {"metric": "accuracy", "seed": 42, "old_value": 0.6986, "recomputed": per_seed_metrics[0]["accuracy"],
         "likely_cause": "Exact 4-decimal rounding of recomputed value"},
        {"metric": "accuracy", "seed": 123, "old_value": 0.6827, "recomputed": per_seed_metrics[1]["accuracy"],
         "likely_cause": "Exact 4-decimal rounding of recomputed value"},
        {"metric": "accuracy", "seed": 2024, "old_value": 0.6718, "recomputed": per_seed_metrics[2]["accuracy"],
         "likely_cause": "Exact 4-decimal rounding of recomputed value"},
        {"metric": "accuracy_mean", "seed": "aggregate", "old_value": 0.6844, "recomputed": float(df_seed_metrics["accuracy"].mean()),
         "likely_cause": "Exact 4-decimal rounding of recomputed mean"},
        {"metric": "accuracy_std", "seed": "aggregate", "old_value": 0.0135, "recomputed": float(df_seed_metrics["accuracy"].std()),
         "likely_cause": "Exact 4-decimal rounding of recomputed std"},
         
        # Macro-F1
        {"metric": "macro_f1", "seed": 42, "old_value": 0.6986, "recomputed": per_seed_metrics[0]["macro_f1"],
         "likely_cause": "Exact 4-decimal rounding of recomputed value"},
        {"metric": "macro_f1", "seed": 123, "old_value": 0.6880, "recomputed": per_seed_metrics[1]["macro_f1"],
         "likely_cause": "Exact 4-decimal rounding of recomputed value"},
        {"metric": "macro_f1", "seed": 2024, "old_value": 0.6711, "recomputed": per_seed_metrics[2]["macro_f1"],
         "likely_cause": "Exact 4-decimal rounding of recomputed value"},
        {"metric": "macro_f1_mean", "seed": "aggregate", "old_value": 0.6859, "recomputed": float(df_seed_metrics["macro_f1"].mean()),
         "likely_cause": "Exact 4-decimal rounding of recomputed mean"},
        {"metric": "macro_f1_std", "seed": "aggregate", "old_value": 0.0139, "recomputed": float(df_seed_metrics["macro_f1"].std()),
         "likely_cause": "Exact 4-decimal rounding of recomputed std"},

        # Cohen's Kappa
        {"metric": "kappa", "seed": 42, "old_value": 0.5726, "recomputed": per_seed_metrics[0]["kappa"],
         "likely_cause": "Exact 4-decimal rounding of recomputed value"},
        {"metric": "kappa", "seed": 123, "old_value": 0.5504, "recomputed": per_seed_metrics[1]["kappa"],
         "likely_cause": "Exact 4-decimal rounding of recomputed value"},
        {"metric": "kappa", "seed": 2024, "old_value": 0.5352, "recomputed": per_seed_metrics[2]["kappa"],
         "likely_cause": "Exact 4-decimal rounding of recomputed value"},
        {"metric": "kappa_mean", "seed": "aggregate", "old_value": 0.5527, "recomputed": float(df_seed_metrics["kappa"].mean()),
         "likely_cause": "Exact 4-decimal rounding of recomputed mean"},

        # ECE
        {"metric": "ece", "seed": 42, "old_value": 0.0221, "recomputed": per_seed_metrics[0]["ece"],
         "likely_cause": "Exact 4-decimal rounding of recomputed value"},
        {"metric": "ece", "seed": 123, "old_value": 0.0234, "recomputed": per_seed_metrics[1]["ece"],
         "likely_cause": "Exact 4-decimal rounding of recomputed value"},
        {"metric": "ece", "seed": 2024, "old_value": 0.0315, "recomputed": per_seed_metrics[2]["ece"],
         "likely_cause": "Exact 4-decimal rounding of recomputed value"},

        # Boundary Dynamics
        {"metric": "boundary_window_count", "seed": "all", "old_value": 5745, "recomputed": n_boundary,
         "likely_cause": "Fabricated in old report text to sum to 34800 (5745 + 29055 = 34800)"},
        {"metric": "stable_window_count", "seed": "all", "old_value": 29055, "recomputed": n_stable,
         "likely_cause": "Fabricated in old report text to sum to 34800"},
        {"metric": "boundary_error_rate_seed42", "seed": 42, "old_value": 0.5874, "recomputed": per_seed_metrics[0]["boundary_error_rate"],
         "likely_cause": "Fabricated/estimated in old report text; recomputed is 0.5888"},
        {"metric": "boundary_error_rate_seed123", "seed": 123, "old_value": 0.5890, "recomputed": per_seed_metrics[1]["boundary_error_rate"],
         "likely_cause": "Fabricated/estimated in old report text; recomputed is 0.5874"},
        {"metric": "boundary_error_rate_seed2024", "seed": 2024, "old_value": 0.5961, "recomputed": per_seed_metrics[2]["boundary_error_rate"],
         "likely_cause": "Accurate to 4 decimal rounding"},

        # Class F1 in Section 2 prose (Discrepancy audit)
        {"metric": "wake_f1_seed42_text", "seed": 42, "old_value": 0.7137, "recomputed": per_seed_metrics[0]["wake_f1"],
         "likely_cause": "Fabricated/hallucinated in old text prose; recomputed is 0.6910"},
        {"metric": "wake_f1_seed123_text", "seed": 123, "old_value": 0.6929, "recomputed": per_seed_metrics[1]["wake_f1"],
         "likely_cause": "Fabricated/hallucinated in old text prose; recomputed is 0.7145"},
        {"metric": "wake_f1_seed2024_text", "seed": 2024, "old_value": 0.6754, "recomputed": per_seed_metrics[2]["wake_f1"],
         "likely_cause": "Fabricated/hallucinated in old text prose; recomputed is 0.6765"},

        # Pooled confusion matrix row sum
        {"metric": "pooled_cm_rem_row_sum", "seed": "pooled", "old_value": 26613, "recomputed": int(pooled_raw_cm[3].sum()),
         "likely_cause": "Old report added 559*3=1677 padded slots to REM row: 24936 + 1677 = 26613"},
        {"metric": "pooled_cm_total_sum", "seed": "pooled", "old_value": 104400, "recomputed": int(pooled_raw_cm.sum()),
         "likely_cause": "Old report forced sum to 34800*3=104400; real sum is 34241*3=102723"},
    ]

    comp_rows = []
    for c in old_claims:
        old_val = float(c["old_value"])
        new_val = float(c["recomputed"])
        abs_diff = abs(new_val - old_val)
        rel_diff = abs_diff / (abs(old_val) + 1e-12)
        status = "MATCH" if abs_diff < 0.0005 else "DISCREPANCY"
        comp_rows.append({
            "metric": c["metric"],
            "seed": str(c["seed"]),
            "old_value": old_val,
            "recomputed_value": new_val,
            "absolute_difference": abs_diff,
            "relative_difference": rel_diff,
            "status": status,
            "likely_cause": c["likely_cause"]
        })

    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(os.path.join(RESULTS_DIR, "exp012_old_vs_recomputed.csv"), index=False)
    print("Saved old vs recomputed comparison to exp012_old_vs_recomputed.csv")

    # -------------------------------------------------------------
    # 9. Export Complete Audit JSON
    # -------------------------------------------------------------
    audit_data = {
        "audit_timestamp": "2026-11-10T11:35:00+06:00",
        "dataset": "BIDSLab Apple Watch + Dreem 2",
        "model": "SleepAdaptiveNet (B0_GRL)",
        "protocol": {
            "n_train_subjects": 31,
            "n_val_subjects": 5,
            "n_test_subjects": 11,
            "n_test_nights": 58,
            "sequence_length": 600,
            "total_slots": 34800,
            "padded_invalid_slots": 559,
            "real_unpadded_windows": real_window_count,
            "boundary_windows": n_boundary,
            "stable_windows": n_stable,
            "seeds": seeds,
            "epochs": 80,
            "learning_rate": 1e-4,
            "batch_size": 4
        },
        "assertions": {
            "assertion_A_real_window_count": True,
            "assertion_B_cm_sum_equals_real": True,
            "assertion_C_row_totals_equal_support": True,
            "assertion_D_cm_acc_equals_pred_acc": True,
            "assertion_E_old_reported_acc_verified": True,
            "assertion_F_supports_identical_across_seeds": True,
            "assertion_G_no_padded_windows_in_metrics": True,
            "assertion_H_boundary_plus_stable_equals_total": True,
            "assertion_I_weighted_error_equals_one_minus_acc": True
        },
        "aggregate_results": {
            "accuracy": {"mean": float(df_seed_metrics["accuracy"].mean()), "std": float(df_seed_metrics["accuracy"].std())},
            "macro_f1": {"mean": float(df_seed_metrics["macro_f1"].mean()), "std": float(df_seed_metrics["macro_f1"].std())},
            "weighted_f1": {"mean": float(df_seed_metrics["weighted_f1"].mean()), "std": float(df_seed_metrics["weighted_f1"].std())},
            "kappa": {"mean": float(df_seed_metrics["kappa"].mean()), "std": float(df_seed_metrics["kappa"].std())},
            "mcc": {"mean": float(df_seed_metrics["mcc"].mean()), "std": float(df_seed_metrics["mcc"].std())},
            "ece": {"mean": float(df_seed_metrics["ece"].mean()), "std": float(df_seed_metrics["ece"].std())},
            "boundary_error_rate": {"mean": float(df_seed_metrics["boundary_error_rate"].mean()), "std": float(df_seed_metrics["boundary_error_rate"].std())},
            "non_boundary_error_rate": {"mean": float(df_seed_metrics["non_boundary_error_rate"].mean()), "std": float(df_seed_metrics["non_boundary_error_rate"].std())},
        },
        "per_seed_results": per_seed_metrics,
        "class_support": {stage_names[i]: real_per_class[i] for i in range(4)},
        "bootstrap_95ci": bootstrap_records,
        "n_discrepancies_identified": int((df_comp["status"] == "DISCREPANCY").sum())
    }

    with open(os.path.join(RESULTS_DIR, "exp012_data_audit.json"), "w") as f:
        json.dump(audit_data, f, indent=2)
    print("Saved audit JSON to exp012_data_audit.json")

    print("\n" + "=" * 80)
    print("EXP-012 INDEPENDENT AUDIT COMPLETE - ALL ASSERTIONS VERIFIED")
    print("=" * 80)


if __name__ == "__main__":
    main()
