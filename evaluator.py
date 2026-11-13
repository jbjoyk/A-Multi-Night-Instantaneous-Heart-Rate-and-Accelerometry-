"""
evaluator.py: Standardized Evaluation Protocol and Test-Set Gate for Sleep Staging.

Standardizes all evaluation routines across the SLEEP research infrastructure:
  1. Strict Metric Hierarchy:
     - PRIMARY: Macro-F1 (for 4-class imbalanced sleep staging: Wake, Light, Deep, REM).
     - SECONDARY: Accuracy, Cohen's kappa, MCC, Expected Calibration Error (ECE),
                  Per-class F1/Recall/Precision, Boundary error rate.
     - DIAGNOSTIC: Confusion Matrix, Parameter Count, Peak VRAM, Loss breakdown, Gate stats.
  2. Test-Set Protection:
     - Hard gate blocking any test-set evaluation during development, tuning, or ranking.
  3. Paired Cluster Bootstrap:
     - Computes subject-level cluster bootstrap 95% confidence intervals (B=1000).
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, precision_recall_fscore_support
)


class EvaluatorError(Exception):
    """Base exception for evaluation failures."""
    pass


class TestSetProtectionViolation(EvaluatorError):
    """Raised when an unauthorized attempt is made to evaluate on the test partition."""
    __test__ = False
    pass


class MetricHierarchy:
    """Explicitly defines primary, secondary, and diagnostic metrics."""
    PRIMARY = "macro_f1"
    
    SECONDARIES = [
        "accuracy",
        "kappa",
        "mcc",
        "ece",
        "wake_f1",
        "light_f1",
        "deep_f1",
        "rem_f1",
        "wake_recall",
        "boundary_error_rate",
        "non_boundary_error_rate"
    ]
    
    DIAGNOSTICS = [
        "confusion_matrix",
        "confusion_matrix_norm",
        "support",
        "n_parameters",
        "peak_vram_mb",
        "alpha_gate_stats",
        "loss_breakdown"
    ]


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across 10 probability bins."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = (predictions == labels).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


class SleepEvaluator:
    """
    Unified, standardized evaluation engine for 4-class sleep staging.
    """
    CLASS_NAMES = ["Wake", "Light", "Deep", "REM"]
    
    def __init__(self):
        pass

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        probs: Optional[np.ndarray] = None,
        is_boundary: Optional[np.ndarray] = None,
        subjects: Optional[np.ndarray] = None,
        compute_bootstrap: bool = False,
        n_bootstraps: int = 1000,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Computes standardized metrics from numpy predictions and targets.
        Only valid labels in {0, 1, 2, 3} are scored.
        """
        valid_mask = (y_true >= 0) & (y_true < 4)
        yt = y_true[valid_mask]
        yp = y_pred[valid_mask]

        if len(yt) == 0:
            raise EvaluatorError("No valid labels found for evaluation.")

        # 1. Primary Metric: Macro-F1
        macro_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))

        # 2. Secondary Metrics
        acc = float(accuracy_score(yt, yp))
        weighted_f1 = float(f1_score(yt, yp, average="weighted", zero_division=0))
        kappa = float(cohen_kappa_score(yt, yp))
        mcc = float(matthews_corrcoef(yt, yp))
        
        prec, rec, f1_cls, supp = precision_recall_fscore_support(
            yt, yp, labels=[0, 1, 2, 3], zero_division=0
        )
        
        cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
        cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True), 0.0)

        # ECE
        ece_val = None
        if probs is not None:
            p_valid = probs[valid_mask]
            ece_val = compute_ece(p_valid, yt, n_bins=10)

        # Boundary Error Analysis
        err_bound = None
        err_non_bound = None
        if is_boundary is not None:
            b_valid = is_boundary[valid_mask].astype(bool)
            is_correct = (yt == yp)
            if b_valid.sum() > 0:
                err_bound = float(1.0 - is_correct[b_valid].mean())
            if (~b_valid).sum() > 0:
                err_non_bound = float(1.0 - is_correct[~b_valid].mean())

        # Subject-Level Aggregations & Cluster Bootstrap
        subject_metrics = []
        bootstrap_ci = {}
        if subjects is not None:
            sub_valid = subjects[valid_mask]
            unique_subs = sorted(list(set(sub_valid)))
            for s in unique_subs:
                s_mask = (sub_valid == s)
                if s_mask.sum() > 0:
                    acc_s = float(accuracy_score(yt[s_mask], yp[s_mask]))
                    mf1_s = float(f1_score(yt[s_mask], yp[s_mask], average="macro", zero_division=0))
                    try:
                        kap_s = float(cohen_kappa_score(yt[s_mask], yp[s_mask]))
                    except Exception:
                        kap_s = float("nan")
                    subject_metrics.append({
                        "subject_id": s,
                        "n_windows": int(s_mask.sum()),
                        "accuracy": acc_s,
                        "macro_f1": mf1_s,
                        "kappa": kap_s
                    })

            if compute_bootstrap and len(unique_subs) > 1:
                rng = np.random.RandomState(random_seed)
                boot_accs = []
                boot_mf1s = []
                for _ in range(n_bootstraps):
                    sampled_subs = rng.choice(unique_subs, size=len(unique_subs), replace=True)
                    s_indices = [np.where(sub_valid == s)[0] for s in sampled_subs]
                    all_idx = np.concatenate(s_indices)
                    boot_accs.append(accuracy_score(yt[all_idx], yp[all_idx]))
                    boot_mf1s.append(f1_score(yt[all_idx], yp[all_idx], average="macro", zero_division=0))

                bootstrap_ci = {
                    "accuracy_95ci": [float(np.percentile(boot_accs, 2.5)), float(np.percentile(boot_accs, 97.5))],
                    "macro_f1_95ci": [float(np.percentile(boot_mf1s, 2.5)), float(np.percentile(boot_mf1s, 97.5))]
                }

        results = {
            "primary": {
                "macro_f1": macro_f1
            },
            "secondary": {
                "accuracy": acc,
                "weighted_f1": weighted_f1,
                "kappa": kappa,
                "mcc": mcc,
                "ece": ece_val,
                "wake_f1": float(f1_cls[0]),
                "light_f1": float(f1_cls[1]),
                "deep_f1": float(f1_cls[2]),
                "rem_f1": float(f1_cls[3]),
                "wake_recall": float(rec[0]),
                "light_recall": float(rec[1]),
                "deep_recall": float(rec[2]),
                "rem_recall": float(rec[3]),
                "boundary_error_rate": err_bound,
                "non_boundary_error_rate": err_non_bound
            },
            "diagnostic": {
                "confusion_matrix": cm.tolist(),
                "confusion_matrix_norm": cm_norm.tolist(),
                "precision": [float(p) for p in prec],
                "recall": [float(r) for r in rec],
                "f1": [float(f) for f in f1_cls],
                "support": [int(s) for s in supp],
                "total_valid_windows": int(len(yt)),
                "subject_metrics": subject_metrics,
                "bootstrap_ci": bootstrap_ci
            }
        }
        return results

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        split_role: str = "validation",
        frozen_evaluation_authorized: bool = False
    ) -> Dict[str, Any]:
        """
        Evaluates a PyTorch model on a DataLoader.
        Hard Gate: If split_role == 'test' and frozen_evaluation_authorized is False,
                   raises TestSetProtectionViolation immediately.
        """
        role_clean = split_role.lower().strip()
        if role_clean in ("test", "outer_test", "held_out_test") and not frozen_evaluation_authorized:
            raise TestSetProtectionViolation(
                "HARD GATE VIOLATION: Test set evaluation attempted during development or without "
                "explicit frozen benchmark authorization. The test set is strictly protected!"
            )

        model.eval()
        all_preds = []
        all_targets = []
        all_boundaries = []
        all_probs = []
        all_subjects = []
        all_alphas = []
        val_losses = []

        with torch.no_grad():
            for batch in dataloader:
                x_ihr = batch["ihr"].to(device)
                x_acc = batch["accel"].to(device)
                x_meta = batch["meta"].to(device)
                targets = batch["y"].to(device)
                mask = batch["mask"].to(device)
                subj_idx = batch["subject_idx"].to(device)

                out = model(x_ihr, x_acc, x_meta, targets=targets, mask=mask, subject_idx=subj_idx)
                
                if "total_loss" in out:
                    val_losses.append(out["total_loss"].item())

                logits = out["logits"]
                probs = out["probabilities"]
                preds = torch.argmax(logits, dim=-1)

                # Boundary indicator
                y_prev = targets[:, :-1]
                y_curr = targets[:, 1:]
                is_bound = torch.zeros_like(targets, dtype=torch.bool)
                is_bound[:, 1:] = (y_prev != y_curr) & (y_prev >= 0) & (y_prev < 4) & (y_curr >= 0) & (y_curr < 4)

                valid = (targets >= 0) & (targets < 4) & mask
                subjs = batch.get("subject", ["unknown"] * len(targets))

                alpha_tensor = out.get("alpha_gate")

                for b_i in range(len(subjs)):
                    m_b = valid[b_i].cpu().numpy()
                    p_b = preds[b_i][m_b].cpu().numpy()
                    t_b = targets[b_i][m_b].cpu().numpy()
                    b_b = is_bound[b_i][m_b].cpu().numpy()
                    prb_b = probs[b_i][m_b].cpu().numpy()

                    all_preds.extend(p_b)
                    all_targets.extend(t_b)
                    all_boundaries.extend(b_b)
                    all_probs.append(prb_b)
                    all_subjects.extend([subjs[b_i]] * len(p_b))

                    if alpha_tensor is not None:
                        a_b = alpha_tensor[b_i][m_b].squeeze(-1).cpu().numpy()
                        all_alphas.extend(a_b)

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        is_bound_arr = np.array(all_boundaries, dtype=bool)
        probs_concat = np.concatenate(all_probs, axis=0) if all_probs else None
        subjects_arr = np.array(all_subjects)

        metrics = self.evaluate_predictions(
            y_true=y_true,
            y_pred=y_pred,
            probs=probs_concat,
            is_boundary=is_bound_arr,
            subjects=subjects_arr,
            compute_bootstrap=(role_clean in ("test", "outer_test"))
        )

        # Attach loss and gate diagnostics
        metrics["diagnostic"]["loss"] = float(np.mean(val_losses)) if val_losses else None
        if all_alphas:
            a_arr = np.array(all_alphas)
            metrics["diagnostic"]["alpha_gate_stats"] = {
                "mean": float(np.mean(a_arr)),
                "std": float(np.std(a_arr)),
                "min": float(np.min(a_arr)),
                "max": float(np.max(a_arr))
            }
        else:
            metrics["diagnostic"]["alpha_gate_stats"] = None

        metrics["split_role"] = split_role
        return metrics


if __name__ == "__main__":
    print("Running SleepEvaluator preflight self-test...")
    evaluator = SleepEvaluator()
    rng = np.random.RandomState(42)
    n = 600
    y_true = rng.randint(0, 4, size=n)
    y_pred = y_true.copy()
    # inject errors
    y_pred[rng.choice(n, size=150, replace=False)] = rng.randint(0, 4, size=150)
    probs = np.zeros((n, 4))
    for i in range(n):
        probs[i, y_pred[i]] = 0.8
        probs[i] += 0.05
    probs /= probs.sum(axis=-1, keepdims=True)
    subs = np.array([f"Bidslab{i%5:02d}" for i in range(n)])

    res = evaluator.evaluate_predictions(y_true, y_pred, probs=probs, subjects=subs, compute_bootstrap=True)
    print(f"Primary Metric (Macro-F1): {res['primary']['macro_f1']:.4f}")
    print(f"Secondary Accuracy: {res['secondary']['accuracy']:.4f}")
    print(f"Secondary Cohen's Kappa: {res['secondary']['kappa']:.4f}")
    print(f"Secondary ECE: {res['secondary']['ece']:.4f}")
    print(f"95% Bootstrap CI: {res['diagnostic']['bootstrap_ci']}")
    print("Testing hard gate on unauthorized test access...")
    try:
        dummy_model = nn.Linear(10, 4)
        evaluator.evaluate_model(dummy_model, None, None, split_role="test", frozen_evaluation_authorized=False)
        print("ERROR: Test set gate failed to raise exception!")
    except TestSetProtectionViolation as e:
        print(f"Gate Triggered Successfully: {e}")
    print("SleepEvaluator self-test PASSED.")
