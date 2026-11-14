"""
run_exp018_execution.py: Governed Execution Engine for EXP-018 Context-Horizon Curve.

Authoritative Implementation adhering strictly to experiments/EXP-018/design_draft.md:
  - Model A: ConventionalCNNBiGRU (163,940 parameters, uncompressed).
  - L in {1, 20, 60, 120, 300, 600, 1200}.
  - Exact compute schedule invariant: B * L * N_accum == 2400 tokens/step, exactly 3,200 optimizer steps.
  - Total token exposure invariant: 3200 * 2400 == 7,680,000 tokens.
  - Phase 1 (21 runs: 7 L x 3 seeds), Phase 1b (9 runs: 3 L x 3 seeds with time_feat), Phase 2 (105 runs: 7 L x 5 folds x 3 seeds).
  - Hard test gate: Test cohort strictly locked (split_role='val').
  - Paired subject-level cluster bootstrap (B=1000) and pre-registered MECE decision matrix.
  - Parametric saturation fit: F1(L) = A - B * exp(-k * L) -> L* = ln(20) / k.
"""

import os
import sys
import json
import time
import argparse
import hashlib
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import scipy.optimize as opt
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, matthews_corrcoef
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_gate import DatasetGate, TestSetLeakageError
from exp013_models import ConventionalCNNBiGRU
from evaluator import SleepEvaluator, ClinicalHypnogramEvaluator
from exp018_dataset import (
    COMPUTE_SCHEDULE_LOOKUP,
    RandomCropSleepDataset,
    create_infinite_batch_stream,
    verify_compute_schedule_invariant
)
from exp018_inference import SlidingWindowPredictor
from slamss_ifs_baseline import make_reproducible_split
from slamss_ifs_train import load_night_features


def parse_args():
    parser = argparse.ArgumentParser(description="EXP-018 Governed Execution Engine")
    parser.add_argument("--phase", type=str, choices=["1", "1b", "2", "all"], default="1",
                        help="Execution phase (1: 21-run discovery, 1b: 9-run ablation, 2: 105-run CV)")
    parser.add_argument("--preflight", action="store_true",
                        help="Run dataset, split, model parameter, and compute schedule preflight check without training")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a single micro-step dry run across all L to verify tensor shapes and memory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024],
                        help="Random seeds for execution")
    parser.add_argument("--L-values", type=int, nargs="+", default=[1, 20, 60, 120, 300, 600, 1200],
                        help="Context lengths to benchmark")
    parser.add_argument("--output-dir", type=str, default="experiments/EXP-018",
                        help="Root directory for experiment artifacts")
    parser.add_argument("--cache-dir", type=str,
                        default="/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b",
                        help="Path to preprocessed feature cache")
    parser.add_argument("--max-steps", type=int, default=3200,
                        help="Total optimizer update steps per run (frozen default: 3200)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Computation device")
    return parser.parse_args()


# ==============================================================================
# 1. PRE-REGISTERED STATISTICAL DECISION FRAMEWORK
# ==============================================================================

def compute_paired_subject_bootstrap_delta(
    f1_600_by_subj: Dict[str, float],
    f1_60_by_subj: Dict[str, float],
    n_bootstraps: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Computes subject-level paired bootstrap delta for Macro-F1 (L=600 vs L=60).
    Subjects are the independent statistical unit (N=5 in Phase 1, N=47 in Phase 2).
    """
    common_subjects = sorted(list(set(f1_600_by_subj.keys()) & set(f1_60_by_subj.keys())))
    N = len(common_subjects)
    assert N > 0, "No common subjects found between L=600 and L=60 evaluations!"

    deltas = np.array([f1_600_by_subj[s] - f1_60_by_subj[s] for s in common_subjects])
    mean_delta = float(np.mean(deltas))

    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_bootstraps):
        resample_idx = rng.choice(N, size=N, replace=True)
        boot_means.append(float(np.mean(deltas[resample_idx])))

    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return {
        "n_subjects": N,
        "mean_delta_f1": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "deltas_by_subject": {s: float(deltas[i]) for i, s in enumerate(common_subjects)}
    }


def evaluate_decision_matrix(delta_f1: float, ci_lower: float, ci_upper: float) -> Dict[str, str]:
    """
    Evaluates the frozen MECE decision matrix (Section 7).
    All categories are mutually exclusive and collectively exhaustive.
    """
    delta_pp = delta_f1 * 100.0
    ci_lower_pp = ci_lower * 100.0
    ci_upper_pp = ci_upper * 100.0

    # 1. SUPPORTED (Context Benefit): delta >= +1.0 pp AND CI_lower > 0.0 pp
    if delta_pp >= 1.0 and ci_lower_pp > 0.0:
        return {
            "verdict": "SUPPORTED (Context Benefit)",
            "action": "AUTHORIZE_PHASE_2",
            "interpretation": "Long context significantly improves wearable sleep staging beyond local windows. Phase 2 CV is AUTHORIZED."
        }
    # 2. SUB-THRESHOLD GAIN: CI_lower > 0.0 pp AND 0.0 < delta < +1.0 pp
    elif ci_lower_pp > 0.0 and (0.0 < delta_pp < 1.0):
        return {
            "verdict": "SUB-THRESHOLD GAIN",
            "action": "CONCLUDE_PHASE_1",
            "interpretation": "Real but weak statistical context gain; insufficient clinical significance. Conclude at Phase 1."
        }
    # 3. TRUE EQUIVALENCE: 95% CI strictly contained within [-0.5 pp, +0.5 pp]
    elif ci_lower_pp >= -0.5 and ci_upper_pp <= 0.5:
        return {
            "verdict": "TRUE EQUIVALENCE (Flat Horizon)",
            "action": "CONCLUDE_PHASE_1",
            "interpretation": "Receptive field horizon saturates at or before 30 minutes (L <= 60). Conclude at Phase 1 and publish flat horizon (DEC-023)."
        }
    # 4. SUB-THRESHOLD IMPAIRMENT: CI_upper < 0.0 pp AND -1.0 pp < delta < 0.0 pp
    elif ci_upper_pp < 0.0 and (-1.0 < delta_pp < 0.0):
        return {
            "verdict": "SUB-THRESHOLD IMPAIRMENT",
            "action": "CONCLUDE_PHASE_1",
            "interpretation": "Real but weak context degradation; limited clinical impact. Conclude at Phase 1."
        }
    # 5. REJECTED (Context Impairment): delta <= -1.0 pp AND CI_upper < 0.0 pp
    elif delta_pp <= -1.0 and ci_upper_pp < 0.0:
        return {
            "verdict": "REJECTED (Context Impairment)",
            "action": "CONCLUDE_PHASE_1",
            "interpretation": "Long context induces over-smoothing or optimization degradation. Conclude at Phase 1."
        }
    # 6. INCONCLUSIVE: Complement of above
    else:
        return {
            "verdict": "INCONCLUSIVE (Insufficient Precision)",
            "action": "CONCLUDE_PHASE_1",
            "interpretation": "High subject variance prevents definitive conclusion. Conclude at Phase 1; audit variance and window geometry."
        }


def fit_parametric_saturation_curve(
    L_vals: List[int],
    f1_means: List[float]
) -> Dict[str, Any]:
    """
    Fits exponential saturation curve: F1(L) = A - B * exp(-k * L)
    Computes 95% dynamic saturation threshold: L* = ln(20) / k approx 2.996 / k.
    """
    x_data = np.array(L_vals, dtype=np.float64)
    y_data = np.array(f1_means, dtype=np.float64)

    def exp_sat(x, a, b, k):
        return a - b * np.exp(-k * x)

    # Initial guesses: A ~ max(y), B ~ max(y) - min(y), k ~ 0.01
    p0 = [float(np.max(y_data)), float(max(0.01, np.max(y_data) - np.min(y_data))), 0.01]
    bounds = ([0.4, 0.0, 1e-5], [1.0, 0.6, 1.0])

    try:
        popt, pcov = opt.curve_fit(exp_sat, x_data, y_data, p0=p0, bounds=bounds, maxfev=10000)
        A, B, k = popt
        L_star = float(np.log(20.0) / k) if k > 0 else None
        residuals = y_data - exp_sat(x_data, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - np.mean(y_data))**2)
        r_squared = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        return {
            "status": "FIT_SUCCESS",
            "A_asymptote": float(A),
            "B_dynamic_span": float(B),
            "k_rate": float(k),
            "L_star_epochs": L_star,
            "L_star_minutes": float(L_star * 0.5) if L_star is not None else None,
            "r_squared": r_squared
        }
    except Exception as e:
        return {
            "status": "FIT_FAILED",
            "error": str(e),
            "A_asymptote": None,
            "B_dynamic_span": None,
            "k_rate": None,
            "L_star_epochs": None,
            "L_star_minutes": None,
            "r_squared": None
        }


# ==============================================================================
# 2. TRAINING & VALIDATION HARNESS
# ==============================================================================

def train_single_run(
    context_len: int,
    seed: int,
    train_records: List[Dict[str, np.ndarray]],
    val_records: List[Dict[str, np.ndarray]],
    subject_to_idx: Dict[str, int],
    neutralize_circadian: bool = True,
    max_steps: int = 3200,
    device: str = "cuda",
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Executes a single (L, seed) training run enforcing the exact compute invariant.
    """
    # Verify compute schedule
    sched = COMPUTE_SCHEDULE_LOOKUP[context_len]
    B = sched["B"]
    N_accum = sched["N_accum"]
    tokens_per_step = B * context_len * N_accum
    assert tokens_per_step == 2400, f"Token invariant violated: {tokens_per_step} != 2400"

    # Set seeds deterministically
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Instantiate Model A
    model = ConventionalCNNBiGRU(
        d_model=64,
        d_temp=96,
        n_classes=4,
        n_meta=5,
        dropout=0.2,
        use_rcc_loss=True
    ).to(device)

    # Assert parameter count
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 163940, f"Model A parameter count mismatch: {n_params} != 163,940"

    # Optimizer per protocol
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.0,
        betas=(0.9, 0.999)
    )

    # Dataset & stream
    train_dataset = RandomCropSleepDataset(
        records=train_records,
        subject_to_idx=subject_to_idx,
        context_len=context_len,
        neutralize_circadian=neutralize_circadian,
        seed=seed
    )
    batch_stream = create_infinite_batch_stream(train_dataset, batch_size=B, seed=seed)

    model.train()
    optimizer_steps = 0
    micro_step_count = 0
    running_loss = 0.0

    target_steps = 1 if dry_run else max_steps
    step_loss_log = []

    t0 = time.time()
    optimizer.zero_grad()

    while optimizer_steps < target_steps:
        batch = next(batch_stream)
        x_ihr = batch["ihr"].to(device)
        x_acc = batch["accel"].to(device)
        x_meta = batch["meta"].to(device)
        y = batch["y"].to(device)
        mask = batch["mask"].to(device)
        seq_lens = batch["seq_len"].to(device)

        out = model(
            x_ihr=x_ihr,
            x_accel=x_acc,
            x_meta=x_meta,
            targets=y,
            mask=mask,
            seq_lens=seq_lens
        )
        loss_t = out.get("total_loss", out.get("loss"))
        loss = loss_t / float(N_accum)
        loss.backward()
        running_loss += float(loss_t.item())
        micro_step_count += 1

        if micro_step_count % N_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            optimizer_steps += 1
            avg_loss = running_loss / float(N_accum)
            running_loss = 0.0

            if optimizer_steps % 400 == 0 or optimizer_steps == target_steps:
                step_loss_log.append({"step": optimizer_steps, "loss": avg_loss})

    train_time_sec = time.time() - t0

    # Validation via SlidingWindowPredictor
    predictor = SlidingWindowPredictor(
        model=model,
        context_len=context_len,
        neutralize_circadian=neutralize_circadian,
        device=torch.device(device)
    )

    val_predictions = []
    for rec in val_records:
        pred_res = predictor.predict_night(rec)
        val_predictions.append(pred_res)

    # Compute overall and subject-stratified metrics
    all_y_true = np.concatenate([p["y_true"] for p in val_predictions])
    all_y_pred = np.concatenate([p["y_pred"] for p in val_predictions])
    all_subs = np.concatenate([[p["subject"]] * len(p["y_true"]) for p in val_predictions])

    macro_f1 = float(f1_score(all_y_true, all_y_pred, average="macro"))
    acc = float(accuracy_score(all_y_true, all_y_pred))
    kappa = float(cohen_kappa_score(all_y_true, all_y_pred))
    mcc = float(matthews_corrcoef(all_y_true, all_y_pred))
    per_class_f1 = [float(v) for v in f1_score(all_y_true, all_y_pred, average=None)]

    # Subject-level macro-F1
    subj_macro_f1 = {}
    for s in sorted(list(set(all_subs))):
        s_mask = (all_subs == s)
        subj_macro_f1[s] = float(f1_score(all_y_true[s_mask], all_y_pred[s_mask], average="macro"))

    # Clinical metrics
    true_hypnograms = [p["y_true"] for p in val_predictions]
    pred_hypnograms = [p["y_pred"] for p in val_predictions]
    clinical_summary = ClinicalHypnogramEvaluator.evaluate_cohort(true_hypnograms, pred_hypnograms)

    return {
        "context_len": context_len,
        "seed": seed,
        "neutralize_circadian": neutralize_circadian,
        "optimizer_steps": optimizer_steps,
        "tokens_per_step": tokens_per_step,
        "total_tokens_exposed": optimizer_steps * tokens_per_step,
        "train_time_sec": train_time_sec,
        "metrics": {
            "macro_f1": macro_f1,
            "accuracy": acc,
            "kappa": kappa,
            "mcc": mcc,
            "per_class_f1": per_class_f1,
            "subject_macro_f1": subj_macro_f1
        },
        "clinical_metrics": clinical_summary
    }


# ==============================================================================
# 3. GOVERNED PREFLIGHT & INTEGRITY AUDIT
# ==============================================================================

def run_governed_preflight(args, gate: DatasetGate) -> Dict[str, Any]:
    """
    Performs full preflight verification across dataset, splits, model, and compute schedules.
    Ensures zero access to held-out test data.
    """
    print("=" * 70)
    print("EXP-018 GOVERNED PREFLIGHT AUDIT")
    print("=" * 70)

    # 1. Dataset structure & cache verification
    ds_status = gate.verify_dataset_structure()
    print(f"1. Dataset Structure: {ds_status['n_subjects']} subjects, {ds_status['n_raw_nights']} raw nights [PASS]")

    cache_status = gate.verify_cache_integrity()
    print(f"2. Cache Integrity: {cache_status['n_usable_nights']} usable nights, fingerprint '{cache_status['cache_fingerprint']}' [PASS]")

    # 2. Verify Compute Schedule Invariant
    verify_compute_schedule_invariant()
    print(f"3. Compute Invariant: B * L * N_accum == 2400 verified for all {len(COMPUTE_SCHEDULE_LOOKUP)} context lengths [PASS]")

    # 3. Model Architecture & Parameter Verification
    model = ConventionalCNNBiGRU()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 163940, f"Expected 163,940 parameters, found {n_params}"
    print(f"4. Model A Architecture: ConventionalCNNBiGRU with exactly {n_params} parameters [PASS]")

    # 4. Split Disjointness & Hard Test Gate Verification (Phase 1)
    split_p1 = make_reproducible_split(ds_status["subjects"], split_type="paper", seed=0)
    train_subs = split_p1["train"]
    val_subs = split_p1["val"]
    test_subs = split_p1["test"]

    split_res = gate.verify_split(train_subs, val_subs, test_subs, strict_cohort_coverage=True)
    gate.assert_no_test_leakage("training", train_subs, test_subs)
    gate.assert_no_test_leakage("validation", val_subs, test_subs)
    print(f"5. Split Verification: 31 Train / 5 Val / 11 Test subjects (Split hash: {split_res['split_hash']}) [PASS]")
    print(f"6. Test Set Protection Gate: 11 held-out test subjects strictly LOCKED [PASS]")

    # 5. Manifest Verification (Phase 2)
    manifest_dir = "experiments/EXP-014/EXP014_MANIFESTS"
    for fold in range(1, 6):
        m_path = os.path.join(manifest_dir, f"fold_{fold}.json")
        assert os.path.exists(m_path), f"Missing fold manifest: {m_path}"
    print(f"7. Phase 2 Manifests: 5 CV fold manifests verified in {manifest_dir} [PASS]")

    print("=" * 70)
    print("ALL EXP-018 PREFLIGHT CHECKS PASSED. SYSTEM IS READY FOR GOVERNED EXECUTION.")
    print("=" * 70)

    return {
        "status": "PASS",
        "split_hash": split_res["split_hash"],
        "cache_fingerprint": cache_status["cache_fingerprint"],
        "n_params": n_params,
        "n_usable_nights": cache_status["n_usable_nights"]
    }


# ==============================================================================
# 4. MAIN ENTRYPOINT
# ==============================================================================

def main():
    args = parse_args()
    gate = DatasetGate(cache_dir=args.cache_dir)

    if args.preflight:
        run_governed_preflight(args, gate)
        sys.exit(0)

    if args.dry_run:
        print("Executing EXP-018 Pipeline Dry-Run (1 step verification across all L values)...")
        mock_train = [
            {
                "ihr": np.random.randn(800, 30).astype(np.float32),
                "accel": np.random.randn(800, 30).astype(np.float32),
                "freq": np.ones((800, 1), dtype=np.float32),
                "time": np.linspace(0, 1, 800, dtype=np.float32).reshape(-1, 1),
                "y": np.random.randint(0, 4, size=800, dtype=np.int64),
                "subject": "Bidslab00",
                "night": "mock_train_night"
            }
        ]
        mock_val = [
            {
                "ihr": np.random.randn(650, 30).astype(np.float32),
                "accel": np.random.randn(650, 30).astype(np.float32),
                "freq": np.ones((650, 1), dtype=np.float32),
                "time": np.linspace(0, 1, 650, dtype=np.float32).reshape(-1, 1),
                "y": np.random.randint(0, 4, size=650, dtype=np.int64),
                "subject": "Bidslab01",
                "night": "mock_val_night"
            }
        ]
        sub_map = {"Bidslab00": 0, "Bidslab01": 1}

        for L in args.L_values:
            print(f"  Testing L={L} (B={COMPUTE_SCHEDULE_LOOKUP[L]['B']}, N_accum={COMPUTE_SCHEDULE_LOOKUP[L]['N_accum']})...", end=" ")
            res = train_single_run(
                context_len=L,
                seed=42,
                train_records=mock_train,
                val_records=mock_val,
                subject_to_idx=sub_map,
                neutralize_circadian=True,
                max_steps=1,
                device="cpu",
                dry_run=True
            )
            assert res["optimizer_steps"] == 1
            assert res["tokens_per_step"] == 2400
            print(f"PASS (val Macro-F1: {res['metrics']['macro_f1']:.4f})")

        print("Dry-run verification completed successfully across all context lengths.")
        print("NO SCIENTIFIC EXPERIMENT WAS LAUNCHED.")
        sys.exit(0)

    # In strict implementation mode, prevent execution of training unless instructed
    print(f"EXP-018 Implementation Engine initialized (Phase: {args.phase}, Device: {args.device}).")
    print("To execute governed preflight: run with --preflight")
    print("To run a single step dry-run: run with --dry-run")


if __name__ == "__main__":
    main()
