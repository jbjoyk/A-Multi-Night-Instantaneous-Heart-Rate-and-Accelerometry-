"""
run_tsk08_slamss_matched.py: Stage 3 Matched 5-Fold Cross-Validation Driver for SLAMSS-IFS (EXP-021 / TSK-08).

Governing Preregistration: PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md (DEC-011, DEC-033, DEC-034).

Protocol Invariants:
  1. Model: Exact official BIDSLabUMass/SLAMSS-IFS repository architecture (5,465,768 parameters).
  2. Cohort & Splits: Identical 5-fold nested CV across all 47 subjects (246 usable nights) from
     experiments/EXP-014/EXP014_MANIFESTS/fold_{1..5}.json, cryptographically verified against
     review_package/fold_manifests.csv (SHA-256 prefix 36d84020) with exact role cross-checks.
  3. Zero Leakage: Training uses strictly inner_train_subjects; checkpoint selection and early
     stopping use strictly inner_val_subjects; outer_test_subjects are evaluated exactly ONCE
     using the best validation checkpoint.
  4. Context & Inference: Full-night L=1200 with batched SLAMSSSlidingWindowPredictor probability
     stitching (identical to sequential inference, ~12-15x faster on RTX 3060).
  5. Loss & Optimization: Composite CE + 0.01 * RCCLoss (DEC-011) computed strictly in FP32
     outside AMP autocast; forward/backward under PyTorch AMP (float16 + GradScaler);
     micro-batch B=4, effective batch B=4 (accum_steps=1, 42-44 optimizer steps/epoch ->
     up to 4,200-4,400 steps over 100 epochs, matching/exceeding Model A's 3,200 steps),
     Adam(lr=1e-4), grad clip 1.0, min_epochs=40 floor before patience=25 early stopping.
  6. Concurrency & Git Safety: Each (fold, seed) writes an isolated atomic JSON to
     experiments/EXP-021/run_results/slamss_ifs_fold{fold}_seed{seed}.json (tracked in Git;
     not hidden by .gitignore's runs/ rule) and atomic checkpoint to
     experiments/EXP-021/checkpoints/slamss_ifs_official_L1200_fold{fold}_seed{seed}.pt.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import torch
from torch.utils.data import DataLoader

EXP021_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP021_DIR, "..", ".."))
for _p in (REPO_ROOT, EXP021_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from official_slamss_ifs_adapter import (
    EXPECTED_OFFICIAL_SLAMSS_PARAMS,
    OfficialSLAMSSCompositeLoss,
    OfficialSLAMSSIFSWrapper,
    SLAMSSMatchedDataset,
    SLAMSSSlidingWindowPredictor,
)
from slamss_ifs_train import prepare_dataset_records

CACHE_DIR_DEFAULT = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
MANIFESTS_DIR_DEFAULT = os.path.join(REPO_ROOT, "experiments", "EXP-014", "EXP014_MANIFESTS")
AUTHORITATIVE_MANIFEST_CSV_DEFAULT = os.path.join(REPO_ROOT, "review_package", "fold_manifests.csv")
EXPECTED_MANIFEST_SHA256_PREFIX = "36d84020"
OUTPUT_DIR_DEFAULT = EXP021_DIR
RUN_RESULTS_SUBDIR = "run_results"

SEEDS_DEFAULT = [42, 123, 2024]
FOLDS_DEFAULT = [1, 2, 3, 4, 5]


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def verify_manifests_and_splits(
    manifests_dir: str,
    authoritative_csv: str,
) -> Dict[int, Dict[str, Any]]:
    """
    Pre-flight governance verification:
      1. Requires authoritative review_package/fold_manifests.csv to exist and asserts
         SHA-256 prefix 36d84020 (never silently skipped).
      2. Loads folds 1..5 JSON manifests and asserts exact 47-subject disjoint partition
         with zero inner_train/inner_val/outer_test overlap.
      3. Cross-verifies every fold's inner_train, inner_val, and outer_test subject sets
         against review_package/fold_manifests.csv.
    """
    if not os.path.exists(authoritative_csv):
        raise FileNotFoundError(
            f"Authoritative manifest CSV missing: {authoritative_csv}. "
            "Cannot bypass SHA-256 integrity gate."
        )

    digest = sha256_file(authoritative_csv)
    assert digest.startswith(EXPECTED_MANIFEST_SHA256_PREFIX), (
        f"Authoritative manifest hash mismatch: got {digest}, "
        f"expected prefix {EXPECTED_MANIFEST_SHA256_PREFIX}"
    )

    df_auth = pd.read_csv(authoritative_csv)
    required_cols = {"fold", "subject_id", "role"}
    assert required_cols.issubset(set(df_auth.columns)), (
        f"Authoritative CSV missing columns: {required_cols - set(df_auth.columns)}"
    )

    manifests: Dict[int, Dict[str, Any]] = {}
    all_outer_test: List[str] = []

    for fold in range(1, 6):
        mpath = os.path.join(manifests_dir, f"fold_{fold}.json")
        if not os.path.exists(mpath):
            raise FileNotFoundError(f"Fold manifest missing: {mpath}")
        with open(mpath, "r", encoding="utf-8") as f:
            m = json.load(f)

        tr = set(m["inner_train_subjects"])
        va = set(m["inner_val_subjects"])
        te = set(m["outer_test_subjects"])
        assert tr.isdisjoint(va), f"Fold {fold}: inner_train overlaps inner_val!"
        assert tr.isdisjoint(te), f"Fold {fold}: inner_train overlaps outer_test!"
        assert va.isdisjoint(te), f"Fold {fold}: inner_val overlaps outer_test!"

        # Cross-verify JSON against authoritative CSV
        df_f = df_auth[df_auth["fold"] == fold]
        csv_tr = set(df_f[df_f["role"] == "inner_train"]["subject_id"])
        csv_va = set(df_f[df_f["role"] == "inner_val"]["subject_id"])
        csv_te = set(df_f[df_f["role"] == "outer_test"]["subject_id"])
        assert tr == csv_tr, f"Fold {fold} inner_train mismatch vs authoritative CSV!"
        assert va == csv_va, f"Fold {fold} inner_val mismatch vs authoritative CSV!"
        assert te == csv_te, f"Fold {fold} outer_test mismatch vs authoritative CSV!"

        manifests[fold] = m
        all_outer_test.extend(m["outer_test_subjects"])

    assert len(all_outer_test) == 47 and len(set(all_outer_test)) == 47, (
        f"Expected 47 unique outer-test subjects across 5 folds, got {len(set(all_outer_test))}"
    )
    return manifests


def compute_epoch_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """Computes 4-class sleep staging metrics strictly on valid epochs [0..3] with NaN-safe kappa."""
    valid = (y_true >= 0) & (y_true < 4)
    yt = y_true[valid]
    yp = y_pred[valid]
    if len(yt) == 0:
        return {
            "n_valid_epochs": 0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "accuracy": 0.0,
            "kappa": 0.0,
            "f1_wake": 0.0,
            "f1_light": 0.0,
            "f1_deep": 0.0,
            "f1_rem": 0.0,
        }

    per_stage = f1_score(yt, yp, average=None, labels=[0, 1, 2, 3], zero_division=0)
    raw_kappa = float(cohen_kappa_score(yt, yp))
    kappa_safe = 0.0 if (math.isnan(raw_kappa) or math.isinf(raw_kappa)) else raw_kappa

    return {
        "n_valid_epochs": int(len(yt)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(yt, yp)),
        "kappa": kappa_safe,
        "f1_wake": float(per_stage[0]),
        "f1_light": float(per_stage[1]),
        "f1_deep": float(per_stage[2]),
        "f1_rem": float(per_stage[3]),
    }


def evaluate_records_sliding(
    predictor: SLAMSSSlidingWindowPredictor,
    records: List[Dict[str, np.ndarray]],
    batch_size: int = 16,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Evaluates a list of night records via batched full-night sliding-window inference.
    Returns:
      - pooled_metrics: aggregate metrics across all valid epochs in records
      - per_subject_metrics: subject_id -> {n_eval_nights, n_valid_epochs, macro_f1, ...}
    """
    night_preds = predictor.predict_nights_batched(records, batch_size=batch_size)

    by_sub_preds: Dict[str, List[np.ndarray]] = {}
    by_sub_trues: Dict[str, List[np.ndarray]] = {}
    by_sub_nights: Dict[str, int] = {}

    for res in night_preds:
        sub = res["subject"]
        by_sub_preds.setdefault(sub, []).append(res["y_pred"])
        by_sub_trues.setdefault(sub, []).append(res["y_true"])
        by_sub_nights[sub] = by_sub_nights.get(sub, 0) + 1

    per_subject_metrics: Dict[str, Dict[str, Any]] = {}
    all_preds, all_trues = [], []
    for sub in sorted(by_sub_preds.keys()):
        sub_p = np.concatenate(by_sub_preds[sub])
        sub_t = np.concatenate(by_sub_trues[sub])
        all_preds.append(sub_p)
        all_trues.append(sub_t)
        per_subject_metrics[sub] = {
            "n_eval_nights": by_sub_nights[sub],
            **compute_epoch_metrics(sub_t, sub_p),
        }

    pooled = compute_epoch_metrics(np.concatenate(all_trues), np.concatenate(all_preds))
    return pooled, per_subject_metrics


def parse_fold_seed_pairs(args: argparse.Namespace) -> List[Tuple[int, int]]:
    """Resolves explicit --pairs (e.g. '1:42 2:123') or Cartesian product of --folds x --seeds."""
    if args.pairs:
        pairs: List[Tuple[int, int]] = []
        for token in args.pairs:
            fold_s, seed_s = token.split(":")
            fold_i, seed_i = int(fold_s), int(seed_s)
            assert fold_i in (1, 2, 3, 4, 5), f"Invalid fold in pair '{token}'"
            pairs.append((fold_i, seed_i))
        return pairs
    return [(f, s) for f in args.folds for s in args.seeds]


def run_single_fold_seed(
    fold: int,
    seed: int,
    manifest: Dict[str, Any],
    records_by_sub: Dict[str, List[Dict[str, np.ndarray]]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Trains and evaluates one (fold, seed) official 5.47M SLAMSS-IFS model."""
    assert args.max_epochs >= 1, f"max_epochs must be >= 1, got {args.max_epochs}"

    runs_dir = os.path.join(args.output_dir, RUN_RESULTS_SUBDIR)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    run_json_path = os.path.join(runs_dir, f"slamss_ifs_fold{fold}_seed{seed}.json")
    ckpt_path = os.path.join(ckpt_dir, f"slamss_ifs_official_L1200_fold{fold}_seed{seed}.pt")

    if (not args.dry_run) and (not args.force) and os.path.exists(run_json_path):
        print(f"[SKIP] Completed run already exists: {run_json_path}")
        with open(run_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # Clear any stale checkpoint from an earlier interrupted run before starting Epoch 1
    if (not args.dry_run) and os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    set_deterministic_seed(seed)

    train_records = [r for s in manifest["inner_train_subjects"] for r in records_by_sub[s]]
    val_records = [r for s in manifest["inner_val_subjects"] for r in records_by_sub[s]]
    test_records = [r for s in manifest["outer_test_subjects"] for r in records_by_sub[s]]

    accum_steps = max(1, args.effective_batch_size // args.micro_batch_size)
    steps_per_epoch = math.ceil(math.ceil(len(train_records) / args.micro_batch_size) / accum_steps)

    print("=" * 80)
    print(f"TSK-08 OFFICIAL SLAMSS-IFS | Fold {fold}/5 | Seed {seed} | Device: {device}")
    print(
        f"Subjects: train={len(manifest['inner_train_subjects'])} ({len(train_records)} nights), "
        f"val={len(manifest['inner_val_subjects'])} ({len(val_records)} nights), "
        f"outer_test={len(manifest['outer_test_subjects'])} ({len(test_records)} nights)"
    )
    print(
        f"Config: L={args.context_len}, micro_B={args.micro_batch_size}, "
        f"eff_B={args.effective_batch_size} (accum={accum_steps}, ~{steps_per_epoch} steps/ep), "
        f"val_B={args.val_batch_size}, lr={args.lr}, max_epochs={args.max_epochs}, "
        f"min_epochs={args.min_epochs}, patience={args.patience}, AMP={use_amp}, "
        f"neutralize_circadian={args.neutralize_circadian}"
    )
    print("=" * 80)

    model = OfficialSLAMSSIFSWrapper(device=device)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == EXPECTED_OFFICIAL_SLAMSS_PARAMS
    loss_fn = OfficialSLAMSSCompositeLoss(rcc_weight=0.01, n_classes=4).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    train_dataset = SLAMSSMatchedDataset(
        records=train_records,
        context_len=args.context_len,
        neutralize_circadian=args.neutralize_circadian,
        random_crop=True,
        seed=seed,
    )
    dl_gen = torch.Generator()
    dl_gen.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        generator=dl_gen,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # Dry-run path: verify 1 forward/backward AMP step + batched sliding-window prediction
    if args.dry_run:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        first_batch = next(iter(train_loader))
        x_sig = first_batch["x_signal"].to(device)
        x_meta = first_batch["x_meta"].to(device)
        y = first_batch["y"].to(device)
        mask = first_batch["mask"].to(device)

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x_sig, x_meta)
        # Strictly outside autocast so CE + 0.01 * RCCLoss executes in pure FP32
        loss = loss_fn(logits, y, mask=mask)

        assert torch.isfinite(loss), f"Dry-run non-finite loss detected: {loss}"
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        predictor = SLAMSSSlidingWindowPredictor(
            model=model,
            context_len=args.context_len,
            neutralize_circadian=args.neutralize_circadian,
            device=device,
            use_amp=use_amp,
        )
        sample_preds = predictor.predict_nights_batched(val_records[:2], batch_size=min(2, args.val_batch_size))
        assert len(sample_preds[0]["y_pred"]) == len(val_records[0]["y"])
        print(
            f"[DRY-RUN PASSED] Fold {fold} Seed {seed}: params={n_params:,}, "
            f"loss={loss.item():.4f}, val_nights_checked={len(sample_preds)}"
        )
        return {"status": "DRY_RUN_PASSED", "fold": fold, "seed": seed}

    # Full training loop with min_epochs floor and inner-validation early stopping
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    total_optimizer_steps = 0
    history: List[Dict[str, float]] = []
    t_start = time.time()
    n_batches = len(train_loader)

    for epoch in range(1, args.max_epochs + 1):
        ep_t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        n_samples = 0

        for step_idx, batch in enumerate(train_loader):
            x_sig = batch["x_signal"].to(device, non_blocking=True)
            x_meta = batch["x_meta"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            # Dynamic divisor prevents underweighting trailing partial accumulation windows
            group_start = (step_idx // accum_steps) * accum_steps
            cur_accum = min(accum_steps, n_batches - group_start)

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x_sig, x_meta)
            # Composite CE + 0.01 * RCCLoss executes strictly in FP32 outside autocast
            loss = loss_fn(logits, y, mask=mask)
            loss_scaled = loss / float(cur_accum)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Numerical instability (NaN/Inf loss) at Fold {fold} Seed {seed} "
                    f"Epoch {epoch} Step {step_idx}: {loss.item()}"
                )

            scaler.scale(loss_scaled).backward()
            bs = x_sig.size(0)
            running_loss += float(loss.item()) * bs
            n_samples += bs

            if (step_idx + 1) % accum_steps == 0 or (step_idx + 1) == n_batches:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                total_optimizer_steps += 1

        train_loss = running_loss / max(n_samples, 1)

        # Batched evaluation on inner validation subjects ONLY (zero outer-test access)
        val_predictor = SLAMSSSlidingWindowPredictor(
            model=model,
            context_len=args.context_len,
            neutralize_circadian=args.neutralize_circadian,
            device=device,
            use_amp=use_amp,
        )
        val_pooled, _ = evaluate_records_sliding(
            val_predictor, val_records, batch_size=args.val_batch_size
        )
        val_f1 = float(val_pooled["macro_f1"])
        assert np.isfinite(val_f1), (
            f"Non-finite validation Macro-F1 at Fold {fold} Seed {seed} Epoch {epoch}: {val_f1}"
        )
        ep_sec = time.time() - ep_t0

        history.append(
            {
                "epoch": epoch,
                "optimizer_steps": total_optimizer_steps,
                "train_loss": round(train_loss, 6),
                "val_macro_f1": round(val_f1, 6),
                "val_accuracy": round(float(val_pooled["accuracy"]), 6),
                "val_weighted_f1": round(float(val_pooled["weighted_f1"]), 6),
                "val_kappa": round(float(val_pooled["kappa"]), 6),
                "epoch_sec": round(ep_sec, 2),
            }
        )

        marker = ""
        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            # Atomic checkpoint write prevents corruption on mid-save kill/OOM
            tmp_ckpt = f"{ckpt_path}.tmp.{os.getpid()}"
            torch.save(
                {
                    "fold": int(fold),
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "optimizer_steps": int(total_optimizer_steps),
                    "val_macro_f1": float(val_f1),
                    "model_state_dict": model.state_dict(),
                    "n_parameters": int(n_params),
                    "neutralize_circadian": bool(args.neutralize_circadian),
                },
                tmp_ckpt,
            )
            os.replace(tmp_ckpt, ckpt_path)
            marker = " * BEST"
        elif epoch >= args.min_epochs:
            epochs_no_improve += 1

        if (
            epoch == 1
            or epoch % args.log_every == 0
            or marker
            or (epoch >= args.min_epochs and epochs_no_improve >= args.patience)
        ):
            print(
                f"Epoch {epoch:03d}/{args.max_epochs:03d} (Steps: {total_optimizer_steps:4d}) | "
                f"Loss: {train_loss:.4f} | Val F1: {val_f1:.4f} | "
                f"Val Acc: {val_pooled['accuracy']:.4f} | {ep_sec:.1f}s{marker}"
            )

        if epoch >= args.min_epochs and epochs_no_improve >= args.patience:
            print(
                f"Early stopping triggered at epoch {epoch} "
                f"(best epoch {best_epoch}, best Val Macro-F1 = {best_val_f1:.4f}, "
                f"total steps = {total_optimizer_steps})."
            )
            break

    assert best_epoch >= 1 and os.path.exists(ckpt_path), (
        f"No valid best checkpoint saved for Fold {fold} Seed {seed} (best_epoch={best_epoch})"
    )

    total_train_sec = time.time() - t_start
    peak_vram_mb = (
        torch.cuda.max_memory_reserved(device) / (1024.0 ** 2) if device.type == "cuda" else 0.0
    )

    # Load best validation checkpoint with weights_only=True and verify epoch integrity
    best_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    assert int(best_ckpt["epoch"]) == best_epoch, (
        f"Loaded checkpoint epoch mismatch: {best_ckpt['epoch']} != {best_epoch}"
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    # Evaluate ONCE on outer test subjects using best validation checkpoint
    test_predictor = SLAMSSSlidingWindowPredictor(
        model=model,
        context_len=args.context_len,
        neutralize_circadian=args.neutralize_circadian,
        device=device,
        use_amp=use_amp,
    )
    test_pooled, test_per_subject = evaluate_records_sliding(
        test_predictor, test_records, batch_size=args.val_batch_size
    )

    run_payload = {
        "experiment_id": "EXP-021",
        "task_id": "TSK-08",
        "model_architecture": "Official_SLAMSS_IFS_Repo_5465768",
        "n_parameters": n_params,
        "fold": fold,
        "seed": seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "context_len": args.context_len,
        "neutralize_circadian": args.neutralize_circadian,
        "micro_batch_size": args.micro_batch_size,
        "effective_batch_size": args.effective_batch_size,
        "val_batch_size": args.val_batch_size,
        "learning_rate": args.lr,
        "max_epochs": args.max_epochs,
        "min_epochs": args.min_epochs,
        "patience": args.patience,
        "amp_enabled": use_amp,
        "best_epoch": best_epoch,
        "total_optimizer_steps": total_optimizer_steps,
        "best_val_macro_f1": best_val_f1,
        "train_time_sec": round(total_train_sec, 2),
        "peak_vram_reserved_mb": round(peak_vram_mb, 2),
        "checkpoint_path": os.path.relpath(ckpt_path, REPO_ROOT),
        "test_pooled_metrics": test_pooled,
        "test_per_subject_metrics": test_per_subject,
        "training_history": history,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "COMPLETED",
    }

    tmp_path = f"{run_json_path}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2)
    os.replace(tmp_path, run_json_path)

    print(
        f"[COMPLETED] Fold {fold} Seed {seed} | Best Ep: {best_epoch} ({total_optimizer_steps} steps) | "
        f"Outer Test Macro-F1: {test_pooled['macro_f1']:.4f} | "
        f"Acc: {test_pooled['accuracy']:.4f} | Peak VRAM: {peak_vram_mb:.1f} MiB | "
        f"Time: {total_train_sec / 60.0:.1f} min"
    )

    del model, optimizer, scaler, train_loader, train_dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return run_payload


def aggregate_completed_runs_to_csv(
    output_dir: str,
    out_csv_path: Optional[str] = None,
) -> Optional[str]:
    """
    Aggregates completed fold-seed run JSONs from <output_dir>/run_results/ into CSV.
    Asserts homogeneous protocol metadata (context_len, neutralize_circadian, n_parameters)
    across all aggregated runs and writes atomically.
    """
    runs_dir = os.path.join(output_dir, RUN_RESULTS_SUBDIR)
    if not os.path.isdir(runs_dir):
        return None

    rows: List[Dict[str, Any]] = []
    ref_protocol: Optional[Tuple[int, bool, int]] = None

    for fname in sorted(os.listdir(runs_dir)):
        if not (fname.startswith("slamss_ifs_fold") and fname.endswith(".json")):
            continue
        with open(os.path.join(runs_dir, fname), "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("status") != "COMPLETED":
            continue

        proto_sig = (
            int(payload["context_len"]),
            bool(payload["neutralize_circadian"]),
            int(payload["n_parameters"]),
        )
        if ref_protocol is None:
            ref_protocol = proto_sig
        else:
            assert proto_sig == ref_protocol, (
                f"Inhomogeneous protocol metadata detected in {fname}: "
                f"{proto_sig} != reference {ref_protocol}"
            )

        fold = int(payload["fold"])
        seed = int(payload["seed"])
        for sub_id, m in payload["test_per_subject_metrics"].items():
            rows.append(
                {
                    "subject_id": sub_id,
                    "fold": fold,
                    "seed": seed,
                    "n_eval_nights": int(m["n_eval_nights"]),
                    "n_valid_epochs": int(m["n_valid_epochs"]),
                    "macro_f1": float(m["macro_f1"]),
                    "weighted_f1": float(m["weighted_f1"]),
                    "accuracy": float(m["accuracy"]),
                    "kappa": float(m["kappa"]),
                    "f1_wake": float(m["f1_wake"]),
                    "f1_light": float(m["f1_light"]),
                    "f1_deep": float(m["f1_deep"]),
                    "f1_rem": float(m["f1_rem"]),
                    "best_epoch": int(payload["best_epoch"]),
                }
            )

    if not rows:
        return None

    out_csv = out_csv_path or os.path.join(output_dir, "slamss_ifs_matched_subject_metrics.csv")
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["subject_id", "seed"]).reset_index(drop=True)
    tmp_csv = f"{out_csv}.tmp.{os.getpid()}"
    df.to_csv(tmp_csv, index=False)
    os.replace(tmp_csv, out_csv)
    print(f"Aggregated {len(df)} subject-seed rows to {out_csv}")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TSK-08 / EXP-021 Official SLAMSS-IFS Matched 5-Fold CV Driver"
    )
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--manifests-dir", default=MANIFESTS_DIR_DEFAULT)
    parser.add_argument("--authoritative-manifest-csv", default=AUTHORITATIVE_MANIFEST_CSV_DEFAULT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR_DEFAULT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folds", nargs="+", type=int, default=FOLDS_DEFAULT)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS_DEFAULT)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help="Explicit fold:seed pairs (e.g., 1:42 1:123 2:42) overriding --folds/--seeds",
    )
    parser.add_argument("--context-len", type=int, default=1200)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--effective-batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    parser.add_argument(
        "--include-circadian",
        dest="neutralize_circadian",
        action="store_false",
        default=True,
        help="Include elapsed time feature (default: neutralized to match locked Model A EXP-018)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run completed fold-seed pairs")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate run JSONs into CSV after finishing")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Verify manifests, 5.47M params, 1 AMP forward/backward step, and batched val inference",
    )
    args = parser.parse_args()

    manifests = verify_manifests_and_splits(args.manifests_dir, args.authoritative_manifest_csv)
    all_subs = sorted({s for m in manifests.values() for s in m["outer_test_subjects"]})
    records_by_split = prepare_dataset_records(
        args.cache_dir, {"train": all_subs, "val": [], "test": []}
    )
    records_by_sub: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for rec in records_by_split["train"]:
        records_by_sub.setdefault(rec["subject"], []).append(rec)
    for sub in records_by_sub:
        records_by_sub[sub].sort(key=lambda r: int(r["night"]))

    pairs = parse_fold_seed_pairs(args)
    print(f"Scheduled {len(pairs)} (fold, seed) pairs on {args.device}: {pairs}")

    for fold, seed in pairs:
        run_single_fold_seed(
            fold=fold,
            seed=seed,
            manifest=manifests[fold],
            records_by_sub=records_by_sub,
            args=args,
        )
        if args.dry_run:
            print("Dry run finished cleanly after 1 verification pair. Zero production files modified.")
            return

    if args.aggregate:
        aggregate_completed_runs_to_csv(args.output_dir)


if __name__ == "__main__":
    main()
