"""
EXP-029 5-Fold Nested Cross-Validation Driver
File: experiments/EXP-029/run_exp029_cv.py

Executes 5-fold subject-disjoint CV x 3 seeds (42, 123, 2024) for EXP-029 arms:
  - cnn_bigru_200k (Arm b, 223,268 params)
  - film_only      (Arm c, 172,260 params)
  - pyramid_only   (Arm d, 182,468 params)
  - full_v5        (Arm e, 190,788 params)
  (plus --loss-type ce sensitivity on full_v5).

Protocol Parity with EXP-021:
  - Adam(lr=1e-4, betas=(0.9, 0.999)), B=4, val_B=16, max_epochs=100,
    min_epochs=40, patience=25, grad_clip=1.0, random_crop=True, L=1200.
  - CUDA AMP float16 forward/backward with CE + 0.01 * RCCLoss computed
    strictly in FP32 outside torch.amp.autocast.
  - Verifies review_package/fold_manifests.csv SHA-256 prefix 36d84020.
  - Supports --smoke for 2-epoch dry-run into isolated smoke_out/ without
    touching production checkpoints or run_results.
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

EXP029_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP029_DIR, "..", ".."))
EXP018_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-018")
for _p in (REPO_ROOT, EXP029_DIR, EXP018_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp018_dataset import RandomCropSleepDataset
from exp029_calibration import (
    LogitStitchingSlidingWindowPredictor,
    estimate_pi_src_from_records,
)
from exp029_models import ARM_ALIASES, build_exp029_model
from slamss_ifs_train import load_night_features

CACHE_DIR_DEFAULT = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
MANIFESTS_DIR_DEFAULT = os.path.join(REPO_ROOT, "experiments", "EXP-014", "EXP014_MANIFESTS")
AUTHORITATIVE_MANIFEST_CSV_DEFAULT = os.path.join(REPO_ROOT, "review_package", "fold_manifests.csv")
EXPECTED_MANIFEST_SHA256_PREFIX = "36d84020"
OUTPUT_DIR_DEFAULT = EXP029_DIR

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
    if not os.path.exists(authoritative_csv):
        raise FileNotFoundError(f"Authoritative manifest CSV missing: {authoritative_csv}")

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

        df_f = df_auth[df_auth["fold"] == fold]
        assert tr == set(df_f[df_f["role"] == "inner_train"]["subject_id"])
        assert va == set(df_f[df_f["role"] == "inner_val"]["subject_id"])
        assert te == set(df_f[df_f["role"] == "outer_test"]["subject_id"])

        manifests[fold] = m
        all_outer_test.extend(m["outer_test_subjects"])

    assert len(all_outer_test) == 47 and len(set(all_outer_test)) == 47, (
        f"Expected 47 unique outer-test subjects, got {len(set(all_outer_test))}"
    )
    return manifests


def load_all_records_by_subject(cache_dir: str) -> Dict[str, List[Dict[str, np.ndarray]]]:
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"Feature cache directory missing: {cache_dir}")
    by_sub: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for fn in sorted(os.listdir(cache_dir)):
        if not fn.endswith(".npz"):
            continue
        sub, night = fn[:-4].split("__")
        rec = load_night_features(cache_dir, sub, night)
        if rec is not None and len(rec["y"]) >= 600:
            by_sub.setdefault(sub, []).append(rec)
    return by_sub


def compute_epoch_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
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


def compute_stage_loss_fp32(
    model: torch.nn.Module,
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Computes CE + 0.01 * RCCLoss strictly in FP32 outside AMP autocast."""
    B, T, C = logits.shape
    device_type = logits.device.type if logits.device.type in {"cuda", "cpu"} else "cpu"
    with torch.amp.autocast(device_type=device_type, enabled=False):
        logits_fp32 = logits.float()
        flat_logits = logits_fp32.reshape(-1, C)
        flat_targets = targets.reshape(-1)
        valid = (flat_targets >= 0) & (flat_targets < C)
        if mask is not None:
            valid = valid & mask.reshape(-1).bool()
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, dtype=torch.float32)

        per_epoch_ce = model.ce_loss(flat_logits, torch.clamp(flat_targets, 0, C - 1))
        ce_term = per_epoch_ce.view(B, T)[valid.view(B, T)].mean()

        if getattr(model, "use_rcc_loss", True):
            raw_rcc = model.rcc_loss(flat_logits, flat_targets)
            v_mask = valid.float().unsqueeze(-1).expand(-1, C)
            rcc_term = -torch.sum(raw_rcc * v_mask) / valid.sum().clamp(min=1.0)
            return ce_term + 0.01 * rcc_term
        return ce_term


def evaluate_records(
    predictor: LogitStitchingSlidingWindowPredictor,
    records: List[Dict[str, np.ndarray]],
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    by_sub_preds: Dict[str, List[np.ndarray]] = {}
    by_sub_trues: Dict[str, List[np.ndarray]] = {}
    by_sub_nights: Dict[str, int] = {}

    for rec in records:
        res = predictor.predict_night(rec)
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


def get_run_tag(arm_id: str, loss_type: str) -> str:
    canonical = ARM_ALIASES.get(arm_id, arm_id)
    return canonical if loss_type == "rccloss" else f"{canonical}_{loss_type}"


def parse_fold_seed_pairs(args: argparse.Namespace) -> List[Tuple[int, int]]:
    if args.pairs:
        pairs: List[Tuple[int, int]] = []
        for token in args.pairs:
            f_s, s_s = token.split(":")
            f_i, s_i = int(f_s), int(s_s)
            assert f_i in (1, 2, 3, 4, 5), f"Invalid fold in pair '{token}'"
            pairs.append((f_i, s_i))
        return pairs
    return [(f, s) for f in args.folds for s in args.seeds]


def run_single_fold_seed(
    fold: int,
    seed: int,
    manifest: Dict[str, Any],
    records_by_sub: Dict[str, List[Dict[str, np.ndarray]]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    run_tag = get_run_tag(args.arm, args.loss_type)
    base_out = os.path.join(args.output_dir, "smoke_out") if args.smoke else args.output_dir
    runs_dir = os.path.join(base_out, "run_results", run_tag)
    ckpt_dir = os.path.join(base_out, "checkpoints", run_tag)
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    run_json_path = os.path.join(runs_dir, f"{run_tag}_fold{fold}_seed{seed}.json")
    ckpt_path = os.path.join(ckpt_dir, f"{run_tag}_L1200_fold{fold}_seed{seed}.pt")

    if (not args.smoke) and (not args.force) and os.path.exists(run_json_path):
        print(f"[SKIP] Completed run already exists: {run_json_path}")
        with open(run_json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    device = torch.device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    set_deterministic_seed(seed)

    train_subs = manifest["inner_train_subjects"]
    val_subs = manifest["inner_val_subjects"]
    test_subs = manifest["outer_test_subjects"]
    train_records = [r for s in train_subs for r in records_by_sub[s]]
    val_records = [r for s in val_subs for r in records_by_sub[s]]
    test_records = [r for s in test_subs for r in records_by_sub[s]]

    subject_to_idx = {s: i for i, s in enumerate(sorted(train_subs))}
    train_dataset = RandomCropSleepDataset(
        records=train_records,
        subject_to_idx=subject_to_idx,
        context_len=args.context_len,
        neutralize_circadian=True,
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

    model = build_exp029_model(arm_id=args.arm, loss_type=args.loss_type).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    max_epochs = 2 if args.smoke else args.max_epochs
    min_epochs = 1 if args.smoke else args.min_epochs
    patience = args.patience

    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0
    history: List[Dict[str, float]] = []
    t_start = time.time()

    for epoch in range(1, max_epochs + 1):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            x_ihr = batch["ihr"].to(device, non_blocking=True)
            x_acc = batch["accel"].to(device, non_blocking=True)
            x_meta = batch["meta"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            seq_len = batch["seq_len"]

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x_ihr=x_ihr, x_accel=x_acc, x_meta=x_meta, seq_lens=seq_len)

            # Strictly FP32 loss outside autocast
            loss = compute_stage_loss_fp32(model, out["logits"], y, mask)
            assert torch.isfinite(loss), f"Non-finite loss at epoch {epoch}: {loss}"

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            n_batches += 1

        val_predictor = LogitStitchingSlidingWindowPredictor(
            model=model,
            context_len=args.context_len,
            neutralize_circadian=True,
            device=device,
        )
        val_pooled, _ = evaluate_records(val_predictor, val_records)
        val_f1 = val_pooled["macro_f1"]
        train_loss = running_loss / max(n_batches, 1)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_f1,
            "val_accuracy": val_pooled["accuracy"],
            "elapsed_sec": time.time() - ep_t0,
        })

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0
            tmp_ckpt = ckpt_path + ".tmp"
            torch.save(model.state_dict(), tmp_ckpt)
            os.replace(tmp_ckpt, ckpt_path)
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1 or args.smoke:
            print(
                f"[{run_tag} F{fold} S{seed}] Ep {epoch:03d}/{max_epochs} | "
                f"loss={train_loss:.4f} | val_f1={val_f1*100:.2f}% (best={best_val_f1*100:.2f}% @ {best_epoch})"
            )

        if epoch >= min_epochs and epochs_no_improve >= patience:
            print(f"[{run_tag} F{fold} S{seed}] Early stopping at epoch {epoch} (best={best_epoch}).")
            break

    wall_sec = time.time() - t_start
    peak_vram_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024 ** 2))
        if device.type == "cuda"
        else 0.0
    )

    # Load best checkpoint for final inner_val pi_src & outer_test evaluation
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    pi_src = estimate_pi_src_from_records(
        model=model,
        val_records=val_records,
        context_len=args.context_len,
        neutralize_circadian=True,
        device=device,
    )
    test_predictor = LogitStitchingSlidingWindowPredictor(
        model=model,
        context_len=args.context_len,
        neutralize_circadian=True,
        device=device,
    )
    test_pooled, test_per_sub = evaluate_records(test_predictor, test_records)

    result = {
        "arm": run_tag,
        "fold": fold,
        "seed": seed,
        "n_params": n_params,
        "loss_type": args.loss_type,
        "smoke": bool(args.smoke),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "pi_src_inner_val": pi_src.tolist(),
        "wall_sec": wall_sec,
        "peak_vram_mb": peak_vram_mb,
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": sha256_file(ckpt_path),
        "outer_test_pooled": test_pooled,
        "outer_test_per_subject": test_per_sub,
        "history": history,
    }
    tmp_json = run_json_path + ".tmp"
    with open(tmp_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    os.replace(tmp_json, run_json_path)

    print(
        f"[DONE {run_tag} F{fold} S{seed}] Test F1={test_pooled['macro_f1']*100:.2f}% | "
        f"VRAM={peak_vram_mb:.1f} MB | Time={wall_sec:.1f}s"
    )
    return result


def aggregate_arm_results(args: argparse.Namespace) -> None:
    run_tag = get_run_tag(args.arm, args.loss_type)
    base_out = os.path.join(args.output_dir, "smoke_out") if args.smoke else args.output_dir
    runs_dir = os.path.join(base_out, "run_results", run_tag)

    rows: List[Dict[str, Any]] = []
    for fold in args.folds:
        for seed in args.seeds:
            jpath = os.path.join(runs_dir, f"{run_tag}_fold{fold}_seed{seed}.json")
            if not os.path.exists(jpath):
                raise FileNotFoundError(f"Missing run JSON for aggregation: {jpath}")
            with open(jpath, "r", encoding="utf-8") as f:
                res = json.load(f)
            for sub, m in res["outer_test_per_subject"].items():
                rows.append({
                    "arm": run_tag,
                    "fold": fold,
                    "seed": seed,
                    "subject_id": sub,
                    **m,
                })

    df_all = pd.DataFrame(rows)
    metric_cols = [
        "macro_f1", "weighted_f1", "accuracy", "kappa",
        "f1_wake", "f1_light", "f1_deep", "f1_rem",
    ]
    df_sub = df_all.groupby("subject_id", as_index=False)[metric_cols].mean()
    df_sub["arm"] = run_tag

    sub_csv = os.path.join(base_out, f"{run_tag}_subject_metrics.csv")
    df_sub.to_csv(sub_csv, index=False)

    summary = {
        "arm": run_tag,
        "n_subjects": int(len(df_sub)),
        "mean_metrics": {c: float(df_sub[c].mean()) for c in metric_cols},
        "std_metrics": {c: float(df_sub[c].std(ddof=1)) for c in metric_cols},
    }
    sum_json = os.path.join(base_out, f"{run_tag}_summary.json")
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[AGGREGATED {run_tag}] N={len(df_sub)} subjects | "
        f"Macro-F1={summary['mean_metrics']['macro_f1']*100:.2f}% ± {summary['std_metrics']['macro_f1']*100:.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-029 5-Fold Nested CV Driver")
    parser.add_argument("--arm", type=str, default="full_v5", choices=sorted(ARM_ALIASES.keys()))
    parser.add_argument("--loss-type", type=str, default="rccloss", choices=["rccloss", "ce"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--folds", type=int, nargs="+", default=FOLDS_DEFAULT)
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    parser.add_argument("--pairs", type=str, nargs="*", default=None, help="Explicit fold:seed pairs e.g. 1:42 2:123")
    parser.add_argument("--context-len", type=int, default=1200)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true", help="2-epoch dry-run to isolated smoke_out/")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR_DEFAULT)
    parser.add_argument("--manifests-dir", type=str, default=MANIFESTS_DIR_DEFAULT)
    parser.add_argument("--authoritative-csv", type=str, default=AUTHORITATIVE_MANIFEST_CSV_DEFAULT)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()

    manifests = verify_manifests_and_splits(args.manifests_dir, args.authoritative_csv)
    if args.aggregate:
        aggregate_arm_results(args)
        return

    records_by_sub = load_all_records_by_subject(args.cache_dir)
    pairs = [(1, 42)] if args.smoke and not args.pairs else parse_fold_seed_pairs(args)

    for fold, seed in pairs:
        run_single_fold_seed(fold, seed, manifests[fold], records_by_sub, args)

    if args.smoke:
        # Verify aggregation schema in smoke_out
        args.folds = [pairs[0][0]]
        args.seeds = [pairs[0][1]]
        aggregate_arm_results(args)


if __name__ == "__main__":
    main()
