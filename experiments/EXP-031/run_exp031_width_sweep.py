"""
EXP-031 5-Fold Nested Cross-Validation Width-Sweep Driver
File: experiments/EXP-031/run_exp031_width_sweep.py

Trains the 7 new plain CNN-BiGRU width arms (W050, W075, P190, W125, W150, W200, W250)
across 5 folds x 3 seeds (42, 123, 2024) = 105 runs under the locked EXP-031 recipe.
Reused arms A (Model A, DEC-037) and B (EXP-029 cnn_bigru_200k) are strictly guarded
against retraining.

Reuses EXP-029/EXP-030 helpers (verify_manifests_and_splits, load_all_records_by_subject,
compute_stage_loss_fp32, evaluate_records, LogitStitchingSlidingWindowPredictor).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

EXP031_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP031_DIR, "..", ".."))
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
EXP018_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-018")
for _p in (REPO_ROOT, EXP031_DIR, EXP030_DIR, EXP029_DIR, EXP018_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp018_dataset import RandomCropSleepDataset  # noqa: E402
from exp029_calibration import (  # noqa: E402
    LogitStitchingSlidingWindowPredictor,
    estimate_pi_src_from_records,
)
from exp031_models import (  # noqa: E402
    EXP031_ARMS,
    EXP031_ARMS_ORDER,
    EXP031_NEW_ARMS,
    EXP031_REUSED_PLAIN_ARMS,
    build_exp031_model,
)
from run_exp029_cv import (  # noqa: E402
    AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
    CACHE_DIR_DEFAULT,
    FOLDS_DEFAULT,
    MANIFESTS_DIR_DEFAULT,
    SEEDS_DEFAULT,
    compute_stage_loss_fp32,
    evaluate_records,
    load_all_records_by_subject,
    parse_fold_seed_pairs,
    set_deterministic_seed,
    sha256_file,
    verify_manifests_and_splits,
)

OUTPUT_DIR_DEFAULT = EXP031_DIR


def get_arm_tag(arm: str, loss_type: str) -> str:
    return arm if loss_type == "rccloss" else f"{arm}_{loss_type}"


def validate_trainable_arm(arm: str) -> None:
    if arm in EXP031_REUSED_PLAIN_ARMS:
        raise ValueError(
            f"Arm '{arm}' is a reused historical baseline ({EXP031_ARMS[arm]['reused_oof_csv']}); "
            "do not retrain it under EXP-031."
        )
    if arm not in EXP031_NEW_ARMS:
        raise ValueError(f"Unknown trainable EXP-031 arm '{arm}'. Allowed: {EXP031_NEW_ARMS}")


def run_single_fold_seed(
    arm: str,
    fold: int,
    seed: int,
    manifest: Dict[str, Any],
    records_by_sub: Dict[str, List[Dict[str, np.ndarray]]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    validate_trainable_arm(arm)
    run_tag = get_arm_tag(arm, args.loss_type)
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

    model = build_exp031_model(arm=arm, loss_type=args.loss_type).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    expected_params = int(EXP031_ARMS[arm]["expected_params"])
    assert n_params == expected_params, f"{arm}: expected {expected_params} params, got {n_params}"

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0
    )
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
        "arm": arm,
        "grid_tag": run_tag,
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


def aggregate_arm(
    arm: str,
    loss_type: str,
    folds: List[int],
    seeds: List[int],
    output_dir: str,
    smoke: bool = False,
) -> Dict[str, Any]:
    validate_trainable_arm(arm)
    run_tag = get_arm_tag(arm, loss_type)
    base_out = os.path.join(output_dir, "smoke_out") if smoke else output_dir
    runs_dir = os.path.join(base_out, "run_results", run_tag)

    rows: List[Dict[str, Any]] = []
    n_params = None
    wall_secs = []
    peak_vrams = []
    for fold in folds:
        for seed in seeds:
            jpath = os.path.join(runs_dir, f"{run_tag}_fold{fold}_seed{seed}.json")
            if not os.path.exists(jpath):
                raise FileNotFoundError(f"Missing run JSON for aggregation: {jpath}")
            with open(jpath, "r", encoding="utf-8") as f:
                res = json.load(f)
            n_params = res.get("n_params", n_params)
            if "wall_sec" in res:
                wall_secs.append(float(res["wall_sec"]))
            if "peak_vram_mb" in res:
                peak_vrams.append(float(res["peak_vram_mb"]))
            for sub, m in res["outer_test_per_subject"].items():
                rows.append({
                    "arm": arm,
                    "grid_tag": run_tag,
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
    df_sub["arm"] = arm
    df_sub["grid_tag"] = run_tag

    sub_csv = os.path.join(base_out, f"{run_tag}_subject_metrics.csv")
    df_sub.to_csv(sub_csv, index=False)

    summary = {
        "arm": arm,
        "grid_tag": run_tag,
        "n_params": n_params,
        "n_subjects": int(len(df_sub)),
        "mean_metrics": {c: float(df_sub[c].mean()) for c in metric_cols},
        "std_metrics": {c: float(df_sub[c].std(ddof=1)) if len(df_sub) > 1 else 0.0 for c in metric_cols},
        "mean_wall_sec": float(np.mean(wall_secs)) if wall_secs else 0.0,
        "max_peak_vram_mb": float(np.max(peak_vrams)) if peak_vrams else 0.0,
    }
    sum_json = os.path.join(base_out, f"{run_tag}_summary.json")
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[AGGREGATED {run_tag}] N={len(df_sub)} subjects | "
        f"Macro-F1={summary['mean_metrics']['macro_f1']*100:.2f}% ± {summary['std_metrics']['macro_f1']*100:.2f}%"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-031 Single-Factor Width Sweep Driver")
    parser.add_argument(
        "--arms",
        type=str,
        nargs="+",
        default=EXP031_NEW_ARMS,
        choices=EXP031_ARMS_ORDER,
    )
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
    parser.add_argument("--smoke", action="store_true", help="2-epoch dry-run into isolated smoke_out/")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR_DEFAULT)
    parser.add_argument("--manifests-dir", type=str, default=MANIFESTS_DIR_DEFAULT)
    parser.add_argument("--authoritative-csv", type=str, default=AUTHORITATIVE_MANIFEST_CSV_DEFAULT)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR_DEFAULT)
    args = parser.parse_args()

    for arm in args.arms:
        validate_trainable_arm(arm)

    manifests = verify_manifests_and_splits(args.manifests_dir, args.authoritative_csv)

    if args.aggregate:
        for arm in args.arms:
            aggregate_arm(arm, args.loss_type, args.folds, args.seeds, args.output_dir, smoke=args.smoke)
        return

    records_by_sub = load_all_records_by_subject(args.cache_dir)
    pairs = [(1, 42)] if args.smoke and not args.pairs else parse_fold_seed_pairs(args)

    for arm in args.arms:
        for fold, seed in pairs:
            run_single_fold_seed(arm, fold, seed, manifests[fold], records_by_sub, args)

        if args.smoke:
            aggregate_arm(
                arm,
                args.loss_type,
                [pairs[0][0]],
                [pairs[0][1]],
                args.output_dir,
                smoke=True,
            )
        else:
            run_tag = get_arm_tag(arm, args.loss_type)
            runs_dir = os.path.join(args.output_dir, "run_results", run_tag)
            all_present = all(
                os.path.exists(os.path.join(runs_dir, f"{run_tag}_fold{f}_seed{s}.json"))
                for f in FOLDS_DEFAULT
                for s in SEEDS_DEFAULT
            )
            if all_present:
                aggregate_arm(
                    arm, args.loss_type, FOLDS_DEFAULT, SEEDS_DEFAULT, args.output_dir, smoke=False
                )


if __name__ == "__main__":
    main()
