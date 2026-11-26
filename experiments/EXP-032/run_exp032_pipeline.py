"""
EXP-032 High-Accuracy Benchmark Unified Pipeline
File: experiments/EXP-032/run_exp032_pipeline.py

Strictly implements docs/designs/PREREG_EXP-032_HIGH_ACCURACY_BENCHMARK.md:
  --stage cache_teachers : Pre-caches deterministic L=1200 (stride=600) inner_train
                           windows and EXP-021 (f, s) teacher logits z_T to disk.
  --stage preflight      : Verifies manifest SHA-256, param/MMAC parity, and runs a
                           1-epoch RTX 3060 wall-clock benchmark for M2 (CRF).
  --stage tune           : Stage A pooled inner_val (N=25, seed=42) tuning (<=4 configs/arm),
                           writing stage_a_screening.csv and locked_configs.json.
  --stage eval_outer     : Stage B locked 45-run evaluation (3 arms x 5 folds x 3 seeds),
                           supporting --arms and --pairs sharding across cuda:0 / cuda:1.
  --stage aggregate      : Produces frozen_checkpoint_manifest.json, {arm}_oof_subject_metrics.csv,
                           and exp032_summary.json with Holm-Bonferroni tests & verdicts.
  --stage eval_external  : Zero-shot SleepAccel (em_off only, n=25 qc_clean & N=31 all,
                           delta_NI=-2.50 pp) + interleaved 1T/4T FP32 CPU latency profiling.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
from torch.utils.data import DataLoader, Dataset

EXP032_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP032_DIR, "..", ".."))
EXP031_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-031")
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
EXP027_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-027")
EXP021_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-021")

for _p in (EXP032_DIR, EXP031_DIR, EXP030_DIR, EXP029_DIR, EXP027_DIR, EXP021_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_tsk08_paired_benchmark import participant_bootstrap_ci  # noqa: E402
from eval_exp029_downstream import holm_bonferroni  # noqa: E402
from eval_exp030_downstream import compute_model_mmacs  # noqa: E402
from evaluator import SleepEvaluator  # noqa: E402
from exp032_models import (  # noqa: E402
    CRFLogitStitchingPredictor,
    EXPECTED_M1_PARAMS,
    EXPECTED_M2_PARAMS,
    EXPECTED_MMACS_L1200,
    build_exp032_model,
    compute_exp032_loss,
    compute_fold_class_weights,
)
from official_slamss_ifs_adapter import OfficialSLAMSSIFSWrapper  # noqa: E402
from run_exp027_sleepaccel_inference import extract_8_metrics, load_preprocessed_sleepaccel  # noqa: E402
from run_exp031_width_sweep import (  # noqa: E402
    AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
    CACHE_DIR_DEFAULT,
    FOLDS_DEFAULT,
    MANIFESTS_DIR_DEFAULT,
    SEEDS_DEFAULT,
    RandomCropSleepDataset,
    load_all_records_by_subject,
    parse_fold_seed_pairs,
    set_deterministic_seed,
    sha256_file,
    verify_manifests_and_splits,
)

EXPECTED_MANIFEST_SHA256 = "36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf"
ARMS_ORDER = ["M1", "M2", "M3"]

STAGE_A_M1_M2_GRID = [
    {"dropout": 0.2, "weight_decay": 1e-4},
    {"dropout": 0.2, "weight_decay": 1e-2},
    {"dropout": 0.3, "weight_decay": 1e-4},
    {"dropout": 0.3, "weight_decay": 1e-2},
]

STAGE_A_M3_KD_GRID = [
    {"tau": 2.0, "alpha_kd": 0.25},
    {"tau": 2.0, "alpha_kd": 0.50},
    {"tau": 4.0, "alpha_kd": 0.25},
    {"tau": 4.0, "alpha_kd": 0.50},
]


# ==============================================================================
# Deterministic Teacher-Cached Dataset for M3 (L=1200, stride=600)
# ==============================================================================

def _build_meta_window(
    ihr_w: np.ndarray,
    freq_w: np.ndarray,
    time_w: np.ndarray,
    neutralize_circadian: bool = True,
) -> np.ndarray:
    """Builds exact 5-channel meta tensor [freq, clock, time_feat, hr_mean, hr_std]."""
    L = ihr_w.shape[0]
    hr_mean = np.mean(ihr_w, axis=-1, keepdims=True).astype(np.float32)
    hr_std = np.std(ihr_w, axis=-1, keepdims=True).astype(np.float32)
    clock = np.zeros((L, 1), dtype=np.float32)
    freq_c = freq_w[:, None].astype(np.float32) if freq_w.ndim == 1 else freq_w.astype(np.float32)
    if neutralize_circadian:
        time_c = np.zeros((L, 1), dtype=np.float32)
    else:
        time_c = time_w[:, None].astype(np.float32) if time_w.ndim == 1 else time_w.astype(np.float32)
    return np.concatenate([freq_c, clock, time_c, hr_mean, hr_std], axis=-1)


def extract_deterministic_windows(
    records: List[Dict[str, Any]],
    context_len: int = 1200,
    stride: int = 600,
    neutralize_circadian: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extracts deterministic L=1200 windows (stride=600) from inner_train records.
    Right-pads nights with N < L using 0.0, y=-100, mask=False.
    """
    windows: List[Dict[str, Any]] = []
    L = int(context_len)
    for rec in records:
        N = len(rec["y"])
        if N <= L:
            pad_len = L - N
            ihr_w = np.pad(rec["ihr"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            acc_w = np.pad(rec["accel"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            freq_arr = rec["freq"][:, None] if rec["freq"].ndim == 1 else rec["freq"]
            time_arr = rec["time"][:, None] if rec["time"].ndim == 1 else rec["time"]
            freq_w = np.pad(freq_arr, ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            time_w = np.pad(time_arr, ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            y_w = np.pad(rec["y"], (0, pad_len), mode="constant", constant_values=-100)
            mask_w = np.zeros(L, dtype=bool)
            mask_w[:N] = True
            meta_w = _build_meta_window(ihr_w, freq_w, time_w, neutralize_circadian)
            windows.append({
                "ihr": ihr_w.astype(np.float32),
                "accel": acc_w.astype(np.float32),
                "meta": meta_w.astype(np.float32),
                "y": y_w.astype(np.int64),
                "mask": mask_w,
                "seq_len": int(N),
                "subject": str(rec["subject"]),
                "night": str(rec["night"]),
            })
        else:
            w_starts = sorted(list(set(range(0, N - L + 1, stride)) | {N - L}))
            for w in w_starts:
                ihr_w = rec["ihr"][w : w + L]
                acc_w = rec["accel"][w : w + L]
                freq_w = rec["freq"][w : w + L]
                time_w = rec["time"][w : w + L]
                y_w = rec["y"][w : w + L]
                mask_w = np.ones(L, dtype=bool)
                meta_w = _build_meta_window(ihr_w, freq_w, time_w, neutralize_circadian)
                windows.append({
                    "ihr": ihr_w.astype(np.float32),
                    "accel": acc_w.astype(np.float32),
                    "meta": meta_w.astype(np.float32),
                    "y": y_w.astype(np.int64),
                    "mask": mask_w,
                    "seq_len": int(L),
                    "subject": str(rec["subject"]),
                    "night": str(rec["night"]),
                })
    return windows


class TeacherCachedDataset(Dataset):
    """Loads pre-cached deterministic inner_train windows + fold-matched EXP-021 teacher logits z_T."""

    def __init__(self, cache_path: str, expected_inner_train_subjects: set):
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing M3 teacher cache: {cache_path}. Run --stage cache_teachers first."
            )
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        cached_subs = set(payload["subjects"])
        assert cached_subs.issubset(expected_inner_train_subjects), (
            f"LEAKAGE DETECTED in {cache_path}: cached subjects outside inner_train!"
        )
        self.ihr = payload["ihr"]
        self.accel = payload["accel"]
        self.meta = payload["meta"]
        self.y = payload["y"]
        self.mask = payload["mask"]
        self.seq_len = payload["seq_len"]
        self.teacher_logits = payload["teacher_logits"]

    def __len__(self) -> int:
        return int(self.ihr.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "ihr": self.ihr[idx],
            "accel": self.accel[idx],
            "meta": self.meta[idx],
            "y": self.y[idx],
            "mask": self.mask[idx],
            "seq_len": self.seq_len[idx],
            "teacher_logits": self.teacher_logits[idx],
        }


# ==============================================================================
# Stage 1: Deterministic Teacher Logit Pre-Caching (EXP-021 -> M3)
# ==============================================================================

def _parse_pairs_str(pairs_str: str) -> List[Tuple[int, int]]:
    if not pairs_str.strip():
        return [(f, s) for f in FOLDS_DEFAULT for s in SEEDS_DEFAULT]
    out: List[Tuple[int, int]] = []
    for tok in pairs_str.split(","):
        tok = tok.strip()
        if not tok:
            continue
        f_s, s_s = tok.split(":")
        out.append((int(f_s), int(s_s)))
    return out


def stage_cache_teachers(args: argparse.Namespace) -> None:
    assert sha256_file(args.manifest_csv) == EXPECTED_MANIFEST_SHA256, (
        "Authoritative fold_manifests.csv SHA-256 mismatch!"
    )
    manifests = verify_manifests_and_splits(args.manifests_dir, args.manifest_csv)
    records_by_sub = load_all_records_by_subject(args.cache_dir)

    cache_out_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(cache_out_dir, exist_ok=True)
    device = torch.device(args.device)

    pairs = _parse_pairs_str(args.pairs)

    for fold, seed in pairs:
        out_path = os.path.join(cache_out_dir, f"teacher_logits_f{fold}_s{seed}.pt")
        if (not args.force) and os.path.exists(out_path):
            print(f"[SKIP] Teacher cache exists: {out_path}")
            continue

        manifest = manifests[fold]
        train_subs = sorted(manifest["inner_train_subjects"])
        train_records = [r for s in train_subs for r in records_by_sub[s]]
        windows = extract_deterministic_windows(
            train_records, context_len=args.context_len, stride=600, neutralize_circadian=True
        )

        ckpt_path = os.path.join(
            EXP021_DIR, "checkpoints", f"slamss_ifs_official_L1200_fold{fold}_seed{seed}.pt"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing EXP-021 teacher checkpoint: {ckpt_path}")

        teacher = OfficialSLAMSSIFSWrapper(device=device)
        ckpt_obj = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt_obj["model_state_dict"] if isinstance(ckpt_obj, dict) and "model_state_dict" in ckpt_obj else ckpt_obj
        teacher.load_state_dict(state_dict, strict=True)
        teacher.eval()

        n_win = len(windows)
        L = args.context_len
        ihr_all = torch.stack([torch.from_numpy(w["ihr"]) for w in windows])
        acc_all = torch.stack([torch.from_numpy(w["accel"]) for w in windows])
        meta_all = torch.stack([torch.from_numpy(w["meta"]) for w in windows])
        y_all = torch.stack([torch.from_numpy(w["y"]) for w in windows])
        mask_all = torch.stack([torch.from_numpy(w["mask"]) for w in windows])
        seq_len_all = torch.tensor([w["seq_len"] for w in windows], dtype=torch.int64)
        z_t_all = torch.empty((n_win, L, 4), dtype=torch.float32)

        t0 = time.time()
        bs = 4
        with torch.no_grad():
            for i in range(0, n_win, bs):
                ihr_b = ihr_all[i : i + bs].to(device)
                acc_b = acc_all[i : i + bs].to(device)
                meta_b = meta_all[i : i + bs].to(device)
                B_cur = ihr_b.shape[0]
                # OfficialSLAMSSIFSWrapper expects x_signal: (B, L*30, 2) = [IHR, Accel_mag]
                x_sig = torch.stack([ihr_b.reshape(B_cur, L * 30), acc_b.reshape(B_cur, L * 30)], dim=-1)
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits_b = teacher(x_sig, meta_b)
                z_t_all[i : i + bs] = logits_b.float().cpu()

        assert torch.isfinite(z_t_all).all(), f"Non-finite teacher logits for fold={fold}, seed={seed}"
        torch.save(
            {
                "fold": fold,
                "seed": seed,
                "teacher_ckpt": ckpt_path,
                "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                "subjects": sorted(list({w["subject"] for w in windows})),
                "ihr": ihr_all,
                "accel": acc_all,
                "meta": meta_all,
                "y": y_all,
                "mask": mask_all,
                "seq_len": seq_len_all,
                "teacher_logits": z_t_all,
            },
            out_path,
        )
        print(f"[CACHED] fold={fold} seed={seed}: {n_win} windows saved to {out_path} ({time.time()-t0:.1f}s)")
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()


# ==============================================================================
# Participant-Level Evaluation Helper (Unweighted Participant Mean Macro-F1)
# ==============================================================================

def evaluate_participant_metrics(
    model: torch.nn.Module,
    records: List[Dict[str, Any]],
    device: torch.device,
    context_len: int = 1200,
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """
    Evaluates records via CRFLogitStitchingPredictor, averaging each subject's nights
    first and returning (unweighted_mean_macro_f1, per_subject_metrics_dict).
    """
    model.eval()
    predictor = CRFLogitStitchingPredictor(
        model=model, context_len=context_len, neutralize_circadian=True, device=device
    )
    evaluator = SleepEvaluator()
    by_sub: Dict[str, List[Dict[str, float]]] = {}

    for rec in records:
        pred = predictor.predict_night(rec)
        y_true = np.asarray(pred["y_true"], dtype=np.int64)
        y_pred = np.asarray(pred["y_pred"], dtype=np.int64)
        m = extract_8_metrics(evaluator, y_true, y_pred)
        by_sub.setdefault(str(rec["subject"]), []).append(m)

    per_sub: Dict[str, Dict[str, float]] = {}
    for sub, m_list in sorted(by_sub.items()):
        per_sub[sub] = {k: float(np.mean([x[k] for x in m_list])) for k in m_list[0]}

    mean_f1 = float(np.mean([v["macro_f1"] for v in per_sub.values()]))
    return mean_f1, per_sub


# ==============================================================================
# Core Training Routine (Shared by Stage A & Stage B)
# ==============================================================================

def train_single_run(
    arm: str,
    fold: int,
    seed: int,
    config: Dict[str, Any],
    manifest: Dict[str, Any],
    records_by_sub: Dict[str, List[Dict[str, Any]]],
    device: torch.device,
    args: argparse.Namespace,
    eval_outer_test: bool,
    ckpt_save_path: Optional[str] = None,
) -> Dict[str, Any]:
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

    class_weights = compute_fold_class_weights(train_records).to(device)

    if arm == "M3":
        cache_path = os.path.join(args.output_dir, "cache", f"teacher_logits_f{fold}_s{seed}.pt")
        train_dataset: Dataset = TeacherCachedDataset(cache_path, set(train_subs))
    else:
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
    n_workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.micro_batch_size,
        shuffle=True,
        generator=dl_gen,
        num_workers=n_workers,
        persistent_workers=(n_workers > 0),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    dropout = float(config["dropout"])
    wd = float(config["weight_decay"])
    tau = float(config.get("tau", 2.0))
    alpha_kd = float(config.get("alpha_kd", 0.25))

    model = build_exp032_model(arm=arm, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_f1 = -1.0
    best_val_per_sub: Dict[str, Dict[str, float]] = {}
    best_epoch = 0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    epochs_no_improve = 0
    t_start = time.time()

    for epoch in range(1, args.max_epochs + 1):
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
            z_t = batch["teacher_logits"].to(device, non_blocking=True) if arm == "M3" else None

            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x_ihr=x_ihr, x_accel=x_acc, x_meta=x_meta, seq_lens=seq_len)

            loss = compute_exp032_loss(
                arm=arm,
                model=model,
                student_logits=out["logits"],
                targets=y,
                seq_mask=mask,
                class_weights=class_weights,
                teacher_logits=z_t,
                tau=tau,
                alpha_kd=alpha_kd,
            )
            assert torch.isfinite(loss), f"Non-finite loss at {arm} f{fold} s{seed} ep{epoch}: {loss}"

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item())
            n_batches += 1

        scheduler.step()

        # Best-checkpoint tracking & patience start strictly at epoch >= min_epochs (40)
        if epoch >= args.min_epochs:
            val_f1, val_sub = evaluate_participant_metrics(model, val_records, device, args.context_len)
            if val_f1 > best_val_f1 + 1e-6:
                best_val_f1 = val_f1
                best_val_per_sub = val_sub
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                break

    wall_sec = time.time() - t_start
    assert best_state is not None, f"No best checkpoint captured for {arm} f{fold} s{seed}"
    model.load_state_dict(best_state)
    if ckpt_save_path is not None:
        os.makedirs(os.path.dirname(ckpt_save_path), exist_ok=True)
        torch.save(best_state, ckpt_save_path)

    result: Dict[str, Any] = {
        "arm": arm,
        "fold": fold,
        "seed": seed,
        "config": config,
        "n_params": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "best_inner_val_macro_f1": best_val_f1,
        "inner_val_per_subject": best_val_per_sub,
        "wall_sec": wall_sec,
        "peak_vram_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
    }

    if eval_outer_test:
        test_f1, test_sub = evaluate_participant_metrics(model, test_records, device, args.context_len)
        result["outer_test_macro_f1"] = test_f1
        result["outer_test_per_subject"] = test_sub

    return result


# ==============================================================================
# Stage 2: Preflight Gate (Unit Verification + 1-Epoch RTX 3060 Benchmark)
# ==============================================================================

def stage_preflight(args: argparse.Namespace) -> None:
    print("=== EXP-032 Preflight Gate ===")
    sha = sha256_file(args.manifest_csv)
    assert sha == EXPECTED_MANIFEST_SHA256, f"Manifest SHA mismatch: {sha}"
    manifests = verify_manifests_and_splits(args.manifests_dir, args.manifest_csv)
    print(f"[PASS] fold_manifests.csv SHA-256 verified ({sha[:12]}...) & 5-fold splits disjoint.")

    for arm, exp_p in [("M1", EXPECTED_M1_PARAMS), ("M2", EXPECTED_M2_PARAMS), ("M3", EXPECTED_M2_PARAMS)]:
        m = build_exp032_model(arm=arm, dropout=0.2)
        mmacs = compute_model_mmacs(m, seq_len=1200, epoch_len=30)
        assert abs(mmacs - EXPECTED_MMACS_L1200) < 0.05, f"{arm} MMACs mismatch: {mmacs}"
        print(f"[PASS] {arm}: params={sum(p.numel() for p in m.parameters()):,}, mmacs_L1200={mmacs:.2f}")

    # 1-epoch RTX 3060 benchmark on M2 (CRF) with real Fold 1 inner_train data
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    records_by_sub = load_all_records_by_subject(args.cache_dir)
    train_subs = manifests[1]["inner_train_subjects"]
    train_records = [r for s in train_subs for r in records_by_sub[s]]
    cw = compute_fold_class_weights(train_records).to(device)
    ds = RandomCropSleepDataset(
        records=train_records,
        subject_to_idx={s: i for i, s in enumerate(sorted(train_subs))},
        context_len=1200,
        neutralize_circadian=True,
        seed=42,
    )
    n_workers = 2 if device.type == "cuda" else 0
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=True,
        num_workers=n_workers,
        persistent_workers=(n_workers > 0),
        pin_memory=(device.type == "cuda"),
    )
    model = build_exp032_model("M2", dropout=0.2).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # Warmup 1 batch so cold CUDA/cuDNN init does not distort steady-state epoch timing
    model.train()
    first_batch = next(iter(loader))
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        _w_out = model(
            x_ihr=first_batch["ihr"].to(device),
            x_accel=first_batch["accel"].to(device),
            x_meta=first_batch["meta"].to(device),
            seq_lens=first_batch["seq_len"],
        )
    _w_loss = compute_exp032_loss(
        "M2", model, _w_out["logits"], first_batch["y"].to(device), first_batch["mask"].to(device), cw
    )
    scaler.scale(_w_loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    t0 = time.time()
    losses = []
    for batch in loader:
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(
                x_ihr=batch["ihr"].to(device),
                x_accel=batch["accel"].to(device),
                x_meta=batch["meta"].to(device),
                seq_lens=batch["seq_len"],
            )
        loss = compute_exp032_loss(
            "M2", model, out["logits"], batch["y"].to(device), batch["mask"].to(device), cw
        )
        assert torch.isfinite(loss), f"Preflight non-finite loss: {loss}"
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.item()))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ep_sec = time.time() - t0
    est_70ep_min = (ep_sec * 70.0) / 60.0
    est_100ep_min = (ep_sec * 100.0) / 60.0
    peak_mb = float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    print(
        f"[PASS] 1-Epoch M2 CRF Benchmark on {device}: {ep_sec:.2f}s/epoch "
        f"(est. 70-epoch early-stop={est_70ep_min:.2f} min, max 100-epoch={est_100ep_min:.2f} min), "
        f"mean_loss={np.mean(losses):.4f}, peak_vram={peak_mb:.1f} MB"
    )
    assert est_70ep_min <= 5.0, f"CRF epoch exceeds PREREG <= 5.0 min/run gate: {ep_sec:.2f}s/epoch ({est_70ep_min:.2f} min)"


# ==============================================================================
# Stage A: Pooled inner_val Tuning (N=25, seed=42)
# ==============================================================================

def stage_tune(args: argparse.Namespace) -> None:
    assert sha256_file(args.manifest_csv) == EXPECTED_MANIFEST_SHA256
    manifests = verify_manifests_and_splits(args.manifests_dir, args.manifest_csv)
    records_by_sub = load_all_records_by_subject(args.cache_dir)
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)
    stage_a_dir = os.path.join(args.output_dir, "stage_a_runs")
    os.makedirs(stage_a_dir, exist_ok=True)
    locked_path = os.path.join(args.output_dir, "locked_configs.json")
    screen_csv = os.path.join(args.output_dir, "stage_a_screening.csv")

    locked: Dict[str, Any] = {}
    if os.path.exists(locked_path):
        with open(locked_path, "r", encoding="utf-8") as f:
            locked = json.load(f)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()] if args.arms else ARMS_ORDER
    req_cids = [int(x.strip()) for x in args.config_ids.split(",") if x.strip()] if args.config_ids else [0, 1, 2, 3]

    for arm in arms:
        if arm in ("M1", "M2"):
            grid = STAGE_A_M1_M2_GRID
        else:
            assert "M2" in locked, "Stage A must finalize M2 in locked_configs.json before M3."
            base_m2 = locked["M2"]["config"]
            grid = [{**base_m2, **kd} for kd in STAGE_A_M3_KD_GRID]

        for cid in req_cids:
            cfg = grid[cid]
            cfg_json = os.path.join(stage_a_dir, f"{arm}_cfg{cid}.json")
            if (not args.force) and os.path.exists(cfg_json):
                print(f"[SKIP] Stage A config already completed: {cfg_json}")
                continue

            fold_sub_f1s: Dict[Tuple[int, str], float] = {}
            for fold in FOLDS_DEFAULT:
                res = train_single_run(
                    arm=arm,
                    fold=fold,
                    seed=42,
                    config=cfg,
                    manifest=manifests[fold],
                    records_by_sub=records_by_sub,
                    device=device,
                    args=args,
                    eval_outer_test=False,
                )
                for sub, sm in res["inner_val_per_subject"].items():
                    fold_sub_f1s[(int(fold), str(sub))] = float(sm["macro_f1"])

            assert len(fold_sub_f1s) == 25, (
                f"Stage A expected 25 (fold, subject) inner_val entries, got {len(fold_sub_f1s)}"
            )
            pooled_f1 = float(np.mean(list(fold_sub_f1s.values())))
            rec = {
                "arm": arm,
                "config_id": cid,
                **cfg,
                "n_inner_val_fold_subjects": len(fold_sub_f1s),
                "pooled_inner_val_macro_f1": pooled_f1,
            }
            with open(cfg_json, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)
            print(f"[STAGE-A] {arm} cfg#{cid} {cfg} -> pooled inner_val Macro-F1={pooled_f1*100:.2f}% (N={len(fold_sub_f1s)})")

        # Finalize arm into locked_configs.json and stage_a_screening.csv once all 4 configs exist
        arm_recs = []
        for cid in range(len(grid)):
            p_c = os.path.join(stage_a_dir, f"{arm}_cfg{cid}.json")
            if os.path.exists(p_c):
                with open(p_c, "r", encoding="utf-8") as f:
                    arm_recs.append(json.load(f))
        if len(arm_recs) == len(grid):
            best_rec = max(arm_recs, key=lambda r: r["pooled_inner_val_macro_f1"])
            best_cfg = {k: best_rec[k] for k in grid[0].keys()}
            if os.path.exists(locked_path):
                with open(locked_path, "r", encoding="utf-8") as f:
                    locked = json.load(f)
            locked[arm] = {
                "config_id": int(best_rec["config_id"]),
                "config": best_cfg,
                "pooled_inner_val_macro_f1": float(best_rec["pooled_inner_val_macro_f1"]),
            }
            with open(locked_path, "w", encoding="utf-8") as f:
                json.dump(locked, f, indent=2)
            print(f"[LOCKED] {arm}: {locked[arm]}")

    # Rebuild full stage_a_screening.csv cleanly from atomic stage_a_runs/*.json
    all_recs = []
    for a in ARMS_ORDER:
        for cid in range(4):
            p_c = os.path.join(stage_a_dir, f"{a}_cfg{cid}.json")
            if os.path.exists(p_c):
                with open(p_c, "r", encoding="utf-8") as f:
                    all_recs.append(json.load(f))
    if all_recs:
        pd.DataFrame(all_recs).to_csv(screen_csv, index=False)


# ==============================================================================
# Stage B: Locked Outer Evaluation (45 Runs)
# ==============================================================================

def stage_eval_outer(args: argparse.Namespace) -> None:
    assert sha256_file(args.manifest_csv) == EXPECTED_MANIFEST_SHA256
    manifests = verify_manifests_and_splits(args.manifests_dir, args.manifest_csv)
    records_by_sub = load_all_records_by_subject(args.cache_dir)
    device = torch.device(args.device)

    locked_path = os.path.join(args.output_dir, "locked_configs.json")
    if not os.path.exists(locked_path):
        raise FileNotFoundError(f"Missing {locked_path}. Run --stage tune first.")
    with open(locked_path, "r", encoding="utf-8") as f:
        locked = json.load(f)

    arms = args.arms.split(",") if args.arms else ARMS_ORDER
    pairs = _parse_pairs_str(args.pairs)

    runs_dir = os.path.join(args.output_dir, "run_results")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    for arm in arms:
        cfg = locked[arm]["config"]
        for fold, seed in pairs:
            run_json = os.path.join(runs_dir, f"{arm}_fold{fold}_seed{seed}.json")
            ckpt_path = os.path.join(ckpt_dir, f"{arm}_fold{fold}_seed{seed}.pt")
            if (not args.force) and os.path.exists(run_json) and os.path.exists(ckpt_path):
                print(f"[SKIP] Completed: {run_json}")
                continue
            res = train_single_run(
                arm=arm,
                fold=fold,
                seed=seed,
                config=cfg,
                manifest=manifests[fold],
                records_by_sub=records_by_sub,
                device=device,
                args=args,
                eval_outer_test=True,
                ckpt_save_path=ckpt_path,
            )
            with open(run_json, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)
            print(
                f"[OUTER] {arm} f{fold} s{seed}: best_ep={res['best_epoch']} "
                f"inner_val={res['best_inner_val_macro_f1']*100:.2f}% "
                f"outer_test={res['outer_test_macro_f1']*100:.2f}% ({res['wall_sec']:.1f}s)"
            )


# ==============================================================================
# Stage C: Aggregate OOF Metrics, Paired Tests & Exhaustive Verdict Rule (§5.1-5.2)
# ==============================================================================

def _paired_vs_ref(
    a_series: pd.Series,
    b_series: pd.Series,
    ni_margin_pp: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Paired participant-level comparison (A - B) in percentage points with:
      - Exact subject_id index alignment assertion
      - 10,000-resample participant bootstrap 95% CI (seed=42)
      - Two-sided paired Wilcoxon signed-rank test
      - Optional one-sided non-inferiority Wilcoxon test on (Delta - ni_margin_pp > 0)
        where ni_margin_pp = -2.50 tests Delta + 2.50 pp > 0.
    """
    a_s = a_series.sort_index()
    b_s = b_series.sort_index()
    assert list(a_s.index) == list(b_s.index), (
        f"Subject index mismatch in paired comparison: {len(a_s)} vs {len(b_s)}"
    )
    a = a_s.values.astype(np.float64)
    b = b_s.values.astype(np.float64)
    diff = (a - b) * 100.0
    mean_d = float(np.mean(diff))
    ci_lo, ci_hi = participant_bootstrap_ci(diff, n_boot=10000, seed=42)
    stat_w, p_two = stats.wilcoxon(diff, alternative="two-sided", zero_method="wilcox")

    out: Dict[str, Any] = {
        "n": int(len(diff)),
        "a_mean_pct": float(np.mean(a) * 100.0),
        "b_mean_pct": float(np.mean(b) * 100.0),
        "delta_pp": mean_d,
        "ci95_low_pp": float(ci_lo),
        "ci95_high_pp": float(ci_hi),
        "wilcoxon_stat": float(stat_w),
        "wilcoxon_p_two_sided": float(p_two),
        "wins_a": int(np.sum(diff > 0)),
    }
    if ni_margin_pp is not None:
        shifted = diff - float(ni_margin_pp)  # e.g. diff - (-2.50) = diff + 2.50
        stat_ni, p_ni = stats.wilcoxon(shifted, alternative="greater", zero_method="wilcox")
        out["ni_margin_pp"] = float(ni_margin_pp)
        out["wilcoxon_ni_stat"] = float(stat_ni)
        out["wilcoxon_p_ni_one_sided"] = float(p_ni)
        out["ci95_low_exceeds_ni"] = bool(ci_lo > float(ni_margin_pp))
    return out


def evaluate_exp032_verdicts(
    primary_stats: Dict[str, Dict[str, Any]],
    acc_stats: Dict[str, Dict[str, Any]],
    sa_single_ni: Dict[str, Dict[str, Any]],
    sa_ens_ni: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Applies exact PREREG_EXP-032 §5.1-§5.2 multiplicity and mutually exclusive,
    exhaustive per-arm and overall verdict logic:
      1. Primary family: Holm-Bonferroni across 3 arms {M1, M2, M3}.
      2. Qualifying arms: K_qualifying arms with delta_pp > 0 and p_holm < 0.05.
      3. Secondary family: Holm-Bonferroni across ALL (3 * K_qualifying) secondary tests:
           (a) BIDSleep N=47 Accuracy superiority (delta_pp > 0, two-sided p_holm < 0.05)
           (b) SleepAccel n=25 mean_single_model NI (ci95_low > -2.50 and one-sided p_holm_ni < 0.05)
           (c) SleepAccel n=25 prob_ensemble NI (ci95_low > -2.50 and one-sided p_holm_ni < 0.05)
      4. Per-arm exhaustive partition:
           - ARM_SUPPORTED:    primary pass AND all 3 secondary gates pass
           - ARM_INCONCLUSIVE: (primary pass AND >=1 secondary gate fails) OR
                               (primary p_holm >= 0.05 AND unadjusted p < 0.05 AND delta_pp >= +0.50)
           - ARM_NOT_SUPPORTED: otherwise
      5. Overall verdict:
           - SUPPORTED: >=1 ARM_SUPPORTED (noting standalone M1/M2 vs teacher-distilled M3)
           - INCONCLUSIVE: 0 ARM_SUPPORTED and >=1 ARM_INCONCLUSIVE
           - NOT SUPPORTED: all 3 ARM_NOT_SUPPORTED
    """
    raw_primary_ps = [float(primary_stats[arm]["wilcoxon_p_two_sided"]) for arm in ARMS_ORDER]
    holm_primary_ps = holm_bonferroni(raw_primary_ps)
    for arm, ph in zip(ARMS_ORDER, holm_primary_ps):
        primary_stats[arm]["wilcoxon_p_holm"] = float(ph)

    qualifying = [
        arm for arm in ARMS_ORDER
        if (primary_stats[arm]["delta_pp"] > 0.0) and (primary_stats[arm]["wilcoxon_p_holm"] < 0.05)
    ]

    sec_tests_keys: List[Tuple[str, str]] = []
    sec_raw_ps: List[float] = []
    for arm in qualifying:
        sec_tests_keys.append((arm, "bidsleep_acc"))
        sec_raw_ps.append(float(acc_stats[arm]["wilcoxon_p_two_sided"]))
        sec_tests_keys.append((arm, "sleepaccel_single_ni"))
        sec_raw_ps.append(float(sa_single_ni[arm]["wilcoxon_p_ni_one_sided"]))
        sec_tests_keys.append((arm, "sleepaccel_ens_ni"))
        sec_raw_ps.append(float(sa_ens_ni[arm]["wilcoxon_p_ni_one_sided"]))

    sec_holm_map: Dict[Tuple[str, str], float] = {}
    if sec_raw_ps:
        sec_holm = holm_bonferroni(sec_raw_ps)
        for key, ph in zip(sec_tests_keys, sec_holm):
            sec_holm_map[key] = float(ph)

    arm_verdicts: Dict[str, Any] = {}
    for arm in ARMS_ORDER:
        p_raw = float(primary_stats[arm]["wilcoxon_p_two_sided"])
        p_h = float(primary_stats[arm]["wilcoxon_p_holm"])
        d_pp = float(primary_stats[arm]["delta_pp"])
        primary_pass = (d_pp > 0.0) and (p_h < 0.05)

        if primary_pass:
            acc_ph = sec_holm_map[(arm, "bidsleep_acc")]
            sa_s_ph = sec_holm_map[(arm, "sleepaccel_single_ni")]
            sa_e_ph = sec_holm_map[(arm, "sleepaccel_ens_ni")]

            g1 = (acc_stats[arm]["delta_pp"] > 0.0) and (acc_ph < 0.05)
            g2 = bool(sa_single_ni[arm]["ci95_low_exceeds_ni"]) and (sa_s_ph < 0.05)
            g3 = bool(sa_ens_ni[arm]["ci95_low_exceeds_ni"]) and (sa_e_ph < 0.05)
            all_sec = g1 and g2 and g3
            verdict = "ARM_SUPPORTED" if all_sec else "ARM_INCONCLUSIVE"
            gates = {
                "bidsleep_acc_pass": g1, "bidsleep_acc_p_holm": acc_ph,
                "sleepaccel_single_ni_pass": g2, "sleepaccel_single_ni_p_holm": sa_s_ph,
                "sleepaccel_ens_ni_pass": g3, "sleepaccel_ens_ni_p_holm": sa_e_ph,
            }
        else:
            trend_pass = (p_raw < 0.05) and (d_pp >= 0.50)
            verdict = "ARM_INCONCLUSIVE" if trend_pass else "ARM_NOT_SUPPORTED"
            gates = None

        arm_verdicts[arm] = {
            "verdict": verdict,
            "primary_pass": primary_pass,
            "is_teacher_distilled": (arm == "M3"),
            "secondary_gates": gates,
        }

    supported_arms = [a for a in ARMS_ORDER if arm_verdicts[a]["verdict"] == "ARM_SUPPORTED"]
    inconclusive_arms = [a for a in ARMS_ORDER if arm_verdicts[a]["verdict"] == "ARM_INCONCLUSIVE"]

    if supported_arms:
        overall = "SUPPORTED"
        standalone_supported = [a for a in supported_arms if a in ("M1", "M2")]
        attribution = (
            f"standalone ({', '.join(standalone_supported)})"
            if standalone_supported
            else "teacher-distilled M3 only"
        )
    elif inconclusive_arms:
        overall = "INCONCLUSIVE"
        attribution = f"inconclusive arms: {', '.join(inconclusive_arms)}"
    else:
        overall = "NOT SUPPORTED"
        attribution = "all 3 arms ARM_NOT_SUPPORTED"

    return {
        "k_qualifying": len(qualifying),
        "n_secondary_tests": len(sec_raw_ps),
        "per_arm": arm_verdicts,
        "overall_verdict": overall,
        "attribution": attribution,
    }


def stage_aggregate(args: argparse.Namespace) -> None:
    runs_dir = os.path.join(args.output_dir, "run_results")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")

    manifest_entries = []
    arm_dfs: Dict[str, pd.DataFrame] = {}

    for arm in ARMS_ORDER:
        sub_seed_rows = []
        for fold in FOLDS_DEFAULT:
            for seed in SEEDS_DEFAULT:
                rj = os.path.join(runs_dir, f"{arm}_fold{fold}_seed{seed}.json")
                cp = os.path.join(ckpt_dir, f"{arm}_fold{fold}_seed{seed}.pt")
                if not os.path.exists(rj) or not os.path.exists(cp):
                    raise FileNotFoundError(f"Missing completed outer run: {rj} / {cp}")
                manifest_entries.append({
                    "arm": arm, "fold": fold, "seed": seed,
                    "checkpoint_relpath": os.path.relpath(cp, REPO_ROOT),
                    "sha256": sha256_file(cp),
                })
                with open(rj, "r", encoding="utf-8") as f:
                    d = json.load(f)
                for sub, sm in d["outer_test_per_subject"].items():
                    sub_seed_rows.append({"subject_id": sub, "fold": fold, "seed": seed, **sm})

        df_raw = pd.DataFrame(sub_seed_rows)
        df_sub = df_raw.groupby("subject_id", as_index=False).mean(numeric_only=True)
        assert len(df_sub) == 47, f"{arm}: expected 47 OOF subjects, got {len(df_sub)}"
        df_sub.to_csv(os.path.join(args.output_dir, f"{arm}_oof_subject_metrics.csv"), index=False)
        arm_dfs[arm] = df_sub.set_index("subject_id").sort_index()

    with open(os.path.join(args.output_dir, "frozen_checkpoint_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"manifest_sha256": EXPECTED_MANIFEST_SHA256, "checkpoints": manifest_entries}, f, indent=2)

    # Load Official SLAMSS-IFS (EXP-021) and W100 (EXP-031-A1) references
    slamss_csv = os.path.join(EXP021_DIR, "slamss_ifs_matched_subject_metrics.csv")
    df_s = pd.read_csv(slamss_csv).groupby("subject_id", as_index=False).mean(numeric_only=True).set_index("subject_id").sort_index()

    w100_csv = os.path.join(EXP031_DIR, "runs", "W100", "W100_subject_metrics.csv")
    df_w100 = pd.read_csv(w100_csv).set_index("subject_id").sort_index()

    primary_raw_ps = []
    primary_stats: Dict[str, Any] = {}
    for arm in ARMS_ORDER:
        res = _paired_vs_ref(arm_dfs[arm]["macro_f1"], df_s["macro_f1"])
        primary_stats[arm] = res
        primary_raw_ps.append(res["wilcoxon_p_two_sided"])

    primary_holm = holm_bonferroni(primary_raw_ps)
    for arm, p_h in zip(ARMS_ORDER, primary_holm):
        primary_stats[arm]["wilcoxon_p_holm"] = float(p_h)

    summary = {
        "experiment_id": "EXP-032",
        "primary_bidsleep_macro_f1_vs_slamss": primary_stats,
        "accuracy_bidsleep_vs_slamss": {
            arm: _paired_vs_ref(arm_dfs[arm]["accuracy"], df_s["accuracy"])
            for arm in ARMS_ORDER
        },
        "descriptive_contrasts_macro_f1": {
            "M1_minus_W100": _paired_vs_ref(arm_dfs["M1"]["macro_f1"], df_w100["macro_f1"]),
            "M2_minus_M1": _paired_vs_ref(arm_dfs["M2"]["macro_f1"], arm_dfs["M1"]["macro_f1"]),
            "M3_minus_M2": _paired_vs_ref(arm_dfs["M3"]["macro_f1"], arm_dfs["M2"]["macro_f1"]),
        },
        "verdicts": "PENDING_STAGE_EVAL_EXTERNAL",
    }
    out_json = os.path.join(args.output_dir, "exp032_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[AGGREGATED] Wrote {out_json} and frozen_checkpoint_manifest.json ({len(manifest_entries)} ckpts).")


# ==============================================================================
# Stage D: Zero-Shot External SleepAccel (em_off) & Interleaved CPU Latency
# ==============================================================================

def verify_frozen_checkpoint_manifest(output_dir: str) -> Dict[Tuple[str, int, int], str]:
    """Enforces M3: verifies all 45 frozen checkpoint SHA-256 digests before touching SleepAccel."""
    manifest_path = os.path.join(output_dir, "frozen_checkpoint_manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Missing frozen_checkpoint_manifest.json at {manifest_path}. Run --stage aggregate first."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data["checkpoints"]
    assert len(entries) == 45, f"Expected 45 frozen checkpoints, got {len(entries)}"
    ckpt_map: Dict[Tuple[str, int, int], str] = {}
    for entry in entries:
        abs_p = os.path.join(REPO_ROOT, entry["checkpoint_relpath"])
        if not os.path.exists(abs_p):
            raise FileNotFoundError(f"Frozen checkpoint missing: {abs_p}")
        actual_sha = sha256_file(abs_p)
        assert actual_sha == entry["sha256"], f"SHA-256 mismatch for {abs_p}!"
        ckpt_map[(str(entry["arm"]), int(entry["fold"]), int(entry["seed"]))] = abs_p
    return ckpt_map


def stage_eval_external(args: argparse.Namespace) -> None:
    from exp030_models import ScaledConventionalCNNBiGRU  # local import for W100 latency comparator

    device = torch.device(args.device)
    ckpt_map = verify_frozen_checkpoint_manifest(args.output_dir)

    locked_path = os.path.join(args.output_dir, "locked_configs.json")
    with open(locked_path, "r", encoding="utf-8") as f:
        locked = json.load(f)

    prep_dir = os.path.join(EXP027_DIR, "cache", "preprocessed")
    sa_records = load_preprocessed_sleepaccel(prep_dir=prep_dir, qc_only=False)
    evaluator = SleepEvaluator()

    # Load Official SLAMSS-IFS SleepAccel reference (EXP-027)
    ref_sa_csv = os.path.join(EXP027_DIR, "exp027_per_subject_metrics.csv")
    df_ref_sa = pd.read_csv(ref_sa_csv)
    df_ref_sa["subject_id"] = df_ref_sa["subject_id"].astype(str)
    ref_slamss_qc = df_ref_sa[(df_ref_sa["arm"] == "SLAMSS_IFS") & (df_ref_sa["qc_pass"] == True)]
    ref_single = ref_slamss_qc[ref_slamss_qc["eval_mode"] == "mean_single_model"].set_index("subject_id")["macro_f1"]
    ref_ens = ref_slamss_qc[ref_slamss_qc["eval_mode"] == "prob_ensemble"].set_index("subject_id")["macro_f1"]

    sa_summary: Dict[str, Any] = {}
    sa_single_ni: Dict[str, Dict[str, Any]] = {}
    sa_ens_ni: Dict[str, Dict[str, Any]] = {}

    for arm in ARMS_ORDER:
        dropout = float(locked[arm]["config"]["dropout"])
        models = []
        for fold in FOLDS_DEFAULT:
            for seed in SEEDS_DEFAULT:
                m = build_exp032_model(arm=arm, dropout=dropout).to(device)
                sd = torch.load(ckpt_map[(arm, fold, seed)], map_location=device, weights_only=False)
                m.load_state_dict(sd, strict=True)
                m.eval()
                models.append(CRFLogitStitchingPredictor(m, context_len=1200, neutralize_circadian=True, device=device))

        rows = []
        for rec in sa_records:
            y_true = np.asarray(rec["y"], dtype=np.int64)
            single_ms = []
            prob_list = []
            for pred_eng in models:
                out = pred_eng.predict_night(rec)
                single_ms.append(extract_8_metrics(evaluator, y_true, np.asarray(out["y_pred"], dtype=np.int64)))
                prob_list.append(np.asarray(out["probs"], dtype=np.float64))

            mean_m = {k: float(np.mean([x[k] for x in single_ms])) for k in single_ms[0]}
            ens_pred = np.argmax(np.mean(prob_list, axis=0), axis=-1)
            ens_m = extract_8_metrics(evaluator, y_true, ens_pred)
            rows.append({"subject_id": str(rec["subject"]), "qc_pass": bool(rec["qc_pass"]), "em_mode": "em_off", "eval_mode": "mean_single_model", **mean_m})
            rows.append({"subject_id": str(rec["subject"]), "qc_pass": bool(rec["qc_pass"]), "em_mode": "em_off", "eval_mode": "prob_ensemble", **ens_m})

        df_sa = pd.DataFrame(rows)
        df_sa["subject_id"] = df_sa["subject_id"].astype(str)
        df_sa.to_csv(os.path.join(args.output_dir, f"{arm}_sleepaccel_metrics.csv"), index=False)

        qc = df_sa[df_sa["qc_pass"] == True]
        arm_single_qc = qc[qc["eval_mode"] == "mean_single_model"].set_index("subject_id")["macro_f1"]
        arm_ens_qc = qc[qc["eval_mode"] == "prob_ensemble"].set_index("subject_id")["macro_f1"]

        sa_single_ni[arm] = _paired_vs_ref(arm_single_qc, ref_single, ni_margin_pp=-2.50)
        sa_ens_ni[arm] = _paired_vs_ref(arm_ens_qc, ref_ens, ni_margin_pp=-2.50)

        sa_summary[arm] = {
            "qc_clean_n25_mean_single_macro_f1": float(arm_single_qc.mean() * 100.0),
            "qc_clean_n25_prob_ensemble_macro_f1": float(arm_ens_qc.mean() * 100.0),
            "all_n31_mean_single_macro_f1": float(df_sa[df_sa["eval_mode"] == "mean_single_model"]["macro_f1"].mean() * 100.0),
            "all_n31_prob_ensemble_macro_f1": float(df_sa[df_sa["eval_mode"] == "prob_ensemble"]["macro_f1"].mean() * 100.0),
            "paired_ni_vs_slamss_single": sa_single_ni[arm],
            "paired_ni_vs_slamss_ensemble": sa_ens_ni[arm],
        }

    with open(os.path.join(args.output_dir, "exp032_sleepaccel_summary.json"), "w", encoding="utf-8") as f:
        json.dump(sa_summary, f, indent=2)

    # Finalize §5.1-§5.2 verdicts in exp032_summary.json
    summary_path = os.path.join(args.output_dir, "exp032_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        summary["verdicts"] = evaluate_exp032_verdicts(
            primary_stats=summary["primary_bidsleep_macro_f1_vs_slamss"],
            acc_stats=summary["accuracy_bidsleep_vs_slamss"],
            sa_single_ni=sa_single_ni,
            sa_ens_ni=sa_ens_ni,
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    # Interleaved round-robin 1T / 4T FP32 CPU latency benchmark alongside W100 & SLAMSS-IFS
    cpu_results: Dict[str, Any] = {}
    x_ihr = torch.randn(1, 1200, 30)
    x_acc = torch.randn(1, 1200, 30)
    x_meta = torch.randn(1, 1200, 5)
    x_sig_slamss = torch.stack([x_ihr.reshape(1, 1200 * 30), x_acc.reshape(1, 1200 * 30)], dim=-1)

    bench_names = ["W100", "M1", "M2", "M3", "SLAMSS_IFS"]
    for n_threads in (1, 4):
        torch.set_num_threads(n_threads)
        mods: Dict[str, torch.nn.Module] = {
            "W100": ScaledConventionalCNNBiGRU(d_enc=64, d_temp=96, head_hidden=64).eval(),
            "M1": build_exp032_model(arm="M1", dropout=0.2).eval(),
            "M2": build_exp032_model(arm="M2", dropout=0.2).eval(),
            "M3": build_exp032_model(arm="M3", dropout=0.2).eval(),
            "SLAMSS_IFS": OfficialSLAMSSIFSWrapper(device=torch.device("cpu")).eval(),
        }
        timings: Dict[str, List[float]] = {k: [] for k in bench_names}
        with torch.no_grad():
            # 1 warmup round + 5 timed interleaved rounds
            for r_idx in range(6):
                for name in bench_names:
                    t0 = time.perf_counter()
                    if name == "SLAMSS_IFS":
                        _ = mods[name](x_sig_slamss, x_meta)
                    else:
                        out = mods[name](x_ihr=x_ihr, x_accel=x_acc, x_meta=x_meta)
                        if getattr(mods[name], "crf", None) is not None:
                            mods[name].crf.viterbi_decode(out["logits"])
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    if r_idx >= 1:
                        timings[name].append(dt_ms)
        cpu_results[f"fp32_{n_threads}t_ms"] = {k: float(np.median(timings[k])) for k in bench_names}

    with open(os.path.join(args.output_dir, "exp032_cpu_latency.json"), "w", encoding="utf-8") as f:
        json.dump(cpu_results, f, indent=2)
    print("[EXTERNAL] Wrote exp032_sleepaccel_summary.json, finalized verdicts, and exp032_cpu_latency.json.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EXP-032 High-Accuracy Benchmark Pipeline")
    p.add_argument("--stage", required=True, choices=["cache_teachers", "preflight", "tune", "eval_outer", "aggregate", "eval_external"])
    p.add_argument("--arms", default="", help="Comma-separated arms subset, e.g. M1,M2")
    p.add_argument("--config-ids", default="", help="Comma-separated Stage A config indices, e.g. 0,1")
    p.add_argument("--pairs", default="", help="Comma-separated fold:seed pairs, e.g. 1:42,1:123")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output-dir", default=EXP032_DIR)
    p.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    p.add_argument("--manifests-dir", default=MANIFESTS_DIR_DEFAULT)
    p.add_argument("--manifest-csv", default=AUTHORITATIVE_MANIFEST_CSV_DEFAULT)
    p.add_argument("--context-len", type=int, default=1200)
    p.add_argument("--micro-batch-size", type=int, default=4)
    p.add_argument("--val-batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--max-epochs", type=int, default=100)
    p.add_argument("--min-epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--force", action="store_true", default=False)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dispatch = {
        "cache_teachers": stage_cache_teachers,
        "preflight": stage_preflight,
        "tune": stage_tune,
        "eval_outer": stage_eval_outer,
        "aggregate": stage_aggregate,
        "eval_external": stage_eval_external,
    }
    dispatch[args.stage](args)


if __name__ == "__main__":
    main()
