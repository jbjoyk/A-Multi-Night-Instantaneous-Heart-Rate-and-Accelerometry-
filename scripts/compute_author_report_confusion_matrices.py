#!/usr/bin/env python3
"""
Compute exact 4x4 OOF confusion matrices (Wake, Light, Deep, REM) from saved
checkpoints (BIDSleep N=47, 5-fold CV x 3 seeds) and saved prediction .npz
files (SleepAccel qc_clean n=25, zero-shot) for:
  - Model A (163,940 params)
  - Official SLAMSS-IFS (5,465,768 params)
  - p1_100k (98,940 params)
  - P190 (190,940 params)

Saves:
  - author_report_bundle/data/confusion_matrices.json
  - author_report_bundle/data/confusion_matrices_long.csv
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in [
    REPO_ROOT,
    REPO_ROOT / "experiments" / "EXP-021",
    REPO_ROOT / "experiments" / "EXP-029",
    REPO_ROOT / "experiments" / "EXP-030",
    REPO_ROOT / "experiments" / "EXP-031",
    REPO_ROOT / "experiments" / "STAGE2_FINAL_MODEL",
]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from exp013_models import ConventionalCNNBiGRU  # noqa: E402
from exp029_calibration import LogitStitchingSlidingWindowPredictor  # noqa: E402
from exp030_models import build_exp030_model  # noqa: E402
from exp031_models import build_exp031_model  # noqa: E402
from official_slamss_ifs_adapter import (  # noqa: E402
    OfficialSLAMSSIFSWrapper,
    SLAMSSSlidingWindowPredictor,
)
from run_exp029_cv import (  # noqa: E402
    AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
    CACHE_DIR_DEFAULT,
    MANIFESTS_DIR_DEFAULT,
    compute_epoch_metrics,
    load_all_records_by_subject,
    verify_manifests_and_splits,
)

STAGES = ["Wake", "Light", "Deep", "REM"]
FOLDS = [1, 2, 3, 4, 5]
SEEDS = [42, 123, 2024]


def confusion_4x4(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    valid = (y_true >= 0) & (y_true < 4)
    yt = y_true[valid].astype(np.int64)
    yp = y_pred[valid].astype(np.int64)
    cm = np.bincount(yt * 4 + yp, minlength=16).reshape(4, 4)
    return cm


def eval_bidsleep_model(
    model_name: str,
    manifests: dict,
    records_by_sub: dict,
    device: torch.device,
) -> Dict[str, object]:
    cm_total = np.zeros((4, 4), dtype=np.int64)
    sub_f1s: Dict[str, List[float]] = {}

    for fold in FOLDS:
        test_subs = manifests[fold]["outer_test_subjects"]
        test_records = [r for s in test_subs for r in records_by_sub[s]]
        for seed in SEEDS:
            if model_name == "Model A":
                ckpt = REPO_ROOT / "experiments" / "EXP-018" / "checkpoints" / f"modelA_L1200_fold{fold}_seed{seed}.pt"
                model = ConventionalCNNBiGRU().to(device).eval()
                state = torch.load(ckpt, map_location=device, weights_only=True)
                model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
                pred = LogitStitchingSlidingWindowPredictor(model, context_len=1200, device=device)
                outs = [pred.predict_night(r) for r in test_records]
            elif model_name == "Official SLAMSS-IFS":
                ckpt = REPO_ROOT / "experiments" / "EXP-021" / "checkpoints" / f"slamss_ifs_official_L1200_fold{fold}_seed{seed}.pt"
                model = OfficialSLAMSSIFSWrapper(device=device).eval()
                state = torch.load(ckpt, map_location=device, weights_only=True)
                model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
                pred = SLAMSSSlidingWindowPredictor(model, context_len=1200, neutralize_circadian=True, device=device, use_amp=True)
                outs = pred.predict_nights_batched(test_records, batch_size=16)
            elif model_name == "p1_100k":
                ckpt = REPO_ROOT / "experiments" / "EXP-030" / "checkpoints" / "p1_100k" / f"p1_100k_L1200_fold{fold}_seed{seed}.pt"
                model = build_exp030_model("p1_100k").to(device).eval()
                state = torch.load(ckpt, map_location=device, weights_only=True)
                model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
                pred = LogitStitchingSlidingWindowPredictor(model, context_len=1200, device=device)
                outs = [pred.predict_night(r) for r in test_records]
            elif model_name == "P190":
                ckpt = REPO_ROOT / "experiments" / "EXP-031" / "checkpoints" / "P190" / f"P190_L1200_fold{fold}_seed{seed}.pt"
                model = build_exp031_model("P190").to(device).eval()
                state = torch.load(ckpt, map_location=device, weights_only=True)
                model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
                pred = LogitStitchingSlidingWindowPredictor(model, context_len=1200, device=device)
                outs = [pred.predict_night(r) for r in test_records]
            else:
                raise ValueError(model_name)

            by_sub_t: Dict[str, List[np.ndarray]] = {}
            by_sub_p: Dict[str, List[np.ndarray]] = {}
            for r, o in zip(test_records, outs):
                yt = np.asarray(o["y_true"])
                yp = np.asarray(o["y_pred"])
                cm_total += confusion_4x4(yt, yp)
                by_sub_t.setdefault(r["subject"], []).append(yt)
                by_sub_p.setdefault(r["subject"], []).append(yp)

            for sub in test_subs:
                m = compute_epoch_metrics(np.concatenate(by_sub_t[sub]), np.concatenate(by_sub_p[sub]))
                sub_f1s.setdefault(sub, []).append(m["macro_f1"])

    macro_f1_mean = float(np.mean([np.mean(v) for v in sub_f1s.values()]) * 100.0)
    cm_norm = cm_total / cm_total.sum(axis=1, keepdims=True)
    return {
        "cohort": "BIDSleep (N=47, 5-fold OOF x 3 seeds)",
        "n_subjects": len(sub_f1s),
        "n_valid_epochs_pooled_3seed": int(cm_total.sum()),
        "participant_mean_macro_f1_pct": macro_f1_mean,
        "counts": cm_total.tolist(),
        "row_normalized": cm_norm.tolist(),
    }


def eval_sleepaccel_npz(model_name: str, pattern: str) -> Dict[str, object]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")

    cm_pooled_15 = np.zeros((4, 4), dtype=np.int64)
    cm_ens = np.zeros((4, 4), dtype=np.int64)
    n_qc = 0
    sub_f1s_single = []
    sub_f1s_ens = []

    for fp in files:
        z = np.load(fp)
        if not bool(z["qc_pass"]):
            continue
        n_qc += 1
        yt = z["y_true"]
        ckpt_preds = z["ckpt_preds"]  # (15, N)
        ens_preds = np.argmax(z["ens_probs"], axis=-1)

        f1s_this = []
        for k in range(ckpt_preds.shape[0]):
            cm_pooled_15 += confusion_4x4(yt, ckpt_preds[k])
            f1s_this.append(compute_epoch_metrics(yt, ckpt_preds[k])["macro_f1"])
        sub_f1s_single.append(float(np.mean(f1s_this)))

        cm_ens += confusion_4x4(yt, ens_preds)
        sub_f1s_ens.append(compute_epoch_metrics(yt, ens_preds)["macro_f1"])

    assert n_qc == 25, f"Expected 25 qc_clean subjects, got {n_qc} for {model_name}"
    return {
        "cohort": "SleepAccel (n=25 qc_clean, zero-shot)",
        "n_subjects": n_qc,
        "n_valid_epochs_single_pass": int(cm_ens.sum()),
        "n_valid_epochs_pooled_15ckpt": int(cm_pooled_15.sum()),
        "mean_single_model_macro_f1_pct": float(np.mean(sub_f1s_single) * 100.0),
        "prob_ensemble_macro_f1_pct": float(np.mean(sub_f1s_ens) * 100.0),
        "counts_pooled_15ckpt": cm_pooled_15.tolist(),
        "row_normalized_pooled_15ckpt": (cm_pooled_15 / cm_pooled_15.sum(axis=1, keepdims=True)).tolist(),
        "counts_ensemble": cm_ens.tolist(),
        "row_normalized_ensemble": (cm_ens / cm_ens.sum(axis=1, keepdims=True)).tolist(),
    }


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    manifests = verify_manifests_and_splits(MANIFESTS_DIR_DEFAULT, AUTHORITATIVE_MANIFEST_CSV_DEFAULT)
    records_by_sub = load_all_records_by_subject(CACHE_DIR_DEFAULT)

    out: Dict[str, Dict[str, object]] = {"BIDSleep": {}, "SleepAccel": {}}

    for name in ["Model A", "Official SLAMSS-IFS", "p1_100k", "P190"]:
        print(f"Evaluating BIDSleep OOF: {name}...")
        res = eval_bidsleep_model(name, manifests, records_by_sub, device)
        out["BIDSleep"][name] = res
        print(f"  -> {name} BIDSleep Macro-F1 = {res['participant_mean_macro_f1_pct']:.2f}% (epochs={res['n_valid_epochs_pooled_3seed']:,})")

    sa_patterns = {
        "Model A": str(REPO_ROOT / "experiments" / "EXP-027" / "cache" / "predictions" / "*__ModelA.npz"),
        "Official SLAMSS-IFS": str(REPO_ROOT / "experiments" / "EXP-027" / "cache" / "predictions" / "*__SLAMSS_IFS.npz"),
        "p1_100k": str(REPO_ROOT / "experiments" / "EXP-030" / "sleepaccel_cache" / "predictions" / "*__p1_100k_em_off.npz"),
        "P190": str(REPO_ROOT / "experiments" / "EXP-031" / "sleepaccel_cache" / "predictions" / "*__P190_em_off.npz"),
    }
    for name, pat in sa_patterns.items():
        res = eval_sleepaccel_npz(name, pat)
        out["SleepAccel"][name] = res
        print(f"  -> {name} SleepAccel single={res['mean_single_model_macro_f1_pct']:.2f}%, ens={res['prob_ensemble_macro_f1_pct']:.2f}%")

    data_dir = REPO_ROOT / "author_report_bundle" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    with open(data_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    rows = []
    for ds_key, models in out.items():
        for m_name, d in models.items():
            counts = d["counts"] if ds_key == "BIDSleep" else d["counts_pooled_15ckpt"]
            norm = d["row_normalized"] if ds_key == "BIDSleep" else d["row_normalized_pooled_15ckpt"]
            for i, true_stage in enumerate(STAGES):
                for j, pred_stage in enumerate(STAGES):
                    rows.append({
                        "dataset": ds_key,
                        "model": m_name,
                        "true_stage": true_stage,
                        "pred_stage": pred_stage,
                        "count": int(counts[i][j]),
                        "row_pct": round(float(norm[i][j]) * 100.0, 2),
                    })
    pd.DataFrame(rows).to_csv(data_dir / "confusion_matrices_long.csv", index=False)
    print(f"Saved confusion matrices to {data_dir}")


if __name__ == "__main__":
    main()
