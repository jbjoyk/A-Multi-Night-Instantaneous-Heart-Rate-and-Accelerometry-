"""
EXP-033 Execution Pipeline:
  Part A: Nested selection, FT-k (k=1..4), FT-other-2 (5 donors), GATE-tau
          across W100 (primary) and p1_100k, P190, full_v5 (descriptive).
  Part B: W100 zero-shot SleepAccel (em_off, n=25 & N=31) + Bland-Altman.

File: experiments/EXP-033/run_exp033_pipeline.py
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

EXP033_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP033_DIR, "..", ".."))
EXP031_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-031")
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
EXP028_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-028")
EXP027_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-027")
EXP019_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-019")
STAGE2_DIR = os.path.join(REPO_ROOT, "experiments", "STAGE2_FINAL_MODEL")

for _p in (REPO_ROOT, EXP033_DIR, EXP031_DIR, EXP030_DIR, EXP029_DIR, EXP028_DIR, EXP027_DIR, EXP019_DIR, STAGE2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_exp030_downstream import _compute_min_stage_frac, _paired_summary  # noqa: E402
from evaluator import ClinicalHypnogramEvaluator, SleepEvaluator  # noqa: E402
from exp029_calibration import LogitStitchingSlidingWindowPredictor  # noqa: E402
from exp033_core import (  # noqa: E402
    FOLDS,
    SEEDS,
    build_exp033_backbone,
    get_backbone_ckpt_map,
    get_backbone_oof_csv,
    metrics_from_night_preds,
    predict_nights,
    run_single_ft,
    verify_inputs_sha256,
)
from run_exp027_sleepaccel_inference import extract_8_metrics, load_preprocessed_sleepaccel  # noqa: E402
from run_exp028_sleepaccel_bland_altman import SUMMARY_METRICS, compute_bland_altman_stats  # noqa: E402
from run_exp029_cv import (  # noqa: E402
    AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
    CACHE_DIR_DEFAULT,
    MANIFESTS_DIR_DEFAULT,
    load_all_records_by_subject,
    verify_manifests_and_splits,
)

FT_GRID: List[Dict[str, Any]] = [
    {"lr": lr, "epochs": ep, "scope": sc}
    for lr in (3e-5, 1e-4, 3e-4)
    for ep in (5, 15)
    for sc in ("head_gru", "all")
]


def _select_best_config(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Selection metric: mean subject-level Macro-F1 over seeds.
    Ties go to smaller lr, then fewer epochs, then head_gru before all.
    """
    def sort_key(item: Dict[str, Any]) -> Tuple[float, float, int, int]:
        return (
            -round(item["mean_macro_f1"], 6),
            float(item["lr"]),
            int(item["epochs"]),
            0 if item["scope"] == "head_gru" else 1,
        )

    best = sorted(candidates, key=sort_key)[0]
    return copy.deepcopy(best)


def run_part_a_backbone(backbone: str, device_str: str) -> None:
    t0 = time.time()
    verify_inputs_sha256()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    manifests = verify_manifests_and_splits(MANIFESTS_DIR_DEFAULT, AUTHORITATIVE_MANIFEST_CSV_DEFAULT)
    records_by_sub_raw = load_all_records_by_subject(CACHE_DIR_DEFAULT)
    records_by_sub = {
        sub: sorted(recs, key=lambda r: int(r["night"]))
        for sub, recs in records_by_sub_raw.items()
    }
    ckpt_map = get_backbone_ckpt_map(backbone)
    oof_df = pd.read_csv(get_backbone_oof_csv(backbone)).set_index("subject_id")

    print(f"[EXP-033 Part A] Backbone={backbone} | Device={device}", flush=True)

    # 1. Load frozen checkpoints once on CPU and compute frozen ZS predictions across all outer-test nights
    base_states: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
    zs_night_preds: Dict[Tuple[str, int], List[Dict[str, np.ndarray]]] = {}
    zs_all_nights_f1: Dict[str, List[float]] = {}

    for fold in FOLDS:
        outer_subs = manifests[fold]["outer_test_subjects"]
        for seed in SEEDS:
            st = torch.load(ckpt_map[(fold, seed)], map_location="cpu", weights_only=True)
            base_states[(fold, seed)] = st
            m_zs = build_exp033_backbone(backbone).to(device).eval()
            m_zs.load_state_dict(copy.deepcopy(st))
            for sub in outer_subs:
                recs = records_by_sub[sub]
                preds = predict_nights(m_zs, recs, device)
                zs_night_preds[(sub, seed)] = preds
                m_all = metrics_from_night_preds(preds)
                zs_all_nights_f1.setdefault(sub, []).append(float(m_all["macro_f1"]))
            del m_zs

    # Assert ZS Macro-F1 over all nights matches OOF per-subject value
    max_oof_diff = 0.0
    for sub, f1_list in zs_all_nights_f1.items():
        zs_mean = float(np.mean(f1_list))
        oof_val = float(oof_df.loc[sub, "macro_f1"])
        diff = abs(zs_mean - oof_val)
        max_oof_diff = max(max_oof_diff, diff)
        if backbone == "W100" and diff > 1e-4:
            raise AssertionError(
                f"ZS OOF mismatch for {backbone} {sub}: zs={zs_mean:.6f} vs oof={oof_val:.6f}"
            )
    print(f"[{backbone}] ZS OOF check passed (max abs diff = {max_oof_diff:.2e})", flush=True)

    # 2. Nested inner-validation hyperparameter selection (5 inner-val subjects/fold x 3 seeds x 12 configs)
    selected_by_fold: Dict[int, Dict[str, Any]] = {}
    for fold in FOLDS:
        inner_subs = [s for s in manifests[fold]["inner_val_subjects"] if len(records_by_sub[s]) >= 3]
        candidates: List[Dict[str, Any]] = []
        for cfg in FT_GRID:
            sub_seed_scores: Dict[str, List[float]] = {s: [] for s in inner_subs}
            for seed in SEEDS:
                st = base_states[(fold, seed)]
                for s in inner_subs:
                    recs = records_by_sub[s]
                    m_ft, _, _ = run_single_ft(
                        backbone=backbone,
                        base_state=st,
                        cal_recs=recs[:2],
                        lr=cfg["lr"],
                        epochs=cfg["epochs"],
                        scope=cfg["scope"],
                        seed=seed,
                        device=device,
                    )
                    eval_preds = predict_nights(m_ft, recs[2:], device)
                    sub_seed_scores[s].append(float(metrics_from_night_preds(eval_preds)["macro_f1"]))
                    del m_ft
            sub_means = [float(np.mean(sub_seed_scores[s])) for s in inner_subs]
            candidates.append({
                "lr": cfg["lr"],
                "epochs": cfg["epochs"],
                "scope": cfg["scope"],
                "mean_macro_f1": float(np.mean(sub_means)),
                "n_inner_val_subjects": len(inner_subs),
            })
        best_cfg = _select_best_config(candidates)
        selected_by_fold[fold] = {
            "selected": best_cfg,
            "all_candidates": candidates,
        }
        print(
            f"[{backbone}] Fold {fold} selected: lr={best_cfg['lr']} epochs={best_cfg['epochs']} "
            f"scope={best_cfg['scope']} (inner_val={best_cfg['mean_macro_f1']*100:.2f}%)",
            flush=True,
        )

    sel_path = os.path.join(EXP033_DIR, f"{backbone}_selected_configs.json")
    with open(sel_path, "w", encoding="utf-8") as f:
        json.dump(selected_by_fold, f, indent=2)

    # 3. Outer test evaluation: FT-k (k=1..4) and FT-other-2 (5 donor draws)
    metric_rows: List[Dict[str, Any]] = []
    night_rows: List[Dict[str, Any]] = []
    hash_rows: List[Dict[str, Any]] = []

    for fold in FOLDS:
        cfg = selected_by_fold[fold]["selected"]
        outer_subs = [s for s in manifests[fold]["outer_test_subjects"] if len(records_by_sub[s]) >= 3]
        donor_pool = [s for s in manifests[fold]["inner_train_subjects"] if len(records_by_sub[s]) >= 2]

        for sub in outer_subs:
            recs = records_by_sub[sub]
            n_usable = len(recs)
            min_cal_frac = _compute_min_stage_frac(recs[:2])
            sub_num = int(re.sub(r"\D", "", sub) or "0")

            for seed in SEEDS:
                st = base_states[(fold, seed)]
                zs_preds_all = zs_night_preds[(sub, seed)]

                # FT-k for k = 1, 2, 3, 4
                for k in (1, 2, 3, 4):
                    if n_usable < k + 1:
                        continue
                    zs_eval_m = metrics_from_night_preds(zs_preds_all[k:])
                    zs_e4_m = metrics_from_night_preds(zs_preds_all[4:]) if n_usable >= 5 else None

                    m_ft, h_before, h_after = run_single_ft(
                        backbone=backbone,
                        base_state=st,
                        cal_recs=recs[:k],
                        lr=cfg["lr"],
                        epochs=cfg["epochs"],
                        scope=cfg["scope"],
                        seed=seed,
                        device=device,
                    )
                    hash_rows.append({
                        "backbone": backbone, "fold": fold, "subject_id": sub, "seed": seed,
                        "arm": f"FT-{k}", "donor_id": "self", "hash_before": h_before, "hash_after": h_after,
                    })

                    ft_eval_preds = predict_nights(m_ft, recs[k:], device)
                    ft_eval_m = metrics_from_night_preds(ft_eval_preds)
                    ft_e4_m = metrics_from_night_preds(ft_eval_preds[4 - k:]) if n_usable >= 5 else None
                    del m_ft

                    metric_rows.append({
                        "backbone": backbone,
                        "fold": fold,
                        "subject_id": sub,
                        "seed": seed,
                        "arm": f"FT-{k}",
                        "k": k,
                        "n_usable": n_usable,
                        "min_cal_stage_frac": min_cal_frac,
                        "zs_macro_f1": float(zs_eval_m["macro_f1"]),
                        "ft_macro_f1": float(ft_eval_m["macro_f1"]),
                        "delta_macro_f1": float(ft_eval_m["macro_f1"] - zs_eval_m["macro_f1"]),
                        "zs_f1_wake": float(zs_eval_m["f1_wake"]),
                        "zs_f1_light": float(zs_eval_m["f1_light"]),
                        "zs_f1_deep": float(zs_eval_m["f1_deep"]),
                        "zs_f1_rem": float(zs_eval_m["f1_rem"]),
                        "ft_f1_wake": float(ft_eval_m["f1_wake"]),
                        "ft_f1_light": float(ft_eval_m["f1_light"]),
                        "ft_f1_deep": float(ft_eval_m["f1_deep"]),
                        "ft_f1_rem": float(ft_eval_m["f1_rem"]),
                        "e4_zs_macro_f1": float(zs_e4_m["macro_f1"]) if zs_e4_m is not None else np.nan,
                        "e4_ft_macro_f1": float(ft_e4_m["macro_f1"]) if ft_e4_m is not None else np.nan,
                    })

                    if k == 2:
                        for idx, (rec_i, zp, fp) in enumerate(zip(recs[2:], zs_preds_all[2:], ft_eval_preds), start=3):
                            zm_i = metrics_from_night_preds([zp])
                            fm_i = metrics_from_night_preds([fp])
                            night_rows.append({
                                "backbone": backbone, "fold": fold, "subject_id": sub, "seed": seed,
                                "eval_night_pos": idx, "night": int(rec_i["night"]),
                                "zs_macro_f1": float(zm_i["macro_f1"]),
                                "ft_macro_f1": float(fm_i["macro_f1"]),
                                "delta_macro_f1": float(fm_i["macro_f1"] - zm_i["macro_f1"]),
                            })

                # FT-other-2: 5 independent donor draws from same fold's training set
                rng = np.random.default_rng(100000 * fold + 1000 * seed + sub_num)
                donors = list(rng.choice(donor_pool, size=5, replace=False))
                donor_f1s: List[float] = []
                donor_stages: Dict[str, List[float]] = {k: [] for k in ("f1_wake", "f1_light", "f1_deep", "f1_rem")}
                zs_eval2_m = metrics_from_night_preds(zs_preds_all[2:])

                for d_idx, donor_id in enumerate(donors):
                    donor_recs = records_by_sub[str(donor_id)]
                    m_don, h_before, h_after = run_single_ft(
                        backbone=backbone,
                        base_state=st,
                        cal_recs=donor_recs[:2],
                        lr=cfg["lr"],
                        epochs=cfg["epochs"],
                        scope=cfg["scope"],
                        seed=seed,
                        device=device,
                    )
                    hash_rows.append({
                        "backbone": backbone, "fold": fold, "subject_id": sub, "seed": seed,
                        "arm": f"FT-other-2_draw{d_idx}", "donor_id": str(donor_id),
                        "hash_before": h_before, "hash_after": h_after,
                    })
                    don_preds = predict_nights(m_don, recs[2:], device)
                    don_m = metrics_from_night_preds(don_preds)
                    donor_f1s.append(float(don_m["macro_f1"]))
                    for stg in donor_stages:
                        donor_stages[stg].append(float(don_m[stg]))
                    del m_don

                ft_other_mean = float(np.mean(donor_f1s))
                metric_rows.append({
                    "backbone": backbone,
                    "fold": fold,
                    "subject_id": sub,
                    "seed": seed,
                    "arm": "FT-other-2",
                    "k": 2,
                    "n_usable": n_usable,
                    "min_cal_stage_frac": min_cal_frac,
                    "zs_macro_f1": float(zs_eval2_m["macro_f1"]),
                    "ft_macro_f1": ft_other_mean,
                    "delta_macro_f1": float(ft_other_mean - float(zs_eval2_m["macro_f1"])),
                    "zs_f1_wake": float(zs_eval2_m["f1_wake"]),
                    "zs_f1_light": float(zs_eval2_m["f1_light"]),
                    "zs_f1_deep": float(zs_eval2_m["f1_deep"]),
                    "zs_f1_rem": float(zs_eval2_m["f1_rem"]),
                    "ft_f1_wake": float(np.mean(donor_stages["f1_wake"])),
                    "ft_f1_light": float(np.mean(donor_stages["f1_light"])),
                    "ft_f1_deep": float(np.mean(donor_stages["f1_deep"])),
                    "ft_f1_rem": float(np.mean(donor_stages["f1_rem"])),
                    "e4_zs_macro_f1": np.nan,
                    "e4_ft_macro_f1": np.nan,
                })

    pd.DataFrame(metric_rows).to_csv(os.path.join(EXP033_DIR, f"{backbone}_subject_seed_metrics.csv"), index=False)
    pd.DataFrame(night_rows).to_csv(os.path.join(EXP033_DIR, f"{backbone}_night_index_metrics.csv"), index=False)
    pd.DataFrame(hash_rows).to_csv(os.path.join(EXP033_DIR, f"{backbone}_state_dict_hashes.csv"), index=False)
    elapsed = time.time() - t0
    print(f"[EXP-033 Part A] Completed {backbone} in {elapsed/60.0:.2f} min ({len(hash_rows)} FT runs).", flush=True)


@torch.no_grad()
def run_part_b_sleepaccel(device_str: str = "cuda:0") -> None:
    t0 = time.time()
    verify_inputs_sha256()
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    prep_dir = os.path.join(EXP027_DIR, "cache", "preprocessed")
    pred_dir = os.path.join(EXP033_DIR, "sleepaccel_cache", "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    all_records = load_preprocessed_sleepaccel(prep_dir, qc_only=False)
    evaluator = SleepEvaluator()
    ckpt_map = get_backbone_ckpt_map("W100")

    probs_raw: Dict[str, List[np.ndarray]] = {r["subject"]: [] for r in all_records}
    for fold in FOLDS:
        for seed in SEEDS:
            model = build_exp033_backbone("W100").to(device).eval()
            model.load_state_dict(torch.load(ckpt_map[(fold, seed)], map_location=device, weights_only=True))
            pred_off = LogitStitchingSlidingWindowPredictor(model=model, context_len=1200, device=device)
            for rec in all_records:
                out = pred_off.predict_night(rec)
                probs_raw[rec["subject"]].append(out["probs"].astype(np.float32))
            del model

    summary_rows: List[Dict[str, Any]] = []
    ba_rows: List[Dict[str, Any]] = []
    for rec in all_records:
        sub = rec["subject"]
        y_true = rec["y"]
        ckpt_probs = np.stack(probs_raw[sub], axis=0)
        ckpt_preds = np.argmax(ckpt_probs, axis=-1)
        ens_probs = np.mean(ckpt_probs, axis=0)
        ens_pred = np.argmax(ens_probs, axis=-1)

        np.savez_compressed(
            os.path.join(pred_dir, f"{sub}__W100_em_off.npz"),
            y_true=y_true,
            ckpt_preds=ckpt_preds,
            ens_probs=ens_probs,
            qc_pass=rec["qc_pass"],
        )

        ckpt_m = [extract_8_metrics(evaluator, y_true, ckpt_preds[c]) for c in range(15)]
        mean_m = {k: float(np.mean([m[k] for m in ckpt_m])) for k in ckpt_m[0]}
        ens_m = extract_8_metrics(evaluator, y_true, ens_pred)
        summary_rows.append({"subject_id": str(sub), "qc_pass": bool(rec["qc_pass"]), "em_mode": "em_off", "eval_mode": "mean_single_model", **mean_m})
        summary_rows.append({"subject_id": str(sub), "qc_pass": bool(rec["qc_pass"]), "em_mode": "em_off", "eval_mode": "prob_ensemble", **ens_m})

        valid = (y_true >= 0) & (y_true < 4)
        yt_v = y_true[valid]
        ref_ba = ClinicalHypnogramEvaluator.compute_night_metrics(yt_v)
        ckpt_ba = [ClinicalHypnogramEvaluator.compute_night_metrics(ckpt_preds[c][valid]) for c in range(15)]
        ens_ba = ClinicalHypnogramEvaluator.compute_night_metrics(ens_pred[valid])
        for k in SUMMARY_METRICS:
            ba_rows.append({
                "subject_id": str(sub), "qc_pass": bool(rec["qc_pass"]), "em_mode": "em_off",
                "eval_mode": "mean_single_model", "metric": k,
                "ref_val": float(ref_ba[k]), "pred_val": float(np.mean([m[k] for m in ckpt_ba])),
            })
            ba_rows.append({
                "subject_id": str(sub), "qc_pass": bool(rec["qc_pass"]), "em_mode": "em_off",
                "eval_mode": "prob_ensemble", "metric": k,
                "ref_val": float(ref_ba[k]), "pred_val": float(ens_ba[k]),
            })

    df_sum = pd.DataFrame(summary_rows)
    df_sum.to_csv(os.path.join(EXP033_DIR, "W100_sleepaccel_metrics.csv"), index=False)

    df_ba = pd.DataFrame(ba_rows)
    df_ba["diff"] = df_ba["pred_val"] - df_ba["ref_val"]
    ba_summary: List[Dict[str, Any]] = []
    for cohort_name, qc_filter in [("qc_clean", True), ("all", None)]:
        for eval_mode in ["mean_single_model", "prob_ensemble"]:
            for k in SUMMARY_METRICS:
                sub_df = df_ba[(df_ba["eval_mode"] == eval_mode) & (df_ba["metric"] == k)]
                if qc_filter is not None:
                    sub_df = sub_df[sub_df["qc_pass"]]
                ba_stats = compute_bland_altman_stats(sub_df["ref_val"].values, sub_df["pred_val"].values)
                ba_summary.append({"cohort": cohort_name, "em_mode": "em_off", "eval_mode": eval_mode, "metric": k, **ba_stats})
    pd.DataFrame(ba_summary).to_csv(os.path.join(EXP033_DIR, "W100_sleepaccel_bland_altman.csv"), index=False)

    # Paired comparisons vs frozen Model A, SLAMSS-IFS, p1_100k, and P190
    df_ref_f1 = pd.read_csv(os.path.join(EXP027_DIR, "exp027_per_subject_metrics.csv"))
    df_ref_f1["subject_id"] = df_ref_f1["subject_id"].astype(str)

    df_p1 = pd.read_csv(os.path.join(EXP030_DIR, "p1_100k_sleepaccel_metrics.csv"))
    df_p1["subject_id"] = df_p1["subject_id"].astype(str)

    df_p190 = pd.read_csv(os.path.join(EXP031_DIR, "P190_sleepaccel_metrics.csv"))
    df_p190["subject_id"] = df_p190["subject_id"].astype(str)

    paired_out: Dict[str, Any] = {"backbone": "W100", "f1_comparisons": []}
    for cohort_name, qc_filter in [("qc_clean", True), ("all", None)]:
        for eval_mode in ["mean_single_model", "prob_ensemble"]:
            df_p = df_sum[df_sum["eval_mode"] == eval_mode]
            if qc_filter is not None:
                df_p = df_p[df_p["qc_pass"]]

            refs = [
                ("ModelA", df_ref_f1[(df_ref_f1["arm"] == "ModelA") & (df_ref_f1["eval_mode"] == eval_mode)]),
                ("SLAMSS_IFS", df_ref_f1[(df_ref_f1["arm"] == "SLAMSS_IFS") & (df_ref_f1["eval_mode"] == eval_mode)]),
                ("p1_100k", df_p1[(df_p1["em_mode"] == "em_off") & (df_p1["eval_mode"] == eval_mode)]),
                ("P190", df_p190[(df_p190["em_mode"] == "em_off") & (df_p190["eval_mode"] == eval_mode)]),
            ]
            for ref_name, df_r in refs:
                if qc_filter is not None:
                    df_r = df_r[df_r["qc_pass"]]
                m = pd.merge(df_p, df_r, on="subject_id", suffixes=("_prim", "_ref"))
                f1_res = _paired_summary(
                    m["macro_f1_prim"].values - m["macro_f1_ref"].values,
                    m["macro_f1_prim"].values,
                    m["macro_f1_ref"].values,
                    scale=100.0,
                )
                paired_out["f1_comparisons"].append({
                    "cohort": cohort_name,
                    "eval_mode": eval_mode,
                    "reference": ref_name,
                    "n_subjects": int(len(m)),
                    **f1_res,
                })

    with open(os.path.join(EXP033_DIR, "W100_sleepaccel_paired_comparisons.json"), "w", encoding="utf-8") as f:
        json.dump(paired_out, f, indent=2)

    agg = {}
    for cohort_name, qc_filter in [("qc_clean", True), ("all", None)]:
        for eval_mode in ["mean_single_model", "prob_ensemble"]:
            sub_df = df_sum[df_sum["eval_mode"] == eval_mode]
            if qc_filter is not None:
                sub_df = sub_df[sub_df["qc_pass"]]
            agg[f"em_off__{eval_mode}__{cohort_name}_macro_f1"] = float(sub_df["macro_f1"].mean() * 100.0)
            agg[f"em_off__{eval_mode}__{cohort_name}_accuracy"] = float(sub_df["accuracy"].mean() * 100.0)
    with open(os.path.join(EXP033_DIR, "W100_sleepaccel_summary.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)

    print(f"[EXP-033 Part B] Completed W100 SleepAccel evaluation in {time.time()-t0:.1f}s.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-033 Pipeline")
    parser.add_argument("--backbone", type=str, choices=["W100", "p1_100k", "P190", "full_v5"], default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--part-b", action="store_true")
    args = parser.parse_args()

    if args.backbone is not None:
        run_part_a_backbone(args.backbone, args.device)
    if args.part_b:
        run_part_b_sleepaccel(args.device)


if __name__ == "__main__":
    main()
