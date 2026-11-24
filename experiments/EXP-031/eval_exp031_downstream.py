"""
EXP-031 Section 10 Downstream Evaluation Suite (Restricted to Section 9 Selected Backbone)
File: experiments/EXP-031/eval_exp031_downstream.py

Reads selected_arm from experiments/EXP-031/selected_backbone.json and runs:
  1. W0-B Dynamic INT8 OOF parity (N=47) on selected_arm
  2. Strictly zero-shot SleepAccel (n=25 qc_clean & N=31 all, em_off only)
     reporting primary single-model mean & secondary 15-model ensemble with
     seed=42 paired bootstrap CIs alongside A (ModelA), full_v5, and SLAMSS-IFS
  3. EXP-020 FT-2 & nested per-fold GATE_tau personalization (N=46) on selected_arm,
     reported alongside A and full_v5
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
import torch.ao.quantization as quantization

EXP031_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP031_DIR, "..", ".."))
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
EXP028_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-028")
EXP027_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-027")
EXP020_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-020")
EXP019_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-019")
STAGE2_DIR = os.path.join(REPO_ROOT, "experiments", "STAGE2_FINAL_MODEL")

for _p in (REPO_ROOT, EXP031_DIR, EXP030_DIR, EXP029_DIR, EXP028_DIR, EXP027_DIR, EXP020_DIR, EXP019_DIR, STAGE2_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from analyze_exp031_confirmatory import profile_single_arm_cpu  # noqa: E402
from analyze_tsk08_paired_benchmark import participant_bootstrap_ci  # noqa: E402
from eval_exp030_downstream import (  # noqa: E402
    TAU_GRID,
    _compute_min_stage_frac,
    _paired_summary,
    select_tau_star_nested,
)
from evaluator import ClinicalHypnogramEvaluator, SleepEvaluator  # noqa: E402
from exp019_adaptation_core import prepare_calibration_batch  # noqa: E402
from exp029_calibration import LogitStitchingSlidingWindowPredictor  # noqa: E402
from exp031_models import EXP031_ELIGIBLE_INT8_ARMS, build_exp031_model  # noqa: E402
from run_exp019_stage_d import evaluate_records_macro_f1  # noqa: E402
from run_exp027_sleepaccel_inference import extract_8_metrics, load_preprocessed_sleepaccel  # noqa: E402
from run_exp028_sleepaccel_bland_altman import SUMMARY_METRICS, compute_bland_altman_stats  # noqa: E402
from run_exp029_cv import (  # noqa: E402
    AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
    CACHE_DIR_DEFAULT,
    MANIFESTS_DIR_DEFAULT,
    compute_epoch_metrics,
    compute_stage_loss_fp32,
    load_all_records_by_subject,
    set_deterministic_seed,
    sha256_file,
    verify_manifests_and_splits,
)

FOLDS = [1, 2, 3, 4, 5]
SEEDS = [42, 123, 2024]


def load_selected_backbone(exp031_dir: str = EXP031_DIR) -> Dict[str, Any]:
    path = os.path.join(exp031_dir, "selected_backbone.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing selected_backbone.json at {path}. Run analyze_exp031_confirmatory.py first."
        )
    with open(path, "r", encoding="utf-8") as f:
        sel = json.load(f)
    arm = sel["selected_arm"]
    if arm not in EXP031_ELIGIBLE_INT8_ARMS:
        raise ValueError(f"Selected arm '{arm}' is not in eligible plain widths {EXP031_ELIGIBLE_INT8_ARMS}")
    return sel


def load_arm_checkpoint_map(exp031_dir: str, arm: str) -> Dict[Tuple[int, int], str]:
    """
    Returns (fold, seed) -> checkpoint_path for any of the 5 eligible plain arms
    (W050, W075, A, P190, B).
    """
    ckpt_map: Dict[Tuple[int, int], str] = {}
    if arm == "A":
        manifest_path = os.path.join(STAGE2_DIR, "frozen_checkpoint_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Frozen manifest missing: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for entry in data["checkpoints"]:
            fold, seed = int(entry["fold"]), int(entry["seed"])
            abs_p = os.path.join(REPO_ROOT, entry["checkpoint_relpath"])
            if not os.path.exists(abs_p):
                raise FileNotFoundError(f"Frozen checkpoint missing: {abs_p}")
            assert sha256_file(abs_p) == entry["sha256"], f"SHA-256 mismatch: {abs_p}"
            ckpt_map[(fold, seed)] = abs_p
        return ckpt_map

    if arm == "B":
        for fold in FOLDS:
            for seed in SEEDS:
                p = os.path.join(EXP029_DIR, "checkpoints", "cnn_bigru_200k", f"cnn_bigru_200k_L1200_fold{fold}_seed{seed}.pt")
                if not os.path.exists(p):
                    raise FileNotFoundError(f"Reused arm B checkpoint missing: {p}")
                ckpt_map[(fold, seed)] = p
        return ckpt_map

    for fold in FOLDS:
        for seed in SEEDS:
            p = os.path.join(exp031_dir, "checkpoints", arm, f"{arm}_L1200_fold{fold}_seed{seed}.pt")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Checkpoint missing for {arm}: {p}")
            ckpt_map[(fold, seed)] = p
    return ckpt_map


def evaluate_w0b_int8_oof(
    exp031_dir: str,
    selected_arm: str,
    cache_dir: str = CACHE_DIR_DEFAULT,
    manifests_dir: str = MANIFESTS_DIR_DEFAULT,
    authoritative_csv: str = AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
) -> Dict[str, Any]:
    profile = profile_single_arm_cpu(selected_arm, benchmark_latency=True)
    ckpt_map = load_arm_checkpoint_map(exp031_dir, selected_arm)
    manifests = verify_manifests_and_splits(manifests_dir, authoritative_csv)
    records_by_sub = load_all_records_by_subject(cache_dir)

    fp32_sub_scores: Dict[str, List[float]] = {}
    int8_sub_scores: Dict[str, List[float]] = {}

    for fold in FOLDS:
        test_subs = manifests[fold]["outer_test_subjects"]
        test_records = [r for s in test_subs for r in records_by_sub[s]]
        for seed in SEEDS:
            ckpt = ckpt_map[(fold, seed)]
            m_fp32 = build_exp031_model(selected_arm).cpu().eval()
            m_fp32.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
            m_int8 = quantization.quantize_dynamic(
                copy.deepcopy(m_fp32), {nn.Linear, nn.GRU}, dtype=torch.qint8
            ).eval()

            p_fp32 = LogitStitchingSlidingWindowPredictor(m_fp32, context_len=1200, device=torch.device("cpu"))
            p_int8 = LogitStitchingSlidingWindowPredictor(m_int8, context_len=1200, device=torch.device("cpu"))

            for pred, store in [(p_fp32, fp32_sub_scores), (p_int8, int8_sub_scores)]:
                by_sub_t: Dict[str, List[np.ndarray]] = {}
                by_sub_p: Dict[str, List[np.ndarray]] = {}
                for rec in test_records:
                    out = pred.predict_night(rec)
                    by_sub_t.setdefault(rec["subject"], []).append(out["y_true"])
                    by_sub_p.setdefault(rec["subject"], []).append(out["y_pred"])
                for sub in test_subs:
                    m = compute_epoch_metrics(np.concatenate(by_sub_t[sub]), np.concatenate(by_sub_p[sub]))
                    store.setdefault(sub, []).append(m["macro_f1"])

    subs = sorted(fp32_sub_scores.keys())
    fp32_means = np.array([np.mean(fp32_sub_scores[s]) for s in subs]) * 100.0
    int8_means = np.array([np.mean(int8_sub_scores[s]) for s in subs]) * 100.0
    diff = int8_means - fp32_means
    ci_low, ci_high = participant_bootstrap_ci(diff, n_boot=10000, alpha=0.05, seed=42)
    _, p_wil = stats.wilcoxon(diff) if np.any(diff != 0) else (0.0, 1.0)

    result = {
        **profile,
        "n_subjects": len(subs),
        "fp32_macro_f1_pct": float(np.mean(fp32_means)),
        "int8_macro_f1_pct": float(np.mean(int8_means)),
        "delta_pp": float(np.mean(diff)),
        "ci95_low_pp": float(ci_low),
        "ci95_high_pp": float(ci_high),
        "wilcoxon_p": float(p_wil),
    }
    out_path = os.path.join(exp031_dir, f"{selected_arm}_w0b_int8_profile.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


@torch.no_grad()
def evaluate_sleepaccel_isolated(
    exp031_dir: str,
    selected_arm: str,
    device_str: str = "cuda:0",
) -> Dict[str, Any]:
    prep_dir = os.path.join(EXP027_DIR, "cache", "preprocessed")
    pred_dir = os.path.join(exp031_dir, "sleepaccel_cache", "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    all_records = load_preprocessed_sleepaccel(prep_dir, qc_only=False)
    evaluator = SleepEvaluator()
    ckpt_map = load_arm_checkpoint_map(exp031_dir, selected_arm)

    probs_raw: Dict[str, List[np.ndarray]] = {r["subject"]: [] for r in all_records}
    for fold in FOLDS:
        for seed in SEEDS:
            model = build_exp031_model(selected_arm).to(device).eval()
            model.load_state_dict(torch.load(ckpt_map[(fold, seed)], map_location=device, weights_only=True))
            pred_off = LogitStitchingSlidingWindowPredictor(model=model, context_len=1200, device=device)
            for rec in all_records:
                out = pred_off.predict_night(rec)
                probs_raw[rec["subject"]].append(out["probs"].astype(np.float32))

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
            os.path.join(pred_dir, f"{sub}__{selected_arm}_em_off.npz"),
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
    df_sum.to_csv(os.path.join(exp031_dir, f"{selected_arm}_sleepaccel_metrics.csv"), index=False)

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
    pd.DataFrame(ba_summary).to_csv(os.path.join(exp031_dir, f"{selected_arm}_sleepaccel_bland_altman.csv"), index=False)

    # Paired comparisons vs pre-declared Section 10 references: A (ModelA), full_v5, SLAMSS_IFS
    df_ref_f1 = pd.read_csv(os.path.join(EXP027_DIR, "exp027_per_subject_metrics.csv"))
    df_ref_f1["subject_id"] = df_ref_f1["subject_id"].astype(str)
    df_v5_f1 = pd.read_csv(os.path.join(EXP029_DIR, "full_v5_sleepaccel_metrics.csv"))
    sub_col = "subject_id" if "subject_id" in df_v5_f1.columns else "subject"
    df_v5_f1["subject_id"] = df_v5_f1[sub_col].astype(str)

    paired_out: Dict[str, Any] = {"selected_arm": selected_arm, "f1_comparisons": []}
    for cohort_name, qc_filter in [("qc_clean", True), ("all", None)]:
        for eval_mode in ["mean_single_model", "prob_ensemble"]:
            df_p = df_sum[df_sum["eval_mode"] == eval_mode]
            if qc_filter is not None:
                df_p = df_p[df_p["qc_pass"]]

            refs = [
                ("ModelA", df_ref_f1[(df_ref_f1["arm"] == "ModelA") & (df_ref_f1["eval_mode"] == eval_mode)]),
                ("full_v5", df_v5_f1[(df_v5_f1["em_mode"] == "em_off") & (df_v5_f1["eval_mode"] == eval_mode)]),
                ("SLAMSS_IFS", df_ref_f1[(df_ref_f1["arm"] == "SLAMSS_IFS") & (df_ref_f1["eval_mode"] == eval_mode)]),
            ]
            for ref_name, df_r in refs:
                if selected_arm == "A" and ref_name == "ModelA":
                    paired_out["f1_comparisons"].append({
                        "cohort": cohort_name, "eval_mode": eval_mode, "reference": ref_name,
                        "self_comparison_guarded": True, "delta_pp": 0.0, "wilcoxon_p": 1.0,
                    })
                    continue
                if qc_filter is not None:
                    df_r = df_r[df_r["qc_pass"]]
                m = pd.merge(df_p, df_r, on="subject_id", suffixes=("_prim", "_ref"))
                f1_res = _paired_summary(
                    m["macro_f1_prim"].values - m["macro_f1_ref"].values,
                    m["macro_f1_prim"].values, m["macro_f1_ref"].values, scale=100.0
                )
                paired_out["f1_comparisons"].append({
                    "cohort": cohort_name, "eval_mode": eval_mode, "reference": ref_name,
                    "n_subjects": int(len(m)), "self_comparison_guarded": False, **f1_res,
                })

    with open(os.path.join(exp031_dir, f"{selected_arm}_sleepaccel_paired_comparisons.json"), "w", encoding="utf-8") as f:
        json.dump(paired_out, f, indent=2)

    agg = {}
    for cohort_name, qc_filter in [("qc_clean", True), ("all", None)]:
        for eval_mode in ["mean_single_model", "prob_ensemble"]:
            sub_df = df_sum[df_sum["eval_mode"] == eval_mode]
            if qc_filter is not None:
                sub_df = sub_df[sub_df["qc_pass"]]
            agg[f"em_off__{eval_mode}__{cohort_name}_macro_f1"] = float(sub_df["macro_f1"].mean() * 100.0)
    with open(os.path.join(exp031_dir, f"{selected_arm}_sleepaccel_summary.json"), "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    return agg


def clone_arm_model_with_state(
    arm: str,
    base_state: Dict[str, torch.Tensor],
    device: torch.device,
    scope_regex: str = r"(classifier_head|temp_norm|temporal_gru)",
) -> Tuple[nn.Module, List[nn.Parameter]]:
    model = build_exp031_model(arm).to(device)
    model.load_state_dict(copy.deepcopy(base_state))
    pattern = re.compile(scope_regex)
    for name, p in model.named_parameters():
        p.requires_grad = bool(pattern.search(name))
    trainable = [p for p in model.parameters() if p.requires_grad]
    return model, trainable


def _run_ft2_single_subject(
    arm: str,
    base_state: Dict[str, torch.Tensor],
    cal_recs: List[Dict[str, np.ndarray]],
    eval_recs: List[Dict[str, np.ndarray]],
    seed: int,
    device: torch.device,
    lr: float = 1e-4,
    epochs: int = 20,
    scope_regex: str = r"(classifier_head|temp_norm|temporal_gru)",
) -> Tuple[float, float]:
    m_zs = build_exp031_model(arm).to(device).eval()
    m_zs.load_state_dict(copy.deepcopy(base_state))
    p_zs = LogitStitchingSlidingWindowPredictor(m_zs, context_len=1200, device=device)
    f1_zs = evaluate_records_macro_f1(p_zs, eval_recs)

    set_deterministic_seed(seed)
    m_ft, trainable = clone_arm_model_with_state(arm, base_state, device, scope_regex)
    opt = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    batch = prepare_calibration_batch(cal_recs, context_len=1200, device=str(device))
    m_ft.train()
    for _ in range(epochs):
        opt.zero_grad()
        out = m_ft(batch["x_ihr"], batch["x_acc"], batch["x_meta"], seq_lens=batch.get("seq_lens", batch.get("seq_len")))
        loss = compute_stage_loss_fp32(m_ft, out["logits"], batch["y"], batch["mask"])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        opt.step()
    m_ft.eval()
    p_ft = LogitStitchingSlidingWindowPredictor(m_ft, context_len=1200, device=device)
    f1_ft = evaluate_records_macro_f1(p_ft, eval_recs)
    return f1_zs, f1_ft


def evaluate_exp020_gate_tau(
    exp031_dir: str,
    selected_arm: str,
    device_str: str = "cuda:0",
    lr: float = 1e-4,
    epochs: int = 20,
    cache_dir: str = CACHE_DIR_DEFAULT,
    manifests_dir: str = MANIFESTS_DIR_DEFAULT,
    authoritative_csv: str = AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
) -> Dict[str, Any]:
    manifests = verify_manifests_and_splits(manifests_dir, authoritative_csv)
    records_by_sub = load_all_records_by_subject(cache_dir)
    ckpt_map = load_arm_checkpoint_map(exp031_dir, selected_arm)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    scope_regex = r"(classifier_head|temp_norm|temporal_gru)"

    rows: List[Dict[str, Any]] = []
    per_fold_selection: Dict[str, Any] = {}

    for fold in FOLDS:
        inner_subs = [s for s in manifests[fold]["inner_val_subjects"] if len(records_by_sub[s]) >= 3]
        outer_subs = [s for s in manifests[fold]["outer_test_subjects"] if len(records_by_sub[s]) >= 3]

        inner_zs: Dict[str, List[float]] = {s: [] for s in inner_subs}
        inner_ft2: Dict[str, List[float]] = {s: [] for s in inner_subs}
        inner_min_frac: Dict[str, float] = {}
        for s in inner_subs:
            recs = sorted(records_by_sub[s], key=lambda r: int(r["night"]))
            inner_min_frac[s] = _compute_min_stage_frac(recs[:2])

        for seed in SEEDS:
            base_state = torch.load(ckpt_map[(fold, seed)], map_location=device, weights_only=True)
            for s in inner_subs:
                recs = sorted(records_by_sub[s], key=lambda r: int(r["night"]))
                f1_zs, f1_ft = _run_ft2_single_subject(
                    selected_arm, base_state, recs[:2], recs[2:], seed, device, lr, epochs, scope_regex
                )
                inner_zs[s].append(f1_zs)
                inner_ft2[s].append(f1_ft)

        inner_zs_mean = {s: float(np.mean(inner_zs[s])) for s in inner_subs}
        inner_ft2_mean = {s: float(np.mean(inner_ft2[s])) for s in inner_subs}
        tau_star, tie_used = select_tau_star_nested(inner_subs, inner_zs_mean, inner_ft2_mean, inner_min_frac, TAU_GRID)

        outer_zs: Dict[str, List[float]] = {s: [] for s in outer_subs}
        outer_ft2: Dict[str, List[float]] = {s: [] for s in outer_subs}
        outer_min_frac: Dict[str, float] = {}
        for s in outer_subs:
            recs = sorted(records_by_sub[s], key=lambda r: int(r["night"]))
            outer_min_frac[s] = _compute_min_stage_frac(recs[:2])

        for seed in SEEDS:
            base_state = torch.load(ckpt_map[(fold, seed)], map_location=device, weights_only=True)
            for s in outer_subs:
                recs = sorted(records_by_sub[s], key=lambda r: int(r["night"]))
                f1_zs, f1_ft = _run_ft2_single_subject(
                    selected_arm, base_state, recs[:2], recs[2:], seed, device, lr, epochs, scope_regex
                )
                outer_zs[s].append(f1_zs)
                outer_ft2[s].append(f1_ft)

        n_abstained_fold = 0
        for s in outer_subs:
            abstained = outer_min_frac[s] < tau_star
            if abstained:
                n_abstained_fold += 1
            zs_m = float(np.mean(outer_zs[s]))
            ft_m = float(np.mean(outer_ft2[s]))
            gate_m = zs_m if abstained else ft_m
            rows.append({
                "subject_id": s,
                "fold": fold,
                "tau_star_fold": tau_star,
                "min_stage_frac": outer_min_frac[s],
                "abstained": abstained,
                "zs_macro_f1": zs_m,
                "ft2_macro_f1": ft_m,
                "gate_tau_macro_f1": gate_m,
            })

        per_fold_selection[str(fold)] = {
            "tau_star": tau_star,
            "tie_breaker_used": tie_used,
            "n_inner_subs": len(inner_subs),
            "n_outer_subs": len(outer_subs),
            "n_outer_abstained": n_abstained_fold,
        }

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(exp031_dir, f"{selected_arm}_exp020_gate_tau_metrics.csv"), index=False)

    ft2_vs_zs = _paired_summary(df["ft2_macro_f1"].values - df["zs_macro_f1"].values, df["ft2_macro_f1"].values, df["zs_macro_f1"].values, scale=100.0)
    gate_vs_zs = _paired_summary(df["gate_tau_macro_f1"].values - df["zs_macro_f1"].values, df["gate_tau_macro_f1"].values, df["zs_macro_f1"].values, scale=100.0)
    gate_vs_ft2 = _paired_summary(df["gate_tau_macro_f1"].values - df["ft2_macro_f1"].values, df["gate_tau_macro_f1"].values, df["ft2_macro_f1"].values, scale=100.0)

    # Reference summaries (A and full_v5)
    ref_summaries: Dict[str, Any] = {}
    exp020_sum = os.path.join(EXP020_DIR, "exp020_summary_metrics.json")
    if os.path.exists(exp020_sum):
        with open(exp020_sum, "r", encoding="utf-8") as f:
            ref_summaries["ModelA_EXP020"] = json.load(f)
    v5_sum = os.path.join(EXP029_DIR, "full_v5_exp020_gate_tau_summary.json")
    if os.path.exists(v5_sum):
        with open(v5_sum, "r", encoding="utf-8") as f:
            ref_summaries["full_v5"] = json.load(f)

    summary = {
        "selected_arm": selected_arm,
        "n_subjects": int(len(df)),
        "per_fold_tau_selection": per_fold_selection,
        "n_abstained_total": int(df["abstained"].sum()),
        "zs_mean_pct": float(df["zs_macro_f1"].mean() * 100.0),
        "ft2_mean_pct": float(df["ft2_macro_f1"].mean() * 100.0),
        "gate_tau_mean_pct": float(df["gate_tau_macro_f1"].mean() * 100.0),
        "ft2_vs_zs": ft2_vs_zs,
        "gate_vs_zs": gate_vs_zs,
        "gate_vs_ft2": gate_vs_ft2,
        "references": ref_summaries,
    }
    with open(os.path.join(exp031_dir, f"{selected_arm}_exp020_gate_tau_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="EXP-031 Section 10 Downstream Evaluation Suite")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["int8", "sleepaccel", "gate_tau"],
        choices=["int8", "sleepaccel", "gate_tau"],
    )
    parser.add_argument("--exp031-dir", type=str, default=EXP031_DIR)
    args = parser.parse_args()

    selection = load_selected_backbone(args.exp031_dir)
    selected_arm = selection["selected_arm"]
    print(f"[EXP-031 DOWNSTREAM] Section 9 Selected Backbone = {selected_arm} ({selection['selected_params']:,} params)")

    if "int8" in args.tasks:
        print(json.dumps(evaluate_w0b_int8_oof(args.exp031_dir, selected_arm), indent=2))
    if "sleepaccel" in args.tasks:
        print(json.dumps(evaluate_sleepaccel_isolated(args.exp031_dir, selected_arm, args.device), indent=2))
    if "gate_tau" in args.tasks:
        print(json.dumps(evaluate_exp020_gate_tau(args.exp031_dir, selected_arm, args.device), indent=2))


if __name__ == "__main__":
    main()
