#!/usr/bin/env python3
"""
run_exp020_regularized_gated_adaptation.py
==========================================
EXP-020 Exploratory Closure (cuda:0):
  - Arm (a): FT-2-KL (logit-space KL regularization toward frozen ZS)
  - Arm (b): GATE_tau (calibration min-stage-fraction abstention gate)
  - Arm (c): Combined FT-2-KL + GATE_tau (strictly per-fold nested gate)

Strictly adheres to docs/designs/exp019b_exp020_closure_spec.md.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import f1_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in [
    REPO_ROOT,
    os.path.join(REPO_ROOT, "experiments", "EXP-019"),
    os.path.join(REPO_ROOT, "experiments", "EXP-021"),
    os.path.join(REPO_ROOT, "experiments", "STAGE2_FINAL_MODEL"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp019_adaptation_core import prepare_calibration_batch
from run_exp019_stage_d import evaluate_records_macro_f1
from analyze_tsk08_paired_benchmark import participant_bootstrap_ci
from final_model_pipeline import (
    FinalModelConfig,
    FinalSleepStagingSystem,
    SCOPE_REGEX_MAP,
    audit_calibration_stage_coverage,
)
from slamss_ifs_train import load_night_features

DEFAULT_CACHE_DIR = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
DEFAULT_MANIFEST_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-014", "EXP014_MANIFESTS")
DEFAULT_FROZEN_MANIFEST = os.path.join(REPO_ROOT, "experiments", "STAGE2_FINAL_MODEL", "frozen_checkpoint_manifest.json")
DEFAULT_NIGHT_INVENTORY = os.path.join(REPO_ROOT, "review_package", "night_inventory.csv")
DEFAULT_STAGE_D_JSON = os.path.join(REPO_ROOT, "experiments", "EXP-019", "exp019_stage_d_selected_configs.json")
DEFAULT_EXP019_METRICS = os.path.join(REPO_ROOT, "experiments", "EXP-019", "exp019_subject_metrics.csv")

LAMBDA_KL_GRID = [0.1, 1.0, 10.0]
TAU_GRID = [0.01, 0.02, 0.03, 0.05]
FOLDS = [1, 2, 3, 4, 5]
SEEDS = [42, 123, 2024]


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def restore_pristine_model(
    system: FinalSleepStagingSystem,
    scope_regex: str = r".*",
    allow_unloaded: bool = False,
) -> nn.Module:
    """
    W0-A invariant: clones a fresh ConventionalCNNBiGRU from system._frozen_state,
    resets requires_grad=True on ALL parameters before applying scope_regex,
    and returns the model in .eval() mode.
    """
    clone = system._clone_frozen_backbone(allow_unloaded=allow_unloaded)
    for param in clone.parameters():
        param.requires_grad = True
    pattern = re.compile(scope_regex)
    for name, param in clone.named_parameters():
        param.requires_grad = bool(pattern.search(name))
    clone.eval()
    return clone


def compute_kl_regularized_loss(
    logits_theta: torch.Tensor,
    p_zs: torch.Tensor,
    base_loss: torch.Tensor,
    valid_mask: torch.Tensor,
    lambda_kl: float,
) -> torch.Tensor:
    """
    Exact EXP-020 objective:
      L_total = L_FT-2 + lambda_kl * mean_{valid}(KL(p_ZS || softmax(z_theta)))
    When lambda_kl == 0.0, returns base_loss untouched for bit-exact FT-2 parity.
    """
    if lambda_kl == 0.0:
        return base_loss
    log_p_theta = F.log_softmax(logits_theta, dim=-1)
    kl_epoch = F.kl_div(log_p_theta, p_zs, reduction="none").sum(dim=-1)
    denom = valid_mask.sum().float().clamp(min=1.0)
    kl_mean = (kl_epoch * valid_mask.float()).sum() / denom
    return base_loss + float(lambda_kl) * kl_mean


def adapt_supervised_ft_kl(
    system: FinalSleepStagingSystem,
    cal_records: List[Dict[str, np.ndarray]],
    lr: float,
    epochs: int,
    scope_regex: str,
    lambda_kl: float,
    seed: int,
    allow_unloaded: bool = False,
) -> nn.Module:
    """
    Runs supervised FT-2-KL adaptation on a fresh pristine clone of system._frozen_state.
    """
    set_deterministic_seed(seed)
    device_str = str(system.device)
    model = restore_pristine_model(system, scope_regex=scope_regex, allow_unloaded=allow_unloaded)

    batch = prepare_calibration_batch(cal_records, context_len=1200, device=device_str)
    # Precompute p_ZS in .eval() with torch.no_grad()
    model.eval()
    with torch.no_grad():
        zs_out = model(
            x_ihr=batch["x_ihr"],
            x_accel=batch["x_acc"],
            x_meta=batch["x_meta"],
            seq_lens=batch["seq_lens"],
        )
        p_zs = F.softmax(zs_out["logits"], dim=-1).detach()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(f"No trainable parameters matched scope_regex: '{scope_regex}'")

    optimizer = torch.optim.Adam(trainable_params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(
            x_ihr=batch["x_ihr"],
            x_accel=batch["x_acc"],
            x_meta=batch["x_meta"],
            targets=batch["y"],
            mask=batch["mask"],
            seq_lens=batch["seq_lens"],
        )
        total_loss = compute_kl_regularized_loss(
            logits_theta=out["logits"],
            p_zs=p_zs,
            base_loss=out["total_loss"],
            valid_mask=batch["mask"],
            lambda_kl=lambda_kl,
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

    model.eval()
    return model


def compute_calibration_min_stage_frac(cal_records: List[Dict[str, np.ndarray]]) -> float:
    """Computes unrounded min_{c in 0..3} P_C(y=c) over calibration nights."""
    all_y = np.concatenate([rec["y"] for rec in cal_records])
    valid_y = all_y[(all_y >= 0) & (all_y < 4)]
    if len(valid_y) == 0:
        return 0.0
    counts = [int(np.sum(valid_y == c)) for c in range(4)]
    return float(min(counts) / float(len(valid_y)))


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Computes Holm-Bonferroni step-down adjusted p-values."""
    m = len(p_values)
    order = np.argsort(p_values)
    adj = np.empty(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = p_values[idx] * (m - rank)
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return [float(x) for x in adj]


def load_subject_records(
    subject_id: str,
    usable_nights: List[str],
    cache_dir: str,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    """Returns (cal_records[:2], eval_records[2:])."""
    recs = []
    for nid in usable_nights:
        r = load_night_features(cache_dir, subject_id, str(nid))
        if r is None:
            raise FileNotFoundError(f"Missing {subject_id}__{nid}.npz in {cache_dir}")
        recs.append(r)
    assert len(recs) >= 3, f"Subject {subject_id} has < 3 usable nights ({len(recs)})"
    return recs[:2], recs[2:]


def run_exp020(
    cache_dir: str = DEFAULT_CACHE_DIR,
    manifest_dir: str = DEFAULT_MANIFEST_DIR,
    frozen_manifest_json: str = DEFAULT_FROZEN_MANIFEST,
    night_inventory_csv: str = DEFAULT_NIGHT_INVENTORY,
    stage_d_json: str = DEFAULT_STAGE_D_JSON,
    exp019_metrics_csv: str = DEFAULT_EXP019_METRICS,
    output_dir: str = os.path.join(REPO_ROOT, "experiments", "EXP-020"),
    device: str = "cuda:0",
    seed_boot: int = 42,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    with open(frozen_manifest_json, "r", encoding="utf-8") as f:
        frozen_manifest = json.load(f)
    sha_map = {(c["fold"], c["seed"]): c["sha256"] for c in frozen_manifest["checkpoints"]}

    with open(stage_d_json, "r", encoding="utf-8") as f:
        stage_d = json.load(f)

    inv_df = pd.read_csv(night_inventory_csv)
    inv_df = inv_df[inv_df["usable"].astype(bool)].copy()
    inv_df["night_int"] = inv_df["night"].astype(int)
    inv_df = inv_df.sort_values(["subject_id", "night_int"])
    usable_by_sub = {
        str(s): [str(n) for n in g["night_int"].tolist()]
        for s, g in inv_df.groupby("subject_id", sort=True)
    }

    cfg = FinalModelConfig.from_json()
    system = FinalSleepStagingSystem(config=cfg, device=device)

    inner_val_selection: Dict[str, Any] = {"folds": {}}
    outer_rows: List[Dict[str, Any]] = []

    for fold in FOLDS:
        with open(os.path.join(manifest_dir, f"fold_{fold}.json"), "r", encoding="utf-8") as f:
            f_manifest = json.load(f)
        inner_subs = [str(s) for s in f_manifest["inner_val_subjects"] if len(usable_by_sub.get(str(s), [])) >= 3]
        outer_subs = [str(s) for s in f_manifest["outer_test_subjects"] if len(usable_by_sub.get(str(s), [])) >= 3]

        fold_cfg = stage_d["folds"][str(fold)]["FT-2"]
        lr = float(fold_cfg["lr"])
        epochs = int(fold_cfg["epochs"])
        scope = str(fold_cfg["scope"])
        scope_regex = SCOPE_REGEX_MAP.get(scope, scope)

        # 1. Inner-val evaluation across 3 seeds (STRICTLY NO OUTER-TEST ACCESS)
        inner_scores: Dict[str, Dict[Any, List[float]]] = {
            "ZS": {s: [] for s in inner_subs},
            "FT-2": {s: [] for s in inner_subs},
            "FT-2-KL": {(lam, s): [] for lam in LAMBDA_KL_GRID for s in inner_subs},
        }
        inner_min_frac = {
            s: compute_calibration_min_stage_frac(load_subject_records(s, usable_by_sub[s], cache_dir)[0])
            for s in inner_subs
        }

        for seed in SEEDS:
            system.load_frozen_checkpoint(fold=fold, seed=seed, expected_sha256=sha_map[(fold, seed)])
            for sub in inner_subs:
                cal_recs, eval_recs = load_subject_records(sub, usable_by_sub[sub], cache_dir)
                zs_model = restore_pristine_model(system, scope_regex=r".*")
                zs_f1 = evaluate_records_macro_f1(system.make_predictor(zs_model), eval_recs)
                inner_scores["ZS"][sub].append(zs_f1)

                ft2_model = adapt_supervised_ft_kl(system, cal_recs, lr, epochs, scope_regex, lambda_kl=0.0, seed=seed)
                ft2_f1 = evaluate_records_macro_f1(system.make_predictor(ft2_model), eval_recs)
                inner_scores["FT-2"][sub].append(ft2_f1)

                for lam in LAMBDA_KL_GRID:
                    kl_model = adapt_supervised_ft_kl(system, cal_recs, lr, epochs, scope_regex, lambda_kl=lam, seed=seed)
                    kl_f1 = evaluate_records_macro_f1(system.make_predictor(kl_model), eval_recs)
                    inner_scores["FT-2-KL"][(lam, sub)].append(kl_f1)

        inner_zs_mean = {s: float(np.mean(inner_scores["ZS"][s])) for s in inner_subs}
        inner_ft2_mean = {s: float(np.mean(inner_scores["FT-2"][s])) for s in inner_subs}

        # Select lambda_KL* (4-decimal round, tie-breaker smallest lambda_KL)
        lam_candidates = []
        for lam in LAMBDA_KL_GRID:
            mean_f1 = float(np.mean([np.mean(inner_scores["FT-2-KL"][(lam, s)]) for s in inner_subs]))
            lam_candidates.append((round(mean_f1, 4), lam, mean_f1))
        lam_candidates.sort(key=lambda x: (-x[0], x[1]))
        best_lam = float(lam_candidates[0][1])
        lam_tie = len([x for x in lam_candidates if x[0] == lam_candidates[0][0]]) > 1

        # Select tau* for GATE_tau (4-decimal round, tie-breaker smallest tau)
        tau_candidates = []
        for tau in TAU_GRID:
            gated_f1s = [
                inner_zs_mean[s] if inner_min_frac[s] < tau else inner_ft2_mean[s]
                for s in inner_subs
            ]
            mean_f1 = float(np.mean(gated_f1s))
            tau_candidates.append((round(mean_f1, 4), tau, mean_f1))
        tau_candidates.sort(key=lambda x: (-x[0], x[1]))
        best_tau = float(tau_candidates[0][1])
        tau_tie = len([x for x in tau_candidates if x[0] == tau_candidates[0][0]]) > 1

        # Arm (c) per-fold nested gate
        inner_kl_best = {s: float(np.mean(inner_scores["FT-2-KL"][(best_lam, s)])) for s in inner_subs}
        delta_kl_vs_zs_inner = float(np.mean([inner_kl_best[s] - inner_zs_mean[s] for s in inner_subs]))
        comb_candidates = []
        for tau in TAU_GRID:
            n_abst = sum(1 for s in inner_subs if inner_min_frac[s] < tau)
            comb_f1s = [
                inner_zs_mean[s] if inner_min_frac[s] < tau else inner_kl_best[s]
                for s in inner_subs
            ]
            mean_f1 = float(np.mean(comb_f1s))
            comb_candidates.append((round(mean_f1, 4), tau, mean_f1, n_abst))
        comb_candidates.sort(key=lambda x: (-x[0], x[1]))
        best_tau_comb = float(comb_candidates[0][1])
        n_abst_comb = int(comb_candidates[0][3])
        comb_inner_mean = float(comb_candidates[0][2])
        kl_inner_mean = float(np.mean(list(inner_kl_best.values())))
        arm_c_unlocked = (
            (delta_kl_vs_zs_inner >= 0.0)
            and (n_abst_comb >= 1)
            and (comb_inner_mean > kl_inner_mean)
        )
        arm_c_status = "UNLOCKED" if arm_c_unlocked else f"FOLD_{fold}_COMBINED_SKIPPED_BY_INNER_VAL_GATE"

        inner_val_selection["folds"][str(fold)] = {
            "lambda_kl_star": best_lam,
            "lambda_tie_breaker_used": lam_tie,
            "tau_star": best_tau,
            "tau_tie_breaker_used": tau_tie,
            "tau_comb_star": best_tau_comb,
            "arm_c_gate_status": arm_c_status,
            "delta_kl_vs_zs_inner": delta_kl_vs_zs_inner,
        }

        # 2. Outer-test evaluation for fold (using selected lambda_kl_star, tau_star, tau_comb_star)
        outer_scores: Dict[str, Dict[str, List[float]]] = {
            "ZS": {s: [] for s in outer_subs},
            "FT-2": {s: [] for s in outer_subs},
            "FT-2-KL": {s: [] for s in outer_subs},
        }
        outer_min_frac = {
            s: compute_calibration_min_stage_frac(load_subject_records(s, usable_by_sub[s], cache_dir)[0])
            for s in outer_subs
        }
        for seed in SEEDS:
            system.load_frozen_checkpoint(fold=fold, seed=seed, expected_sha256=sha_map[(fold, seed)])
            for sub in outer_subs:
                cal_recs, eval_recs = load_subject_records(sub, usable_by_sub[sub], cache_dir)
                zs_m = restore_pristine_model(system, scope_regex=r".*")
                outer_scores["ZS"][sub].append(evaluate_records_macro_f1(system.make_predictor(zs_m), eval_recs))

                ft2_m = adapt_supervised_ft_kl(system, cal_recs, lr, epochs, scope_regex, lambda_kl=0.0, seed=seed)
                outer_scores["FT-2"][sub].append(evaluate_records_macro_f1(system.make_predictor(ft2_m), eval_recs))

                kl_m = adapt_supervised_ft_kl(system, cal_recs, lr, epochs, scope_regex, lambda_kl=best_lam, seed=seed)
                outer_scores["FT-2-KL"][sub].append(evaluate_records_macro_f1(system.make_predictor(kl_m), eval_recs))

        for sub in outer_subs:
            zs_v = float(np.mean(outer_scores["ZS"][sub]))
            ft2_v = float(np.mean(outer_scores["FT-2"][sub]))
            kl_v = float(np.mean(outer_scores["FT-2-KL"][sub]))
            mf = outer_min_frac[sub]
            gate_v = zs_v if mf < best_tau else ft2_v
            comb_v = (zs_v if mf < best_tau_comb else kl_v) if arm_c_unlocked else kl_v
            row = {
                "subject_id": sub,
                "fold": fold,
                "min_stage_frac_cal": mf,
                "ZS": zs_v,
                "FT-2": ft2_v,
                "FT-2-KL": kl_v,
                "GATE_tau": gate_v,
                "GATE_abstained": bool(mf < best_tau),
                "FT-2-KL_GATE_tau": comb_v,
                "arm_c_unlocked": arm_c_unlocked,
                "lambda_kl_star": best_lam,
                "tau_star": best_tau,
            }
            for t_fixed in TAU_GRID:
                row[f"GATE_fixed_{t_fixed}"] = zs_v if mf < t_fixed else ft2_v
            outer_rows.append(row)

    outer_df = pd.DataFrame(outer_rows).sort_values("subject_id").reset_index(drop=True)
    outer_csv = os.path.join(output_dir, "exp020_subject_metrics.csv")
    outer_df.to_csv(outer_csv, index=False)

    sel_json = os.path.join(output_dir, "exp020_inner_val_selection.json")
    with open(sel_json, "w", encoding="utf-8") as f:
        json.dump(inner_val_selection, f, indent=2)

    # Summary statistics & Holm-Bonferroni across primary arms
    arms = ["FT-2-KL", "GATE_tau", "FT-2-KL_GATE_tau"]
    raw_pvals = []
    arm_summaries: Dict[str, Any] = {}
    zs_arr = outer_df["ZS"].values * 100.0
    ft2_arr = outer_df["FT-2"].values * 100.0

    for arm in arms:
        arm_arr = outer_df[arm].values * 100.0
        d_zs = arm_arr - zs_arr
        d_ft2 = arm_arr - ft2_arr
        ci_zs = participant_bootstrap_ci(d_zs, n_boot=10000, seed=seed_boot)
        ci_ft2 = participant_bootstrap_ci(d_ft2, n_boot=10000, seed=seed_boot)
        _, p_wil = stats.wilcoxon(d_zs) if np.any(d_zs != 0) else (0.0, 1.0)
        raw_pvals.append(float(p_wil))
        arm_summaries[arm] = {
            "mean_macro_f1": float(np.mean(outer_df[arm].values)),
            "delta_vs_zs_pp": float(np.mean(d_zs)),
            "delta_vs_zs_ci95_pp": [ci_zs[0], ci_zs[1]],
            "wilcoxon_p_vs_zs_raw": float(p_wil),
            "delta_vs_ft2_pp": float(np.mean(d_ft2)),
            "delta_vs_ft2_ci95_pp": [ci_ft2[0], ci_ft2[1]],
            "tail_p5_pp": float(np.percentile(d_zs, 5.0)),
            "tail_p10_pp": float(np.percentile(d_zs, 10.0)),
            "tail_min_pp": float(np.min(d_zs)),
            "n_harm_le_neg5pp": int(np.sum(d_zs <= -5.0)),
        }

    holm_p = holm_bonferroni(raw_pvals)
    for arm, hp in zip(arms, holm_p):
        arm_summaries[arm]["wilcoxon_p_vs_zs_holm"] = hp

    b34 = outer_df[outer_df["subject_id"] == "Bidslab34"].iloc[0].to_dict()
    summary = {
        "classification": "DESIGN_INFORMED_EXPLORATORY_CLOSURE",
        "n_outer_subjects": len(outer_df),
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": device,
            "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() and "cuda" in device else "cpu",
        },
        "arms": arm_summaries,
        "bidslab34_audit": b34,
    }
    sum_json = os.path.join(output_dir, "exp020_summary_metrics.json")
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_md = os.path.join(output_dir, "EXP020_CLOSURE_REPORT.md")
    lines = [
        "# EXP-020 Regularized & Gated Adaptation Closure Report (Exploratory)",
        "",
        "> **Prior-Access Notice:** Design-informed exploratory closure on BIDSleep; no confirmatory claims.",
        "",
        "| Arm | Mean Macro-F1 | Δ vs ZS (pp) | 95% CI (pp) | Holm p | Δ vs FT-2 (pp) | N_harm (≤ -5pp) | Min Δ (pp) |",
        "|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for arm in arms:
        s = arm_summaries[arm]
        lines.append(
            f"| `{arm}` | {s['mean_macro_f1']:.4f} | {s['delta_vs_zs_pp']:+.2f} | "
            f"[{s['delta_vs_zs_ci95_pp'][0]:+.2f}, {s['delta_vs_zs_ci95_pp'][1]:+.2f}] | "
            f"{s['wilcoxon_p_vs_zs_holm']:.4f} | {s['delta_vs_ft2_pp']:+.2f} | "
            f"{s['n_harm_le_neg5pp']} | {s['tail_min_pp']:+.2f} |"
        )
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {"subject_csv": outer_csv, "selection_json": sel_json, "summary_json": sum_json, "report": report_md}


def main():
    parser = argparse.ArgumentParser(description="EXP-020 Regularized & Gated Adaptation Closure")
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "experiments", "EXP-020"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    out = run_exp020(cache_dir=args.cache_dir, output_dir=args.output_dir, device=args.device)
    print(f"EXP-020 complete: {out['report']}")


if __name__ == "__main__":
    main()
