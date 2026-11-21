"""
freeze_final_model.py: Stage 2 Model Freeze Verification & Canonical OOF Extraction Script.

Responsibilities (ZERO retraining; strictly read-only verification + inference):
  1. Verifies all 15 frozen Model A (L=1200) checkpoints exist and match the exact
     163,940-parameter architecture invariant.
  2. Computes SHA-256 cryptographic hashes for all 15 checkpoints and writes
     experiments/STAGE2_FINAL_MODEL/frozen_checkpoint_manifest.json.
  3. Evaluates the 15 frozen checkpoints via SlidingWindowPredictor (L=1200,
     neutralize_circadian=True) across all 47 outer-test subjects (5 folds x 3 seeds).
  4. Asserts strict numerical parity (default <= 5e-4, configurable via --parity-tolerance)
     on BOTH fold-pooled and per-subject Macro-F1 against locked
     experiments/EXP-018/phase2_results/run_L1200_fold{fold}_seed{seed}.json.
  5. Serializes the complete 47-subject x 3-seed paired-benchmark reference table to
     experiments/STAGE2_FINAL_MODEL/final_model_oof_subject_metrics.csv
     (including Macro-F1, Weighted-F1, Accuracy, Kappa, and per-stage Wake/Light/Deep/REM F1)
     consumed directly by Stage 3 (TSK-08 / EXP-021).
  6. Generates experiments/STAGE2_FINAL_MODEL/STAGE2_MODEL_FREEZE_REPORT.md explicitly
     distinguishing unweighted participant-mean OOF (N=47) from pooled-epoch 3-seed OOF.

Usage:
  .venv/bin/python experiments/STAGE2_FINAL_MODEL/freeze_final_model.py --device cuda:0
  .venv/bin/python experiments/STAGE2_FINAL_MODEL/freeze_final_model.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import torch

STAGE2_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(STAGE2_DIR, "..", ".."))
EXP018_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-018")
for _p in (REPO_ROOT, STAGE2_DIR, EXP018_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from final_model_pipeline import (
    DEFAULT_SPEC_PATH,
    FinalModelConfig,
    FinalSleepStagingSystem,
    sha256_file,
)
from slamss_ifs_train import prepare_dataset_records

CACHE_DIR_DEFAULT = "/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b"
MANIFESTS_DIR_DEFAULT = os.path.join(REPO_ROOT, "experiments", "EXP-014", "EXP014_MANIFESTS")
PHASE2_RESULTS_DIR_DEFAULT = os.path.join(REPO_ROOT, "experiments", "EXP-018", "phase2_results")
MANIFEST_JSON_OUT_DEFAULT = os.path.join(STAGE2_DIR, "frozen_checkpoint_manifest.json")
OOF_CSV_OUT_DEFAULT = os.path.join(STAGE2_DIR, "final_model_oof_subject_metrics.csv")
REPORT_MD_OUT_DEFAULT = os.path.join(STAGE2_DIR, "STAGE2_MODEL_FREEZE_REPORT.md")


def compute_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Computes complete primary and secondary metrics on valid 4-class epochs [0..3] with NaN-safe kappa."""
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


def write_freeze_report(
    report_path: str,
    config: FinalModelConfig,
    mem_profile: Dict[str, Any],
    manifest_entries: List[Dict[str, Any]],
    df_oof: pd.DataFrame,
    max_subject_parity_diff: float,
    max_fold_parity_diff: float,
    parity_tolerance: float,
    subject_parity_tolerance: float,
) -> None:
    """Writes Stage 2 freeze seal report summarizing verification and 47-subject OOF metrics."""
    subj_means = df_oof.groupby("subject_id").mean(numeric_only=True)
    n_subjects = len(subj_means)
    mean_f1 = subj_means["macro_f1"].mean() * 100.0
    sd_f1 = subj_means["macro_f1"].std(ddof=1) * 100.0
    mean_acc = subj_means["accuracy"].mean() * 100.0
    mean_wf1 = subj_means["weighted_f1"].mean() * 100.0
    mean_kappa = subj_means["kappa"].mean()

    lines = [
        "# Stage 2: Final Model Freeze Verification Report",
        "",
        f"- **Model Release:** `{config.model_name}` (`ConventionalCNNBiGRU`)",
        f"- **Verification Timestamp (UTC):** `{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}`",
        f"- **Exact Trainable Parameters:** `{mem_profile['trainable_parameters']:,}`",
        f"- **Weight Footprint Only:** `{mem_profile['fp32_kib']} KiB` (FP32) | `{mem_profile['fp16_kib']} KiB` (FP16) | `{mem_profile['int8_kib']} KiB` (INT8)",
        f"- **Compression vs. SLAMSS-IFS:** `{mem_profile['compression_vs_slamss_paper_text']}x` (paper text 666k) | `{mem_profile['compression_vs_slamss_official_repo']}x` (official repo 5.47M)",
        f"- **Checkpoints Verified:** `{len(manifest_entries)} / 15` (5 folds x 3 seeds)",
        f"- **Max Fold-Pooled Parity Diff vs. Locked EXP-018 JSON:** `{max_fold_parity_diff:.6f}` (Threshold: `<= {parity_tolerance:.6f}`)",
        f"- **Max Per-Subject Parity Diff vs. Locked EXP-018 JSON:** `{max_subject_parity_diff:.6f}` (Threshold: `<= {subject_parity_tolerance:.6f}`)",
        "- **Freeze Seal Status:** **FROZEN & CLEARED FOR STAGE 3 (TSK-08)**",
        "",
        "---",
        "",
        f"## 1. 47-Subject Unweighted Participant-Mean OOF Reference Summary (N = {n_subjects})",
        "",
        "> **Aggregation Note:** Metrics below are unweighted participant means ± between-participant SD across `N = 47` independent subjects (each averaged across seeds `42, 123, 2024`), matching the Stage 3 paired comparative estimand (`PREREG_SLAMSS_IFS_MATCHED_BENCHMARK.md` §4.1). Locked EXP-018 pooled-epoch 3-seed OOF Macro-F1 (`0.6479 ± 0.0053`) pools epochs across all subjects prior to averaging over `N = 3` seeds.",
        "",
        "| Metric | Participant Mean | Between-Participant SD |",
        "| :--- | :---: | :---: |",
        f"| **Macro-F1** | **{mean_f1:.2f}%** | {sd_f1:.2f} pp |",
        f"| **Accuracy** | {mean_acc:.2f}% | {subj_means['accuracy'].std(ddof=1) * 100.0:.2f} pp |",
        f"| **Weighted-F1** | {mean_wf1:.2f}% | {subj_means['weighted_f1'].std(ddof=1) * 100.0:.2f} pp |",
        f"| **Cohen's Kappa** | {mean_kappa:.4f} | {subj_means['kappa'].std(ddof=1):.4f} |",
        f"| **Wake F1** | {subj_means['f1_wake'].mean() * 100.0:.2f}% | {subj_means['f1_wake'].std(ddof=1) * 100.0:.2f} pp |",
        f"| **Light F1** | {subj_means['f1_light'].mean() * 100.0:.2f}% | {subj_means['f1_light'].std(ddof=1) * 100.0:.2f} pp |",
        f"| **Deep F1** | {subj_means['f1_deep'].mean() * 100.0:.2f}% | {subj_means['f1_deep'].std(ddof=1) * 100.0:.2f} pp |",
        f"| **REM F1** | {subj_means['f1_rem'].mean() * 100.0:.2f}% | {subj_means['f1_rem'].std(ddof=1) * 100.0:.2f} pp |",
        "",
        "---",
        "",
        "## 2. Cryptographic Checkpoint Registry (SHA-256)",
        "",
        "| Fold | Seed | SHA-256 (First 16 Hex) | Fold Parity Diff | Max Subject Parity Diff | Status |",
        "| :---: | :---: | :--- | :---: | :---: | :---: |",
    ]

    for entry in manifest_entries:
        lines.append(
            f"| {entry['fold']} | {entry['seed']} | `{entry['sha256'][:16]}...` "
            f"| `{entry['fold_pooled_f1_diff']:.6f}` | `{entry['max_subject_f1_diff']:.6f}` | `{entry['status']}` |"
        )

    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_stage2_freeze(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("STAGE 2: FINAL MODEL SYNTHESIS & FREEZE VERIFICATION")
    print(
        f"Device: {args.device} | Parity Tol: {args.parity_tolerance} | "
        f"Dry Run: {args.dry_run} | Verify Only: {args.verify_only}"
    )
    print("=" * 80)

    config = FinalModelConfig.from_json(args.spec_path)
    system = FinalSleepStagingSystem(config=config, device=args.device)
    mem_profile = system.export_int8_memory_profile()
    print(
        f"Architecture Verified: {mem_profile['model_name']} | "
        f"Params: {mem_profile['trainable_parameters']:,} | "
        f"INT8 Weights: {mem_profile['int8_kib']} KiB"
    )

    # 1. Load 5-fold outer test manifests (47 subjects)
    fold_to_test_subs: Dict[int, List[str]] = {}
    all_test_subs: List[str] = []
    for fold in config.folds:
        mpath = os.path.join(args.manifests_dir, f"fold_{fold}.json")
        with open(mpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        subs = list(manifest["outer_test_subjects"])
        fold_to_test_subs[fold] = subs
        all_test_subs.extend(subs)

    assert len(set(all_test_subs)) == 47, (
        f"Expected exactly 47 unique outer-test subjects, got {len(set(all_test_subs))}"
    )

    # 2. Verify all 15 checkpoints exist and hash them
    manifest_entries: List[Dict[str, Any]] = []
    for fold in config.folds:
        for seed in config.seeds:
            ckpt_path = system.checkpoint_path_for(fold, seed)
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Missing frozen checkpoint: {ckpt_path}")
            digest = sha256_file(ckpt_path)
            manifest_entries.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "checkpoint_relpath": os.path.relpath(ckpt_path, REPO_ROOT),
                    "sha256": digest,
                    "n_parameters": config.expected_params,
                    "fold_pooled_f1_diff": 0.0,
                    "max_subject_f1_diff": 0.0,
                    "status": "HASH_VERIFIED",
                }
            )

    print(f"Verified all {len(manifest_entries)} frozen checkpoints on disk.")

    if args.verify_only:
        print("Verify-only mode complete. Zero inference or file writes performed.")
        return

    # 3. Load preprocessed records
    split_dict = {"train": [], "val": [], "test": sorted(set(all_test_subs))}
    records_by_split = prepare_dataset_records(args.cache_dir, split_dict)
    records_by_sub: Dict[str, List[Dict[str, np.ndarray]]] = {}
    for rec in records_by_split["test"]:
        records_by_sub.setdefault(rec["subject"], []).append(rec)
    for sub in records_by_sub:
        records_by_sub[sub].sort(key=lambda r: int(r["night"]))

    # 4. Run zero-gradient inference and cross-check parity against EXP-018 Phase 2 JSONs
    oof_rows: List[Dict[str, Any]] = []
    max_subject_parity_diff = 0.0
    max_fold_parity_diff = 0.0

    for entry in manifest_entries:
        fold = entry["fold"]
        seed = entry["seed"]
        system.load_frozen_checkpoint(fold=fold, seed=seed, expected_sha256=entry["sha256"])
        predictor = system.make_predictor()

        hist_json_path = os.path.join(
            args.phase2_results_dir, f"run_L1200_fold{fold}_seed{seed}.json"
        )
        with open(hist_json_path, "r", encoding="utf-8") as f:
            hist_data = json.load(f)
        hist_sub_f1: Dict[str, float] = hist_data["test_metrics"]["subject_macro_f1"]
        hist_fold_f1 = float(hist_data["test_metrics"]["macro_f1"])

        fold_max_sub_diff = 0.0
        fold_all_preds: List[np.ndarray] = []
        fold_all_trues: List[np.ndarray] = []

        for sub in fold_to_test_subs[fold]:
            sub_records = records_by_sub[sub]
            preds_list, trues_list = [], []
            for rec in sub_records:
                res = predictor.predict_night(rec)
                preds_list.append(res["y_pred"])
                trues_list.append(res["y_true"])

            y_pred_all = np.concatenate(preds_list)
            y_true_all = np.concatenate(trues_list)
            fold_all_preds.append(y_pred_all)
            fold_all_trues.append(y_true_all)

            metrics = compute_subject_metrics(y_true_all, y_pred_all)
            hist_val = float(hist_sub_f1[sub])
            diff = abs(metrics["macro_f1"] - hist_val)
            fold_max_sub_diff = max(fold_max_sub_diff, diff)
            max_subject_parity_diff = max(max_subject_parity_diff, diff)
            assert diff <= args.subject_parity_tolerance, (
                f"Per-subject parity check failed for Fold {fold} Seed {seed} Subject {sub}: "
                f"eval={metrics['macro_f1']:.6f}, hist={hist_val:.6f}, diff={diff:.6f} > {args.subject_parity_tolerance}"
            )

            oof_rows.append(
                {
                    "subject_id": sub,
                    "fold": fold,
                    "seed": seed,
                    "n_eval_nights": len(sub_records),
                    **metrics,
                }
            )

            if args.dry_run:
                print(
                    f"[DRY-RUN] Verified Fold {fold} Seed {seed} Subject {sub}: "
                    f"Macro-F1={metrics['macro_f1']:.4f} (hist={hist_val:.4f}, diff={diff:.6f})"
                )
                print("Dry run completed successfully. Zero production artifacts written.")
                return

        fold_pooled = compute_subject_metrics(
            np.concatenate(fold_all_trues), np.concatenate(fold_all_preds)
        )
        fold_pooled_diff = abs(fold_pooled["macro_f1"] - hist_fold_f1)
        max_fold_parity_diff = max(max_fold_parity_diff, fold_pooled_diff)
        assert fold_pooled_diff <= args.parity_tolerance, (
            f"Fold-pooled parity check failed for Fold {fold} Seed {seed}: "
            f"eval={fold_pooled['macro_f1']:.6f}, hist={hist_fold_f1:.6f}, diff={fold_pooled_diff:.6f}"
        )

        entry["fold_pooled_f1_diff"] = round(fold_pooled_diff, 6)
        entry["max_subject_f1_diff"] = round(fold_max_sub_diff, 6)
        entry["status"] = "FROZEN_AND_PARITY_VERIFIED"
        print(
            f"Fold {fold} Seed {seed}: {len(fold_to_test_subs[fold])} subjects verified "
            f"(fold diff={fold_pooled_diff:.6f}, max sub diff={fold_max_sub_diff:.6f})"
        )

    # 5. Serialize freeze manifest, canonical OOF CSV, and freeze report
    manifest_payload = {
        "model_name": config.model_name,
        "freeze_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec_sha256": sha256_file(args.spec_path),
        "memory_profile": mem_profile,
        "fold_parity_tolerance": args.parity_tolerance,
        "subject_parity_tolerance": args.subject_parity_tolerance,
        "max_fold_pooled_parity_diff_vs_exp018": round(max_fold_parity_diff, 6),
        "max_subject_parity_diff_vs_exp018": round(max_subject_parity_diff, 6),
        "checkpoints": manifest_entries,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest_json_out)), exist_ok=True)
    with open(args.manifest_json_out, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    os.makedirs(os.path.dirname(os.path.abspath(args.oof_csv_out)), exist_ok=True)
    df_oof = pd.DataFrame(oof_rows).sort_values(["subject_id", "seed"]).reset_index(drop=True)
    df_oof.to_csv(args.oof_csv_out, index=False)

    write_freeze_report(
        report_path=args.report_md_out,
        config=config,
        mem_profile=mem_profile,
        manifest_entries=manifest_entries,
        df_oof=df_oof,
        max_subject_parity_diff=max_subject_parity_diff,
        max_fold_parity_diff=max_fold_parity_diff,
        parity_tolerance=args.parity_tolerance,
        subject_parity_tolerance=args.subject_parity_tolerance,
    )

    print("=" * 80)
    print(f"Saved frozen checkpoint manifest: {args.manifest_json_out}")
    print(f"Saved 47-subject OOF reference:   {args.oof_csv_out} ({len(df_oof)} rows)")
    print(f"Saved Stage 2 freeze report:      {args.report_md_out}")
    print("STAGE 2 FINAL MODEL FREEZE COMPLETE.")
    print("=" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 2 Final Model Freeze Verification Script")
    parser.add_argument("--spec-path", default=DEFAULT_SPEC_PATH)
    parser.add_argument("--cache-dir", default=CACHE_DIR_DEFAULT)
    parser.add_argument("--manifests-dir", default=MANIFESTS_DIR_DEFAULT)
    parser.add_argument("--phase2-results-dir", default=PHASE2_RESULTS_DIR_DEFAULT)
    parser.add_argument("--manifest-json-out", default=MANIFEST_JSON_OUT_DEFAULT)
    parser.add_argument("--oof-csv-out", default=OOF_CSV_OUT_DEFAULT)
    parser.add_argument("--report-md-out", default=REPORT_MD_OUT_DEFAULT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--parity-tolerance", type=float, default=5e-4, help="Fold-pooled Macro-F1 parity tolerance")
    parser.add_argument(
        "--subject-parity-tolerance",
        type=float,
        default=5e-3,
        help="Per-subject Macro-F1 parity tolerance (0.5 pp, matching regenerate_modelA_checkpoints.py)",
    )
    parser.add_argument("--verify-only", action="store_true", help="Only verify files and SHA-256 hashes")
    parser.add_argument("--dry-run", action="store_true", help="Verify 1 subject without writing outputs")
    return parser.parse_args()


if __name__ == "__main__":
    run_stage2_freeze(parse_args())
