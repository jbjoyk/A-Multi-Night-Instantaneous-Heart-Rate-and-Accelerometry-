"""
regenerate_modelA_checkpoints.py: Regenerate and verify 15 EXP-018 Phase 2 Model A (L=1200) checkpoints.

Protocol Governance:
  - Model A: ConventionalCNNBiGRU (163,940 parameters).
  - L=1200, B=2, N_accum=1 (tokens_per_step=2400, 3200 steps).
  - Seeds: 42, 123, 2024.
  - Folds: 1, 2, 3, 4, 5 (from experiments/EXP-014/EXP014_MANIFESTS/fold_{fold}.json).
  - Train set: inner_train_subjects.
  - Val set: inner_val_subjects.
  - Test set: outer_test_subjects (for historical verification only).
  - Parity assertion: Test Macro-F1 matches historical JSON within +-0.5 pp.
  - Serialization: experiments/EXP-018/checkpoints/modelA_L1200_fold{fold}_seed{seed}.pt
"""

import os
import sys

# Ensure repository root and EXP-018 directory are in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
exp018_dir = os.path.abspath(os.path.dirname(__file__))
for path_to_add in [repo_root, exp018_dir]:
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)

import json
import time
import argparse
from typing import Dict, List, Any
import numpy as np
import torch

from dataset_gate import DatasetGate
from exp013_models import ConventionalCNNBiGRU
from exp018_dataset import (
    COMPUTE_SCHEDULE_LOOKUP,
    RandomCropSleepDataset,
    create_infinite_batch_stream
)
from exp018_inference import SlidingWindowPredictor
from slamss_ifs_train import prepare_dataset_records

# Reuse evaluate_records from run_exp018_execution
from run_exp018_execution import evaluate_records


def train_and_save_checkpoint(
    fold: int,
    seed: int,
    train_records: List[Dict[str, np.ndarray]],
    val_records: List[Dict[str, np.ndarray]],
    test_records: List[Dict[str, np.ndarray]],
    subject_to_idx: Dict[str, int],
    checkpoint_dir: str,
    historical_json_path: str,
    device: str = "cuda:0",
    max_steps: int = 3200,
    tolerance: float = 0.005  # 0.5 pp
) -> Dict[str, Any]:
    checkpoint_path = os.path.join(checkpoint_dir, f"modelA_L1200_fold{fold}_seed{seed}.pt")
    context_len = 1200
    sched = COMPUTE_SCHEDULE_LOOKUP[context_len]
    B = sched["B"]
    N_accum = sched["N_accum"]

    # Load historical JSON for assertion
    with open(historical_json_path, "r") as f:
        hist_data = json.load(f)
    hist_test_f1 = hist_data["test_metrics"]["macro_f1"]

    if os.path.exists(checkpoint_path):
        print(f"[CHECKPOINT EXISTS] {checkpoint_path}. Verifying weights and parity...")
        model = ConventionalCNNBiGRU(
            d_model=64,
            d_temp=96,
            n_classes=4,
            n_meta=5,
            dropout=0.2,
            use_rcc_loss=True
        ).to(device)
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        predictor = SlidingWindowPredictor(
            model=model,
            context_len=context_len,
            neutralize_circadian=True,
            device=torch.device(device)
        )
        eval_metrics = evaluate_records(predictor, test_records)
        eval_f1 = eval_metrics["macro_f1"]
        diff = abs(eval_f1 - hist_test_f1)
        print(f"  Verified Checkpoint Test Macro-F1: {eval_f1:.4f} vs Historical: {hist_test_f1:.4f} (diff: {diff:.5f})")
        assert diff <= tolerance, f"Checkpoint verification failed: diff {diff} > {tolerance}"
        return {
            "fold": fold,
            "seed": seed,
            "status": "CACHED_AND_VERIFIED",
            "test_macro_f1": eval_f1,
            "hist_test_macro_f1": hist_test_f1,
            "checkpoint_path": checkpoint_path
        }

    # Deterministic seeding
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = ConventionalCNNBiGRU(
        d_model=64,
        d_temp=96,
        n_classes=4,
        n_meta=5,
        dropout=0.2,
        use_rcc_loss=True
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 163940, f"Model A parameter count mismatch: {n_params} != 163,940"

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.0,
        betas=(0.9, 0.999)
    )

    train_dataset = RandomCropSleepDataset(
        records=train_records,
        subject_to_idx=subject_to_idx,
        context_len=context_len,
        neutralize_circadian=True,
        seed=seed
    )
    batch_stream = create_infinite_batch_stream(train_dataset, batch_size=B, seed=seed)

    model.train()
    optimizer_steps = 0
    micro_step_count = 0
    running_loss = 0.0
    t0 = time.time()
    optimizer.zero_grad()

    while optimizer_steps < max_steps:
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

            if optimizer_steps % 800 == 0 or optimizer_steps == max_steps:
                print(f"  Step {optimizer_steps}/{max_steps} | Loss: {running_loss / float(N_accum):.4f}")
            running_loss = 0.0

    train_time = time.time() - t0
    print(f"  Training completed in {train_time:.1f}s. Evaluating against outer test set...")

    # Evaluation
    predictor = SlidingWindowPredictor(
        model=model,
        context_len=context_len,
        neutralize_circadian=True,
        device=torch.device(device)
    )
    test_metrics = evaluate_records(predictor, test_records)
    eval_f1 = test_metrics["macro_f1"]
    diff = abs(eval_f1 - hist_test_f1)

    print(f"  Result Test Macro-F1: {eval_f1:.4f} | Historical: {hist_test_f1:.4f} | Diff: {diff:.5f}")
    assert diff <= tolerance, f"Parity assertion failed for Fold {fold} Seed {seed}: |{eval_f1:.4f} - {hist_test_f1:.4f}| = {diff:.5f} > {tolerance}"

    # Save checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"  [SAVED] Checkpoint saved successfully to {checkpoint_path}")

    return {
        "fold": fold,
        "seed": seed,
        "status": "REGENERATED_AND_VERIFIED",
        "test_macro_f1": eval_f1,
        "hist_test_macro_f1": hist_test_f1,
        "parity_diff": diff,
        "train_time_sec": train_time,
        "checkpoint_path": checkpoint_path
    }


def main():
    parser = argparse.ArgumentParser(description="Regenerate EXP-018 Model A L=1200 Checkpoints")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Folds to regenerate")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024], help="Random seeds")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--cache-dir", type=str,
                        default="/home/rj2826/ai-research/sleep/sleepchannelnet_v8_preprocessing_audit_work_REAL/cache/prep_57738d534b",
                        help="Cache directory")
    parser.add_argument("--manifest-dir", type=str, default="experiments/EXP-014/EXP014_MANIFESTS", help="Manifest directory")
    parser.add_argument("--output-dir", type=str, default="experiments/EXP-018", help="EXP-018 directory")
    args = parser.parse_args()

    checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    phase2_dir = os.path.join(args.output_dir, "phase2_results")

    # Load gate and records
    gate = DatasetGate(cache_dir=args.cache_dir)
    ds_status = gate.verify_dataset_structure()
    all_subs = ds_status["subjects"]
    records_dict = prepare_dataset_records(args.cache_dir, {"train": all_subs, "val": [], "test": []})
    all_nights = records_dict["train"]
    records_by_sub = {s: [] for s in all_subs}
    for r in all_nights:
        records_by_sub[r["subject"]].append(r)
    sub_to_idx = {s: i for i, s in enumerate(all_subs)}

    results = []
    for fold in args.folds:
        manifest_path = os.path.join(args.manifest_dir, f"fold_{fold}.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        inner_train_subs = manifest["inner_train_subjects"]
        inner_val_subs = manifest["inner_val_subjects"]
        outer_test_subs = manifest["outer_test_subjects"]

        train_records = [r for s in inner_train_subs for r in records_by_sub[s]]
        val_records = [r for s in inner_val_subs for r in records_by_sub[s]]
        test_records = [r for s in outer_test_subs for r in records_by_sub[s]]

        for seed in args.seeds:
            hist_json = os.path.join(phase2_dir, f"run_L1200_fold{fold}_seed{seed}.json")
            assert os.path.exists(hist_json), f"Historical JSON missing: {hist_json}"
            print(f"\n=======================================================")
            print(f"Processing Fold {fold}/5, Seed {seed} on {args.device}")
            print(f"=======================================================")
            res = train_and_save_checkpoint(
                fold=fold,
                seed=seed,
                train_records=train_records,
                val_records=val_records,
                test_records=test_records,
                subject_to_idx=sub_to_idx,
                checkpoint_dir=checkpoint_dir,
                historical_json_path=hist_json,
                device=args.device
            )
            results.append(res)

    print("\n" + "=" * 70)
    print("REGENERATION RUN SUMMARY")
    print("=" * 70)
    for r in results:
        print(f"Fold {r['fold']} Seed {r['seed']}: {r['status']} | Macro-F1: {r['test_macro_f1']:.4f} (Hist: {r['hist_test_macro_f1']:.4f})")


if __name__ == "__main__":
    main()
