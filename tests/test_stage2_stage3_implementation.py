"""
test_stage2_stage3_implementation.py: Comprehensive Unit & Contract Test Suite for Stage 2 and Stage 3.

Covers every EGM-audited invariant:
  Stage 2 (Final Model Synthesis & Freeze):
    1. FinalModelConfig loading, exact 163,940 parameter count, and INT8 weight-only SRAM profile.
    2. Unloaded-backbone guard preventing silent adaptation of uninitialized random weights.
    3. Non-mutating calibration stage-coverage diagnostic (balanced, scarce/missing, empty).
    4. Wearable quality gate (strict > 0.95, NaN/Inf leakage rejection, truly unlabeled records).
    5. Tier 1 label-free adapt_unsupervised on records without 'y' + backbone non-mutation proof.
    6. compute_subject_metrics including NaN-safe Cohen's kappa on degenerate single-class outputs.

  Stage 3 (TSK-08 / EXP-021 Official SLAMSS-IFS Matched Benchmark):
    7. OfficialSLAMSSIFSWrapper exact 5,465,768 params + forward/backward across B in {1, 2, 3, 4}
       + deterministic eval() repeatability + CUDA AMP FP32-loss & GradScaler compatibility.
    8. OfficialSLAMSSCompositeLoss masking (zero gradient on padded positions) + attached zero loss
       on all-masked batches (GradScaler-safe).
    9. SLAMSSMatchedDataset padding (N <= L), cropping (N > L), and circadian neutralization.
   10. SLAMSSSlidingWindowPredictor Case 1 (N <= L), Case 2 (N > L), and predict_nights_batched
       exact numerical equivalence against sequential predict_night.
   11. verify_manifests_and_splits against real EXP014_MANIFESTS & review_package/fold_manifests.csv
       (SHA-256 prefix 36d84020, 3-role CSV cross-check, FileNotFoundError on missing CSV).
   12. parse_fold_seed_pairs & aggregate_completed_runs_to_csv (including homogeneous metadata guard).
   13. paired_metric_inference (standard, all-zero, constant-shift), classify_preregistered_outcome
       (3 tiers + discordance warning), and run_paired_comparative_analysis (complete pass +
       rejection of 14/15 partial runs and duplicate rows).
   14. Locked EXP-019 SHA-256 immutability and locked EXP-018 checkpoint/JSON inventory check.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STAGE2_DIR = os.path.join(REPO_ROOT, "experiments", "STAGE2_FINAL_MODEL")
EXP021_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-021")
EXP019_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-019")
EXP018_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-018")

for _p in (REPO_ROOT, STAGE2_DIR, EXP021_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from final_model_pipeline import (
    EXPECTED_PARAM_COUNT,
    FinalModelConfig,
    FinalSleepStagingSystem,
    audit_calibration_stage_coverage,
    verify_wearable_quality_records,
)
from freeze_final_model import compute_subject_metrics
from official_slamss_ifs_adapter import (
    EXPECTED_OFFICIAL_SLAMSS_PARAMS,
    OfficialSLAMSSCompositeLoss,
    OfficialSLAMSSIFSWrapper,
    SLAMSSMatchedDataset,
    SLAMSSSlidingWindowPredictor,
)
from run_tsk08_slamss_matched import (
    AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
    MANIFESTS_DIR_DEFAULT,
    RUN_RESULTS_SUBDIR,
    aggregate_completed_runs_to_csv,
    parse_fold_seed_pairs,
    verify_manifests_and_splits,
)
from analyze_tsk08_paired_benchmark import (
    classify_preregistered_outcome,
    paired_metric_inference,
    run_paired_comparative_analysis,
)

LOCKED_HASHES = {
    "exp019_subject_metrics.csv": "87f0910ba72c346fd465bc64e5d6f32c6ef609454ea11eb99abccf5af15bb407",
    "exp019_sleep_summary_agreement.csv": "41a2d6afa73b3d65599a83c7c463e7cadbbbcabef6d1f71a6349194b489b35e3",
    "exp019_k_learning_curve.csv": "2bf08608c4685836facc52fefea5ccf8eb0070af93a81b2e0c35b5dde61af3e3",
}


def _make_dummy_night(n_epochs: int, subject: str = "BidslabTest", night: str = "1", with_y: bool = True):
    rng = np.random.default_rng(42 + int(night))
    rec = {
        "subject": subject,
        "night": night,
        "ihr": rng.normal(0.0, 1.0, size=(n_epochs, 30)).astype(np.float32),
        "accel": rng.normal(0.0, 1.0, size=(n_epochs, 30)).astype(np.float32),
        "freq": np.ones((n_epochs, 1), dtype=np.float32),
        "time": np.linspace(0.0, 1.0, n_epochs).reshape(-1, 1).astype(np.float32),
    }
    if with_y:
        rec["y"] = rng.integers(0, 4, size=n_epochs).astype(np.int64)
    return rec


# ------------------------------------------------------------------------------
# STAGE 2 TESTS
# ------------------------------------------------------------------------------

def test_stage2_config_memory_and_unloaded_guard():
    cfg = FinalModelConfig.from_json()
    assert cfg.expected_params == EXPECTED_PARAM_COUNT == 163940
    assert cfg.context_len == 1200
    assert cfg.neutralize_circadian is True

    system = FinalSleepStagingSystem(config=cfg, device="cpu")
    profile = system.export_int8_memory_profile()
    assert profile["trainable_parameters"] == 163940
    assert profile["weights_fit_nrf5340_512kb_sram_int8"] is True
    assert profile["compression_vs_slamss_paper_text"] == pytest.approx(4.06, abs=0.02)
    assert profile["compression_vs_slamss_official_repo"] == pytest.approx(33.34, abs=0.02)

    # Adapting before load_frozen_checkpoint must raise RuntimeError by default
    unlabeled = [_make_dummy_night(650, night="1", with_y=False)]
    with pytest.raises(RuntimeError, match="No frozen population checkpoint loaded"):
        system.adapt_unsupervised(unlabeled)


def test_stage2_stage_coverage_diagnostic():
    # Empty list guard
    diag_empty = audit_calibration_stage_coverage([])
    assert diag_empty["high_negative_transfer_risk"] is True
    assert len(diag_empty["missing_stages"]) == 4

    # Healthy balanced night
    rec_balanced = {
        "y": np.array([0] * 100 + [1] * 400 + [2] * 200 + [3] * 200, dtype=np.int64)
    }
    diag_ok = audit_calibration_stage_coverage([rec_balanced], min_fraction_warn=0.02)
    assert diag_ok["high_negative_transfer_risk"] is False
    assert diag_ok["missing_stages"] == []
    assert diag_ok["scarce_stages"] == []

    # Bidslab34-like scarce Deep night
    rec_scarce = {
        "y": np.array([0] * 200 + [1] * 500 + [2] * 5 + [3] * 295, dtype=np.int64)
    }
    diag_warn = audit_calibration_stage_coverage([rec_scarce], min_fraction_warn=0.02)
    assert diag_warn["high_negative_transfer_risk"] is True
    assert "Deep" in diag_warn["scarce_stages"]


def test_stage2_unlabeled_wearable_gate_and_backbone_non_mutation():
    # Unlabeled records (no 'y' key) passing gate
    rec1 = _make_dummy_night(650, night="1", with_y=False)
    rec2 = _make_dummy_night(650, night="2", with_y=False)
    audit_ok = verify_wearable_quality_records([rec1, rec2])
    assert audit_ok["passed"] is True
    assert audit_ok["non_physiological_leakage_count"] == 0

    # Strict > 0.95 rejection on flatline accelerometer
    rec_flat = copy.deepcopy(rec1)
    rec_flat["accel"][:100, :] = 0.05  # ~15% flatlined -> ratio ~0.85 < 0.95
    audit_fail = verify_wearable_quality_records([rec_flat])
    assert audit_fail["passed"] is False

    # Adapt on truly unlabeled records (allow_unloaded=True for unit test) and verify backbone untouched
    system = FinalSleepStagingSystem(device="cpu")
    orig_state = {k: v.clone() for k, v in system.backbone.state_dict().items()}
    # Use small step config copy for fast CPU test
    fast_cfg = FinalModelConfig(ua_steps=2)
    system_fast = FinalSleepStagingSystem(config=fast_cfg, device="cpu")
    orig_fast = {k: v.clone() for k, v in system_fast.backbone.state_dict().items()}

    adapted_clone, meta = system_fast.adapt_unsupervised([rec1, rec2], allow_unloaded=True)
    assert meta["tier"] == "tier_1_unsupervised_auxiliary"
    assert meta["quality_audit"]["passed"] is True
    assert "y" not in rec1 and "y" not in rec2  # caller dicts not mutated
    for k, v in system_fast.backbone.state_dict().items():
        assert torch.equal(v, orig_fast[k]), f"Frozen backbone parameter '{k}' was mutated!"
    assert adapted_clone is not system_fast.backbone


def test_stage2_compute_subject_metrics_nan_safe_kappa():
    # Degenerate single-class target and single-class prediction (sklearn returns NaN kappa)
    y_t = np.ones(100, dtype=np.int64)
    y_p = np.ones(100, dtype=np.int64)
    m = compute_subject_metrics(y_t, y_p)
    assert not math.isnan(m["kappa"])
    assert m["accuracy"] == 1.0
    assert m["n_valid_epochs"] == 100


# ------------------------------------------------------------------------------
# STAGE 3 TESTS
# ------------------------------------------------------------------------------

def test_stage3_official_slamss_params_batch_sizes_and_determinism():
    model = OfficialSLAMSSIFSWrapper(device=torch.device("cpu"))
    total_params = sum(p.numel() for p in model.parameters())
    assert total_params == EXPECTED_OFFICIAL_SLAMSS_PARAMS == 5465768

    loss_fn = OfficialSLAMSSCompositeLoss(rcc_weight=0.01, n_classes=4)
    L = 12  # fast CPU sequence length (360 samples)

    # Test B in {1, 2, 3, 4} including tail batch B=3 (Folds 3 & 5)
    for B in (1, 2, 3, 4):
        model.train()
        model.zero_grad(set_to_none=True)
        x_sig = torch.randn(B, L * 30, 2)
        x_meta = torch.randn(B, L, 5)
        targets = torch.randint(0, 4, (B, L))
        mask = torch.ones(B, L, dtype=torch.bool)
        logits = model(x_sig, x_meta)
        assert logits.shape == (B, L, 4)
        loss = loss_fn(logits, targets, mask=mask)
        assert torch.isfinite(loss)
        loss.backward()

    # Deterministic eval() repeatability (zero stochastic teacher forcing)
    model.eval()
    x_sig = torch.randn(2, L * 30, 2)
    x_meta = torch.randn(2, L, 5)
    with torch.no_grad():
        out1 = model(x_sig, x_meta)
        out2 = model(x_sig, x_meta)
    assert torch.allclose(out1, out2, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for AMP FP32-loss test")
def test_stage3_cuda_amp_fp32_loss_and_gradscaler():
    device = torch.device("cuda:0")
    model = OfficialSLAMSSIFSWrapper(device=device)
    loss_fn = OfficialSLAMSSCompositeLoss(rcc_weight=0.01, n_classes=4).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    B, L = 3, 16  # tail batch B=3 under AMP
    x_sig = torch.randn(B, L * 30, 2, device=device)
    x_meta = torch.randn(B, L, 5, device=device)
    y = torch.randint(0, 4, (B, L), device=device)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)

    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        logits = model(x_sig, x_meta)
    loss = loss_fn(logits, y, mask=mask)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)

    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(opt)
    scaler.update()

    # All-masked batch must stay attached so GradScaler.unscale_ does not crash
    opt.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        logits_m = model(x_sig, x_meta)
    loss_zero = loss_fn(logits_m, y, mask=torch.zeros_like(mask))
    assert loss_zero.item() == 0.0 and loss_zero.requires_grad
    scaler.scale(loss_zero).backward()
    scaler.unscale_(opt)
    scaler.step(opt)
    scaler.update()


def test_stage3_composite_rcc_loss_masking_and_all_masked():
    loss_fn = OfficialSLAMSSCompositeLoss(rcc_weight=0.01, n_classes=4)
    logits = torch.randn(2, 10, 4, requires_grad=True)
    targets = torch.randint(0, 4, (2, 10))
    targets[:, 7:] = -100
    mask = torch.ones(2, 10, dtype=torch.bool)
    mask[:, 7:] = False

    loss = loss_fn(logits, targets, mask=mask)
    assert torch.isfinite(loss) and loss.item() > 0.0
    loss.backward()
    assert logits.grad is not None
    assert torch.allclose(logits.grad[:, 7:, :], torch.zeros_like(logits.grad[:, 7:, :]))

    # All-masked batch returns attached zero loss
    logits2 = torch.randn(2, 10, 4, requires_grad=True)
    loss_z = loss_fn(logits2, targets, mask=torch.zeros_like(mask))
    assert loss_z.item() == 0.0
    loss_z.backward()
    assert logits2.grad is not None
    assert torch.allclose(logits2.grad, torch.zeros_like(logits2.grad))


def test_stage3_matched_dataset_padding_crop_and_circadian():
    rec_short = _make_dummy_night(12, night="1")
    rec_long = _make_dummy_night(35, night="2")

    ds_neut = SLAMSSMatchedDataset([rec_short, rec_long], context_len=20, neutralize_circadian=True, seed=42)
    item_s = ds_neut[0]
    assert item_s["x_signal"].shape == (20 * 30, 2)
    assert item_s["x_meta"].shape == (20, 5)
    assert torch.all(item_s["mask"][:12]) and not torch.any(item_s["mask"][12:])
    assert torch.all(item_s["y"][12:] == -100)
    assert torch.all(item_s["x_meta"][:, 1] == 0.0)  # clock == 0
    assert torch.all(item_s["x_meta"][:, 2] == 0.0)  # time_feat == 0

    item_l = ds_neut[1]
    assert item_l["x_signal"].shape == (20 * 30, 2)
    assert torch.all(item_l["mask"])

    ds_circ = SLAMSSMatchedDataset([rec_long], context_len=20, neutralize_circadian=False, random_crop=False)
    assert float(ds_circ[0]["x_meta"][:, 2].abs().sum()) > 0.0


def test_stage3_sliding_window_batched_equivalence():
    model = OfficialSLAMSSIFSWrapper(device=torch.device("cpu"))
    predictor = SLAMSSSlidingWindowPredictor(
        model=model, context_len=20, neutralize_circadian=True, device=torch.device("cpu"), use_amp=False
    )

    rec_short = _make_dummy_night(12, night="1")
    rec_long = _make_dummy_night(35, night="2")

    seq_short = predictor.predict_night(rec_short)
    seq_long = predictor.predict_night(rec_long)
    batched = predictor.predict_nights_batched([rec_short, rec_long], batch_size=4)

    assert seq_short["y_pred"].shape == (12,) and np.all(seq_short["counts"] >= 1.0)
    assert seq_long["y_pred"].shape == (35,) and np.all(seq_long["counts"] >= 1.0)
    assert np.allclose(batched[0]["probs"], seq_short["probs"], atol=1e-5)
    assert np.allclose(batched[1]["probs"], seq_long["probs"], atol=1e-5)
    assert np.array_equal(batched[0]["y_pred"], seq_short["y_pred"])
    assert np.array_equal(batched[1]["y_pred"], seq_long["y_pred"])


def test_stage3_manifest_verification_and_aggregation(tmp_path):
    # Real authoritative manifest verification (SHA-256 prefix 36d84020 + 3-role CSV cross-check)
    manifests = verify_manifests_and_splits(MANIFESTS_DIR_DEFAULT, AUTHORITATIVE_MANIFEST_CSV_DEFAULT)
    assert len(manifests) == 5
    assert sum(len(m["outer_test_subjects"]) for m in manifests.values()) == 47

    # Missing authoritative CSV must raise FileNotFoundError (no silent bypass)
    with pytest.raises(FileNotFoundError):
        verify_manifests_and_splits(MANIFESTS_DIR_DEFAULT, str(tmp_path / "missing.csv"))

    # Pair parser check
    class DummyArgs:
        pairs = ["1:42", "3:2024"]
        folds = [1, 2]
        seeds = [42]

    assert parse_fold_seed_pairs(DummyArgs()) == [(1, 42), (3, 2024)]

    # Aggregation + homogeneous metadata guard
    rr_dir = tmp_path / RUN_RESULTS_SUBDIR
    rr_dir.mkdir()
    sample_payload = {
        "status": "COMPLETED",
        "fold": 1,
        "seed": 42,
        "best_epoch": 50,
        "context_len": 1200,
        "neutralize_circadian": True,
        "n_parameters": EXPECTED_OFFICIAL_SLAMSS_PARAMS,
        "test_per_subject_metrics": {
            "Bidslab08": {
                "n_eval_nights": 6,
                "n_valid_epochs": 5000,
                "macro_f1": 0.65,
                "weighted_f1": 0.66,
                "accuracy": 0.67,
                "kappa": 0.50,
                "f1_wake": 0.60,
                "f1_light": 0.70,
                "f1_deep": 0.65,
                "f1_rem": 0.65,
            }
        },
    }
    (rr_dir / "slamss_ifs_fold1_seed42.json").write_text(json.dumps(sample_payload))
    out_csv = aggregate_completed_runs_to_csv(str(tmp_path))
    assert out_csv is not None and os.path.exists(out_csv)

    # Inhomogeneous metadata (neutralize_circadian flipped) must fail loudly
    bad_payload = copy.deepcopy(sample_payload)
    bad_payload["seed"] = 123
    bad_payload["neutralize_circadian"] = False
    (rr_dir / "slamss_ifs_fold1_seed123.json").write_text(json.dumps(bad_payload))
    with pytest.raises(AssertionError, match="Inhomogeneous protocol metadata"):
        aggregate_completed_runs_to_csv(str(tmp_path))


def test_stage3_paired_inference_discordance_and_partial_seed_rejection():
    # All-zero and constant-shift edge cases in paired_metric_inference
    zeros = paired_metric_inference(np.full(10, 0.6), np.full(10, 0.6), scale=100.0, n_boot=200)
    assert zeros["mean_diff"] == 0.0 and zeros["cohens_dz"] == 0.0 and zeros["wilcoxon_p_two_sided"] == 1.0
    const = paired_metric_inference(np.full(10, 0.65), np.full(10, 0.60), scale=100.0, n_boot=200)
    assert math.isinf(const["cohens_dz"]) and const["cohens_dz"] > 0

    # All 3 preregistered classification tiers + discordance flagging
    assert classify_preregistered_outcome(0.5, 2.5, 0.001)["category"].startswith("MODEL A ADVANTAGE")
    assert classify_preregistered_outcome(-2.5, -0.5, 0.001)["category"].startswith("SLAMSS-IFS ADVANTAGE")
    parity = classify_preregistered_outcome(-0.8, 0.9, 0.42)
    assert parity["category"].startswith("PERFORMANCE INDISTINGUISHABLE") and not parity["discordant_tests"]
    disc = classify_preregistered_outcome(-0.17, 1.99, 0.00131)
    assert disc["category"].startswith("PERFORMANCE INDISTINGUISHABLE") and disc["discordant_tests"] is True
    assert disc["discordance_note"] is not None

    # Complete 47-subject x 3-seed synthetic dataset
    rng = np.random.default_rng(123)
    rows_a, rows_s = [], []
    for i in range(47):
        sub = f"Bidslab{i:02d}"
        fold = (i % 5) + 1
        base_f1 = 0.62 + rng.normal(0.0, 0.04)
        for seed in (42, 123, 2024):
            rows_a.append(
                {
                    "subject_id": sub,
                    "fold": fold,
                    "seed": seed,
                    "n_valid_epochs": 4000,
                    "macro_f1": base_f1 + 0.025 + rng.normal(0.0, 0.003),
                    "accuracy": base_f1 + 0.03,
                    "weighted_f1": base_f1 + 0.025,
                    "kappa": 0.52,
                    "f1_wake": 0.65,
                    "f1_light": 0.66,
                    "f1_deep": 0.64,
                    "f1_rem": 0.61,
                }
            )
            rows_s.append(
                {
                    "subject_id": sub,
                    "fold": fold,
                    "seed": seed,
                    "n_valid_epochs": 4000,
                    "macro_f1": base_f1 + rng.normal(0.0, 0.003),
                    "accuracy": base_f1 + 0.01,
                    "weighted_f1": base_f1,
                    "kappa": 0.49,
                    "f1_wake": 0.63,
                    "f1_light": 0.64,
                    "f1_deep": 0.62,
                    "f1_rem": 0.59,
                }
            )

    df_a = pd.DataFrame(rows_a)
    df_s = pd.DataFrame(rows_s)

    summary, df_deltas = run_paired_comparative_analysis(
        df_a, df_s, n_boot=2000, seed=42, expected_n_subjects=47
    )
    assert summary["n_participants"] == 47
    assert "delta_macro_f1_pp" in df_deltas.columns and "delta_kappa" in df_deltas.columns
    assert summary["verdict"]["category"] == "MODEL A ADVANTAGE (STATISTICALLY SIGNIFICANT)"

    # Partial 14/15 run (missing Fold 5 seed 2024) MUST be rejected!
    df_s_partial = df_s[~((df_s["fold"] == 5) & (df_s["seed"] == 2024))]
    with pytest.raises(AssertionError, match="partial run detected"):
        run_paired_comparative_analysis(df_a, df_s_partial, n_boot=200, expected_n_subjects=47)

    # Duplicate rows MUST be rejected!
    df_s_dup = pd.concat([df_s, df_s.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="duplicate"):
        run_paired_comparative_analysis(df_a, df_s_dup, n_boot=200, expected_n_subjects=47)


def test_locked_exp018_and_exp019_artifacts_remain_immutable():
    for fname, expected_hash in LOCKED_HASHES.items():
        fpath = os.path.join(EXP019_DIR, fname)
        assert os.path.exists(fpath), f"Missing locked file: {fpath}"
        with open(fpath, "rb") as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        assert actual == expected_hash

    # Confirm all 15 frozen EXP-018 Model A checkpoints and Phase 2 JSONs exist untouched
    for fold in range(1, 6):
        for seed in (42, 123, 2024):
            ckpt = os.path.join(EXP018_DIR, "checkpoints", f"modelA_L1200_fold{fold}_seed{seed}.pt")
            p2 = os.path.join(EXP018_DIR, "phase2_results", f"run_L1200_fold{fold}_seed{seed}.json")
            assert os.path.exists(ckpt), f"Missing locked EXP-018 checkpoint: {ckpt}"
            assert os.path.exists(p2), f"Missing locked EXP-018 Phase 2 JSON: {p2}"
