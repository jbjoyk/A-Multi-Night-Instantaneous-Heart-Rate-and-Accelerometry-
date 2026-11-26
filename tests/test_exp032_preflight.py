"""
EXP-032 Preflight Verification & Unit Tests
File: tests/test_exp032_preflight.py

Verifies all locked preflight invariants from docs/designs/PREREG_EXP-032_HIGH_ACCURACY_BENCHMARK.md:
  1. Exact parameter counts (M1=1,402,468; M2/M3=1,402,492 <= 1,500,000) & L=1200 MMACs (4,053.08).
  2. LinearChainCRF4 interior -100 marginalization, trailing padding, extreme logits (+/-50),
     finite FP32 gradients, O(1) scale parity between L_CRF_mean and tau^2 * L_KL_masked,
     and exact Viterbi vs. brute-force sequence score parity on T=6.
  3. CRFLogitStitchingPredictor full-night stitching on T > 1200 (T=1500).
  4. review_package/fold_manifests.csv SHA-256, zero subject overlap, and strict (f, s)
     inner_train-only teacher isolation.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

TESTS_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
EXP032_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-032")
EXP031_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-031")
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
EXP021_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-021")

for _p in (EXP032_DIR, EXP031_DIR, EXP030_DIR, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from eval_exp030_downstream import compute_model_mmacs  # noqa: E402
from exp032_models import (  # noqa: E402
    CRFLogitStitchingPredictor,
    EXPECTED_M1_PARAMS,
    EXPECTED_M2_PARAMS,
    EXPECTED_M3_PARAMS,
    EXPECTED_MMACS_L1200,
    LinearChainCRF4,
    build_exp032_model,
    compute_exp032_loss,
    compute_fold_class_weights,
    compute_masked_kl_fp32,
)
from run_exp031_width_sweep import (  # noqa: E402
    AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
    MANIFESTS_DIR_DEFAULT,
    sha256_file,
    verify_manifests_and_splits,
)
from run_exp032_pipeline import (  # noqa: E402
    EXPECTED_MANIFEST_SHA256,
    TeacherCachedDataset,
    _paired_vs_ref,
    evaluate_exp032_verdicts,
)


def test_exact_param_counts_and_mmacs():
    for arm, exp_p in [("M1", EXPECTED_M1_PARAMS), ("M2", EXPECTED_M2_PARAMS), ("M3", EXPECTED_M3_PARAMS)]:
        model = build_exp032_model(arm=arm, dropout=0.2)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == exp_p, f"{arm}: expected {exp_p}, got {n_params}"
        assert n_params <= 1_500_000, f"{arm} exceeds 1.5M budget: {n_params}"

        mmacs = compute_model_mmacs(model, seq_len=1200, epoch_len=30)
        assert abs(mmacs - EXPECTED_MMACS_L1200) < 0.05, f"{arm}: expected {EXPECTED_MMACS_L1200}, got {mmacs:.2f}"


def test_crf_interior_ignore_trailing_pad_and_extreme_logits():
    torch.manual_seed(42)
    crf = LinearChainCRF4(num_tags=4)
    with torch.no_grad():
        crf.transitions.copy_(torch.randn(4, 4) * 0.5)
        crf.start_transitions.copy_(torch.randn(4) * 0.2)
        crf.end_transitions.copy_(torch.randn(4) * 0.2)

    B, T = 2, 64
    # Extreme logits (+/- 50)
    emissions = (torch.randn(B, T, 4) * 25.0).clamp(-50.0, 50.0).requires_grad_(True)
    tags = torch.randint(0, 4, (B, T))
    seq_mask = torch.ones((B, T), dtype=torch.bool)

    # Inject interior -100 spans inside valid night bounds
    tags[0, 10:18] = -100
    tags[1, 5:8] = -100
    tags[1, 25:30] = -100

    # Trailing padding on sample 1
    seq_mask[1, 48:] = False
    tags[1, 48:] = -100

    cw = torch.tensor([1.2, 0.7, 1.1, 1.0], dtype=torch.float32)  # mean == 1.0
    loss = crf.compute_loss(emissions, tags, seq_mask, class_weights=cw)

    assert torch.isfinite(loss), f"CRF loss non-finite: {loss}"
    assert loss.item() >= 0.0, f"CRF NLL negative: {loss.item()}"
    loss.backward()

    assert emissions.grad is not None and torch.isfinite(emissions.grad).all()
    assert crf.transitions.grad is not None and torch.isfinite(crf.transitions.grad).all()
    assert crf.start_transitions.grad is not None and torch.isfinite(crf.start_transitions.grad).all()
    assert crf.end_transitions.grad is not None and torch.isfinite(crf.end_transitions.grad).all()

    marginals = crf.forward_backward_marginals(emissions.detach(), seq_mask)
    assert torch.isfinite(marginals).all()
    # Valid positions sum to 1.0; padded positions are 0.0
    assert torch.allclose(marginals[0].sum(dim=-1), torch.ones(T), atol=1e-5)
    assert torch.allclose(marginals[1, :48].sum(dim=-1), torch.ones(48), atol=1e-5)
    assert torch.allclose(marginals[1, 48:], torch.zeros((T - 48, 4)), atol=1e-6)


def test_crf_viterbi_exact_vs_brute_force_short_seq():
    torch.manual_seed(123)
    crf = LinearChainCRF4(num_tags=4)
    with torch.no_grad():
        crf.start_transitions.copy_(torch.randn(4))
        crf.end_transitions.copy_(torch.randn(4))
        crf.transitions.copy_(torch.randn(4, 4))

    T = 6
    emissions = torch.randn(1, T, 4)
    seq_mask = torch.ones((1, T), dtype=torch.bool)

    viterbi_path = crf.viterbi_decode(emissions, seq_mask)[0].tolist()

    # Brute-force enumerate all 4^6 = 4096 sequences
    emit_np = emissions[0].numpy()
    start_np = crf.start_transitions.detach().numpy()
    end_np = crf.end_transitions.detach().numpy()
    trans_np = crf.transitions.detach().numpy()

    best_score = -1e18
    best_seq = None
    for seq in itertools.product(range(4), repeat=T):
        score = float(start_np[seq[0]] + emit_np[0, seq[0]])
        for t in range(1, T):
            score += float(trans_np[seq[t - 1], seq[t]] + emit_np[t, seq[t]])
        score += float(end_np[seq[-1]])
        if score > best_score:
            best_score = score
            best_seq = list(seq)

    assert viterbi_path == best_seq, f"Viterbi {viterbi_path} != brute-force {best_seq}"


def test_loss_scale_parity_l_crf_mean_vs_kl():
    torch.manual_seed(7)
    model = build_exp032_model("M3", dropout=0.2)
    B, T = 2, 1200
    z_s = torch.randn(B, T, 4)
    z_t = torch.randn(B, T, 4)
    y = torch.randint(0, 4, (B, T))
    y[0, 100:120] = -100
    mask = torch.ones((B, T), dtype=torch.bool)
    mask[1, 950:] = False
    y[1, 950:] = -100

    cw = torch.tensor([1.15, 0.70, 1.10, 1.05], dtype=torch.float32)
    l_crf = model.crf.compute_loss(z_s, y, mask, class_weights=cw).item()
    tau = 2.0
    l_kl_scaled = (tau ** 2) * compute_masked_kl_fp32(z_s, z_t, y, mask, tau=tau).item()
    l_m3 = compute_exp032_loss("M3", model, z_s, y, mask, cw, teacher_logits=z_t, tau=tau, alpha_kd=0.5).item()

    # Both L_CRF_mean and tau^2 * L_KL_masked must be O(1) (no 1000x sequence-sum mismatch)
    assert 0.1 < l_crf < 10.0, f"L_CRF_mean out of O(1) scale: {l_crf}"
    assert 0.05 < l_kl_scaled < 10.0, f"tau^2 * L_KL_masked out of O(1) scale: {l_kl_scaled}"
    assert 0.1 < l_m3 < 10.0, f"L_M3 out of O(1) scale: {l_m3}"


def test_crf_logit_stitching_long_night():
    torch.manual_seed(0)
    model = build_exp032_model("M2", dropout=0.2).eval()
    predictor = CRFLogitStitchingPredictor(
        model=model, context_len=1200, neutralize_circadian=True, device=torch.device("cpu")
    )
    N = 1500
    rec = {
        "ihr": np.random.randn(N, 30).astype(np.float32),
        "accel": np.random.randn(N, 30).astype(np.float32),
        "freq": np.ones((N, 1), dtype=np.float32) * 0.2,
        "time": np.linspace(0, 1, N, dtype=np.float32)[:, None],
        "y": np.random.randint(0, 4, size=N).astype(np.int64),
        "subject": "SYNTH_01",
        "night": "1",
    }
    out = predictor.predict_night(rec)
    assert out["y_pred"].shape == (N,)
    assert out["probs"].shape == (N, 4)
    assert np.all((out["y_pred"] >= 0) & (out["y_pred"] < 4))
    assert np.allclose(out["probs"].sum(axis=-1), np.ones(N), atol=1e-5)


def test_manifest_sha256_disjointness_and_teacher_isolation(tmp_path):
    actual_sha = sha256_file(AUTHORITATIVE_MANIFEST_CSV_DEFAULT)
    assert actual_sha == EXPECTED_MANIFEST_SHA256, f"Manifest SHA mismatch: {actual_sha}"

    manifests = verify_manifests_and_splits(MANIFESTS_DIR_DEFAULT, AUTHORITATIVE_MANIFEST_CSV_DEFAULT)
    assert len(manifests) == 5

    # Verify all 15 EXP-021 teacher checkpoints exist
    for fold in [1, 2, 3, 4, 5]:
        for seed in [42, 123, 2024]:
            ckpt = os.path.join(
                EXP021_DIR, "checkpoints", f"slamss_ifs_official_L1200_fold{fold}_seed{seed}.pt"
            )
            assert os.path.exists(ckpt), f"Missing EXP-021 teacher checkpoint: {ckpt}"

    # Verify TeacherCachedDataset rejects any cache containing an outer_test or inner_val subject
    bad_cache = tmp_path / "bad_teacher_cache.pt"
    f1_val_sub = manifests[1]["inner_val_subjects"][0]
    torch.save(
        {
            "subjects": [manifests[1]["inner_train_subjects"][0], f1_val_sub],
            "ihr": torch.zeros((2, 1200, 30)),
            "accel": torch.zeros((2, 1200, 30)),
            "meta": torch.zeros((2, 1200, 5)),
            "y": torch.zeros((2, 1200), dtype=torch.int64),
            "mask": torch.ones((2, 1200), dtype=torch.bool),
            "seq_len": torch.tensor([1200, 1200], dtype=torch.int64),
            "teacher_logits": torch.zeros((2, 1200, 4)),
        },
        bad_cache,
    )
    with pytest.raises(AssertionError, match="LEAKAGE DETECTED"):
        TeacherCachedDataset(str(bad_cache), set(manifests[1]["inner_train_subjects"]))


def test_compute_fold_class_weights_normalized():
    recs = [
        {"y": np.array([0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 3, 3, -100, -100], dtype=np.int64)},
    ]
    w = compute_fold_class_weights(recs, n_classes=4)
    assert w.shape == (4,)
    assert abs(float(w.mean().item()) - 1.0) < 1e-5
    # Majority class 1 must have smallest weight
    assert w[1] < w[0] and w[1] < w[2] and w[1] < w[3]


def test_paired_vs_ref_bootstrap_ni_and_index_alignment():
    rng = np.random.default_rng(42)
    subs = [f"S{i:02d}" for i in range(25)]
    b = pd.Series(rng.uniform(0.55, 0.70, size=25), index=subs)
    a = b + 0.015  # +1.50 pp improvement
    res = _paired_vs_ref(a, b, ni_margin_pp=-2.50)
    assert abs(res["delta_pp"] - 1.50) < 1e-6
    assert res["ci95_low_pp"] > -2.50
    assert res["ci95_low_exceeds_ni"] is True
    assert res["wilcoxon_p_ni_one_sided"] < 0.001

    # Misaligned subject IDs must raise AssertionError
    b_bad = b.copy()
    b_bad.index = [f"X{i:02d}" for i in range(25)]
    with pytest.raises(AssertionError, match="Subject index mismatch"):
        _paired_vs_ref(a, b_bad)


def test_stage_a_fold_subject_pooling_n25():
    manifests = verify_manifests_and_splits(MANIFESTS_DIR_DEFAULT, AUTHORITATIVE_MANIFEST_CSV_DEFAULT)
    fold_sub_keys = set()
    unique_subs = set()
    for fold in [1, 2, 3, 4, 5]:
        for sub in manifests[fold]["inner_val_subjects"]:
            fold_sub_keys.add((fold, sub))
            unique_subs.add(sub)
    assert len(fold_sub_keys) == 25, f"Expected 25 (fold, sub) inner_val pairs, got {len(fold_sub_keys)}"
    assert len(unique_subs) == 21, "4 subjects appear in two folds' inner_val; keying by (fold, sub) is required"


def test_verdict_rule_exhaustive_and_multiplicity():
    # Case 1: M1 & M2 pass primary (K_qualifying=2 -> 6 secondary tests), M2 passes all 3 secondary gates, M1 fails acc
    # M3 has p_raw=0.03, p_holm>=0.05, delta=+0.60 -> ARM_INCONCLUSIVE (trend branch)
    primary = {
        "M1": {"delta_pp": 1.0, "wilcoxon_p_two_sided": 0.001},
        "M2": {"delta_pp": 1.4, "wilcoxon_p_two_sided": 0.0005},
        "M3": {"delta_pp": 0.60, "wilcoxon_p_two_sided": 0.03},
    }
    acc = {
        "M1": {"delta_pp": 0.2, "wilcoxon_p_two_sided": 0.20},  # fails acc
        "M2": {"delta_pp": 1.2, "wilcoxon_p_two_sided": 0.001},
        "M3": {"delta_pp": 0.5, "wilcoxon_p_two_sided": 0.04},
    }
    sa_s = {
        "M1": {"ci95_low_exceeds_ni": True, "wilcoxon_p_ni_one_sided": 0.002},
        "M2": {"ci95_low_exceeds_ni": True, "wilcoxon_p_ni_one_sided": 0.001},
        "M3": {"ci95_low_exceeds_ni": True, "wilcoxon_p_ni_one_sided": 0.002},
    }
    sa_e = {
        "M1": {"ci95_low_exceeds_ni": True, "wilcoxon_p_ni_one_sided": 0.002},
        "M2": {"ci95_low_exceeds_ni": True, "wilcoxon_p_ni_one_sided": 0.001},
        "M3": {"ci95_low_exceeds_ni": True, "wilcoxon_p_ni_one_sided": 0.002},
    }
    # Force M3 to fail Holm by giving p=0.049 (still <0.05 unadjusted) with M1/M2 passing
    primary_c1 = {k: dict(v) for k, v in primary.items()}
    primary_c1["M3"]["wilcoxon_p_two_sided"] = 0.06  # NOT_SUPPORTED
    v1 = evaluate_exp032_verdicts(primary_c1, acc, sa_s, sa_e)
    assert v1["k_qualifying"] == 2
    assert v1["n_secondary_tests"] == 6
    assert v1["per_arm"]["M2"]["verdict"] == "ARM_SUPPORTED"
    assert v1["per_arm"]["M1"]["verdict"] == "ARM_INCONCLUSIVE"
    assert v1["per_arm"]["M3"]["verdict"] == "ARM_NOT_SUPPORTED"
    assert v1["overall_verdict"] == "SUPPORTED"
    assert "standalone" in v1["attribution"]

    # Case 2: All 3 arms fail Holm, M3 hits unadjusted trend (p=0.03, delta=+0.60) -> INCONCLUSIVE
    primary_c2 = {
        "M1": {"delta_pp": -0.2, "wilcoxon_p_two_sided": 0.50},
        "M2": {"delta_pp": 0.1, "wilcoxon_p_two_sided": 0.40},
        "M3": {"delta_pp": 0.60, "wilcoxon_p_two_sided": 0.03},  # p_holm = 0.09 >= 0.05
    }
    v2 = evaluate_exp032_verdicts(primary_c2, acc, sa_s, sa_e)
    assert v2["k_qualifying"] == 0
    assert v2["per_arm"]["M3"]["verdict"] == "ARM_INCONCLUSIVE"
    assert v2["overall_verdict"] == "INCONCLUSIVE"

    # Case 3: All 3 arms fail -> NOT SUPPORTED
    primary_c3 = {
        "M1": {"delta_pp": -0.2, "wilcoxon_p_two_sided": 0.50},
        "M2": {"delta_pp": 0.1, "wilcoxon_p_two_sided": 0.40},
        "M3": {"delta_pp": 0.2, "wilcoxon_p_two_sided": 0.30},
    }
    v3 = evaluate_exp032_verdicts(primary_c3, acc, sa_s, sa_e)
    assert v3["overall_verdict"] == "NOT SUPPORTED"

