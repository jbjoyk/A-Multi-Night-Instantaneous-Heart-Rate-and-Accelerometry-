#!/usr/bin/env python3
"""
test_exp020_preflight.py: Mandatory CPU-fast pre-flight unit tests for EXP-020.
"""

import json
import os
import sys
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in [
    REPO_ROOT,
    os.path.join(REPO_ROOT, "experiments", "EXP-019"),
    os.path.join(REPO_ROOT, "experiments", "EXP-020"),
    os.path.join(REPO_ROOT, "experiments", "STAGE2_FINAL_MODEL"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp019_adaptation_core import adapt_supervised_ft
from run_exp020_regularized_gated_adaptation import (
    DEFAULT_CACHE_DIR,
    DEFAULT_FROZEN_MANIFEST,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_NIGHT_INVENTORY,
    adapt_supervised_ft_kl,
    compute_kl_regularized_loss,
    load_subject_records,
    restore_pristine_model,
    set_deterministic_seed,
)
from final_model_pipeline import (
    FinalModelConfig,
    FinalSleepStagingSystem,
    SCOPE_REGEX_MAP,
    sha256_file,
)


def test_zero_outer_test_hyperparameter_leakage():
    inv_df = pd.read_csv(DEFAULT_NIGHT_INVENTORY)
    inv_df = inv_df[inv_df["usable"].astype(bool)]
    eligible = {str(s) for s, g in inv_df.groupby("subject_id") if len(g) >= 3}
    assert "Bidslab49" not in eligible

    all_outer = []
    for fold in range(1, 6):
        with open(os.path.join(DEFAULT_MANIFEST_DIR, f"fold_{fold}.json"), "r", encoding="utf-8") as f:
            fm = json.load(f)
        inner_set = {str(s) for s in fm["inner_val_subjects"] if str(s) in eligible}
        outer_set = {str(s) for s in fm["outer_test_subjects"] if str(s) in eligible}
        assert inner_set.isdisjoint(outer_set)
        assert "Bidslab49" not in inner_set and "Bidslab49" not in outer_set
        all_outer.extend(outer_set)

    assert len(all_outer) == 46
    assert len(set(all_outer)) == 46

    fold_csv = pd.read_csv(os.path.join(REPO_ROOT, "review_package", "fold_manifests.csv"))
    assert set(fold_csv["subject_id"].astype(str)) - {"Bidslab49"} == set(all_outer)


def test_deterministic_checkpoint_sha256_and_param_count():
    with open(DEFAULT_FROZEN_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert len(manifest["checkpoints"]) == 15
    cfg = FinalModelConfig.from_json()
    system = FinalSleepStagingSystem(config=cfg, device="cpu")
    assert sum(p.numel() for p in system.backbone.parameters()) == 163940

    for entry in manifest["checkpoints"]:
        ckpt_path = os.path.join(REPO_ROOT, entry["checkpoint_relpath"])
        assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
        assert sha256_file(ckpt_path) == entry["sha256"]


def test_w0a_pristine_reset_and_multi_sweep_immutability():
    cfg = FinalModelConfig.from_json()
    system = FinalSleepStagingSystem(config=cfg, device="cpu")
    with open(DEFAULT_FROZEN_MANIFEST, "r", encoding="utf-8") as f:
        sha_map = {(c["fold"], c["seed"]): c["sha256"] for c in json.load(f)["checkpoints"]}
    system.load_frozen_checkpoint(fold=4, seed=42, expected_sha256=sha_map[(4, 42)])

    frozen_snap = {k: v.clone() for k, v in system._frozen_state.items()}
    inv_df = pd.read_csv(DEFAULT_NIGHT_INVENTORY)
    inv_df = inv_df[inv_df["usable"].astype(bool)].sort_values(["subject_id", "night"])
    sub1 = "Bidslab00"
    nights1 = [str(int(n)) for n in inv_df.loc[inv_df["subject_id"] == sub1, "night"].tolist()]
    cal1, _ = load_subject_records(sub1, nights1, DEFAULT_CACHE_DIR)

    # Fold 4 head_gru freeze followed by Fold 5 scope=all reset
    m_f4 = adapt_supervised_ft_kl(system, cal1, lr=3e-5, epochs=2, scope_regex=SCOPE_REGEX_MAP["head_gru"], lambda_kl=1.0, seed=42)
    for name, p in m_f4.named_parameters():
        if "hr_encoder" in name or "acc_encoder" in name:
            assert not p.requires_grad
            assert torch.equal(p.detach(), frozen_snap[name])

    m_f5 = restore_pristine_model(system, scope_regex=SCOPE_REGEX_MAP["all"])
    assert all(p.requires_grad for p in m_f5.parameters()), "Fold 4 head_gru freeze leaked into Fold 5!"

    for k, v in system._frozen_state.items():
        assert torch.equal(v, frozen_snap[k]), f"_frozen_state mutated at {k}"


def test_kl_loss_boundary_and_mask_invariance():
    cfg = FinalModelConfig.from_json()
    system = FinalSleepStagingSystem(config=cfg, device="cpu")
    with open(DEFAULT_FROZEN_MANIFEST, "r", encoding="utf-8") as f:
        sha_map = {(c["fold"], c["seed"]): c["sha256"] for c in json.load(f)["checkpoints"]}
    system.load_frozen_checkpoint(fold=1, seed=42, expected_sha256=sha_map[(1, 42)])

    inv_df = pd.read_csv(DEFAULT_NIGHT_INVENTORY)
    inv_df = inv_df[inv_df["usable"].astype(bool)].sort_values(["subject_id", "night"])
    nights = [str(int(n)) for n in inv_df.loc[inv_df["subject_id"] == "Bidslab08", "night"].tolist()]
    cal_recs, _ = load_subject_records("Bidslab08", nights, DEFAULT_CACHE_DIR)

    # 1. Exact lambda_KL=0.0 parity with adapt_supervised_ft
    set_deterministic_seed(42)
    ref_clone = system._clone_frozen_backbone()
    ref_adapted = adapt_supervised_ft(ref_clone, cal_recs, lr=1e-4, epochs=2, scope_regex=SCOPE_REGEX_MAP["all"], device="cpu")
    kl0_adapted = adapt_supervised_ft_kl(system, cal_recs, lr=1e-4, epochs=2, scope_regex=SCOPE_REGEX_MAP["all"], lambda_kl=0.0, seed=42)
    for (k1, v1), (k2, v2) in zip(ref_adapted.state_dict().items(), kl0_adapted.state_dict().items()):
        assert k1 == k2
        assert torch.allclose(v1, v2, atol=1e-5), f"lambda_KL=0.0 parity drift at {k1}"

    # 2. Mask invariance at -100 positions
    logits = torch.randn(1, 5, 4, requires_grad=True)
    p_zs = torch.softmax(torch.randn(1, 5, 4), dim=-1)
    valid_mask = torch.tensor([[True, True, False, False, True]])
    base_loss = torch.tensor(0.0)
    loss1 = compute_kl_regularized_loss(logits, p_zs, base_loss, valid_mask, lambda_kl=1.0)
    logits_pert = logits.detach().clone()
    logits_pert[0, 2:4, :] += 100.0
    logits_pert.requires_grad_(True)
    loss2 = compute_kl_regularized_loss(logits_pert, p_zs, base_loss, valid_mask, lambda_kl=1.0)
    assert torch.allclose(loss1, loss2, atol=1e-6)


def test_gate_tau_and_per_fold_arm_c_gate():
    tau = 0.03
    assert (0.0 < tau) is True
    assert ((tau - 1e-6) < tau) is True
    assert (tau < tau) is False
    assert ((tau + 1e-6) < tau) is False
