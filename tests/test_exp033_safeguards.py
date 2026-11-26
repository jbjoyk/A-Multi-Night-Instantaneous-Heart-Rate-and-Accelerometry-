"""
Pre-run and post-run safeguard tests for EXP-033.
Verifies:
  1. SHA-256 digests of fold_manifests.csv and all 15 W100 checkpoints.
  2. Bitwise identical ZS predictions for a subject before and after fine-tuning
     another subject, plus state-dict hash immutability of the frozen base_state.
"""

import os
import sys

import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP033_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-033")
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
for _p in (REPO_ROOT, EXP033_DIR, EXP029_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp033_core import (  # noqa: E402
    build_exp033_backbone,
    get_backbone_ckpt_map,
    predict_nights,
    run_single_ft,
    state_dict_sha256,
    verify_inputs_sha256,
)
from run_exp029_cv import CACHE_DIR_DEFAULT, load_all_records_by_subject  # noqa: E402


def test_sha256_manifests_and_w100_checkpoints():
    res = verify_inputs_sha256()
    assert res["w100_checkpoints_verified"] == 15


def test_zs_bitwise_identical_after_other_subject_finetuning():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt_map = get_backbone_ckpt_map("W100")
    base_state = torch.load(ckpt_map[(1, 42)], map_location="cpu", weights_only=True)
    base_hash_before = state_dict_sha256(base_state)

    records_by_sub = load_all_records_by_subject(CACHE_DIR_DEFAULT)
    eligible = sorted(s for s, recs in records_by_sub.items() if len(recs) >= 3)
    sub_a, sub_b = eligible[0], eligible[1]
    recs_a = sorted(records_by_sub[sub_a], key=lambda r: int(r["night"]))
    recs_b = sorted(records_by_sub[sub_b], key=lambda r: int(r["night"]))

    # ZS on subject A before fine-tuning subject B
    m_zs_before = build_exp033_backbone("W100").to(device).eval()
    m_zs_before.load_state_dict(base_state)
    preds_before = predict_nights(m_zs_before, recs_a[2:], device)

    # Fine-tune on subject B
    m_ft_b, hash_b_before, hash_b_after = run_single_ft(
        backbone="W100",
        base_state=base_state,
        cal_recs=recs_b[:2],
        lr=1e-4,
        epochs=5,
        scope="all",
        seed=42,
        device=device,
    )
    assert hash_b_before == base_hash_before
    assert hash_b_after != hash_b_before
    del m_ft_b

    # ZS on subject A after fine-tuning subject B
    base_hash_after = state_dict_sha256(base_state)
    assert base_hash_after == base_hash_before

    m_zs_after = build_exp033_backbone("W100").to(device).eval()
    m_zs_after.load_state_dict(base_state)
    preds_after = predict_nights(m_zs_after, recs_a[2:], device)

    assert len(preds_before) == len(preds_after)
    for pb, pa in zip(preds_before, preds_after):
        assert np.array_equal(pb["y_pred"], pa["y_pred"])
        assert np.array_equal(pb["probs"], pa["probs"])
