"""
EXP-031 Single-Factor Width Sweep Model Factory
File: experiments/EXP-031/exp031_models.py

Reuses ScaledConventionalCNNBiGRU from experiments/EXP-030/exp030_models.py
without mutating EXP-030 or exp013_models.py / sleep_adaptive_net.py.
Covers all 9 plain CNN-BiGRU widths ordered by parameter count:
  W050 (43,108), W075 (93,796), A (163,940), P190 (190,940), B (223,268),
  W125 (253,540), W150 (362,596), W200 (639,076), W250 (993,380).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import torch.nn as nn

EXP031_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP031_DIR, "..", ".."))
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
for _p in (REPO_ROOT, EXP030_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp030_models import ScaledConventionalCNNBiGRU  # noqa: E402


EXP031_ARMS_ORDER: List[str] = [
    "W050",
    "W075",
    "A",
    "P190",
    "B",
    "W125",
    "W150",
    "W200",
    "W250",
]

EXP031_NEW_ARMS: List[str] = [
    "W050",
    "W075",
    "P190",
    "W125",
    "W150",
    "W200",
    "W250",
]

EXP031_REUSED_PLAIN_ARMS: List[str] = ["A", "B"]

# Section 9 deployment constraint: Dynamic INT8 in-memory payload < 400 KiB
# Strictly the 5 eligible plain CNN-BiGRU widths (V5 / full_v5 is excluded).
EXP031_ELIGIBLE_INT8_ARMS: List[str] = ["W050", "W075", "A", "P190", "B"]

FULL_V5_PARAMS = 190788

EXP031_ARMS: Dict[str, Dict[str, Any]] = {
    "W050": {
        "w": 0.50,
        "d_enc": 32,
        "d_temp": 48,
        "head_hidden": 32,
        "expected_params": 43108,
        "expected_int8_kib": 81.00,
        "eligible_int8_400kib": True,
        "is_new": True,
        "reused_oof_csv": None,
    },
    "W075": {
        "w": 0.75,
        "d_enc": 48,
        "d_temp": 72,
        "head_hidden": 48,
        "expected_params": 93796,
        "expected_int8_kib": 176.96,
        "eligible_int8_400kib": True,
        "is_new": True,
        "reused_oof_csv": None,
    },
    "A": {
        "w": 1.00,
        "d_enc": 64,
        "d_temp": 96,
        "head_hidden": 64,
        "expected_params": 163940,
        "expected_int8_kib": 309.91,
        "eligible_int8_400kib": True,
        "is_new": False,
        "reused_oof_csv": os.path.join(REPO_ROOT, "experiments", "EXP-030", "p2_164k_subject_metrics.csv"),
    },
    "P190": {
        "w": 1.08,
        "d_enc": 72,
        "d_temp": 100,
        "head_hidden": 64,
        "expected_params": 190940,
        "expected_int8_kib": 375.16,
        "eligible_int8_400kib": True,
        "is_new": True,
        "reused_oof_csv": None,
    },
    "B": {
        "w": 1.17,
        "d_enc": 64,
        "d_temp": 128,
        "head_hidden": 64,
        "expected_params": 223268,
        "expected_int8_kib": 368.04,
        "eligible_int8_400kib": True,
        "is_new": False,
        "reused_oof_csv": os.path.join(REPO_ROOT, "experiments", "EXP-029", "cnn_bigru_200k_subject_metrics.csv"),
    },
    "W125": {
        "w": 1.25,
        "d_enc": 80,
        "d_temp": 120,
        "head_hidden": 80,
        "expected_params": 253540,
        "expected_int8_kib": 479.86,
        "eligible_int8_400kib": False,
        "is_new": True,
        "reused_oof_csv": None,
    },
    "W150": {
        "w": 1.50,
        "d_enc": 96,
        "d_temp": 144,
        "head_hidden": 96,
        "expected_params": 362596,
        "expected_int8_kib": 686.82,
        "eligible_int8_400kib": False,
        "is_new": True,
        "reused_oof_csv": None,
    },
    "W200": {
        "w": 2.00,
        "d_enc": 128,
        "d_temp": 192,
        "head_hidden": 128,
        "expected_params": 639076,
        "expected_int8_kib": 1211.72,
        "eligible_int8_400kib": False,
        "is_new": True,
        "reused_oof_csv": None,
    },
    "W250": {
        "w": 2.50,
        "d_enc": 160,
        "d_temp": 240,
        "head_hidden": 160,
        "expected_params": 993380,
        "expected_int8_kib": 1884.63,
        "eligible_int8_400kib": False,
        "is_new": True,
        "reused_oof_csv": None,
    },
}


def build_exp031_model(arm: str, loss_type: str = "rccloss") -> nn.Module:
    """
    Factory for EXP-031 plain width-scaled ConventionalCNNBiGRU models.
    arm must be one of EXP031_ARMS_ORDER.
    loss_type in {"rccloss", "ce"}.
    """
    if arm not in EXP031_ARMS:
        raise ValueError(f"Unknown EXP-031 arm '{arm}'. Valid arms: {EXP031_ARMS_ORDER}")
    if loss_type not in {"rccloss", "ce"}:
        raise ValueError(f"Invalid loss_type '{loss_type}'. Expected 'rccloss' or 'ce'.")
    cfg = EXP031_ARMS[arm]
    return ScaledConventionalCNNBiGRU(
        d_enc=int(cfg["d_enc"]),
        d_temp=int(cfg["d_temp"]),
        head_hidden=int(cfg["head_hidden"]),
        n_classes=4,
        n_meta=5,
        dropout=0.2,
        use_rcc_loss=(loss_type == "rccloss"),
    )
