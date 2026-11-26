"""
EXP-033 Core Utilities, SHA-256 Verification, Model Factory, and Adaptation Primitives.
File: experiments/EXP-033/exp033_core.py
"""

from __future__ import annotations

import copy
import hashlib
import os
import re
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

EXP033_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP033_DIR, "..", ".."))
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
EXP031_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-031")
EXP019_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-019")

for _p in (REPO_ROOT, EXP029_DIR, EXP030_DIR, EXP031_DIR, EXP019_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp019_adaptation_core import prepare_calibration_batch  # noqa: E402
from exp029_calibration import LogitStitchingSlidingWindowPredictor  # noqa: E402
from exp029_models import build_exp029_model  # noqa: E402
from exp030_models import ScaledConventionalCNNBiGRU, EXP030_GRID  # noqa: E402
from exp031_models import EXP031_ARMS  # noqa: E402
from run_exp029_cv import (  # noqa: E402
    AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
    CACHE_DIR_DEFAULT,
    MANIFESTS_DIR_DEFAULT,
    compute_epoch_metrics,
    set_deterministic_seed,
    sha256_file,
)

FOLDS = [1, 2, 3, 4, 5]
SEEDS = [42, 123, 2024]

EXPECTED_MANIFEST_SHA256 = "36d840207e937081655438a9aa4170a682cfb90e9bac3fb63c8ec0d9b15adfcf"

EXPECTED_W100_CKPT_SHA256: Dict[Tuple[int, int], str] = {
    (1, 42): "9cbf86756d8f8cb037f7a8b8afd69ba2d125bd4b32abb4bcc7f870b6ea0e124b",
    (1, 123): "d2a1b68448d32f5dc32a20e00e301c11f382ba13fa8e61df8eefbe160d20fee1",
    (1, 2024): "b24597ecc7aafcfddc628e6f980a97bbc8d08ffdb4715876932272194e58bef1",
    (2, 42): "f080601580dbb1f2783804a82e472faaa0810050506cb4fd6cbe55b7971d8c09",
    (2, 123): "7b3568e6e88155275581e063829d134a52402c07c6dfcd59fda60ffd7b80b76e",
    (2, 2024): "06ec27de92fadfa46e001c98d9e379e4bfbc1a508c6ebfbc331d1383ff688a1c",
    (3, 42): "70363741a45e28d545d530de891ca350eba2f1c0c71d05b6f980cafbc05b870b",
    (3, 123): "c74d5527050783b91fdadc2d195d5da84b84d712f2ba7896be3588f764a2af92",
    (3, 2024): "a11a17c4a3b8de3e1eaab56179c60955f5ebe34e11d4a8a7fdee693324fd7add",
    (4, 42): "fdac65033e373fae7ad4895bf1e7c42f4709a466200b626da2a4925827f2253f",
    (4, 123): "6881ce59c9a03f7c2e6bb97943a7310a94d91c3e81716c630a8728d4a16e65a9",
    (4, 2024): "88627e07f496c7a980371bcee97887cb4433bc3116d8922b92b6cb42319d3837",
    (5, 42): "8cb2e46da46df11b4c56094ab31b29f8076822823723095dea56baea194a490f",
    (5, 123): "08323ad53fe82cdba60c56998dd3df8b40377eb53bf08f241f9c54eaf7c030ec",
    (5, 2024): "dd87d1275a5fe5f5fce1d1999348df9af174751055382b4b9ab4eb204804deda",
}

BACKBONE_ORDER = ["W100", "p1_100k", "P190", "full_v5"]


def state_dict_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    """Deterministic SHA-256 hash over sorted state_dict keys and tensor bytes."""
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        h.update(k.encode("utf-8"))
        t = state_dict[k].detach().cpu().contiguous()
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def verify_inputs_sha256(
    manifest_csv: str = AUTHORITATIVE_MANIFEST_CSV_DEFAULT,
) -> Dict[str, Any]:
    """Verify fold_manifests.csv and all 15 W100 checkpoint SHA-256 digests."""
    actual_manifest = sha256_file(manifest_csv)
    if actual_manifest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(
            f"fold_manifests.csv SHA-256 mismatch: expected {EXPECTED_MANIFEST_SHA256}, got {actual_manifest}"
        )
    ckpt_dir = os.path.join(EXP031_DIR, "runs", "W100", "checkpoints")
    for (fold, seed), expected_hash in sorted(EXPECTED_W100_CKPT_SHA256.items()):
        path = os.path.join(ckpt_dir, f"W100_L1200_fold{fold}_seed{seed}.pt")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing W100 checkpoint: {path}")
        actual = sha256_file(path)
        if actual != expected_hash:
            raise RuntimeError(
                f"W100 checkpoint SHA-256 mismatch for fold={fold}, seed={seed}: "
                f"expected {expected_hash}, got {actual}"
            )
    return {
        "manifest_sha256": actual_manifest,
        "w100_checkpoints_verified": len(EXPECTED_W100_CKPT_SHA256),
    }


def build_exp033_backbone(backbone: str) -> nn.Module:
    if backbone == "W100":
        cfg = EXP031_ARMS["A"]
        return ScaledConventionalCNNBiGRU(
            d_enc=cfg["d_enc"],
            d_temp=cfg["d_temp"],
            head_hidden=cfg["head_hidden"],
        )
    if backbone == "P190":
        cfg = EXP031_ARMS["P190"]
        return ScaledConventionalCNNBiGRU(
            d_enc=cfg["d_enc"],
            d_temp=cfg["d_temp"],
            head_hidden=cfg["head_hidden"],
        )
    if backbone == "p1_100k":
        cfg = EXP030_GRID["p1_100k"]
        return ScaledConventionalCNNBiGRU(
            d_enc=cfg["d_enc"],
            d_temp=cfg["d_temp"],
            head_hidden=cfg["head_hidden"],
        )
    if backbone == "full_v5":
        return build_exp029_model("full_v5")
    raise ValueError(f"Unknown backbone: {backbone}")


def get_backbone_ckpt_map(backbone: str) -> Dict[Tuple[int, int], str]:
    ckpt_map: Dict[Tuple[int, int], str] = {}
    for fold in FOLDS:
        for seed in SEEDS:
            if backbone == "W100":
                p = os.path.join(EXP031_DIR, "runs", "W100", "checkpoints", f"W100_L1200_fold{fold}_seed{seed}.pt")
            elif backbone == "P190":
                p = os.path.join(EXP031_DIR, "checkpoints", "P190", f"P190_L1200_fold{fold}_seed{seed}.pt")
            elif backbone == "p1_100k":
                p = os.path.join(EXP030_DIR, "checkpoints", "p1_100k", f"p1_100k_L1200_fold{fold}_seed{seed}.pt")
            elif backbone == "full_v5":
                p = os.path.join(EXP029_DIR, "checkpoints", "full_v5", f"full_v5_L1200_fold{fold}_seed{seed}.pt")
            else:
                raise ValueError(f"Unknown backbone: {backbone}")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing checkpoint for {backbone} fold={fold} seed={seed}: {p}")
            ckpt_map[(fold, seed)] = p
    return ckpt_map


def get_backbone_oof_csv(backbone: str) -> str:
    if backbone == "W100":
        return os.path.join(EXP031_DIR, "runs", "W100", "W100_subject_metrics.csv")
    if backbone == "P190":
        return os.path.join(EXP031_DIR, "P190_subject_metrics.csv")
    if backbone == "p1_100k":
        return os.path.join(EXP030_DIR, "p1_100k_subject_metrics.csv")
    if backbone == "full_v5":
        return os.path.join(EXP029_DIR, "full_v5_subject_metrics.csv")
    raise ValueError(f"Unknown backbone: {backbone}")


def get_scope_regex(backbone: str, scope: str) -> str:
    if scope == "all":
        return r".*"
    if scope == "head_gru":
        if backbone == "full_v5":
            return r"(classifier_head|coarse_gru|fusion_proj|temp_norm)"
        return r"(classifier_head|temporal_gru.*weight_.*l1|temporal_gru.*bias_.*l1|temp_norm)"
    raise ValueError(f"Unknown scope: {scope}")


def run_single_ft(
    backbone: str,
    base_state: Dict[str, torch.Tensor],
    cal_recs: List[Dict[str, np.ndarray]],
    lr: float,
    epochs: int,
    scope: str,
    seed: int,
    device: torch.device,
) -> Tuple[nn.Module, str, str]:
    """
    Fine-tune a fresh deep copy of base_state on cal_recs using Adam,
    batch_size=len(cal_recs), grad clip 1.0, and CE + 0.01*RCCLoss.
    Returns (adapted_model, hash_before, hash_after).
    """
    set_deterministic_seed(seed)
    fresh_state = copy.deepcopy(base_state)
    hash_before = state_dict_sha256(fresh_state)

    model = build_exp033_backbone(backbone).to(device)
    model.load_state_dict(fresh_state)

    pattern = re.compile(get_scope_regex(backbone, scope))
    for name, p in model.named_parameters():
        p.requires_grad = bool(pattern.search(name))
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(f"No trainable parameters matched scope={scope} for backbone={backbone}")

    optimizer = torch.optim.Adam(trainable, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
    batch = prepare_calibration_batch(cal_recs, context_len=1200, device=str(device))

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
        loss = out["total_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

    model.eval()
    hash_after = state_dict_sha256(model.state_dict())
    return model, hash_before, hash_after


@torch.no_grad()
def predict_nights(
    model: nn.Module,
    records: List[Dict[str, np.ndarray]],
    device: torch.device,
) -> List[Dict[str, np.ndarray]]:
    predictor = LogitStitchingSlidingWindowPredictor(model=model, context_len=1200, device=device)
    return [predictor.predict_night(r) for r in records]


def metrics_from_night_preds(preds: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
    if not preds:
        return {"n_valid_epochs": 0, "macro_f1": 0.0, "f1_wake": 0.0, "f1_light": 0.0, "f1_deep": 0.0, "f1_rem": 0.0}
    yt = np.concatenate([p["y_true"] for p in preds])
    yp = np.concatenate([p["y_pred"] for p in preds])
    return compute_epoch_metrics(yt, yp)
