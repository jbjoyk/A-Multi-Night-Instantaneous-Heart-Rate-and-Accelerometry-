"""
exp019_adaptation_core.py: Core PyTorch adaptation library for EXP-019.

Authoritative Implementation adhering strictly to design_draft.md:
  - Model A: ConventionalCNNBiGRU (163,940 parameters).
  - Physical mini-batching: B=2 calibration nights right-padded to L=1200.
  - adapt_supervised_ft: fine-tuning with RCC loss and selective parameter freezing.
  - adapt_unsupervised_infomax: InfoMax with Shannon entropy + marginal diversity penalty + anti-collapse safeguard.
  - train_per_subject_scratch: scratch training with linear LR warmup.
  - compute_wearable_quality_mask: 30 <= HR <= 180 bpm and non-zero accelerometry variance.
  - reset_model_weights: in-memory state restoration via deep copy.
"""

import os
import sys
import copy
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure repo root and EXP-018 are accessible
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
exp018_dir = os.path.abspath(os.path.join(repo_root, "experiments", "EXP-018"))
for p in [repo_root, exp018_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from exp013_models import ConventionalCNNBiGRU


def compute_wearable_quality_mask(
    ihr_array: np.ndarray,
    accel_array: np.ndarray,
    raw_bpm_array: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Computes boolean quality mask strictly adhering to PREREG_EXP-019_S3_AMENDMENT.md:
    1. Accelerometry Validity: Scale-invariant intra-epoch variance check (Var_t(m) > 0.0),
       which correctly screens out sensor freeze, disconnected hardware, or zero-padding.
    2. Heart Rate Validity:
       - If raw_bpm_array or unstandardized BPM is provided (mean > 25.0):
         enforces physiological plausibility bounds (30 <= HR <= 180 bpm).
       - If ihr_array consists of standardized z-scores (empirical [-4, +10]):
         verifies valid non-NaN non-extreme physiological range (|z| <= 10.0).
    ihr_array: shape (T, 30) or (T,)
    accel_array: shape (T, 30) or (T,)
    raw_bpm_array: optional shape (T, 30) or (T,) with unstandardized BPM values
    Returns:
    mask: shape (T,), dtype bool
    """
    if ihr_array.ndim == 2:
        hr_mean = np.mean(ihr_array, axis=-1)
    else:
        hr_mean = ihr_array

    if accel_array.ndim == 2:
        acc_var = np.var(accel_array, axis=-1)
    else:
        acc_var = np.zeros_like(hr_mean) + 1.0

    valid_acc = (acc_var > 1e-6)

    if raw_bpm_array is not None:
        raw_mean = np.mean(raw_bpm_array, axis=-1) if raw_bpm_array.ndim == 2 else raw_bpm_array
        valid_hr = (raw_mean >= 30.0) & (raw_mean <= 180.0)
    elif np.nanmax(np.abs(hr_mean)) > 25.0:
        # Detected unstandardized raw BPM array
        valid_hr = (hr_mean >= 30.0) & (hr_mean <= 180.0)
    else:
        # Standardized z-score representation from prep_57738d534b:
        # Screen out non-finite (NaN/Inf) and extreme sensor clipping (|z| > 10.0)
        valid_hr = np.isfinite(hr_mean) & (np.abs(hr_mean) <= 10.0)

    return (valid_hr & valid_acc).astype(bool)


def reset_model_weights(
    model: nn.Module,
    reference_state_dict: Dict[str, torch.Tensor]
) -> None:
    """Restores model weights to reference state via in-memory deep copy."""
    model.load_state_dict(copy.deepcopy(reference_state_dict))


def prepare_calibration_batch(
    cal_records: List[Dict[str, np.ndarray]],
    context_len: int = 1200,
    device: str = "cuda:0"
) -> Dict[str, torch.Tensor]:
    """
    Prepares a physical mini-batch (B=len(cal_records), L=context_len) from calibration nights.
    Nights are right-padded to context_len with valid targets and lengths protected.
    Circadian features are neutralized (time_feat = 0, clock = 0).
    """
    B = len(cal_records)
    L = context_len

    x_ihr_list = []
    x_acc_list = []
    x_meta_list = []
    y_list = []
    mask_list = []
    seq_lens_list = []
    wearable_mask_list = []

    for rec in cal_records:
        N = min(len(rec["y"]), L)
        seq_lens_list.append(N)

        # Slice up to L
        ihr_slice = rec["ihr"][:N]
        acc_slice = rec["accel"][:N]
        freq_slice = rec["freq"][:N]
        time_slice = rec["time"][:N]
        y_slice = rec["y"][:N]

        # Pad to L
        pad_len = L - N
        ihr_pad = np.pad(ihr_slice, ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
        acc_pad = np.pad(acc_slice, ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
        freq_pad = np.pad(freq_slice, ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
        time_pad = np.pad(time_slice, ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)

        # Meta calculation with neutralized circadian features
        hr_mean = ihr_pad.mean(axis=1, keepdims=True)
        hr_std = ihr_pad.std(axis=1, keepdims=True)
        clock = np.zeros_like(time_pad)
        time_feat = np.zeros_like(time_pad)  # neutralize circadian
        meta_pad = np.concatenate([freq_pad, clock, time_feat, hr_mean, hr_std], axis=-1)

        # Targets: pad with -100
        y_pad = np.pad(y_slice, (0, pad_len), mode="constant", constant_values=-100)
        mask_pad = (y_pad >= 0) & (y_pad < 4)

        # Wearable quality mask
        w_mask_slice = compute_wearable_quality_mask(ihr_slice, acc_slice)
        w_mask_pad = np.pad(w_mask_slice, (0, pad_len), mode="constant", constant_values=False)

        x_ihr_list.append(ihr_pad)
        x_acc_list.append(acc_pad)
        x_meta_list.append(meta_pad)
        y_list.append(y_pad)
        mask_list.append(mask_pad)
        wearable_mask_list.append(w_mask_pad)

    return {
        "x_ihr": torch.from_numpy(np.stack(x_ihr_list).astype(np.float32)).to(device),
        "x_acc": torch.from_numpy(np.stack(x_acc_list).astype(np.float32)).to(device),
        "x_meta": torch.from_numpy(np.stack(x_meta_list).astype(np.float32)).to(device),
        "y": torch.from_numpy(np.stack(y_list).astype(np.int64)).to(device),
        "mask": torch.from_numpy(np.stack(mask_list).astype(bool)).to(device),
        "seq_lens": torch.tensor(seq_lens_list, dtype=torch.int64).to(device),
        "wearable_mask": torch.from_numpy(np.stack(wearable_mask_list).astype(bool)).to(device)
    }


def adapt_supervised_ft(
    model: nn.Module,
    cal_records: List[Dict[str, np.ndarray]],
    lr: float,
    epochs: int,
    scope_regex: str,
    device: str = "cuda:0"
) -> nn.Module:
    """
    Executes supervised fine-tuning on calibration nights using RCC loss and selective parameter freezing.
    1 epoch = 1 optimizer step over the physical mini-batch of k=2 calibration nights.
    """
    pattern = re.compile(scope_regex)
    for name, param in model.named_parameters():
        param.requires_grad = bool(pattern.search(name))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError(f"No trainable parameters matched scope_regex: '{scope_regex}'")

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.0
    )

    batch = prepare_calibration_batch(cal_records, context_len=1200, device=device)
    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(
            x_ihr=batch["x_ihr"],
            x_accel=batch["x_acc"],
            x_meta=batch["x_meta"],
            targets=batch["y"],
            mask=batch["mask"],
            seq_lens=batch["seq_lens"]
        )
        loss = out["total_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

    model.eval()
    return model


def adapt_unsupervised_infomax(
    model: nn.Module,
    cal_records: List[Dict[str, np.ndarray]],
    lr: float,
    steps: int,
    lambda_div: float = 1.0,
    prior_entropy: float = 1.3863,
    device: str = "cuda:0"
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Executes unsupervised InfoMax adaptation with marginal diversity penalty and collapse abort assertion.
    Feature encoder backbone is frozen; only temporal GRU and classifier head are adapted.
    H_prior = ln(4) approx 1.3863 nats. Abort threshold = 0.5 * H_prior = 0.6931 nats.
    Returns:
      (adapted_model, info_dict)
    """
    scope_regex = r"(classifier_head|temporal_gru.*weight_.*l1|temporal_gru.*bias_.*l1|temp_norm)"
    pattern = re.compile(scope_regex)
    for name, param in model.named_parameters():
        param.requires_grad = bool(pattern.search(name))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.0
    )

    batch = prepare_calibration_batch(cal_records, context_len=1200, device=device)
    wearable_mask = batch["wearable_mask"]  # (B, L)
    abort_threshold = 0.5 * prior_entropy  # 0.6931 nats

    model.train()
    aborted = False
    abort_step = None
    final_marginal_entropy = 0.0

    for step in range(steps):
        optimizer.zero_grad()
        out = model(
            x_ihr=batch["x_ihr"],
            x_accel=batch["x_acc"],
            x_meta=batch["x_meta"],
            seq_lens=batch["seq_lens"]
        )
        logits = out["logits"]  # (B, L, 4)
        probs = F.softmax(logits, dim=-1)  # (B, L, 4)

        # Filter strictly by wearable quality mask
        valid_probs = probs[wearable_mask]  # (M, 4)
        if valid_probs.shape[0] == 0:
            break

        # 1. Shannon entropy per epoch: H(p) = - sum(p * ln(p))
        per_epoch_entropy = - torch.sum(valid_probs * torch.log(valid_probs + 1e-8), dim=-1)
        loss_ent = per_epoch_entropy.mean()

        # 2. Marginal diversity penalty: H(mean(p))
        p_mean = valid_probs.mean(dim=0)  # (4,)
        marginal_entropy = - torch.sum(p_mean * torch.log(p_mean + 1e-8))
        final_marginal_entropy = float(marginal_entropy.item())

        # 3. Collapse abort safeguard
        if marginal_entropy.item() < abort_threshold:
            aborted = True
            abort_step = step
            break

        loss = loss_ent - lambda_div * marginal_entropy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

    model.eval()
    info = {
        "aborted": aborted,
        "abort_step": abort_step,
        "final_marginal_entropy": final_marginal_entropy,
        "completed_steps": abort_step if aborted else steps
    }
    return model, info


def train_per_subject_scratch(
    cal_records: List[Dict[str, np.ndarray]],
    lr: float,
    steps: int,
    device: str = "cuda:0"
) -> nn.Module:
    """
    Trains a clean Model A from scratch exclusively on the subject's calibration nights.
    Uses linear learning rate warmup over the first 20 steps (or 10% of steps).
    """
    model = ConventionalCNNBiGRU(
        d_model=64,
        d_temp=96,
        n_classes=4,
        n_meta=5,
        dropout=0.2,
        use_rcc_loss=True
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.0
    )

    warmup_steps = min(20, max(1, steps // 10))
    batch = prepare_calibration_batch(cal_records, context_len=1200, device=device)
    model.train()

    for step in range(steps):
        # Linear LR warmup
        if step < warmup_steps:
            cur_lr = lr * float(step + 1) / float(warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr
        else:
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        optimizer.zero_grad()
        out = model(
            x_ihr=batch["x_ihr"],
            x_accel=batch["x_acc"],
            x_meta=batch["x_meta"],
            targets=batch["y"],
            mask=batch["mask"],
            seq_lens=batch["seq_lens"]
        )
        loss = out["total_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    model.eval()
    return model
