"""
EXP-029 Unified Pre-Softmax Logit-Stitching & Post-Hoc Prior Calibration
File: experiments/EXP-029/exp029_calibration.py

Subclasses exp018_inference.SlidingWindowPredictor without modifying historical
files, implementing:
  1. Unified pre-softmax logit stitching across both N <= L and N > L.
  2. Zero-parameter DampedEMCalibrator (Saerens et al., 2002) with locked
     a priori hyperparameters: alpha=0.5, n_iter=3, T=1.0, and empirical pi_src
     computed from per-fold inner_val_subjects.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EXP029_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP029_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exp018_inference import SlidingWindowPredictor


class DampedEMCalibrator:
    """
    Zero-tuning Saerens et al. (2002) damped EM prior calibrator.
    Locked a priori parameters: alpha=0.5, n_iter=3, T=1.0.
    """

    def __init__(
        self,
        pi_src: np.ndarray,
        alpha: float = 0.5,
        n_iter: int = 3,
        temperature: float = 1.0,
        eps: float = 1e-8,
    ):
        pi_arr = np.asarray(pi_src, dtype=np.float64).reshape(-1)
        if pi_arr.shape != (4,):
            raise ValueError(f"pi_src must have shape (4,), got {pi_arr.shape}")
        if np.any(~np.isfinite(pi_arr)) or np.sum(pi_arr) <= 0:
            raise ValueError(f"Invalid pi_src: {pi_arr}")
        pi_arr = np.clip(pi_arr, eps, None)
        self.pi_src = pi_arr / np.sum(pi_arr)
        self.alpha = float(alpha)
        self.n_iter = int(n_iter)
        self.temperature = float(temperature)
        self.eps = float(eps)

    def calibrate(self, probs: np.ndarray) -> np.ndarray:
        """
        Applies damped EM prior adjustment to an (N, 4) probability matrix.
        Safe on extreme label-shift nights (e.g., 0% Deep sleep).
        """
        p = np.asarray(probs, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 4:
            raise ValueError(f"Expected probs of shape (N, 4), got {p.shape}")
        if len(p) == 0:
            return p.copy()

        p = np.clip(p, 0.0, None)
        row_sums = np.maximum(p.sum(axis=1, keepdims=True), 1e-12)
        p = p / row_sums

        pi_curr = self.pi_src.copy()
        for _ in range(self.n_iter):
            ratio = np.clip(pi_curr / self.pi_src, self.eps, 1.0 / self.eps)
            weighted = p * ratio[None, :]
            denom = np.maximum(weighted.sum(axis=1, keepdims=True), 1e-12)
            post = weighted / denom
            pi_em = post.mean(axis=0)
            pi_curr = (1.0 - self.alpha) * pi_curr + self.alpha * pi_em
            pi_curr = np.clip(pi_curr, self.eps, None)
            pi_curr = pi_curr / np.sum(pi_curr)

        final_ratio = np.clip(pi_curr / self.pi_src, self.eps, 1.0 / self.eps)
        final_weighted = p * final_ratio[None, :]
        final_denom = np.maximum(final_weighted.sum(axis=1, keepdims=True), 1e-12)
        return final_weighted / final_denom


class LogitStitchingSlidingWindowPredictor(SlidingWindowPredictor):
    """
    Subclass of SlidingWindowPredictor performing unified pre-softmax logit
    stitching across both N <= L and N > L, followed by a single post-stitch
    softmax (with optional DampedEMCalibrator).
    """

    def __init__(
        self,
        model: nn.Module,
        context_len: int = 1200,
        neutralize_circadian: bool = True,
        device: Optional[torch.device] = None,
        prior_calibrator: Optional[DampedEMCalibrator] = None,
        temperature: float = 1.0,
    ):
        super().__init__(
            model=model,
            context_len=context_len,
            neutralize_circadian=neutralize_circadian,
            device=device,
        )
        self.prior_calibrator = prior_calibrator
        self.temperature = (
            prior_calibrator.temperature if prior_calibrator is not None else float(temperature)
        )

    @torch.no_grad()
    def predict_night(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accumulates pre-softmax logits across windows for both N <= L and N > L,
        applies unified post-stitch softmax, and optionally runs DampedEMCalibrator.
        """
        N = len(record["y"])
        L = self.context_len

        logit_sum = np.zeros((N, 4), dtype=np.float64)
        counts = np.zeros(N, dtype=np.float64)

        if N <= L:
            pad_len = L - N
            ihr_pad = np.pad(record["ihr"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            acc_pad = np.pad(record["accel"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            freq_pad = np.pad(record["freq"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            time_pad = np.pad(record["time"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)

            x_ihr, x_acc, x_meta, seq_lens_t = self._prepare_window_tensors(
                ihr_pad, acc_pad, freq_pad, time_pad, seq_len=N
            )
            out = self.model(
                x_ihr=x_ihr,
                x_accel=x_acc,
                x_meta=x_meta,
                seq_lens=seq_lens_t,
            )
            logits_valid = out["logits"].squeeze(0)[:N].float().cpu().numpy()
            logit_sum[:] = logits_valid
            counts[:] = 1.0
        else:
            stride = max(1, L // 2)
            margin = 0 if L == 1 else int(np.floor(0.25 * L))
            w_starts = sorted(list(set(range(0, N - L + 1, stride)) | {N - L}))

            for w in w_starts:
                ihr_w = record["ihr"][w : w + L]
                acc_w = record["accel"][w : w + L]
                freq_w = record["freq"][w : w + L]
                time_w = record["time"][w : w + L]

                x_ihr, x_acc, x_meta, seq_lens_t = self._prepare_window_tensors(
                    ihr_w, acc_w, freq_w, time_w, seq_len=L
                )
                out = self.model(
                    x_ihr=x_ihr,
                    x_accel=x_acc,
                    x_meta=x_meta,
                    seq_lens=seq_lens_t,
                )
                logits_w = out["logits"].squeeze(0).float().cpu().numpy()

                if L == 1:
                    logit_sum[w] += logits_w[0]
                    counts[w] += 1.0
                elif w == 0:
                    end_idx = L - margin
                    logit_sum[0:end_idx] += logits_w[0:end_idx]
                    counts[0:end_idx] += 1.0
                elif w == N - L:
                    dest_start = w + margin
                    logit_sum[dest_start:N] += logits_w[margin:L]
                    counts[dest_start:N] += 1.0
                else:
                    dest_start = w + margin
                    dest_end = w + L - margin
                    logit_sum[dest_start:dest_end] += logits_w[margin : L - margin]
                    counts[dest_start:dest_end] += 1.0

        assert np.all(counts > 0), (
            f"Coverage gap in sliding-window logit stitching: {np.sum(counts == 0)} epochs have 0 predictions!"
        )

        stitched_logits = logit_sum / np.maximum(counts[:, None], 1e-12)
        temp = max(self.temperature, 1e-6)
        probs = F.softmax(torch.from_numpy(stitched_logits) / temp, dim=-1).numpy()

        if self.prior_calibrator is not None:
            probs = self.prior_calibrator.calibrate(probs)

        preds = np.argmax(probs, axis=-1)

        return {
            "y_true": record["y"],
            "y_pred": preds,
            "probs": probs,
            "logits": stitched_logits,
            "counts": counts,
            "subject": record["subject"],
            "night": record["night"],
            "n_epochs": N,
        }


@torch.no_grad()
def estimate_pi_src_from_records(
    model: nn.Module,
    val_records: List[Dict[str, Any]],
    context_len: int = 1200,
    neutralize_circadian: bool = True,
    device: Optional[torch.device] = None,
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Computes empirical source prior pi_src as the mean predicted softmax
    marginal across all valid epochs in a fold's inner_val_subjects.
    """
    predictor = LogitStitchingSlidingWindowPredictor(
        model=model,
        context_len=context_len,
        neutralize_circadian=neutralize_circadian,
        device=device,
        prior_calibrator=None,
        temperature=temperature,
    )
    prob_chunks: List[np.ndarray] = []
    for rec in val_records:
        out = predictor.predict_night(rec)
        y = np.asarray(rec["y"])
        valid = (y >= 0) & (y < 4)
        if np.any(valid):
            prob_chunks.append(out["probs"][valid])
    if not prob_chunks:
        return np.full(4, 0.25, dtype=np.float64)
    all_probs = np.concatenate(prob_chunks, axis=0)
    pi_src = all_probs.mean(axis=0).astype(np.float64)
    pi_src = np.clip(pi_src, 1e-8, None)
    return pi_src / np.sum(pi_src)
