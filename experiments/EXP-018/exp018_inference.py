"""
exp018_inference.py: Sliding-Window Full-Night Inference Engine for EXP-018.

Authoritative Implementation:
  - Supports arbitrary native night length N across all context lengths L in {1, 20, 60, 120, 300, 600, 1200}.
  - Case 1 (N <= L): Right-pads to L with zero-padding protection (seq_lens=N passed to pack_padded_sequence),
    bypasses margin discarding (m=0), and retains predictions [0, N) directly.
  - Case 2 (N > L): Strided sliding window with W=L, stride S=max(1, floor(L/2)), margin discard m=floor(0.25*L),
    asymmetric boundary window anchoring at w=0 and w=N-L, and emission probability averaging.
  - Assertions: Verifies zero gap in nocturnal prediction coverage (every epoch has count >= 1).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SlidingWindowPredictor:
    """
    Standardized full-night inference engine with probability aggregation.
    """
    def __init__(
        self,
        model: nn.Module,
        context_len: int,
        neutralize_circadian: bool = True,
        device: Optional[torch.device] = None
    ):
        self.model = model
        self.context_len = context_len
        self.neutralize_circadian = neutralize_circadian
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model.to(self.device)
        self.model.eval()

    def _prepare_window_tensors(
        self,
        ihr_slice: np.ndarray,
        acc_slice: np.ndarray,
        freq_slice: np.ndarray,
        time_slice: np.ndarray,
        seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Formats numpy window slices into torch tensors with circadian control."""
        L = len(ihr_slice)
        hr_mean = ihr_slice.mean(axis=1, keepdims=True)
        hr_std = ihr_slice.std(axis=1, keepdims=True)
        clock = np.zeros_like(time_slice)

        if self.neutralize_circadian:
            time_feat = np.zeros_like(time_slice)
        else:
            time_feat = time_slice

        meta_slice = np.concatenate([freq_slice, clock, time_feat, hr_mean, hr_std], axis=-1)

        x_ihr = torch.from_numpy(ihr_slice.astype(np.float32)).unsqueeze(0).to(self.device)   # (1, L, 30)
        x_acc = torch.from_numpy(acc_slice.astype(np.float32)).unsqueeze(0).to(self.device)   # (1, L, 30)
        x_meta = torch.from_numpy(meta_slice.astype(np.float32)).unsqueeze(0).to(self.device) # (1, L, 5)
        seq_lens_t = torch.tensor([seq_len], dtype=torch.int64)

        return x_ihr, x_acc, x_meta, seq_lens_t

    @torch.no_grad()
    def predict_night(
        self,
        record: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """
        Runs sliding-window inference over an entire night, returning dense probabilities and stage predictions.
        """
        N = len(record["y"])
        L = self.context_len

        # Case 1: Short Night (N <= L)
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
                seq_lens=seq_lens_t
            )
            logits = out["logits"].squeeze(0)  # (L, 4)
            probs = F.softmax(logits, dim=-1).cpu().numpy()  # (L, 4)

            # Bypass margin discard entirely (m=0), retain valid [0, N)
            probs_valid = probs[:N]
            preds_valid = np.argmax(probs_valid, axis=-1)
            counts = np.ones(N, dtype=np.float64)

            return {
                "y_true": record["y"],
                "y_pred": preds_valid,
                "probs": probs_valid,
                "counts": counts,
                "subject": record["subject"],
                "night": record["night"],
                "n_epochs": N
            }

        # Case 2: Standard Night (N > L)
        stride = max(1, L // 2)
        margin = 0 if L == 1 else int(np.floor(0.25 * L))

        prob_sum = np.zeros((N, 4), dtype=np.float64)
        counts = np.zeros(N, dtype=np.float64)

        # Pre-register window starts: regular stride points plus boundary end anchor (N - L)
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
                seq_lens=seq_lens_t
            )
            logits = out["logits"].squeeze(0)  # (L, 4)
            probs_w = F.softmax(logits, dim=-1).cpu().numpy()  # (L, 4)

            if L == 1:
                prob_sum[w] += probs_w[0]
                counts[w] += 1.0
            elif w == 0:
                # Boundary start: retain [0, L - margin)
                end_idx = L - margin
                prob_sum[0 : end_idx] += probs_w[0 : end_idx]
                counts[0 : end_idx] += 1.0
            elif w == N - L:
                # Boundary end: retain [margin, L)
                start_in_w = margin
                len_segment = L - margin
                dest_start = w + margin
                prob_sum[dest_start : N] += probs_w[start_in_w : L]
                counts[dest_start : N] += 1.0
            else:
                # Interior window: retain central slice [margin, L - margin)
                start_in_w = margin
                end_in_w = L - margin
                dest_start = w + margin
                dest_end = w + L - margin
                prob_sum[dest_start : dest_end] += probs_w[start_in_w : end_in_w]
                counts[dest_start : dest_end] += 1.0

        # Hard invariant: Assert complete coverage (no unpredicted gap epochs)
        assert np.all(counts > 0), (
            f"Coverage gap in sliding-window inference: {np.sum(counts == 0)} epochs have 0 predictions!"
        )

        final_probs = prob_sum / np.maximum(counts[:, None], 1e-12)
        final_preds = np.argmax(final_probs, axis=-1)

        return {
            "y_true": record["y"],
            "y_pred": final_preds,
            "probs": final_probs,
            "counts": counts,
            "subject": record["subject"],
            "night": record["night"],
            "n_epochs": N
        }
