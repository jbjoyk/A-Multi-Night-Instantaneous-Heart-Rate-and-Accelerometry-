"""
official_slamss_ifs_adapter.py: Official 5,465,768-Parameter SLAMSS-IFS Adapter (EXP-021 / TSK-08).

Strictly wraps the official BIDSLabUMass/SLAMSS-IFS repository implementation selected under DEC-011:
  - File: experiments/EXP-005/official_slamss_ifs/model/SLAMSS_IFS_Model.py
  - CNN subnetwork:     5,166,272 trainable parameters
  - Seq2Seq subnetwork:   299,496 trainable parameters
  - Total verified:     5,465,768 trainable parameters (33.34x Model A's 163,940 params)

Key Integrity Guarantees:
  1. Zero label leakage: Seq2Seq receives an all-zero start/target tensor with
     teacher_forcing_ratio=0.0 at BOTH train and inference time (preventing t=0 target leakage).
  2. Exact DEC-011 composite loss: L_total = L_CE(masked) + 0.01 * L_RCC(masked) computed in FP32
     for numerical stability under PyTorch AMP.
  3. Protocol-matched L=1200 dataset and sliding-window inference engine mirroring
     experiments/EXP-018/exp018_dataset.py and exp018_inference.py (100% coverage invariant).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

EXP021_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP021_DIR, "..", ".."))
OFFICIAL_SLAMSS_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-005", "official_slamss_ifs")

for _p in (REPO_ROOT, OFFICIAL_SLAMSS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from model.SLAMSS_IFS_Model import Attention, CNN, Decoder, Encoder, Seq2Seq
from test_rccloss_audit import RCCLoss, get_official_weights

EXPECTED_OFFICIAL_SLAMSS_PARAMS = 5465768


class OfficialSLAMSSIFSWrapper(nn.Module):
    """
    Unified nn.Module wrapper combining official SLAMSS-IFS CNN and Seq2Seq modules.
    Enforces exact parameter count (5,465,768) and zero-leakage teacher_forcing_ratio=0.0.
    """

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        init_device = device or torch.device("cpu")
        enc_hid_dim = 96
        dec_hid_dim = 96
        in_w = 2
        output_dim = 4

        attn = Attention(enc_hid_dim, dec_hid_dim)
        enc = Encoder(in_w, in_w, enc_hid_dim, dec_hid_dim, 0.0)
        dec = Decoder(output_dim, output_dim, enc_hid_dim, dec_hid_dim, 0.0, attn)

        self.cnn = CNN()
        self.s2s = Seq2Seq(enc, dec, init_device)
        self.n_classes = output_dim

        cnn_params = sum(p.numel() for p in self.cnn.parameters())
        s2s_params = sum(p.numel() for p in self.s2s.parameters())
        total_params = cnn_params + s2s_params
        assert total_params == EXPECTED_OFFICIAL_SLAMSS_PARAMS, (
            f"Official SLAMSS-IFS parameter mismatch: got {total_params:,} "
            f"(CNN={cnn_params:,}, Seq2Seq={s2s_params:,}), "
            f"expected {EXPECTED_OFFICIAL_SLAMSS_PARAMS:,}"
        )
        self.to(init_device)

    def to(self, *args: Any, **kwargs: Any) -> "OfficialSLAMSSIFSWrapper":
        super().to(*args, **kwargs)
        # Keep internal Seq2Seq device attribute synchronized with parameter device
        param_device = next(self.parameters()).device
        self.s2s.device = param_device
        return self

    def forward(
        self,
        x_signal: torch.Tensor,
        x_meta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_signal: (B, L * 30, 2) tensor of [IHR, Accel_mag] at 1 Hz
            x_meta:   (B, L, 5) tensor of per-epoch metadata channels
        Returns:
            logits:   (B, L, 4) raw stage emission logits

        Note (DEC-011): Upstream SLAMSS_IFS_Model.py does not pack padded sequences or mask
        attention keys; right-padded zero tails for short nights (N < L) are preserved
        unmodified for faithful official repository reproduction while masked out of the loss.
        """
        batch_size, seq_len, meta_dim = x_meta.shape
        assert meta_dim == 5, f"Expected x_meta last dim == 5, got {meta_dim}"
        assert x_signal.shape == (batch_size, seq_len * 30, 2), (
            f"Expected x_signal shape {(batch_size, seq_len * 30, 2)}, got {tuple(x_signal.shape)}"
        )

        if self.s2s.device != x_signal.device:
            self.s2s.device = x_signal.device

        # All-zero start/target placeholder prevents ground-truth label leakage at t=0
        trg_zeros = torch.zeros(
            (batch_size, seq_len, self.n_classes),
            device=x_signal.device,
            dtype=x_signal.dtype,
        )
        cnn_feat = self.cnn(x_signal, x_meta)
        logits = self.s2s(cnn_feat, x_signal, trg_zeros, teacher_forcing_ratio=0.0)
        return logits


class OfficialSLAMSSCompositeLoss(nn.Module):
    """
    DEC-011 / Official SLAMSS-IFS composite loss:
      L_total = L_CE(masked) + 0.01 * L_RCC(masked)
    Executes strictly with AMP autocast disabled in float32 so torch.mm inside RCCLoss
    never downcasts to float16. Normalized by valid masked epoch count so loss scale is
    invariant to padding length and prevents GradScaler FP16 overflow.
    """

    def __init__(self, rcc_weight: float = 0.01, n_classes: int = 4):
        super().__init__()
        self.rcc_weight = float(rcc_weight)
        self.n_classes = int(n_classes)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        fn_w, fp_w = get_official_weights()
        self.rcc_loss = RCCLoss(fn_weights=fn_w, fp_weights=fp_w, cls=n_classes)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        with torch.amp.autocast(device_type=logits.device.type, enabled=False):
            logits_f32 = logits.float().contiguous().view(-1, self.n_classes)
            targets_flat = targets.contiguous().view(-1)

            valid = (targets_flat >= 0) & (targets_flat < self.n_classes)
            if mask is not None:
                valid = valid & mask.contiguous().view(-1).bool()

            if not valid.any():
                # Keep zero loss attached to the autograd graph so GradScaler.unscale_() succeeds
                return logits_f32.sum() * 0.0

            clamped_targets = torch.clamp(targets_flat, 0, self.n_classes - 1)
            ce_per_epoch = self.ce_loss(logits_f32, clamped_targets)
            ce_term = ce_per_epoch[valid].mean()

            raw_rcc = self.rcc_loss(logits_f32, clamped_targets)
            v_mask = valid.float().unsqueeze(-1).expand(-1, self.n_classes)
            rcc_term = -torch.sum(raw_rcc * v_mask) / valid.sum().float().clamp(min=1.0)

            return ce_term + self.rcc_weight * rcc_term


def build_slamss_window_arrays(
    ihr_slice: np.ndarray,
    acc_slice: np.ndarray,
    freq_slice: np.ndarray,
    time_slice: np.ndarray,
    neutralize_circadian: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs protocol-matched (L*30, 2) signal and (L, 5) metadata arrays matching
    Model A's exact EXP-018 feature representation:
      x_signal[:, 0] = IHR (1 Hz, L*30)
      x_signal[:, 1] = Accel magnitude (1 Hz, L*30)
      x_meta         = [freq, clock=0, time_feat, hr_mean, hr_std]
    """
    L = len(ihr_slice)
    x_signal = np.empty((L * 30, 2), dtype=np.float32)
    x_signal[:, 0] = ihr_slice.reshape(-1).astype(np.float32)
    x_signal[:, 1] = acc_slice.reshape(-1).astype(np.float32)

    hr_mean = ihr_slice.mean(axis=1, keepdims=True)
    hr_std = ihr_slice.std(axis=1, keepdims=True)
    clock = np.zeros_like(time_slice)
    time_feat = np.zeros_like(time_slice) if neutralize_circadian else time_slice

    x_meta = np.concatenate([freq_slice, clock, time_feat, hr_mean, hr_std], axis=-1).astype(
        np.float32
    )
    return x_signal, x_meta


class SLAMSSMatchedDataset(Dataset):
    """
    Protocol-matched full-night dataset (L=1200) for training official SLAMSS-IFS.
    Mirrors experiments/EXP-018/exp018_dataset.py::RandomCropSleepDataset:
      - N > L: uniform random start crop of length L (train) or head slice (deterministic).
      - N <= L: right-pad signals/meta with 0.0, targets with -100, mask=False on padded positions.
    """

    def __init__(
        self,
        records: List[Dict[str, np.ndarray]],
        context_len: int = 1200,
        neutralize_circadian: bool = True,
        random_crop: bool = True,
        seed: Optional[int] = None,
    ):
        assert len(records) > 0, "Records list cannot be empty"
        self.records = records
        self.context_len = int(context_len)
        self.neutralize_circadian = bool(neutralize_circadian)
        self.random_crop = bool(random_crop)
        self.rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        native_len = len(rec["y"])
        L = self.context_len

        if native_len > L:
            s = int(self.rng.randint(0, native_len - L + 1)) if self.random_crop else 0
            e = s + L
            ihr_slice = rec["ihr"][s:e]
            acc_slice = rec["accel"][s:e]
            freq_slice = rec["freq"][s:e]
            time_slice = rec["time"][s:e]
            y_slice = rec["y"][s:e]
            mask_slice = np.ones(L, dtype=bool)
        else:
            pad_len = L - native_len
            ihr_slice = np.pad(rec["ihr"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            acc_slice = np.pad(rec["accel"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            freq_slice = np.pad(rec["freq"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            time_slice = np.pad(rec["time"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
            y_slice = np.pad(rec["y"], (0, pad_len), mode="constant", constant_values=-100)
            mask_slice = np.zeros(L, dtype=bool)
            mask_slice[:native_len] = True

        x_signal, x_meta = build_slamss_window_arrays(
            ihr_slice=ihr_slice,
            acc_slice=acc_slice,
            freq_slice=freq_slice,
            time_slice=time_slice,
            neutralize_circadian=self.neutralize_circadian,
        )

        return {
            "x_signal": torch.from_numpy(x_signal),
            "x_meta": torch.from_numpy(x_meta),
            "y": torch.from_numpy(y_slice.astype(np.int64)),
            "mask": torch.from_numpy(mask_slice),
            "subject": rec["subject"],
            "night": rec["night"],
        }


class SLAMSSSlidingWindowPredictor:
    """
    Full-night sliding-window inference engine for OfficialSLAMSSIFSWrapper.
    Exactly mirrors experiments/EXP-018/exp018_inference.py::SlidingWindowPredictor:
      - Case 1 (N <= L): right-pad to L, bypass margin discard (m=0), slice [0:N].
      - Case 2 (N > L): W=L, stride=max(1, L//2), margin=floor(0.25*L), boundary anchors
        at w=0 and w=N-L, emission probability averaging, and complete coverage assertion.
    """

    def __init__(
        self,
        model: OfficialSLAMSSIFSWrapper,
        context_len: int = 1200,
        neutralize_circadian: bool = True,
        device: Optional[torch.device] = None,
        use_amp: bool = True,
    ):
        self.model = model
        self.context_len = int(context_len)
        self.neutralize_circadian = bool(neutralize_circadian)
        self.device = device or (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.use_amp = bool(use_amp and self.device.type == "cuda")
        self.model.to(self.device)
        self.model.eval()

    def _infer_windows_batched_probs(
        self,
        sig_list: List[np.ndarray],
        meta_list: List[np.ndarray],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Runs batched inference over pre-built (L*30, 2) / (L, 5) windows under torch.no_grad."""
        n_win = len(sig_list)
        if n_win == 0:
            return np.zeros((0, self.context_len, 4), dtype=np.float32)

        all_probs: List[np.ndarray] = []
        for start in range(0, n_win, batch_size):
            end = min(start + batch_size, n_win)
            x_sig = torch.from_numpy(np.stack(sig_list[start:end], axis=0)).to(
                self.device, non_blocking=True
            )
            x_meta = torch.from_numpy(np.stack(meta_list[start:end], axis=0)).to(
                self.device, non_blocking=True
            )
            with torch.amp.autocast(
                device_type=self.device.type, dtype=torch.float16, enabled=self.use_amp
            ):
                logits = self.model(x_sig, x_meta)  # (b, L, 4)
            probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.concatenate(all_probs, axis=0)

    def _infer_window_probs(
        self,
        ihr_slice: np.ndarray,
        acc_slice: np.ndarray,
        freq_slice: np.ndarray,
        time_slice: np.ndarray,
    ) -> np.ndarray:
        x_sig_np, x_meta_np = build_slamss_window_arrays(
            ihr_slice=ihr_slice,
            acc_slice=acc_slice,
            freq_slice=freq_slice,
            time_slice=time_slice,
            neutralize_circadian=self.neutralize_circadian,
        )
        return self._infer_windows_batched_probs([x_sig_np], [x_meta_np], batch_size=1)[0]

    @torch.no_grad()
    def predict_nights_batched(
        self,
        records: List[Dict[str, np.ndarray]],
        batch_size: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        Batched full-night sliding-window inference across multiple nights.
        Produces mathematically identical probabilities to calling predict_night() sequentially
        while avoiding launch-bound B=1 Seq2Seq overhead (~12-15x faster on RTX 3060).
        """
        if not records:
            return []

        L = self.context_len
        stride = max(1, L // 2)
        margin = 0 if L == 1 else int(np.floor(0.25 * L))

        sig_list: List[np.ndarray] = []
        meta_list: List[np.ndarray] = []
        # (rec_idx, is_short, w_start, N)
        win_index: List[Tuple[int, bool, int, int]] = []

        prob_sums: List[np.ndarray] = []
        counts_list: List[np.ndarray] = []

        for rec_idx, record in enumerate(records):
            N = len(record["y"])
            prob_sums.append(np.zeros((N, 4), dtype=np.float64))
            counts_list.append(np.zeros(N, dtype=np.float64))

            if N <= L:
                pad_len = L - N
                ihr_pad = np.pad(record["ihr"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
                acc_pad = np.pad(record["accel"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
                freq_pad = np.pad(record["freq"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
                time_pad = np.pad(record["time"], ((0, pad_len), (0, 0)), mode="constant", constant_values=0.0)
                xs, xm = build_slamss_window_arrays(
                    ihr_pad, acc_pad, freq_pad, time_pad, neutralize_circadian=self.neutralize_circadian
                )
                sig_list.append(xs)
                meta_list.append(xm)
                win_index.append((rec_idx, True, 0, N))
            else:
                w_starts = sorted(list(set(range(0, N - L + 1, stride)) | {N - L}))
                for w in w_starts:
                    xs, xm = build_slamss_window_arrays(
                        record["ihr"][w : w + L],
                        record["accel"][w : w + L],
                        record["freq"][w : w + L],
                        record["time"][w : w + L],
                        neutralize_circadian=self.neutralize_circadian,
                    )
                    sig_list.append(xs)
                    meta_list.append(xm)
                    win_index.append((rec_idx, False, w, N))

        all_probs = self._infer_windows_batched_probs(sig_list, meta_list, batch_size=batch_size)

        for k, (rec_idx, is_short, w, N) in enumerate(win_index):
            probs_w = all_probs[k]
            if is_short:
                prob_sums[rec_idx][:N] += probs_w[:N]
                counts_list[rec_idx][:N] += 1.0
            elif L == 1:
                prob_sums[rec_idx][w] += probs_w[0]
                counts_list[rec_idx][w] += 1.0
            elif w == 0:
                end_idx = L - margin
                prob_sums[rec_idx][0:end_idx] += probs_w[0:end_idx]
                counts_list[rec_idx][0:end_idx] += 1.0
            elif w == N - L:
                dest_start = w + margin
                prob_sums[rec_idx][dest_start:N] += probs_w[margin:L]
                counts_list[rec_idx][dest_start:N] += 1.0
            else:
                dest_start = w + margin
                dest_end = w + L - margin
                prob_sums[rec_idx][dest_start:dest_end] += probs_w[margin : L - margin]
                counts_list[rec_idx][dest_start:dest_end] += 1.0

        outputs: List[Dict[str, Any]] = []
        for rec_idx, record in enumerate(records):
            counts = counts_list[rec_idx]
            assert np.all(counts > 0), (
                f"Coverage gap in batched SLAMSS inference for {record['subject']}/{record['night']}: "
                f"{int(np.sum(counts == 0))} epochs uncovered"
            )
            final_probs = prob_sums[rec_idx] / np.maximum(counts[:, None], 1e-12)
            final_preds = np.argmax(final_probs, axis=-1)
            outputs.append(
                {
                    "y_true": record["y"],
                    "y_pred": final_preds,
                    "probs": final_probs,
                    "counts": counts,
                    "subject": record["subject"],
                    "night": record["night"],
                    "n_epochs": len(record["y"]),
                }
            )
        return outputs

    @torch.no_grad()
    def predict_night(self, record: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return self.predict_nights_batched([record], batch_size=1)[0]
