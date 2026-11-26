"""
EXP-032 High-Accuracy Benchmark Models, Exact 4-State Linear-Chain CRF & Stitched Decoding
File: experiments/EXP-032/exp032_models.py

Strictly implements docs/designs/PREREG_EXP-032_HIGH_ACCURACY_BENCHMARK.md:
  - MultiScaleTCNBlock (d_model=224, d_branch=64, kernels=(11, 15, 21), dilations 1 & 2)
  - M1: MultiScaleTCNBiGRU (1,402,468 params, 4,053.08 MMACs at L=1200)
  - LinearChainCRF4: FP32-forced exact forward-algorithm log-partition, interior -100
    marginalization, clean trailing mask termination, per-epoch class-weighted L_CRF_mean
    normalized to O(1) scale, exact Viterbi decode, and forward-backward marginals.
  - M2: MultiScaleTCNBiGRU + LinearChainCRF4 (1,402,492 params)
  - M3: Fold-matched EXP-021 Official SLAMSS-IFS teacher distillation + M2 CRF (1,402,492 params)
  - CRFLogitStitchingPredictor: stitches pre-CRF emission logits z_S across stride=600 windows
    for T > 1200 nights, running exact Viterbi + forward-backward marginals once per full night.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

EXP032_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP032_DIR, "..", ".."))
EXP030_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-030")
EXP029_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-029")
for _p in (REPO_ROOT, EXP030_DIR, EXP029_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp029_calibration import LogitStitchingSlidingWindowPredictor  # noqa: E402
from exp030_models import ScaledModalityEncoder  # noqa: E402

EXPECTED_M1_PARAMS = 1402468
EXPECTED_M2_PARAMS = 1402492
EXPECTED_M3_PARAMS = 1402492
EXPECTED_MMACS_L1200 = 4053.08


# ==============================================================================
# Parallel Associative-Scan FP32 Linear-Chain CRF Primitives
# ==============================================================================

def _crf_forward_log_partition(
    emissions: torch.Tensor,
    seq_mask: torch.Tensor,
    start_transitions: torch.Tensor,
    end_transitions: torch.Tensor,
    transitions: torch.Tensor,
) -> torch.Tensor:
    """
    Exact parallel associative-scan forward algorithm computing log Z(x) in FP32
    in ceil(log2(T)) = 11 parallel log-semiring matmuls (exact to FP32 precision).
    Args:
        emissions: (B, T, C) float32
        seq_mask:  (B, T) bool (True for valid night positions, False for trailing pad)
        start_transitions: (C,) float32
        end_transitions:   (C,) float32
        transitions:       (C, C) float32, transitions[i, j] = score(i -> j)
    Returns:
        log_z: (B,) float32
    """
    B, T, C = emissions.shape
    v0 = start_transitions.unsqueeze(0) + emissions[:, 0, :]  # (B, C)
    if T == 1:
        return torch.logsumexp(v0 + end_transitions.unsqueeze(0), dim=-1)

    # Transition matrices M[:, t-1, i, j] = transitions[i, j] + emissions[:, t, j]
    # Padded steps (mask == False) use the log-semiring identity matrix (0 diag, -1e9 off-diag)
    M = transitions.view(1, 1, C, C) + emissions[:, 1:, :].unsqueeze(2)  # (B, T-1, C, C)
    log_eye = torch.full((C, C), -1e9, dtype=emissions.dtype, device=emissions.device)
    log_eye.fill_diagonal_(0.0)
    mask_steps = seq_mask[:, 1:].unsqueeze(-1).unsqueeze(-1)  # (B, T-1, 1, 1)
    M = torch.where(mask_steps, M, log_eye.view(1, 1, C, C))

    # Parallel associative reduction in ceil(log2(T-1)) steps
    while M.shape[1] > 1:
        n = M.shape[1]
        if n % 2 == 1:
            last = M[:, -1:]
            m_even = M[:, 0:-1:2]
            m_odd = M[:, 1::2]
            m_pair = torch.logsumexp(m_even.unsqueeze(-1) + m_odd.unsqueeze(-3), dim=-2)
            M = torch.cat([m_pair, last], dim=1)
        else:
            m_even = M[:, 0::2]
            m_odd = M[:, 1::2]
            M = torch.logsumexp(m_even.unsqueeze(-1) + m_odd.unsqueeze(-3), dim=-2)

    m_total = M.squeeze(1)  # (B, C, C)
    alpha = torch.logsumexp(v0.unsqueeze(-1) + m_total, dim=-2) + end_transitions.unsqueeze(0)
    return torch.logsumexp(alpha, dim=-1)


def _crf_forward_backward_marginals(
    emissions: torch.Tensor,
    seq_mask: torch.Tensor,
    start_transitions: torch.Tensor,
    end_transitions: torch.Tensor,
    transitions: torch.Tensor,
) -> torch.Tensor:
    """
    Exact forward-backward marginal posteriors P_crf(y_t | x_{1:T}) in FP32.
    Args:
        emissions: (T, B, C) float32
        seq_mask:  (T, B) bool
    Returns:
        marginals: (B, T, C) float32 probabilities summing to 1.0 over C on valid positions
    """
    T, B, C = emissions.shape
    trans = transitions.unsqueeze(0)  # (1, C, C)

    # Forward pass storing alpha_t for all t
    alphas = torch.empty((T, B, C), dtype=emissions.dtype, device=emissions.device)
    alpha = start_transitions.unsqueeze(0) + emissions[0]
    alphas[0] = alpha
    for t in range(1, T):
        mask_t = seq_mask[t].unsqueeze(1)
        next_alpha = torch.logsumexp(alpha.unsqueeze(2) + trans, dim=1) + emissions[t]
        alpha = torch.where(mask_t, next_alpha, alpha)
        alphas[t] = alpha

    log_z = torch.logsumexp(alpha + end_transitions.unsqueeze(0), dim=1)  # (B,)

    # Backward pass: beta[t, b, i] = log sum of paths from t+1..L-1 given state i at t
    betas = torch.empty((T, B, C), dtype=emissions.dtype, device=emissions.device)
    beta = end_transitions.unsqueeze(0).expand(B, C).clone()
    betas[T - 1] = beta
    for t in range(T - 2, -1, -1):
        next_mask = seq_mask[t + 1].unsqueeze(1)  # (B, 1)
        # trans (1, C, C) + emissions[t+1] (B, 1, C) + beta (B, 1, C) -> logsumexp over next state dim=2
        next_beta = torch.logsumexp(
            trans + emissions[t + 1].unsqueeze(1) + beta.unsqueeze(1), dim=2
        )
        beta = torch.where(next_mask, next_beta, beta)
        betas[t] = beta

    log_marginals = alphas + betas - log_z.unsqueeze(0).unsqueeze(2)  # (T, B, C)
    probs = torch.softmax(log_marginals, dim=2)  # numerically exact normalization
    probs = probs * seq_mask.unsqueeze(2).to(probs.dtype)
    return probs.transpose(0, 1).contiguous()


def _crf_viterbi_decode(
    emissions: torch.Tensor,
    seq_mask: torch.Tensor,
    start_transitions: torch.Tensor,
    end_transitions: torch.Tensor,
    transitions: torch.Tensor,
) -> torch.Tensor:
    """
    Exact masked Viterbi decoding in FP32.
    Args:
        emissions: (T, B, C) float32
        seq_mask:  (T, B) bool
    Returns:
        best_tags: (B, T) int64 (0 on trailing padded positions)
    """
    T, B, C = emissions.shape
    trans = transitions.unsqueeze(0)  # (1, C, C)
    score = start_transitions.unsqueeze(0) + emissions[0]  # (B, C)
    backpointers = torch.zeros((T, B, C), dtype=torch.long, device=emissions.device)

    for t in range(1, T):
        mask_t = seq_mask[t].unsqueeze(1)  # (B, 1)
        # (B, C, 1) + (1, C, C) -> max over prev state dim=1
        broadcast_score = score.unsqueeze(2) + trans
        max_score, best_prev = broadcast_score.max(dim=1)
        next_score = max_score + emissions[t]
        score = torch.where(mask_t, next_score, score)
        backpointers[t] = best_prev

    score = score + end_transitions.unsqueeze(0)
    _, best_last_tag = score.max(dim=1)  # (B,)

    seq_lens = seq_mask.long().sum(dim=0).clamp(min=1)  # (B,)
    best_tags = torch.zeros((B, T), dtype=torch.long, device=emissions.device)

    for b in range(B):
        length = int(seq_lens[b].item())
        tag = int(best_last_tag[b].item())
        best_tags[b, length - 1] = tag
        for t in range(length - 1, 0, -1):
            tag = int(backpointers[t, b, tag].item())
            best_tags[b, t - 1] = tag

    return best_tags


class LinearChainCRF4(nn.Module):
    """
    Exact 4-State Linear-Chain Conditional Random Field (24 learnable parameters).
    Strictly executes in FP32 outside AMP autocast.

    Interior -100 (ignore_index) Marginalization:
      - Positions with (seq_mask == True) & (tags == -100) have their emission potentials
        zeroed across all 4 states and are marginalized via exact log-sum-exp in the
        clamped numerator partition Z_clamped(x, y_obs).
      - Because Z_clamped(x, y_obs) <= Z(x) always holds, NLL = log Z(x) - log Z_clamped >= 0
        is guaranteed for any combination of interior -100 spans, trailing padding, and extreme logits.
    """

    def __init__(self, num_tags: int = 4):
        super().__init__()
        self.num_tags = int(num_tags)
        self.start_transitions = nn.Parameter(torch.zeros(self.num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(self.num_tags))
        self.transitions = nn.Parameter(torch.zeros(self.num_tags, self.num_tags))

    def compute_loss(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        seq_mask: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes per-epoch mean-normalized class-weighted CRF NLL (L_CRF_mean) in FP32.
        Args:
            emissions: (B, T, 4) pre-CRF emission logits
            tags:      (B, T) int64 targets in {0, 1, 2, 3} or -100
            seq_mask:  (B, T) bool (True for valid night bounds, False for trailing pad)
            class_weights: optional (4,) float32 tensor with mean == 1.0
        Returns:
            scalar FP32 tensor (O(1) per-epoch scale)
        """
        with torch.amp.autocast(device_type=emissions.device.type, enabled=False):
            emit = emissions.float()
            mask = seq_mask.bool()
            y = tags.long()
            valid = mask & (y >= 0) & (y < self.num_tags)

            # Edge case: batch has 0 valid labeled epochs
            if not valid.any():
                return emit.sum() * 0.0

            # Weight emission potentials at valid positions by w_{y_t}; zero out interior -100 emissions
            if class_weights is not None:
                cw = class_weights.to(device=emit.device, dtype=torch.float32)
                safe_y = y.clamp(min=0, max=self.num_tags - 1)
                w_bt = torch.where(valid, cw[safe_y], torch.zeros_like(emit[:, :, 0]))
                emit_eff = emit * w_bt.unsqueeze(-1)
                denom = w_bt.sum().clamp(min=1e-6)
            else:
                emit_eff = torch.where(valid.unsqueeze(-1), emit, torch.zeros_like(emit))
                denom = valid.float().sum().clamp(min=1.0)

            # Build clamped emissions for numerator Z_clamped(x, y_obs):
            #   - Observed positions (valid == True): mask out all tags != y_t with -1e9
            #   - Interior -100 positions (mask == True & valid == False): keep all 4 states at 0.0 (marginalized)
            safe_y = y.clamp(min=0, max=self.num_tags - 1)
            one_hot = F.one_hot(safe_y, num_classes=self.num_tags).bool()
            clamp_mask = valid.unsqueeze(-1) & (~one_hot)
            emit_clamped = emit_eff.masked_fill(clamp_mask, -1e9)

            # Batch both unconstrained and clamped emissions together for a single 11-step scan
            emit_both = torch.cat([emit_eff, emit_clamped], dim=0)  # (2*B, T, C)
            mask_both = torch.cat([mask, mask], dim=0)  # (2*B, T)

            start_t = self.start_transitions.float()
            end_t = self.end_transitions.float()
            trans_t = self.transitions.float()

            log_z_both = _crf_forward_log_partition(emit_both, mask_both, start_t, end_t, trans_t)
            B = emit.shape[0]
            log_z = log_z_both[:B]
            log_z_clamped = log_z_both[B:]

            # Exact non-negative NLL per sequence, normalized by sum of valid weights -> O(1) scale
            nll_seq = (log_z - log_z_clamped).clamp(min=0.0)
            return nll_seq.sum() / denom

    @torch.no_grad()
    def forward_backward_marginals(
        self,
        emissions: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes calibrated per-epoch posterior probabilities P_crf(y_t | x_{1:T}) in FP32.
        Args:
            emissions: (B, T, 4)
            seq_mask:  optional (B, T) bool (defaults to all True)
        Returns:
            probs: (B, T, 4) float32
        """
        with torch.amp.autocast(device_type=emissions.device.type, enabled=False):
            emit = emissions.float()
            B, T, _ = emit.shape
            if seq_mask is None:
                mask = torch.ones((B, T), dtype=torch.bool, device=emit.device)
            else:
                mask = seq_mask.bool()
            return _crf_forward_backward_marginals(
                emit.transpose(0, 1).contiguous(),
                mask.transpose(0, 1).contiguous(),
                self.start_transitions.float(),
                self.end_transitions.float(),
                self.transitions.float(),
            )

    @torch.no_grad()
    def viterbi_decode(
        self,
        emissions: torch.Tensor,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Exact MAP sequence decoding via Viterbi algorithm in FP32.
        Args:
            emissions: (B, T, 4)
            seq_mask:  optional (B, T) bool (defaults to all True)
        Returns:
            best_tags: (B, T) int64
        """
        with torch.amp.autocast(device_type=emissions.device.type, enabled=False):
            emit = emissions.float()
            B, T, _ = emit.shape
            if seq_mask is None:
                mask = torch.ones((B, T), dtype=torch.bool, device=emit.device)
            else:
                mask = seq_mask.bool()
            return _crf_viterbi_decode(
                emit.transpose(0, 1).contiguous(),
                mask.transpose(0, 1).contiguous(),
                self.start_transitions.float(),
                self.end_transitions.float(),
                self.transitions.float(),
            )


# ==============================================================================
# Multi-Scale Large-Kernel Residual TCN + BiGRU Architecture (M1, M2, M3)
# ==============================================================================

class MultiScaleTCNBlock(nn.Module):
    """
    Bottleneck multi-scale dilated residual 1D convolution block.
    Targets Official SLAMSS-IFS cnn3 wide-kernel (k=11, 15, 21) sleep-cycle context.
    """

    def __init__(
        self,
        d_model: int = 224,
        d_branch: int = 64,
        kernels: Tuple[int, ...] = (11, 15, 21),
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv1d(d_model, d_branch, kernel_size=1),
            nn.GroupNorm(8, d_branch),
            nn.GELU(),
        )
        self.branches = nn.ModuleList([
            nn.Conv1d(
                d_branch,
                d_branch,
                kernel_size=k,
                padding=((k - 1) * dilation) // 2,
                dilation=dilation,
            )
            for k in kernels
        ])
        self.proj = nn.Sequential(
            nn.Conv1d(d_branch * len(kernels), d_model, kernel_size=1),
            nn.GroupNorm(8, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model, T)
        z = self.reduce(x)
        ms = torch.cat([b(z) for b in self.branches], dim=1)
        return x + self.proj(ms)


class MultiScaleTCNBiGRU(nn.Module):
    """
    Shared EXP-032 high-accuracy backbone:
      - M1 (use_crf=False): 1,402,468 params, 4,053.08 MMACs at L=1200
      - M2 / M3 (use_crf=True): 1,402,492 params, 4,053.08 MMACs at L=1200
    Preserves the standard forward() contract (x_ihr, x_accel, x_meta, seq_lens) -> {"logits": z_S}.
    """

    def __init__(
        self,
        d_enc: int = 128,
        d_model: int = 224,
        d_branch: int = 64,
        d_gru: int = 128,
        head_hidden: int = 128,
        n_classes: int = 4,
        n_meta: int = 5,
        dropout: float = 0.2,
        use_crf: bool = False,
    ):
        super().__init__()
        self.use_crf = bool(use_crf)
        self.n_classes = int(n_classes)

        self.hr_encoder = ScaledModalityEncoder(in_features=1, d_enc=d_enc, dropout=dropout)
        self.acc_encoder = ScaledModalityEncoder(in_features=1, d_enc=d_enc, dropout=dropout)
        self.meta_proj = nn.Sequential(
            nn.Linear(n_meta, 32),
            nn.LayerNorm(32),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(d_enc * 2 + 32, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.tcn = nn.Sequential(
            MultiScaleTCNBlock(d_model, d_branch, (11, 15, 21), dilation=1, dropout=dropout),
            MultiScaleTCNBlock(d_model, d_branch, (11, 15, 21), dilation=2, dropout=dropout),
        )
        self.temporal_gru = nn.GRU(
            d_model,
            d_gru,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.post_gru_norm = nn.LayerNorm(d_gru * 2)
        self.head = nn.Sequential(
            nn.Linear(d_gru * 2, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )
        self.crf = LinearChainCRF4(num_tags=n_classes) if self.use_crf else None

    def forward(
        self,
        x_ihr: torch.Tensor,
        x_accel: Optional[torch.Tensor] = None,
        x_meta: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        x_acc: Optional[torch.Tensor] = None,
        meta: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Support both (x_ihr, x_accel, x_meta) and (x_ihr, x_acc, meta) kwargs
        acc = x_accel if x_accel is not None else x_acc
        assert acc is not None, "Must provide x_accel or x_acc"
        m = x_meta if x_meta is not None else meta

        B, T, _ = x_ihr.shape
        h_hr = self.hr_encoder(x_ihr)
        h_acc = self.acc_encoder(acc)
        if m is None:
            m = torch.zeros((B, T, 5), device=x_ihr.device, dtype=x_ihr.dtype)
        h_meta = self.meta_proj(m)

        h = self.fusion(torch.cat([h_hr, h_acc, h_meta], dim=-1))
        h = self.tcn(h.transpose(1, 2)).transpose(1, 2)

        if seq_lens is not None:
            lens_cpu = seq_lens.detach().to("cpu", dtype=torch.long).clamp(min=1, max=T)
            packed = pack_padded_sequence(h, lens_cpu, batch_first=True, enforce_sorted=False)
            gru_out, _ = self.temporal_gru(packed)
            h, _ = pad_packed_sequence(gru_out, batch_first=True, total_length=T)
        else:
            h, _ = self.temporal_gru(h)

        h = self.post_gru_norm(h)
        logits = self.head(h)
        return {"logits": logits}


def build_exp032_model(arm: str, dropout: float = 0.2) -> MultiScaleTCNBiGRU:
    """Factory enforcing exact locked parameter counts for M1, M2, and M3."""
    if arm == "M1":
        model = MultiScaleTCNBiGRU(dropout=dropout, use_crf=False)
        expected = EXPECTED_M1_PARAMS
    elif arm in ("M2", "M3"):
        model = MultiScaleTCNBiGRU(dropout=dropout, use_crf=True)
        expected = EXPECTED_M2_PARAMS
    else:
        raise ValueError(f"Unknown EXP-032 arm: {arm!r} (expected 'M1', 'M2', or 'M3')")

    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == expected, f"{arm}: expected {expected:,} params, got {n_params:,}"
    return model


# ==============================================================================
# Locked Per-Fold Class Weights & Loss Functions (FP32)
# ==============================================================================

def compute_fold_class_weights(train_records: list, n_classes: int = 4) -> torch.Tensor:
    """
    Computes per-fold inverse-sqrt class weights strictly from inner_train valid epochs:
        w_tilde_c = (N_valid / (4 * N_c))^0.5
        w_c       = w_tilde_c / mean(w_tilde)
    Guarantees mean(w_c) == 1.0.
    """
    counts = np.zeros(n_classes, dtype=np.float64)
    for rec in train_records:
        y = np.asarray(rec["y"], dtype=np.int64)
        valid = (y >= 0) & (y < n_classes)
        for c in range(n_classes):
            counts[c] += float(np.sum(y[valid] == c))

    n_valid = float(np.sum(counts))
    assert n_valid > 0 and np.all(counts > 0), f"Invalid inner_train class counts: {counts}"
    w_tilde = np.sqrt(n_valid / (float(n_classes) * counts))
    w = w_tilde / np.mean(w_tilde)
    return torch.tensor(w, dtype=torch.float32)


def compute_weighted_ce_fp32(
    logits: torch.Tensor,
    targets: torch.Tensor,
    seq_mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    """
    Per-epoch weighted cross-entropy L_CE_weighted in FP32 over valid = (y != -100) & mask.
    """
    with torch.amp.autocast(device_type=logits.device.type, enabled=False):
        z = logits.float()
        y = targets.long()
        valid = seq_mask.bool() & (y >= 0) & (y < z.shape[-1])
        if not valid.any():
            return z.sum() * 0.0

        cw = class_weights.to(device=z.device, dtype=torch.float32)
        safe_y = y.clamp(min=0, max=z.shape[-1] - 1)
        ce_per_epoch = F.cross_entropy(
            z.reshape(-1, z.shape[-1]),
            safe_y.reshape(-1),
            reduction="none",
        ).view_as(y)
        w_bt = torch.where(valid, cw[safe_y], torch.zeros_like(ce_per_epoch))
        return (ce_per_epoch * w_bt).sum() / w_bt.sum().clamp(min=1e-6)


def compute_masked_kl_fp32(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    seq_mask: torch.Tensor,
    tau: float = 2.0,
) -> torch.Tensor:
    """
    Per-epoch masked KL divergence L_KL_masked in FP32:
      F.kl_div(log_softmax(z_S / tau), softmax(z_T / tau), reduction='none').sum(dim=-1)
    averaged strictly over valid epochs ((y != -100) & mask).
    """
    with torch.amp.autocast(device_type=student_logits.device.type, enabled=False):
        z_s = student_logits.float()
        z_t = teacher_logits.float().detach()
        y = targets.long()
        valid = seq_mask.bool() & (y >= 0) & (y < z_s.shape[-1])
        if not valid.any():
            return z_s.sum() * 0.0

        t_val = float(tau)
        log_p_s = F.log_softmax(z_s / t_val, dim=-1)
        p_t = F.softmax(z_t / t_val, dim=-1)
        kl_per_epoch = F.kl_div(log_p_s, p_t, reduction="none").sum(dim=-1)
        return (kl_per_epoch * valid.float()).sum() / valid.float().sum().clamp(min=1.0)


def compute_exp032_loss(
    arm: str,
    model: MultiScaleTCNBiGRU,
    student_logits: torch.Tensor,
    targets: torch.Tensor,
    seq_mask: torch.Tensor,
    class_weights: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    tau: float = 2.0,
    alpha_kd: float = 0.25,
) -> torch.Tensor:
    """
    Unified locked FP32 loss dispatcher matching PREREG_EXP-032 Section 2.2:
      - M1: L_CE_weighted(z_S, y, w_c)
      - M2: L_CRF_mean(z_S, y, mask, w_c) + 0.1 * L_CE_weighted(z_S, y, w_c)
      - M3: (1 - alpha_kd) * L_M2 + alpha_kd * tau^2 * L_KL_masked(z_S, z_T, y, mask, tau)
    """
    l_ce = compute_weighted_ce_fp32(student_logits, targets, seq_mask, class_weights)
    if arm == "M1":
        return l_ce

    assert model.crf is not None, f"{arm} requires model.crf to be enabled"
    l_crf = model.crf.compute_loss(student_logits, targets, seq_mask, class_weights=class_weights)
    l_m2 = l_crf + 0.1 * l_ce
    if arm == "M2":
        return l_m2

    if arm == "M3":
        assert teacher_logits is not None, "M3 requires teacher_logits z_T"
        l_kl = compute_masked_kl_fp32(student_logits, teacher_logits, targets, seq_mask, tau=tau)
        return (1.0 - float(alpha_kd)) * l_m2 + float(alpha_kd) * (float(tau) ** 2) * l_kl

    raise ValueError(f"Unknown EXP-032 arm: {arm!r}")


# ==============================================================================
# CRF-Aware Full-Night Logit Stitching Predictor
# ==============================================================================

class CRFLogitStitchingPredictor(LogitStitchingSlidingWindowPredictor):
    """
    Stitches pre-CRF emission logits z_S across stride=600 windows (L=1200) for T > 1200 nights.
      - For M1 (model.crf is None): probs = softmax(z_stitched), y_pred = argmax(probs)
      - For M2/M3 (model.crf is not None): runs LinearChainCRF4 once on the full-night
        stitched sequence:
          * probs  = forward_backward_marginals(z_stitched)  [for prob_ensemble]
          * y_pred = viterbi_decode(z_stitched)              [for single-checkpoint hard hypnogram]
    """

    @torch.no_grad()
    def predict_night(self, record: Dict[str, Any]) -> Dict[str, Any]:
        base = super().predict_night(record)
        crf = getattr(self.model, "crf", None)
        if crf is None:
            return base

        stitched_logits = base["logits"]  # (N, 4) float64 numpy
        z_t = torch.from_numpy(stitched_logits.astype(np.float32)).unsqueeze(0).to(self.device)
        probs = crf.forward_backward_marginals(z_t).squeeze(0).cpu().numpy()
        preds = crf.viterbi_decode(z_t).squeeze(0).cpu().numpy()

        base["probs"] = probs
        base["y_pred"] = preds
        return base
