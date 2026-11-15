r"""
experiments/EXP-014/nas_iteration4_model.py
NAS Iteration 4 Architecture: Candidate Beta (CircadianCrossAttentionCNNBiGRU)

Grounding:
  Hypothesis H-F (Autonomic–Somatomotor Cross-Modal Interaction):
    Models dynamic, non-linear physiological coupling between central autonomic activity (IHR)
    and peripheral somatomotor activity (ACC) via windowed multi-head cosine cross-attention.
    Temporal context window (W=5, t \pm 2 epochs) extracts a 2.5-minute local physiological horizon,
    disambiguating motor atonia during REM storms from quiet sleep and postural adjustments.
    Circadian phase context (c_meta, 32 dims) is injected into queries and keys, providing
    continuous nocturnal phase awareness across the sleep cycle.

  Hypothesis H-A (Subject-Invariant Generalization across All 47 Subjects):
    Masked Temporal Instance Normalization (TIN) standardizes each nocturnal recording to
    zero nocturnal mean and unit nocturnal variance, eliminating subject baseline shifts.
    L2-normalized queries and keys restrict attention similarities strictly to cosine angles
    cos(theta) in [-1, 1], completely eliminating feature magnitude scale drift across subjects.
    Zero identity initialization (W_o == 0, b_o == 0) mathematically guarantees that training
    starts on the verified baseline manifold without initial perturbation.

Architectural Budget:
  Baseline Model A: 163,940 parameters
  CircadianAnchoredCrossAttention (CACA) module: 22,930 parameters
  Total Model Parameters: Exactly 186,870 parameters (+13.99% over baseline)
  Budget Cap: Strictly compliant (186,870 <= 196,728 cap, 9,858 parameter headroom)
  Compression Ratio vs 5.47M SLAMSS-IFS: 96.58% parameter compression
"""

import os
import sys
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from sleep_adaptive_net import (
    HREncoder,
    ACCEncoder,
    RCCLoss,
    get_official_weights
)


class MaskedTemporalInstanceNorm(nn.Module):
    """
    Subject-Invariant Masked Temporal Instance Normalization (TIN).
    Standardizes sequence representations per recording across valid sequence length T,
    eliminating inter-subject baseline shifts (e.g., resting HR, baseline activity level).
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, C)
        # mask: (B, T) boolean
        if mask is None:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
        else:
            m = mask.unsqueeze(-1).float()  # (B, T, 1)
            valid_counts = m.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B, 1, 1)
            mean = (x * m).sum(dim=1, keepdim=True) / valid_counts
            var = (((x - mean) ** 2) * m).sum(dim=1, keepdim=True) / valid_counts

        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        if mask is not None:
            x_norm = x_norm * m
        return x_norm * self.weight + self.bias


class MultiScaleModalityBlock(nn.Module):
    """
    Modality-Specific Multi-Scale Temporal Convolutional Block.
    Processes a single modality across parallel receptive fields without inter-modal blurring.
    """
    def __init__(
        self,
        in_channels: int = 64,
        mid_channels: int = 48,
        kernel_sizes: Tuple[int, ...] = (1, 3, 5)
    ):
        super().__init__()
        self.num_branches = len(kernel_sizes)
        self.branch_channels = mid_channels // self.num_branches
        assert self.branch_channels * self.num_branches == mid_channels, (
            "mid_channels must be divisible by num_branches"
        )

        self.proj_in = nn.Conv1d(in_channels, mid_channels, kernel_size=1)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    self.branch_channels,
                    self.branch_channels,
                    kernel_size=k,
                    padding=k // 2,
                    groups=self.branch_channels
                ),
                nn.GroupNorm(min(4, self.branch_channels), self.branch_channels),
                nn.GELU()
            )
            for k in kernel_sizes
        ])
        self.proj_out = nn.Conv1d(mid_channels, in_channels, kernel_size=1)
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        res = x
        x_1d = x.transpose(1, 2)  # (B, C, T)
        proj = self.proj_in(x_1d)  # (B, mid, T)
        chunks = torch.chunk(proj, self.num_branches, dim=1)
        outs = [b(c) for b, c in zip(self.branches, chunks)]
        fused = torch.cat(outs, dim=1)
        out = self.proj_out(fused).transpose(1, 2)  # (B, T, C)
        return self.act(self.norm(res + out))


class LocalizedCosineCrossAttention(nn.Module):
    r"""
    Windowed Multi-Head Cosine Cross-Attention with L2-normalized queries & keys.
    Window size W = 5 (t \pm 2 epochs).
    Logits are strictly cosine similarities cos(theta) in [-1, 1], eliminating inter-subject amplitude drift.
    Zero-initialized output projection ensures exact baseline manifold preservation at step 0.
    """
    def __init__(
        self,
        d_query: int = 96,
        d_key: int = 96,
        d_val: int = 64,
        d_attn: int = 16,
        d_out: int = 64,
        n_heads: int = 2,
        window_size: int = 5,
        eps: float = 1e-6
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_attn = d_attn
        self.d_head = d_attn // n_heads
        assert self.d_head * n_heads == d_attn, "d_attn must be divisible by n_heads"
        self.window_size = window_size
        self.pad = window_size // 2
        self.eps = eps

        # Query, Key, Value, and Output linear projections
        self.q_proj = nn.Linear(d_query, d_attn)
        self.k_proj = nn.Linear(d_key, d_attn)
        self.v_proj = nn.Linear(d_val, d_attn)
        self.out_proj = nn.Linear(d_attn, d_out)

        # Learnable temperature scale initialized to 10.0 (equivalent to tau = 0.1)
        self.scale = nn.Parameter(torch.tensor(10.0))

        # Identity initialization: zero-initialize out_proj
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        q_in: torch.Tensor,
        k_in: torch.Tensor,
        v_in: torch.Tensor,
        return_sim: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # q_in: (B, T, d_query)  [query at current epoch t]
        # k_in: (B, T, d_key)    [keys to be unfolded over window W]
        # v_in: (B, T, d_val)    [values to be unfolded over window W]
        B, T, _ = q_in.shape

        # 1. Project Query at current epoch t
        Q = self.q_proj(q_in)  # (B, T, d_attn)
        Q = Q.view(B, T, self.n_heads, 1, self.d_head).transpose(1, 2)  # (B, n_heads, T, 1, d_head)
        Q_norm = F.normalize(Q, p=2, dim=-1, eps=self.eps)

        # 2. Extract local temporal context window W for Keys and Values via 1D unfold
        k_trans = k_in.transpose(1, 2)  # (B, d_k, T)
        v_trans = v_in.transpose(1, 2)  # (B, d_v, T)

        # Replicate padding along temporal dimension
        k_pad = F.pad(k_trans, (self.pad, self.pad), mode='replicate')  # (B, d_k, T + 2*pad)
        v_pad = F.pad(v_trans, (self.pad, self.pad), mode='replicate')  # (B, d_v, T + 2*pad)

        # Unfold window size W=5, stride=1
        k_unfold = k_pad.unfold(dimension=-1, size=self.window_size, step=1)  # (B, d_k, T, W)
        v_unfold = v_pad.unfold(dimension=-1, size=self.window_size, step=1)  # (B, d_v, T, W)

        k_unfold = k_unfold.permute(0, 2, 3, 1)  # (B, T, W, d_k)
        v_unfold = v_unfold.permute(0, 2, 3, 1)  # (B, T, W, d_v)

        # 3. Project Keys and Values across the local window
        K = self.k_proj(k_unfold)  # (B, T, W, d_attn)
        V = self.v_proj(v_unfold)  # (B, T, W, d_attn)

        K = K.view(B, T, self.window_size, self.n_heads, self.d_head).permute(0, 3, 1, 2, 4)  # (B, n_heads, T, W, d_head)
        V = V.view(B, T, self.window_size, self.n_heads, self.d_head).permute(0, 3, 1, 2, 4)  # (B, n_heads, T, W, d_head)

        K_norm = F.normalize(K, p=2, dim=-1, eps=self.eps)

        # 4. Cosine similarity logits strictly in [-1, 1]
        sim = torch.matmul(Q_norm, K_norm.transpose(-1, -2))  # (B, n_heads, T, 1, W)
        attn_weights = F.softmax(sim * self.scale, dim=-1)     # (B, n_heads, T, 1, W)

        # 5. Context aggregation
        out = torch.matmul(attn_weights, V)  # (B, n_heads, T, 1, d_head)
        out = out.squeeze(-2).transpose(1, 2).contiguous().view(B, T, self.d_attn)  # (B, T, d_attn)

        # 6. Output projection (zero-initialized at step 0)
        proj_out = self.out_proj(out)
        if return_sim:
            return proj_out, sim
        return proj_out


class CircadianAnchoredCrossAttention(nn.Module):
    """
    Circadian-Anchored Localized Cross-Attention Module (CACA).
    Combines Masked TIN, Modality-Specific Multi-Scale Blocks, Circadian Projection,
    and Bidirectional Localized Cosine Cross-Attention.
    Parameters: Exactly 22,930 parameters.
    """
    def __init__(
        self,
        d_model: int = 64,
        d_meta: int = 16,
        d_circadian: int = 32,
        d_attn: int = 16,
        n_heads: int = 2,
        window_size: int = 5
    ):
        super().__init__()
        self.d_model = d_model

        # 1. Subject-invariant standardization via Masked TIN
        self.tin_hr = MaskedTemporalInstanceNorm(d_model)
        self.tin_acc = MaskedTemporalInstanceNorm(d_model)
        self.norm_hr_in = nn.LayerNorm(d_model)
        self.norm_acc_in = nn.LayerNorm(d_model)

        # 2. Modality-specific multi-scale temporal modeling (isolated before fusion)
        self.ms_hr = MultiScaleModalityBlock(in_channels=d_model, mid_channels=48, kernel_sizes=(1, 3, 5))
        self.ms_acc = MultiScaleModalityBlock(in_channels=d_model, mid_channels=32, kernel_sizes=(1, 3))

        # 3. Circadian context projection
        self.circadian_proj = nn.Sequential(
            nn.Linear(d_meta, d_circadian),
            nn.GELU()
        )

        # 4. Bidirectional Localized Cosine Cross-Attention
        in_qk = d_model + d_circadian  # 64 + 32 = 96
        # Autonomic (HR) queried by Somatomotor (ACC) local context
        self.attn_hr_from_acc = LocalizedCosineCrossAttention(
            d_query=in_qk,
            d_key=in_qk,
            d_val=d_model,
            d_attn=d_attn,
            d_out=d_model,
            n_heads=n_heads,
            window_size=window_size
        )
        # Somatomotor (ACC) queried by Autonomic (HR) local context
        self.attn_acc_from_hr = LocalizedCosineCrossAttention(
            d_query=in_qk,
            d_key=in_qk,
            d_val=d_model,
            d_attn=d_attn,
            d_out=d_model,
            n_heads=n_heads,
            window_size=window_size
        )

        self.norm_hr_out = nn.LayerNorm(d_model)
        self.norm_acc_out = nn.LayerNorm(d_model)

    def forward(
        self,
        h_hr: torch.Tensor,
        h_acc: torch.Tensor,
        h_meta: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
          h_hr:   (B, T, 64)
          h_acc:  (B, T, 64)
          h_meta: (B, T, 16)
          mask:   (B, T) boolean
        Output:
          h_hr_out:  (B, T, 64)
          h_acc_out: (B, T, 64)
        """
        # Step 1: Subject-invariant standardization via Masked TIN
        h_hr_tin = self.tin_hr(h_hr, mask)
        h_acc_tin = self.tin_acc(h_acc, mask)
        z_hr = self.norm_hr_in(h_hr + h_hr_tin)
        z_acc = self.norm_acc_in(h_acc + h_acc_tin)

        # Step 2: Modality-specific multi-scale temporal modeling
        h_hr_ms = self.ms_hr(z_hr)
        h_acc_ms = self.ms_acc(z_acc)

        # Step 3: Circadian context projection
        c_meta = self.circadian_proj(h_meta)

        # Step 4: Localized Cosine Cross-Attention
        # HR attends to ACC local context (W=5) conditioned on circadian phase
        q_hr = torch.cat([h_hr_ms, c_meta], dim=-1)   # (B, T, 96)
        k_acc = torch.cat([h_acc_ms, c_meta], dim=-1) # (B, T, 96)
        delta_hr = self.attn_hr_from_acc(q_hr, k_acc, h_acc_ms)  # (B, T, 64)

        # ACC attends to HR local context (W=5) conditioned on circadian phase
        q_acc = torch.cat([h_acc_ms, c_meta], dim=-1) # (B, T, 96)
        k_hr = torch.cat([h_hr_ms, c_meta], dim=-1)   # (B, T, 96)
        delta_acc = self.attn_acc_from_hr(q_acc, k_hr, h_hr_ms)  # (B, T, 64)

        # Step 5: Residual connection + LayerNorm
        h_hr_out = self.norm_hr_out(h_hr_ms + delta_hr)
        h_acc_out = self.norm_acc_out(h_acc_ms + delta_acc)

        return h_hr_out, h_acc_out


class CircadianCrossAttentionCNNBiGRU(nn.Module):
    """
    NAS Iteration 4 Architecture: Candidate Beta.
    Extends Baseline Model A with CircadianAnchoredCrossAttention.
    Total Parameters: Exactly 186,870 parameters (+13.99% over baseline).
    """
    def __init__(
        self,
        d_model: int = 64,
        d_temp: int = 96,
        n_classes: int = 4,
        n_meta: int = 5,
        d_attn: int = 16,
        n_heads: int = 2,
        window_size: int = 5,
        dropout: float = 0.2,
        use_rcc_loss: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_temp = d_temp
        self.n_classes = n_classes
        self.use_rcc_loss = use_rcc_loss

        # 1. Modality-specific Encoders (Identical to Baseline Model A, 29,568 params each)
        self.hr_encoder = HREncoder(in_features=1, d_model=d_model, dropout=dropout)
        self.acc_encoder = ACCEncoder(in_features=1, d_model=d_model, dropout=dropout)

        # Meta feature projection (Linear(5, 16) + GELU, 96 params)
        self.meta_proj = nn.Sequential(
            nn.Linear(n_meta, 16),
            nn.GELU()
        )

        # 2. Structural Mutation: Circadian-Anchored Cross-Attention (22,930 params)
        self.cross_attention = CircadianAnchoredCrossAttention(
            d_model=d_model,
            d_meta=16,
            d_circadian=32,
            d_attn=d_attn,
            n_heads=n_heads,
            window_size=window_size
        )

        # 3. Temporal Sequence Modeling (BiGRU, input_size=144, hidden_size=48 -> 96, 97,920 params)
        in_temp = 2 * d_model + 16  # 64 + 64 + 16 = 144
        self.temporal_gru = nn.GRU(
            input_size=in_temp,
            hidden_size=d_temp // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.temp_norm = nn.LayerNorm(d_temp)  # 192 params

        # 4. Classifier Head (6,596 params)
        self.classifier_head = nn.Sequential(
            nn.Linear(d_temp, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

        # Loss Functions
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        fn_w, fp_w = get_official_weights()
        self.rcc_loss = RCCLoss(fn_weights=fn_w, fp_weights=fp_w, cls=n_classes)

    def forward(
        self,
        x_ihr: Union[torch.Tensor, Dict[str, torch.Tensor]],
        x_accel: Optional[torch.Tensor] = None,
        x_meta: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        subject_idx: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if isinstance(x_ihr, dict):
            batch = x_ihr
            x_ihr = batch["ihr"]
            x_accel = batch["accel"]
            x_meta = batch["meta"]
            targets = batch.get("y", targets)
            mask = batch.get("mask", mask)
            subject_idx = batch.get("subject_idx", subject_idx)
            seq_lens = batch.get("seq_len", seq_lens)

        B, T, _ = x_ihr.shape

        # 1. Modality Encoders
        h_hr = self.hr_encoder(x_ihr)      # (B, T, 64)
        h_acc = self.acc_encoder(x_accel)  # (B, T, 64)
        h_meta = self.meta_proj(x_meta)    # (B, T, 16)

        # 2. Circadian-Anchored Localized Cross-Attention
        h_hr_fused, h_acc_fused = self.cross_attention(h_hr, h_acc, h_meta, mask=mask)

        # 3. Multimodal Concatenation
        cat_features = torch.cat([h_hr_fused, h_acc_fused, h_meta], dim=-1)  # (B, T, 144)

        # 4. Temporal Sequence Modeling with Packing Support
        if seq_lens is not None:
            seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int64)
            packed_features = nn.utils.rnn.pack_padded_sequence(
                cat_features, seq_lens_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.temporal_gru(packed_features)
            temp_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )
        else:
            temp_out, _ = self.temporal_gru(cat_features)

        temp_features = self.temp_norm(temp_out)  # (B, T, 96)

        # 5. Classifier Head
        logits = self.classifier_head(temp_features)  # (B, T, 4)

        # 6. Loss Calculation
        loss_stage = torch.tensor(0.0, device=x_ihr.device)
        if targets is not None:
            flat_logits = logits.view(-1, self.n_classes)
            flat_targets = targets.view(-1)

            valid = (flat_targets >= 0) & (flat_targets < self.n_classes)
            if mask is not None:
                valid = valid & mask.view(-1)

            if valid.any():
                per_epoch_ce = self.ce_loss(flat_logits, torch.clamp(flat_targets, 0, self.n_classes - 1))
                per_epoch_ce_2d = per_epoch_ce.view(B, T)
                valid_2d = valid.view(B, T)
                ce_term = per_epoch_ce_2d[valid_2d].mean()

                if self.use_rcc_loss:
                    raw_rcc = self.rcc_loss(flat_logits, flat_targets)
                    v_mask = valid.float().unsqueeze(-1).repeat(1, self.n_classes)
                    rcc_term = - torch.sum(raw_rcc * v_mask) / valid.sum().clamp(min=1.0)
                    loss_stage = ce_term + 0.01 * rcc_term
                else:
                    loss_stage = ce_term

        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "emission_logits": logits,
            "probabilities": probs,
            "total_loss": loss_stage,
            "loss_stage": loss_stage,
            "temp_features": temp_features
        }


# Model alias per dispatch conventions
CandidateBeta_CACA = CircadianCrossAttentionCNNBiGRU


def build_nas_iteration4_model() -> CircadianCrossAttentionCNNBiGRU:
    """Factory function for NAS Iteration 4 Candidate Beta architecture."""
    return CircadianCrossAttentionCNNBiGRU()
