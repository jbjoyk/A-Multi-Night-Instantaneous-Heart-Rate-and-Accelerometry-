r"""
.agents/teamwork/teamwork_explorer_r5_1/proposed_nas_iteration5_model.py
NAS Iteration 5 Architecture: Candidate Gamma (InterEpochSkipAttentionCNNBiGRU)

Grounding:
  Hypothesis H-F (Autonomic–Somatomotor Cross-Modal Interaction & Direct Skip Highway):
    - Overcomes the 2-layer BiGRU information bottleneck by creating a direct skip highway
      from the 144-dimensional intra-epoch representation (64 IHR + 64 ACC + 16 Meta)
      directly into the classification head: Linear(96 [BiGRU] + 48 [Context] + 144 [Skip] = 288 -> 64).
    - Preserves high-frequency autonomic-somatomotor features (e.g. instantaneous motor stillness
      coinciding with sympathetic cardiac surges during REM storms) that were previously
      destroyed or attenuated when forced through the 48-dim recurrent cell over 600 time steps.
    - Provides a direct 1-step backpropagation gradient path (\partial L / \partial e_intra)
      from the classification loss directly into hr_encoder and acc_encoder, completely bypassing
      the 600-step BPTT Jacobian of the recurrent network.

  Hypothesis H-A (Subject-Invariant Generalization & Sequence-Level Temporal Context):
    - Subject-Invariant Standardization: Masked Temporal Instance Normalization (TIN) normalizes
      nocturnal sequences per recording across valid epochs to zero nocturnal mean and unit
      variance, eliminating inter-subject baseline shifts in resting heart rate and accelerometer sensitivity.
    - Inter-Epoch Sequence-Level Multi-Head Attention (IESA): Implements 4-head scaled dot-product
      attention across the entire sequence length T (up to 600–1200 epochs). Every epoch t has direct O(1)
      receptive field access to any epoch across the night, allowing the network to recognize
      90-minute ultradian sleep cycle dynamics (180 epochs) and resolve REM vs Light sleep confusion.
    - Sequence Padding Protection: Strict boolean attention key-masking (-1e9 prior to softmax)
      guarantees zero attention weight to padded epochs, preventing artifact leakage on variable-length recordings.
    - Step-0 Stability: Zero-identity initialization on the attention output projection (out_proj == 0)
      mathematically guarantees that at step 0, attn_context is identically zero, ensuring the initial forward
      pass operates stably on the verified baseline manifold without random perturbation.

Architectural Budget & Compression:
  Baseline Model A: 163,940 parameters
  Candidate Gamma (IESA): Exactly 193,156 parameters (+17.82% over baseline)
  Budget Cap: Strictly compliant (193,156 <= 196,728 cap; 3,572 parameter headroom)
  Compression Ratio vs 5.47M published SLAMSS-IFS paper benchmark: 96.47% parameter compression.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from exp013_models import ConventionalCNNBiGRU
from sleep_adaptive_net import (
    HREncoder,
    ACCEncoder,
    RCCLoss,
    get_official_weights
)


class MaskedTemporalInstanceNorm(nn.Module):
    """
    Subject-Invariant Masked Temporal Instance Normalization (TIN).
    Standardizes nocturnal sequence representations per recording across valid sequence length T,
    eliminating inter-subject baseline shifts (e.g. resting HR, accelerometer baseline sensitivity).
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


class InterEpochSequenceAttention(nn.Module):
    r"""
    Inter-Epoch Sequence-Level Multi-Head Scaled Dot-Product Attention (IESA).
    Enables direct O(1) receptive field access across all epochs in the nocturnal recording (T = 600 epochs),
    overcoming the exponential gradient and information decay of 2-layer BiGRU over 90-minute ultradian cycles.

    Features:
      - 4 attention heads specialized in micro-continuity, stage transitions, ultradian cycles, and sleep anchors.
      - Exact sequence padding key-masking (-1e9 before softmax) to eliminate padded token leakage.
      - Zero-identity output projection initialization (out_proj == 0) for step-0 stability.
    """
    def __init__(
        self,
        d_in: int = 96,
        d_attn: int = 48,
        n_heads: int = 4,
        dropout: float = 0.2,
        zero_init: bool = True
    ):
        super().__init__()
        self.d_in = d_in
        self.d_attn = d_attn
        self.n_heads = n_heads
        self.d_k = d_attn // n_heads
        assert self.d_k * n_heads == d_attn, f"d_attn ({d_attn}) must be divisible by n_heads ({n_heads})"

        self.q_proj = nn.Linear(d_in, d_attn)
        self.k_proj = nn.Linear(d_in, d_attn)
        self.v_proj = nn.Linear(d_in, d_attn)
        self.out_proj = nn.Linear(d_attn, d_attn)
        self.norm = nn.LayerNorm(d_attn)
        self.dropout = nn.Dropout(dropout)

        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
          x: (B, T, d_in) sequence features from BiGRU
          mask: (B, T) boolean mask where True indicates valid epoch, False is padded
        Output:
          attn_context: (B, T, d_attn) global sequence-level context
          attn_weights: (B, n_heads, T, T) attention alignment matrix
        """
        B, T, _ = x.shape

        # Linear projections & head splitting
        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, n_heads, T, d_k)

        # Scaled dot-product scores: (B, n_heads, T, T)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        # Mask invalid key positions (padding protection)
        if mask is not None:
            key_mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            scores = scores.masked_fill(~key_mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context aggregation
        context = torch.matmul(attn_weights, v)  # (B, n_heads, T, d_k)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_attn)  # (B, T, d_attn)

        # Output projection and normalization
        out = self.norm(self.out_proj(context))

        # Zero-out context at padded positions
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()

        return out, attn_weights


class InterEpochSkipAttentionCNNBiGRU(nn.Module):
    r"""
    NAS Iteration 5 Candidate Gamma Architecture.
    Combines:
      1. Modality-Specific Encoders (HREncoder 64, ACCEncoder 64, MetaProj 16)
      2. Subject-Invariant Masked Temporal Instance Normalization (TIN)
      3. 2-layer Bidirectional GRU (hidden 48 -> 96) with packed sequence protection
      4. Inter-Epoch Sequence Multi-Head Attention (IESA, d_attn=48, 4 heads)
      5. Direct Inter-Epoch Feature Skip Connection (144 dims) bypassing recurrent model
      6. Multi-Stream Classification Head: Linear(96 [GRU] + 48 [Attn] + 144 [Skip] = 288 -> 64 -> 4)

    Total Parameters: Exactly 193,156 parameters (96.47% compression vs 5.47M SLAMSS-IFS).
    """
    def __init__(
        self,
        d_model: int = 64,
        d_temp: int = 96,
        d_attn: int = 48,
        n_heads: int = 4,
        n_classes: int = 4,
        n_meta: int = 5,
        dropout: float = 0.2,
        concat_context: bool = True,
        zero_init: bool = True,
        use_rcc_loss: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_temp = d_temp
        self.d_attn = d_attn
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.concat_context = concat_context
        self.use_rcc_loss = use_rcc_loss

        # 1. Modality-specific Encoders (Identical to Baseline Model A, 29,568 params each)
        base = ConventionalCNNBiGRU(
            d_model=d_model,
            d_temp=d_temp,
            n_classes=n_classes,
            n_meta=n_meta,
            dropout=dropout
        )
        self.hr_encoder = base.hr_encoder
        self.acc_encoder = base.acc_encoder

        # Meta feature projection (Linear(5, 16) + GELU, 96 params)
        self.meta_proj = base.meta_proj

        # 2. Subject-invariant Masked Temporal Instance Normalization (512 params)
        self.tin_hr = MaskedTemporalInstanceNorm(d_model)
        self.tin_acc = MaskedTemporalInstanceNorm(d_model)
        self.norm_hr = nn.LayerNorm(d_model)
        self.norm_acc = nn.LayerNorm(d_model)

        # 3. Recurrent Temporal Sequence Modeling (97,920 params)
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

        # 4. Inter-Epoch Sequence Multi-Head Attention (16,416 params)
        self.inter_epoch_attn = InterEpochSequenceAttention(
            d_in=d_temp,
            d_attn=d_attn,
            n_heads=n_heads,
            dropout=dropout,
            zero_init=zero_init
        )

        # 5. Multi-Stream Classifier Head (18,884 params)
        # Combines BiGRU state (96) + Attention context (48) + Direct Skip features (144) = 288 dims
        in_clf = d_temp + (d_attn if concat_context else 0) + in_temp
        self.classifier_head = nn.Sequential(
            nn.Linear(in_clf, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

        # Stage loss functions
        self.ce_loss = base.ce_loss
        self.rcc_loss = base.rcc_loss

    def forward(
        self,
        x_ihr: Union[torch.Tensor, Dict[str, torch.Tensor]],
        x_accel: Optional[torch.Tensor] = None,
        x_meta: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        subject_idx: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass supporting both tensor arguments and dictionary batch input.
        Matches SleepAdaptiveNet / Model A return dictionary interface.
        """
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

        # Step 1: Modality-Specific Feature Extraction
        h_hr = self.hr_encoder(x_ihr)      # (B, T, 64)
        h_acc = self.acc_encoder(x_accel)  # (B, T, 64)
        h_meta = self.meta_proj(x_meta)    # (B, T, 16)

        # Step 2: Subject-Invariant Standardization via Masked TIN
        h_hr_tin = self.tin_hr(h_hr, mask)
        h_acc_tin = self.tin_acc(h_acc, mask)
        z_hr = self.norm_hr(h_hr + h_hr_tin)
        z_acc = self.norm_acc(h_acc + h_acc_tin)

        # Intra-epoch fused multimodal representation (144 dims)
        e_intra = torch.cat([z_hr, z_acc, h_meta], dim=-1)  # (B, T, 144)

        # Step 3: Recurrent Temporal Sequence Modeling with Packing Support
        if seq_lens is not None:
            seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int64)
            packed = nn.utils.rnn.pack_padded_sequence(
                e_intra, seq_lens_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.temporal_gru(packed)
            temp_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )
        else:
            temp_out, _ = self.temporal_gru(e_intra)

        temp_features = self.temp_norm(temp_out)  # (B, T, 96)

        # Step 4: Inter-Epoch Sequence-Level Attention Context
        attn_context, attn_weights = self.inter_epoch_attn(temp_features, mask=mask)  # (B, T, 48)

        # Step 5: Multi-Stream Skip Integration
        # Direct skip connection bypassing recurrent model straight to classifier head
        if self.concat_context:
            combined = torch.cat([temp_features, attn_context, e_intra], dim=-1)  # (B, T, 288)
        else:
            combined = torch.cat([temp_features + attn_context, e_intra], dim=-1)  # (B, T, 240)

        # Step 6: Classification Logits Emission
        logits = self.classifier_head(combined)  # (B, T, 4)

        # Step 7: Loss Calculation with RCC Loss Support
        loss_stage = torch.tensor(0.0, device=x_ihr.device)
        if targets is not None:
            flat_logits = logits.view(-1, self.n_classes)
            flat_targets = targets.view(-1)

            valid = (flat_targets >= 0) & (flat_targets < self.n_classes)
            if mask is not None:
                valid = valid & mask.view(-1)

            if valid.any():
                per_epoch_ce = self.ce_loss(
                    flat_logits, torch.clamp(flat_targets, 0, self.n_classes - 1)
                )
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
            "temp_features": temp_features,
            "attn_context": attn_context,
            "attn_weights": attn_weights,
            "skip_features": e_intra
        }


# Model alias per dispatch conventions
CandidateGamma_IESA = InterEpochSkipAttentionCNNBiGRU


def build_nas_iteration5_model(
    concat_context: bool = True,
    zero_init: bool = True
) -> InterEpochSkipAttentionCNNBiGRU:
    """Factory function for NAS Iteration 5 Candidate Gamma architecture."""
    return InterEpochSkipAttentionCNNBiGRU(
        concat_context=concat_context,
        zero_init=zero_init
    )
