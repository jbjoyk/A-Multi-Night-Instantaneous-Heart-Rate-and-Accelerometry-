"""
EXP-029 Model Definitions & Factory
File: experiments/EXP-029/exp029_models.py

Implements UltradianPyramidalCoupler and its ablation variants strictly matching
docs/designs/exp029_ultradian_pyramidal_coupler.md:
  - Arm (b) 'b_200k' / 'cnn_bigru_200k': ConventionalCNNBiGRU(d_temp=128) [223,268 params]
  - Arm (c) 'c_film' / 'film_only': 2L BiGRU (d_temp=96) + Sub-Epoch FiLM [172,260 params]
  - Arm (d) 'd_pyramid' / 'pyramid_only': 2-Scale Pyramidal BiGRU, no FiLM [182,468 params]
  - Arm (e) 'e_full' / 'full_v5': Full Ultradian-Pyramidal-Coupler [190,788 params]

Preserves the exact 14-key forward() return dict, seq_lens=None fallback,
zero-init FiLM residual gates (weight=0, bias=0), dropout=0.0 on 1-layer BiGRUs,
and FP32 loss calculation outside torch.amp.autocast.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

EXP029_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP029_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from exp013_models import ConventionalCNNBiGRU
from sleep_adaptive_net import RCCLoss, TemporalConvBlock, get_official_weights


ARM_ALIASES: Dict[str, str] = {
    "b_200k": "cnn_bigru_200k",
    "cnn_bigru_200k": "cnn_bigru_200k",
    "c_film": "film_only",
    "film_only": "film_only",
    "d_pyramid": "pyramid_only",
    "pyramid_only": "pyramid_only",
    "e_full": "full_v5",
    "full_v5": "full_v5",
}


class PrePoolEncoder(nn.Module):
    """
    1D CNN encoder exposing pre-pooled (B*T, 64, 7) ~0.233 Hz sub-epoch maps
    before AdaptiveAvgPool1d(1) -> Dropout -> Linear(64, 64) -> LayerNorm(64).
    Preserves exact 29,568 parameter count of HREncoder / ACCEncoder.
    """

    def __init__(self, in_features: int = 1, d_model: int = 64, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.prepool = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool1d(2),  # 30 -> 15
            TemporalConvBlock(32, 64, kernel_size=5),
            nn.MaxPool1d(2),  # 15 -> 7
            TemporalConvBlock(64, d_model, kernel_size=3),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward_prepool(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sub-epoch feature map of shape (B*T, d_model, 7)."""
        B, T, L = x.shape
        x_flat = x.reshape(B * T, 1, L)
        return self.prepool(x_flat)

    def pool_and_project(self, feat_seq: torch.Tensor, B: int, T: int) -> torch.Tensor:
        """Pools (B*T, d_model, 7) -> (B, T, d_model)."""
        pooled = self.dropout(self.pool(feat_seq).squeeze(-1))
        out = self.norm(self.proj(pooled))
        return out.view(B, T, self.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return self.pool_and_project(self.forward_prepool(x), B, T)


class IntraEpochFiLM(nn.Module):
    """
    Sub-epoch (~0.233 Hz) pre-pooling ACC -> HR residual FiLM conditioner.
    Zero-initialized Conv1d(64, 128, kernel_size=1) guarantees exact identity
    (gamma=1.0, beta=0.0) at initialization step 0.
    """

    def __init__(self, d_model: int = 64):
        super().__init__()
        self.film_proj = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, h_hr_seq: torch.Tensor, h_acc_seq: torch.Tensor) -> torch.Tensor:
        d_gamma, d_beta = torch.chunk(self.film_proj(h_acc_seq), 2, dim=1)
        gamma = 1.0 + torch.tanh(d_gamma)
        beta = d_beta
        return gamma * h_hr_seq + beta


class MaskWeightedPool1d(nn.Module):
    """
    4x mask-weighted temporal average pooling over valid epochs [0, seq_len).
    Right-pads with zeros when T % 4 != 0 before reshaping.
    """

    def __init__(self, scale_factor: int = 4):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C)
            valid_mask: (B, T, 1) float tensor with 1.0 for valid epochs and 0.0 for padded.
        Returns:
            (B, ceil(T / scale_factor), C)
        """
        B, T, C = x.shape
        k = self.scale_factor
        x_masked = x * valid_mask
        pad_t = (k - (T % k)) % k
        if pad_t > 0:
            x_masked = F.pad(x_masked, (0, 0, 0, pad_t), value=0.0)
            m_pad = F.pad(valid_mask, (0, 0, 0, pad_t), value=0.0)
        else:
            m_pad = valid_mask

        T_c = (T + pad_t) // k
        x_reshaped = x_masked.view(B, T_c, k, C)
        m_reshaped = m_pad.view(B, T_c, k, 1)
        denom = m_reshaped.sum(dim=2).clamp(min=1.0)
        return x_reshaped.sum(dim=2) / denom


class UltradianPyramidalCoupler(nn.Module):
    """
    EXP-029 Ultradian-Pyramidal-Coupler (v5) and its structural ablations:
      - 'film_only': 2-layer BiGRU (d_temp=96) + IntraEpochFiLM [172,260 params]
      - 'pyramid_only': 2-scale pyramidal BiGRU (30s fine + 2-min coarse), no FiLM [182,468 params]
      - 'full_v5': IntraEpochFiLM + 2-scale pyramidal BiGRU [190,788 params]
    """

    def __init__(
        self,
        arm_id: str = "full_v5",
        d_model: int = 64,
        d_temp: int = 96,
        n_classes: int = 4,
        n_meta: int = 5,
        dropout: float = 0.2,
        use_rcc_loss: bool = True,
    ):
        super().__init__()
        canonical = ARM_ALIASES.get(arm_id, arm_id)
        if canonical not in {"film_only", "pyramid_only", "full_v5"}:
            raise ValueError(
                f"UltradianPyramidalCoupler arm_id must be one of "
                f"{{'film_only', 'pyramid_only', 'full_v5'}} (got '{arm_id}')"
            )

        self.arm_id = canonical
        self.d_model = d_model
        self.d_temp = d_temp
        self.n_classes = n_classes
        self.use_rcc_loss = use_rcc_loss
        self.use_film = canonical in {"film_only", "full_v5"}
        self.use_pyramid = canonical in {"pyramid_only", "full_v5"}

        # 1. Sub-Epoch Pre-Pooling Encoders + Meta Projection
        self.hr_encoder = PrePoolEncoder(in_features=1, d_model=d_model, dropout=dropout)
        self.acc_encoder = PrePoolEncoder(in_features=1, d_model=d_model, dropout=dropout)
        self.meta_proj = nn.Sequential(
            nn.Linear(n_meta, 16),
            nn.GELU(),
        )

        # 2. Module 1: Optional Sub-Epoch ACC -> HR Residual FiLM
        if self.use_film:
            self.intra_film = IntraEpochFiLM(d_model=d_model)

        # 3. Module 2: Temporal Sequence Backbone
        in_temp = 2 * d_model + 16  # 144
        if self.use_pyramid:
            self.fine_gru = nn.GRU(
                input_size=in_temp,
                hidden_size=d_temp // 2,  # 48/dir -> 96
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0,
            )
            self.mask_pool = MaskWeightedPool1d(scale_factor=4)
            self.coarse_gru = nn.GRU(
                input_size=d_temp,
                hidden_size=d_temp // 2,  # 48/dir -> 96
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0,
            )
            self.fusion_proj = nn.Linear(2 * d_temp, d_temp)  # 192 -> 96
        else:
            self.temporal_gru = nn.GRU(
                input_size=in_temp,
                hidden_size=d_temp // 2,  # 48/dir -> 96
                num_layers=2,
                batch_first=True,
                bidirectional=True,
                dropout=dropout,
            )

        self.temp_norm = nn.LayerNorm(d_temp)

        # 4. Classifier Head (identical to ConventionalCNNBiGRU)
        self.classifier_head = nn.Sequential(
            nn.Linear(d_temp, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

        # 5. Primary Stage Loss Functions
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        fn_w, fp_w = get_official_weights()
        self.rcc_loss = RCCLoss(fn_weights=fn_w, fp_weights=fp_w, cls=n_classes)

    def _compute_loss_fp32(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Computes CE + 0.01 * RCCLoss strictly in FP32 outside AMP autocast."""
        B, T, _ = logits.shape
        device_type = logits.device.type if logits.device.type in {"cuda", "cpu"} else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            logits_fp32 = logits.float()
            flat_logits = logits_fp32.reshape(-1, self.n_classes)
            flat_targets = targets.reshape(-1)

            valid = (flat_targets >= 0) & (flat_targets < self.n_classes)
            if mask is not None:
                valid = valid & mask.reshape(-1).bool()

            if not valid.any():
                return torch.tensor(0.0, device=logits.device, dtype=torch.float32)

            per_epoch_ce = self.ce_loss(
                flat_logits, torch.clamp(flat_targets, 0, self.n_classes - 1)
            )
            ce_term = per_epoch_ce.view(B, T)[valid.view(B, T)].mean()

            if self.use_rcc_loss:
                raw_rcc = self.rcc_loss(flat_logits, flat_targets)
                v_mask = valid.float().unsqueeze(-1).expand(-1, self.n_classes)
                rcc_term = -torch.sum(raw_rcc * v_mask) / valid.sum().clamp(min=1.0)
                return ce_term + 0.01 * rcc_term
            return ce_term

    def forward(
        self,
        x_ihr: torch.Tensor,
        x_accel: torch.Tensor,
        x_meta: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        subject_idx: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Forward pass preserving the exact 14-key return dictionary of ConventionalCNNBiGRU.
        """
        B, T, _ = x_ihr.shape

        # 1. Sub-Epoch Pre-Pooling Encoding & Optional FiLM Conditioning
        h_hr_seq = self.hr_encoder.forward_prepool(x_ihr)
        h_acc_seq = self.acc_encoder.forward_prepool(x_accel)

        if self.use_film:
            h_hr_seq = self.intra_film(h_hr_seq, h_acc_seq)

        h_hr = self.hr_encoder.pool_and_project(h_hr_seq, B, T)
        h_acc = self.acc_encoder.pool_and_project(h_acc_seq, B, T)
        h_meta = self.meta_proj(x_meta)

        cat_features = torch.cat([h_hr, h_acc, h_meta], dim=-1)  # (B, T, 144)

        # 2. Sequence Length & Valid Mask Handling (supports seq_lens=None)
        if seq_lens is None:
            seq_lens_cpu = torch.full((B,), T, dtype=torch.int64)
            valid_mask = torch.ones((B, T, 1), device=cat_features.device, dtype=cat_features.dtype)
        else:
            seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int64).clamp(min=1, max=T)
            t_idx = torch.arange(T, device=cat_features.device).unsqueeze(0)
            valid_mask = (t_idx < seq_lens_cpu.to(cat_features.device).unsqueeze(1)).unsqueeze(-1)
            valid_mask = valid_mask.to(dtype=cat_features.dtype)

        # 3. Temporal Modeling
        if self.use_pyramid:
            packed_fine = nn.utils.rnn.pack_padded_sequence(
                cat_features, seq_lens_cpu, batch_first=True, enforce_sorted=False
            )
            packed_fine_out, _ = self.fine_gru(packed_fine)
            h_fine, _ = nn.utils.rnn.pad_packed_sequence(
                packed_fine_out, batch_first=True, total_length=T
            )
            h_fine_masked = h_fine * valid_mask

            h_pool = self.mask_pool(h_fine_masked, valid_mask)
            T_c = h_pool.shape[1]
            seq_lens_coarse = ((seq_lens_cpu + 3) // 4).clamp(min=1, max=T_c)

            packed_coarse = nn.utils.rnn.pack_padded_sequence(
                h_pool, seq_lens_coarse, batch_first=True, enforce_sorted=False
            )
            packed_coarse_out, _ = self.coarse_gru(packed_coarse)
            h_coarse, _ = nn.utils.rnn.pad_packed_sequence(
                packed_coarse_out, batch_first=True, total_length=T_c
            )

            h_coarse_up = h_coarse.repeat_interleave(4, dim=1)[:, :T, :] * valid_mask
            temp_out = self.fusion_proj(torch.cat([h_fine_masked, h_coarse_up], dim=-1))
        else:
            packed = nn.utils.rnn.pack_padded_sequence(
                cat_features, seq_lens_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.temporal_gru(packed)
            temp_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )

        temp_features = self.temp_norm(temp_out)
        logits = self.classifier_head(temp_features)

        # 4. FP32 Loss Calculation
        loss_stage = torch.tensor(0.0, device=x_ihr.device, dtype=torch.float32)
        if targets is not None:
            loss_stage = self._compute_loss_fp32(logits, targets, mask)

        probs = F.softmax(logits.float(), dim=-1)

        return {
            "logits": logits,
            "emission_logits": logits,
            "probabilities": probs,
            "alpha_gate": None,
            "confidence": torch.zeros(B, T, 1, device=x_ihr.device, dtype=logits.dtype),
            "entropy": torch.zeros(B, T, 1, device=x_ihr.device, dtype=logits.dtype),
            "total_loss": loss_stage,
            "loss_stage": loss_stage,
            "loss_trans": torch.tensor(0.0, device=x_ihr.device),
            "loss_boundary": torch.tensor(0.0, device=x_ihr.device),
            "loss_adv": torch.tensor(0.0, device=x_ihr.device),
            "loss_conf": torch.tensor(0.0, device=x_ihr.device),
            "boundary_mask": None,
            "temp_features": temp_features,
        }


def build_exp029_model(
    arm_id: str,
    loss_type: str = "rccloss",
    dropout: float = 0.2,
) -> nn.Module:
    """
    Factory for EXP-029 arms:
      - 'b_200k' / 'cnn_bigru_200k': ConventionalCNNBiGRU(d_temp=128) [223,268 params]
      - 'c_film' / 'film_only': UltradianPyramidalCoupler('film_only') [172,260 params]
      - 'd_pyramid' / 'pyramid_only': UltradianPyramidalCoupler('pyramid_only') [182,468 params]
      - 'e_full' / 'full_v5': UltradianPyramidalCoupler('full_v5') [190,788 params]
    """
    canonical = ARM_ALIASES.get(arm_id)
    if canonical is None:
        raise ValueError(f"Unknown EXP-029 arm_id '{arm_id}'. Expected one of {sorted(ARM_ALIASES.keys())}.")
    if loss_type not in {"rccloss", "ce"}:
        raise ValueError(f"Unsupported loss_type '{loss_type}'. Expected 'rccloss' or 'ce'.")

    use_rcc_loss = loss_type == "rccloss"
    if canonical == "cnn_bigru_200k":
        return ConventionalCNNBiGRU(
            d_model=64,
            d_temp=128,
            n_classes=4,
            n_meta=5,
            dropout=dropout,
            use_rcc_loss=use_rcc_loss,
        )
    return UltradianPyramidalCoupler(
        arm_id=canonical,
        d_model=64,
        d_temp=96,
        n_classes=4,
        n_meta=5,
        dropout=dropout,
        use_rcc_loss=use_rcc_loss,
    )
