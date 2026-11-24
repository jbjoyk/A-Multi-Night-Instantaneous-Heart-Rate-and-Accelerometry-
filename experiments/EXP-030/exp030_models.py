"""
EXP-030 Width-Scaled ConventionalCNNBiGRU Model Factory
File: experiments/EXP-030/exp030_models.py

Implements ScaledModalityEncoder and ScaledConventionalCNNBiGRU across the
6-point capacity Pareto grid (p1_100k through p6_1051k), leaving exp013_models.py
and sleep_adaptive_net.py untouched.

At p2_164k (d_enc=64, d_temp=96, head_hidden=64), state_dict keys, shapes,
parameter count (163,940), and forward outputs are bit-for-bit identical to
ConventionalCNNBiGRU() in exp013_models.py.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

EXP030_DIR = os.path.abspath(os.path.dirname(__file__))
REPO_ROOT = os.path.abspath(os.path.join(EXP030_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from sleep_adaptive_net import RCCLoss, TemporalConvBlock, get_official_weights


EXP030_GRID_ORDER = [
    "p1_100k",
    "p2_164k",
    "p3_268k",
    "p4_404k",
    "p5_666k",
    "p6_1051k",
]

EXP030_GRID: Dict[str, Dict[str, int]] = {
    "p1_100k": {"d_enc": 48, "d_temp": 76, "head_hidden": 48, "expected_params": 98940},
    "p2_164k": {"d_enc": 64, "d_temp": 96, "head_hidden": 64, "expected_params": 163940},
    "p3_268k": {"d_enc": 80, "d_temp": 128, "head_hidden": 64, "expected_params": 268260},
    "p4_404k": {"d_enc": 96, "d_temp": 160, "head_hidden": 96, "expected_params": 403524},
    "p5_666k": {"d_enc": 128, "d_temp": 200, "head_hidden": 128, "expected_params": 665556},
    "p6_1051k": {"d_enc": 160, "d_temp": 256, "head_hidden": 128, "expected_params": 1051492},
}


class ScaledModalityEncoder(nn.Module):
    """
    Width-scaled 1D CNN modality encoder matching HREncoder / ACCEncoder
    in sleep_adaptive_net.py with c1 = d_enc // 2 and c2 = d_enc.
    """

    def __init__(self, in_features: int = 1, d_enc: int = 64, dropout: float = 0.2):
        super().__init__()
        if d_enc % 8 != 0:
            raise ValueError(f"d_enc must be divisible by 8 so c1=d_enc//2 is divisible by 4; got {d_enc}")
        self.d_enc = d_enc
        c1 = d_enc // 2
        c2 = d_enc

        self.net = nn.Sequential(
            nn.Conv1d(in_features, c1, kernel_size=5, padding=2),
            nn.GroupNorm(4, c1),
            nn.GELU(),
            nn.MaxPool1d(2),  # 30 -> 15
            TemporalConvBlock(c1, c2, kernel_size=5),
            nn.MaxPool1d(2),  # 15 -> 7
            TemporalConvBlock(c2, c2, kernel_size=3),  # in==out -> res=Identity()
            nn.AdaptiveAvgPool1d(1),  # 7 -> 1
            nn.Dropout(dropout),
        )
        self.proj = nn.Linear(c2, d_enc)
        self.norm = nn.LayerNorm(d_enc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L = x.shape
        x_flat = x.view(B * T, 1, L)
        feat = self.net(x_flat).squeeze(-1)
        feat = self.norm(self.proj(feat))
        return feat.view(B, T, self.d_enc)


class ScaledConventionalCNNBiGRU(nn.Module):
    """
    Width-scaled ConventionalCNNBiGRU preserving the exact 14-key forward()
    contract of ConventionalCNNBiGRU in exp013_models.py.
    """

    def __init__(
        self,
        d_enc: int = 64,
        d_temp: int = 96,
        head_hidden: int = 64,
        n_classes: int = 4,
        n_meta: int = 5,
        dropout: float = 0.2,
        use_rcc_loss: bool = True,
    ):
        super().__init__()
        if d_temp % 2 != 0:
            raise ValueError(f"d_temp must be even for bidirectional GRU; got {d_temp}")
        self.d_enc = d_enc
        self.d_model = d_enc
        self.d_temp = d_temp
        self.head_hidden = head_hidden
        self.n_classes = n_classes
        self.use_rcc_loss = use_rcc_loss

        self.hr_encoder = ScaledModalityEncoder(in_features=1, d_enc=d_enc, dropout=dropout)
        self.acc_encoder = ScaledModalityEncoder(in_features=1, d_enc=d_enc, dropout=dropout)

        self.meta_proj = nn.Sequential(
            nn.Linear(n_meta, 16),
            nn.GELU(),
        )

        in_temp = 2 * d_enc + 16
        self.temporal_gru = nn.GRU(
            input_size=in_temp,
            hidden_size=d_temp // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.temp_norm = nn.LayerNorm(d_temp)

        self.classifier_head = nn.Sequential(
            nn.Linear(d_temp, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, n_classes),
        )

        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        fn_w, fp_w = get_official_weights()
        self.rcc_loss = RCCLoss(fn_weights=fn_w, fp_weights=fp_w, cls=n_classes)

    def forward(
        self,
        x_ihr: torch.Tensor,
        x_accel: torch.Tensor,
        x_meta: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        subject_idx: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, _ = x_ihr.shape

        h_hr = self.hr_encoder(x_ihr)
        h_acc = self.acc_encoder(x_accel)
        h_meta = self.meta_proj(x_meta)

        cat_features = torch.cat([h_hr, h_acc, h_meta], dim=-1)

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

        temp_features = self.temp_norm(temp_out)
        logits = self.classifier_head(temp_features)

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
                ce_term = per_epoch_ce.view(B, T)[valid.view(B, T)].mean()

                if self.use_rcc_loss:
                    raw_rcc = self.rcc_loss(flat_logits, flat_targets)
                    v_mask = valid.float().unsqueeze(-1).repeat(1, self.n_classes)
                    rcc_term = -torch.sum(raw_rcc * v_mask) / valid.sum().clamp(min=1.0)
                    loss_stage = ce_term + 0.01 * rcc_term
                else:
                    loss_stage = ce_term

        probs = F.softmax(logits, dim=-1)

        return {
            "logits": logits,
            "emission_logits": logits,
            "probabilities": probs,
            "alpha_gate": None,
            "confidence": torch.zeros(B, T, 1, device=x_ihr.device),
            "entropy": torch.zeros(B, T, 1, device=x_ihr.device),
            "total_loss": loss_stage,
            "loss_stage": loss_stage,
            "loss_trans": torch.tensor(0.0, device=x_ihr.device),
            "loss_boundary": torch.tensor(0.0, device=x_ihr.device),
            "loss_adv": torch.tensor(0.0, device=x_ihr.device),
            "loss_conf": torch.tensor(0.0, device=x_ihr.device),
            "boundary_mask": None,
            "temp_features": temp_features,
        }


def build_exp030_model(tag: str, loss_type: str = "rccloss") -> nn.Module:
    """
    Factory for EXP-030 width-scaled models.
    tag must be one of EXP030_GRID keys (p1_100k .. p6_1051k).
    loss_type in {"rccloss", "ce"}.
    """
    if tag not in EXP030_GRID:
        raise ValueError(f"Unknown EXP-030 grid tag '{tag}'. Valid tags: {EXP030_GRID_ORDER}")
    if loss_type not in {"rccloss", "ce"}:
        raise ValueError(f"Invalid loss_type '{loss_type}'. Expected 'rccloss' or 'ce'.")
    cfg = EXP030_GRID[tag]
    return ScaledConventionalCNNBiGRU(
        d_enc=cfg["d_enc"],
        d_temp=cfg["d_temp"],
        head_hidden=cfg["head_hidden"],
        n_classes=4,
        n_meta=5,
        dropout=0.2,
        use_rcc_loss=(loss_type == "rccloss"),
    )
