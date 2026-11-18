"""
experiments/EXP-014/nas_iteration3_model.py
NAS Iteration 3 Architecture: Candidate Alpha (CircadianCrossModalCNNBiGRU)

Grounding:
  Hypothesis H-F (Cross-Modal IHR-Accelerometry Interaction):
    Models non-linear physiological coupling between central autonomic activity (IHR)
    and peripheral somatomotor activity (ACC) anchored by circadian phase context (c_meta).
    Modality-specific multi-scale temporal convolutions isolate instantaneous sympathetic bursts
    (30s) from slow vagal decelerations (150s), and motor twitches (30s) from postural shifts (90s).

  Hypothesis H-A (Subject-Invariant Generalization across 47 Subjects):
    Masked Temporal Instance Normalization (TIN) standardizes each nocturnal recording to
    zero nocturnal mean and unit nocturnal variance, eliminating subject baseline shifts.
    Bounded Tanh cross-gating with zero identity initialization guarantees that optimization
    starts near the robust baseline manifold and prevents out-of-distribution drift on unseen subjects.

Architectural Budget:
  Baseline Model A: 163,940 parameters
  CircadianAnchoredCrossModalFusion module: 26,562 parameters
  Total Model Parameters: Exactly 190,502 parameters (+16.20% over baseline)
  Budget Cap: Strictly compliant (190,502 <= 196,728 cap, 6,226 parameter headroom)
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
    Standardizes representations across the nocturnal sequence dimension T for each
    recording independently, completely eliminating inter-subject baseline shifts (e.g. resting HR).
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
    def __init__(self, in_channels: int = 64, mid_channels: int = 48, kernel_sizes: Tuple[int, ...] = (1, 3, 5)):
        super().__init__()
        self.num_branches = len(kernel_sizes)
        self.branch_channels = mid_channels // self.num_branches
        assert self.branch_channels * self.num_branches == mid_channels, "mid_channels must be divisible by num_branches"

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


class CircadianAnchoredCrossModalFusion(nn.Module):
    """
    Structural Mutation Module: Circadian-Anchored Cross-Modal Fusion (CACMF).
    Integrates Masked TIN, Modality-Specific Multi-Scale Blocks, Circadian Projection,
    and Bounded Tanh Cross-Gating.
    Parameters: 26,562.
    """
    def __init__(
        self,
        d_model: int = 64,
        d_meta: int = 16,
        d_circadian: int = 32,
        d_gate: int = 48,
        dropout: float = 0.1
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

        # 4. Bounded Cross-Modal Gating Network
        in_gate = 2 * d_model + d_circadian  # 64 + 64 + 32 = 160
        self.gate_net = nn.Sequential(
            nn.Linear(in_gate, d_gate),
            nn.LayerNorm(d_gate),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_gate, 2 * d_model),
            nn.Tanh()
        )

        # Learnable gating scale parameters initialized to 0.1
        self.alpha_hr = nn.Parameter(torch.tensor(0.1))
        self.alpha_acc = nn.Parameter(torch.tensor(0.1))

        self.norm_hr_out = nn.LayerNorm(d_model)
        self.norm_acc_out = nn.LayerNorm(d_model)

        # Identity initialization: zero-initialize final linear layer
        nn.init.zeros_(self.gate_net[4].weight)
        nn.init.zeros_(self.gate_net[4].bias)

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
          mask:   (B, T)
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

        # Step 4: Bounded cross-modal interaction
        joint = torch.cat([h_hr_ms, h_acc_ms, c_meta], dim=-1)  # (B, T, 160)
        gates = self.gate_net(joint)  # (B, T, 128) strictly bounded in (-1, 1)
        g_hr, g_acc = torch.chunk(gates, 2, dim=-1)  # (B, T, 64) each

        # Step 5: Bounded refinement with residual identity
        h_hr_out = self.norm_hr_out(h_hr_ms * (1.0 + self.alpha_hr * g_hr))
        h_acc_out = self.norm_acc_out(h_acc_ms * (1.0 + self.alpha_acc * g_acc))

        return h_hr_out, h_acc_out


class CircadianCrossModalCNNBiGRU(nn.Module):
    """
    NAS Iteration 3 Architecture: Candidate Alpha.
    Extends Baseline Model A with CircadianAnchoredCrossModalFusion.
    Total Parameters: Exactly 190,502 parameters (+16.20% over baseline).
    """
    def __init__(
        self,
        d_model: int = 64,
        d_temp: int = 96,
        n_classes: int = 4,
        n_meta: int = 5,
        dropout: float = 0.2,
        fusion_dropout: float = 0.1,
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

        # 2. Structural Mutation: Circadian-Anchored Cross-Modal Fusion (26,562 params)
        self.cross_fusion = CircadianAnchoredCrossModalFusion(
            d_model=d_model,
            d_meta=16,
            d_circadian=32,
            d_gate=48,
            dropout=fusion_dropout
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

        # 4. Classifier Head (Linear(96, 64) -> LayerNorm(64) -> GELU -> Dropout -> Linear(64, 4), 6,596 params)
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

        # 2. Circadian-Anchored Cross-Modal Fusion
        h_hr_fused, h_acc_fused = self.cross_fusion(h_hr, h_acc, h_meta, mask=mask)

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
            "temp_features": temp_features
        }


# Model alias per dispatch instructions
CandidateAlpha_CACMF = CircadianCrossModalCNNBiGRU


def build_nas_iteration3_model() -> CircadianCrossModalCNNBiGRU:
    """
    Factory function for NAS Iteration 3 Candidate Alpha architecture.
    """
    return CircadianCrossModalCNNBiGRU()
