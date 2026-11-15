"""
NAS Iteration 1: Mutated PyTorch Model Architecture
File: experiments/EXP-014/nas_iteration1_model.py

Grounding:
  Hypothesis H-A (RESEARCH_DIRECTION_v3_FULL.md):
  Multi-night within-person adaptation leverages repeated nocturnal recordings from
  the same individual where biological dynamics manifest across multiple temporal scales
  (instantaneous autonomic arousals vs stable Slow-Wave Sleep vs REM atonia).
  Standard uniform BiGRU conflates high-frequency single-night noise with multi-epoch state stability.

Structural Mutation:
  Multi-Scale Temporal Convolutional Processing with parallel receptive fields k in {1, 3, 5},
  pointwise projection bottleneck, group/layer normalization, and residual shortcut.

Architectural Budget:
  Baseline Model A: 163,940 parameters
  Maximum allowed (+20% cap): 196,728 parameters
  Pre-recurrent Mutation (MSTF-Pre): 192,692 parameters (+28,752, +17.54%) -> Budget Compliant
  Intra-encoder Mutation (MS-TCB): 173,284 parameters (+9,344, +5.70%) -> Budget Compliant
"""

import sys
import os
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure workspace root is in path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from exp013_models import ConventionalCNNBiGRU
from sleep_adaptive_net import (
    HREncoder,
    ACCEncoder,
    RCCLoss,
    get_official_weights
)


class MultiScaleTemporalFusion(nn.Module):
    """
    Pre-Recurrent Multi-Scale Temporal Fusion (MSTF-Pre) Block.
    Operates across temporal sequence epochs T on concatenated multimodal features (B, T, C_in).
    
    Architecture:
      X (B, T, C_in) -> Transpose to (B, C_in, T)
      -> 1x1 Conv Bottleneck Projection (C_in -> C_mid)
      -> Split into K parallel branches
      -> Parallel Depthwise Convolutions with kernel sizes k in {1, 3, 5}
         (padding = k // 2 preserves sequence length T)
      -> GroupNorm(4, C_branch) + GELU per branch
      -> Channel Concatenation (B, C_mid, T)
      -> 1x1 Conv Restoration Projection (C_mid -> C_in)
      -> Transpose back to (B, T, C_in)
      -> Residual Connection + LayerNorm(C_in) + GELU:
         Y = GELU(LayerNorm(X + Out))
    """
    def __init__(
        self,
        in_channels: int = 144,
        bottleneck_channels: int = 32,
        kernel_sizes: Tuple[int, ...] = (1, 3, 5)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_branches = len(kernel_sizes)
        self.mid_channels = bottleneck_channels * self.num_branches
        
        # 1. Bottleneck projection: 144 -> 96
        self.proj_in = nn.Conv1d(in_channels, self.mid_channels, kernel_size=1)
        
        # 2. Parallel multi-scale depthwise branches: k in {1, 3, 5}
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    bottleneck_channels,
                    bottleneck_channels,
                    kernel_size=k,
                    padding=k // 2,
                    groups=bottleneck_channels
                ),
                nn.GroupNorm(min(4, bottleneck_channels), bottleneck_channels),
                nn.GELU()
            )
            for k in kernel_sizes
        ])
        
        # 3. Restoration projection: 96 -> 144
        self.proj_out = nn.Conv1d(self.mid_channels, in_channels, kernel_size=1)
        
        # 4. Residual normalization and activation
        self.norm = nn.LayerNorm(in_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass preserving exact dimensions: (B, T, C) -> (B, T, C)
        """
        res = x
        x_1d = x.transpose(1, 2) # (B, C, T)
        proj = self.proj_in(x_1d) # (B, C_mid, T)
        chunks = torch.chunk(proj, self.num_branches, dim=1)
        outs = [b(c) for b, c in zip(self.branches, chunks)]
        fused = torch.cat(outs, dim=1) # (B, C_mid, T)
        out = self.proj_out(fused).transpose(1, 2) # (B, T, C)
        return self.act(self.norm(res + out))


class MultiScaleTemporalConvBlock(nn.Module):
    """
    Intra-Encoder Multi-Scale Temporal Convolutional Block (MS-TCB).
    Replaces net.6 in HREncoder and ACCEncoder to process intra-epoch (30s at 1Hz)
    intermediate representations (length L=7).
    
    Architecture:
      Parallel conv branches with k in {1, 3, 5} and channel split (20, 20, 24) -> 64
      -> 1x1 Conv Projection (64 -> 64)
      -> GroupNorm(4, 64) + GELU
      -> Residual Connection: Y = Out + Identity(X)
    """
    def __init__(self, in_channels: int = 64, out_channels: int = 64):
        super().__init__()
        c1, c2, c3 = 20, 20, 24
        assert c1 + c2 + c3 == out_channels, f"Channel sum {c1+c2+c3} != {out_channels}"
        self.b1 = nn.Conv1d(in_channels, c1, kernel_size=1)
        self.b2 = nn.Conv1d(in_channels, c2, kernel_size=3, padding=1)
        self.b3 = nn.Conv1d(in_channels, c3, kernel_size=5, padding=2)
        self.proj = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.gn = nn.GroupNorm(4, out_channels)
        self.act = nn.GELU()
        self.res = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        out = self.act(self.gn(self.proj(out)))
        return out + self.res(x)


class MultiScaleConventionalCNNBiGRU(nn.Module):
    """
    NAS Iteration 1 Mutated Model A Architecture.
    
    Grounded in Hypothesis H-A for multi-night within-person adaptation:
    Extends Model A (ConventionalCNNBiGRU, 163,940 params) with isolated structural
    multi-scale temporal convolutional fusion.
    
    Two mutation types are supported:
      - 'pre_recurrent' (default): MultiScaleTemporalFusion placed between multimodal
        concatenation (144 dims) and temporal BiGRU.
        Parameters: 192,692 (+28,752, +17.54% <= +20% budget cap).
      - 'intra_encoder': MultiScaleTemporalConvBlock replacing net.6 in HREncoder and ACCEncoder.
        Parameters: 173,284 (+9,344, +5.70% <= +20% budget cap).
    """
    def __init__(
        self,
        d_model: int = 64,
        d_temp: int = 96,
        n_classes: int = 4,
        n_meta: int = 5,
        dropout: float = 0.2,
        use_rcc_loss: bool = True,
        mutation_type: str = "pre_recurrent"
    ):
        super().__init__()
        self.d_model = d_model
        self.d_temp = d_temp
        self.n_classes = n_classes
        self.use_rcc_loss = use_rcc_loss
        self.mutation_type = mutation_type.lower()
        
        # 1. Modality-specific Encoders
        self.hr_encoder = HREncoder(in_features=1, d_model=d_model, dropout=dropout)
        self.acc_encoder = ACCEncoder(in_features=1, d_model=d_model, dropout=dropout)
        
        if self.mutation_type == "intra_encoder":
            # Mutate net.6 inside both modality encoders
            self.hr_encoder.net[6] = MultiScaleTemporalConvBlock(d_model, d_model)
            self.acc_encoder.net[6] = MultiScaleTemporalConvBlock(d_model, d_model)
        
        # Meta feature projection
        self.meta_proj = nn.Sequential(
            nn.Linear(n_meta, 16),
            nn.GELU()
        )
        
        # 2. Structural Mutation: Pre-Recurrent Multi-Scale Temporal Fusion
        in_temp = 2 * d_model + 16 # 64 + 64 + 16 = 144
        if self.mutation_type == "pre_recurrent":
            self.mstf = MultiScaleTemporalFusion(
                in_channels=in_temp,
                bottleneck_channels=32,
                kernel_sizes=(1, 3, 5)
            )
        else:
            self.mstf = nn.Identity()
        
        # 3. Temporal Sequence Modeling (BiGRU)
        self.temporal_gru = nn.GRU(
            input_size=in_temp,
            hidden_size=d_temp // 2, # bidirectional -> total d_temp = 96
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.temp_norm = nn.LayerNorm(d_temp)
        
        # 4. Classifier Head
        self.classifier_head = nn.Sequential(
            nn.Linear(d_temp, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        
        # Stage Loss Functions
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
        """
        Forward pass accepting either individual tensors or a batch dictionary.
        Accepts:
          x_ihr: (B, T, 30) or batch dict
          x_accel: (B, T, 30)
          x_meta: (B, T, 5)
          targets: (B, T)
          mask: (B, T)
          subject_idx: (B,)
          seq_lens: (B,)
        """
        # Handle batch dictionary input for complete compatibility
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
        h_hr = self.hr_encoder(x_ihr)     # (B, T, d_model)
        h_acc = self.acc_encoder(x_accel) # (B, T, d_model)
        h_meta = self.meta_proj(x_meta)   # (B, T, 16)
        
        # 2. Concatenation
        cat_features = torch.cat([h_hr, h_acc, h_meta], dim=-1) # (B, T, 144)
        
        # 3. Multi-Scale Temporal Fusion (if enabled)
        fused_features = self.mstf(cat_features) # (B, T, 144)
        
        # 4. Temporal Sequence Modeling with Packing Support
        if seq_lens is not None:
            seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int64)
            packed_features = nn.utils.rnn.pack_padded_sequence(
                fused_features, seq_lens_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.temporal_gru(packed_features)
            temp_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )
        else:
            temp_out, _ = self.temporal_gru(fused_features)
            
        temp_features = self.temp_norm(temp_out) # (B, T, d_temp)
        
        # 5. Classifier Head
        logits = self.classifier_head(temp_features) # (B, T, n_classes)
        
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
MutatedModelA_MSCNNBiGRU = MultiScaleConventionalCNNBiGRU


def build_nas_iteration1_model(mutation_type: str = "pre_recurrent") -> MultiScaleConventionalCNNBiGRU:
    """
    Factory function for NAS Iteration 1 mutated models.
    """
    return MultiScaleConventionalCNNBiGRU(mutation_type=mutation_type)


if __name__ == "__main__":
    print("Testing NAS Iteration 1 Model Architecture...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Pre-recurrent variant
    m1 = MultiScaleConventionalCNNBiGRU(mutation_type="pre_recurrent").to(device)
    p1 = sum(p.numel() for p in m1.parameters())
    print(f"Pre-Recurrent Mutated Model params: {p1:,} (+{p1 - 163940:,}, +{(p1 - 163940)/163940*100:.2f}%)")
    assert p1 <= 196728, f"Parameter count {p1} exceeds 196,728 cap!"
    
    # Intra-encoder variant
    m2 = MultiScaleConventionalCNNBiGRU(mutation_type="intra_encoder").to(device)
    p2 = sum(p.numel() for p in m2.parameters())
    print(f"Intra-Encoder Mutated Model params: {p2:,} (+{p2 - 163940:,}, +{(p2 - 163940)/163940*100:.2f}%)")
    assert p2 <= 196728, f"Parameter count {p2} exceeds 196,728 cap!"
    
    # Forward and backward test
    B, T = 4, 600
    x_ihr = torch.randn(B, T, 30, device=device)
    x_acc = torch.randn(B, T, 30, device=device)
    x_meta = torch.randn(B, T, 5, device=device)
    targets = torch.randint(0, 4, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    seq_lens = torch.full((B,), T, dtype=torch.int64, device=device)
    
    out1 = m1(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)
    assert out1["logits"].shape == (B, T, 4), f"Unexpected shape {out1['logits'].shape}"
    out1["total_loss"].backward()
    assert m1.mstf.proj_in.weight.grad is not None, "Missing gradient in mstf!"
    print("Forward and backward gradient verification: PASSED")
