r"""
experiments/EXP-014/nas_iteration7_model.py
NAS Iteration 7 Architecture: Candidate Epsilon (LocalUltradianAdaptiveNormCNNBiGRU)

Grounding:
  Hypothesis H-F (Autonomic–Somatomotor Cross-Modal Interaction & Direct Skip Highway):
    - Overcomes the recurrent information bottleneck by maintaining a direct 144-dimensional
      skip highway from the intra-epoch representation (64 HR + 64 ACC + 16 Meta) directly
      into the classification head: Linear(96 [BiGRU] + 48 [IESA] + 144 [Skip] = 288 -> 64 -> 4).
    - Preserves high-frequency autonomic-somatomotor coupling (e.g. instantaneous motor stillness
      coinciding with sympathetic cardiac surges during REM storms) bypassing the 600-step
      BPTT Jacobian of the recurrent network.

  Hypothesis H-A (Local Ultradian Adaptive Normalization & Dynamic Circadian Phase Gating):
    - Diagnosed Failure of Candidate Delta: Global whole-night z-scoring (\mu_night, \sigma_night)
      compressed non-stationary early-night sleep (Slow Wave dominance, low baseline HR) and
      late-night sleep (REM dominance, circadian cortisol rise, high baseline HR) into conflicting
      relative bands, producing 3,740 Light->REM and 1,854 REM->Light misclassifications on Fold 1.
    - Local Ultradian Basal Standardization (LUBS): Computes running local mean (\mu_local) and
      running local standard deviation (\sigma_local) across an ultradian window (W = 91 epochs = 45.5 min)
      via vectorized depthwise 1D operations. This centers Light sleep at z_local ~ 0.0 in both early
      and late night, while mapping REM bursts to sharp positive deviations (z_local > +1.5) regardless
      of the slow 8-hour circadian drift.
    - Latent Local Ultradian Adaptive Normalization (Latent LUAN): Replaces global whole-night TIN
      with local ultradian standardizers on latent HR (64) and ACC (64) representations, preventing
      early vs late night latent manifold collision.
    - Dynamic Circadian Phase Gating (DCPG): Extracts the macro-circadian trajectory
      c(t) = [t/T, (mu_local - mu_night)/sigma_night, sigma_local/sigma_night] and modulates the
      multimodal intra-epoch features with an adaptive gate, dynamically tuning sensitivity to
      sympathetic surges as a function of homeostatic sleep pressure and circadian REM propensity.
    - Inter-Epoch Sequence Attention (IESA): 4-head sequence-level attention providing O(1) receptive
      field access across all 600 epochs in the nocturnal recording.

Architectural Budget & Compression:
  - Baseline Model A: 163,940 parameters
  - Candidate Epsilon (LUAN-BiGRU): Exactly 195,708 parameters (+19.38% over baseline)
  - Budget Cap: Strictly compliant (195,708 <= 196,728 cap; 1,020 parameter headroom)
  - Compression Ratio vs 5.47M published SLAMSS-IFS paper benchmark: 96.42% parameter compression.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class LocalUltradianBasalStandardization(nn.Module):
    r"""
    Subject-Invariant Local Ultradian Basal Standardization (LUBS).
    Computes running local mean (\mu_{\text{local}}) and running local standard deviation
    (\sigma_{\text{local}}) over a 45-minute ultradian window (W = 91 epochs) using
    parameter-free depthwise 1D operations with exact padding mask protection.

    Standardizes raw instantaneous heart rate (x_ihr) and meta HR features relative to
    the local ultradian baseline, eliminating the non-stationary drift across the night.
    Also extracts the macro-circadian phase vector c(t) = [t/T, \Delta\mu, \Delta\sigma].
    """
    def __init__(self, window_size: int = 91, eps: float = 1e-5):
        super().__init__()
        self.window_size = window_size
        self.padding = window_size // 2
        self.eps = eps

        # Register non-trainable 1D box kernel buffer: (1, 1, W)
        self.register_buffer("kernel", torch.ones(1, 1, window_size))

        # Learnable affine scale and shift parameters initialized to identity (gamma=1, beta=0)
        self.gamma_ihr = nn.Parameter(torch.ones(1))
        self.beta_ihr = nn.Parameter(torch.zeros(1))
        self.gamma_meta_mean = nn.Parameter(torch.ones(1))
        self.beta_meta_mean = nn.Parameter(torch.zeros(1))
        self.gamma_meta_std = nn.Parameter(torch.ones(1))
        self.beta_meta_std = nn.Parameter(torch.zeros(1))
        self.gamma_trend = nn.Parameter(torch.ones(1))
        self.beta_trend = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        x_ihr: torch.Tensor,
        x_meta: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Args:
          x_ihr: (B, T, 30) raw instantaneous heart rate in bpm
          x_meta: (B, T, 5) meta features [freq, clock, time, hr_mean, hr_std]
          mask: (B, T) boolean mask (True for valid epochs, False for padded)
        Returns:
          z_ihr: (B, T, 30) standardized IHR relative to local ultradian baseline
          z_meta: (B, T, 5) standardized meta features with cols 3 & 4 locally normalized
          circadian_ctx: (B, T, 3) macro-circadian state [time_in_night, delta_mu, delta_sigma]
        """
        B, T, L = x_ihr.shape
        if mask is None:
            m = torch.ones(B, 1, T, dtype=x_ihr.dtype, device=x_ihr.device)
            m_mask = torch.ones(B, T, dtype=x_ihr.dtype, device=x_ihr.device)
        else:
            m = mask.unsqueeze(1).to(dtype=x_ihr.dtype)
            m_mask = mask.to(dtype=x_ihr.dtype)

        # Epoch mean heart rate: (B, 1, T)
        hr_epoch = x_meta[:, :, 3:4].transpose(1, 2)
        masked_hr = hr_epoch * m

        # 1. Local Ultradian Running Mean (45.5 min window)
        local_sum = F.conv1d(masked_hr, self.kernel, padding=self.padding)
        local_count = F.conv1d(m, self.kernel, padding=self.padding).clamp(min=1.0)
        mu_local = local_sum / local_count  # (B, 1, T)
        mu_loc = mu_local.transpose(1, 2)   # (B, T, 1)

        # True sample-level running local variance across all samples in the window
        sample_sq_diff_local = ((x_ihr - mu_loc) ** 2).mean(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, T)
        local_var = F.conv1d(sample_sq_diff_local * m, self.kernel, padding=self.padding) / local_count
        sigma_local = torch.sqrt(local_var + self.eps)  # (B, 1, T)
        sig_loc = sigma_local.transpose(1, 2)           # (B, T, 1)

        # 2. Whole-Night Global Baseline for Macro Circadian Trajectory
        night_count = m.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mu_night = (hr_epoch * m).sum(dim=-1, keepdim=True) / night_count  # (B, 1, 1)
        mu_nit = mu_night.transpose(1, 2)                                  # (B, 1, 1)

        sample_sq_diff_night = ((x_ihr - mu_nit) ** 2).mean(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, T)
        var_night = (sample_sq_diff_night * m).sum(dim=-1, keepdim=True) / night_count
        sigma_night = torch.sqrt(var_night + self.eps)  # (B, 1, 1)
        sig_nit = sigma_night.transpose(1, 2)           # (B, 1, 1)

        m_3d = m_mask.unsqueeze(-1)

        # 3. Standardize raw IHR signal relative to local ultradian baseline
        z_ihr = ((x_ihr - mu_loc) / sig_loc) * self.gamma_ihr + self.beta_ihr
        z_ihr = z_ihr * m_3d

        # 4. Standardize meta HR columns
        norm_meta_mean = ((x_meta[:, :, 3:4] - mu_loc) / sig_loc) * self.gamma_meta_mean + self.beta_meta_mean
        norm_meta_std = (x_meta[:, :, 4:5] / sig_loc) * self.gamma_meta_std + self.beta_meta_std

        # 5. Extract Macro Circadian Context [time_in_night, delta_mu, delta_sigma]
        circ_trend = ((mu_loc - mu_nit) / sig_nit) * self.gamma_trend + self.beta_trend
        circ_var_ratio = sig_loc / sig_nit
        circadian_ctx = torch.cat([x_meta[:, :, 2:3], circ_trend, circ_var_ratio], dim=-1) * m_3d

        # 6. Reconstruct meta features with preserved cols 0..2 and locally standardized cols 3..4
        z_meta = torch.cat([x_meta[:, :, :3], norm_meta_mean, norm_meta_std], dim=-1) * m_3d

        return z_ihr, z_meta, circadian_ctx


class LocalUltradianLatentNorm(nn.Module):
    r"""
    Subject-Invariant Latent Local Ultradian Adaptive Normalization (Latent LUAN).
    Computes running local mean and standard deviation over an ultradian window (W = 91 epochs)
    for each latent channel independently using depthwise 1D conv with groups=num_features.
    Prevents non-stationary latent embedding baseline shift between early-night and late-night sleep.
    """
    def __init__(self, num_features: int, window_size: int = 91, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.window_size = window_size
        self.padding = window_size // 2
        self.eps = eps

        # Register non-trainable depthwise box kernel buffer: (num_features, 1, W)
        self.register_buffer("kernel", torch.ones(num_features, 1, window_size))

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, C)
        B, T, C = x.shape
        if mask is None:
            m = torch.ones(B, 1, T, dtype=x.dtype, device=x.device)
            m_3d = torch.ones(B, T, 1, dtype=x.dtype, device=x.device)
        else:
            m = mask.unsqueeze(1).to(dtype=x.dtype)
            m_3d = mask.unsqueeze(-1).to(dtype=x.dtype)

        x_1d = x.transpose(1, 2)  # (B, C, T)
        x_masked = x_1d * m

        # Depthwise 1D running statistics
        local_sum = F.conv1d(x_masked, self.kernel, padding=self.padding, groups=C)
        local_count = F.conv1d(m, self.kernel[:1], padding=self.padding, groups=1).clamp(min=1.0)
        local_mean = local_sum / local_count
        sq_diff = ((x_1d - local_mean) ** 2) * m
        local_var = F.conv1d(sq_diff, self.kernel, padding=self.padding, groups=C) / local_count
        local_std = torch.sqrt(local_var + self.eps)

        x_norm = ((x_1d - local_mean) / local_std).transpose(1, 2) * m_3d
        out = x_norm * self.weight + self.bias
        return self.norm(x + out)


class DynamicCircadianPhaseGating(nn.Module):
    r"""
    Dynamic Circadian Phase Gating (DCPG).
    Modulates intra-epoch multimodal representations as a function of the circadian trajectory:
      c(t) = [t/T, \Delta\mu(t), \Delta\sigma(t)] \in \mathbb{R}^3
    Uses an MLP projection to generate an adaptive gate g \in (0, 1)^{144}.
    Supports zero_init: when True, fc2 is initialized to zero, ensuring 2 * sigmoid(0) = 1.0,
    providing exact step-0 identity preservation.
    """
    def __init__(
        self,
        in_features: int = 3,
        hidden_dim: int = 16,
        out_features: int = 144,
        zero_init: bool = True
    ):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_features)

        if zero_init:
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)
        else:
            nn.init.normal_(self.fc2.weight, std=0.01)
            nn.init.zeros_(self.fc2.bias)

    def forward(
        self,
        e_intra: torch.Tensor,
        circ_ctx: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # circ_ctx: (B, T, 3)
        # e_intra: (B, T, 144)
        h = self.act(self.norm(self.fc1(circ_ctx)))
        gate = torch.sigmoid(self.fc2(h))  # (B, T, 144)
        # 2.0 * gate is exactly 1.0 at step 0 if zero_init=True
        e_gated = e_intra * (2.0 * gate)
        if mask is not None:
            e_gated = e_gated * mask.unsqueeze(-1).to(dtype=e_gated.dtype)
        return e_gated, gate


class InterEpochSequenceAttention(nn.Module):
    r"""
    Inter-Epoch Sequence-Level Multi-Head Scaled Dot-Product Attention (IESA).
    Enables direct O(1) receptive field access across all epochs in the nocturnal recording (T = 600 epochs),
    overcoming the exponential gradient and information decay of 2-layer BiGRU over 90-minute ultradian cycles.
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
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)

        if mask is not None:
            key_mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(~key_mask, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_attn)
        out = self.norm(self.out_proj(context))

        if mask is not None:
            out = out * mask.unsqueeze(-1).to(dtype=out.dtype)

        return out, attn_weights


class LocalUltradianAdaptiveNormCNNBiGRU(nn.Module):
    r"""
    NAS Iteration 7 Candidate Epsilon Architecture.
    Combines:
      1. Local Ultradian Basal Standardization (LUBS, W=91 epochs = 45.5 min, 8 params)
      2. Modality-Specific Encoders (HREncoder 64, ACCEncoder 64, MetaProj 16, 59,232 params)
      3. Latent Local Ultradian Adaptive Normalization (Latent LUAN on HR & ACC, 512 params)
      4. Dynamic Circadian Phase Gating (DCPG, 2,544 params)
      5. 2-layer Bidirectional GRU (hidden 48 -> 96, 97,920 params) with packed sequence protection
      6. Inter-Epoch Sequence Multi-Head Attention (IESA, d_attn=48, 4 heads, 16,416 params)
      7. Direct Intra-Epoch Feature Skip Connection (144 dims) bypassing recurrent model
      8. Multi-Stream Classification Head: Linear(96 [GRU] + 48 [Attn] + 144 [Skip] = 288 -> 64 -> 4, 18,884 params)

    Total Parameters: Exactly 195,708 parameters (96.42% compression vs 5.47M SLAMSS-IFS).
    Budget Cap Compliance: 195,708 <= 196,728 (1,020 parameter headroom).
    """
    def __init__(
        self,
        d_model: int = 64,
        d_temp: int = 96,
        d_attn: int = 48,
        n_heads: int = 4,
        n_classes: int = 4,
        n_meta: int = 5,
        window_size: int = 91,
        dropout: float = 0.2,
        zero_init: bool = True,
        use_rcc_loss: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_temp = d_temp
        self.d_attn = d_attn
        self.n_heads = n_heads
        self.n_classes = n_classes
        self.n_meta = n_meta
        self.use_rcc_loss = use_rcc_loss

        # 1. Base components for weight matching
        base = ConventionalCNNBiGRU(
            d_model=d_model,
            d_temp=d_temp,
            n_classes=n_classes,
            n_meta=n_meta,
            dropout=dropout
        )
        self.hr_encoder = base.hr_encoder      # 29,568 params
        self.acc_encoder = base.acc_encoder    # 29,568 params
        self.meta_proj = base.meta_proj        # 96 params

        # 2. Local Ultradian Basal Standardization (8 params)
        self.basal_luan = LocalUltradianBasalStandardization(window_size=window_size)

        # 3. Latent Local Ultradian Adaptive Normalization (512 params total)
        self.luan_hr = LocalUltradianLatentNorm(d_model, window_size=window_size)   # 256 params
        self.luan_acc = LocalUltradianLatentNorm(d_model, window_size=window_size)  # 256 params

        # 4. Dynamic Circadian Phase Gating (2,544 params)
        in_temp = 2 * d_model + 16  # 64 + 64 + 16 = 144
        self.circadian_gate = DynamicCircadianPhaseGating(
            in_features=3,
            hidden_dim=16,
            out_features=in_temp,
            zero_init=zero_init
        )

        # 5. Temporal Sequence Modeling (BiGRU, 97,920 params)
        self.temporal_gru = nn.GRU(
            input_size=in_temp,
            hidden_size=d_temp // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.temp_norm = nn.LayerNorm(d_temp)  # 192 params

        # 6. Inter-Epoch Sequence Multi-Head Attention (16,416 params)
        self.inter_epoch_attn = InterEpochSequenceAttention(
            d_in=d_temp,
            d_attn=d_attn,
            n_heads=n_heads,
            dropout=dropout,
            zero_init=zero_init
        )

        # 7. Multi-Stream Classifier Head (18,884 params)
        # Combines BiGRU state (96) + Attention context (48) + Direct Skip features (144) = 288 dims
        in_clf = d_temp + d_attn + in_temp
        self.classifier_head = nn.Sequential(
            nn.Linear(in_clf, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

        # Loss functions
        self.ce_loss = base.ce_loss
        self.rcc_loss = base.rcc_loss

    def forward(
        self,
        x_ihr: Union[torch.Tensor, Dict[str, Any]],
        x_accel: Optional[torch.Tensor] = None,
        x_meta: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        subject_idx: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass supporting:
          1. Individual tensor arguments: model(x_ihr, x_accel, x_meta, targets, mask, ...)
          2. Three-argument positional: model(x, x_meta, mask) where x has shape (B, T, 60)
          3. Two-argument keyword: model(x, x_meta=x_meta, mask=mask)
          4. Dictionary batch input: model(batch_dict)
        """
        # Parse inputs
        if isinstance(x_ihr, dict):
            batch = x_ihr
            x_ihr = batch["ihr"]
            x_accel = batch["accel"]
            x_meta = batch["meta"]
            targets = batch.get("y", targets)
            mask = batch.get("mask", mask)
            subject_idx = batch.get("subject_idx", subject_idx)
            seq_lens = batch.get("seq_len", seq_lens)
        elif x_accel is not None and x_accel.shape[-1] == 5:
            # Positional call model(x, x_meta, mask)
            actual_mask = targets if mask is None else mask
            if x_meta is not None and x_meta.dtype == torch.bool:
                actual_mask = x_meta
            x_meta = x_accel
            mask = actual_mask
            targets = None
            if x_ihr.shape[-1] == 60:
                x_accel = x_ihr[:, :, 30:60]
                x_ihr = x_ihr[:, :, :30]
            else:
                raise ValueError(f"Expected x to have 60 channels when passing (x, x_meta, mask), got {x_ihr.shape[-1]}")
        elif x_accel is None and x_meta is not None:
            # Keyword call model(x, x_meta=meta, mask=mask)
            if x_ihr.shape[-1] == 60:
                x_accel = x_ihr[:, :, 30:60]
                x_ihr = x_ihr[:, :, :30]

        # Handle case where mask is passed positionally as 4th argument (targets parameter)
        if targets is not None and targets.dtype == torch.bool:
            mask = targets
            targets = None

        B, T, _ = x_ihr.shape

        # Step 1: Physiological Local Ultradian Basal Standardization
        z_ihr, z_meta, circ_ctx = self.basal_luan(x_ihr, x_meta, mask=mask)

        # Step 2: Modality-Specific Feature Extraction
        h_hr = self.hr_encoder(z_ihr)       # (B, T, 64)
        h_acc = self.acc_encoder(x_accel)   # (B, T, 64)
        h_meta = self.meta_proj(z_meta)     # (B, T, 16)

        # Step 3: Latent Local Ultradian Adaptive Normalization
        z_hr = self.luan_hr(h_hr, mask=mask)      # (B, T, 64)
        z_acc = self.luan_acc(h_acc, mask=mask)   # (B, T, 64)

        # Step 4: Intra-Epoch Multimodal Representation (144 dims)
        e_intra = torch.cat([z_hr, z_acc, h_meta], dim=-1)  # (B, T, 144)

        # Step 5: Dynamic Circadian Phase Gating
        e_gated, circ_gate = self.circadian_gate(e_intra, circ_ctx, mask=mask)  # (B, T, 144)

        # Step 6: Recurrent Temporal Sequence Modeling with Packing Support
        if seq_lens is not None:
            seq_lens_cpu = seq_lens.to("cpu", dtype=torch.int64)
            packed = nn.utils.rnn.pack_padded_sequence(
                e_gated, seq_lens_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.temporal_gru(packed)
            temp_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=T
            )
        else:
            temp_out, _ = self.temporal_gru(e_gated)

        temp_features = self.temp_norm(temp_out)  # (B, T, 96)
        if mask is not None:
            temp_features = temp_features * mask.unsqueeze(-1).to(dtype=temp_features.dtype)

        # Step 7: Inter-Epoch Sequence Multi-Head Attention Context
        attn_context, attn_weights = self.inter_epoch_attn(temp_features, mask=mask)  # (B, T, 48)

        # Step 8: Multi-Stream Skip Integration
        # Direct intra-epoch skip highway bypassing recurrent model
        combined = torch.cat([temp_features, attn_context, e_intra], dim=-1)  # (B, T, 288)

        # Step 9: Classification Logits Emission
        logits = self.classifier_head(combined)  # (B, T, 4)

        # Step 10: Loss Calculation with RCC Loss Support
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
            "skip_features": e_intra,
            "e_gated": e_gated,
            "circ_gate": circ_gate,
            "circ_ctx": circ_ctx,
            "z_ihr": z_ihr,
            "z_meta": z_meta
        }


# Model aliases per dispatch conventions
CandidateEpsilon_LUAN = LocalUltradianAdaptiveNormCNNBiGRU
CandidateEpsilon_LUAP = LocalUltradianAdaptiveNormCNNBiGRU


def build_nas_iteration7_model(
    window_size: int = 91,
    dropout: float = 0.2,
    zero_init: bool = True,
    use_rcc_loss: bool = True
) -> LocalUltradianAdaptiveNormCNNBiGRU:
    """Factory function for NAS Iteration 7 Candidate Epsilon architecture."""
    return LocalUltradianAdaptiveNormCNNBiGRU(
        window_size=window_size,
        dropout=dropout,
        zero_init=zero_init,
        use_rcc_loss=use_rcc_loss
    )
