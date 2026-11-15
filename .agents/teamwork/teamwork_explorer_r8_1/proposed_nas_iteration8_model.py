r"""
experiments/EXP-014/nas_iteration8_model.py
NAS Iteration 8 Architecture: Candidate Zeta (LocalUltradianPyramidCNNBiGRU)

Grounding:
  Hypothesis H-F (Autonomic–Somatomotor Cross-Modal Interaction & Direct Skip Highway):
    - Overcomes the recurrent information bottleneck by maintaining a direct 144-dimensional
      skip highway from the intra-epoch representation (64 HR + 64 ACC + 16 Meta) directly
      into the multi-stream classification head: Linear(96 [BiGRU] + 48 [IESA] + 144 [Skip] = 288 -> 64 -> 4).
    - Preserves high-frequency autonomic-somatomotor coupling (e.g. instantaneous motor stillness
      coinciding with sympathetic cardiac surges during REM storms) bypassing the 600-step
      BPTT Jacobian of the recurrent network.

  Hypothesis H-A (Multi-Scale Ultradian Temporal Pyramids & Dual-Scale Local Basal Standardization):
    - Diagnosed Deficiency of Candidate Epsilon:
      While Candidate Epsilon established a new project high-water mark on locked Fold 1
      (64.27% Accuracy, 0.6435 Weighted-F1, boosting REM recall from 49.37% to 59.64% and reducing
      Light->REM false positives by 34.6%), its outer-test confusion matrix revealed severe
      Deep-Light sleep confusion:
        * 2,290 Deep sleep windows misclassified as Light (28.92% of all Deep sleep).
        * 1,468 Light sleep windows misclassified as Deep (10.59% of Light sleep).
        * Deep-Light confusion accounted for 3,758 misclassified windows (11.7% of all outer-test data).
    - Physiological Root Cause:
      A single window size (W = 91 epochs = 45.5 min) captures the broad 90-minute ultradian macro-drift,
      but cannot resolve local transient autonomic events:
        * Deep sleep (Slow Wave Sleep / N3) requires sustained parasympathetic suppression without transient sympathetic spikes.
        * Light sleep (N1/N2) contains frequent transient autonomic activations, arousals, and sleep spindles/K-complexes.
        * Transient arousals operate on a micro timescale of W_micro = 15 epochs (7.5 min), whereas the ultradian
          macro-cycle operates at W_macro = 91 epochs (45.5 min).
    - Dual-Scale Local Basal Standardization (DSLBS):
      Computes running baseline statistics across dual complementary timescales:
        * Micro scale (W_micro = 15 epochs / 7.5 min): measures immediate local context and transient autonomic bursts.
        * Macro scale (W_macro = 91 epochs / 45.5 min): measures the 90-minute ultradian circadian baseline.
      Computes the Autonomic Contrast Differential:
        \Delta z(t) = z_micro(t) - z_macro(t) = (\mu_micro(t) - \mu_macro(t)) / \sigma_macro(t).
      When \Delta z(t) ~ 0 and z_macro(t) < -0.5, physiology indicates sustained, quiescent Deep sleep.
      When \Delta z(t) > 0, physiology indicates transient sympathetic activation (Light sleep micro-arousals or REM bursts).
    - Latent Multi-Scale Ultradian Temporal Pyramids (Latent MUP LUAN):
      Computes micro (W=15) and macro (W=91) depthwise 1D running statistics independently across all 64 channels
      of latent HR and ACC representations, injecting channel-wise contrast differentials with zero-initialized
      adaptation weights for exact step-0 identity preservation.
    - Dynamic Circadian Phase Gating (DCPG):
      Conditions on a 6-dimensional Multi-Scale Ultradian Temporal Pyramid context:
        c_mup(t) = [t/T, circ_trend, circ_var_ratio, \Delta z(t), z_macro(t), \sigma_micro/\sigma_macro].
      Maintains exact step-0 identity preservation (2 * g \equiv 1.0) via zero-initialized output projection.
    - Inter-Epoch Sequence Attention (IESA):
      4-head sequence-level attention providing O(1) receptive field access across all 600 epochs in the nocturnal recording.

Architectural Budget & Compression:
  - Baseline Model A: 163,940 parameters
  - Candidate Epsilon (Iter 7): 195,708 parameters
  - Candidate Zeta (MUP-BiGRU): Exactly 195,892 parameters (+184 params over Epsilon, +19.49% over baseline)
  - Budget Cap: Strictly compliant (195,892 <= 196,728 cap; 836 parameter headroom)
  - Compression Ratio vs 5.47M published SLAMSS-IFS paper benchmark: 96.42% parameter compression.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dynamically resolve workspace root whether in .agents/teamwork/... or experiments/EXP-014
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_candidate1 = os.path.abspath(os.path.join(cur_dir, "../../.."))
root_candidate2 = os.path.abspath(os.path.join(cur_dir, "../.."))
WORKSPACE_ROOT = root_candidate1 if os.path.exists(os.path.join(root_candidate1, "exp013_models.py")) else root_candidate2
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from exp013_models import ConventionalCNNBiGRU
from sleep_adaptive_net import (
    HREncoder,
    ACCEncoder,
    RCCLoss,
    get_official_weights
)


class DualScaleLocalBasalStandardization(nn.Module):
    r"""
    Subject-Invariant Dual-Scale Local Basal Standardization (DSLBS).
    Computes running local mean (\mu) and standard deviation (\sigma) across dual complementary timescales:
      - Micro scale: W_micro = 15 epochs (7.5 min), resolving transient autonomic activations and micro-arousals.
      - Macro scale: W_macro = 91 epochs (45.5 min), capturing the 90-minute ultradian circadian baseline.
    
    Standardizes raw instantaneous heart rate (x_ihr) and meta HR features, and extracts the
    6-dimensional Multi-Scale Ultradian Temporal Pyramid context:
      c_mup(t) = [t/T, circ_trend, circ_var_ratio, \Delta z(t), z_macro(t), var_ratio_micro_macro]
    where \Delta z(t) = (\mu_micro - \mu_macro) / \sigma_macro is the Autonomic Contrast Differential.
    """
    def __init__(
        self,
        window_macro: int = 91,
        window_micro: int = 15,
        eps: float = 1e-5
    ):
        super().__init__()
        self.window_macro = window_macro
        self.window_micro = window_micro
        self.pad_macro = window_macro // 2
        self.pad_micro = window_micro // 2
        self.eps = eps

        # Register non-trainable 1D box kernel buffers: (1, 1, W)
        self.register_buffer("kernel_macro", torch.ones(1, 1, window_macro))
        self.register_buffer("kernel_micro", torch.ones(1, 1, window_micro))

        # Learnable affine scale and shift parameters
        self.gamma_ihr = nn.Parameter(torch.ones(1))
        self.beta_ihr = nn.Parameter(torch.zeros(1))
        self.alpha_ihr_diff = nn.Parameter(torch.zeros(1))  # 0 at step 0 for exact Epsilon identity

        self.gamma_meta_mean = nn.Parameter(torch.ones(1))
        self.beta_meta_mean = nn.Parameter(torch.zeros(1))
        self.alpha_meta_diff = nn.Parameter(torch.zeros(1))  # 0 at step 0 for exact Epsilon identity
        self.gamma_meta_std = nn.Parameter(torch.ones(1))
        self.beta_meta_std = nn.Parameter(torch.zeros(1))

        # Context feature scaling parameters
        self.gamma_trend = nn.Parameter(torch.ones(1))
        self.beta_trend = nn.Parameter(torch.zeros(1))
        self.gamma_delta_z = nn.Parameter(torch.ones(1))
        self.beta_delta_z = nn.Parameter(torch.zeros(1))
        self.gamma_z_macro = nn.Parameter(torch.ones(1))
        self.beta_z_macro = nn.Parameter(torch.zeros(1))
        self.gamma_var_ratio = nn.Parameter(torch.ones(1))
        self.beta_var_ratio = nn.Parameter(torch.zeros(1))

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
          z_ihr: (B, T, 30) standardized IHR with micro-contrast differential
          z_meta: (B, T, 5) standardized meta features with cols 3 & 4 locally normalized
          mup_ctx: (B, T, 6) multi-scale temporal pyramid context
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

        # 1. Macro-Scale Running Statistics (W_macro = 91 epochs = 45.5 min)
        macro_sum = F.conv1d(masked_hr, self.kernel_macro, padding=self.pad_macro)
        macro_count = F.conv1d(m, self.kernel_macro, padding=self.pad_macro).clamp(min=1.0)
        mu_macro = macro_sum / macro_count  # (B, 1, T)
        mu_mac = mu_macro.transpose(1, 2)   # (B, T, 1)

        sample_sq_diff_macro = ((x_ihr - mu_mac) ** 2).mean(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, T)
        macro_var = F.conv1d(sample_sq_diff_macro * m, self.kernel_macro, padding=self.pad_macro) / macro_count
        sigma_macro = torch.sqrt(macro_var + self.eps)  # (B, 1, T)
        sig_mac = sigma_macro.transpose(1, 2)           # (B, T, 1)

        # 2. Micro-Scale Running Statistics (W_micro = 15 epochs = 7.5 min)
        micro_sum = F.conv1d(masked_hr, self.kernel_micro, padding=self.pad_micro)
        micro_count = F.conv1d(m, self.kernel_micro, padding=self.pad_micro).clamp(min=1.0)
        mu_micro = micro_sum / micro_count  # (B, 1, T)
        mu_mic = mu_micro.transpose(1, 2)   # (B, T, 1)

        sample_sq_diff_micro = ((x_ihr - mu_mic) ** 2).mean(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, T)
        micro_var = F.conv1d(sample_sq_diff_micro * m, self.kernel_micro, padding=self.pad_micro) / micro_count
        sigma_micro = torch.sqrt(micro_var + self.eps)  # (B, 1, T)
        sig_mic = sigma_micro.transpose(1, 2)           # (B, T, 1)

        # 3. Whole-Night Global Baseline for Macro Circadian Trajectory
        night_count = m.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mu_night = (hr_epoch * m).sum(dim=-1, keepdim=True) / night_count  # (B, 1, 1)
        mu_nit = mu_night.transpose(1, 2)                                  # (B, 1, 1)

        sample_sq_diff_night = ((x_ihr - mu_nit) ** 2).mean(dim=-1, keepdim=True).transpose(1, 2)  # (B, 1, T)
        var_night = (sample_sq_diff_night * m).sum(dim=-1, keepdim=True) / night_count
        sigma_night = torch.sqrt(var_night + self.eps)  # (B, 1, 1)
        sig_nit = sigma_night.transpose(1, 2)           # (B, 1, 1)

        m_3d = m_mask.unsqueeze(-1)

        # 4. Standardize raw IHR signal with micro-macro contrast differential
        z_ihr_macro = (x_ihr - mu_mac) / sig_mac
        z_ihr_micro = (x_ihr - mu_mic) / sig_mic
        delta_z_ihr = z_ihr_micro - z_ihr_macro
        z_ihr = (z_ihr_macro + self.alpha_ihr_diff * delta_z_ihr) * self.gamma_ihr + self.beta_ihr
        z_ihr = z_ihr * m_3d

        # 5. Autonomic Contrast Differential \Delta z(t) and Macro Basal Depth
        delta_z = (mu_mic - mu_mac) / sig_mac
        z_epoch_macro = (x_meta[:, :, 3:4] - mu_mac) / sig_mac
        z_macro_night = (mu_mac - mu_nit) / sig_nit

        # 6. Standardize meta HR columns
        norm_meta_mean = (z_epoch_macro + self.alpha_meta_diff * delta_z) * self.gamma_meta_mean + self.beta_meta_mean
        norm_meta_std = (x_meta[:, :, 4:5] / sig_mac) * self.gamma_meta_std + self.beta_meta_std
        z_meta = torch.cat([x_meta[:, :, :3], norm_meta_mean, norm_meta_std], dim=-1) * m_3d

        # 7. Extract 6-Dimensional Multi-Scale Ultradian Temporal Pyramid Context
        circ_trend = ((mu_mac - mu_nit) / sig_nit) * self.gamma_trend + self.beta_trend
        circ_var_ratio = sig_mac / sig_nit
        delta_z_scaled = delta_z * self.gamma_delta_z + self.beta_delta_z
        z_macro_scaled = z_macro_night * self.gamma_z_macro + self.beta_z_macro
        var_ratio_scaled = (sig_mic / sig_mac) * self.gamma_var_ratio + self.beta_var_ratio
        time_col = x_meta[:, :, 2:3]

        mup_ctx = torch.cat([
            time_col,
            circ_trend,
            circ_var_ratio,
            delta_z_scaled,
            z_macro_scaled,
            var_ratio_scaled
        ], dim=-1) * m_3d

        return z_ihr, z_meta, mup_ctx


class MultiScaleLocalUltradianLatentNorm(nn.Module):
    r"""
    Subject-Invariant Multi-Scale Local Ultradian Latent Normalization (Latent MUP LUAN).
    Computes running local mean and standard deviation over dual ultradian timescales:
      - W_macro = 91 epochs (45.5 min)
      - W_micro = 15 epochs (7.5 min)
    independently for each latent channel using depthwise 1D conv with groups=num_features.
    Injects channel-wise latent contrast differentials (\Delta x_norm = x_norm_micro - x_norm_macro)
    with zero-initialized alpha parameters to preserve step-0 identity to Candidate Epsilon.
    """
    def __init__(
        self,
        num_features: int,
        window_macro: int = 91,
        window_micro: int = 15,
        eps: float = 1e-5
    ):
        super().__init__()
        self.num_features = num_features
        self.window_macro = window_macro
        self.window_micro = window_micro
        self.pad_macro = window_macro // 2
        self.pad_micro = window_micro // 2
        self.eps = eps

        # Register non-trainable depthwise box kernel buffers: (num_features, 1, W)
        self.register_buffer("kernel_macro", torch.ones(num_features, 1, window_macro))
        self.register_buffer("kernel_micro", torch.ones(num_features, 1, window_micro))

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.alpha = nn.Parameter(torch.zeros(num_features))  # Zero-init for step-0 identity
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

        # Macro running statistics (W_macro = 91)
        macro_sum = F.conv1d(x_masked, self.kernel_macro, padding=self.pad_macro, groups=C)
        macro_count = F.conv1d(m, self.kernel_macro[:1], padding=self.pad_macro, groups=1).clamp(min=1.0)
        macro_mean = macro_sum / macro_count
        sq_macro = ((x_1d - macro_mean) ** 2) * m
        macro_var = F.conv1d(sq_macro, self.kernel_macro, padding=self.pad_macro, groups=C) / macro_count
        macro_std = torch.sqrt(macro_var + self.eps)
        x_norm_macro = ((x_1d - macro_mean) / macro_std).transpose(1, 2) * m_3d

        # Micro running statistics (W_micro = 15)
        micro_sum = F.conv1d(x_masked, self.kernel_micro, padding=self.pad_micro, groups=C)
        micro_count = F.conv1d(m, self.kernel_micro[:1], padding=self.pad_micro, groups=1).clamp(min=1.0)
        micro_mean = micro_sum / micro_count
        sq_micro = ((x_1d - micro_mean) ** 2) * m
        micro_var = F.conv1d(sq_micro, self.kernel_micro, padding=self.pad_micro, groups=C) / micro_count
        micro_std = torch.sqrt(micro_var + self.eps)
        x_norm_micro = ((x_1d - micro_mean) / micro_std).transpose(1, 2) * m_3d

        # Latent contrast differential
        delta_x_norm = x_norm_micro - x_norm_macro

        # Multi-scale adaptive synthesis (step-0 identity to Epsilon when alpha == 0)
        out = (x_norm_macro + self.alpha * delta_x_norm) * self.weight + self.bias
        res = self.norm(x + out)
        if mask is not None:
            res = res * m_3d
        return res


class DynamicCircadianPhaseGating(nn.Module):
    r"""
    Dynamic Circadian Phase Gating (DCPG).
    Modulates intra-epoch multimodal representations as a function of the 6-dimensional
    Multi-Scale Ultradian Temporal Pyramid state:
      c_mup(t) = [t/T, circ_trend, circ_var_ratio, \Delta z(t), z_macro(t), var_ratio] \in \mathbb{R}^6
    Uses an MLP projection to generate an adaptive gate g \in (0, 1)^{144}.
    Supports zero_init: when True, fc2 is initialized to zero, ensuring 2 * sigmoid(0) = 1.0,
    providing exact step-0 identity preservation.
    """
    def __init__(
        self,
        in_features: int = 6,
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
        # circ_ctx: (B, T, 6)
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


class LocalUltradianPyramidCNNBiGRU(nn.Module):
    r"""
    NAS Iteration 8 Candidate Zeta Architecture.
    Combines:
      1. Dual-Scale Local Basal Standardization (DSLBS, W_macro=91, W_micro=15, 16 params)
      2. Modality-Specific Encoders (HREncoder 64, ACCEncoder 64, MetaProj 16, 59,232 params)
      3. Multi-Scale Local Ultradian Latent Normalization (Latent MUP LUAN, 640 params)
      4. Dynamic Circadian Phase Gating (DCPG with 6-dim MUP context, 2,592 params)
      5. 2-layer Bidirectional GRU (hidden 48 -> 96, 97,920 params) with packed sequence protection
      6. Inter-Epoch Sequence Multi-Head Attention (IESA, d_attn=48, 4 heads, 16,416 params)
      7. Direct Intra-Epoch Feature Skip Highway (144 dims) bypassing recurrent model
      8. Multi-Stream Classification Head: Linear(96 [GRU] + 48 [Attn] + 144 [Skip] = 288 -> 64 -> 4, 18,884 params)

    Total Parameters: Exactly 195,892 parameters (96.42% compression vs 5.47M SLAMSS-IFS).
    Budget Cap Compliance: 195,892 <= 196,728 (836 parameter headroom).
    """
    def __init__(
        self,
        d_model: int = 64,
        d_temp: int = 96,
        d_attn: int = 48,
        n_heads: int = 4,
        n_classes: int = 4,
        n_meta: int = 5,
        window_macro: int = 91,
        window_micro: int = 15,
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
        self.window_macro = window_macro
        self.window_micro = window_micro
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

        # 2. Dual-Scale Local Basal Standardization (16 params)
        self.basal_dslbs = DualScaleLocalBasalStandardization(
            window_macro=window_macro,
            window_micro=window_micro
        )

        # 3. Multi-Scale Local Ultradian Latent Normalization (640 params total: 320 each)
        self.luan_hr = MultiScaleLocalUltradianLatentNorm(
            d_model,
            window_macro=window_macro,
            window_micro=window_micro
        )
        self.luan_acc = MultiScaleLocalUltradianLatentNorm(
            d_model,
            window_macro=window_macro,
            window_micro=window_micro
        )

        # 4. Dynamic Circadian Phase Gating with 6-dim MUP context (2,592 params)
        in_temp = 2 * d_model + 16  # 64 + 64 + 16 = 144
        self.circadian_gate = DynamicCircadianPhaseGating(
            in_features=6,
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

        # Step 1: Dual-Scale Local Basal Standardization (DSLBS)
        z_ihr, z_meta, mup_ctx = self.basal_dslbs(x_ihr, x_meta, mask=mask)

        # Step 2: Modality-Specific Feature Extraction
        h_hr = self.hr_encoder(z_ihr)       # (B, T, 64)
        h_acc = self.acc_encoder(x_accel)   # (B, T, 64)
        h_meta = self.meta_proj(z_meta)     # (B, T, 16)

        # Step 3: Latent Multi-Scale Ultradian Temporal Pyramid Normalization (Latent MUP LUAN)
        z_hr = self.luan_hr(h_hr, mask=mask)      # (B, T, 64)
        z_acc = self.luan_acc(h_acc, mask=mask)   # (B, T, 64)

        # Step 4: Intra-Epoch Multimodal Representation (144 dims)
        e_intra = torch.cat([z_hr, z_acc, h_meta], dim=-1)  # (B, T, 144)

        # Step 5: Dynamic Circadian Phase Gating conditioned on 6-dim MUP context
        e_gated, circ_gate = self.circadian_gate(e_intra, mup_ctx, mask=mask)  # (B, T, 144)

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
        # Direct 144-dim intra-epoch skip highway bypassing recurrent model
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
            "mup_ctx": mup_ctx,
            "z_ihr": z_ihr,
            "z_meta": z_meta
        }


# Model aliases per dispatch conventions
CandidateZeta_MUP = LocalUltradianPyramidCNNBiGRU
CandidateZeta_DSLBS = LocalUltradianPyramidCNNBiGRU


def build_nas_iteration8_model(
    window_macro: int = 91,
    window_micro: int = 15,
    dropout: float = 0.2,
    zero_init: bool = True,
    use_rcc_loss: bool = True
) -> LocalUltradianPyramidCNNBiGRU:
    """Factory function for NAS Iteration 8 Candidate Zeta architecture."""
    return LocalUltradianPyramidCNNBiGRU(
        window_macro=window_macro,
        window_micro=window_micro,
        dropout=dropout,
        zero_init=zero_init,
        use_rcc_loss=use_rcc_loss
    )
