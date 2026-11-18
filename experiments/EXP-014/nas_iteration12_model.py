r"""
experiments/EXP-014/nas_iteration12_model.py
NAS Iteration 12 Architecture: Candidate Kappa (BoundedMarginSynergyPyramidCNNBiGRU)

Grounding:
  Hypothesis H-F (Cross-Modal Bounded-Margin Synergy - BM-CMS):
    - Diagnosed Deficiency of Candidate Theta (Iteration 10) & Candidate Iota (Iteration 11):
      * Candidate Theta introduced CASCI with Dual-Scale ACC DSLBS, setting an all-time outer-test record
        of 66.84% Acc / 0.6677 WF1 (67.01% Acc / 0.6684 WF1 at Epoch 50) and reducing Light->REM false alarms
        by 33.6% (from 2,829 to 1,879 windows).
        HOWEVER, CASCI's rigid zero-threshold ReLU multiplication (s_rem = ReLU(u_ihr) * ReLU(-u_acc)) caused
        transient twitches (u_acc > 0) to instantly collapse REM synergy to 0.0, surging REM->Light false negatives
        to 2,459 windows.
      * Candidate Iota replaced ReLU with softplus (s_rem = softplus(u_ihr) * softplus(-u_acc)). However, because
        softplus has a strictly positive lower bound (softplus(u) > 0 everywhere), synergy leaked forward in time,
        causing Light->REM false alarms to surge by +28.4% (from 1,879 back up to 2,412 windows), dragging accuracy
        down to 65.12%.
    - Bounded-Margin Cross-Modal Synergy (BM-CMS):
      Candidate Kappa synthesizes the strengths of Theta and Iota while curing their defects:
        1. Sharp ReLU clipping is retained to provide a strict zero lower-bound, preventing synergy leakage
           into Light sleep.
        2. Fixed zero thresholds are replaced with learned asymmetric physiological margins \tau:
           * REM synergy:       s_rem   = ReLU(u_ihr - \tau_ihr_rem) * ReLU(\tau_acc_rem - u_acc)
             where \tau_acc_rem is initialized to +0.25 (allowing physiological twitches up to +0.25
             without killing REM synergy, but strictly zeroing once real body movement u_acc > \tau_acc_rem occurs!).
           * Deep sleep synergy:s_deep  = ReLU(\tau_ihr_deep - u_ihr) * ReLU(\tau_acc_deep - u_acc)
             where \tau_acc_deep is initialized to +0.05.
           * Wake synergy:      s_wake  = ReLU(u_ihr - \tau_ihr_wake) * ReLU(u_acc - \tau_acc_wake)
           * Light sleep synergy:s_light= ReLU(\tau_ihr_light - u_ihr) * ReLU(u_acc - \tau_acc_light)
        3. All 8 margins \tau are learnable parameters initialized to physiologically sound priors.

  Hypothesis H-A (Subject-Invariant Dual-Scale Basal Standardization & Multi-Scale Pyramids):
    - Retains 100% of Candidate Theta's verified foundation:
      * Dual-Scale Local Basal Standardization (DSLBS for IHR, W_micro=15, W_macro=91, 16 params)
      * Dual-Scale Accelerometry Standardization (Dual-Scale ACC DSLBS, 0 params, scale-invariant)
      * Multi-Scale Ultradian Temporal Pyramids (MUP)
      * Latent MUP LUAN on HR 64 & ACC 64 (640 params)
      * Dynamic Circadian Phase Gating (DCPG with 6-dim MUP context, 2,592 params)
      * 2-layer Bidirectional GRU (hidden 48 -> 96, 97,920 params)
      * Inter-Epoch Sequence Multi-Head Attention (IESA, d_attn=48, 4 heads, 16,416 params)
      * Direct 144-dimensional intra-epoch feature skip highway into multi-stream classifier head (18,884 params)

Architectural Budget & Compression:
  - Baseline Model A: 163,940 parameters
  - Candidate Theta (Iter 10): 196,373 parameters
  - Candidate Kappa (Iter 12, Linear Head): Exactly 196,381 parameters (+8 params over Theta)
  - Candidate Kappa (Iter 12, Cosine Head): Exactly 196,378 parameters (+5 params over Theta)
  - Target Budget: Strictly compliant (196,381 <= 196,500 target budget, 119 params headroom)
  - Hard Budget Cap: Strictly compliant (196,381 <= 196,728 cap, 347 params headroom)
  - Parameter Compression: 96.41% compression vs 5.47M published SLAMSS-IFS benchmark.
"""

import os
import sys
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Dynamically resolve workspace root
cur_dir = os.path.dirname(os.path.abspath(__file__))
root_candidate1 = os.path.abspath(os.path.join(cur_dir, "../.."))
root_candidate2 = os.path.abspath(os.path.join(cur_dir, "../../.."))
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
    Dual-Scale Local Basal Standardization (DSLBS).
    Separates short-term ultradian fluctuations (W_micro = 15 epochs = 7.5 min)
    from macro-scale circadian and basal trends (W_macro = 91 epochs = 45.5 min).
    Trainable parameters: Exactly 16.
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

        self.register_buffer("kernel_macro", torch.ones(1, 1, window_macro))
        self.register_buffer("kernel_micro", torch.ones(1, 1, window_micro))

        self.gamma_ihr = nn.Parameter(torch.ones(1))
        self.beta_ihr = nn.Parameter(torch.zeros(1))
        self.alpha_ihr_diff = nn.Parameter(torch.zeros(1))

        self.gamma_meta_mean = nn.Parameter(torch.ones(1))
        self.beta_meta_mean = nn.Parameter(torch.zeros(1))
        self.alpha_meta_diff = nn.Parameter(torch.zeros(1))
        self.gamma_meta_std = nn.Parameter(torch.ones(1))
        self.beta_meta_std = nn.Parameter(torch.zeros(1))

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
        B, T, L = x_ihr.shape
        if mask is None:
            m = torch.ones(B, 1, T, dtype=x_ihr.dtype, device=x_ihr.device)
            m_mask = torch.ones(B, T, dtype=x_ihr.dtype, device=x_ihr.device)
        else:
            m = mask.unsqueeze(1).to(dtype=x_ihr.dtype)
            m_mask = mask.to(dtype=x_ihr.dtype)

        hr_epoch = x_meta[:, :, 3:4].transpose(1, 2)
        masked_hr = hr_epoch * m

        # Macro-Scale Running Statistics (W_macro = 91)
        macro_sum = F.conv1d(masked_hr, self.kernel_macro, padding=self.pad_macro)
        macro_count = F.conv1d(m, self.kernel_macro, padding=self.pad_macro).clamp(min=1.0)
        mu_macro = macro_sum / macro_count
        mu_mac = mu_macro.transpose(1, 2)

        sample_sq_diff_macro = ((x_ihr - mu_mac) ** 2).mean(dim=-1, keepdim=True).transpose(1, 2)
        macro_var = F.conv1d(sample_sq_diff_macro * m, self.kernel_macro, padding=self.pad_macro) / macro_count
        sigma_macro = torch.sqrt(macro_var + self.eps)
        sig_mac = sigma_macro.transpose(1, 2)

        # Micro-Scale Running Statistics (W_micro = 15)
        micro_sum = F.conv1d(masked_hr, self.kernel_micro, padding=self.pad_micro)
        micro_count = F.conv1d(m, self.kernel_micro, padding=self.pad_micro).clamp(min=1.0)
        mu_micro = micro_sum / micro_count
        mu_mic = mu_micro.transpose(1, 2)

        sample_sq_diff_micro = ((x_ihr - mu_mic) ** 2).mean(dim=-1, keepdim=True).transpose(1, 2)
        micro_var = F.conv1d(sample_sq_diff_micro * m, self.kernel_micro, padding=self.pad_micro) / micro_count
        sigma_micro = torch.sqrt(micro_var + self.eps)
        sig_mic = sigma_micro.transpose(1, 2)

        # Whole-Night Global Baseline
        night_count = m.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mu_night = (hr_epoch * m).sum(dim=-1, keepdim=True) / night_count
        mu_nit = mu_night.transpose(1, 2)

        sample_sq_diff_night = ((x_ihr - mu_nit) ** 2).mean(dim=-1, keepdim=True).transpose(1, 2)
        var_night = (sample_sq_diff_night * m).sum(dim=-1, keepdim=True) / night_count
        sigma_night = torch.sqrt(var_night + self.eps)
        sig_nit = sigma_night.transpose(1, 2)

        m_3d = m_mask.unsqueeze(-1)

        # Standardize IHR
        z_ihr_macro = (x_ihr - mu_mac) / sig_mac
        z_ihr_micro = (x_ihr - mu_mic) / sig_mic
        delta_z_ihr = z_ihr_micro - z_ihr_macro
        z_ihr = (z_ihr_macro + self.alpha_ihr_diff * delta_z_ihr) * self.gamma_ihr + self.beta_ihr
        z_ihr = z_ihr * m_3d

        # Autonomic Contrast Differential
        delta_z = (mu_mic - mu_mac) / sig_mac
        z_epoch_macro = (x_meta[:, :, 3:4] - mu_mac) / sig_mac
        z_macro_night = (mu_mac - mu_nit) / sig_nit

        # Standardize Meta features
        norm_meta_mean = (z_epoch_macro + self.alpha_meta_diff * delta_z) * self.gamma_meta_mean + self.beta_meta_mean
        norm_meta_std = (x_meta[:, :, 4:5] / sig_mac) * self.gamma_meta_std + self.beta_meta_std
        z_meta = torch.cat([x_meta[:, :, :3], norm_meta_mean, norm_meta_std], dim=-1) * m_3d

        # Context features
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


class DualScaleAccelerometryDSLBS(nn.Module):
    r"""
    Subject-Invariant Dual-Scale Accelerometry Standardization (Dual-Scale ACC DSLBS).
    Eliminates static DC gravity/tilt offset via within-epoch variance,
    and eliminates arbitrary sensor gain via scale-free contrast ratio.
    Trainable parameters: Exactly 0 (buffer only).
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
        self.eps = eps

        # Non-trainable 1D box filters
        self.register_buffer("kernel_macro", torch.ones(1, 1, window_macro))
        self.register_buffer("kernel_micro", torch.ones(1, 1, window_micro))

    def forward(
        self,
        x_accel: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x_accel: (B, T, 30)
        B, T, L = x_accel.shape
        if mask is None:
            m = torch.ones(B, 1, T, dtype=x_accel.dtype, device=x_accel.device)
            m_3d = torch.ones(B, T, 1, dtype=x_accel.dtype, device=x_accel.device)
        else:
            m = mask.unsqueeze(1).to(dtype=x_accel.dtype)
            m_3d = mask.unsqueeze(-1).to(dtype=x_accel.dtype)

        # 1. Within-epoch dynamic kinetic activity (std across 30 samples)
        mean_epoch = x_accel.mean(dim=-1, keepdim=True)
        a_t = torch.sqrt(torch.mean((x_accel - mean_epoch) ** 2, dim=-1, keepdim=True) + 1e-12)  # (B, T, 1)
        a_1d = a_t.transpose(1, 2)  # (B, 1, T)
        masked_a = a_1d * m

        # 2. Causal Macro-Scale Running Statistics (W_macro = 91 epochs = 45.5 min)
        padded_a_macro = F.pad(masked_a, (self.window_macro - 1, 0))
        padded_m_macro = F.pad(m, (self.window_macro - 1, 0))
        macro_sum = F.conv1d(padded_a_macro, self.kernel_macro, padding=0)
        macro_count = F.conv1d(padded_m_macro, self.kernel_macro, padding=0).clamp(min=1.0)
        mu_macro = (macro_sum / macro_count).transpose(1, 2)  # (B, T, 1)

        padded_sq_macro = F.pad(((a_t - mu_macro) ** 2).transpose(1, 2) * m, (self.window_macro - 1, 0))
        macro_var = F.conv1d(padded_sq_macro, self.kernel_macro, padding=0) / macro_count
        sigma_macro = torch.sqrt(macro_var.clamp(min=1e-12)).transpose(1, 2)  # (B, T, 1)

        # 3. Causal Micro-Scale Running Statistics (W_micro = 15 epochs = 7.5 min)
        padded_a_micro = F.pad(masked_a, (self.window_micro - 1, 0))
        padded_m_micro = F.pad(m, (self.window_micro - 1, 0))
        micro_sum = F.conv1d(padded_a_micro, self.kernel_micro, padding=0)
        micro_count = F.conv1d(padded_m_micro, self.kernel_micro, padding=0).clamp(min=1.0)
        mu_micro = (micro_sum / micro_count).transpose(1, 2)  # (B, T, 1)

        # 4. Strictly Causal Scale-Free Somatomotor Contrast Differential
        delta_z_acc = ((mu_micro - mu_macro) / (sigma_macro + self.eps)) * m_3d
        z_macro_acc = (mu_macro / (sigma_macro + self.eps)) * m_3d

        return delta_z_acc, z_macro_acc


class MultiScaleLocalUltradianLatentNorm(nn.Module):
    r"""
    Subject-Invariant Multi-Scale Local Ultradian Latent Normalization (Latent MUP LUAN).
    Retained directly from Candidate Zeta (640 trainable parameters total for HR and ACC).
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

        self.register_buffer("kernel_macro", torch.ones(num_features, 1, window_macro))
        self.register_buffer("kernel_micro", torch.ones(num_features, 1, window_micro))

        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.alpha = nn.Parameter(torch.zeros(num_features))
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        if mask is None:
            m = torch.ones(B, 1, T, dtype=x.dtype, device=x.device)
            m_3d = torch.ones(B, T, 1, dtype=x.dtype, device=x.device)
        else:
            m = mask.unsqueeze(1).to(dtype=x.dtype)
            m_3d = mask.unsqueeze(-1).to(dtype=x.dtype)

        x_1d = x.transpose(1, 2)
        x_masked = x_1d * m

        # Macro
        macro_sum = F.conv1d(x_masked, self.kernel_macro, padding=self.pad_macro, groups=C)
        macro_count = F.conv1d(m, self.kernel_macro[:1], padding=self.pad_macro, groups=1).clamp(min=1.0)
        macro_mean = macro_sum / macro_count
        sq_macro = ((x_1d - macro_mean) ** 2) * m
        macro_var = F.conv1d(sq_macro, self.kernel_macro, padding=self.pad_macro, groups=C) / macro_count
        macro_std = torch.sqrt(macro_var + self.eps)
        x_norm_macro = ((x_1d - macro_mean) / macro_std).transpose(1, 2) * m_3d

        # Micro
        micro_sum = F.conv1d(x_masked, self.kernel_micro, padding=self.pad_micro, groups=C)
        micro_count = F.conv1d(m, self.kernel_micro[:1], padding=self.pad_micro, groups=1).clamp(min=1.0)
        micro_mean = micro_sum / micro_count
        sq_micro = ((x_1d - micro_mean) ** 2) * m
        micro_var = F.conv1d(sq_micro, self.kernel_micro, padding=self.pad_micro, groups=C) / micro_count
        micro_std = torch.sqrt(micro_var + self.eps)
        x_norm_micro = ((x_1d - micro_mean) / micro_std).transpose(1, 2) * m_3d

        delta_x_norm = x_norm_micro - x_norm_macro
        out = (x_norm_macro + self.alpha * delta_x_norm) * self.weight + self.bias
        res = self.norm(x + out)
        if mask is not None:
            res = res * m_3d
        return res


class BoundedMarginCrossModalSynergy(nn.Module):
    r"""
    Bounded-Margin Cross-Modal Autonomic-Somatomotor Synergy (BM-CMS).
    Solves both Candidate Theta's rigid zero-threshold flaw and Candidate Iota's softplus leakage flaw:
      - Sharp ReLU gating is retained to provide a strict zero lower-bound, preventing synergy leakage
        into Light sleep.
      - Fixed zero thresholds are replaced with learned asymmetric physiological margins \tau:
        * REM synergy:       s_rem   = ReLU(u_ihr - \tau_ihr_rem) * ReLU(\tau_acc_rem - u_acc)
          where \tau_acc_rem is initialized to +0.25 (allowing physiological twitches up to +0.25
          without killing REM synergy, but strictly zeroing once real body movement u_acc > \tau_acc_rem occurs!).
        * Deep sleep synergy:s_deep  = ReLU(\tau_ihr_deep - u_ihr) * ReLU(\tau_acc_deep - u_acc)
        * Wake synergy:      s_wake  = ReLU(u_ihr - \tau_ihr_wake) * ReLU(u_acc - \tau_acc_wake)
        * Light sleep synergy:s_light= ReLU(\tau_ihr_light - u_ihr) * ReLU(u_acc - \tau_acc_light)
    Trainable parameters: Exactly 489 parameters.
    """
    def __init__(
        self,
        in_features: int = 8,
        hidden_dim: int = 3,
        out_features: int = 144,
        n_classes: int = 4,
        zero_init: bool = True,
        use_logit_mod: bool = True,
        init_tau_acc_rem: float = 0.25,
        init_tau_ihr_rem: float = 0.0,
        init_tau_acc_deep: float = 0.05,
        init_tau_ihr_deep: float = 0.0,
        init_tau_acc_wake: float = 0.0,
        init_tau_ihr_wake: float = 0.0,
        init_tau_acc_light: float = 0.0,
        init_tau_ihr_light: float = 0.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.out_features = out_features
        self.n_classes = n_classes
        self.use_logit_mod = use_logit_mod

        # 8 Learnable Asymmetric Margins
        self.tau_ihr_rem = nn.Parameter(torch.tensor(float(init_tau_ihr_rem)))
        self.tau_acc_rem = nn.Parameter(torch.tensor(float(init_tau_acc_rem)))
        self.tau_ihr_deep = nn.Parameter(torch.tensor(float(init_tau_ihr_deep)))
        self.tau_acc_deep = nn.Parameter(torch.tensor(float(init_tau_acc_deep)))
        self.tau_ihr_wake = nn.Parameter(torch.tensor(float(init_tau_ihr_wake)))
        self.tau_acc_wake = nn.Parameter(torch.tensor(float(init_tau_acc_wake)))
        self.tau_ihr_light = nn.Parameter(torch.tensor(float(init_tau_ihr_light)))
        self.tau_acc_light = nn.Parameter(torch.tensor(float(init_tau_acc_light)))

        self.fc1 = nn.Linear(in_features, hidden_dim, bias=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_features, bias=False)

        if use_logit_mod:
            self.fc_logit = nn.Linear(hidden_dim, n_classes, bias=True)

        if zero_init:
            nn.init.zeros_(self.fc2.weight)
            if use_logit_mod:
                nn.init.zeros_(self.fc_logit.weight)
                nn.init.zeros_(self.fc_logit.bias)
        else:
            nn.init.normal_(self.fc2.weight, std=0.01)
            if use_logit_mod:
                nn.init.normal_(self.fc_logit.weight, std=0.01)
                nn.init.zeros_(self.fc_logit.bias)

    def forward(
        self,
        e_intra: torch.Tensor,
        delta_z_ihr: torch.Tensor,
        z_macro_ihr: torch.Tensor,
        delta_z_acc: torch.Tensor,
        z_macro_acc: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, torch.Tensor]:
        # e_intra: (B, T, 144)
        B, T, _ = e_intra.shape

        u_ihr = delta_z_ihr
        u_acc = delta_z_acc

        # 4-Quadrant Bounded Margin Synergy Terms
        s_rem = F.relu(u_ihr - self.tau_ihr_rem) * F.relu(self.tau_acc_rem - u_acc)
        s_deep = F.relu(self.tau_ihr_deep - u_ihr) * F.relu(self.tau_acc_deep - u_acc)
        s_wake = F.relu(u_ihr - self.tau_ihr_wake) * F.relu(u_acc - self.tau_acc_wake)
        s_light = F.relu(self.tau_ihr_light - u_ihr) * F.relu(u_acc - self.tau_acc_light)

        synergies = torch.cat([s_rem, s_deep, s_wake, s_light], dim=-1)  # (B, T, 4)

        # Context vector: (B, T, 8)
        c_bm = torch.cat([
            u_ihr, u_acc, z_macro_ihr, z_macro_acc,
            s_rem, s_deep, s_wake, s_light
        ], dim=-1)

        h = self.act(self.norm(self.fc1(c_bm)))      # (B, T, 3)
        gate = torch.sigmoid(self.fc2(h))            # (B, T, 144)
        e_bm = e_intra * (2.0 * gate)
        delta_logits = self.fc_logit(h) if self.use_logit_mod else None

        if mask is not None:
            m_3d = mask.unsqueeze(-1).to(dtype=e_bm.dtype)
            e_bm = e_bm * m_3d
            gate = gate * m_3d
            if delta_logits is not None:
                delta_logits = delta_logits * m_3d
            c_bm = c_bm * m_3d
            synergies = synergies * m_3d

        return e_bm, gate, delta_logits, c_bm, synergies


class DynamicCircadianPhaseGating(nn.Module):
    r"""
    Dynamic Circadian Phase Gating (DCPG).
    Retained directly from Candidate Zeta (2,592 trainable parameters).
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
        h = self.act(self.norm(self.fc1(circ_ctx)))
        gate = torch.sigmoid(self.fc2(h))
        e_gated = e_intra * (2.0 * gate)
        if mask is not None:
            e_gated = e_gated * mask.unsqueeze(-1).to(dtype=e_gated.dtype)
        return e_gated, gate


class InterEpochSequenceAttention(nn.Module):
    r"""
    Inter-Epoch Sequence-Level Multi-Head Attention (IESA).
    Retained directly from Candidate Zeta (16,416 trainable parameters).
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


class AdaptiveCosineClassifierHead(nn.Module):
    r"""
    Adaptive Cosine / Linear Multi-Stream Classification Head for Sleep Staging.
    Default mode: Linear head (18,884 params) for exact parity with Candidate Theta.
    Optional mode: Cosine head with angular margin (18,881 params).
    """
    def __init__(
        self,
        in_features: int = 288,
        hidden_dim: int = 64,
        n_classes: int = 4,
        dropout: float = 0.2,
        use_cosine: bool = False,
        scale: float = 16.0,
        margin: float = 0.10,
        learnable_scale: bool = True
    ):
        super().__init__()
        self.use_cosine = use_cosine
        self.margin = margin
        self.n_classes = n_classes
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        if use_cosine:
            self.weight = nn.Parameter(torch.empty(n_classes, hidden_dim))
            nn.init.xavier_uniform_(self.weight)
            if learnable_scale:
                self.scale = nn.Parameter(torch.tensor(scale))
            else:
                self.register_buffer("scale", torch.tensor(scale))
        else:
            self.linear = nn.Linear(hidden_dim, n_classes, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self.in_proj(x)
        if self.use_cosine:
            h_norm = F.normalize(h, p=2, dim=-1)
            w_norm = F.normalize(self.weight, p=2, dim=-1)
            cosine = F.linear(h_norm, w_norm)  # (B, T, 4)
            if self.training and targets is not None and self.margin > 0.0:
                one_hot = F.one_hot(targets.clamp(min=0, max=self.n_classes - 1), num_classes=self.n_classes).float()
                cosine_m = cosine - one_hot * self.margin
                logits = self.scale * cosine_m
            else:
                logits = self.scale * cosine
            return logits
        else:
            return self.linear(h)


class BoundedMarginSynergyPyramidCNNBiGRU(nn.Module):
    r"""
    NAS Iteration 12 Candidate Kappa Architecture.
    Bounded-Margin Cross-Modal Synergy Architecture.
    Total Parameters: Exactly 196,381 (with default linear classifier head).
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
        use_rcc_loss: bool = True,
        use_logit_mod: bool = True,
        use_cosine_head: bool = False,
        cosine_scale: float = 16.0,
        cosine_margin: float = 0.10,
        init_tau_acc_rem: float = 0.25,
        init_tau_ihr_rem: float = 0.0,
        init_tau_acc_deep: float = 0.05,
        init_tau_ihr_deep: float = 0.0,
        init_tau_acc_wake: float = 0.0,
        init_tau_ihr_wake: float = 0.0,
        init_tau_acc_light: float = 0.0,
        init_tau_ihr_light: float = 0.0,
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
        self.use_logit_mod = use_logit_mod
        self.use_cosine_head = use_cosine_head

        base = ConventionalCNNBiGRU(
            d_model=d_model,
            d_temp=d_temp,
            n_classes=n_classes,
            n_meta=n_meta,
            dropout=dropout
        )
        self.hr_encoder = base.hr_encoder
        self.acc_encoder = base.acc_encoder
        self.meta_proj = base.meta_proj

        # 1. IHR DSLBS (16 params)
        self.basal_dslbs = DualScaleLocalBasalStandardization(
            window_macro=window_macro,
            window_micro=window_micro
        )

        # 2. ACC DSLBS (0 params, scale-invariant)
        self.acc_dslbs = DualScaleAccelerometryDSLBS(
            window_macro=window_macro,
            window_micro=window_micro
        )

        # 3. Latent MUP LUAN (640 params)
        self.luan_hr = MultiScaleLocalUltradianLatentNorm(
            d_model, window_macro=window_macro, window_micro=window_micro
        )
        self.luan_acc = MultiScaleLocalUltradianLatentNorm(
            d_model, window_macro=window_macro, window_micro=window_micro
        )

        # 4. Bounded-Margin Cross-Modal Synergy (BM-CMS, 489 params)
        in_temp = 2 * d_model + 16  # 144
        self.bm_casci = BoundedMarginCrossModalSynergy(
            in_features=8,
            hidden_dim=3,
            out_features=in_temp,
            n_classes=n_classes,
            zero_init=zero_init,
            use_logit_mod=use_logit_mod,
            init_tau_acc_rem=init_tau_acc_rem,
            init_tau_ihr_rem=init_tau_ihr_rem,
            init_tau_acc_deep=init_tau_acc_deep,
            init_tau_ihr_deep=init_tau_ihr_deep,
            init_tau_acc_wake=init_tau_acc_wake,
            init_tau_ihr_wake=init_tau_ihr_wake,
            init_tau_acc_light=init_tau_acc_light,
            init_tau_ihr_light=init_tau_ihr_light
        )

        # 5. Dynamic Circadian Phase Gating (2,592 params)
        self.circadian_gate = DynamicCircadianPhaseGating(
            in_features=6,
            hidden_dim=16,
            out_features=in_temp,
            zero_init=zero_init
        )

        # 6. BiGRU (97,920 params)
        self.temporal_gru = nn.GRU(
            input_size=in_temp,
            hidden_size=d_temp // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.temp_norm = nn.LayerNorm(d_temp)

        # 7. IESA (16,416 params)
        self.inter_epoch_attn = InterEpochSequenceAttention(
            d_in=d_temp,
            d_attn=d_attn,
            n_heads=n_heads,
            dropout=dropout,
            zero_init=zero_init
        )

        # 8. Multi-Stream Classifier Head (18,884 params in linear mode, 18,881 params in cosine mode)
        in_clf = d_temp + d_attn + in_temp
        self.classifier_head = AdaptiveCosineClassifierHead(
            in_features=in_clf,
            hidden_dim=64,
            n_classes=n_classes,
            dropout=dropout,
            use_cosine=use_cosine_head,
            scale=cosine_scale,
            margin=cosine_margin
        )

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
        if isinstance(x_ihr, dict):
            batch = x_ihr
            x_ihr = batch["ihr"]
            x_accel = batch["accel"]
            x_meta = batch["meta"]
            targets = batch.get("y", targets)
            mask = batch.get("mask", mask)
            seq_lens = batch.get("seq_len", seq_lens)
        elif x_accel is not None and x_accel.shape[-1] == 5:
            actual_mask = targets if mask is None else mask
            if x_meta is not None and x_meta.dtype == torch.bool:
                actual_mask = x_meta
            x_meta = x_accel
            mask = actual_mask
            targets = None
            if x_ihr.shape[-1] == 60:
                x_accel = x_ihr[:, :, 30:60]
                x_ihr = x_ihr[:, :, :30]
        elif x_accel is None and x_meta is not None:
            if x_ihr.shape[-1] == 60:
                x_accel = x_ihr[:, :, 30:60]
                x_ihr = x_ihr[:, :, :30]

        if targets is not None and targets.dtype == torch.bool:
            mask = targets
            targets = None

        B, T, _ = x_ihr.shape

        # Step 1: Dual-Scale IHR DSLBS & Scale-Free ACC DSLBS
        z_ihr, z_meta, mup_ctx = self.basal_dslbs(x_ihr, x_meta, mask=mask)
        delta_z_acc, z_macro_acc = self.acc_dslbs(x_accel, mask=mask)

        delta_z_ihr = mup_ctx[:, :, 3:4]
        z_macro_ihr = mup_ctx[:, :, 4:5]

        # Step 2: Modality Encoders
        h_hr = self.hr_encoder(z_ihr)
        h_acc = self.acc_encoder(x_accel)
        h_meta = self.meta_proj(z_meta)

        # Step 3: Latent Multi-Scale Ultradian Temporal Pyramid Normalization
        z_hr = self.luan_hr(h_hr, mask=mask)
        z_acc = self.luan_acc(h_acc, mask=mask)

        # Step 4: Intra-Epoch representation (144 dims)
        e_intra = torch.cat([z_hr, z_acc, h_meta], dim=-1)

        # Step 5: Bounded-Margin Cross-Modal Synergy
        e_bm, bm_gate, delta_logits, bm_ctx, synergies = self.bm_casci(
            e_intra, delta_z_ihr, z_macro_ihr, delta_z_acc, z_macro_acc, mask=mask
        )

        # Step 6: Dynamic Circadian Phase Gating
        e_gated, circ_gate = self.circadian_gate(e_bm, mup_ctx, mask=mask)

        # Step 7: Temporal BiGRU with Packing Support
        if seq_lens is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                e_gated, seq_lens.to("cpu", dtype=torch.int64), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.temporal_gru(packed)
            temp_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
        else:
            temp_out, _ = self.temporal_gru(e_gated)

        temp_features = self.temp_norm(temp_out)
        if mask is not None:
            temp_features = temp_features * mask.unsqueeze(-1).to(dtype=temp_features.dtype)

        # Step 8: Inter-Epoch Attention
        attn_context, attn_weights = self.inter_epoch_attn(temp_features, mask=mask)

        # Step 9: Multi-Stream Skip Integration (288 dims)
        combined = torch.cat([temp_features, attn_context, e_bm], dim=-1)

        # Step 10: Classification Logits Emission & Prior Modulation
        logits = self.classifier_head(combined, targets=targets)
        if delta_logits is not None:
            logits = logits + delta_logits

        # Step 11: Multi-Task Loss Calculation
        loss_stage = torch.tensor(0.0, device=x_ihr.device)
        if targets is not None:
            flat_logits = logits.view(-1, self.n_classes)
            flat_targets = targets.view(-1)
            valid = (flat_targets >= 0) & (flat_targets < self.n_classes)
            if mask is not None:
                valid = valid & mask.view(-1)
            if valid.any():
                per_epoch_ce = self.ce_loss(flat_logits, torch.clamp(flat_targets, 0, self.n_classes - 1))
                ce_term = per_epoch_ce.view(B, T)[valid.view(B, T)].mean()
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
            "loss": loss_stage,
            "total_loss": loss_stage,
            "loss_stage": loss_stage,
            "bm_gate": bm_gate,
            "circ_gate": circ_gate,
            "attn_weights": attn_weights,
            "delta_z_acc": delta_z_acc,
            "delta_z_ihr": delta_z_ihr,
            "bm_ctx": bm_ctx,
            "synergies": synergies
        }


# Model aliases per dispatch conventions
CandidateKappa_BMCMS = BoundedMarginSynergyPyramidCNNBiGRU
CandidateKappa_Pyramid = BoundedMarginSynergyPyramidCNNBiGRU


def build_nas_iteration12_model(
    window_macro: int = 91,
    window_micro: int = 15,
    dropout: float = 0.2,
    zero_init: bool = True,
    use_rcc_loss: bool = True,
    use_logit_mod: bool = True,
    use_cosine_head: bool = False,
    cosine_scale: float = 16.0,
    cosine_margin: float = 0.10,
    init_tau_acc_rem: float = 0.25,
    init_tau_ihr_rem: float = 0.0,
    init_tau_acc_deep: float = 0.05,
    init_tau_ihr_deep: float = 0.0,
    init_tau_acc_wake: float = 0.0,
    init_tau_ihr_wake: float = 0.0,
    init_tau_acc_light: float = 0.0,
    init_tau_ihr_light: float = 0.0,
) -> BoundedMarginSynergyPyramidCNNBiGRU:
    """Factory function for NAS Iteration 12 Candidate Kappa architecture."""
    return BoundedMarginSynergyPyramidCNNBiGRU(
        window_macro=window_macro,
        window_micro=window_micro,
        dropout=dropout,
        zero_init=zero_init,
        use_rcc_loss=use_rcc_loss,
        use_logit_mod=use_logit_mod,
        use_cosine_head=use_cosine_head,
        cosine_scale=cosine_scale,
        cosine_margin=cosine_margin,
        init_tau_acc_rem=init_tau_acc_rem,
        init_tau_ihr_rem=init_tau_ihr_rem,
        init_tau_acc_deep=init_tau_acc_deep,
        init_tau_ihr_deep=init_tau_ihr_deep,
        init_tau_acc_wake=init_tau_acc_wake,
        init_tau_ihr_wake=init_tau_ihr_wake,
        init_tau_acc_light=init_tau_acc_light,
        init_tau_ihr_light=init_tau_ihr_light,
    )
