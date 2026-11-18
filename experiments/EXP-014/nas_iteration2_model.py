"""
NAS Iteration 2: Mutated PyTorch Model Architecture (Candidate B - Cross-Modal FiLM Fusion)
File: experiments/EXP-014/nas_iteration2_model.py

Grounding:
  Hypothesis H-F (Cross-Modal IHR-Accelerometry Interaction Module):
    Sleep staging from wearable signals requires modeling physiological coupling between
    the central autonomic nervous system (instantaneous heart rate) and the peripheral
    somatomotor system (triaxial accelerometry). In REM sleep, somatomotor atonia coincides
    with autonomic storms; in Wake, high motor activity drives tachycardia; in Deep sleep,
    minimal motor activity coincides with low, stable heart rate.
    Feature-wise Linear Modulation (FiLM) dynamically conditions autonomic representations
    on motor activity, and somatomotor representations on autonomic tone.

  Hypothesis H-A (Subject-Invariant Adaptation across 47 Subjects):
    Normalized, scale-invariant cross-modal interactions with residual identity initialization
    and LayerNorm prevent subject-level baseline offsets (e.g. resting HR variance) from
    diverging optimization, maintaining stability across all 47 subjects.

Architectural Budget:
  Baseline Model A: 163,940 parameters
  CrossModalFiLMFusion module: 29,440 parameters
  Total Model A + FiLM Parameters: Exactly 193,380 parameters (+17.96% over baseline)
  Budget Cap: Strictly within 162k–196k (+20% cap = 196,728)
"""

import os
import sys
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


class CrossModalFiLMFusion(nn.Module):
    """
    Candidate B: Cross-Modal Feature-wise Linear Modulation (FiLM) with Residual Gating.
    Targeting Hypothesis H-F (autonomic-somatomotor coupling) and H-A (subject invariance).

    Modulation Mechanics:
      1. ACC conditions HR:
         [gamma_hr, beta_hr] = W2_hr * GELU(W1_hr * h_acc + b1_hr) + b2_hr
         h_hr_film = LayerNorm(h_hr + Dropout(GELU(gamma_hr * h_hr + beta_hr)))

      2. HR conditions ACC:
         [gamma_acc, beta_acc] = W2_acc * GELU(W1_acc * h_hr + b1_acc) + b2_acc
         h_acc_film = LayerNorm(h_acc + Dropout(GELU(gamma_acc * h_acc + beta_acc)))

      3. Joint Gating Refinement:
         [g_hr, g_acc] = Sigmoid(W_g2 * GELU(W_g1 * [h_hr_film ; h_acc_film] + b_g1) + b_g2)
         h_hr_out = h_hr_film * (1.0 + g_hr)
         h_acc_out = h_acc_film * (1.0 + g_acc)

    Identity Initialization:
      Initializes generator second linear projection layers (W2_hr, b2_hr, W2_acc, b2_acc)
      to zero so training starts near the identity of Baseline Model A.

    Parameters: 29,440.
    """
    def __init__(self, d_model: int = 64, d_cond: int = 32, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_cond = d_cond

        # 1. ACC conditions HR: gamma_hr and beta_hr (64 -> 32 -> 128)
        self.film_gen_hr = nn.Sequential(
            nn.Linear(d_model, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, 2 * d_model)
        )
        self.norm_hr = nn.LayerNorm(d_model)

        # 2. HR conditions ACC: gamma_acc and beta_acc (64 -> 32 -> 128)
        self.film_gen_acc = nn.Sequential(
            nn.Linear(d_model, d_cond),
            nn.GELU(),
            nn.Linear(d_cond, 2 * d_model)
        )
        self.norm_acc = nn.LayerNorm(d_model)

        # 3. Dynamic Cross-Modal Gating Network (128 -> 64 -> 128)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)

        # Identity initialization: zero-init second linear layer
        nn.init.zeros_(self.film_gen_hr[2].weight)
        nn.init.zeros_(self.film_gen_hr[2].bias)
        nn.init.zeros_(self.film_gen_acc[2].weight)
        nn.init.zeros_(self.film_gen_acc[2].bias)

    def forward(
        self,
        h_hr: torch.Tensor,
        h_acc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input:
          h_hr:  (B, T, d_model)
          h_acc: (B, T, d_model)
        Output:
          h_hr_out:  (B, T, d_model)
          h_acc_out: (B, T, d_model)
        """
        # Step 1: Affine feature modulation
        params_hr = self.film_gen_hr(h_acc)
        gamma_hr, beta_hr = torch.chunk(params_hr, 2, dim=-1)
        h_hr_film = self.norm_hr(h_hr + self.dropout(F.gelu(gamma_hr * h_hr + beta_hr)))

        params_acc = self.film_gen_acc(h_hr)
        gamma_acc, beta_acc = torch.chunk(params_acc, 2, dim=-1)
        h_acc_film = self.norm_acc(h_acc + self.dropout(F.gelu(gamma_acc * h_acc + beta_acc)))

        # Step 2: Joint Gating Refinement
        cat_features = torch.cat([h_hr_film, h_acc_film], dim=-1)
        gates = self.gate_net(cat_features)
        gate_hr, gate_acc = torch.chunk(gates, 2, dim=-1)

        h_hr_out = h_hr_film * (1.0 + gate_hr)
        h_acc_out = h_acc_film * (1.0 + gate_acc)

        return h_hr_out, h_acc_out


class CrossModalFiLMCNNBiGRU(nn.Module):
    """
    NAS Iteration 2 Mutated Architecture: Candidate B (Cross-Modal FiLM CNN-BiGRU).

    Extends Baseline Model A with CrossModalFiLMFusion before multimodal concatenation
    and temporal BiGRU sequence modeling.

    Total Parameter Count: Exactly 193,380 parameters.
    """
    def __init__(
        self,
        d_model: int = 64,
        d_temp: int = 96,
        n_classes: int = 4,
        n_meta: int = 5,
        dropout: float = 0.2,
        film_dropout: float = 0.1,
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

        # 2. Structural Mutation: Cross-Modal FiLM Fusion (29,440 params)
        self.cross_film = CrossModalFiLMFusion(
            d_model=d_model,
            d_cond=32,
            dropout=film_dropout
        )

        # 3. Temporal Sequence Modeling (BiGRU, input_size=144, hidden_size=48 -> 96, 97,920 params)
        in_temp = 2 * d_model + 16 # 64 + 64 + 16 = 144
        self.temporal_gru = nn.GRU(
            input_size=in_temp,
            hidden_size=d_temp // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.temp_norm = nn.LayerNorm(d_temp) # 192 params

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

        # 2. Cross-Modal FiLM Modulation & Gating
        h_hr_fused, h_acc_fused = self.cross_film(h_hr, h_acc) # (B, T, 64), (B, T, 64)

        # 3. Multimodal Concatenation
        cat_features = torch.cat([h_hr_fused, h_acc_fused, h_meta], dim=-1) # (B, T, 144)

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
CandidateB_FiLMFusion = CrossModalFiLMCNNBiGRU


def build_nas_iteration2_model() -> CrossModalFiLMCNNBiGRU:
    """
    Factory function for NAS Iteration 2 Candidate B architecture.
    """
    return CrossModalFiLMCNNBiGRU()


if __name__ == "__main__":
    print("Testing NAS Iteration 2 (Candidate B - CrossModalFiLMCNNBiGRU)...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    m = CrossModalFiLMCNNBiGRU().to(device)
    p = sum(param.numel() for param in m.parameters())
    print(f"Total Parameters: {p:,} (Baseline: 163,940, Delta: +{p - 163940:,}, +{(p - 163940)/163940*100:.2f}%)")
    assert p == 193380, f"Expected 193,380 parameters, got {p}"
    assert p <= 196728, f"Parameter count {p} exceeds 196,728 cap!"

    # Forward and backward gradient check
    B, T = 4, 600
    x_ihr = torch.randn(B, T, 30, device=device)
    x_acc = torch.randn(B, T, 30, device=device)
    x_meta = torch.randn(B, T, 5, device=device)
    targets = torch.randint(0, 4, (B, T), device=device)
    mask = torch.ones(B, T, dtype=torch.bool, device=device)
    seq_lens = torch.full((B,), T, dtype=torch.int64, device=device)

    out = m(x_ihr, x_acc, x_meta, targets=targets, mask=mask, seq_lens=seq_lens)
    assert out["logits"].shape == (B, T, 4), f"Unexpected shape {out['logits'].shape}"
    out["total_loss"].backward()
    assert m.cross_film.film_gen_hr[0].weight.grad is not None, "Missing grad in film_gen_hr!"
    assert m.cross_film.film_gen_acc[0].weight.grad is not None, "Missing grad in film_gen_acc!"
    assert m.cross_film.gate_net[0].weight.grad is not None, "Missing grad in gate_net!"
    print("All assertions passed successfully!")
