"""
EXP-013 Model Architecture Definitions & Factory
File: exp013_models.py

Defines the model matrix for EXP-013 Component Attribution:
  - Model A: Matched-capacity conventional CNN-BiGRU (no gate, direct concatenation, no transition, no GRL)
  - Model B: Full adaptive fusion + transition, NO GRL
  - Model C: Adaptive fusion + GRL, NO transition
  - Model D: Fixed equal fusion (0.5/0.5) + transition + GRL
  - Model E: Frozen EXP-012 B0_GRL reference (DO NOT RETRAIN)
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from sleep_adaptive_net import (
    HREncoder,
    ACCEncoder,
    RCCLoss,
    get_official_weights,
    SleepAdaptiveNet
)


class ConventionalCNNBiGRU(nn.Module):
    """
    Model A: Matched-capacity conventional CNN-BiGRU baseline.
    Controls approximately for parameter capacity (~160k parameters, within +/-10% of 162,325)
    to test whether proposed mechanisms provide benefit beyond a conventional architecture.
    
    Architecture:
      HR / ACC input (1 Hz, 30-sample epochs)
      -> HREncoder / ACCEncoder (identical CNNs to SleepAdaptiveNet, d_model=64)
      -> Meta feature projection (Linear(5, 16) + GELU)
      -> Concatenation: [h_HR ; h_ACC ; h_meta] (dimension = 64 + 64 + 16 = 144)
      -> 2-layer Bidirectional GRU (input_size=144, hidden_size=48 -> d_temp=96)
      -> LayerNorm(96)
      -> Classifier Head (Linear(96, 64) -> LayerNorm(64) -> GELU -> Dropout -> Linear(64, 4))
      
    Components explicitly absent:
      - No adaptive reliability gate (direct concatenation)
      - No transition module
      - No subject-invariant GRL
      - No boundary loss
      - No uncertainty head
    """
    def __init__(
        self,
        d_model: int = 64,
        d_temp: int = 96,
        n_classes: int = 4,
        n_meta: int = 5,
        dropout: float = 0.2,
        use_rcc_loss: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_temp = d_temp
        self.n_classes = n_classes
        self.use_rcc_loss = use_rcc_loss
        
        # 1. Modality-specific Encoders (Identical to SleepAdaptiveNet)
        self.hr_encoder = HREncoder(in_features=1, d_model=d_model, dropout=dropout)
        self.acc_encoder = ACCEncoder(in_features=1, d_model=d_model, dropout=dropout)
        
        # Meta feature projection (Identical to SleepAdaptiveNet)
        self.meta_proj = nn.Sequential(
            nn.Linear(n_meta, 16),
            nn.GELU()
        )
        
        # 2. Conventional Concatenation & Temporal Sequence Modeling
        in_temp = 2 * d_model + 16 # 64 + 64 + 16 = 144
        self.temporal_gru = nn.GRU(
            input_size=in_temp,
            hidden_size=d_temp // 2, # bidirectional -> total d_temp = 96
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.temp_norm = nn.LayerNorm(d_temp)
        
        # 3. Primary Classifier Head (Identical to SleepAdaptiveNet)
        self.classifier_head = nn.Sequential(
            nn.Linear(d_temp, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        
        # Primary stage loss functions
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
        subject_idx: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass matching SleepAdaptiveNet return dictionary format.
        """
        B, T, _ = x_ihr.shape
        
        # 1. Conventional Modality Encoding
        h_hr = self.hr_encoder(x_ihr)     # (B, T, d_model)
        h_acc = self.acc_encoder(x_accel) # (B, T, d_model)
        h_meta = self.meta_proj(x_meta)   # (B, T, 16)
        
        # 2. Direct Feature Concatenation (No Adaptive Gate)
        cat_features = torch.cat([h_hr, h_acc, h_meta], dim=-1) # (B, T, 144)
        
        # 3. Temporal Sequence Modeling
        temp_out, _ = self.temporal_gru(cat_features)
        temp_features = self.temp_norm(temp_out) # (B, T, d_temp)
        
        # 4. Classification Emission Logits
        logits = self.classifier_head(temp_features) # (B, T, n_classes)
        
        # 5. Loss Calculation (RCCLoss identical to SleepAdaptiveNet)
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


def build_exp013_model(
    model_id: str,
    n_train_subjects: int = 31,
    loss_type: str = "rccloss"
) -> nn.Module:
    """
    Builds models corresponding strictly to the EXP-013 experimental matrix:
      - 'A': Matched-capacity conventional CNN-BiGRU (no gate, concat, no trans, no GRL)
      - 'B': Full adaptive fusion + transition, NO GRL
      - 'C': Adaptive fusion + GRL, NO transition
      - 'D': Fixed equal fusion (0.5/0.5) + transition + GRL
      - 'E': Frozen EXP-012 B0_GRL reference (Raises error if requested for training)
    """
    model_id = model_id.upper()
    
    if model_id == "A":
        return ConventionalCNNBiGRU(
            d_model=64,
            d_temp=96,
            n_classes=4,
            n_meta=5,
            dropout=0.2,
            use_rcc_loss=(loss_type == "rccloss")
        )
    elif model_id == "B":
        return SleepAdaptiveNet(
            d_model=64,
            d_temp=96,
            n_classes=4,
            n_meta=5,
            n_train_subjects=n_train_subjects,
            dropout=0.2,
            adaptive_fusion=True,
            transition_aware=True,
            boundary_aware=False,
            subject_invariant=False,
            uncertainty_head=False,
            lambda_transition=0.10,
            lambda_boundary=0.0,
            lambda_adv=0.0,
            lambda_conf=0.0,
            use_rcc_loss=(loss_type == "rccloss")
        )
    elif model_id == "C":
        return SleepAdaptiveNet(
            d_model=64,
            d_temp=96,
            n_classes=4,
            n_meta=5,
            n_train_subjects=n_train_subjects,
            dropout=0.2,
            adaptive_fusion=True,
            transition_aware=False,
            boundary_aware=False,
            subject_invariant=True,
            uncertainty_head=False,
            lambda_transition=0.0,
            lambda_boundary=0.0,
            lambda_adv=0.10,
            lambda_conf=0.0,
            use_rcc_loss=(loss_type == "rccloss")
        )
    elif model_id == "D":
        return SleepAdaptiveNet(
            d_model=64,
            d_temp=96,
            n_classes=4,
            n_meta=5,
            n_train_subjects=n_train_subjects,
            dropout=0.2,
            adaptive_fusion=False, # Activates fixed 0.5/0.5 equal fusion in AdaptiveReliabilityFusion
            transition_aware=True,
            boundary_aware=False,
            subject_invariant=True,
            uncertainty_head=False,
            lambda_transition=0.10,
            lambda_boundary=0.0,
            lambda_adv=0.10,
            lambda_conf=0.0,
            use_rcc_loss=(loss_type == "rccloss")
        )
    elif model_id == "E":
        raise ValueError(
            "Model E is the frozen EXP-012 reference. Do NOT retrain Model E. "
            "Use saved EXP-012 artifacts directly."
        )
    else:
        raise ValueError(f"Unknown model_id '{model_id}'. Expected one of ['A', 'B', 'C', 'D'].")
