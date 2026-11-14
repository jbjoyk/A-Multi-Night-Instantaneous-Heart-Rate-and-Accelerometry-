"""
SleepAdaptiveNet: Proposed Modular Architecture for Wearable Sleep Staging.

Hypothesis:
    Wearable sleep staging can be improved by:
      1. Dynamically estimating the reliability of heart-rate and motion signals (Adaptive Reliability Fusion).
      2. Explicitly modeling sleep-stage transitions across temporal context (Transition-Aware Modeling).
      3. Applying targeted learning emphasis to difficult boundary epochs (Boundary-Aware Loss).
      4. Learning representations invariant to subject-specific physiology (Subject-Invariant Representation via GRL).
      5. Outputting calibrated predictive uncertainty/confidence (Uncertainty Head).

All components are strictly modular and independently switchable for clean ablation studies.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from test_rccloss_audit import RCCLoss, get_official_weights


# ==============================================================================
# 1. GRADIENT REVERSAL LAYER (GRL) FOR SUBJECT INVARIANCE
# ==============================================================================

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        # Reverse and scale gradients flowing back to the representation
        output = grad_output.neg() * ctx.alpha
        return output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer (Ganin & Lempitsky, ICML 2015).
    Forward: acts as identity.
    Backward: multiplies gradients by -alpha.
    """
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def set_alpha(self, alpha: float):
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


# ==============================================================================
# 2. MODALITY-SPECIFIC ENCODERS (HR & ACCELEROMETER)
# ==============================================================================

class TemporalConvBlock(nn.Module):
    """Lightweight 1D residual convolution block with GroupNorm and GELU."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.gn = nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels)
        self.act = nn.GELU()
        
        if in_channels != out_channels or stride != 1:
            self.res = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.res = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.conv(x))) + self.res(x)


class HREncoder(nn.Module):
    """
    Encodes 30-sample Instantaneous Heart Rate (IHR) epochs into d_model representations.
    Input:  (B, T, 30) or (B*T, 1, 30)
    Output: (B, T, d_model)
    """
    def __init__(self, in_features: int = 1, d_model: int = 64, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool1d(2), # 30 -> 15
            TemporalConvBlock(32, 64, kernel_size=5),
            nn.MaxPool1d(2), # 15 -> 7
            TemporalConvBlock(64, d_model, kernel_size=3),
            nn.AdaptiveAvgPool1d(1), # 7 -> 1
            nn.Dropout(dropout)
        )
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L = x.shape
        x_flat = x.view(B * T, 1, L) # (B*T, 1, 30)
        feat = self.net(x_flat).squeeze(-1) # (B*T, d_model)
        feat = self.norm(self.proj(feat))
        return feat.view(B, T, self.d_model)


class ACCEncoder(nn.Module):
    """
    Encodes 30-sample Accelerometer magnitude epochs into d_model representations.
    Input:  (B, T, 30) or (B*T, 1, 30)
    Output: (B, T, d_model)
    """
    def __init__(self, in_features: int = 1, d_model: int = 64, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Conv1d(in_features, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool1d(2), # 30 -> 15
            TemporalConvBlock(32, 64, kernel_size=5),
            nn.MaxPool1d(2), # 15 -> 7
            TemporalConvBlock(64, d_model, kernel_size=3),
            nn.AdaptiveAvgPool1d(1), # 7 -> 1
            nn.Dropout(dropout)
        )
        self.proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L = x.shape
        x_flat = x.view(B * T, 1, L)
        feat = self.net(x_flat).squeeze(-1)
        feat = self.norm(self.proj(feat))
        return feat.view(B, T, self.d_model)


# ==============================================================================
# 3. ADAPTIVE MODALITY RELIABILITY FUSION GATE
# ==============================================================================

class AdaptiveReliabilityFusion(nn.Module):
    """
    Learns dynamic modality reliability weight alpha_t in [0, 1]:
      alpha_t = sigmoid(W2 * GELU(W1 [h_HR ; h_ACC] + b1) + b2)
      z_t = alpha_t * h_HR + (1 - alpha_t) * h_ACC
    Interpretation:
      alpha_t -> 1: Heart rate signal is considered reliable/informative.
      alpha_t -> 0: Acceleration signal is considered reliable/informative.
    Supports ablation: if enabled=False, uses uniform static fusion (alpha=0.5).
    """
    def __init__(self, d_model: int = 64, enabled: bool = True):
        super().__init__()
        self.d_model = d_model
        self.enabled = enabled
        
        if self.enabled:
            self.gate_net = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.LayerNorm(d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
        else:
            self.gate_net = None

    def forward(self, h_hr: torch.Tensor, h_acc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h_hr, h_acc: (B, T, d_model)
        if self.enabled:
            cat_feat = torch.cat([h_hr, h_acc], dim=-1) # (B, T, 2 * d_model)
            alpha = self.gate_net(cat_feat) # (B, T, 1)
        else:
            # Static equal weighting ablation
            alpha = torch.full((h_hr.size(0), h_hr.size(1), 1), 0.5, device=h_hr.device)
            
        z = alpha * h_hr + (1.0 - alpha) * h_acc
        return z, alpha


# ==============================================================================
# 4. TRANSITION-AWARE MODELING MODULE
# ==============================================================================

class TransitionAwareModule(nn.Module):
    """
    Explicitly models physiological sleep stage transition dynamics.
    Learns a K x K transition matrix M where M_{j, k} represents the log-compatibility
    of transitioning from stage j at t-1 to stage k at t.
    
    Adjusts emission logits:
      u_t = s_t + gamma * log( sum_j p_{t-1}(j) * softmax(M)_{j, k} + eps )
    Computes transition consistency NLL:
      L_trans = - 1/(T-1) sum_{t=2}^T log( softmax(M)_{y_{t-1}, y_t} )
    Supports ablation: if enabled=False, returns unmodified emission logits and L_trans = 0.
    """
    def __init__(self, n_classes: int = 4, enabled: bool = True, eps: float = 1e-6):
        super().__init__()
        self.n_classes = n_classes
        self.enabled = enabled
        self.eps = eps
        
        # Learnable transition matrix (K, K) initialized close to uniform with positive diagonal
        init_matrix = torch.eye(n_classes) * 1.5 + torch.randn(n_classes, n_classes) * 0.1
        self.transition_matrix = nn.Parameter(init_matrix)
        self.gamma = nn.Parameter(torch.tensor(0.1)) # learnable transition influence scale

    def forward(
        self,
        emission_logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        emission_logits: (B, T, K)
        targets: Optional (B, T)
        mask: Optional (B, T)
        Returns:
          adjusted_logits: (B, T, K)
          loss_trans: scalar Tensor
        """
        if not self.enabled:
            return emission_logits, torch.tensor(0.0, device=emission_logits.device)
            
        B, T, K = emission_logits.shape
        trans_prob = F.softmax(self.transition_matrix, dim=-1) # (K, K), row j sums to 1
        
        # Sequential transition-aware logit refinement
        emiss_probs = F.softmax(emission_logits, dim=-1) # (B, T, K)
        adjusted_logits = emission_logits.clone()
        
        # Transition prior for t >= 1: q_t = p_{t-1} @ trans_prob
        # Vectorized across batch and sequence:
        prev_probs = emiss_probs[:, :-1, :] # (B, T-1, K)
        transition_prior = torch.matmul(prev_probs, trans_prob) # (B, T-1, K)
        log_prior = torch.log(transition_prior + self.eps) # (B, T-1, K)
        
        # Refine logits from t=1..T-1 with scaled transition prior
        adjusted_logits[:, 1:, :] = emission_logits[:, 1:, :] + self.gamma * log_prior
        
        # Transition consistency loss during training
        loss_trans = torch.tensor(0.0, device=emission_logits.device)
        if targets is not None:
            # Pairwise transitions (y_{t-1}, y_t)
            y_prev = targets[:, :-1] # (B, T-1)
            y_curr = targets[:, 1:]  # (B, T-1)
            
            valid = (y_prev >= 0) & (y_prev < K) & (y_curr >= 0) & (y_curr < K)
            if mask is not None:
                valid = valid & mask[:, :-1] & mask[:, 1:]
                
            if valid.any():
                valid_prev = y_prev[valid]
                valid_curr = y_curr[valid]
                # Log probability of observed transitions under learned matrix
                log_trans_prob = F.log_softmax(self.transition_matrix, dim=-1) # (K, K)
                selected_log_probs = log_trans_prob[valid_prev, valid_curr]
                loss_trans = - selected_log_probs.mean()
                
        return adjusted_logits, loss_trans


# ==============================================================================
# 5. SUBJECT DISCRIMINATOR WITH GRADIENT REVERSAL (GRL)
# ==============================================================================

class SubjectDiscriminator(nn.Module):
    """
    Adversarial subject discriminator for subject-invariant representation learning.
    Trained on TRAINING subjects only (never validation or test subjects).
    Encourages temporal representations to be invariant to subject identity.
    Supports ablation: if enabled=False, loss_adv = 0.
    """
    def __init__(
        self,
        in_dim: int = 96,
        n_subjects: int = 31,
        alpha_grl: float = 0.1,
        enabled: bool = True
    ):
        super().__init__()
        self.enabled = enabled
        self.n_subjects = n_subjects
        
        if self.enabled:
            self.grl = GradientReversalLayer(alpha=alpha_grl)
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, n_subjects)
            )
        else:
            self.grl = None
            self.net = None

    def forward(
        self,
        features: torch.Tensor,
        subject_idx: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        features: (B, T, D)
        subject_idx: (B,) integer IDs in {0, ..., n_subjects-1} or -100 for non-training
        Returns:
          subj_logits: (B, T, n_subjects) or None
          loss_adv: scalar Tensor
        """
        if not self.enabled or subject_idx is None:
            return None, torch.tensor(0.0, device=features.device)
            
        B, T, D = features.shape
        rev_features = self.grl(features) # (B, T, D)
        subj_logits = self.net(rev_features) # (B, T, n_subjects)
        
        # Valid training subjects filter (0 <= s < n_subjects)
        valid_b = (subject_idx >= 0) & (subject_idx < self.n_subjects)
        if not valid_b.any():
            return subj_logits, torch.tensor(0.0, device=features.device)
            
        # Target expanded across sequence: (B, T)
        targets_expanded = subject_idx.unsqueeze(1).repeat(1, T)
        
        flat_logits = subj_logits.view(-1, self.n_subjects)
        flat_targets = targets_expanded.view(-1)
        
        valid_flat = valid_b.unsqueeze(1).repeat(1, T).view(-1)
        if mask is not None:
            valid_flat = valid_flat & mask.view(-1)
            
        if not valid_flat.any():
            return subj_logits, torch.tensor(0.0, device=features.device)
            
        loss_adv = F.cross_entropy(flat_logits[valid_flat], flat_targets[valid_flat])
        return subj_logits, loss_adv


# ==============================================================================
# 6. UNCERTAINTY / CONFIDENCE HEAD
# ==============================================================================

class UncertaintyHead(nn.Module):
    """
    Estimates predictive uncertainty and confidence.
    Outputs:
      - predictive entropy H in [0, 1] normalized by log(K)
      - learned confidence score c in [0, 1] predicting model correctness
    Supports ablation: if enabled=False, computes normalized entropy only, loss_conf = 0.
    """
    def __init__(self, in_dim: int = 96, enabled: bool = True):
        super().__init__()
        self.enabled = enabled
        if self.enabled:
            self.head = nn.Sequential(
                nn.Linear(in_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )
        else:
            self.head = None

    def forward(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        features: (B, T, D)
        logits: (B, T, K)
        targets: Optional (B, T)
        mask: Optional (B, T)
        Returns:
          confidence: (B, T, 1) in [0, 1]
          norm_entropy: (B, T, 1) in [0, 1]
          loss_conf: scalar Tensor
        """
        B, T, K = logits.shape
        probs = F.softmax(logits, dim=-1)
        
        # Normalized Predictive Entropy: H / log(K)
        eps = 1e-7
        entropy = - torch.sum(probs * torch.log(probs + eps), dim=-1, keepdim=True) # (B, T, 1)
        norm_entropy = torch.clamp(entropy / torch.log(torch.tensor(float(K), device=logits.device)), 0.0, 1.0)
        
        if not self.enabled:
            # When disabled, confidence is directly 1.0 - norm_entropy
            confidence = 1.0 - norm_entropy
            return confidence, norm_entropy, torch.tensor(0.0, device=logits.device)
            
        confidence = self.head(features) # (B, T, 1)
        
        loss_conf = torch.tensor(0.0, device=logits.device)
        if targets is not None:
            preds = torch.argmax(logits, dim=-1) # (B, T)
            is_correct = (preds == targets).float().unsqueeze(-1) # (B, T, 1)
            
            valid = (targets >= 0) & (targets < K)
            if mask is not None:
                valid = valid & mask
                
            if valid.any():
                valid_mask = valid.unsqueeze(-1)
                flat_conf = confidence[valid_mask]
                flat_target = is_correct[valid_mask]
                loss_conf = F.binary_cross_entropy(flat_conf, flat_target)
                
        return confidence, norm_entropy, loss_conf


# ==============================================================================
# 7. BOUNDARY LOSS HELPER
# ==============================================================================

def compute_boundary_loss(
    per_epoch_loss: torch.Tensor,
    targets: torch.Tensor,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes targeted loss on difficult sleep stage boundary epochs:
      b_t = 1 if y_t != y_{t-1}, 0 otherwise.
    per_epoch_loss: (B, T) unreduced classification loss
    targets: (B, T)
    mask: Optional (B, T)
    Returns:
      loss_boundary: scalar Tensor
      boundary_mask: (B, T) boolean Tensor indicating boundary positions
    """
    B, T = targets.shape
    # Boundary condition: y_t != y_{t-1}
    y_prev = targets[:, :-1]
    y_curr = targets[:, 1:]
    
    b_inner = (y_prev != y_curr) & (y_prev >= 0) & (y_curr >= 0)
    boundary_indicator = torch.zeros(B, T, dtype=torch.bool, device=targets.device)
    boundary_indicator[:, 1:] = b_inner
    
    valid_boundary = boundary_indicator
    if mask is not None:
        valid_boundary = valid_boundary & mask
        
    if not valid_boundary.any():
        return torch.tensor(0.0, device=per_epoch_loss.device), boundary_indicator
        
    loss_boundary = per_epoch_loss[valid_boundary].mean()
    return loss_boundary, boundary_indicator


# ==============================================================================
# 8. COMPLETE MODULAR ARCHITECTURE: SleepAdaptiveNet
# ==============================================================================

class SleepAdaptiveNet(nn.Module):
    """
    Complete Proposed Sleep Staging Architecture.
    
    Components:
      1. Separate HR & ACC Encoders: h_HR, h_ACC
      2. Adaptive Modality Reliability Fusion Gate: alpha_t
      3. Bidirectional Temporal GRU Context Modeling
      4. Transition-Aware Modeling: P(y_t | y_{t-1})
      5. Boundary-Aware Learning: L_boundary
      6. Subject-Invariant Representation via GRL: L_adv
      7. Uncertainty / Confidence Head: confidence, entropy
      
    Configuration Toggles:
      - adaptive_fusion: bool
      - transition_aware: bool
      - boundary_aware: bool
      - subject_invariant: bool
      - uncertainty_head: bool
    """
    def __init__(
        self,
        d_model: int = 64,
        d_temp: int = 96,
        n_classes: int = 4,
        n_meta: int = 5,
        n_train_subjects: int = 31,
        dropout: float = 0.2,
        # Modular ablation toggles
        adaptive_fusion: bool = True,
        transition_aware: bool = True,
        boundary_aware: bool = True,
        subject_invariant: bool = True,
        uncertainty_head: bool = True,
        # Loss hyperparameter weights
        lambda_transition: float = 0.1,
        lambda_boundary: float = 0.5,
        lambda_adv: float = 0.1,
        lambda_conf: float = 0.1,
        use_rcc_loss: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.d_temp = d_temp
        self.n_classes = n_classes
        self.n_train_subjects = n_train_subjects
        
        # Flags
        self.adaptive_fusion_flag = adaptive_fusion
        self.transition_aware_flag = transition_aware
        self.boundary_aware_flag = boundary_aware
        self.subject_invariant_flag = subject_invariant
        self.uncertainty_head_flag = uncertainty_head
        
        # Loss weights
        self.lambda_transition = lambda_transition
        self.lambda_boundary = lambda_boundary
        self.lambda_adv = lambda_adv
        self.lambda_conf = lambda_conf
        self.use_rcc_loss = use_rcc_loss
        
        # 1. Modality-specific Encoders
        self.hr_encoder = HREncoder(in_features=1, d_model=d_model, dropout=dropout)
        self.acc_encoder = ACCEncoder(in_features=1, d_model=d_model, dropout=dropout)
        
        # 2. Adaptive Reliability Fusion Gate
        self.fusion = AdaptiveReliabilityFusion(d_model=d_model, enabled=adaptive_fusion)
        
        # Meta feature projection (clock, circadian time, frequency, HR stats)
        self.meta_proj = nn.Sequential(
            nn.Linear(n_meta, 16),
            nn.GELU()
        )
        
        # 3. Temporal Sequence Modeling
        in_temp = d_model + 16
        self.temporal_gru = nn.GRU(
            input_size=in_temp,
            hidden_size=d_temp // 2, # bidirectional -> total d_temp
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.temp_norm = nn.LayerNorm(d_temp)
        
        # 4. Primary Stage Classifier Head
        self.classifier_head = nn.Sequential(
            nn.Linear(d_temp, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
        
        # 5. Transition-Aware Modeling Module
        self.transition_module = TransitionAwareModule(n_classes=n_classes, enabled=transition_aware)
        
        # 6. Subject Discriminator (Adversarial GRL)
        self.subject_discriminator = SubjectDiscriminator(
            in_dim=d_temp,
            n_subjects=n_train_subjects,
            alpha_grl=0.1,
            enabled=subject_invariant
        )
        
        # 7. Uncertainty / Confidence Head
        self.uncertainty_module = UncertaintyHead(in_dim=d_temp, enabled=uncertainty_head)
        
        # Primary stage losses
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        fn_w, fp_w = get_official_weights()
        self.rcc_loss = RCCLoss(fn_weights=fn_w, fp_weights=fp_w, cls=n_classes)

    def extract_epoch_features(
        self,
        x_ihr: torch.Tensor,
        x_accel: torch.Tensor,
        x_meta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes and fuses HR and ACC epochs.
        x_ihr:   (B, T, 30)
        x_accel: (B, T, 30)
        x_meta:  (B, T, n_meta)
        Returns:
          fused_epochs: (B, T, d_model + 16)
          alpha_gate:   (B, T, 1)
        """
        h_hr = self.hr_encoder(x_ihr)
        h_acc = self.acc_encoder(x_accel)
        z, alpha = self.fusion(h_hr, h_acc)
        
        h_meta = self.meta_proj(x_meta)
        fused = torch.cat([z, h_meta], dim=-1)
        return fused, alpha

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
        Forward pass with comprehensive return dictionary for modular training and analysis.
        """
        B, T, _ = x_ihr.shape
        
        # 1 & 2. Modality encoding and adaptive fusion
        fused_epochs, alpha_gate = self.extract_epoch_features(x_ihr, x_accel, x_meta)
        
        # 3. Temporal sequence modeling
        temp_out, _ = self.temporal_gru(fused_epochs)
        temp_features = self.temp_norm(temp_out) # (B, T, d_temp)
        
        # 4. Primary stage emission logits
        emission_logits = self.classifier_head(temp_features) # (B, T, n_classes)
        
        # 5. Transition-aware modeling
        final_logits, loss_trans = self.transition_module(emission_logits, targets=targets, mask=mask)
        
        # 6. Subject-invariant representation (Adversarial GRL)
        _, loss_adv = self.subject_discriminator(temp_features, subject_idx=subject_idx, mask=mask)
        
        # 7. Uncertainty and confidence estimation
        confidence, norm_entropy, loss_conf = self.uncertainty_module(
            temp_features, final_logits, targets=targets, mask=mask
        )
        
        # Primary stage classification loss & boundary loss
        loss_stage = torch.tensor(0.0, device=x_ihr.device)
        loss_boundary = torch.tensor(0.0, device=x_ihr.device)
        boundary_mask = None
        
        if targets is not None:
            flat_logits = final_logits.view(-1, self.n_classes)
            flat_targets = targets.view(-1)
            
            valid = (flat_targets >= 0) & (flat_targets < self.n_classes)
            if mask is not None:
                valid = valid & mask.view(-1)
                
            if valid.any():
                # Cross-entropy loss per epoch
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
                    
                # Boundary-aware loss
                if self.boundary_aware_flag:
                    loss_boundary, boundary_mask = compute_boundary_loss(per_epoch_ce_2d, targets, mask=mask)
                    
        # Total composite loss
        total_loss = (
            loss_stage
            + self.lambda_transition * loss_trans
            + self.lambda_boundary * loss_boundary
            + self.lambda_adv * loss_adv
            + self.lambda_conf * loss_conf
        )
        
        return {
            "logits": final_logits,
            "emission_logits": emission_logits,
            "probabilities": F.softmax(final_logits, dim=-1),
            "alpha_gate": alpha_gate,
            "confidence": confidence,
            "entropy": norm_entropy,
            "total_loss": total_loss,
            "loss_stage": loss_stage,
            "loss_trans": loss_trans,
            "loss_boundary": loss_boundary,
            "loss_adv": loss_adv,
            "loss_conf": loss_conf,
            "boundary_mask": boundary_mask
        }


# ==============================================================================
# 9. ABLATION CONFIGURATION FACTORY
# ==============================================================================

def build_ablation_model(
    ablation_name: str,
    n_train_subjects: int = 31,
    loss_type: str = "rccloss",
    **kwargs
) -> SleepAdaptiveNet:
    """
    Constructs models corresponding exactly to the required ablation configurations:
      - 'A0': Baseline (Static fusion, no transition, no boundary, no GRL, no uncertainty head)
      - 'A1': Baseline + Adaptive HR/ACC Fusion
      - 'A2': A1 + Transition-Aware Modeling
      - 'A3': A2 + Boundary-Aware Loss
      - 'A4': A3 + Subject-Invariant Representation (GRL)
      - 'A5': A4 + Uncertainty Head (Full Proposed Model)
    Allows overriding hyperparameters (e.g. lambda_boundary) via kwargs.
    """
    configs = {
        "A0": dict(adaptive_fusion=False, transition_aware=False, boundary_aware=False, subject_invariant=False, uncertainty_head=False),
        "A1": dict(adaptive_fusion=True,  transition_aware=False, boundary_aware=False, subject_invariant=False, uncertainty_head=False),
        "A2": dict(adaptive_fusion=True,  transition_aware=True,  boundary_aware=False, subject_invariant=False, uncertainty_head=False),
        "A3": dict(adaptive_fusion=True,  transition_aware=True,  boundary_aware=True,  subject_invariant=False, uncertainty_head=False),
        "A4": dict(adaptive_fusion=True,  transition_aware=True,  boundary_aware=True,  subject_invariant=True,  uncertainty_head=False),
        "A5": dict(adaptive_fusion=True,  transition_aware=True,  boundary_aware=True,  subject_invariant=True,  uncertainty_head=True),
    }
    
    if ablation_name not in configs:
        raise ValueError(f"Unknown ablation name '{ablation_name}'. Supported: {list(configs.keys())}")
        
    cfg = dict(configs[ablation_name])
    cfg.update(kwargs)
    return SleepAdaptiveNet(
        d_model=64,
        d_temp=96,
        n_classes=4,
        n_meta=5,
        n_train_subjects=n_train_subjects,
        use_rcc_loss=(loss_type == "rccloss"),
        **cfg
    )
