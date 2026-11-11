"""
EXP-014 Model Architecture Definitions & Factory
File: exp014_models.py

Defines the complete 5-model matrix for EXP-014 5-Fold Subject Cross-Validation:
  - Model A: Matched-capacity conventional CNN-BiGRU (no gate, direct concat, no transition, no GRL)
  - Model B: SleepAdaptiveNet minus GRL (adaptive fusion ON, transition ON, GRL OFF)
  - Model C: SleepAdaptiveNet minus Transition (adaptive fusion ON, transition OFF, GRL ON)
  - Model D: SleepAdaptiveNet minus Dynamic Gating (fixed 0.5/0.5 fusion, transition ON, GRL ON)
  - Model E: Full SleepAdaptiveNet (adaptive fusion ON, transition ON, GRL ON, boundary OFF, uncertainty OFF)

IMPORTANT ARCHITECTURAL RULE:
In EXP-014, Model E is independently instantiated and trained from scratch for each outer fold
using that fold's inner training subjects. It NEVER loads the historical EXP-012 checkpoint.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from exp013_models import ConventionalCNNBiGRU
from sleep_adaptive_net import SleepAdaptiveNet


def build_exp014_model(
    model_id: str,
    n_train_subjects: int,
    loss_type: str = "rccloss"
) -> nn.Module:
    """
    Factory function instantiating freshly initialized models for EXP-014 cross-validation.
    
    All 5 models are trainable in EXP-014:
      - Model A: ConventionalCNNBiGRU (163,940 parameters)
      - Model B: SleepAdaptiveNet minus GRL (153,974 parameters)
      - Model C: SleepAdaptiveNet minus Transition (162,325 parameters)
      - Model D: SleepAdaptiveNet minus Dynamic Gating (153,876 parameters)
      - Model E: Full SleepAdaptiveNet (162,325 parameters)
      
    Args:
        model_id: 'A', 'B', 'C', 'D', or 'E'
        n_train_subjects: Number of inner training subjects in the specific fold
        loss_type: Loss configuration ('rccloss' or 'ce')
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
            subject_invariant=False, # GRL disabled
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
            transition_aware=False, # Transition module disabled
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
            adaptive_fusion=False, # Fixed 0.5/0.5 equal fusion in AdaptiveReliabilityFusion
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
        # Newly instantiated, untrained Model E for EXP-014 fold training
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
            subject_invariant=True,
            uncertainty_head=False,
            lambda_transition=0.10,
            lambda_boundary=0.0,
            lambda_adv=0.10,
            lambda_conf=0.0,
            use_rcc_loss=(loss_type == "rccloss")
        )
    else:
        raise ValueError(f"Unknown model_id '{model_id}'. Expected one of ['A', 'B', 'C', 'D', 'E'].")
