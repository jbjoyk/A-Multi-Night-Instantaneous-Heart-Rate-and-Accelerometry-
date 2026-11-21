"""
final_model_pipeline.py: Unified Frozen Final Model & Tiered Adaptation Pipeline (Stage 2).

Encapsulates SleepStagingFinalModel_v1.0:
  - Frozen population backbone: Model A (ConventionalCNNBiGRU, exactly 163,940 params).
  - Full-night context inference: SlidingWindowPredictor (L=1200, circadian neutralized).
  - Tier 0: Zero-Shot population inference.
  - Tier 1: Optional quality-gated unsupervised InfoMax auxiliary (TSK-07 / DEC-036),
            supporting truly unlabeled wearable records (no 'y' key required).
  - Tier 2: Fold-locked supervised within-person calibration (TSK-06 / DEC-035) with
            non-mutating stage-coverage safety diagnostics.
  - Hardware SRAM weight footprint calculator (FP32 / FP16 / INT8).

Governing decisions: DEC-029, DEC-031, DEC-033, DEC-034, DEC-035, DEC-036.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP018_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-018")
EXP019_DIR = os.path.join(REPO_ROOT, "experiments", "EXP-019")
for _p in (REPO_ROOT, EXP018_DIR, EXP019_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from exp013_models import ConventionalCNNBiGRU
from exp018_inference import SlidingWindowPredictor
from exp019_adaptation_core import (
    adapt_supervised_ft,
    adapt_unsupervised_infomax,
    compute_wearable_quality_mask,
    reset_model_weights,
)

DEFAULT_SPEC_PATH = os.path.join(os.path.dirname(__file__), "final_model_spec.json")
EXPECTED_PARAM_COUNT = 163940
STAGE_NAMES = ("Wake", "Light", "Deep", "REM")

SCOPE_REGEX_MAP: Dict[str, str] = {
    "head_gru": r"(classifier_head|temporal_gru.*weight_.*l1|temporal_gru.*bias_.*l1|temp_norm)",
    "all": r".*",
}


def sha256_file(path: str) -> str:
    """Computes streaming SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class FinalModelConfig:
    """Immutable configuration loaded from final_model_spec.json."""

    model_name: str = "SleepStagingFinalModel_v1.0"
    d_model: int = 64
    d_temp: int = 96
    n_classes: int = 4
    n_meta: int = 5
    dropout: float = 0.2
    use_rcc_loss: bool = True
    context_len: int = 1200
    neutralize_circadian: bool = True
    expected_params: int = EXPECTED_PARAM_COUNT
    checkpoints_dir: str = os.path.join(REPO_ROOT, "experiments", "EXP-018", "checkpoints")
    stage_d_configs_path: str = os.path.join(
        REPO_ROOT, "experiments", "EXP-019", "exp019_stage_d_selected_configs.json"
    )
    ua_lr: float = 1e-5
    ua_steps: int = 20
    ua_lambda_div: float = 1.0
    ua_prior_entropy: float = float(np.log(4.0))
    gate_valid_epoch_ratio_min: float = 0.95
    gate_min_epochs_per_night: int = 600
    gate_max_non_phys_leakage: int = 0
    min_stage_fraction_warn: float = 0.02
    folds: Tuple[int, ...] = (1, 2, 3, 4, 5)
    seeds: Tuple[int, ...] = (42, 123, 2024)
    raw_spec: Dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_json(cls, spec_path: str = DEFAULT_SPEC_PATH) -> "FinalModelConfig":
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)

        arch = spec["architecture"]["hyperparameters"]
        data_cfg = spec["data_and_context_protocol"]
        pop_cfg = spec["population_backbone_training"]
        tier1 = spec["tiered_personalization_policy"]["tier_1_unsupervised_auxiliary"]
        gate_cfg = tier1["wearable_quality_gate"]
        tier2 = spec["tiered_personalization_policy"]["tier_2_supervised_calibration"]

        ckpt_dir = pop_cfg["checkpoints_dir"]
        if not os.path.isabs(ckpt_dir):
            ckpt_dir = os.path.join(REPO_ROOT, ckpt_dir)

        stage_d_path = tier2["stage_d_fold_configs_json"]
        if not os.path.isabs(stage_d_path):
            stage_d_path = os.path.join(REPO_ROOT, stage_d_path)

        return cls(
            model_name=spec["model_name"],
            d_model=int(arch["d_model"]),
            d_temp=int(arch["d_temp"]),
            n_classes=int(arch["n_classes"]),
            n_meta=int(arch["n_meta"]),
            dropout=float(arch["dropout"]),
            use_rcc_loss=bool(arch["use_rcc_loss"]),
            context_len=int(data_cfg["context_len_epochs"]),
            neutralize_circadian=bool(data_cfg["neutralize_circadian"]),
            expected_params=int(spec["architecture"]["exact_trainable_parameters"]),
            checkpoints_dir=ckpt_dir,
            stage_d_configs_path=stage_d_path,
            ua_lr=float(tier1["lr"]),
            ua_steps=int(tier1["steps"]),
            ua_lambda_div=float(tier1["lambda_div"]),
            ua_prior_entropy=float(tier1["prior_entropy_nats"]),
            gate_valid_epoch_ratio_min=float(gate_cfg["cohort_valid_epoch_ratio_min"]),
            gate_min_epochs_per_night=int(gate_cfg["night_min_valid_epochs"]),
            gate_max_non_phys_leakage=int(gate_cfg.get("non_physiological_leakage_max", 0)),
            min_stage_fraction_warn=float(
                tier2["stage_coverage_safety_diagnostic"]["min_stage_epoch_fraction_warn"]
            ),
            folds=tuple(int(x) for x in pop_cfg["folds"]),
            seeds=tuple(int(x) for x in pop_cfg["seeds"]),
            raw_spec=spec,
        )


def audit_calibration_stage_coverage(
    cal_records: List[Dict[str, np.ndarray]],
    min_fraction_warn: float = 0.02,
) -> Dict[str, Any]:
    """
    Non-mutating diagnostic inspecting sleep-stage class coverage across calibration nights.
    Identifies severely underrepresented or absent stages (e.g., Bidslab34 Deep = 0.68%)
    that represent elevated risk of negative transfer under supervised fine-tuning.
    """
    if not cal_records:
        return {
            "n_calibration_nights": 0,
            "n_valid_epochs": 0,
            "stage_counts": {name: 0 for name in STAGE_NAMES},
            "stage_fractions": {name: 0.0 for name in STAGE_NAMES},
            "missing_stages": list(STAGE_NAMES),
            "scarce_stages": [],
            "high_negative_transfer_risk": True,
        }

    all_y = np.concatenate([rec["y"] for rec in cal_records])
    valid_y = all_y[(all_y >= 0) & (all_y < 4)]
    n_valid = int(len(valid_y))

    counts: Dict[str, int] = {}
    fractions: Dict[str, float] = {}
    missing_stages: List[str] = []
    scarce_stages: List[str] = []

    for idx, name in enumerate(STAGE_NAMES):
        c = int(np.sum(valid_y == idx))
        frac = float(c / n_valid) if n_valid > 0 else 0.0
        counts[name] = c
        fractions[name] = round(frac, 6)
        if c == 0:
            missing_stages.append(name)
        elif frac < min_fraction_warn:
            scarce_stages.append(name)

    return {
        "n_calibration_nights": len(cal_records),
        "n_valid_epochs": n_valid,
        "stage_counts": counts,
        "stage_fractions": fractions,
        "missing_stages": missing_stages,
        "scarce_stages": scarce_stages,
        "high_negative_transfer_risk": bool(missing_stages or scarce_stages),
    }


def verify_wearable_quality_records(
    cal_records: List[Dict[str, np.ndarray]],
    min_valid_ratio: float = 0.95,
    min_epochs_per_night: int = 600,
    max_non_phys_leakage: int = 0,
) -> Dict[str, Any]:
    """
    Evaluates the TSK-07 scale-invariant wearable quality gate over calibration records.
    Operates strictly on wearable arrays ('ihr', 'accel') without requiring ground-truth labels ('y'):
      - valid_epoch_ratio > min_valid_ratio (strict inequality per PREREG_EXP-019_S3_AMENDMENT.md §3.2)
      - min_epochs_per_night >= min_epochs_per_night
      - non_physiological_leakage_count <= max_non_phys_leakage (NaN/Inf leakage check)
    """
    if not cal_records:
        return {
            "total_epochs": 0,
            "valid_epochs": 0,
            "valid_epoch_ratio": 0.0,
            "min_epochs_per_night": 0,
            "non_physiological_leakage_count": 0,
            "passed": False,
        }

    total_epochs = 0
    valid_epochs = 0
    non_phys_leakage = 0
    night_valid_counts: List[int] = []

    for rec in cal_records:
        ihr = rec["ihr"]
        acc = rec["accel"]
        n_len = int(len(ihr))
        raw_bpm = rec.get("raw_bpm", None)
        mask = compute_wearable_quality_mask(ihr[:n_len], acc[:n_len], raw_bpm_array=raw_bpm)
        n_val = int(np.sum(mask))

        # Verify zero NaN/Inf epochs leak through the wearable quality mask
        if n_val > 0:
            passed_ihr = ihr[:n_len][mask]
            passed_acc = acc[:n_len][mask]
            bad_ihr = ~np.all(np.isfinite(passed_ihr), axis=-1) if passed_ihr.ndim == 2 else ~np.isfinite(passed_ihr)
            bad_acc = ~np.all(np.isfinite(passed_acc), axis=-1) if passed_acc.ndim == 2 else ~np.isfinite(passed_acc)
            non_phys_leakage += int(np.sum(bad_ihr | bad_acc))

        total_epochs += n_len
        valid_epochs += n_val
        night_valid_counts.append(n_val)

    ratio = float(valid_epochs / total_epochs) if total_epochs > 0 else 0.0
    min_night = int(min(night_valid_counts)) if night_valid_counts else 0
    passed = (
        (ratio > min_valid_ratio)
        and (min_night >= min_epochs_per_night)
        and (non_phys_leakage <= max_non_phys_leakage)
    )

    return {
        "total_epochs": total_epochs,
        "valid_epochs": valid_epochs,
        "valid_epoch_ratio": ratio,
        "min_epochs_per_night": min_night,
        "non_physiological_leakage_count": non_phys_leakage,
        "passed": passed,
    }


def _ensure_label_placeholder_for_unlabeled(
    cal_records: List[Dict[str, np.ndarray]],
) -> List[Dict[str, np.ndarray]]:
    """
    Returns shallow copies with dummy '-100' targets when 'y' is absent, enabling true
    label-free Tier 1 adaptation without mutating caller dicts.
    """
    sanitized: List[Dict[str, np.ndarray]] = []
    for rec in cal_records:
        if "y" in rec:
            sanitized.append(rec)
        else:
            rec_copy = dict(rec)
            rec_copy["y"] = np.full(len(rec["ihr"]), -100, dtype=np.int64)
            sanitized.append(rec_copy)
    return sanitized


class FinalSleepStagingSystem:
    """
    Production interface for SleepStagingFinalModel_v1.0.
    Preserves clean separation between frozen population weights and per-subject adapted clones.
    """

    def __init__(
        self,
        config: Optional[FinalModelConfig] = None,
        device: Optional[str] = None,
    ):
        self.config = config or FinalModelConfig.from_json()
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.backbone: ConventionalCNNBiGRU = self.build_backbone()
        self._frozen_state: Optional[Dict[str, torch.Tensor]] = None
        self._loaded_checkpoint_path: Optional[str] = None
        self._stage_d_cache: Optional[Dict[str, Any]] = None

    def build_backbone(self) -> ConventionalCNNBiGRU:
        """Instantiates Model A and asserts the exact frozen parameter count (163,940)."""
        model = ConventionalCNNBiGRU(
            d_model=self.config.d_model,
            d_temp=self.config.d_temp,
            n_classes=self.config.n_classes,
            n_meta=self.config.n_meta,
            dropout=self.config.dropout,
            use_rcc_loss=self.config.use_rcc_loss,
        ).to(self.device)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params == self.config.expected_params, (
            f"Final model parameter count invariant violated: "
            f"got {n_params:,}, expected {self.config.expected_params:,}"
        )
        return model

    def checkpoint_path_for(self, fold: int, seed: int) -> str:
        return os.path.join(
            self.config.checkpoints_dir, f"modelA_L1200_fold{fold}_seed{seed}.pt"
        )

    def load_frozen_checkpoint(
        self,
        fold: int,
        seed: int,
        expected_sha256: Optional[str] = None,
    ) -> str:
        """Loads a verified frozen population checkpoint and caches its clean state dict."""
        ckpt_path = self.checkpoint_path_for(fold, seed)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Frozen checkpoint not found: {ckpt_path}")

        actual_sha = sha256_file(ckpt_path)
        if expected_sha256 is not None and actual_sha != expected_sha256:
            raise ValueError(
                f"Checkpoint SHA-256 mismatch for {ckpt_path}: "
                f"expected {expected_sha256}, got {actual_sha}"
            )

        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        self.backbone.load_state_dict(state)
        self.backbone.eval()
        self._frozen_state = copy.deepcopy(self.backbone.state_dict())
        self._loaded_checkpoint_path = ckpt_path
        return actual_sha

    def _clone_frozen_backbone(self, allow_unloaded: bool = False) -> ConventionalCNNBiGRU:
        """Returns an isolated copy of the frozen population backbone for safe adaptation."""
        if self._frozen_state is None and not allow_unloaded:
            raise RuntimeError(
                "No frozen population checkpoint loaded. Call load_frozen_checkpoint(fold, seed) "
                "before running adaptation (or pass allow_unloaded=True for unit testing)."
            )
        clone = self.build_backbone()
        ref_state = self._frozen_state if self._frozen_state is not None else self.backbone.state_dict()
        reset_model_weights(clone, ref_state)
        clone.eval()
        return clone

    def make_predictor(self, model: Optional[nn.Module] = None) -> SlidingWindowPredictor:
        """Creates the standardized L=1200 full-night SlidingWindowPredictor."""
        target_model = model if model is not None else self.backbone
        return SlidingWindowPredictor(
            model=target_model,
            context_len=self.config.context_len,
            neutralize_circadian=self.config.neutralize_circadian,
            device=self.device,
        )

    def predict_night(
        self,
        record: Dict[str, np.ndarray],
        model: Optional[nn.Module] = None,
    ) -> Dict[str, Any]:
        """Tier 0 inference (or inference using an adapted clone) on a single full night."""
        predictor = self.make_predictor(model=model)
        return predictor.predict_night(record)

    def _load_stage_d_configs(self) -> Dict[str, Any]:
        if self._stage_d_cache is None:
            with open(self.config.stage_d_configs_path, "r", encoding="utf-8") as f:
                self._stage_d_cache = json.load(f)
        return self._stage_d_cache

    def adapt_unsupervised(
        self,
        cal_records: List[Dict[str, np.ndarray]],
        enforce_quality_gate: bool = True,
        allow_unloaded: bool = False,
    ) -> Tuple[ConventionalCNNBiGRU, Dict[str, Any]]:
        """
        Tier 1 label-free unsupervised stabilization (TSK-07 / DEC-036).
        Accepts truly unlabeled wearable records (no 'y' key required).
        Never mutates self.backbone; returns an adapted clone and audit metadata.
        """
        quality_audit = verify_wearable_quality_records(
            cal_records,
            min_valid_ratio=self.config.gate_valid_epoch_ratio_min,
            min_epochs_per_night=self.config.gate_min_epochs_per_night,
            max_non_phys_leakage=self.config.gate_max_non_phys_leakage,
        )
        if enforce_quality_gate and not quality_audit["passed"]:
            raise RuntimeError(
                f"Wearable pre-flight quality gate failed: {quality_audit}"
            )

        adapted_model = self._clone_frozen_backbone(allow_unloaded=allow_unloaded)
        sanitized_records = _ensure_label_placeholder_for_unlabeled(cal_records)
        adapted_model, infomax_info = adapt_unsupervised_infomax(
            model=adapted_model,
            cal_records=sanitized_records,
            lr=self.config.ua_lr,
            steps=self.config.ua_steps,
            lambda_div=self.config.ua_lambda_div,
            prior_entropy=self.config.ua_prior_entropy,
            device=str(self.device),
        )
        return adapted_model, {
            "tier": "tier_1_unsupervised_auxiliary",
            "quality_audit": quality_audit,
            "infomax_info": infomax_info,
        }

    def adapt_supervised(
        self,
        cal_records: List[Dict[str, np.ndarray]],
        fold: int,
        allow_unloaded: bool = False,
    ) -> Tuple[ConventionalCNNBiGRU, Dict[str, Any]]:
        """
        Tier 2 supervised within-person calibration (TSK-06 / DEC-035).
        Uses Stage D fold-locked hyperparameters and attaches non-mutating stage-coverage diagnostics.
        Never mutates self.backbone; returns an adapted clone and diagnostic metadata.
        """
        if not cal_records:
            raise ValueError("cal_records must contain at least 1 labeled calibration night.")

        stage_d = self._load_stage_d_configs()
        fold_cfg = stage_d["folds"][str(fold)]["FT-2"]
        lr = float(fold_cfg["lr"])
        epochs = int(fold_cfg["epochs"])
        scope = str(fold_cfg["scope"])
        scope_regex = SCOPE_REGEX_MAP.get(scope, scope)

        coverage_diag = audit_calibration_stage_coverage(
            cal_records, min_fraction_warn=self.config.min_stage_fraction_warn
        )

        adapted_model = self._clone_frozen_backbone(allow_unloaded=allow_unloaded)
        adapted_model = adapt_supervised_ft(
            model=adapted_model,
            cal_records=cal_records,
            lr=lr,
            epochs=epochs,
            scope_regex=scope_regex,
            device=str(self.device),
        )
        return adapted_model, {
            "tier": "tier_2_supervised_calibration",
            "fold": fold,
            "k_nights": len(cal_records),
            "hyperparameters": {"lr": lr, "epochs": epochs, "scope": scope},
            "stage_coverage_diagnostic": coverage_diag,
        }

    def export_int8_memory_profile(self) -> Dict[str, Any]:
        """Computes exact parameter weight memory footprint (FP32, FP16, INT8; weights only)."""
        n_params = sum(p.numel() for p in self.backbone.parameters())
        slamss_paper_params = 666060
        slamss_repo_params = 5465768
        return {
            "model_name": self.config.model_name,
            "trainable_parameters": n_params,
            "fp32_bytes": n_params * 4,
            "fp32_kib": round((n_params * 4) / 1024.0, 2),
            "fp16_bytes": n_params * 2,
            "fp16_kib": round((n_params * 2) / 1024.0, 2),
            "int8_bytes": n_params * 1,
            "int8_kib": round(n_params / 1024.0, 2),
            "compression_vs_slamss_paper_text": round(slamss_paper_params / n_params, 2),
            "compression_vs_slamss_official_repo": round(slamss_repo_params / n_params, 2),
            "weights_fit_nrf5340_512kb_sram_int8": (n_params / 1024.0) < 512.0,
            "sram_note": "Parameter weights only; BiGRU activation buffers and RTOS stack require separate SRAM budgeting.",
        }
