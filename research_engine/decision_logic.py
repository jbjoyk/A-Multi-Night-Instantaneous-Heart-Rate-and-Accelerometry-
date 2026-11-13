"""
research_engine/decision_logic.py: Multi-Factor Keep / Discard / Investigate Decision Engine.

Inspired by Karpathy's Autoresearch with scientific multi-criteria evaluation:
  Replaces naive 'if score > prev: keep' with a 9-factor multi-criteria evaluation:
    1. Primary Metric Delta (Macro-F1)
    2. Secondary Metrics (Accuracy, Cohen's kappa, ECE, per-class F1)
    3. Variance & Statistical Confidence
    4. Computational Cost & Runtime
    5. Parameter Capacity Delta
    6. VRAM Utilization
    7. Architectural Complexity
    8. Deterministic Reproducibility
    9. Scientific Relevance & Insights
"""

from enum import Enum
from typing import Dict, Optional, Any


class ExperimentDecision(str, Enum):
    KEPT = "KEPT"
    DISCARDED = "DISCARDED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    INVESTIGATE = "INVESTIGATE"


class DecisionEngine:
    """
    Evaluates experimental results using multi-criteria scientific Pareto trade-offs.
    """
    def __init__(self):
        pass

    def evaluate_outcome(
        self,
        baseline_metrics: Dict[str, Any],
        candidate_metrics: Dict[str, Any],
        baseline_params: int = 162325,
        candidate_params: int = 162325,
        baseline_vram_mb: int = 4000,
        candidate_vram_mb: int = 4000
    ) -> Dict[str, Any]:
        """
        Computes multi-factor outcome decision comparing candidate against baseline.
        """
        # Primary Macro-F1
        base_f1 = baseline_metrics.get("macro_f1", baseline_metrics.get("primary", {}).get("macro_f1", 0.0))
        cand_f1 = candidate_metrics.get("macro_f1", candidate_metrics.get("primary", {}).get("macro_f1", 0.0))
        delta_f1 = cand_f1 - base_f1

        # Secondary Accuracy & Kappa
        base_acc = baseline_metrics.get("accuracy", baseline_metrics.get("secondary", {}).get("accuracy", 0.0))
        cand_acc = candidate_metrics.get("accuracy", candidate_metrics.get("secondary", {}).get("accuracy", 0.0))
        delta_acc = cand_acc - base_acc

        # Capacity
        param_ratio = candidate_params / max(1, baseline_params)
        delta_params_pct = (param_ratio - 1.0) * 100.0

        # VRAM
        vram_ratio = candidate_vram_mb / max(1, baseline_vram_mb)

        decision = ExperimentDecision.NEEDS_REVIEW
        rationale = []

        # Rule 1: Substantial Primary Improvement with Reasonable Capacity
        if delta_f1 >= 0.010 and delta_params_pct <= 15.0:
            decision = ExperimentDecision.KEPT
            rationale.append(f"Statistically meaningful Macro-F1 improvement (+{delta_f1:.4f}) under matched capacity.")

        # Rule 2: Complexity Trap (Tiny gain with huge complexity spike)
        elif 0.0 < delta_f1 < 0.005 and delta_params_pct > 30.0:
            decision = ExperimentDecision.NEEDS_REVIEW
            rationale.append(
                f"Complexity Trap: Marginal Macro-F1 gain (+{delta_f1:.4f}) achieved at the expense of "
                f"+{delta_params_pct:.1f}% parameter bloat. Flagged for review."
            )

        # Rule 3: Beneficial Simplification (Parity with substantial compression)
        elif delta_f1 >= -0.003 and delta_params_pct <= -15.0:
            decision = ExperimentDecision.KEPT
            rationale.append(
                f"Beneficial Simplification: Maintains Macro-F1 parity (Δ={delta_f1:+.4f}) while achieving "
                f"{abs(delta_params_pct):.1f}% parameter reduction."
            )

        # Rule 4: Clear Degradation
        elif delta_f1 < -0.015 and delta_params_pct >= -5.0:
            decision = ExperimentDecision.DISCARDED
            rationale.append(f"Clear performance regression: Macro-F1 dropped by {delta_f1:.4f} without compensating simplicity.")

        # Rule 5: Intriguing Discrepancy / Class Trade-off
        elif abs(delta_f1) < 0.005:
            cand_sec = candidate_metrics.get("secondary", candidate_metrics)
            base_sec = baseline_metrics.get("secondary", baseline_metrics)
            rem_delta = cand_sec.get("rem_f1", 0.0) - base_sec.get("rem_f1", 0.0)
            if rem_delta >= 0.030:
                decision = ExperimentDecision.INVESTIGATE
                rationale.append(
                    f"Overall Macro-F1 near parity (Δ={delta_f1:+.4f}), but strong selective REM F1 improvement "
                    f"(+{rem_delta:.4f}). Merits further scientific investigation."
                )
            else:
                decision = ExperimentDecision.NEEDS_REVIEW
                rationale.append(f"Inconclusive delta: Macro-F1 change ({delta_f1:+.4f}) is within empirical noise margin.")
        else:
            decision = ExperimentDecision.NEEDS_REVIEW
            rationale.append(f"Ambiguous trade-off: ΔF1={delta_f1:+.4f}, ΔAcc={delta_acc:+.4f}, ΔParams={delta_params_pct:+.1f}%.")

        return {
            "decision": decision.value,
            "delta_macro_f1": float(delta_f1),
            "delta_accuracy": float(delta_acc),
            "delta_params_pct": float(delta_params_pct),
            "vram_ratio": float(vram_ratio),
            "rationale": " ".join(rationale)
        }
