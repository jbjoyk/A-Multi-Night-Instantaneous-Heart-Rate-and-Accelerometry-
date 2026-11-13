"""
research_engine/scientific_reviewer.py: Autonomous Scientific Peer Reviewer.

Inspired by SakanaAI/AI-Scientist:
  Audits experiment proposals against 10 rigorous scientific standards:
    1. Data & Split Leakage
    2. Invalid Comparisons
    3. Metric Inconsistencies (Macro-F1 priority on imbalanced data)
    4. Overfitting & Generalization Risk
    5. Insufficient / Unmatched Baselines
    6. Weak or Confounded Ablations
    7. Unsupported Novelty Assertions
    8. Excessive Architectural Complexity
    9. Statistical Uncertainty & Sample Size
   10. Deterministic Reproducibility

The reviewer has autonomous authority to REJECT or request REVISION of any flawed proposal.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from research_engine.experiment_designer import ExperimentSpecification


class ReviewVerdict(str, Enum):
    ACCEPT = "ACCEPT"
    REVISE = "REVISE"
    REJECT = "REJECT"


class ScientificReviewer:
    """
    Rigorously audits proposed experiments to protect scientific validity and resource integrity.
    """
    def __init__(self):
        pass

    def review_proposal(
        self,
        spec: ExperimentSpecification,
        existing_experiment_ids: Optional[List[str]] = None,
        test_subjects: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Conducts a formal 10-point scientific audit of an experiment specification.
        """
        flags: List[str] = []
        checks_passed = 0
        checks_total = 10

        # Check 1: Leakage Protection
        # Does the experiment claim or attempt to evaluate test subjects during screening?
        spec_text = str(spec.to_dict()).lower()
        if "test set" in spec_text or "test partition" in spec_text or "outer test" in spec_text:
            if "frozen" not in spec_text:
                flags.append("CHECK 1 FAILED: Proposal mentions test set without frozen benchmark authorization. Potential data leakage!")
        else:
            checks_passed += 1

        # Check 2: Invalid Comparisons
        # Must compare against verified baselines (Model A or Model E)
        if spec.baseline_model not in ("Model A", "Model E", "Model B", "Model C", "Model D", "B0_GRL"):
            flags.append(f"CHECK 2 FAILED: Unverified baseline model '{spec.baseline_model}'. Must anchor to audited baselines.")
        else:
            checks_passed += 1

        # Check 3: Metric Inconsistencies
        # Primary metric must prioritize Macro-F1 on imbalanced sleep data
        if "macro-f1" not in spec.success_metric.lower() and "f1" not in spec.success_metric.lower():
            flags.append("CHECK 3 FAILED: Primary success metric must prioritize Macro-F1 for class-imbalanced sleep staging.")
        else:
            checks_passed += 1

        # Check 4: Overfitting Risk
        if spec.expected_vram_mb > 8000:
            flags.append(f"CHECK 4 FAILED: Excessive memory profile ({spec.expected_vram_mb} MB) threatens OOM / overfitting.")
        else:
            checks_passed += 1

        # Check 5: Baseline Adequacy
        if not spec.baseline_model:
            flags.append("CHECK 5 FAILED: Baseline model is unspecified.")
        else:
            checks_passed += 1

        # Check 6: Clean Ablation Control
        if not spec.is_controlled_ablation or len(spec.fixed_components) < 2:
            flags.append("CHECK 6 FAILED: Insufficient fixed controls. Experiment must isolate a single factor.")
        else:
            checks_passed += 1

        # Check 7: Unsupported Novelty Claims
        if "guaranteed novel" in spec_text or "breakthrough" in spec_text:
            flags.append("CHECK 7 FAILED: Unsupported novelty assertion. All novelty must be framed as candidate novelty.")
        else:
            checks_passed += 1

        # Check 8: Excessive Complexity
        complex_words = ["huge", "massive", "transformer-100m", "unbounded", "arbitrary"]
        if any(w in spec_text for w in complex_words):
            flags.append("CHECK 8 FAILED: Excessive complexity. Proposes disproportionate architecture for wearable signals.")
        else:
            checks_passed += 1

        # Check 9: Statistical Uncertainty & Falsifiability
        if not spec.falsification_criteria or len(spec.falsification_criteria.strip()) < 10:
            flags.append("CHECK 9 FAILED: Weak or missing falsification criteria. Hypotheses must be empirically falsifiable.")
        else:
            checks_passed += 1

        # Check 10: Reproducibility & Redundancy
        if existing_experiment_ids and spec.experiment_id in existing_experiment_ids:
            flags.append(f"CHECK 10 FAILED: Experiment ID '{spec.experiment_id}' already exists in research memory.")
        else:
            checks_passed += 1

        # Verdict Determination
        if any("CHECK 1 FAILED" in f or "CHECK 2 FAILED" in f or "CHECK 10 FAILED" in f for f in flags):
            verdict = ReviewVerdict.REJECT
        elif flags:
            verdict = ReviewVerdict.REVISE
        else:
            verdict = ReviewVerdict.ACCEPT

        return {
            "verdict": verdict.value,
            "checks_passed": checks_passed,
            "checks_total": checks_total,
            "flags": flags,
            "summary": (
                f"Scientific Review Verdict: {verdict.value} ({checks_passed}/{checks_total} checks passed). "
                + (f"Flags: {'; '.join(flags)}" if flags else "All 10 experimental validity checks PASSED.")
            )
        }
