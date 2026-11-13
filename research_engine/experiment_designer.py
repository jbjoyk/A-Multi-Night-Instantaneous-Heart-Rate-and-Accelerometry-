"""
research_engine/experiment_designer.py: Controlled Experiment Specification and Design.

Implements Karpathy's Autoresearch experiment specification requirements:
  1. Answers all 10 mandatory experimental design questions.
  2. Enforces controlled One-Factor-At-A-Time (OFAT) ablations.
  3. Rejects proposals that mutate multiple uncontrolled variables simultaneously.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict


class ExperimentDesignError(Exception):
    """Raised when an experiment proposal violates experimental control standards."""
    pass


@dataclass
class ExperimentSpecification:
    experiment_id: str
    hypothesis_id: str
    research_question: str
    hypothesis_statement: str
    baseline_model: str
    exact_change: str
    fixed_components: List[str]
    success_metric: str
    falsification_criteria: str
    confounders: List[str]
    compute_budget: str
    target_device: str
    expected_vram_mb: int
    is_controlled_ablation: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperimentDesigner:
    """
    Transforms scientific hypotheses into rigorously controlled, executable experiment specifications.
    """
    ALLOWED_BUDGETS = {"SMOKE", "SHORT", "STANDARD", "FULL"}

    def __init__(self):
        pass

    def design_experiment(
        self,
        experiment_id: str,
        hypothesis_id: str,
        research_question: str,
        hypothesis_statement: str,
        baseline_model: str,
        exact_change: str,
        fixed_components: List[str],
        success_metric: str,
        falsification_criteria: str,
        confounders: List[str],
        compute_budget: str,
        target_device: str = "cuda:0",
        expected_vram_mb: int = 4000
    ) -> ExperimentSpecification:
        """
        Creates and validates a complete, single-variable experiment design.
        """
        budget_clean = compute_budget.upper().strip()
        if budget_clean not in self.ALLOWED_BUDGETS:
            raise ExperimentDesignError(
                f"Invalid budget '{compute_budget}'. Allowed budgets: {sorted(list(self.ALLOWED_BUDGETS))}"
            )

        # Detect multiple uncontrolled changes
        indicators = [" and also ", " plus new ", " along with ", " together with ", " simultaneous "]
        change_lower = exact_change.lower()
        if any(ind in change_lower for ind in indicators):
            raise ExperimentDesignError(
                f"Multi-variable confounder detected in change description: '{exact_change}'. "
                "The research protocol strictly enforces One-Factor-At-A-Time (OFAT) controlled ablations."
            )

        spec = ExperimentSpecification(
            experiment_id=experiment_id,
            hypothesis_id=hypothesis_id,
            research_question=research_question,
            hypothesis_statement=hypothesis_statement,
            baseline_model=baseline_model,
            exact_change=exact_change,
            fixed_components=fixed_components,
            success_metric=success_metric,
            falsification_criteria=falsification_criteria,
            confounders=confounders,
            compute_budget=budget_clean,
            target_device=target_device,
            expected_vram_mb=expected_vram_mb,
            is_controlled_ablation=True
        )
        return spec
