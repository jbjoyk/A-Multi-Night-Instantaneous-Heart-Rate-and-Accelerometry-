"""
research_engine/hypothesis_agent.py: Hypothesis Generation and Framing Agent.

Inspired by SakanaAI/AI-Scientist:
  1. Formulates structured scientific hypotheses grounded in prior experiments, DEC-021 audit findings, and literature.
  2. Enforces mandatory 10-point schema for every hypothesis:
       - hypothesis
       - motivation
       - expected_mechanism
       - experiment_design
       - expected_outcome
       - novelty_rationale (strictly stamped as 'candidate novelty requiring literature verification')
       - feasibility
       - risks
       - required_compute
       - evaluation_criteria
  3. Precludes unsubstantiated novelty assertions.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from research_engine.literature_agent import LiteratureAgent, EvidenceCategory


@dataclass
class ResearchHypothesis:
    hypothesis_id: str
    research_question: str
    hypothesis: str
    motivation: str
    expected_mechanism: str
    experiment_design: str
    expected_outcome: str
    novelty_rationale: str
    feasibility: str
    risks: str
    required_compute: str
    evaluation_criteria: str
    status: str = "PROPOSED"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Enforce novelty labeling rule
        if "candidate novelty requiring literature verification" not in d["novelty_rationale"].lower():
            d["novelty_rationale"] = (
                f"{d['novelty_rationale']} [candidate novelty requiring literature verification]"
            )
        return d


class HypothesisAgent:
    """
    Generates and formalizes testable scientific hypotheses for wearable sleep staging.
    """
    def __init__(self, literature_agent: Optional[LiteratureAgent] = None):
        self.literature_agent = literature_agent or LiteratureAgent()

    def create_hypothesis(
        self,
        hypothesis_id: str,
        research_question: str,
        hypothesis: str,
        motivation: str,
        expected_mechanism: str,
        experiment_design: str,
        expected_outcome: str,
        novelty_rationale: str,
        feasibility: str,
        risks: str,
        required_compute: str,
        evaluation_criteria: str
    ) -> ResearchHypothesis:
        """
        Creates and validates a structured research hypothesis.
        """
        # Ensure novelty rationale is never treated as a guaranteed fact
        sanitized_novelty = novelty_rationale.strip()
        if not sanitized_novelty.endswith("[candidate novelty requiring literature verification]"):
            sanitized_novelty = f"{sanitized_novelty} (candidate novelty requiring literature verification)"

        hyp = ResearchHypothesis(
            hypothesis_id=hypothesis_id,
            research_question=research_question,
            hypothesis=hypothesis,
            motivation=motivation,
            expected_mechanism=expected_mechanism,
            experiment_design=experiment_design,
            expected_outcome=expected_outcome,
            novelty_rationale=sanitized_novelty,
            feasibility=feasibility,
            risks=risks,
            required_compute=required_compute,
            evaluation_criteria=evaluation_criteria,
            status="PROPOSED"
        )
        return hyp

    def get_grounded_candidates(self) -> List[ResearchHypothesis]:
        """
        Returns pre-formulated hypotheses directly addressing the open DEC-021 audit findings.
        """
        candidates = [
            self.create_hypothesis(
                hypothesis_id="HYP-001",
                research_question="RQ-001: Does removing the dummy zero-placeholder clock feature impact baseline performance?",
                hypothesis="Removing the all-zero 'clock' feature from x_meta will not degrade Macro-F1 and will reduce input dimension by 1.",
                motivation="DEC-021 audit identified that 'clock' in ControlledSleepDataset is hardcoded to np.zeros_like(time_feat), contributing zero temporal signal.",
                expected_mechanism="Linear projection layer in meta_proj currently expends parameter capacity learning near-zero weights for a useless column.",
                experiment_design="Train Model A with 4 meta features vs standard 5 meta features for 10 epochs (SHORT budget) on seed 42.",
                expected_outcome="Macro-F1 parity (delta within ±0.002) with 16 fewer parameters in meta_proj.",
                novelty_rationale="Component pruning of previously overlooked dummy input channel",
                feasibility="High; requires minor slice adjustment in ControlledSleepDataset and meta_proj dimension update.",
                risks="Minimal; model capacity remains virtually unchanged.",
                required_compute="SHORT (10 epochs, ~2 minutes on 1x RTX 3060)",
                evaluation_criteria="Validation Macro-F1 parity (Δ >= -0.005) and zero loss divergence."
            ),
            self.create_hypothesis(
                hypothesis_id="HYP-002",
                research_question="RQ-002: Does un-attenuated GRL gradient scaling improve subject-invariant representation learning?",
                hypothesis="Increasing effective GRL gradient scale from -0.010 to -0.050 via scheduled alpha will reduce subject classification accuracy while boosting staging Macro-F1.",
                motivation="DEC-021 audit revealed that lambda_adv=0.10 * alpha_grl=0.10 caused a 100-fold gradient attenuation (-0.010) with set_alpha never called.",
                expected_mechanism="A stronger adversarial gradient forces the temporal BiGRU to purge individual heart-rate baseline offsets.",
                experiment_design="Compare fixed -0.010 scale against Ganin et al. scheduled alpha (reaching 0.50) under SHORT budget (10 epochs).",
                expected_outcome="Higher validation Macro-F1 (+0.010 to +0.020) and lower subject discriminator accuracy.",
                novelty_rationale="Recalibration of domain-adversarial gradient dynamics in wearable sleep staging",
                feasibility="High; uses existing ReverseLayerF with set_alpha called per epoch.",
                risks="Adversarial instability if gradient scale is increased too aggressively.",
                required_compute="SHORT (10 epochs, ~2.5 minutes on 1x RTX 3060)",
                evaluation_criteria="Validation Macro-F1 gain > +0.008 without destabilizing validation loss."
            ),
            self.create_hypothesis(
                hypothesis_id="HYP-003",
                research_question="RQ-003: What is the impact of full-night sequences versus the 5-hour crop on REM sleep recall?",
                hypothesis="Evaluating models on full nights (up to 10 hours) will restore late-night REM sleep representation and increase overall REM recall by >= 5 percentage points.",
                motivation="DEC-021 audit verified that T=600 drops 28.55% of labeled data, causing an artificial 18% relative depletion of REM sleep.",
                expected_mechanism="REM episodes lengthen circadianly across the night; full-night sequences present the complete physiological sleep cycle.",
                experiment_design="Compare validation set metrics on T=600 vs full available night length using zero-padded or sliding window evaluation.",
                expected_outcome="REM recall increases from ~60% to >= 66%; Deep sleep proportion normalizes to ~19%.",
                novelty_rationale="Quantification of sequence-truncation bias on wearable sleep stage architecture",
                feasibility="Medium; requires padding or variable-length DataLoader handling.",
                risks="Increased VRAM consumption for sequence lengths T > 600.",
                required_compute="STANDARD (80 epochs, ~15 minutes on 1x RTX 3060)",
                evaluation_criteria="Per-class REM recall gain >= +5.0 pp and normalized macro distribution."
            )
        ]
        return candidates
