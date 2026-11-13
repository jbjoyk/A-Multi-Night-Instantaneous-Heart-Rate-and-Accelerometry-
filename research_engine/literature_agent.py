"""
research_engine/literature_agent.py: Literature-Grounded Evidence Management and Fact Auditing.

Inspired by SakanaAI/AI-Scientist:
  1. Manages verified literature context and citations.
  2. Categorizes every proposition into:
       - VERIFIED_LITERATURE_EVIDENCE (peer-reviewed published findings with citation)
       - PROJECT_OBSERVATION (empirically verified results from this repository)
       - LLM_INFERENCE (deductive reasoning based on verified facts)
       - SPECULATIVE_HYPOTHESIS (untested conjectures requiring experimentation)
  3. Rejects unsubstantiated claims of novelty: any claim not backed by literature
     is classified as "candidate novelty requiring literature verification".
  4. Explicitly acknowledges that automated external literature retrieval is:
     LITERATURE_RETRIEVAL_NOT_IMPLEMENTED
"""

from enum import Enum
from typing import Dict, List, Optional, Any


class EvidenceCategory(str, Enum):
    VERIFIED_LITERATURE_EVIDENCE = "VERIFIED_LITERATURE_EVIDENCE"
    PROJECT_OBSERVATION = "PROJECT_OBSERVATION"
    LLM_INFERENCE = "LLM_INFERENCE"
    SPECULATIVE_HYPOTHESIS = "SPECULATIVE_HYPOTHESIS"


class LiteratureAgent:
    """
    Maintains literature evidence, prevents unverified claims, and grounds hypotheses.
    """
    LITERATURE_RETRIEVAL_STATUS = "LITERATURE_RETRIEVAL_NOT_IMPLEMENTED"

    def __init__(self):
        self.literature_index: Dict[str, Dict[str, Any]] = self._init_default_literature()
        self.project_observations: Dict[str, Dict[str, Any]] = self._init_default_observations()

    def search_external_literature(self, query: str) -> Dict[str, Any]:
        """
        Explicitly reports that automated external literature search/retrieval is NOT implemented.
        Prevents hallucinated external literature evidence from entering the system.
        """
        return {
            "status": self.LITERATURE_RETRIEVAL_STATUS,
            "results": [],
            "query": query,
            "message": (
                "LITERATURE_RETRIEVAL_NOT_IMPLEMENTED: Automated external literature retrieval "
                "is disabled. Literature grounding is strictly restricted to locally audited and "
                "verified citations to prevent LLM hallucinations."
            )
        }

    def _init_default_literature(self) -> Dict[str, Dict[str, Any]]:
        return {
            "LIT-001": {
                "key": "SLAMSS-IFS_TBME2026",
                "title": "SLAMSS-IFS: Sleep Staging with Multi-Night Instantaneous Heart Rate and Accelerometry via Invariant Feature Space",
                "authors": "Hosseini et al.",
                "venue": "IEEE Transactions on Biomedical Engineering (TBME), 2026",
                "dataset": "PhysioNet BIDSLab (47 subjects, 253 nights)",
                "reported_metrics": {
                    "accuracy": 0.7104,
                    "macro_f1": 0.7079,
                    "n_parameters": 5470000
                },
                "key_findings": [
                    "Reports 71.04% 4-class accuracy using 5.47M parameter architecture.",
                    "Introduces composite Relative Class Cost (RCC) loss.",
                    "Uses 1 Hz resampled instantaneous heart rate (IHR) and triaxial accelerometry."
                ]
            },
            "LIT-002": {
                "key": "Ganin_ICML2015",
                "title": "Unsupervised Domain Adaptation by Backpropagation",
                "authors": "Ganin, Y. and Lempitsky, V.",
                "venue": "ICML 2015",
                "key_findings": [
                    "Introduces Gradient Reversal Layer (GRL) for domain-adversarial invariance.",
                    "Schedules reversal scale: p = 2/(1 + exp(-10*progress)) - 1."
                ]
            },
            "LIT-003": {
                "key": "PhysioNet_BIDSLab",
                "title": "A Multi-Night Instantaneous Heart Rate and Accelerometry Dataset with EEG Sleep Stage Labels",
                "authors": "BIDSLab UMass",
                "venue": "PhysioNet v1.0.0, 2023",
                "key_findings": [
                    "Cohort of 47 healthy human subjects, 253 total recorded nights.",
                    "Apple Watch PPG and 50 Hz triaxial accelerometer.",
                    "Dreem 2 concurrent EEG headband for 30s polysomnography staging."
                ]
            }
        }

    def _init_default_observations(self) -> Dict[str, Dict[str, Any]]:
        return {
            "OBS-001": {
                "source": "EXP-012",
                "finding": "SleepAdaptiveNet (B0_GRL, 162k params) achieved 68.44% accuracy on the official 11-subject test partition."
            },
            "OBS-002": {
                "source": "EXP-013",
                "finding": "On the 11-subject test partition, SleepAdaptiveNet outperformed matched baseline by +1.79% accuracy (+0.0199 Macro-F1)."
            },
            "OBS-003": {
                "source": "EXP-014",
                "finding": "Across all 47 subjects in 5-fold CV (75 runs), Model E (66.06%) and Model A (66.63%) achieved parity (bootstrap 95% CI crosses zero). Subject variance (ICC=0.963) dominates performance."
            },
            "OBS-004": {
                "source": "DEC-021_AUDIT",
                "finding": "Sequence truncation T=600 drops 28.55% of labeled epochs, enriching Deep sleep (+6.35 pp) and depleting REM (-4.73 pp). Dummy clock feature in x_meta contains constant zeros. GRL effective gradient scale was -0.010 (100x attenuated)."
            }
        }

    def audit_claim(self, claim_text: str, category: EvidenceCategory, citation_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Audits an assertion. Rejects ungrounded claims of literature fact or novelty.
        """
        if category == EvidenceCategory.VERIFIED_LITERATURE_EVIDENCE:
            if not citation_id or citation_id not in self.literature_index:
                return {
                    "approved": False,
                    "reclassified_category": EvidenceCategory.SPECULATIVE_HYPOTHESIS,
                    "reason": f"Claimed as literature evidence but citation '{citation_id}' is not in verified literature index.",
                    "novelty_status": "candidate novelty requiring literature verification"
                }
            return {
                "approved": True,
                "category": EvidenceCategory.VERIFIED_LITERATURE_EVIDENCE,
                "citation": self.literature_index[citation_id],
                "novelty_status": "documented in literature"
            }

        if category == EvidenceCategory.PROJECT_OBSERVATION:
            return {
                "approved": True,
                "category": EvidenceCategory.PROJECT_OBSERVATION,
                "novelty_status": "project empirical finding"
            }

        # Any speculative claim or inference cannot claim established novelty
        return {
            "approved": True,
            "category": category,
            "novelty_status": "candidate novelty requiring literature verification"
        }

    def format_grounding_summary(self) -> str:
        """Returns a concise summary of active literature and empirical observations."""
        lines = ["### Verified Literature Index:"]
        for k, v in self.literature_index.items():
            lines.append(f"- **[{k}] {v['key']}**: {v['title']} ({v['venue']})")
        lines.append("\n### Project Baseline Observations:")
        for k, v in self.project_observations.items():
            lines.append(f"- **[{k}] {v['source']}**: {v['finding']}")
        return "\n".join(lines)
