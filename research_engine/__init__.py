"""
research_engine package.
"""

from research_engine.resource_manager import GPUResourceManager, ResourceAllocationError
from research_engine.literature_agent import LiteratureAgent, EvidenceCategory
from research_engine.hypothesis_agent import HypothesisAgent, ResearchHypothesis
from research_engine.experiment_designer import ExperimentDesigner, ExperimentSpecification, ExperimentDesignError
from research_engine.scientific_reviewer import ScientificReviewer, ReviewVerdict
from research_engine.decision_logic import DecisionEngine, ExperimentDecision
from research_engine.experiment_engine import ExperimentEngine

__all__ = [
    "GPUResourceManager",
    "ResourceAllocationError",
    "LiteratureAgent",
    "EvidenceCategory",
    "HypothesisAgent",
    "ResearchHypothesis",
    "ExperimentDesigner",
    "ExperimentSpecification",
    "ExperimentDesignError",
    "ScientificReviewer",
    "ReviewVerdict",
    "DecisionEngine",
    "ExperimentDecision",
    "ExperimentEngine"
]
