"""
Prometheus - Self-Improving Code Evolution Engine
A Darwinian code evolution system that generates, evaluates, mutates, and selects
optimal code variants through multi-objective optimization.
"""

__version__ = "1.0.0"
__author__ = "Prometheus Team"

from .core.genome import Genome
from .core.evaluator import Evaluator
from .core.population import Population
from .evolution.generator import Generator
from .evolution.mutator import Mutator
from .evolution.selector import Selector
from .knowledge.base import KnowledgeBase
from .dream.loop import DreamLoop

__all__ = [
    "Genome",
    "Evaluator", 
    "Population",
    "Generator",
    "Mutator",
    "Selector",
    "KnowledgeBase",
    "DreamLoop",
]
