"""
Genome Module - Represents code as evolvable genetic material.

Each Genome encapsulates:
- Source code as a string
- Metadata (language, complexity, fitness scores)
- Mutation history
- Phenotype mapping
"""

from __future__ import annotations
import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional
import uuid


class GenomeType(Enum):
    """Types of genetic material in the evolution system."""
    FUNCTION = auto()
    CLASS = auto()
    MODULE = auto()
    ALGORITHM = auto()
    TEMPLATE = auto()


class MutationType(Enum):
    """Types of mutations that can be applied."""
    POINT = auto()           # Single point mutation
    BLOCK = auto()           # Block-level mutation
    REFACTOR = auto()        # Structural refactoring
    OPTIMIZE = auto()        # Performance optimization
    CROSSOVER = auto()       # Genetic crossover
    INSERT = auto()          # Insertion mutation
    DELETE = auto()          # Deletion mutation
    DUPLICATE = auto()        # Gene duplication
    SHRINK = auto()          # Genome shrinking
    GROW = auto()            # Genome growth


@dataclass
class FitnessScores:
    """Multi-objective fitness scores for a genome."""
    correctness: float = 0.0      # Test passing rate
    performance: float = 0.0      # Execution speed
    complexity: float = 0.0       # Cyclomatic complexity (lower is better)
    maintainability: float = 0.0   # Code quality metrics
    memory: float = 0.0           # Memory usage (lower is better)
    readability: float = 0.0      # Code readability score
    
    # Weighted composite score
    composite: float = field(default=0.0)
    
    # Dominance ranking for multi-objective optimization
    pareto_rank: int = 0
    
    def update_composite(self, weights: dict[str, float]) -> float:
        """Calculate weighted composite score."""
        self.composite = sum([
            self.correctness * weights.get("correctness", 0.3),
            self.performance * weights.get("performance", 0.2),
            (1 - self.complexity) * weights.get("complexity", 0.1),
            self.maintainability * weights.get("maintainability", 0.15),
            (1 - self.memory) * weights.get("memory", 0.1),
            self.readability * weights.get("readability", 0.15),
        ])
        return self.composite
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return {
            "correctness": self.correctness,
            "performance": self.performance,
            "complexity": self.complexity,
            "maintainability": self.maintainability,
            "memory": self.memory,
            "readability": self.readability,
            "composite": self.composite,
            "pareto_rank": self.pareto_rank,
        }


@dataclass
class MutationHistory:
    """Tracks mutation history for a genome."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    mutations: list[dict] = field(default_factory=list)
    generation: int = 0
    parent_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def add_mutation(self, mutation_type: MutationType, details: dict):
        """Record a mutation event."""
        self.mutations.append({
            "type": mutation_type.name,
            "details": details,
            "timestamp": time.time(),
        })
    
    def get_diversity_score(self) -> float:
        """Calculate diversity based on mutation history."""
        if not self.mutations:
            return 1.0
        unique_types = len(set(m["type"] for m in self.mutations))
        return min(1.0, unique_types / len(MutationType))


@dataclass
class Genome:
    """
    Represents an evolvable unit of code.
    
    The Genome is the fundamental unit of evolution in Prometheus.
    It wraps source code with metadata necessary for genetic algorithms.
    """
    
    # Core attributes
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source: str = ""
    language: str = "python"
    genome_type: GenomeType = GenomeType.FUNCTION
    
    # Phenotype mapping
    ast: Optional[Any] = None  # Abstract syntax tree representation
    bytecode: Optional[bytes] = None
    
    # Evaluation data
    fitness: FitnessScores = field(default_factory=FitnessScores)
    mutation_history: MutationHistory = field(default_factory=MutationHistory)
    
    # Metadata
    metadata: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    generation: int = 0
    
    # Caching
    _hash: Optional[str] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize derived fields after construction."""
        if self.source and not self._hash:
            self._compute_hash()
    
    def _compute_hash(self):
        """Compute content hash for deduplication."""
        content = f"{self.language}:{self.source}"
        self._hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @property
    def hash(self) -> str:
        """Get content hash."""
        if self._hash is None:
            self._compute_hash()
        return self._hash
    
    @property
    def size(self) -> int:
        """Get source code size in characters."""
        return len(self.source)
    
    @property
    def line_count(self) -> int:
        """Get number of lines in source."""
        return len(self.source.splitlines())
    
    def clone(self) -> Genome:
        """Create a deep copy of this genome."""
        import copy
        return Genome(
            source=self.source,
            language=self.language,
            genome_type=self.genome_type,
            metadata=copy.deepcopy(self.metadata),
            generation=self.generation,
        )
    
    def mutate(self, mutator: Callable[[Genome], Genome]) -> Genome:
        """Apply a mutation function to create a new genome."""
        new_genome = mutator(self)
        new_genome.mutation_history.parent_id = self.id
        new_genome.generation = self.generation + 1
        return new_genome
    
    def crossover(self, other: Genome) -> tuple[Genome, Genome]:
        """Perform crossover with another genome."""
        from .crossover import single_point_crossover
        return single_point_crossover(self, other)
    
    def evaluate(self, test_suite: Optional[list] = None) -> FitnessScores:
        """Evaluate fitness scores for this genome."""
        from ..core.evaluator import Evaluator
        evaluator = Evaluator()
        return evaluator.evaluate(self, test_suite)
    
    def to_dict(self) -> dict:
        """Serialize genome to dictionary."""
        return {
            "id": self.id,
            "source": self.source,
            "language": self.language,
            "genome_type": self.genome_type.name,
            "fitness": self.fitness.to_dict(),
            "mutation_history": {
                "id": self.mutation_history.id,
                "mutations": self.mutation_history.mutations,
                "generation": self.mutation_history.generation,
            },
            "metadata": self.metadata,
            "hash": self.hash,
            "created_at": self.created_at,
            "generation": self.generation,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Genome:
        """Deserialize genome from dictionary."""
        fitness = FitnessScores(**data.get("fitness", {}))
        history = MutationHistory(
            id=data.get("mutation_history", {}).get("id", str(uuid.uuid4())),
            mutations=data.get("mutation_history", {}).get("mutations", []),
            generation=data.get("mutation_history", {}).get("generation", 0),
        )
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            source=data.get("source", ""),
            language=data.get("language", "python"),
            genome_type=GenomeType[data.get("genome_type", "FUNCTION")],
            metadata=data.get("metadata", {}),
            generation=data.get("generation", 0),
            fitness=fitness,
            mutation_history=history,
        )
    
    def __str__(self) -> str:
        return f"Genome(id={self.id[:8]}, type={self.genome_type.name}, fitness={self.fitness.composite:.3f})"
    
    def __repr__(self) -> str:
        return self.__str__()
