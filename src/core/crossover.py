"""
Crossover Module - Genetic crossover operations for genomes.

Provides various crossover strategies:
- Single-point crossover
- Two-point crossover
- Uniform crossover
- Semantic-aware crossover
"""

from __future__ import annotations
import ast
import random
from typing import Optional
from .genome import Genome, GenomeType


def single_point_crossover(
    parent1: Genome, 
    parent2: Genome
) -> tuple[Genome, Genome]:
    """
    Perform single-point crossover between two genomes.
    
    Args:
        parent1: First parent genome
        parent2: Second parent genome
        
    Returns:
        Tuple of two child genomes
    """
    # Choose crossover point
    max_point = min(len(parent1.source), len(parent2.source))
    if max_point < 2:
        # If genomes are too short, return clones
        return parent1.clone(), parent2.clone()
    
    point = random.randint(1, max_point - 1)
    
    # Create children
    child1_source = parent1.source[:point] + parent2.source[point:]
    child2_source = parent2.source[:point] + parent1.source[point:]
    
    child1 = Genome(
        source=child1_source,
        language=parent1.language,
        genome_type=parent1.genome_type,
        generation=max(parent1.generation, parent2.generation) + 1,
    )
    child1.mutation_history.parent_id = parent1.id
    
    child2 = Genome(
        source=child2_source,
        language=parent2.language,
        genome_type=parent2.genome_type,
        generation=max(parent1.generation, parent2.generation) + 1,
    )
    child2.mutation_history.parent_id = parent2.id
    
    return child1, child2


def two_point_crossover(
    parent1: Genome, 
    parent2: Genome
) -> tuple[Genome, Genome]:
    """
    Perform two-point crossover between two genomes.
    """
    max_point = min(len(parent1.source), len(parent2.source))
    if max_point < 3:
        return single_point_crossover(parent1, parent2)
    
    points = sorted(random.sample(range(1, max_point), 2))
    point1, point2 = points
    
    # Create children
    child1_source = (
        parent1.source[:point1] + 
        parent2.source[point1:point2] + 
        parent1.source[point2:]
    )
    child2_source = (
        parent2.source[:point1] + 
        parent1.source[point1:point2] + 
        parent2.source[point2:]
    )
    
    child1 = Genome(
        source=child1_source,
        language=parent1.language,
        genome_type=parent1.genome_type,
        generation=max(parent1.generation, parent2.generation) + 1,
    )
    
    child2 = Genome(
        source=child2_source,
        language=parent2.language,
        genome_type=parent2.genome_type,
        generation=max(parent1.generation, parent2.generation) + 1,
    )
    
    return child1, child2


def uniform_crossover(
    parent1: Genome, 
    parent2: Genome,
    probability: float = 0.5
) -> tuple[Genome, Genome]:
    """
    Perform uniform crossover with given probability.
    
    Each gene (character) is randomly chosen from either parent.
    """
    max_len = max(len(parent1.source), len(parent2.source))
    
    child1_parts = []
    child2_parts = []
    
    for i in range(max_len):
        if random.random() < probability:
            c1, c2 = parent1.source[i] if i < len(parent1.source) else '', \
                     parent2.source[i] if i < len(parent2.source) else ''
        else:
            c2, c1 = parent1.source[i] if i < len(parent1.source) else '', \
                     parent2.source[i] if i < len(parent2.source) else ''
        
        child1_parts.append(c1)
        child2_parts.append(c2)
    
    child1 = Genome(
        source=''.join(child1_parts),
        language=parent1.language,
        genome_type=parent1.genome_type,
        generation=max(parent1.generation, parent2.generation) + 1,
    )
    
    child2 = Genome(
        source=''.join(child2_parts),
        language=parent2.language,
        genome_type=parent2.genome_type,
        generation=max(parent1.generation, parent2.generation) + 1,
    )
    
    return child1, child2


def semantic_crossover(
    parent1: Genome, 
    parent2: Genome
) -> tuple[Genome, Genome]:
    """
    Semantic-aware crossover that tries to preserve code structure.
    
    Attempts to crossover at function/class boundaries.
    """
    try:
        tree1 = ast.parse(parent1.source)
        tree2 = ast.parse(parent2.source)
        
        # Get top-level function/class boundaries
        boundaries1 = _get_boundaries(tree1)
        boundaries2 = _get_boundaries(tree2)
        
        if not boundaries1 or not boundaries2:
            return single_point_crossover(parent1, parent2)
        
        # Try to crossover at boundaries
        boundary1 = random.choice(boundaries1)
        boundary2 = random.choice(boundaries2)
        
        # Extract sections
        section1_a = parent1.source[:boundary1]
        section1_b = parent1.source[boundary1:]
        section2_a = parent2.source[:boundary2]
        section2_b = parent2.source[boundary2:]
        
        child1_source = section1_a + section2_b
        child2_source = section2_a + section1_b
        
        # Verify syntax
        try:
            ast.parse(child1_source)
            ast.parse(child2_source)
        except SyntaxError:
            return single_point_crossover(parent1, parent2)
        
        child1 = Genome(
            source=child1_source,
            language=parent1.language,
            genome_type=parent1.genome_type,
            generation=max(parent1.generation, parent2.generation) + 1,
        )
        
        child2 = Genome(
            source=child2_source,
            language=parent2.language,
            genome_type=parent2.genome_type,
            generation=max(parent1.generation, parent2.generation) + 1,
        )
        
        return child1, child2
        
    except SyntaxError:
        return single_point_crossover(parent1, parent2)


def _get_boundaries(tree: ast.AST) -> list[int]:
    """Get line numbers of top-level definitions."""
    boundaries = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            boundaries.append(node.lineno)
    return sorted(set(boundaries)) if boundaries else []


class CrossoverStrategy:
    """Strategy pattern for crossover operations."""
    
    STRATEGIES = {
        "single_point": single_point_crossover,
        "two_point": two_point_crossover,
        "uniform": uniform_crossover,
        "semantic": semantic_crossover,
    }
    
    @classmethod
    def get_strategy(cls, name: str):
        """Get crossover strategy by name."""
        return cls.STRATEGIES.get(name, single_point_crossover)
    
    @classmethod
    def random_strategy(cls):
        """Get a random crossover strategy."""
        return random.choice(list(cls.STRATEGIES.values()))
