"""
Selector Module - Multi-objective selection for evolution.

Provides various selection strategies:
- Tournament selection
- Fitness-proportional selection
- Rank-based selection
- Multi-objective selection (NSGA-II, NSGA-III)
- Pareto-based selection
"""

from __future__ import annotations
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
import numpy as np

from ..core.genome import Genome, FitnessScores
from ..core.population import Population


class SelectionMethod(Enum):
    """Selection method types."""
    TOURNAMENT = auto()
    FITNESS_PROPORTIONAL = auto()
    RANK = auto()
    TRUNCATION = auto()
    NSGA2 = auto()
    NSGA3 = auto()
    SPEA2 = auto()
    RANDOM = auto()


@dataclass
class SelectionConfig:
    """Configuration for selection."""
    method: SelectionMethod = SelectionMethod.TOURNAMENT
    tournament_size: int = 3
    elite_size: int = 5
    
    # Multi-objective parameters
    crowding_distance: bool = True
    reference_directions: Optional[list] = None
    
    # Diversity preservation
    niching: bool = True
    niche_count: int = 2
    
    # Stochastic parameters
    selection_pressure: float = 0.7  # 0-1, higher = more selective
    
    
class SelectionResult:
    """Result of selection operation."""
    selected: list[Genome]
    discarded: list[Genome]
    metadata: dict = field(default_factory=dict)


class Selector:
    """
    Multi-objective selection for evolutionary algorithms.
    
    Provides various selection strategies for selecting
    parents and survivors in the evolutionary process.
    """
    
    def __init__(self, config: Optional[SelectionConfig] = None):
        """Initialize selector with configuration."""
        self.config = config or SelectionConfig()
        self.selection_history: list[SelectionResult] = []
    
    def select_parents(
        self, 
        population: Population, 
        count: int = 2
    ) -> list[Genome]:
        """
        Select parent genomes for reproduction.
        
        Args:
            population: Current population
            count: Number of parents to select
            
        Returns:
            List of selected parent genomes
        """
        method_map = {
            SelectionMethod.TOURNAMENT: self._tournament_select,
            SelectionMethod.FITNESS_PROPORTIONAL: self._fitness_proportional_select,
            SelectionMethod.RANK: self._rank_select,
            SelectionMethod.TRUNCATION: self._truncation_select,
            SelectionMethod.RANDOM: self._random_select,
        }
        
        select_func = method_map.get(self.config.method, self._tournament_select)
        
        parents = []
        for _ in range(count):
            parent = select_func(population)
            parents.append(parent)
        
        return parents
    
    def select_survivors(
        self, 
        population: Population, 
        offspring: list[Genome],
        evaluate_offspring: bool = True
    ) -> list[Genome]:
        """
        Select survivors from combined population and offspring.
        
        Args:
            population: Current population
            offspring: New offspring genomes
            evaluate_offspring: Whether to evaluate offspring fitness
            
        Returns:
            List of surviving genomes
        """
        # Combine population and offspring
        combined = list(population.genomes) + list(offspring)
        
        # Evaluate if needed
        if evaluate_offspring:
            from ..core.evaluator import Evaluator
            evaluator = Evaluator()
            for g in combined:
                if g.fitness.composite == 0:
                    g.fitness = evaluator.evaluate(g)
        
        # Apply multi-objective optimization if configured
        if self.config.method in [SelectionMethod.NSGA2, SelectionMethod.NSGA3, SelectionMethod.SPEA2]:
            survivors = self._multi_objective_select(combined, population.config.size)
        else:
            survivors = self._environmental_selection(combined, population.config.size)
        
        # Record selection
        discarded = [g for g in combined if g not in survivors]
        result = SelectionResult(
            selected=survivors,
            discarded=discarded,
            metadata={"method": self.config.method.name},
        )
        self.selection_history.append(result)
        
        return survivors
    
    # Tournament Selection
    
    def _tournament_select(self, population: Population) -> Genome:
        """Select using tournament selection."""
        tournament_size = self.config.tournament_size
        
        # Select tournament participants
        tournament = random.sample(
            list(population.genomes),
            min(tournament_size, len(population.genomes))
        )
        
        # Select winner based on multi-objective criteria
        if self.config.niching:
            # Diversity-aware tournament
            return self._diversity_tournament(tournament)
        else:
            # Standard fitness tournament
            return max(tournament, key=lambda g: g.fitness.composite)
    
    def _diversity_tournament(self, tournament: list[Genome]) -> Genome:
        """Tournament selection with diversity consideration."""
        # Calculate crowding distances
        crowding = self._calculate_crowding(tournament)
        
        # Combined score: fitness + diversity bonus
        scores = []
        for g in tournament:
            fitness_score = g.fitness.composite
            diversity_score = crowding.get(g.id, 0)
            
            # Weighted combination
            combined = (
                self.config.selection_pressure * fitness_score +
                (1 - self.config.selection_pressure) * diversity_score
            )
            scores.append((g, combined))
        
        return max(scores, key=lambda x: x[1])[0]
    
    # Fitness-Proportional Selection
    
    def _fitness_proportional_select(self, population: Population) -> Genome:
        """Select using fitness-proportional (roulette wheel) selection."""
        genomes = list(population.genomes)
        
        # Handle zero or negative fitness
        min_fitness = min(g.fitness.composite for g in genomes)
        if min_fitness < 0:
            # Shift fitness values to be positive
            shift = abs(min_fitness) + 1
            adjusted = [g.fitness.composite + shift for g in genomes]
        else:
            adjusted = [g.fitness.composite for g in genomes]
        
        total = sum(adjusted)
        if total == 0:
            return random.choice(genomes)
        
        # Normalize to probabilities
        probs = [f / total for f in adjusted]
        
        return random.choices(genomes, weights=probs, k=1)[0]
    
    # Rank Selection
    
    def _rank_select(self, population: Population) -> Genome:
        """Select using rank-based selection."""
        genomes = list(population.genomes)
        
        # Sort by fitness
        sorted_genomes = sorted(genomes, key=lambda g: g.fitness.composite)
        
        # Assign ranks (1 = worst, n = best)
        ranks = {g: i + 1 for i, g in enumerate(sorted_genomes)}
        
        # Calculate selection probabilities based on rank
        n = len(genomes)
        total_rank = n * (n + 1) / 2
        probs = [ranks[g] / total_rank for g in genomes]
        
        return random.choices(genomes, weights=probs, k=1)[0]
    
    # Truncation Selection
    
    def _truncation_select(self, population: Population) -> Genome:
        """Select from top fraction of population."""
        genomes = list(population.genomes)
        threshold = int(len(genomes) * self.config.selection_pressure)
        
        # Select from top performers
        sorted_genomes = sorted(genomes, key=lambda g: g.fitness.composite, reverse=True)
        candidates = sorted_genomes[:max(threshold, 1)]
        
        return random.choice(candidates)
    
    # Random Selection
    
    def _random_select(self, population: Population) -> Genome:
        """Select randomly."""
        return random.choice(list(population.genomes))
    
    # Multi-Objective Selection
    
    def _multi_objective_select(
        self, 
        genomes: list[Genome], 
        target_size: int
    ) -> list[Genome]:
        """Multi-objective selection using NSGA-II/III."""
        # Fast non-dominated sorting
        fronts = self._fast_non_dominated_sort(genomes)
        
        selected = []
        front_idx = 0
        
        while len(selected) + len(fronts[front_idx]) <= target_size:
            # Add entire front
            selected.extend(fronts[front_idx])
            front_idx += 1
            
            if front_idx >= len(fronts):
                break
        
        # Partial last front - use crowding distance
        if len(selected) < target_size and front_idx < len(fronts):
            remaining = target_size - len(selected)
            partial_front = fronts[front_idx]
            
            # Calculate crowding distances
            crowding = self._calculate_crowding(partial_front)
            
            # Sort by crowding distance (descending)
            sorted_front = sorted(
                partial_front,
                key=lambda g: crowding.get(g.id, 0),
                reverse=True
            )
            
            selected.extend(sorted_front[:remaining])
        
        return selected
    
    def _fast_non_dominated_sort(self, genomes: list[Genome]) -> list[list[Genome]]:
        """Fast non-dominated sorting (NSGA-II)."""
        fronts = [[]]
        
        domination_count = {g.id: 0 for g in genomes}
        dominated_solutions = {g.id: [] for g in genomes}
        
        for i, p in enumerate(genomes):
            for j, q in enumerate(genomes):
                if i == j:
                    continue
                    
                if self._dominates(p, q):
                    dominated_solutions[p.id].append(q)
                elif self._dominates(q, p):
                    domination_count[p.id] += 1
            
            if domination_count[p.id] == 0:
                fronts[0].append(p)
        
        # Build subsequent fronts
        current_front = 0
        while fronts[current_front]:
            next_front = []
            
            for p in fronts[current_front]:
                for q_id in dominated_solutions[p.id]:
                    domination_count[q_id] -= 1
                    if domination_count[q_id] == 0:
                        # Find the genome with this ID
                        for g in genomes:
                            if g.id == q_id:
                                next_front.append(g)
                                break
            
            current_front += 1
            if next_front:
                fronts.append(next_front)
        
        return [f for f in fronts if f]  # Remove empty fronts
    
    def _dominates(self, p: Genome, q: Genome) -> bool:
        """Check if p dominates q (Pareto dominance)."""
        fp = p.fitness
        fq = q.fitness
        
        # Check if p is better in at least one objective and not worse in any
        better_in_any = False
        
        objectives = [
            (fp.correctness, fq.correctness),  # Maximize
            (fp.performance, fq.performance),  # Maximize
            (1 - fp.complexity, 1 - fq.complexity),  # Minimize complexity
            (fp.maintainability, fq.maintainability),  # Maximize
            (1 - fp.memory, 1 - fq.memory),  # Minimize memory
            (fp.readability, fq.readability),  # Maximize
        ]
        
        for p_val, q_val in objectives:
            if p_val > q_val:
                better_in_any = True
            elif p_val < q_val:
                return False
        
        return better_in_any
    
    def _calculate_crowding(self, genomes: list[Genome]) -> dict[str, float]:
        """Calculate crowding distance for genomes."""
        if len(genomes) <= 2:
            return {g.id: float('inf') for g in genomes}
        
        crowding = {g.id: 0.0 for g in genomes}
        
        objectives = [
            ('correctness', True),   # maximize
            ('performance', True),   # maximize
            ('complexity', False),   # minimize
            ('maintainability', True),  # maximize
            ('memory', False),        # minimize
            ('readability', True),   # maximize
        ]
        
        for obj_name, maximize in objectives:
            # Sort by objective value
            sorted_genomes = sorted(
                genomes,
                key=lambda g: getattr(g.fitness, obj_name)
            )
            
            # Boundary points get infinite distance
            crowding[sorted_genomes[0].id] = float('inf')
            crowding[sorted_genomes[-1].id] = float('inf')
            
            # Calculate range
            obj_values = [getattr(g.fitness, obj_name) for g in sorted_genomes]
            value_range = max(obj_values) - min(obj_values)
            
            if value_range == 0:
                continue
            
            # Calculate crowding distance
            for i in range(1, len(sorted_genomes) - 1):
                g = sorted_genomes[i]
                prev_val = getattr(sorted_genomes[i-1].fitness, obj_name)
                next_val = getattr(sorted_genomes[i+1].fitness, obj_name)
                
                if maximize:
                    crowding[g.id] += (next_val - prev_val) / value_range
                else:
                    crowding[g.id] += (prev_val - next_val) / value_range
        
        return crowding
    
    def _environmental_selection(
        self, 
        genomes: list[Genome], 
        target_size: int
    ) -> list[Genome]:
        """Environmental selection to reduce population size."""
        if len(genomes) <= target_size:
            return genomes
        
        # Use NSGA-II style selection
        return self._multi_objective_select(genomes, target_size)
    
    # Utility methods
    
    def select_elite(
        self, 
        genomes: list[Genome], 
        elite_size: Optional[int] = None
    ) -> list[Genome]:
        """Select elite (best) genomes."""
        elite_size = elite_size or self.config.elite_size
        
        sorted_genomes = sorted(
            genomes,
            key=lambda g: g.fitness.composite,
            reverse=True
        )
        
        return sorted_genomes[:elite_size]
    
    def select_diverse(
        self, 
        genomes: list[Genome], 
        count: int
    ) -> list[Genome]:
        """Select diverse set of genomes."""
        if len(genomes) <= count:
            return genomes
        
        selected = [genomes[0]]  # Start with first genome
        remaining = list(genomes[1:])
        
        while len(selected) < count and remaining:
            # Find genome farthest from selected
            best_distance = -1
            best_genome = None
            
            for g in remaining:
                min_distance = min(
                    self._genome_distance(g, s) 
                    for s in selected
                )
                if min_distance > best_distance:
                    best_distance = min_distance
                    best_genome = g
            
            if best_genome:
                selected.append(best_genome)
                remaining.remove(best_genome)
        
        return selected
    
    def _genome_distance(self, g1: Genome, g2: Genome) -> float:
        """Calculate distance between two genomes."""
        # Hamming distance on source code
        max_len = max(len(g1.source), len(g2.source))
        if max_len == 0:
            return 0
        
        matching = sum(
            1 for i in range(min(len(g1.source), len(g2.source)))
            if g1.source[i] == g2.source[i]
        )
        
        return 1 - (matching / max_len)


# Convenience function
def select_parents(population: Population, count: int = 2) -> list[Genome]:
    """Quickly select parents from population."""
    selector = Selector()
    return selector.select_parents(population, count)
