"""
Population Module - Manages populations of genomes for evolution.

The Population class manages the population of genomes, including:
- Population initialization and maintenance
- Diversity tracking
- Generation management
- Population statistics
"""

from __future__ import annotations
import random
import statistics
from dataclasses import dataclass, field
from typing import Optional, Callable, Iterator
from collections import defaultdict

from .genome import Genome, FitnessScores, GenomeType
from .evaluator import Evaluator


@dataclass
class PopulationConfig:
    """Configuration for population management."""
    size: int = 50
    elite_size: int = 5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    tournament_size: int = 3
    max_stagnation: int = 20
    diversity_threshold: float = 0.1
    
    # Selection parameters
    selection_pressure: float = 0.5  # 0 = random, 1 = fitness-based
    
    # Diversity maintenance
    niching: bool = True
    niche_radius: float = 0.2
    
    # Age-based evolution
    allow_aging: bool = False
    max_age: int = 50
    aging_rate: float = 0.01


@dataclass
class PopulationStats:
    """Statistics about the population."""
    generation: int = 0
    size: int = 0
    
    # Fitness statistics
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    worst_fitness: float = 0.0
    fitness_std: float = 0.0
    
    # Diversity metrics
    diversity: float = 0.0
    unique_genomes: int = 0
    
    # Performance
    avg_execution_time: float = 0.0
    avg_complexity: float = 0.0
    
    # Convergence
    stagnation_counter: int = 0
    converged: bool = False
    
    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "size": self.size,
            "best_fitness": self.best_fitness,
            "avg_fitness": self.avg_fitness,
            "worst_fitness": self.worst_fitness,
            "fitness_std": self.fitness_std,
            "diversity": self.diversity,
            "unique_genomes": self.unique_genomes,
            "stagnation_counter": self.stagnation_counter,
            "converged": self.converged,
        }


class Population:
    """
    Manages a population of genomes for evolutionary computation.
    
    The Population class provides:
    - Population initialization and maintenance
    - Statistics tracking
    - Diversity management
    - Generation advancement
    """
    
    def __init__(
        self, 
        config: Optional[PopulationConfig] = None,
        evaluator: Optional[Evaluator] = None,
    ):
        """Initialize population with configuration."""
        self.config = config or PopulationConfig()
        self.evaluator = evaluator or Evaluator()
        
        # Population data
        self.genomes: list[Genome] = []
        self.generation: int = 0
        self.best_genome: Optional[Genome] = None
        
        # History for tracking
        self.history: list[PopulationStats] = []
        self.best_ever: Optional[Genome] = None
        
        # Convergence tracking
        self.stagnation: int = 0
        self.converged: bool = False
        
    def initialize(
        self, 
        template: Optional[str] = None,
        genome_type: GenomeType = GenomeType.FUNCTION,
    ) -> None:
        """
        Initialize the population.
        
        Args:
            template: Optional template code to base initial genomes on
            genome_type: Type of genomes to create
        """
        self.genomes = []
        
        for i in range(self.config.size):
            genome = Genome(
                source=template or self._generate_initial_code(i),
                language="python",
                genome_type=genome_type,
                generation=0,
            )
            self.genomes.append(genome)
        
        # Evaluate initial population
        self._evaluate_all()
        
        # Update best genome
        self._update_best()
        
    def _generate_initial_code(self, seed: int) -> str:
        """Generate initial code for a genome."""
        random.seed(seed)
        
        templates = [
            """def solve(input_data):
    '''Solve the given problem.'''
    return input_data
""",
            """def solve(n):
    '''Calculate result for n.'''
    if n <= 0:
        return 0
    return n * 2
""",
            """def solve(data):
    '''Process data.'''
    result = []
    for item in data:
        result.append(item)
    return result
""",
            """def solve(x, y):
    '''Combine x and y.'''
    return x + y
""",
            """def solve(n):
    '''Recursive solution.'''
    if n <= 1:
        return n
    return solve(n-1) + solve(n-2)
""",
        ]
        
        return random.choice(templates)
    
    def _evaluate_all(self) -> None:
        """Evaluate all genomes in the population."""
        for genome in self.genomes:
            if genome.fitness.composite == 0.0:
                genome.fitness = self.evaluator.evaluate(genome)
    
    def _update_best(self) -> None:
        """Update the best genome in the population."""
        if not self.genomes:
            return
            
        sorted_genomes = sorted(
            self.genomes, 
            key=lambda g: g.fitness.composite, 
            reverse=True
        )
        self.best_genome = sorted_genomes[0]
        
        # Update best ever if this is better
        if (self.best_ever is None or 
            self.best_genome.fitness.composite > self.best_ever.fitness.composite):
            self.best_ever = self.best_genome.clone()
    
    def select_parent(self) -> Genome:
        """
        Select a parent genome using configured selection method.
        
        Returns:
            Selected parent genome
        """
        pressure = self.config.selection_pressure
        
        if pressure < 0.3:
            # Mostly random selection
            return random.choice(self.genomes)
        
        elif pressure < 0.7:
            # Tournament selection
            return self._tournament_select()
        
        else:
            # Fitness-proportional selection
            return self._fitness_proportional_select()
    
    def _tournament_select(self) -> Genome:
        """Select genome via tournament selection."""
        tournament = random.sample(
            self.genomes, 
            min(self.config.tournament_size, len(self.genomes))
        )
        return max(tournament, key=lambda g: g.fitness.composite)
    
    def _fitness_proportional_select(self) -> Genome:
        """Select genome via fitness-proportional selection (roulette wheel)."""
        fitnesses = [g.fitness.composite for g in self.genomes]
        total_fitness = sum(fitnesses)
        
        if total_fitness <= 0:
            return random.choice(self.genomes)
        
        # Normalize to probabilities
        probs = [f / total_fitness for f in fitnesses]
        
        # Weighted random selection
        return random.choices(self.genomes, weights=probs, k=1)[0]
    
    def select_parents(self, count: int = 2) -> list[Genome]:
        """Select multiple parents for reproduction."""
        return [self.select_parent() for _ in range(count)]
    
    def get_elite(self) -> list[Genome]:
        """Get the elite genomes (top performers)."""
        sorted_genomes = sorted(
            self.genomes,
            key=lambda g: g.fitness.composite,
            reverse=True
        )
        return sorted_genomes[:self.config.elite_size]
    
    def add_genome(self, genome: Genome) -> None:
        """Add a genome to the population."""
        self.genomes.append(genome)
    
    def remove_genome(self, genome: Genome) -> None:
        """Remove a genome from the population."""
        if genome in self.genomes:
            self.genomes.remove(genome)
    
    def replace_genome(self, old: Genome, new: Genome) -> None:
        """Replace a genome in the population."""
        idx = self.genomes.index(old) if old in self.genomes else -1
        if idx >= 0:
            self.genomes[idx] = new
    
    def get_statistics(self) -> PopulationStats:
        """Calculate current population statistics."""
        if not self.genomes:
            return PopulationStats()
        
        fitnesses = [g.fitness.composite for g in self.genomes]
        unique_hashes = set(g.hash for g in self.genomes)
        
        stats = PopulationStats(
            generation=self.generation,
            size=len(self.genomes),
            best_fitness=max(fitnesses) if fitnesses else 0.0,
            avg_fitness=statistics.mean(fitnesses) if fitnesses else 0.0,
            worst_fitness=min(fitnesses) if fitnesses else 0.0,
            fitness_std=statistics.stdev(fitnesses) if len(fitnesses) > 1 else 0.0,
            diversity=self._calculate_diversity(),
            unique_genomes=len(unique_hashes),
            stagnation_counter=self.stagnation,
            converged=self.converged,
        )
        
        return stats
    
    def _calculate_diversity(self) -> float:
        """
        Calculate population diversity.
        
        Returns:
            Diversity score from 0.0 (identical) to 1.0 (highly diverse)
        """
        if len(self.genomes) <= 1:
            return 1.0
        
        # Calculate genetic diversity based on source code differences
        total_distance = 0.0
        comparisons = 0
        
        for i, g1 in enumerate(self.genomes):
            for g2 in self.genomes[i+1:]:
                distance = self._hamming_distance(g1.source, g2.source)
                total_distance += distance
                comparisons += 1
        
        if comparisons == 0:
            return 1.0
        
        avg_distance = total_distance / comparisons
        
        # Normalize to 0-1 range (assuming max distance of 1000)
        return min(1.0, avg_distance / 1000)
    
    def _hamming_distance(self, s1: str, s2: str) -> float:
        """Calculate normalized Hamming distance between two strings."""
        max_len = max(len(s1), len(s2))
        if max_len == 0:
            return 0.0
        
        min_len = min(len(s1), len(s2))
        matching = sum(1 for i in range(min_len) if s1[i] == s2[i])
        
        return 1.0 - (matching / max_len)
    
    def enforce_size(self) -> None:
        """Enforce population size limits."""
        # Remove worst genomes if population is too large
        while len(self.genomes) > self.config.size:
            sorted_genomes = sorted(
                self.genomes,
                key=lambda g: g.fitness.composite
            )
            self.genomes = sorted_genomes[:-1]
        
        # If population is too small, add mutated copies of best
        while len(self.genomes) < self.config.size:
            if self.genomes:
                best = max(self.genomes, key=lambda g: g.fitness.composite)
                mutated = best.clone()
                mutated.id = f"{mutated.id}_dup"
                self.genomes.append(mutated)
            else:
                break
    
    def next_generation(self) -> None:
        """Advance to the next generation."""
        self.generation += 1
        
        # Update generation numbers
        for genome in self.genomes:
            genome.generation = self.generation
        
        # Record statistics
        stats = self.get_statistics()
        self.history.append(stats)
        
        # Check for stagnation
        if len(self.history) >= 2:
            prev_fitness = self.history[-2].best_fitness
            curr_fitness = stats.best_fitness
            
            if curr_fitness <= prev_fitness:
                self.stagnation += 1
            else:
                self.stagnation = 0
        
        # Check for convergence
        if self.stagnation >= self.config.max_stagnation:
            self.converged = True
        
        # Check diversity
        if stats.diversity < self.config.diversity_threshold:
            self._inject_diversity()
    
    def _inject_diversity(self) -> None:
        """Inject new random genomes to increase diversity."""
        # Replace some worst genomes with random new ones
        sorted_genomes = sorted(
            self.genomes,
            key=lambda g: g.fitness.composite
        )
        
        num_to_replace = max(1, len(self.genomes) // 4)
        
        for i in range(num_to_replace):
            if sorted_genomes[i] in self.genomes:
                self.genomes.remove(sorted_genomes[i])
        
        # Add new random genomes
        for _ in range(num_to_replace):
            self.genomes.append(Genome(
                source=self._generate_initial_code(random.randint(0, 1000)),
                generation=self.generation,
            ))
    
    def niching(self) -> None:
        """
        Apply niching to maintain diversity in fitness landscape.
        
        Reduces competition between similar genomes to preserve diversity.
        """
        if not self.config.niching:
            return
        
        # Calculate niches
        niches: dict[int, list[Genome]] = defaultdict(list)
        
        for genome in self.genomes:
            # Simple niching based on fitness bucket
            niche_id = int(genome.fitness.composite * 10)
            niches[niche_id].append(genome)
        
        # Adjust fitness based on niche crowding
        for niche_id, niche_genomes in niches.items():
            if len(niche_genomes) > 2:
                # Reduce fitness for crowded niches
                crowding_factor = 1.0 / (1.0 + len(niche_genomes) * self.config.niche_radius)
                for genome in niche_genomes:
                    genome.fitness.composite *= crowding_factor
    
    def prune_duplicates(self) -> int:
        """Remove duplicate genomes from population."""
        seen = set()
        removed = 0
        
        new_genomes = []
        for genome in self.genomes:
            if genome.hash not in seen:
                seen.add(genome.hash)
                new_genomes.append(genome)
            else:
                removed += 1
        
        self.genomes = new_genomes
        return removed
    
    def archive(self) -> list[Genome]:
        """Get genomes suitable for archiving to knowledge base."""
        # Archive elite genomes and diverse samples
        archived = []
        
        # Archive best genomes
        elite = self.get_elite()
        archived.extend(elite)
        
        # Archive diverse samples
        if len(self.genomes) > self.config.elite_size:
            non_elite = [g for g in self.genomes if g not in elite]
            
            # Sample from different fitness ranges
            fitness_ranges = 3
            range_size = len(non_elite) // fitness_ranges
            
            for i in range(fitness_ranges):
                start = i * range_size
                end = start + range_size if i < fitness_ranges - 1 else len(non_elite)
                range_genomes = sorted(
                    non_elite[start:end], 
                    key=lambda g: g.fitness.composite,
                    reverse=True
                )
                if range_genomes:
                    archived.append(range_genomes[0])
        
        return archived
    
    def __iter__(self) -> Iterator[Genome]:
        """Iterate over genomes in population."""
        return iter(self.genomes)
    
    def __len__(self) -> int:
        """Get population size."""
        return len(self.genomes)
    
    def __getitem__(self, index: int) -> Genome:
        """Get genome by index."""
        return self.genomes[index]
