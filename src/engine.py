"""
Prometheus Evolution Engine - Main orchestration module.

This module ties together all components:
- Generator: Creates code variants
- Mutator: Applies genetic mutations
- Selector: Multi-objective selection
- Evaluator: Measures fitness
- Knowledge Base: Stores patterns
- Dream Loop: Knowledge distillation
"""

from __future__ import annotations
import time
import random
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Iterator
from pathlib import Path

from .core.genome import Genome, GenomeType, FitnessScores
from .core.population import Population, PopulationConfig, PopulationStats
from .core.evaluator import Evaluator
from .core.metrics import MetricsCollector
from .evolution.generator import Generator, GenerationConfig, ThinkingLevel
from .evolution.mutator import Mutator, MutationOperator
from .evolution.selector import Selector, SelectionConfig, SelectionMethod
from .knowledge.base import KnowledgeBase, PatternType
from .dream.loop import DreamLoop, DreamConfig, DreamResult


@dataclass
class EvolutionConfig:
    """Main configuration for the evolution engine."""
    # Population
    population_size: int = 50
    elite_size: int = 5
    
    # Evolution parameters
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    
    # Thinking levels
    thinking_level: ThinkingLevel = ThinkingLevel.DELIBERATE
    thinking_budget: int = 500
    
    # Termination
    target_fitness: float = 0.95
    stagnation_limit: int = 20
    
    # Output
    log_interval: int = 5
    save_best: bool = True
    checkpoint_dir: Optional[Path] = None
    
    # Knowledge integration
    use_knowledge_base: bool = True
    knowledge_base_size: int = 1000
    
    # Dream loop
    enable_dream_loop: bool = True
    dream_interval: int = 10


@dataclass
class EvolutionResult:
    """Result of evolution process."""
    success: bool
    best_genome: Optional[Genome]
    generations: int
    duration_seconds: float
    final_population: list[Genome]
    metrics: dict
    dream_results: list[DreamResult] = field(default_factory=list)
    
    @property
    def best_fitness(self) -> float:
        return self.best_genome.fitness.composite if self.best_genome else 0.0


class Prometheus:
    """
    Self-Improving Code Evolution Engine.
    
    Prometheus orchestrates the entire Darwinian code evolution process:
    1. Initialize population
    2. Evaluate fitness
    3. Select parents
    4. Generate offspring (mutation/crossover)
    5. Evaluate offspring
    6. Select survivors
    7. Dream loop (periodic knowledge distillation)
    8. Repeat until termination
    """
    
    def __init__(
        self, 
        config: Optional[EvolutionConfig] = None,
        seed_template: Optional[str] = None,
    ):
        """Initialize Prometheus evolution engine."""
        self.config = config or EvolutionConfig()
        self.seed_template = seed_template
        
        # Initialize components
        self.evaluator = Evaluator()
        self.population = Population(
            config=PopulationConfig(
                size=self.config.population_size,
                elite_size=self.config.elite_size,
                mutation_rate=self.config.mutation_rate,
                crossover_rate=self.config.crossover_rate,
            ),
            evaluator=self.evaluator,
        )
        
        self.generator = Generator(
            config=GenerationConfig(
                thinking_level=self.config.thinking_level,
                thinking_budget_tokens=self.config.thinking_budget,
            )
        )
        
        self.mutator = Mutator()
        
        self.selector = Selector(
            config=SelectionConfig(
                method=SelectionMethod.NSGA2,
                elite_size=self.config.elite_size,
            )
        )
        
        self.knowledge_base = KnowledgeBase(
            config={"max_patterns": self.config.knowledge_base_size}
        ) if self.config.use_knowledge_base else None
        
        self.dream_loop = DreamLoop(
            knowledge_base=self.knowledge_base,
            config=DreamConfig(
                enabled=self.config.enable_dream_loop,
                interval_generations=self.config.dream_interval,
            )
        ) if self.knowledge_base else None
        
        self.metrics = MetricsCollector()
        
        # State
        self.current_generation = 0
        self.is_running = False
        self.should_stop = False
        
        # Callbacks
        self.on_generation: Optional[Callable] = None
        self.on_dream: Optional[Callable] = None
        self.on_best_found: Optional[Callable] = None
    
    def initialize(self) -> None:
        """Initialize the evolution process."""
        # Initialize population
        self.population.initialize(
            template=self.seed_template,
            genome_type=GenomeType.FUNCTION,
        )
        
        self.current_generation = 0
        self.metrics.record(0, **self.population.get_statistics().to_dict())
    
    def evolve(
        self, 
        max_generations: Optional[int] = None,
        target_fitness: Optional[float] = None,
    ) -> EvolutionResult:
        """
        Run the evolution process.
        
        Args:
            max_generations: Maximum generations to run
            target_fitness: Target fitness to stop at
            
        Returns:
            EvolutionResult with final state
        """
        max_generations = max_generations or self.config.max_generations
        target_fitness = target_fitness or self.config.target_fitness
        
        self.is_running = True
        self.should_stop = False
        
        start_time = time.time()
        best_ever = None
        dream_results = []
        
        # Initialize if not already done
        if not self.population.genomes:
            self.initialize()
        
        try:
            for generation in range(self.current_generation, max_generations):
                if self.should_stop:
                    break
                
                self.current_generation = generation + 1
                
                # Run generation
                self._run_generation()
                
                # Check for new best
                current_best = self.population.best_genome
                if current_best and (
                    best_ever is None or 
                    current_best.fitness.composite > best_ever.fitness.composite
                ):
                    best_ever = current_best.clone()
                    
                    # Fire callback
                    if self.on_best_found:
                        self.on_best_found(best_ever)
                
                # Log progress
                if generation % self.config.log_interval == 0:
                    self._log_progress(generation, best_ever)
                
                # Dream loop
                if self.dream_loop and self.dream_loop.should_dream(generation):
                    dream_result = self.dream_loop.dream()
                    dream_results.append(dream_result)
                    
                    if self.on_dream:
                        self.on_dream(dream_result)
                
                # Fire generation callback
                if self.on_generation:
                    self.on_generation(generation, self.population)
                
                # Check termination
                if best_ever and best_ever.fitness.composite >= target_fitness:
                    break
                
                if self.population.converged:
                    break
        
        finally:
            self.is_running = False
        
        duration = time.time() - start_time
        
        return EvolutionResult(
            success=best_ever is not None and best_ever.fitness.composite >= target_fitness,
            best_genome=best_ever,
            generations=self.current_generation,
            duration_seconds=duration,
            final_population=list(self.population.genomes),
            metrics=self.metrics.get_summary(),
            dream_results=dream_results,
        )
    
    def _run_generation(self) -> None:
        """Run one generation of evolution."""
        # 1. Evaluate all genomes
        self.population._evaluate_all()
        
        # 2. Archive for dream loop
        if self.dream_loop:
            for genome in self.population.genomes:
                self.dream_loop.archive(genome)
        
        # 3. Select parents
        parents = []
        for _ in range(self.config.population_size // 2):
            parent_pair = self.selector.select_parents(self.population, count=2)
            parents.append(parent_pair)
        
        # 4. Generate offspring
        offspring = []
        
        for parent1, parent2 in parents:
            # Crossover
            if random.random() < self.config.crossover_rate:
                from .core.crossover import CrossoverStrategy
                child1, child2 = CrossoverStrategy.single_strategy(parent1, parent2)
            else:
                child1 = parent1.clone()
                child2 = parent2.clone()
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self.mutator.mutate(child1)
            
            if random.random() < self.config.mutation_rate:
                child2 = self.mutator.mutate(child2)
            
            offspring.extend([child1, child2])
        
        # 5. Evaluate offspring
        for g in offspring:
            g.fitness = self.evaluator.evaluate(g)
        
        # 6. Select survivors
        survivors = self.selector.select_survivors(self.population, offspring)
        
        # 7. Update population
        self.population.genomes = survivors
        self.population.enforce_size()
        self.population.next_generation()
        
        # 8. Record metrics
        stats = self.population.get_statistics()
        self.metrics.record(
            self.current_generation,
            **stats.to_dict()
        )
    
    def _log_progress(self, generation: int, best: Optional[Genome]) -> None:
        """Log evolution progress."""
        stats = self.population.get_statistics()
        
        print(f"\n{'='*60}")
        print(f"Generation {generation}")
        print(f"{'='*60}")
        print(f"Population: {stats.size} genomes")
        print(f"Best fitness: {stats.best_fitness:.4f}")
        print(f"Avg fitness: {stats.avg_fitness:.4f}")
        print(f"Diversity: {stats.diversity:.4f}")
        
        if best:
            print(f"\nBest genome ({best.id[:8]}):")
            print(f"  Lines: {best.line_count}")
            print(f"  Correctness: {best.fitness.correctness:.2f}")
            print(f"  Performance: {best.fitness.performance:.2f}")
        
        if self.dream_loop:
            dream_stats = self.dream_loop.get_statistics()
            print(f"\nDream loop stats:")
            print(f"  Dreams executed: {dream_stats['dream_count']}")
            print(f"  Archives: {dream_stats['archives_size']}")
        
        print(f"{'='*60}\n")
    
    def stop(self) -> None:
        """Stop the evolution process."""
        self.should_stop = True
    
    def get_best(self) -> Optional[Genome]:
        """Get the best genome found so far."""
        return self.population.best_ever
    
    def get_knowledge_stats(self) -> Optional[dict]:
        """Get knowledge base statistics."""
        if self.knowledge_base:
            return self.knowledge_base.get_statistics()
        return None
    
    def save_checkpoint(self, path: Optional[Path] = None) -> Path:
        """Save evolution checkpoint."""
        path = path or (self.config.checkpoint_dir / f"checkpoint_{int(time.time())}.json")
        
        import json
        data = {
            "generation": self.current_generation,
            "population": [g.to_dict() for g in self.population.genomes],
            "best": self.population.best_ever.to_dict() if self.population.best_ever else None,
            "metrics": self.metrics.get_summary(),
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return path
    
    def load_checkpoint(self, path: Path) -> None:
        """Load evolution checkpoint."""
        import json
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        self.current_generation = data["generation"]
        self.population.genomes = [Genome.from_dict(g) for g in data["population"]]
        
        if data.get("best"):
            self.population.best_ever = Genome.from_dict(data["best"])
    
    def __repr__(self) -> str:
        return f"Prometheus(gen={self.current_generation}, pop={len(self.population)})"


# Convenience function
def evolve(
    problem: str,
    template: Optional[str] = None,
    max_generations: int = 100,
) -> EvolutionResult:
    """
    Quick evolution function.
    
    Args:
        problem: Problem description
        template: Optional seed template
        max_generations: Maximum generations
        
    Returns:
        EvolutionResult
    """
    engine = Prometheus(
        config=EvolutionConfig(
            max_generations=max_generations,
        ),
        seed_template=template,
    )
    return engine.evolve()
