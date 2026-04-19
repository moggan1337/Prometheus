"""Tests for Prometheus evolution engine."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.genome import Genome, GenomeType, FitnessScores
from src.core.evaluator import Evaluator
from src.core.population import Population, PopulationConfig
from src.evolution.generator import Generator, ThinkingLevel
from src.evolution.mutator import Mutator, MutationOperator
from src.evolution.selector import Selector, SelectionMethod
from src.knowledge.base import KnowledgeBase, Pattern, PatternType
from src.dream.loop import DreamLoop


class TestGenome:
    """Tests for Genome class."""
    
    def test_creation(self):
        """Test basic genome creation."""
        genome = Genome(source="def test(): return 42")
        assert genome.source == "def test(): return 42"
        assert genome.language == "python"
        assert genome.id is not None
        assert genome.hash is not None
    
    def test_clone(self):
        """Test genome cloning."""
        original = Genome(source="def test(): pass")
        cloned = original.clone()
        
        assert cloned.id != original.id
        assert cloned.source == original.source
    
    def test_fitness_scores(self):
        """Test fitness score calculation."""
        fitness = FitnessScores(
            correctness=0.8,
            performance=0.7,
            complexity=0.3,
            maintainability=0.9,
            memory=0.2,
            readability=0.8,
        )
        
        fitness.update_composite({
            "correctness": 0.3,
            "performance": 0.2,
            "complexity": 0.1,
            "maintainability": 0.15,
            "memory": 0.1,
            "readability": 0.15,
        })
        
        assert fitness.composite > 0.5


class TestEvaluator:
    """Tests for Evaluator class."""
    
    def test_basic_evaluation(self):
        """Test basic genome evaluation."""
        genome = Genome(source="def test(): return 42")
        evaluator = Evaluator()
        
        fitness = evaluator.evaluate(genome)
        
        assert fitness.correctness >= 0
        assert fitness.performance >= 0
        assert fitness.composite >= 0


class TestPopulation:
    """Tests for Population class."""
    
    def test_initialization(self):
        """Test population initialization."""
        config = PopulationConfig(size=10)
        population = Population(config=config)
        
        population.initialize()
        
        assert len(population) == 10
        assert population.generation == 0
    
    def test_selection(self):
        """Test parent selection."""
        config = PopulationConfig(size=20)
        population = Population(config=config)
        population.initialize()
        
        selector = Selector()
        parent = selector.select_parents(population, count=1)[0]
        
        assert parent is not None
        assert isinstance(parent, Genome)
    
    def test_statistics(self):
        """Test population statistics."""
        config = PopulationConfig(size=10)
        population = Population(config=config)
        population.initialize()
        
        stats = population.get_statistics()
        
        assert stats.size == 10
        assert "best_fitness" in stats.to_dict()


class TestGenerator:
    """Tests for Generator class."""
    
    def test_generation(self):
        """Test code generation."""
        generator = Generator()
        result = generator.generate({})
        
        assert result.success
        assert result.genome is not None
    
    def test_thinking_levels(self):
        """Test different thinking levels."""
        for level in [ThinkingLevel.INSTINCTIVE, ThinkingLevel.DELIBERATE]:
            generator = Generator()
            generator.config.thinking_level = level
            
            result = generator.generate({})
            
            assert result.success


class TestMutator:
    """Tests for Mutator class."""
    
    def test_mutation(self):
        """Test basic mutation."""
        genome = Genome(source="def test(): return 42")
        mutator = Mutator()
        
        mutated = mutator.mutate(genome)
        
        assert mutated is not None
    
    def test_refactor(self):
        """Test refactoring mutation."""
        genome = Genome(source="def test(x): return x + 1")
        mutator = Mutator()
        
        mutated = mutator.refactor(genome)
        
        assert mutated is not None
    
    def test_add_type_hints(self):
        """Test adding type hints."""
        genome = Genome(source="def test(x): return x")
        mutator = Mutator()
        
        mutated = mutator.add_type_hints(genome)
        
        assert "->" in mutated.source or "int" in mutated.source


class TestSelector:
    """Tests for Selector class."""
    
    def test_tournament_selection(self):
        """Test tournament selection."""
        config = PopulationConfig(size=20)
        population = Population(config=config)
        population.initialize()
        
        selector = Selector()
        parents = selector.select_parents(population, count=5)
        
        assert len(parents) == 5
    
    def test_survivor_selection(self):
        """Test survivor selection."""
        config = PopulationConfig(size=20)
        population = Population(config=config)
        population.initialize()
        
        selector = Selector()
        
        # Create offspring
        offspring = [Genome(source=f"def gen{i}(): return {i}") for i in range(10)]
        
        survivors = selector.select_survivors(population, offspring)
        
        assert len(survivors) == config.size


class TestKnowledgeBase:
    """Tests for KnowledgeBase class."""
    
    def test_add_pattern(self):
        """Test adding patterns."""
        kb = KnowledgeBase()
        
        pattern = Pattern(
            id="test1",
            name="test_pattern",
            pattern_type=PatternType.STRUCTURE,
            source="def test(): pass",
            description="A test pattern",
        )
        
        kb.add(pattern)
        
        assert len(kb) == 1
    
    def test_query(self):
        """Test pattern querying."""
        kb = KnowledgeBase()
        
        # Add patterns
        for i, ptype in enumerate([PatternType.STRUCTURE, PatternType.ALGORITHM]):
            pattern = Pattern(
                id=f"test{i}",
                name=f"pattern_{i}",
                pattern_type=ptype,
                source=f"def test{i}(): pass",
                description=f"Pattern {i}",
            )
            kb.add(pattern)
        
        results = kb.query(pattern_type=PatternType.STRUCTURE)
        
        assert len(results) == 1
    
    def test_similar_patterns(self):
        """Test finding similar patterns."""
        kb = KnowledgeBase()
        
        pattern = Pattern(
            id="test1",
            name="fibonacci",
            pattern_type=PatternType.ALGORITHM,
            source="def fib(n): return fib(n-1) + fib(n-2)",
            description="Fibonacci pattern",
        )
        kb.add(pattern)
        
        similar = kb.find_similar("def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)")
        
        assert len(similar) >= 1


class TestDreamLoop:
    """Tests for DreamLoop class."""
    
    def test_archive(self):
        """Test genome archiving."""
        kb = KnowledgeBase()
        dream = DreamLoop(kb)
        
        genome = Genome(source="def test(): return 42")
        genome.fitness.composite = 0.8
        
        dream.archive(genome)
        
        assert len(dream.genome_archive) == 1
    
    def test_should_dream(self):
        """Test dream trigger logic."""
        kb = KnowledgeBase()
        dream = DreamLoop(kb, config=DreamConfig(interval_generations=5))
        
        # Add some genomes
        for i in range(5):
            g = Genome(source=f"def test{i}(): return {i}")
            g.fitness.composite = 0.7
            dream.archive(g)
        
        # Should trigger at generation 5
        assert dream.should_dream(5)


class TestIntegration:
    """Integration tests."""
    
    def test_full_evolution(self):
        """Test full evolution cycle."""
        from src.engine import Prometheus, EvolutionConfig
        
        config = EvolutionConfig(
            population_size=10,
            max_generations=3,
            enable_dream_loop=False,
        )
        
        engine = Prometheus(config=config)
        result = engine.evolve()
        
        assert result.generations <= 3
        assert result.final_population is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
