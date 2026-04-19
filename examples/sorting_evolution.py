"""Example: Sorting Algorithm Evolution"""

from src.engine import Prometheus, EvolutionConfig
from src.core.genome import Genome
from src.core.evaluator import Evaluator


def evaluate_sorting(genome, evaluator):
    """Custom evaluation for sorting algorithms."""
    test_cases = [
        ([3, 1, 4, 1, 5], [1, 1, 3, 4, 5]),
        ([9, 8, 7, 6, 5, 4], [4, 5, 6, 7, 8, 9]),
        ([1], [1]),
        ([2, 1], [1, 2]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
    ]
    
    passed = 0
    for input_data, expected in test_cases:
        try:
            # Execute the sorting function
            exec_globals = {"__name__": "__test__"}
            exec(compile(genome.source, "<test>", "exec"), exec_globals)
            
            if 'sort' in exec_globals:
                result = exec_globals['sort'](input_data.copy())
                if result == expected:
                    passed += 1
        except Exception:
            pass
    
    # Update fitness
    genome.fitness.correctness = passed / len(test_cases)
    genome.fitness = evaluator.evaluate(genome)
    
    return genome.fitness


def main():
    print("=" * 60)
    print("Sorting Algorithm Evolution")
    print("=" * 60)
    
    # Template for sorting
    template = '''
def sort(data):
    """Sort a list in ascending order."""
    # Implement your sorting algorithm here
    return sorted(data)
'''
    
    config = EvolutionConfig(
        population_size=30,
        max_generations=20,
        target_fitness=0.9,
        enable_dream_loop=True,
        dream_interval=10,
    )
    
    engine = Prometheus(config=config, seed_template=template)
    
    # Override evaluation
    evaluator = Evaluator()
    
    def custom_evaluate(genome):
        return evaluate_sorting(genome, evaluator)
    
    # Set up callbacks
    def on_generation(gen, pop):
        stats = pop.get_statistics()
        print(f"Gen {gen}: Best={stats.best_fitness:.4f}, "
              f"Passed={int(stats.best_fitness * 5)}/5 tests")
    
    engine.on_generation = on_generation
    
    print("\nEvolving sorting algorithms...\n")
    result = engine.evolve()
    
    print("\n" + "=" * 60)
    print("Evolution Complete!")
    print("=" * 60)
    
    if result.best_genome:
        print(f"\nBest Sorting Algorithm:")
        print("-" * 40)
        print(result.best_genome.source)
        print(f"\nTest Cases Passed: {int(result.best_genome.fitness.correctness * 5)}/5")


if __name__ == "__main__":
    main()
