"""Example: Basic Evolution"""

from src.engine import Prometheus, EvolutionConfig
from src.evolution.generator import ThinkingLevel


def main():
    print("=" * 60)
    print("Prometheus Basic Evolution Example")
    print("=" * 60)
    
    # Configure evolution
    config = EvolutionConfig(
        population_size=20,
        max_generations=10,
        target_fitness=0.8,
        thinking_level=ThinkingLevel.DELIBERATE,
        enable_dream_loop=True,
        dream_interval=5,
    )
    
    # Create engine
    engine = Prometheus(config=config)
    
    # Set up callbacks
    def on_generation(gen, pop):
        stats = pop.get_statistics()
        print(f"Gen {gen}: Best={stats.best_fitness:.4f}, "
              f"Avg={stats.avg_fitness:.4f}, "
              f"Div={stats.diversity:.4f}")
    
    def on_best_found(genome):
        print(f"\n🎉 New best! Fitness: {genome.fitness.composite:.4f}")
    
    def on_dream(dream_result):
        print(f"\n💭 Dream complete:")
        print(f"   Patterns: {dream_result.patterns_extracted}")
        print(f"   Insights: {len(dream_result.insights)}")
        print(f"   Strategies: {len(dream_result.strategies)}")
    
    engine.on_generation = on_generation
    engine.on_best_found = on_best_found
    engine.on_dream = on_dream
    
    # Run evolution
    print("\nStarting evolution...\n")
    result = engine.evolve()
    
    # Report results
    print("\n" + "=" * 60)
    print("Evolution Complete!")
    print("=" * 60)
    print(f"Success: {result.success}")
    print(f"Generations: {result.generations}")
    print(f"Duration: {result.duration_seconds:.2f}s")
    print(f"Best Fitness: {result.best_fitness:.4f}")
    
    if result.best_genome:
        print(f"\nBest Code:")
        print("-" * 40)
        print(result.best_genome.source)
    
    # Knowledge base stats
    if engine.knowledge_base:
        kb_stats = engine.knowledge_base.get_statistics()
        print(f"\nKnowledge Base:")
        print(f"  Total patterns: {kb_stats['total_patterns']}")
        print(f"  Avg effectiveness: {kb_stats['avg_effectiveness']:.4f}")


if __name__ == "__main__":
    main()
