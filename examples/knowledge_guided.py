"""Example: Knowledge-Guided Evolution"""

from src.engine import Prometheus, EvolutionConfig
from src.knowledge.base import KnowledgeBase, Pattern, PatternType


def main():
    print("=" * 60)
    print("Knowledge-Guided Evolution Example")
    print("=" * 60)
    
    # Create and populate knowledge base
    kb = KnowledgeBase()
    
    # Add known patterns
    patterns = [
        Pattern(
            id="fibonacci_1",
            name="fibonacci_recursive",
            pattern_type=PatternType.ALGORITHM,
            source='''
def fibonacci(n):
    """Calculate nth Fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
''',
            description="Basic recursive Fibonacci",
            tags=["recursion", "fibonacci", "sequence"]
        ),
        Pattern(
            id="fibonacci_2",
            name="fibonacci_memoized",
            pattern_type=PatternType.OPTIMIZATION,
            source='''
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci(n):
    """Calculate nth Fibonacci number with memoization."""
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
''',
            description="Memoized Fibonacci for performance",
            tags=["recursion", "fibonacci", "memoization", "optimization"]
        ),
        Pattern(
            id="binary_search",
            name="binary_search",
            pattern_type=PatternType.ALGORITHM,
            source='''
def binary_search(arr, target):
    """Search for target in sorted array."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
''',
            description="Efficient binary search",
            tags=["search", "binary", "sorted", "algorithm"]
        ),
    ]
    
    for p in patterns:
        kb.add(p)
        p.use(fitness=0.8)  # Simulate usage with good fitness
    
    print(f"\nKnowledge base initialized with {len(kb)} patterns")
    print(f"Patterns: {[p.name for p in kb]}")
    
    # Configure evolution
    config = EvolutionConfig(
        population_size=25,
        max_generations=15,
        use_knowledge_base=True,
        knowledge_base_size=1000,
    )
    
    engine = Prometheus(config=config)
    
    # Use knowledge base during evolution
    def on_generation(gen, pop):
        if gen % 5 == 0:
            best = pop.best_genome
            
            # Query knowledge base
            similar = kb.find_similar(best.source) if best else []
            
            print(f"Gen {gen}: Best={pop.get_statistics().best_fitness:.4f}")
            print(f"  Similar patterns: {len(similar)}")
            
            # Query by type
            algorithms = kb.query(pattern_type=PatternType.ALGORITHM)
            print(f"  Algorithms in KB: {len(algorithms)}")
    
    engine.on_generation = on_generation
    
    print("\nStarting knowledge-guided evolution...\n")
    result = engine.evolve()
    
    print("\n" + "=" * 60)
    print("Evolution Complete!")
    print("=" * 60)
    
    # Final knowledge base stats
    final_stats = engine.get_knowledge_stats()
    if final_stats:
        print(f"\nFinal Knowledge Base:")
        print(f"  Total patterns: {final_stats['total_patterns']}")
        print(f"  Avg effectiveness: {final_stats['avg_effectiveness']:.4f}")


if __name__ == "__main__":
    main()
