# Prometheus - Self-Improving Code Evolution Engine

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Evolution-Darwinian-ff69b4.svg" alt="Evolution">
</p>

Prometheus is a **Darwinian code evolution engine** that implements natural selection principles to automatically improve code. Inspired by biological evolution, it generates code variants, evaluates their fitness, applies genetic mutations, and selects the fittest individuals for the next generation.

## 🎬 Demo
![Prometheus Demo](demo.gif)

*Self-improving code evolution engine*

## Screenshots
| Component | Preview |
|-----------|---------|
| Evolution View | ![evolution](screenshots/evolution.png) |
| Genome Browser | ![genome](screenshots/genome-browser.png) |
| Fitness Landscape | ![landscape](screenshots/fitness-landscape.png) |

## Visual Description
Evolution view shows population diversity and improvement. Genome browser displays code genes with mutation effects. Fitness landscape plots performance across solution space.

---


## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
  - [Genome](#genome)
  - [Population](#population)
  - [Generator](#generator)
  - [Mutator](#mutator)
  - [Selector](#selector)
  - [Evaluator](#evaluator)
  - [Knowledge Base](#knowledge-base)
  - [Dream Loop](#dream-loop)
- [Thinking Levels](#thinking-levels)
- [Genetic Evolution Process](#genetic-evolution-process)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Prometheus implements a complete evolutionary computation system for code improvement:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PROMETHEUS ENGINE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│   │Generator │───▶│Mutator   │───▶│Evaluator │───▶│Selector  │      │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
│        ▲                                                   │         │
│        │                                                   ▼         │
│   ┌──────────┐                                      ┌──────────┐      │
│   │Knowledge  │◀─────────────────────────────────────│Population│      │
│   │Base       │                                      └──────────┘      │
│   └──────────┘                                            │           │
│        ▲                                                  │           │
│        │                                                  ▼           │
│   ┌──────────┐                                      ┌──────────┐      │
│   │Dream Loop│                                      │  Next   │      │
│   └──────────┘                                      │Generation│      │
│                                                     └──────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Features

- **Multi-objective Optimization**: Simultaneous optimization of correctness, performance, complexity, and maintainability
- **Thinking Levels**: System-1, System-2, and System-3 cognitive processing for generation
- **Pattern Memory**: Knowledge base stores successful patterns for future guidance
- **Dream Loop**: Periodic knowledge distillation and strategic planning
- **Genetic Mutations**: Multiple mutation operators (point, structural, semantic, quality)
- **Pareto Selection**: NSGA-II/III multi-objective selection algorithms

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ARCHITECTURE LAYERS                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    APPLICATION LAYER                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │   │
│  │  │ Prometheus  │  │   Evolver    │  │  Analyzer   │        │   │
│  │  │   Engine    │  │   CLI        │  │   Tool      │        │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                │                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                      EVOLUTION LAYER                         │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │   │
│  │  │Generator │  │ Mutator  │  │ Selector │  │Evaluator │   │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                │                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                       CORE LAYER                             │   │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │   │
│  │  │ Genome   │  │Population│  │Metrics   │  │Crossover │    │   │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                │                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    COGNITION LAYER                            │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │   │
│  │  │ Thinking     │  │  Dream       │  │  Knowledge   │     │   │
│  │  │ Engine       │  │  Loop        │  │  Base        │     │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
Prometheus/
├── src/
│   ├── __init__.py              # Main package
│   ├── engine.py                # Main evolution engine
│   ├── core/                    # Core data structures
│   │   ├── __init__.py
│   │   ├── genome.py           # Genome representation
│   │   ├── evaluator.py        # Fitness evaluation
│   │   ├── population.py       # Population management
│   │   ├── metrics.py          # Metrics collection
│   │   └── crossover.py        # Genetic crossover
│   ├── evolution/              # Evolution operators
│   │   ├── __init__.py
│   │   ├── generator.py        # Code generation
│   │   ├── mutator.py          # Genetic mutations
│   │   └── selector.py         # Multi-objective selection
│   ├── thinking/               # Thinking levels
│   │   ├── __init__.py
│   │   └── levels.py           # System 1/2/3 thinking
│   ├── knowledge/              # Knowledge base
│   │   ├── __init__.py
│   │   ├── base.py             # Pattern storage
│   │   └── storage.py          # Storage backends
│   └── dream/                  # Dream loop
│       ├── __init__.py
│       └── loop.py             # Knowledge distillation
├── tests/                      # Test suite
├── examples/                  # Usage examples
├── docs/                      # Documentation
└── README.md                  # This file
```

---

## Core Components

### Genome

The `Genome` class represents an evolvable unit of code. It encapsulates:

- **Source Code**: The actual code being evolved
- **Fitness Scores**: Multi-objective fitness values
- **Mutation History**: Track genetic changes
- **Metadata**: Additional information about the genome

```python
from src.core.genome import Genome

# Create a genome
genome = Genome(
    source="def solve(n): return n * 2",
    language="python"
)

# Access properties
print(genome.id)        # Unique identifier
print(genome.hash)      # Content hash
print(genome.size)      # Source size
print(genome.line_count) # Number of lines
```

#### Fitness Scores

```python
from src.core.genome import FitnessScores

fitness = FitnessScores(
    correctness=0.9,       # Test passing rate (0-1)
    performance=0.8,      # Execution speed (0-1)
    complexity=0.3,       # Cyclomatic complexity (0-1, lower is better)
    maintainability=0.85, # Code quality (0-1)
    memory=0.2,           # Memory usage (0-1, lower is better)
    readability=0.9,       # Code readability (0-1)
)

# Calculate composite score
fitness.update_composite({
    "correctness": 0.3,
    "performance": 0.2,
    "complexity": 0.1,
    "maintainability": 0.15,
    "memory": 0.1,
    "readability": 0.15,
})
```

### Population

The `Population` class manages a collection of genomes:

```python
from src.core.population import Population, PopulationConfig

config = PopulationConfig(
    size=50,              # Population size
    elite_size=5,         # Number of elite genomes
    mutation_rate=0.1,    # Mutation probability
    crossover_rate=0.7,   # Crossover probability
    tournament_size=3,    # Tournament selection size
)

population = Population(config=config)
population.initialize(template="def solve(): pass")
```

### Generator

The `Generator` creates new code variants with thinking budgets:

```python
from src.evolution.generator import Generator, ThinkingLevel

generator = Generator()

# Generate with specific thinking level
result = generator.generate(
    context={
        "problem": "sort a list",
        "name": "sort",
        "params": "data"
    },
    parent=None  # Optional parent for evolution
)

print(result.genome.source)
print(result.thinking_trace)  # Reasoning steps
```

#### Thinking Levels

| Level | Description | Token Budget | Use Case |
|-------|-------------|--------------|----------|
| INSTINCTIVE | Fast pattern matching | ~50 | Quick drafts |
| DELIBERATE | Step-by-step analysis | ~200 | Standard generation |
| REFLECTIVE | Self-critiquing | ~400 | Quality-focused |
| META | Self-modifying | ~500 | Complex problems |

### Mutator

The `Mutator` applies genetic mutations to genomes:

```python
from src.evolution.mutator import Mutator, MutationOperator

mutator = Mutator()

# Apply random mutation
mutated = mutator.mutate(genome)

# Apply specific mutation
mutated = mutator.mutate(genome, operator=MutationOperator.REFACTOR)

# Apply multiple mutations
offspring = mutator.mutate_multiple(genome, count=3)
```

#### Mutation Operators

| Category | Operators | Description |
|----------|-----------|-------------|
| **Point** | SUBSTITUTE, INSERT, DELETE, DUPLICATE | Character/line level changes |
| **Structural** | REFACTOR, INLINE, EXTRACT | Code structure changes |
| **Optimization** | OPTIMIZE, CACHE, UNROLL, VECTORIZE | Performance improvements |
| **Semantic** | TYPE_CHANGE, LOOP_CONVERSION, RECURSION_TO_LOOP | Behavior changes |
| **Quality** | ADD_COMMENTS, ADD_TYPE_HINTS, IMPROVE_READABILITY | Code quality |

### Selector

The `Selector` implements multi-objective selection:

```python
from src.evolution.selector import Selector, SelectionMethod

selector = Selector()

# Select parents
parents = selector.select_parents(population, count=2)

# Select survivors from combined population + offspring
survivors = selector.select_survivors(population, offspring)
```

#### Selection Methods

| Method | Description | Best For |
|--------|-------------|----------|
| TOURNAMENT | Competition-based selection | General use |
| FITNESS_PROPORTIONAL | Roulette wheel | Weighted fitness |
| RANK | Rank-based selection | Fitness scaling |
| TRUNCATION | Top fraction selection | Quick convergence |
| NSGA2 | Non-dominated sorting | Multi-objective |
| NSGA3 | Reference direction-based | Many objectives |

### Evaluator

The `Evaluator` measures genome fitness:

```python
from src.core.evaluator import Evaluator, TestCase

evaluator = Evaluator()

# Define test cases
tests = [
    TestCase(
        name="basic_test",
        code="result = solve([3, 1, 2])",
        expected_output=[1, 2, 3]
    ),
]

# Evaluate genome
fitness = evaluator.evaluate(genome, test_suite=tests)
```

### Knowledge Base

The `KnowledgeBase` stores successful patterns:

```python
from src.knowledge.base import KnowledgeBase, Pattern, PatternType

kb = KnowledgeBase()

# Add pattern
pattern = Pattern(
    name="quick_sort",
    pattern_type=PatternType.ALGORITHM,
    source="def quicksort(arr): ...",
    description="Efficient sorting algorithm"
)
kb.add(pattern)

# Query patterns
results = kb.query(pattern_type=PatternType.ALGORITHM, min_effectiveness=0.7)

# Find similar patterns
similar = kb.find_similar(source_code)
```

### Dream Loop

The `DreamLoop` implements periodic knowledge distillation:

```python
from src.dream.loop import DreamLoop

dream = DreamLoop(knowledge_base=kb)

# Archive genomes during evolution
dream.archive(successful_genome)

# Check if it's time to dream
if dream.should_dream(current_generation):
    result = dream.dream()
    print(result.patterns_extracted)
    print(result.insights)
    print(result.strategies)
```

---

## Thinking Levels

Prometheus implements a multi-level cognitive architecture inspired by dual-process theory:

### System 1: Instinctive

```python
# Fast, pattern-matching generation
generator.config.thinking_level = ThinkingLevel.INSTINCTIVE

# Characteristics:
# - Low token budget (~50 tokens)
# - Pattern recognition
# - Quick responses
# - ~10ms execution time
```

### System 2: Deliberate

```python
# Slow, analytical generation
generator.config.thinking_level = ThinkingLevel.DELIBERATE

# Characteristics:
# - Medium token budget (~200 tokens)
# - Step-by-step reasoning
# - Problem decomposition
# - ~100ms execution time
```

### System 3: Metacognitive

```python
# Self-aware, self-modifying generation
generator.config.thinking_level = ThinkingLevel.METACOGNITIVE

# Characteristics:
# - High token budget (~500 tokens)
# - Strategy selection
# - Performance monitoring
# - Reflection and adaptation
```

---

## Genetic Evolution Process

The evolution process follows these steps:

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVOLUTION CYCLE                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐                                                    │
│  │ INITIAL  │ ◀──────────────────────────────────────┐          │
│  │POPULATION│                                        │          │
│  └────┬─────┘                                        │          │
│       │                                              │          │
│       ▼                                              │          │
│  ┌──────────┐                                        │          │
│  │ EVALUATE │  Run tests, benchmarks, metrics        │          │
│  └────┬─────┘                                        │          │
│       │                                              │          │
│       ▼                                              │          │
│  ┌──────────┐                                        │          │
│  │  SELECT  │  Multi-objective selection (NSGA-II)   │          │
│  └────┬─────┘                                        │          │
│       │                                              │          │
│       ▼                                              │          │
│  ┌──────────┐     ┌──────────┐                       │          │
│  │GENERATE  │────▶│ MUTATE   │  Genetic operators    │          │
│  │ OFFSPRING│     └──────────┘                       │          │
│  └────┬─────┘                                        │          │
│       │                                              │          │
│       ▼                                              │          │
│  ┌──────────┐                                        │          │
│  │  DREAM   │  Periodic knowledge distillation      │          │
│  │   LOOP   │                                        │          │
│  └────┬─────┘                                        │          │
│       │                                              │          │
│       ▼                                              │          │
│  ┌──────────┐                                        │          │
│  │TERMINATE?│  Target fitness or max generations     │          │
│  └────┬─────┘                                        │          │
│       │                                              │          │
│       │ Yes                                          │          │
│       ▼                                              │          │
│  ┌──────────┐                                        │          │
│  │  OUTPUT  │  Best solution                         │          │
│  └──────────┘                                        │          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Process

1. **Initialization**: Create initial population from seed template or random generation
2. **Evaluation**: Run tests, benchmarks, and quality metrics on each genome
3. **Selection**: Select parents using multi-objective tournament selection
4. **Reproduction**: 
   - Crossover: Combine genetic material from parents
   - Mutation: Apply genetic mutations (point, structural, semantic)
5. **Survivor Selection**: Choose next generation using NSGA-II
6. **Dream Loop** (periodic):
   - Consolidate: Extract patterns from successful genomes
   - Reflect: Analyze what worked and what didn't
   - Abstract: Generalize patterns into principles
   - Plan: Formulate future strategies
7. **Repeat**: Continue until termination criteria met

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Install from Source

```bash
# Clone the repository
git clone https://github.com/moggan1337/Prometheus.git
cd Prometheus

# Install dependencies
pip install -r requirements.txt

# Install Prometheus
pip install -e .
```

### Requirements

```
numpy>=1.21.0
pytest>=7.0.0
```

---

## Quick Start

### Basic Evolution

```python
from src.engine import Prometheus, EvolutionConfig

# Configure evolution
config = EvolutionConfig(
    population_size=50,
    max_generations=100,
    target_fitness=0.9,
)

# Create engine
engine = Prometheus(config=config)

# Run evolution
result = engine.evolve()

# Get best genome
best = result.best_genome
print(f"Best fitness: {result.best_fitness}")
print(f"Best code:\n{best.source}")
```

### With Custom Template

```python
from src.engine import Prometheus, EvolutionConfig

template = """
def solve(data):
    '''
    Implement your solution here.
    '''
    # Your starting code
    return data
"""

engine = Prometheus(
    config=EvolutionConfig(max_generations=50),
    seed_template=template
)

result = engine.evolve()
```

### Using Callbacks

```python
from src.engine import Prometheus, EvolutionConfig

engine = Prometheus()

# Set up callbacks
def on_generation(gen, pop):
    print(f"Generation {gen}: Best={pop.get_statistics().best_fitness:.4f}")

def on_best_found(genome):
    print(f"New best found! Fitness: {genome.fitness.composite:.4f}")

engine.on_generation = on_generation
engine.on_best_found = on_best_found

result = engine.evolve()
```

---

## Usage Examples

### Example 1: Sorting Algorithm Evolution

```python
from src.engine import Prometheus, EvolutionConfig
from src.core.genome import Genome, GenomeType

# Template for sorting
sorting_template = """
def sort(data):
    '''Sort a list of numbers.'''
    # Implement sorting algorithm
    return data
"""

config = EvolutionConfig(
    population_size=30,
    max_generations=50,
    thinking_level=ThinkingLevel.DELIBERATE,
)

engine = Prometheus(config=config, seed_template=sorting_template)

# Custom evaluation callback
def evaluate_sorting(genome):
    test_cases = [
        ([3, 1, 4, 1, 5], [1, 1, 3, 4, 5]),
        ([9, 8, 7, 6], [6, 7, 8, 9]),
        ([1], [1]),
    ]
    
    passed = 0
    for input_data, expected in test_cases:
        try:
            exec_globals = {}
            exec(genome.source, exec_globals)
            result = exec_globals['sort'](input_data)
            if result == expected:
                passed += 1
        except:
            pass
    
    return passed / len(test_cases)

result = engine.evolve()
print(f"Best sorting algorithm:\n{result.best_genome.source}")
```

### Example 2: Multi-Objective Optimization

```python
from src.engine import Prometheus, EvolutionConfig
from src.core.evaluator import Evaluator

# Configure for multi-objective optimization
config = EvolutionConfig(
    population_size=100,
    max_generations=200,
)

engine = Prometheus(config=config)

# Run with custom evaluator
def custom_evaluate(genome):
    evaluator = Evaluator()
    
    # Run standard evaluation
    fitness = evaluator.evaluate(genome)
    
    # Add custom objectives
    # (e.g., specific metrics for your domain)
    
    return fitness

# Evolution with custom evaluation
# (Engine uses default evaluator by default)
result = engine.evolve()

# Analyze Pareto front
pareto_genomes = get_pareto_front(result.final_population)
for g in pareto_genomes:
    print(f"Fitness: {g.fitness.composite}")
    print(f"  Correctness: {g.fitness.correctness}")
    print(f"  Performance: {g.fitness.performance}")
    print(f"  Complexity: {g.fitness.complexity}")
```

### Example 3: Knowledge-Guided Evolution

```python
from src.engine import Prometheus, EvolutionConfig
from src.knowledge.base import KnowledgeBase, Pattern, PatternType

# Create knowledge base
kb = KnowledgeBase()

# Add known good patterns
kb.add(Pattern(
    name="binary_search",
    pattern_type=PatternType.ALGORITHM,
    source="def binary_search(arr, target): ...",
    description="Efficient search algorithm"
))

# Use knowledge base in evolution
config = EvolutionConfig(
    use_knowledge_base=True,
    knowledge_base_size=1000,
)

engine = Prometheus(config=config)

# Query knowledge during evolution
def on_generation(gen, pop):
    if gen % 10 == 0:
        # Find similar patterns to best genome
        best = pop.best_genome
        similar = kb.find_similar(best.source)
        print(f"Generation {gen}: Found {len(similar)} similar patterns")

engine.on_generation = on_generation

result = engine.evolve()
```

### Example 4: Using the Dream Loop

```python
from src.engine import Prometheus, EvolutionConfig
from src.dream.loop import DreamConfig

# Configure dream loop
dream_config = DreamConfig(
    enabled=True,
    interval_generations=10,    # Run every 10 generations
    consolidation_threshold=0.7, # Min fitness to extract pattern
    min_patterns_for_abstraction=5,
)

config = EvolutionConfig(
    enable_dream_loop=True,
    dream_interval=10,
)

engine = Prometheus(config=config)

# Set up dream callback
def on_dream(dream_result):
    print(f"Dream complete!")
    print(f"  Patterns extracted: {dream_result.patterns_extracted}")
    print(f"  Insights: {dream_result.insights}")
    print(f"  Strategies: {dream_result.strategies}")

engine.on_dream = on_dream

result = engine.evolve()

# Get recommended strategies
strategies = engine.dream_loop.get_recommended_strategies()
for strategy in strategies:
    print(f"Strategy: {strategy.name}")
    print(f"  Expected improvement: {strategy.expected_improvement:.2%}")
```

---

## Configuration

### EvolutionConfig

```python
@dataclass
class EvolutionConfig:
    # Population parameters
    population_size: int = 50          # Number of genomes
    elite_size: int = 5                # Best genomes to preserve
    
    # Evolution parameters
    max_generations: int = 100         # Stop condition
    mutation_rate: float = 0.1         # Mutation probability
    crossover_rate: float = 0.7        # Crossover probability
    
    # Thinking parameters
    thinking_level: ThinkingLevel = ThinkingLevel.DELIBERATE
    thinking_budget: int = 500
    
    # Termination
    target_fitness: float = 0.95      # Stop when reached
    stagnation_limit: int = 20         # Generations without improvement
    
    # Output
    log_interval: int = 5              # Print progress every N gens
    save_best: bool = True
    checkpoint_dir: Optional[Path] = None
    
    # Knowledge integration
    use_knowledge_base: bool = True
    knowledge_base_size: int = 1000
    
    # Dream loop
    enable_dream_loop: bool = True
    dream_interval: int = 10
```

### PopulationConfig

```python
@dataclass
class PopulationConfig:
    size: int = 50
    elite_size: int = 5
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    tournament_size: int = 3
    max_stagnation: int = 20
    diversity_threshold: float = 0.1
```

---

## API Reference

### Core Classes

| Class | Description |
|-------|-------------|
| `Genome` | Represents an evolvable unit of code |
| `Population` | Manages a population of genomes |
| `FitnessScores` | Multi-objective fitness values |
| `PopulationStats` | Population statistics |

### Evolution Classes

| Class | Description |
|-------|-------------|
| `Generator` | Creates code variants |
| `Mutator` | Applies genetic mutations |
| `Selector` | Multi-objective selection |
| `Evaluator` | Measures fitness |

### Cognition Classes

| Class | Description |
|-------|-------------|
| `ThinkingEngine` | Multi-level cognitive processing |
| `DreamLoop` | Knowledge distillation |
| `KnowledgeBase` | Pattern storage and retrieval |

---

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_genome.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Project Structure

```
Prometheus/
├── src/                    # Source code
│   ├── core/               # Core data structures
│   ├── evolution/           # Evolution operators
│   ├── thinking/           # Thinking levels
│   ├── knowledge/          # Knowledge base
│   └── dream/              # Dream loop
├── tests/                  # Test suite
├── examples/               # Usage examples
├── docs/                   # Documentation
└── README.md
```

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Submit a pull request

---

## License

MIT License - see LICENSE file for details.

---

<p align="center">
  Built with 🧬 by Prometheus
</p>
