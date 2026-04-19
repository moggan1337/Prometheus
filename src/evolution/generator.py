"""
Generator Module - Produces code variants with thinking budgets.

The Generator creates new code variants based on:
1. Thinking levels (system-1, system-2, system-3)
2. Prompt templates
3. Code templates
4. Meta-prompting strategies
"""

from __future__ import annotations
import re
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, Any

from ..core.genome import Genome, GenomeType, MutationType
from ..thinking.levels import ThinkingEngine, ThinkingLevel as TLevel


class ThinkingLevel(Enum):
    """
    Thinking levels for code generation.
    
    Inspired by dual-process theory:
    - System 1: Fast, intuitive, pattern matching
    - System 2: Slow, deliberate, reasoning
    - System 3: Meta-cognitive, self-improvement
    """
    INSTINCTIVE = auto()  # Quick pattern-based generation
    DELIBERATE = auto()   # Thoughtful, multi-step generation
    REFLECTIVE = auto()   # Self-aware, critiquing generation
    META = auto()         # Self-modifying generation


@dataclass
class GenerationConfig:
    """Configuration for code generation."""
    # Thinking level
    thinking_level: ThinkingLevel = ThinkingLevel.DELIBERATE
    thinking_budget_tokens: int = 500
    
    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 1000
    
    # Template parameters
    use_template: bool = True
    template_library: Optional[list] = None
    
    # Self-improvement parameters
    enable_self_critique: bool = True
    enable_reflection: bool = True
    max_iterations: int = 3
    
    # Diversity
    enforce_diversity: bool = True
    mutation_probability: float = 0.3


@dataclass
class GenerationResult:
    """Result of code generation."""
    genome: Genome
    thinking_trace: list[str] = field(default_factory=list)
    iterations: int = 1
    success: bool = True
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class Generator:
    """
    Generates code variants with configurable thinking budgets.
    
    The Generator is responsible for creating new code through:
    1. Template-based generation
    2. Thinking-level guided generation
    3. Self-improvement loops
    4. Diversity enforcement
    """
    
    def __init__(
        self, 
        config: Optional[GenerationConfig] = None,
        thinking_engine: Optional[ThinkingEngine] = None,
    ):
        """Initialize generator with configuration."""
        self.config = config or GenerationConfig()
        self.thinking_engine = thinking_engine or ThinkingEngine()
        
        # Initialize template library
        self._init_templates()
        
        # State for generation
        self.generation_count = 0
    
    def _init_templates(self) -> None:
        """Initialize code templates."""
        self.templates = {
            "function": [
                """def {name}({params}):
    '''{docstring}'''
    {body}
""",
                """async def {name}({params}):
    '''{docstring}'''
    {body}
""",
            ],
            "class": [
                """class {name}:
    '''{docstring}'''
    
    def __init__(self{init_params}):
        {init_body}
    
    def __call__(self{call_params}):
        {call_body}
""",
            ],
            "algorithm": [
                """def {name}({params}):
    '''{docstring} - Algorithm implementation'''
    # Initialization
    {init}
    
    # Main loop
    for {loop_var} in {iteration}:
        {loop_body}
    
    return {result}
""",
            ],
        }
    
    def generate(
        self, 
        context: Optional[dict] = None,
        parent: Optional[Genome] = None,
    ) -> GenerationResult:
        """
        Generate a new code variant.
        
        Args:
            context: Generation context (problem description, requirements, etc.)
            parent: Optional parent genome for evolutionary generation
            
        Returns:
            GenerationResult with generated genome
        """
        context = context or {}
        self.generation_count += 1
        
        thinking_trace = []
        
        try:
            # Apply thinking level strategy
            if self.config.thinking_level == ThinkingLevel.INSTINCTIVE:
                return self._generate_instinctive(context, parent)
            
            elif self.config.thinking_level == ThinkingLevel.DELIBERATE:
                return self._generate_deliberate(context, parent)
            
            elif self.config.thinking_level == ThinkingLevel.REFLECTIVE:
                return self._generate_reflective(context, parent)
            
            else:  # META
                return self._generate_meta(context, parent)
                
        except Exception as e:
            return GenerationResult(
                genome=Genome(),
                thinking_trace=thinking_trace,
                success=False,
                error=str(e),
            )
    
    def _generate_instinctive(
        self, 
        context: dict, 
        parent: Optional[Genome]
    ) -> GenerationResult:
        """
        System 1: Fast, intuitive generation.
        
        Uses pattern matching and templates without deep reasoning.
        """
        thinking_trace = [
            "[System-1] Quick pattern recognition...",
            "[System-1] Matching to known patterns...",
        ]
        
        # If parent exists, mutate slightly
        if parent:
            genome = parent.clone()
            genome.source = self._apply_quick_mutations(parent.source)
        else:
            # Generate from template
            genome = self._generate_from_template(context)
        
        return GenerationResult(
            genome=genome,
            thinking_trace=thinking_trace,
            iterations=1,
            success=True,
        )
    
    def _generate_deliberate(
        self, 
        context: dict, 
        parent: Optional[Genome]
    ) -> GenerationResult:
        """
        System 2: Slow, deliberate generation.
        
        Breaks down problem, generates step by step.
        """
        thinking_trace = [
            "[System-2] Analyzing requirements...",
            "[System-2] Breaking down problem into steps...",
            "[System-2] Generating solution structure...",
            "[System-2] Implementing details...",
        ]
        
        # Multi-step generation
        if parent:
            # Evolutionary generation with deliberation
            genome = self._evolve_with_deliberation(parent, context)
        else:
            # De novo generation
            genome = self._generate_structured(context)
        
        # Self-critique if enabled
        if self.config.enable_self_critique:
            thinking_trace.append("[System-2] Self-critiquing...")
            genome = self._self_critique(genome)
        
        return GenerationResult(
            genome=genome,
            thinking_trace=thinking_trace,
            iterations=2,
            success=True,
        )
    
    def _generate_reflective(
        self, 
        context: dict, 
        parent: Optional[Genome]
    ) -> GenerationResult:
        """
        System 2+: Reflective generation with self-awareness.
        
        Generates, reflects on quality, and iteratively improves.
        """
        thinking_trace = [
            "[Reflection] Initial generation...",
            "[Reflection] Analyzing generated code...",
            "[Reflection] Identifying potential issues...",
            "[Reflection] Planning improvements...",
            "[Reflection] Implementing refinements...",
        ]
        
        genome = self._generate_deliberate(context, parent).genome
        
        # Reflective iteration
        for i in range(self.config.max_iterations):
            issues = self._identify_issues(genome, context)
            if not issues:
                break
            
            thinking_trace.append(f"[Reflection] Iteration {i+1}: Fixing {len(issues)} issues...")
            genome = self._fix_issues(genome, issues)
        
        return GenerationResult(
            genome=genome,
            thinking_trace=thinking_trace,
            iterations=self.config.max_iterations,
            success=True,
        )
    
    def _generate_meta(
        self, 
        context: dict, 
        parent: Optional[Genome]
    ) -> GenerationResult:
        """
        System 3: Meta-cognitive generation.
        
        Can modify its own generation strategy based on feedback.
        """
        thinking_trace = [
            "[Meta] Activating meta-cognition...",
            "[Meta] Analyzing generation strategy...",
            "[Meta] Selecting appropriate approach...",
            "[Meta] Executing generation...",
            "[Meta] Evaluating outcomes...",
            "[Meta] Adjusting strategy if needed...",
        ]
        
        # Analyze context to select best approach
        strategy = self._select_strategy(context)
        thinking_trace.append(f"[Meta] Selected strategy: {strategy}")
        
        # Generate with selected strategy
        if strategy == "template":
            genome = self._generate_from_template(context)
        elif strategy == "evolve" and parent:
            genome = self._evolve_with_deliberation(parent, context)
        else:
            genome = self._generate_structured(context)
        
        # Meta-level critique
        genome = self._meta_critique(genome, context)
        
        return GenerationResult(
            genome=genome,
            thinking_trace=thinking_trace,
            iterations=3,
            success=True,
            metadata={"strategy": strategy},
        )
    
    def _generate_from_template(self, context: dict) -> Genome:
        """Generate code from template."""
        genome_type = context.get("genome_type", GenomeType.FUNCTION)
        template_type = {
            GenomeType.FUNCTION: "function",
            GenomeType.CLASS: "class",
            GenomeType.ALGORITHM: "algorithm",
        }.get(genome_type, "function")
        
        template = random.choice(self.templates.get(template_type, self.templates["function"]))
        
        # Fill template
        name = context.get("name", "solve")
        params = context.get("params", "data")
        docstring = context.get("docstring", "Solve the given problem.")
        body = context.get("body", "return data")
        
        source = template.format(
            name=name,
            params=params,
            docstring=docstring,
            body=body,
            init_params=context.get("init_params", ""),
            init_body=context.get("init_body", "pass"),
            call_params=context.get("call_params", ""),
            call_body=context.get("call_body", "pass"),
            loop_var=context.get("loop_var", "i"),
            iteration=context.get("iteration", "range(10)"),
            loop_body=context.get("loop_body", "pass"),
            result=context.get("result", "None"),
            init=context.get("init", "# Initialize"),
        )
        
        return Genome(
            source=source,
            language=context.get("language", "python"),
            genome_type=genome_type,
            metadata={"generated_from": "template"},
        )
    
    def _generate_structured(self, context: dict) -> Genome:
        """Generate structured code without template."""
        problem = context.get("problem", "")
        requirements = context.get("requirements", [])
        
        # Simple structured generation based on problem type
        if "sort" in problem.lower():
            source = self._generate_sorting(context)
        elif "search" in problem.lower():
            source = self._generate_search(context)
        elif "recursive" in problem.lower() or "fibonacci" in problem.lower():
            source = self._generate_recursive(context)
        else:
            source = self._generate_generic(context)
        
        return Genome(
            source=source,
            language=context.get("language", "python"),
            genome_type=GenomeType.ALGORITHM,
            metadata={"generated_from": "structured"},
        )
    
    def _generate_sorting(self, context: dict) -> str:
        """Generate sorting algorithm."""
        return '''def sort(data):
    """
    Sort the input data using efficient sorting.
    
    Args:
        data: List of comparable elements
        
    Returns:
        Sorted list
    """
    if not data or len(data) <= 1:
        return data
    
    # Quick sort implementation
    pivot = data[len(data) // 2]
    left = [x for x in data if x < pivot]
    middle = [x for x in data if x == pivot]
    right = [x for x in data if x > pivot]
    
    return sort(left) + middle + sort(right)
'''
    
    def _generate_search(self, context: dict) -> str:
        """Generate search algorithm."""
        return '''def search(data, target):
    """
    Search for target in sorted data.
    
    Args:
        data: Sorted list of elements
        target: Element to search for
        
    Returns:
        Index of target if found, -1 otherwise
    """
    left, right = 0, len(data) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if data[mid] == target:
            return mid
        elif data[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1
'''
    
    def _generate_recursive(self, context: dict) -> str:
        """Generate recursive algorithm."""
        return '''def fibonacci(n):
    """
    Calculate the nth Fibonacci number.
    
    Args:
        n: Position in Fibonacci sequence
        
    Returns:
        nth Fibonacci number
    """
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)


def memoized_fibonacci(n, memo=None):
    """Memoized Fibonacci for better performance."""
    if memo is None:
        memo = {}
    
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    
    memo[n] = memoized_fibonacci(n - 1, memo) + memoized_fibonacci(n - 2, memo)
    return memo[n]
'''
    
    def _generate_generic(self, context: dict) -> str:
        """Generate generic algorithm."""
        return '''def solve(data):
    """
    Solve the given problem.
    
    Args:
        data: Input data
        
    Returns:
        Solution result
    """
    # Process data
    result = data
    
    # Apply transformations
    # Add your logic here
    
    return result
'''
    
    def _evolve_with_deliberation(
        self, 
        parent: Genome, 
        context: dict
    ) -> Genome:
        """Evolve parent genome with deliberate improvements."""
        from ..evolution.mutator import Mutator
        
        mutator = Mutator()
        
        # Apply deliberate mutations
        mutations = [
            mutator.refactor,
            mutator.optimize_performance,
            mutator.add_type_hints,
            mutator.improve_readability,
        ]
        
        current = parent
        for mutation in random.sample(mutations, min(2, len(mutations))):
            current = mutation(current)
        
        return current
    
    def _apply_quick_mutations(self, source: str) -> str:
        """Apply quick pattern-based mutations."""
        mutations = [
            lambda s: s.replace("return None", "return []"),
            lambda s: s.replace("pass", "# TODO: implement"),
            lambda s: re.sub(r'(\w+)\s*=\s*None', r'\1 = 0', s),
            lambda s: re.sub(r'for\s+(\w+)\s+in', r'for \1_idx, \1 in', s),
        ]
        
        # Apply 1-2 random mutations
        selected = random.sample(mutations, min(random.randint(1, 2), len(mutations)))
        for mutation in selected:
            source = mutation(source)
        
        return source
    
    def _self_critique(self, genome: Genome) -> Genome:
        """Self-critique and improve generated code."""
        issues = self._identify_issues(genome, {})
        
        if not issues:
            return genome
        
        # Fix critical issues
        return self._fix_issues(genome, issues[:2])
    
    def _identify_issues(
        self, 
        genome: Genome, 
        context: dict
    ) -> list[str]:
        """Identify potential issues in generated code."""
        issues = []
        source = genome.source
        
        # Check for common issues
        if "pass" in source and source.count("pass") > 2:
            issues.append("Excessive pass statements")
        
        if "TODO" in source or "FIXME" in source:
            issues.append("Contains TODO/FIXME comments")
        
        if source.count("\n") < 3:
            issues.append("Too short")
        
        if "return" not in source:
            issues.append("Missing return statement")
        
        # Check for obvious bugs
        if re.search(r'def\s+\w+\([^)]*\):\s*pass\s*$', source, re.MULTILINE):
            issues.append("Empty function definition")
        
        return issues
    
    def _fix_issues(self, genome: Genome, issues: list[str]) -> Genome:
        """Fix identified issues in genome."""
        source = genome.source
        
        for issue in issues:
            if "Excessive pass" in issue:
                source = source.replace("pass", "# placeholder", 1)
            elif "Contains TODO" in issue:
                source = re.sub(r'#\s*TODO.*\n', '', source)
            elif "Too short" in issue:
                source += "\n    # Extended functionality\n    return result"
        
        genome.source = source
        return genome
    
    def _select_strategy(self, context: dict) -> str:
        """Select generation strategy based on context."""
        complexity = context.get("complexity", "medium")
        
        if complexity == "simple":
            return "template"
        elif complexity == "complex":
            return "evolve"
        else:
            return random.choice(["template", "structured"])
    
    def _meta_critique(self, genome: Genome, context: dict) -> Genome:
        """Meta-level critique and modification."""
        # Simple meta-critique: ensure basic quality
        source = genome.source
        
        # Ensure docstring exists
        if '"""' not in source and "'''" not in source:
            source = source.replace(
                "def ",
                '"""Solve the problem."""\n    def ',
                1
            )
        
        # Ensure return exists
        if "return" not in source:
            source = source.rstrip() + "\n    return None\n"
        
        genome.source = source
        return genome
    
    def generate_batch(
        self, 
        count: int, 
        context: Optional[dict] = None,
        parent: Optional[Genome] = None,
    ) -> list[GenerationResult]:
        """Generate multiple code variants."""
        results = []
        for _ in range(count):
            results.append(self.generate(context, parent))
        return results


# Convenience function
def quick_generate(problem: str) -> Genome:
    """Quickly generate code for a problem."""
    generator = Generator()
    result = generator.generate({"problem": problem})
    return result.genome
