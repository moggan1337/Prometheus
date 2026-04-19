"""
Mutator Module - Genetic mutations for code evolution.

Provides various mutation operators:
- Point mutations
- Structural mutations
- Refactoring mutations
- Optimization mutations
- Semantic mutations
"""

from __future__ import annotations
import re
import ast
import random
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable, Any
from collections import defaultdict

from ..core.genome import Genome, MutationType, GenomeType


class MutationOperator(Enum):
    """Types of mutation operators."""
    # Point mutations
    SUBSTITUTE = auto()       # Replace characters/patterns
    INSERT = auto()           # Insert new code
    DELETE = auto()           # Delete code segments
    DUPLICATE = auto()        # Duplicate code segments
    
    # Structural mutations
    REFACTOR = auto()         # Refactoring changes
    INLINE = auto()           # Inline functions/variables
    EXTRACT = auto()          # Extract to function/variable
    
    # Optimization mutations
    OPTIMIZE = auto()         # Performance optimization
    CACHE = auto()            # Add caching
    UNROLL = auto()           # Loop unrolling
    VECTORIZE = auto()        # Vectorize operations
    
    # Semantic mutations
    TYPE_CHANGE = auto()      # Change data types
    LOOP_CONVERSION = auto()  # Convert between loop types
    RECURSION_TO_LOOP = auto() # Convert recursion to iteration
    LOOP_TO_RECURSION = auto() # Convert iteration to recursion
    
    # Quality mutations
    ADD_COMMENTS = auto()     # Add documentation
    ADD_TYPE_HINTS = auto()   # Add type annotations
    IMPROVE_READABILITY = auto() # Improve code style
    REMOVE_DEAD_CODE = auto() # Remove unused code


@dataclass
class MutationResult:
    """Result of a mutation operation."""
    success: bool
    mutated_genome: Optional[Genome] = None
    mutation_type: MutationType = MutationType.POINT
    details: dict = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class Mutator:
    """
    Applies genetic mutations to genomes.
    
    The Mutator provides various mutation strategies:
    - Point mutations (character/line level)
    - Structural mutations (AST level)
    - Semantic mutations (behavior changing)
    - Quality mutations (improvement focused)
    """
    
    def __init__(self, config: Optional[dict] = None):
        """Initialize mutator with configuration."""
        self.config = config or self._default_config()
        self.mutation_history: list[MutationResult] = []
    
    def _default_config(self) -> dict:
        """Get default mutator configuration."""
        return {
            "point_mutation_rate": 0.4,
            "block_mutation_rate": 0.3,
            "refactor_rate": 0.2,
            "optimize_rate": 0.1,
            "max_mutations_per_genome": 5,
            "preserve_syntax": True,
            "mutation_strength": 0.3,  # 0-1, how aggressive
        }
    
    def mutate(
        self, 
        genome: Genome, 
        operator: Optional[MutationOperator] = None
    ) -> Genome:
        """
        Apply mutation to a genome.
        
        Args:
            genome: Genome to mutate
            operator: Specific mutation operator, or None for random
            
        Returns:
            Mutated genome
        """
        # Select operator
        if operator is None:
            operator = self._select_operator()
        
        # Apply mutation based on operator type
        mutation_methods = {
            MutationOperator.SUBSTITUTE: self._substitute_mutation,
            MutationOperator.INSERT: self._insert_mutation,
            MutationOperator.DELETE: self._delete_mutation,
            MutationOperator.DUPLICATE: self._duplicate_mutation,
            MutationOperator.REFACTOR: self.refactor,
            MutationOperator.INLINE: self._inline_mutation,
            MutationOperator.EXTRACT: self._extract_mutation,
            MutationOperator.OPTIMIZE: self.optimize_performance,
            MutationOperator.CACHE: self._add_cache_mutation,
            MutationOperator.TYPE_CHANGE: self._type_change_mutation,
            MutationOperator.ADD_COMMENTS: self.add_comments,
            MutationOperator.ADD_TYPE_HINTS: self.add_type_hints,
            MutationOperator.IMPROVE_READABILITY: self.improve_readability,
            MutationOperator.REMOVE_DEAD_CODE: self.remove_dead_code,
            MutationOperator.RECURSION_TO_LOOP: self._recursion_to_loop,
            MutationOperator.LOOP_TO_RECURSION: self._loop_to_recursion,
        }
        
        mutation_func = mutation_methods.get(operator, self._substitute_mutation)
        
        try:
            mutated = mutation_func(genome)
            mutated.generation = genome.generation + 1
            mutated.mutation_history.parent_id = genome.id
            
            # Record mutation
            result = MutationResult(
                success=True,
                mutated_genome=mutated,
                mutation_type=self._operator_to_mutation_type(operator),
                details={"operator": operator.name},
            )
            self.mutation_history.append(result)
            
            return mutated
            
        except Exception as e:
            # Return clone on failure
            return genome.clone()
    
    def _select_operator(self) -> MutationOperator:
        """Select mutation operator based on configuration."""
        rates = {
            MutationOperator.SUBSTITUTE: self.config["point_mutation_rate"],
            MutationOperator.INSERT: self.config["block_mutation_rate"] / 2,
            MutationOperator.DELETE: self.config["block_mutation_rate"] / 2,
            MutationOperator.REFACTOR: self.config["refactor_rate"],
            MutationOperator.OPTIMIZE: self.config["optimize_rate"],
            MutationOperator.ADD_TYPE_HINTS: 0.05,
            MutationOperator.IMPROVE_READABILITY: 0.05,
        }
        
        # Normalize and select
        total = sum(rates.values())
        probs = {op: rate / total for op, rate in rates.items()}
        
        return random.choices(
            list(probs.keys()),
            weights=list(probs.values()),
            k=1
        )[0]
    
    def _operator_to_mutation_type(self, operator: MutationOperator) -> MutationType:
        """Map operator to mutation type."""
        mapping = {
            MutationOperator.SUBSTITUTE: MutationType.POINT,
            MutationOperator.INSERT: MutationType.INSERT,
            MutationOperator.DELETE: MutationType.DELETE,
            MutationOperator.DUPLICATE: MutationType.DUPLICATE,
            MutationOperator.REFACTOR: MutationType.REFACTOR,
            MutationOperator.OPTIMIZE: MutationType.OPTIMIZE,
        }
        return mapping.get(operator, MutationType.POINT)
    
    # Point Mutations
    
    def _substitute_mutation(self, genome: Genome) -> Genome:
        """Substitute characters or patterns."""
        source = genome.source
        
        # Common substitutions
        substitutions = [
            # Boolean swaps
            (r'\bTrue\b', 'False'),
            (r'\bFalse\b', 'True'),
            # None swaps
            (r'\bNone\b', '[]'),
            (r'\[\]', 'None'),
            # Comparison flips
            (r'>=', '>'),
            (r'<=', '<'),
            (r'>', '>='),
            (r'<', '<='),
            # Operator swaps
            (r'\*\s*\*', '*'),
            (r'\+\s*\+', '+'),
            # Keyword swaps
            (r'\breturn\b', 'pass\n    return'),
            # Loop swaps
            (r'while\b', 'for _ in'),
            # List comprehension to loop
            (r'\[.*for.*in.*\]', '# List comprehension'),
        ]
        
        # Select and apply substitution
        sub = random.choice(substitutions)
        if re.search(sub[0], source):
            new_source = re.sub(sub[0], sub[1], source, count=1)
            
            mutated = genome.clone()
            mutated.source = new_source
            return mutated
        
        return genome.clone()
    
    def _insert_mutation(self, genome: Genome) -> Genome:
        """Insert new code segments."""
        source = genome.source
        
        # Possible insertions
        insertions = [
            ("# Optimization: ", 0),
            ("\n    result = None\n", source.rfind(":")),
            ("\n    # Handle edge cases\n", source.rfind(":")),
            ("    try:\n", source.rfind(":")),
            ("    # Cache for performance\n", source.rfind(":")),
        ]
        
        pos, content = random.choice(insertions)
        
        new_source = source[:pos] + content + source[pos:]
        
        mutated = genome.clone()
        mutated.source = new_source
        return mutated
    
    def _delete_mutation(self, genome: Genome) -> Genome:
        """Delete code segments."""
        source = genome.source
        lines = source.split('\n')
        
        if len(lines) <= 2:
            return genome.clone()
        
        # Find deletable lines
        deletable = [i for i, line in enumerate(lines) 
                     if line.strip() and not line.strip().startswith('#')
                     and 'def ' not in line and 'class ' not in line
                     and 'return' not in line]
        
        if deletable:
            idx = random.choice(deletable)
            del lines[idx]
            
            mutated = genome.clone()
            mutated.source = '\n'.join(lines)
            return mutated
        
        return genome.clone()
    
    def _duplicate_mutation(self, genome: Genome) -> Genome:
        """Duplicate code segments."""
        source = genome.source
        lines = source.split('\n')
        
        if len(lines) < 3:
            return genome.clone()
        
        # Find duplicable blocks
        blocks = []
        current_block = []
        
        for i, line in enumerate(lines):
            current_block.append(line)
            if line.strip() and not line.strip().startswith('#'):
                if len(current_block) >= 2:
                    blocks.append((i - len(current_block) + 1, list(current_block)))
                current_block = []
        
        if blocks:
            idx, block = random.choice(blocks)
            lines.insert(idx + len(block), *block)
            
            mutated = genome.clone()
            mutated.source = '\n'.join(lines)
            return mutated
        
        return genome.clone()
    
    # Structural Mutations
    
    def refactor(self, genome: Genome) -> Genome:
        """Apply refactoring mutations."""
        refactorings = [
            self._rename_variables,
            self._simplify_conditionals,
            self._consolidate_loops,
            self._extract_method,
        ]
        
        refactor = random.choice(refactorings)
        return refactor(genome)
    
    def _rename_variables(self, genome: Genome) -> Genome:
        """Rename variables to more meaningful names."""
        source = genome.source
        
        # Find single-letter variables
        single_letter = re.findall(r'\b([a-z])\b(?!\s*=)', source)
        
        if single_letter:
            letter = random.choice(single_letter)
            names = {'i': 'index', 'j': 'counter', 'k': 'iterator', 
                    'n': 'count', 'x': 'value', 'y': 'other'}
            new_name = names.get(letter, 'var')
            
            # Replace in non-keyword contexts
            pattern = rf'\b{letter}\b(?!\s*[=+])'
            new_source = re.sub(pattern, new_name, source, count=1)
            
            mutated = genome.clone()
            mutated.source = new_source
            return mutated
        
        return genome.clone()
    
    def _simplify_conditionals(self, genome: Genome) -> Genome:
        """Simplify conditional expressions."""
        source = genome.source
        
        # Simplify patterns
        simplifications = [
            (r'if\s+x\s+==\s+True:', 'if x:'),
            (r'if\s+x\s+==\s+False:', 'if not x:'),
            (r'if\s+\(x\s*<', 'if x <'),
            (r'\(x\s+and\s+y\)', 'x and y'),
        ]
        
        for pattern, replacement in simplifications:
            if re.search(pattern, source):
                new_source = re.sub(pattern, replacement, source, count=1)
                mutated = genome.clone()
                mutated.source = new_source
                return mutated
        
        return genome.clone()
    
    def _consolidate_loops(self, genome: Genome) -> Genome:
        """Consolidate consecutive similar loops."""
        source = genome.source
        
        # Simple consolidation patterns
        if 'for i in range' in source and source.count('for i in range') >= 2:
            # Simple deduplication attempt
            new_source = re.sub(r'for i in range\([^)]+\):\s*pass', '', source, count=1)
            
            mutated = genome.clone()
            mutated.source = new_source
            return mutated
        
        return genome.clone()
    
    def _extract_method(self, genome: Genome) -> Genome:
        """Extract code to a new method."""
        source = genome.source
        lines = source.split('\n')
        
        # Find extractable blocks (simple heuristic)
        for i, line in enumerate(lines):
            if 'for ' in line or 'while ' in line:
                # Insert helper function before
                helper = '''
def _helper(data):
    """Extracted helper function."""
    return data
'''
                lines.insert(i, helper)
                
                mutated = genome.clone()
                mutated.source = '\n'.join(lines)
                return mutated
        
        return genome.clone()
    
    def _inline_mutation(self, genome: Genome) -> Genome:
        """Inline function calls or variables."""
        source = genome.source
        
        # Simple inlining: remove redundant assignments
        inlined = re.sub(r'(\w+)\s*=\s*\1\b', r'\1', source)
        
        mutated = genome.clone()
        mutated.source = inlined
        return mutated
    
    def _extract_mutation(self, genome: Genome) -> Genome:
        """Extract repeated code to a function."""
        return self._extract_method(genome)
    
    # Optimization Mutations
    
    def optimize_performance(self, genome: Genome) -> Genome:
        """Apply performance optimizations."""
        optimizations = [
            self._add_list_comprehension,
            self._use_set_lookup,
            self._add_memoization,
            self._preallocate_collections,
        ]
        
        opt = random.choice(optimizations)
        return opt(genome)
    
    def _add_list_comprehension(self, genome: Genome) -> Genome:
        """Convert loops to list comprehensions where applicable."""
        source = genome.source
        
        # Pattern: for x in y: result.append(x)
        pattern = r'for\s+(\w+)\s+in\s+([^:]+):\s*\n\s*result\.append\(\1\)'
        if re.search(pattern, source):
            new_source = re.sub(
                pattern,
                r'result = [\1 for \1 in \2]',
                source
            )
            
            mutated = genome.clone()
            mutated.source = new_source
            return mutated
        
        return genome.clone()
    
    def _use_set_lookup(self, genome: Genome) -> Genome:
        """Convert list membership tests to set lookups."""
        source = genome.source
        
        # Pattern: x in [a, b, c]
        pattern = r'in\s+\[([^]]+)\]'
        matches = re.findall(pattern, source)
        
        if matches:
            for match in matches:
                items = [x.strip() for x in match.split(',')]
                if len(items) >= 3:
                    set_literal = '{' + match + '}'
                    new_source = source.replace(f'[{match}]', set_literal, 1)
                    
                    mutated = genome.clone()
                    mutated.source = new_source
                    return mutated
        
        return genome.clone()
    
    def _add_memoization(self, genome: Genome) -> Genome:
        """Add memoization/caching."""
        source = genome.source
        
        if 'def ' in source and 'fibonacci' in source.lower():
            # Add simple memoization
            memo_code = '''
from functools import lru_cache

'''
            new_source = memo_code + source
            
            # Add decorator to function
            new_source = re.sub(
                r'def\s+(\w+)\(',
                r'@lru_cache(maxsize=None)\ndef \1(',
                new_source,
                count=1
            )
            
            mutated = genome.clone()
            mutated.source = new_source
            return mutated
        
        return genome.clone()
    
    def _preallocate_collections(self, genome: Genome) -> Genome:
        """Preallocate collections for better performance."""
        source = genome.source
        
        # Pattern: result = [] followed by appends
        if 'result = []' in source and '.append(' in source:
            # Try to infer size
            append_count = source.count('.append(')
            
            new_source = source.replace(
                'result = []',
                f'result = [None] * {append_count}'
            )
            
            mutated = genome.clone()
            mutated.source = new_source
            return mutated
        
        return genome.clone()
    
    def _add_cache_mutation(self, genome: Genome) -> Genome:
        """Add caching mechanism."""
        return self._add_memoization(genome)
    
    # Semantic Mutations
    
    def _type_change_mutation(self, genome: Genome) -> Genome:
        """Change data types."""
        source = genome.source
        
        type_changes = [
            (r'\[\]', 'set()'),
            (r'set\(\)', '[]'),
            (r'\{\}', '{: None}'),
        ]
        
        for pattern, replacement in type_changes:
            if re.search(pattern, source):
                new_source = re.sub(pattern, replacement, source, count=1)
                
                mutated = genome.clone()
                mutated.source = new_source
                return mutated
        
        return genome.clone()
    
    def _recursion_to_loop(self, genome: Genome) -> Genome:
        """Convert recursive function to iterative."""
        source = genome.source
        
        if 'def ' in source and 'return ' in source:
            # Simple heuristic conversion
            # This is a simplified version; real implementation would need AST analysis
            
            if 'fibonacci' in source.lower():
                new_source = '''
def fibonacci_iterative(n):
    """Iterative Fibonacci implementation."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b
'''
                mutated = genome.clone()
                mutated.source = new_source
                return mutated
        
        return genome.clone()
    
    def _loop_to_recursion(self, genome: Genome) -> Genome:
        """Convert iterative function to recursive."""
        source = genome.source
        
        if 'for ' in source and 'range' in source:
            # Simple heuristic
            # Would need more sophisticated AST analysis for real implementation
            return genome.clone()
        
        return genome.clone()
    
    def _loop_conversion(self, genome: Genome) -> Genome:
        """Convert between loop types."""
        source = genome.source
        
        # while to for
        if 'while ' in source:
            new_source = re.sub(r'while\s+(\w+)\s*<\s*(\d+):', 
                               r'for \1 in range(\2):', source, count=1)
            if new_source != source:
                mutated = genome.clone()
                mutated.source = new_source
                return mutated
        
        return genome.clone()
    
    # Quality Mutations
    
    def add_comments(self, genome: Genome) -> Genome:
        """Add helpful comments."""
        source = genome.source
        
        # Add module docstring if missing
        if not (source.startswith('"""') or source.startswith("'''")):
            if 'def ' in source:
                idx = source.find('def ')
                comment = '"""\nModule docstring.\n"""\n\n'
                new_source = comment + source[:idx] + source[idx:]
                
                mutated = genome.clone()
                mutated.source = new_source
                return mutated
        
        return genome.clone()
    
    def add_type_hints(self, genome: Genome) -> Genome:
        """Add type annotations."""
        source = genome.source
        
        # Add return type hints
        if 'def ' in source and '->' not in source:
            # Add simple return type
            new_source = re.sub(
                r'def\s+(\w+)\s*\(([^)]*)\):',
                r'def \1(\2) -> None:',
                source,
                count=1
            )
            
            # Add parameter type hints
            params = re.search(r'def\s+\w+\s*\(([^)]*)\)', new_source)
            if params and params.group(1).strip():
                param_str = params.group(1)
                typed_params = ', '.join(f'{p}: int' for p in param_str.split(','))
                
                new_source = re.sub(
                    r'def\s+(\w+)\s*\([^)]*\):',
                    f'def \\1({typed_params}):',
                    new_source,
                    count=1
                )
            
            mutated = genome.clone()
            mutated.source = new_source
            return mutated
        
        return genome.clone()
    
    def improve_readability(self, genome: Genome) -> Genome:
        """Improve code readability."""
        source = genome.source
        
        improvements = [
            # Add blank lines around operators
            (r'(\S)\s*([+\-*/])\s*(\S)', r'\1 \2 \3'),
            # Fix spacing in function calls
            (r'(\w+)\(\s+', r'\1('),
            (r'\s+\)', r')'),
        ]
        
        new_source = source
        for pattern, replacement in improvements:
            new_source = re.sub(pattern, replacement, new_source)
        
        if new_source != source:
            mutated = genome.clone()
            mutated.source = new_source
            return mutated
        
        return genome.clone()
    
    def remove_dead_code(self, genome: Genome) -> Genome:
        """Remove unreachable or unused code."""
        source = genome.source
        
        # Remove empty try blocks
        empty_try = r'try:\s*pass\s*except[^:]+:\s*pass'
        new_source = re.sub(empty_try, 'pass', source)
        
        # Remove commented-out code
        new_source = re.sub(r'#\s*\w+\s*=\s*[^\n]+\n(?=\n)', '', new_source)
        
        mutated = genome.clone()
        mutated.source = new_source
        return mutated
    
    # Batch operations
    
    def mutate_multiple(
        self, 
        genome: Genome, 
        count: Optional[int] = None
    ) -> list[Genome]:
        """Apply multiple mutations to create diverse offspring."""
        count = count or random.randint(1, self.config["max_mutations_per_genome"])
        
        current = genome
        results = []
        
        for _ in range(count):
            current = self.mutate(current)
            results.append(current)
        
        return results


# Convenience function
def quick_mutate(source: str) -> str:
    """Quickly mutate source code."""
    genome = Genome(source=source)
    mutator = Mutator()
    mutated = mutator.mutate(genome)
    return mutated.source
