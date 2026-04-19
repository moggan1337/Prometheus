"""
Thinking Levels Module - Implements multi-level cognitive processing.

Inspired by dual-process theory and modern reasoning systems:
- System 1: Fast, instinctive processing
- System 2: Slow, deliberate reasoning  
- System 3: Meta-cognitive self-improvement

Each level has different:
- Token budgets
- Processing strategies
- Quality vs speed tradeoffs
"""

from __future__ import annotations
import time
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Protocol
from abc import ABC, abstractmethod


class ThinkingLevel(Enum):
    """
    Cognitive processing levels.
    
    Each level represents a different mode of reasoning with
    associated tradeoffs between speed, depth, and resource usage.
    """
    INSTINCT = auto()    # Fast, pattern-matching (System 1)
    REASONED = auto()    # Deliberate analysis (System 2)
    ANALYTICAL = auto()  # Deep analysis (System 2+)
    METACOGNITIVE = auto()  # Self-aware reasoning (System 3)


@dataclass
class ThinkingConfig:
    """Configuration for thinking levels."""
    level: ThinkingLevel = ThinkingLevel.REASONED
    token_budget: int = 500
    max_time_ms: int = 5000
    enable_reflection: bool = True
    enable_self_correction: bool = True
    depth_limit: int = 5


@dataclass 
class Thought:
    """Represents a single thought or reasoning step."""
    content: str
    level: ThinkingLevel
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    parent: Optional[Thought] = None
    children: list[Thought] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def add_child(self, thought: 'Thought') -> None:
        """Add a child thought."""
        thought.parent = self
        self.children.append(thought)


@dataclass
class ThinkingResult:
    """Result of thinking process."""
    thoughts: list[Thought]
    level: ThinkingLevel
    tokens_used: int
    time_elapsed_ms: float
    final_answer: Any = None
    confidence: float = 1.0
    reflection: Optional[str] = None


class ThinkingStrategy(ABC):
    """Abstract base class for thinking strategies."""
    
    @abstractmethod
    def think(self, problem: str, config: ThinkingConfig) -> ThinkingResult:
        """Execute thinking process."""
        pass


class InstinctStrategy(ThinkingStrategy):
    """
    System 1: Fast, instinctive thinking.
    
    Characteristics:
    - Low token budget
    - Pattern matching
    - Quick responses
    - High speed, lower accuracy
    """
    
    def think(self, problem: str, config: ThinkingConfig) -> ThinkingResult:
        """Execute instinct-driven thinking."""
        start_time = time.time()
        
        # Quick pattern matching
        thoughts = []
        
        # Initial recognition
        thought = Thought(
            content=f"Quick pattern recognition for: {problem[:50]}...",
            level=ThinkingLevel.INSTINCT,
            confidence=0.7,
        )
        thoughts.append(thought)
        
        # Fast heuristic response
        response = self._fast_heuristic(problem)
        
        thought2 = Thought(
            content=f"Pattern matched: {response['pattern']}",
            level=ThinkingLevel.INSTINCT,
            confidence=0.8,
            parent=thought,
        )
        thoughts.append(thought2)
        thought.add_child(thought2)
        
        tokens_used = 50
        elapsed = (time.time() - start_time) * 1000
        
        return ThinkingResult(
            thoughts=thoughts,
            level=ThinkingLevel.INSTINCT,
            tokens_used=tokens_used,
            time_elapsed_ms=elapsed,
            final_answer=response['answer'],
            confidence=0.7,
        )
    
    def _fast_heuristic(self, problem: str) -> dict:
        """Apply fast heuristic matching."""
        # Simple keyword-based matching
        patterns = {
            'sort': 'Use built-in sorted()',
            'search': 'Use binary search or dict lookup',
            'loop': 'Consider list comprehension',
            'recursive': 'Consider memoization',
            'optimize': 'Profile first, then optimize',
        }
        
        matched = []
        for keyword, suggestion in patterns.items():
            if keyword in problem.lower():
                matched.append(suggestion)
        
        return {
            'pattern': matched[0] if matched else 'No pattern matched',
            'answer': matched[0] if matched else 'Standard approach',
        }


class ReasonedStrategy(ThinkingStrategy):
    """
    System 2: Deliberate reasoning.
    
    Characteristics:
    - Medium token budget
    - Step-by-step analysis
    - Moderate speed and accuracy
    """
    
    def think(self, problem: str, config: ThinkingConfig) -> ThinkingResult:
        """Execute deliberate reasoning."""
        start_time = time.time()
        
        thoughts = []
        
        # Problem decomposition
        thought1 = Thought(
            content=f"Decomposing problem: {problem[:100]}",
            level=ThinkingLevel.REASONED,
            confidence=0.9,
        )
        thoughts.append(thought1)
        
        # Identify components
        components = self._decompose_problem(problem)
        
        thought2 = Thought(
            content=f"Identified {len(components)} key components: {', '.join(components[:3])}",
            level=ThinkingLevel.REASONED,
            confidence=0.85,
            parent=thought1,
        )
        thoughts.append(thought2)
        thought1.add_child(thought2)
        
        # Analyze each component
        for i, component in enumerate(components[:config.depth_limit]):
            thought = Thought(
                content=f"Analyzing component {i+1}: {component}",
                level=ThinkingLevel.REASONED,
                confidence=0.85,
            )
            thoughts.append(thought)
        
        # Synthesize solution
        thought3 = Thought(
            content="Synthesizing solution from analyzed components",
            level=ThinkingLevel.REASONED,
            confidence=0.9,
        )
        thoughts.append(thought3)
        
        # Generate response
        solution = self._synthesize_solution(problem, components)
        
        tokens_used = 200
        elapsed = (time.time() - start_time) * 1000
        
        return ThinkingResult(
            thoughts=thoughts,
            level=ThinkingLevel.REASONED,
            tokens_used=tokens_used,
            time_elapsed_ms=elapsed,
            final_answer=solution,
            confidence=0.85,
        )
    
    def _decompose_problem(self, problem: str) -> list[str]:
        """Decompose problem into components."""
        # Simple decomposition based on keywords
        keywords = ['input', 'process', 'output', 'edge', 'case', 
                   'optimize', 'handle', 'return', 'calculate']
        
        components = []
        words = problem.lower().split()
        
        for keyword in keywords:
            if keyword in words:
                components.append(keyword)
        
        if not components:
            components = ['general', 'implementation']
        
        return components
    
    def _synthesize_solution(self, problem: str, components: list) -> str:
        """Synthesize solution from components."""
        return f"Solution addressing {', '.join(components)}"


class AnalyticalStrategy(ThinkingStrategy):
    """
    System 2+: Deep analytical thinking.
    
    Characteristics:
    - High token budget
    - Thorough analysis
    - Multiple perspectives
    - Verification steps
    """
    
    def think(self, problem: str, config: ThinkingConfig) -> ThinkingResult:
        """Execute deep analytical thinking."""
        start_time = time.time()
        
        thoughts = []
        
        # Deep problem analysis
        thought1 = Thought(
            content=f"Deep analysis of problem: {problem}",
            level=ThinkingLevel.ANALYTICAL,
            confidence=1.0,
        )
        thoughts.append(thought1)
        
        # Multiple perspective analysis
        perspectives = ['functional', 'performance', 'readability', 'maintainability']
        
        perspective_thoughts = []
        for perspective in perspectives:
            thought = Thought(
                content=f"Analyzing from {perspective} perspective",
                level=ThinkingLevel.ANALYTICAL,
                confidence=0.9,
                parent=thought1,
            )
            thoughts.append(thought)
            perspective_thoughts.append(thought)
        
        # Identify tradeoffs
        thought_tradeoffs = Thought(
            content=f"Identified tradeoffs between perspectives",
            level=ThinkingLevel.ANALYTICAL,
            confidence=0.85,
        )
        thoughts.append(thought_tradeoffs)
        
        # Generate optimized solution
        solution = self._generate_optimized_solution(problem, perspectives)
        
        # Verification step
        thought_verify = Thought(
            content="Verifying solution correctness and efficiency",
            level=ThinkingLevel.ANALYTICAL,
            confidence=0.9,
        )
        thoughts.append(thought_verify)
        
        tokens_used = 400
        elapsed = (time.time() - start_time) * 1000
        
        return ThinkingResult(
            thoughts=thoughts,
            level=ThinkingLevel.ANALYTICAL,
            tokens_used=tokens_used,
            time_elapsed_ms=elapsed,
            final_answer=solution,
            confidence=0.9,
        )
    
    def _generate_optimized_solution(self, problem: str, perspectives: list) -> str:
        """Generate solution optimized for multiple perspectives."""
        return f"Optimized solution balancing {', '.join(perspectives)}"


class MetacognitiveStrategy(ThinkingStrategy):
    """
    System 3: Meta-cognitive self-aware thinking.
    
    Characteristics:
    - Self-monitoring
    - Strategy selection
    - Performance optimization
    - Learning from feedback
    """
    
    def think(self, problem: str, config: ThinkingConfig) -> ThinkingResult:
        """Execute meta-cognitive thinking."""
        start_time = time.time()
        
        thoughts = []
        
        # Self-awareness
        thought1 = Thought(
            content="Activating meta-cognitive monitoring",
            level=ThinkingLevel.METACOGNITIVE,
            confidence=1.0,
        )
        thoughts.append(thought1)
        
        # Strategy selection
        thought2 = Thought(
            content=f"Selecting optimal thinking strategy for problem type",
            level=ThinkingLevel.METACOGNITIVE,
            confidence=0.95,
            parent=thought1,
        )
        thoughts.append(thought2)
        
        # Problem classification
        problem_type = self._classify_problem(problem)
        
        thought3 = Thought(
            content=f"Classified as {problem_type} problem",
            level=ThinkingLevel.METACOGNITIVE,
            confidence=0.9,
            parent=thought2,
        )
        thoughts.append(thought3)
        
        # Strategy adaptation
        adapted_strategy = self._adapt_strategy(problem_type, config)
        
        thought4 = Thought(
            content=f"Adapted strategy: {adapted_strategy}",
            level=ThinkingLevel.METACOGNITIVE,
            confidence=0.85,
        )
        thoughts.append(thought4)
        
        # Execute adapted thinking
        # ... (would delegate to appropriate strategy)
        
        # Performance monitoring
        thought5 = Thought(
            content="Monitoring thinking performance and adjusting",
            level=ThinkingLevel.METACOGNITIVE,
            confidence=0.9,
        )
        thoughts.append(thought5)
        
        # Reflection
        reflection = self._generate_reflection(problem_type)
        
        tokens_used = 450
        elapsed = (time.time() - start_time) * 1000
        
        return ThinkingResult(
            thoughts=thoughts,
            level=ThinkingLevel.METACOGNITIVE,
            tokens_used=tokens_used,
            time_elapsed_ms=elapsed,
            final_answer=f"Meta-cognitively optimized solution",
            confidence=0.95,
            reflection=reflection,
        )
    
    def _classify_problem(self, problem: str) -> str:
        """Classify problem type."""
        classifications = {
            'algorithmic': ['sort', 'search', 'graph', 'tree', 'dynamic'],
            'optimization': ['minimize', 'maximize', 'optimal', 'efficient'],
            'data': ['process', 'transform', 'analyze', 'filter'],
            'computational': ['calculate', 'compute', 'evaluate', 'recursive'],
        }
        
        problem_lower = problem.lower()
        
        for ptype, keywords in classifications.items():
            if any(kw in problem_lower for kw in keywords):
                return ptype
        
        return 'general'
    
    def _adapt_strategy(self, problem_type: str, config: ThinkingConfig) -> str:
        """Adapt strategy based on problem type."""
        adaptations = {
            'algorithmic': 'Deep analysis with complexity consideration',
            'optimization': 'Multi-objective evaluation',
            'data': 'Pipeline thinking with validation',
            'computational': 'Recursive decomposition',
            'general': 'Standard reasoned approach',
        }
        return adaptations.get(problem_type, 'Standard approach')
    
    def _generate_reflection(self, problem_type: str) -> str:
        """Generate meta-cognitive reflection."""
        return f"Approached {problem_type} problem with appropriate depth and verification"


class ThinkingEngine:
    """
    Multi-level thinking engine.
    
    Provides access to different levels of cognitive processing
    with configurable budgets and strategies.
    """
    
    def __init__(self, config: Optional[ThinkingConfig] = None):
        """Initialize thinking engine."""
        self.config = config or ThinkingConfig()
        self.strategies = {
            ThinkingLevel.INSTINCT: InstinctStrategy(),
            ThinkingLevel.REASONED: ReasonedStrategy(),
            ThinkingLevel.ANALYTICAL: AnalyticalStrategy(),
            ThinkingLevel.METACOGNITIVE: MetacognitiveStrategy(),
        }
        
        self.thinking_history: list[ThinkingResult] = []
    
    def think(
        self, 
        problem: str, 
        level: Optional[ThinkingLevel] = None
    ) -> ThinkingResult:
        """
        Execute thinking process.
        
        Args:
            problem: The problem to think about
            level: Specific thinking level, or None for auto-selection
            
        Returns:
            ThinkingResult with thoughts and final answer
        """
        level = level or self.config.level
        
        # Check token budget
        if self.config.token_budget < 100:
            level = ThinkingLevel.INSTINCT
        elif self.config.token_budget < 300:
            level = ThinkingLevel.REASONED
        
        # Execute thinking
        strategy = self.strategies.get(level, self.strategies[ThinkingLevel.REASONED])
        result = strategy.think(problem, self.config)
        
        # Record history
        self.thinking_history.append(result)
        
        return result
    
    def think_with_budget(
        self, 
        problem: str, 
        token_budget: int
    ) -> ThinkingResult:
        """
        Think with specific token budget.
        
        Args:
            problem: Problem to solve
            token_budget: Available tokens
            
        Returns:
            ThinkingResult
        """
        # Select appropriate level based on budget
        if token_budget < 100:
            level = ThinkingLevel.INSTINCT
        elif token_budget < 300:
            level = ThinkingLevel.REASONED
        elif token_budget < 500:
            level = ThinkingLevel.ANALYTICAL
        else:
            level = ThinkingLevel.METACOGNITIVE
        
        # Temporarily override config
        original_budget = self.config.token_budget
        self.config.token_budget = token_budget
        
        result = self.think(problem, level)
        
        # Restore config
        self.config.token_budget = original_budget
        
        return result
    
    def think_adaptive(
        self, 
        problem: str, 
        time_limit_ms: int = 5000
    ) -> ThinkingResult:
        """
        Adaptive thinking with time limit.
        
        Args:
            problem: Problem to solve
            time_limit_ms: Time limit in milliseconds
            
        Returns:
            ThinkingResult with best effort
        """
        start_time = time.time()
        
        # Start with fast level
        result = self.think(problem, ThinkingLevel.INSTINCT)
        
        # Check if more time available
        elapsed = (time.time() - start_time) * 1000
        remaining = time_limit_ms - elapsed
        
        if remaining > 2000 and result.confidence < 0.8:
            # Try higher level
            result2 = self.think(problem, ThinkingLevel.REASONED)
            if result2.confidence > result.confidence:
                result = result2
        
        return result
    
    def get_thinking_stats(self) -> dict:
        """Get statistics about thinking history."""
        if not self.thinking_history:
            return {"total_thinks": 0}
        
        return {
            "total_thinks": len(self.thinking_history),
            "avg_confidence": sum(r.confidence for r in self.thinking_history) / len(self.thinking_history),
            "avg_tokens": sum(r.tokens_used for r in self.thinking_history) / len(self.thinking_history),
            "avg_time_ms": sum(r.time_elapsed_ms for r in self.thinking_history) / len(self.thinking_history),
            "level_distribution": {
                level.name: sum(1 for r in self.thinking_history if r.level == level)
                for level in ThinkingLevel
            },
        }


# Convenience function
def think(problem: str, level: ThinkingLevel = ThinkingLevel.REASONED) -> ThinkingResult:
    """Quickly execute thinking process."""
    engine = ThinkingEngine()
    return engine.think(problem, level)
