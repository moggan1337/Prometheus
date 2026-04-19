"""
Dream Loop Module - Periodic knowledge distillation and reflection.

The Dream Loop implements a self-improvement cycle where:
1. Consolidation: Patterns are distilled from recent evolution
2. Reflection: Successes and failures are analyzed
3. Abstraction: General principles are extracted
4. Planning: Future strategies are formulated
"""

from __future__ import annotations
import time
import random
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from collections import defaultdict

from ..core.genome import Genome, FitnessScores
from ..knowledge.base import KnowledgeBase, Pattern, PatternType


class DreamPhase(Enum):
    """Phases of the dream/consolidation loop."""
    IDLE = auto()
    CONSOLIDATION = auto()  # Distilling patterns
    REFLECTION = auto()     # Analyzing outcomes
    ABSTRACTION = auto()    # Extracting principles
    PLANNING = auto()       # Strategy formulation
    INTEGRATION = auto()    # Updating knowledge base


@dataclass
class DreamResult:
    """Result of a dream loop cycle."""
    phase: DreamPhase
    duration_ms: float
    patterns_extracted: int = 0
    insights: list[str] = field(default_factory=list)
    strategies: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return self.patterns_extracted > 0 or len(self.insights) > 0


@dataclass
class DreamConfig:
    """Configuration for dream loop."""
    enabled: bool = True
    interval_generations: int = 10  # Run every N generations
    min_patterns_for_abstraction: int = 5
    
    # Consolidation parameters
    consolidation_threshold: float = 0.7  # Min fitness to extract pattern
    max_patterns_per_cycle: int = 10
    
    # Reflection parameters
    reflection_depth: int = 3  # How deep to analyze
    analyze_failures: bool = True
    
    # Abstraction parameters
    abstraction_level: int = 2  # How abstract the principles
    min_occurrences: int = 3    # Min occurrences for generalization
    
    # Planning parameters
    planning_horizon: int = 5   # How far ahead to plan
    adaptive_strategy: bool = True


@dataclass
class Insight:
    """Represents an insight from the dream process."""
    id: str
    content: str
    source_patterns: list[str]
    confidence: float
    abstraction_level: int
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "source_patterns": self.source_patterns,
            "confidence": self.confidence,
            "abstraction_level": self.abstraction_level,
            "created_at": self.created_at,
        }


@dataclass
class Strategy:
    """Represents a strategic insight for future evolution."""
    id: str
    name: str
    description: str
    rationale: str
    applicable_conditions: list[str]
    expected_improvement: float
    confidence: float
    created_at: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "rationale": self.rationale,
            "applicable_conditions": self.applicable_conditions,
            "expected_improvement": self.expected_improvement,
            "confidence": self.confidence,
            "created_at": self.created_at,
        }


class DreamLoop:
    """
    Implements periodic knowledge distillation and reflection.
    
    The Dream Loop runs periodically (e.g., every N generations) to:
    1. Consolidate: Extract successful patterns from recent genomes
    2. Reflect: Analyze what worked and what didn't
    3. Abstract: Generalize patterns into higher-level principles
    4. Plan: Formulate strategies for future evolution
    5. Integrate: Update the knowledge base with insights
    """
    
    def __init__(
        self, 
        knowledge_base: KnowledgeBase,
        config: Optional[DreamConfig] = None
    ):
        """Initialize dream loop."""
        self.knowledge_base = knowledge_base
        self.config = config or DreamConfig()
        
        # State
        self.phase = DreamPhase.IDLE
        self.last_dream_time = 0.0
        self.dream_count = 0
        
        # Archives for consolidation
        self.genome_archive: list[Genome] = []
        self.success_archive: list[Genome] = []
        self.failure_archive: list[Genome] = []
        
        # Insights and strategies
        self.insights: list[Insight] = []
        self.strategies: list[Strategy] = []
        
        # Statistics
        self.dream_history: list[DreamResult] = []
    
    def archive(self, genome: Genome) -> None:
        """Archive a genome for future consolidation."""
        self.genome_archive.append(genome)
        
        if genome.fitness.composite >= self.config.consolidation_threshold:
            self.success_archive.append(genome)
        else:
            self.failure_archive.append(genome)
    
    def should_dream(self, current_generation: int) -> bool:
        """Check if it's time to run the dream loop."""
        if not self.config.enabled:
            return False
        
        # Run at configured intervals
        if current_generation % self.config.interval_generations != 0:
            return False
        
        # Run if we have enough archived data
        if len(self.success_archive) < 3:
            return False
        
        # Run if enough time has passed
        time_since_dream = time.time() - self.last_dream_time
        if time_since_dream < 60:  # Minimum 1 minute between dreams
            return False
        
        return True
    
    def dream(self) -> DreamResult:
        """
        Execute the dream loop cycle.
        
        Returns:
            DreamResult with insights and strategies
        """
        start_time = time.time()
        result = DreamResult(phase=DreamPhase.IDLE, duration_ms=0)
        
        try:
            # Phase 1: Consolidation
            self.phase = DreamPhase.CONSOLIDATION
            result.phase = DreamPhase.CONSOLIDATION
            patterns = self._consolidate()
            result.patterns_extracted = len(patterns)
            
            # Phase 2: Reflection
            self.phase = DreamPhase.REFLECTION
            insights = self._reflect()
            result.insights = [i.content for i in insights]
            self.insights.extend(insights)
            
            # Phase 3: Abstraction
            self.phase = DreamPhase.ABSTRACTION
            more_insights = self._abstract(patterns)
            result.insights.extend([i.content for i in more_insights])
            self.insights.extend(more_insights)
            
            # Phase 4: Planning
            self.phase = DreamPhase.PLANNING
            strategies = self._plan()
            result.strategies = [s.name for s in strategies]
            self.strategies.extend(strategies)
            
            # Phase 5: Integration
            self.phase = DreamPhase.INTEGRATION
            self._integrate(patterns, insights, strategies)
            
            # Update state
            self.last_dream_time = time.time()
            self.dream_count += 1
            
            # Cleanup archives
            self._cleanup_archives()
            
        finally:
            self.phase = DreamPhase.IDLE
        
        result.duration_ms = (time.time() - start_time) * 1000
        self.dream_history.append(result)
        
        return result
    
    def _consolidate(self) -> list[Pattern]:
        """
        Phase 1: Consolidate patterns from successful genomes.
        
        Extracts patterns from the best performing genomes.
        """
        patterns = []
        
        # Sort by fitness
        sorted_genomes = sorted(
            self.success_archive,
            key=lambda g: g.fitness.composite,
            reverse=True
        )
        
        # Extract patterns from top performers
        for genome in sorted_genomes[:self.config.max_patterns_per_cycle]:
            pattern = self.knowledge_base.extract_from_genome(genome)
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def _reflect(self) -> list[Insight]:
        """
        Phase 2: Reflect on successes and failures.
        
        Analyzes what worked and what didn't.
        """
        insights = []
        
        # Analyze successful patterns
        if self.success_archive:
            best = max(self.success_archive, key=lambda g: g.fitness.composite)
            worst = min(self.success_archive, key=lambda g: g.fitness.composite)
            
            # Generate insights from comparison
            insight = Insight(
                id=f"insight_{int(time.time() * 1000)}",
                content=f"Best performing pattern achieves {best.fitness.composite:.2f} fitness vs average {sum(g.fitness.composite for g in self.success_archive) / len(self.success_archive):.2f}",
                source_patterns=[best.id],
                confidence=0.8,
                abstraction_level=1,
            )
            insights.append(insight)
        
        # Analyze failures
        if self.config.analyze_failures and self.failure_archive:
            failure_rate = len(self.failure_archive) / max(1, len(self.genome_archive))
            
            if failure_rate > 0.5:
                insight = Insight(
                    id=f"insight_{int(time.time() * 1000) + 1}",
                    content=f"High failure rate ({failure_rate:.1%}) suggests need for mutation diversity or different selection strategy",
                    source_patterns=[g.id for g in self.failure_archive[:3]],
                    confidence=0.7,
                    abstraction_level=2,
                )
                insights.append(insight)
        
        # Analyze fitness components
        if len(self.success_archive) >= 3:
            avg_correctness = sum(g.fitness.correctness for g in self.success_archive) / len(self.success_archive)
            avg_performance = sum(g.fitness.performance for g in self.success_archive) / len(self.success_archive)
            
            if avg_correctness > avg_performance:
                insight = Insight(
                    id=f"insight_{int(time.time() * 1000) + 2}",
                    content="Correctness outperforms performance - consider optimization focus",
                    source_patterns=[g.id for g in self.success_archive[:3]],
                    confidence=0.6,
                    abstraction_level=2,
                )
                insights.append(insight)
        
        return insights
    
    def _abstract(self, patterns: list[Pattern]) -> list[Insight]:
        """
        Phase 3: Abstract higher-level principles.
        
        Generalizes patterns into reusable principles.
        """
        insights = []
        
        if len(patterns) < self.config.min_patterns_for_abstraction:
            return insights
        
        # Group patterns by type
        by_type = defaultdict(list)
        for p in patterns:
            by_type[p.pattern_type].append(p)
        
        # Generate type-specific insights
        for ptype, type_patterns in by_type.items():
            if len(type_patterns) >= self.config.min_occurrences:
                # Analyze common elements
                common_elements = self._find_common_elements(type_patterns)
                
                if common_elements:
                    insight = Insight(
                        id=f"insight_{int(time.time() * 1000)}",
                        content=f"{ptype.name} patterns share: {', '.join(common_elements[:3])}",
                        source_patterns=[p.id for p in type_patterns],
                        confidence=0.75,
                        abstraction_level=self.config.abstraction_level,
                    )
                    insights.append(insight)
        
        # Generate cross-type insights
        if len(by_type) > 1:
            types = list(by_type.keys())
            insight = Insight(
                id=f"insight_{int(time.time() * 1000) + 1}",
                content=f"Evolutionary success involves {len(types)} different pattern types: {', '.join(t.name for t in types)}",
                source_patterns=[p.id for p in patterns[:5]],
                confidence=0.7,
                abstraction_level=3,
            )
            insights.append(insight)
        
        # Generate optimization insight
        optimization_patterns = by_type.get(PatternType.OPTIMIZATION, [])
        if optimization_patterns:
            best_opt = max(optimization_patterns, key=lambda p: p.effectiveness)
            insight = Insight(
                id=f"insight_{int(time.time() * 1000) + 2}",
                content=f"Most effective optimization pattern improves fitness by {best_opt.avg_fitness:.2f}",
                source_patterns=[best_opt.id],
                confidence=0.85,
                abstraction_level=2,
            )
            insights.append(insight)
        
        return insights
    
    def _find_common_elements(self, patterns: list[Pattern]) -> list[str]:
        """Find common elements across patterns."""
        if not patterns:
            return []
        
        # Simple keyword-based commonality
        all_keywords = []
        for pattern in patterns:
            keywords = self._extract_keywords(pattern.source)
            all_keywords.extend(keywords)
        
        # Count frequencies
        from collections import Counter
        counter = Counter(all_keywords)
        
        # Return keywords appearing in multiple patterns
        common = [kw for kw, count in counter.items() if count >= len(patterns) * 0.5]
        return common[:5]
    
    def _extract_keywords(self, source: str) -> list[str]:
        """Extract keywords from source code."""
        import re
        
        keywords = []
        
        # Extract function names
        func_names = re.findall(r'def\s+(\w+)', source)
        keywords.extend(func_names)
        
        # Extract common programming constructs
        constructs = re.findall(r'\b(for|while|if|return|try|except|with)\b', source)
        keywords.extend(constructs)
        
        # Extract common method names
        methods = re.findall(r'\.(\w+)\(', source)
        keywords.extend(methods)
        
        return keywords
    
    def _plan(self) -> list[Strategy]:
        """
        Phase 4: Formulate strategies for future evolution.
        
        Based on insights, generates actionable strategies.
        """
        strategies = []
        
        # Analyze recent performance
        recent_success_rate = (
            len(self.success_archive) / max(1, len(self.genome_archive))
        )
        
        # Strategy 1: Mutation diversity
        if recent_success_rate < 0.5:
            strategy = Strategy(
                id=f"strategy_{int(time.time() * 1000)}",
                name="increase_mutation_diversity",
                description="Apply broader mutation operators to escape local optima",
                rationale=f"Success rate ({recent_success_rate:.1%}) indicates potential local optima",
                applicable_conditions=["low_success_rate", "stagnation"],
                expected_improvement=0.1,
                confidence=0.8,
            )
            strategies.append(strategy)
        
        # Strategy 2: Crossover focus
        if len(self.success_archive) >= 5:
            strategy = Strategy(
                id=f"strategy_{int(time.time() * 1000) + 1}",
                name="focus_crossover",
                description="Emphasize crossover operations to combine successful traits",
                rationale="Sufficient successful patterns available for meaningful crossover",
                applicable_conditions=["diverse_successful_patterns"],
                expected_improvement=0.15,
                confidence=0.75,
            )
            strategies.append(strategy)
        
        # Strategy 3: Elite preservation
        strategy = Strategy(
            id=f"strategy_{int(time.time() * 1000) + 2}",
            name="preserve_elite",
            description="Increase elite preservation to maintain best solutions",
            rationale="Best solutions should be protected from destructive mutations",
            applicable_conditions=["always"],
            expected_improvement=0.05,
            confidence=0.7,
        )
        strategies.append(strategy)
        
        # Strategy 4: Adaptive selection
        if self.config.adaptive_strategy:
            strategy = Strategy(
                id=f"strategy_{int(time.time() * 1000) + 3}",
                name="adaptive_selection_pressure",
                description="Adjust selection pressure based on diversity levels",
                rationale="Maintain diversity while selecting for improvement",
                applicable_conditions=["low_diversity", "high_diversity"],
                expected_improvement=0.08,
                confidence=0.65,
            )
            strategies.append(strategy)
        
        # Strategy 5: Knowledge-guided mutation
        if len(self.knowledge_base) > 10:
            strategy = Strategy(
                id=f"strategy_{int(time.time() * 1000) + 4}",
                name="knowledge_guided_mutation",
                description="Use knowledge base patterns to guide mutation directions",
                rationale=f"Knowledge base contains {len(self.knowledge_base)} patterns",
                applicable_conditions=["has_knowledge_base"],
                expected_improvement=0.12,
                confidence=0.7,
            )
            strategies.append(strategy)
        
        return strategies
    
    def _integrate(
        self, 
        patterns: list[Pattern],
        insights: list[Insight],
        strategies: list[Strategy]
    ) -> None:
        """
        Phase 5: Integrate findings into knowledge base.
        
        Adds patterns, insights, and strategies to the knowledge base.
        """
        # Add patterns
        for pattern in patterns:
            # Check for duplicates
            similar = self.knowledge_base.find_similar(pattern.source, threshold=0.8)
            if not similar:
                self.knowledge_base.add(pattern)
        
        # Store insights and strategies in metadata
        # (In a full implementation, these would be stored separately)
        pass
    
    def _cleanup_archives(self) -> None:
        """Clean up old archives to prevent memory bloat."""
        max_archive_size = self.config.interval_generations * 100
        
        # Keep only recent entries
        if len(self.genome_archive) > max_archive_size:
            self.genome_archive = self.genome_archive[-max_archive_size:]
        
        if len(self.success_archive) > max_archive_size // 2:
            self.success_archive = self.success_archive[-max_archive_size // 2:]
        
        if len(self.failure_archive) > max_archive_size // 2:
            self.failure_archive = self.failure_archive[-max_archive_size // 2:]
    
    def get_insights(self, min_confidence: float = 0.5) -> list[Insight]:
        """Get insights above confidence threshold."""
        return [i for i in self.insights if i.confidence >= min_confidence]
    
    def get_strategies(self, min_confidence: float = 0.5) -> list[Strategy]:
        """Get strategies above confidence threshold."""
        return [s for s in self.strategies if s.confidence >= min_confidence]
    
    def get_recommended_strategies(self) -> list[Strategy]:
        """Get strategies recommended for current state."""
        strategies = self.get_strategies(min_confidence=0.6)
        
        # Sort by expected improvement
        strategies.sort(key=lambda s: s.expected_improvement, reverse=True)
        
        return strategies[:3]  # Top 3 strategies
    
    def get_statistics(self) -> dict:
        """Get dream loop statistics."""
        return {
            "dream_count": self.dream_count,
            "last_dream_time": self.last_dream_time,
            "current_phase": self.phase.name,
            "archives_size": {
                "total": len(self.genome_archive),
                "success": len(self.success_archive),
                "failure": len(self.failure_archive),
            },
            "insights_count": len(self.insights),
            "strategies_count": len(self.strategies),
            "avg_dream_duration_ms": (
                sum(r.duration_ms for r in self.dream_history) / len(self.dream_history)
                if self.dream_history else 0
            ),
        }
