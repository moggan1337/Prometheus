"""
Knowledge Base Module - Stores and retrieves code patterns.

The Knowledge Base stores:
- Successful code patterns
- Anti-patterns (what NOT to do)
- Optimization patterns
- Refactoring patterns
- Performance insights
"""

from __future__ import annotations
import json
import time
import hashlib
import re
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Any, Iterator
from pathlib import Path
import numpy as np

from ..core.genome import Genome
from .storage import StorageBackend, JSONStorageBackend


class PatternType(Enum):
    """Types of patterns stored in knowledge base."""
    STRUCTURE = auto()      # Code structure patterns
    ALGORITHM = auto()     # Algorithm patterns
    OPTIMIZATION = auto()  # Performance patterns
    REFACTOR = auto()      # Refactoring patterns
    ANTIPATTERN = auto()   # Anti-patterns to avoid
    HEURISTIC = auto()     # Problem-solving heuristics
    TEMPLATE = auto()      # Code templates


@dataclass
class Pattern:
    """
    Represents a code pattern in the knowledge base.
    
    Patterns capture successful code solutions that can be
    reused and evolved in future generations.
    """
    id: str
    name: str
    pattern_type: PatternType
    
    # Content
    source: str
    description: str
    code_context: str = ""
    
    # Metadata
    language: str = "python"
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    
    # Effectiveness metrics
    success_rate: float = 0.5  # How often this pattern leads to success
    avg_fitness: float = 0.0   # Average fitness when used
    fitness_variance: float = 0.0
    
    # Applicability
    applicability: str = ""  # When is this pattern applicable?
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)
    
    # Relationships
    related_patterns: list[str] = field(default_factory=list)
    parent_pattern: Optional[str] = None
    
    # Tags for retrieval
    tags: list[str] = field(default_factory=list)
    
    # Evolution tracking
    generation: int = 0
    lineage: list[str] = field(default_factory=list)
    
    @property
    def hash(self) -> str:
        """Get pattern hash for deduplication."""
        content = f"{self.pattern_type.name}:{self.source}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @property
    def complexity(self) -> float:
        """Calculate pattern complexity."""
        return len(self.source) / 1000.0  # Normalized by 1000 chars
    
    @property
    def effectiveness(self) -> float:
        """Calculate overall effectiveness score."""
        return (
            self.success_rate * 0.4 +
            min(1.0, self.avg_fitness) * 0.3 +
            (1.0 - min(1.0, self.fitness_variance)) * 0.3
        )
    
    def use(self, fitness: Optional[float] = None) -> None:
        """Record pattern usage."""
        self.use_count += 1
        self.last_used = time.time()
        
        if fitness is not None:
            # Update rolling statistics
            n = self.use_count
            old_avg = self.avg_fitness
            self.avg_fitness = old_avg + (fitness - old_avg) / n
            
            # Update variance (simplified)
            if n > 1:
                self.fitness_variance = (
                    (n - 2) / (n - 1) * self.fitness_variance +
                    (fitness - old_avg) ** 2 / n
                )
    
    def to_dict(self) -> dict:
        """Serialize pattern to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "pattern_type": self.pattern_type.name,
            "source": self.source,
            "description": self.description,
            "code_context": self.code_context,
            "language": self.language,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "use_count": self.use_count,
            "success_rate": self.success_rate,
            "avg_fitness": self.avg_fitness,
            "fitness_variance": self.fitness_variance,
            "applicability": self.applicability,
            "preconditions": self.preconditions,
            "postconditions": self.postconditions,
            "related_patterns": self.related_patterns,
            "parent_pattern": self.parent_pattern,
            "tags": self.tags,
            "generation": self.generation,
            "lineage": self.lineage,
            "hash": self.hash,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> Pattern:
        """Deserialize pattern from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            pattern_type=PatternType[data.get("pattern_type", "STRUCTURE")],
            source=data["source"],
            description=data.get("description", ""),
            code_context=data.get("code_context", ""),
            language=data.get("language", "python"),
            created_at=data.get("created_at", time.time()),
            last_used=data.get("last_used", time.time()),
            use_count=data.get("use_count", 0),
            success_rate=data.get("success_rate", 0.5),
            avg_fitness=data.get("avg_fitness", 0.0),
            fitness_variance=data.get("fitness_variance", 0.0),
            applicability=data.get("applicability", ""),
            preconditions=data.get("preconditions", []),
            postconditions=data.get("postconditions", []),
            related_patterns=data.get("related_patterns", []),
            parent_pattern=data.get("parent_pattern"),
            tags=data.get("tags", []),
            generation=data.get("generation", 0),
            lineage=data.get("lineage", []),
        )
    
    def __str__(self) -> str:
        return f"Pattern({self.name}, type={self.pattern_type.name}, effectiveness={self.effectiveness:.2f})"
    
    def __repr__(self) -> str:
        return self.__str__()


class KnowledgeBase:
    """
    Stores and retrieves code patterns for evolutionary guidance.
    
    The Knowledge Base provides:
    - Pattern storage and retrieval
    - Similarity-based pattern matching
    - Pattern effectiveness tracking
    - Automatic pattern extraction
    """
    
    def __init__(
        self, 
        storage: Optional[StorageBackend] = None,
        config: Optional[dict] = None
    ):
        """Initialize knowledge base."""
        self.storage = storage or JSONStorageBackend()
        self.config = config or self._default_config()
        
        # In-memory index for fast access
        self._patterns: dict[str, Pattern] = {}
        self._index_by_type: dict[PatternType, list[str]] = {pt: [] for pt in PatternType}
        self._index_by_tag: dict[str, list[str]] = {}
        self._similarity_index: dict[str, list[tuple[str, float]]] = {}
        
        # Load existing patterns
        self._load()
    
    def _default_config(self) -> dict:
        """Get default configuration."""
        return {
            "max_patterns": 10000,
            "similarity_threshold": 0.7,
            "effectiveness_threshold": 0.5,
            "auto_extraction": True,
            "pattern_aging": True,
            "max_age_days": 365,
        }
    
    def _load(self) -> None:
        """Load patterns from storage."""
        patterns = self.storage.load_all()
        for pattern in patterns:
            self._add_to_index(pattern)
    
    def _add_to_index(self, pattern: Pattern) -> None:
        """Add pattern to in-memory indexes."""
        self._patterns[pattern.id] = pattern
        
        # Index by type
        self._index_by_type[pattern.pattern_type].append(pattern.id)
        
        # Index by tags
        for tag in pattern.tags:
            if tag not in self._index_by_tag:
                self._index_by_tag[tag] = []
            self._index_by_tag[tag].append(pattern.id)
        
        # Build similarity index (lazy)
        self._similarity_index.clear()
    
    # Core operations
    
    def add(self, pattern: Pattern) -> None:
        """Add a pattern to the knowledge base."""
        # Check capacity
        if len(self._patterns) >= self.config["max_patterns"]:
            self._evict_old_patterns()
        
        # Add to storage
        self.storage.save(pattern)
        
        # Add to index
        self._add_to_index(pattern)
        
        # Update similarity index
        self._update_similarity_index(pattern)
    
    def get(self, pattern_id: str) -> Optional[Pattern]:
        """Get pattern by ID."""
        return self._patterns.get(pattern_id)
    
    def remove(self, pattern_id: str) -> bool:
        """Remove pattern from knowledge base."""
        if pattern_id not in self._patterns:
            return False
        
        pattern = self._patterns[pattern_id]
        
        # Remove from storage
        self.storage.delete(pattern_id)
        
        # Remove from indexes
        del self._patterns[pattern_id]
        self._index_by_type[pattern.pattern_type].remove(pattern_id)
        
        for tag in pattern.tags:
            if tag in self._index_by_tag:
                self._index_by_tag[tag].remove(pattern_id)
        
        self._similarity_index.clear()
        
        return True
    
    # Query operations
    
    def query(
        self,
        pattern_type: Optional[PatternType] = None,
        tags: Optional[list[str]] = None,
        min_effectiveness: float = 0.0,
        limit: int = 10
    ) -> list[Pattern]:
        """
        Query patterns by criteria.
        
        Args:
            pattern_type: Filter by pattern type
            tags: Filter by tags
            min_effectiveness: Minimum effectiveness score
            limit: Maximum results
            
        Returns:
            List of matching patterns
        """
        candidates = set(self._patterns.keys())
        
        # Filter by type
        if pattern_type:
            type_ids = set(self._index_by_type.get(pattern_type, []))
            candidates &= type_ids
        
        # Filter by tags
        if tags:
            tag_ids = set()
            for tag in tags:
                tag_ids |= set(self._index_by_tag.get(tag, []))
            candidates &= tag_ids
        
        # Filter by effectiveness
        results = []
        for pid in candidates:
            pattern = self._patterns[pid]
            if pattern.effectiveness >= min_effectiveness:
                results.append(pattern)
        
        # Sort by effectiveness and limit
        results.sort(key=lambda p: p.effectiveness, reverse=True)
        return results[:limit]
    
    def find_similar(
        self, 
        source: str, 
        threshold: Optional[float] = None,
        limit: int = 5
    ) -> list[tuple[Pattern, float]]:
        """
        Find patterns similar to source code.
        
        Args:
            source: Source code to compare
            threshold: Minimum similarity (0-1)
            limit: Maximum results
            
        Returns:
            List of (pattern, similarity) tuples
        """
        threshold = threshold or self.config["similarity_threshold"]
        
        # Lazy build similarity index
        if not self._similarity_index:
            self._build_similarity_index()
        
        source_hash = self._compute_code_hash(source)
        
        # Get similar from index
        similar = self._similarity_index.get(source_hash, [])
        
        # Filter and return
        return [(self._patterns[pid], sim) 
                for pid, sim in similar 
                if sim >= threshold][:limit]
    
    def find_by_context(
        self, 
        context: str, 
        limit: int = 5
    ) -> list[Pattern]:
        """Find patterns matching code context."""
        context_lower = context.lower()
        
        results = []
        for pattern in self._patterns.values():
            if context_lower in pattern.code_context.lower():
                results.append(pattern)
            elif any(word in pattern.source.lower() 
                    for word in context.split()[:5]):
                results.append(pattern)
        
        results.sort(key=lambda p: p.effectiveness, reverse=True)
        return results[:limit]
    
    # Pattern extraction
    
    def extract_from_genome(self, genome: Genome) -> Optional[Pattern]:
        """
        Extract a pattern from a successful genome.
        
        Args:
            genome: Source genome
            
        Returns:
            Extracted pattern or None
        """
        if genome.fitness.composite < 0.7:
            return None
        
        # Generate pattern name
        name = self._generate_pattern_name(genome)
        
        # Determine pattern type
        pattern_type = self._classify_pattern(genome)
        
        pattern = Pattern(
            id=f"pattern_{int(time.time() * 1000)}",
            name=name,
            pattern_type=pattern_type,
            source=genome.source,
            description=f"Extracted from genome {genome.id[:8]}",
            language=genome.language,
            generation=genome.generation,
            avg_fitness=genome.fitness.composite,
            success_rate=genome.fitness.correctness,
            tags=self._extract_tags(genome),
        )
        
        return pattern
    
    def _generate_pattern_name(self, genome: Genome) -> str:
        """Generate descriptive pattern name."""
        # Extract function/class name
        import re
        match = re.search(r'def\s+(\w+)', genome.source)
        if match:
            return f"{match.group(1)}_pattern"
        
        match = re.search(r'class\s+(\w+)', genome.source)
        if match:
            return f"{match.group(1)}_class_pattern"
        
        return f"pattern_{genome.hash}"
    
    def _classify_pattern(self, genome: Genome) -> PatternType:
        """Classify pattern type based on content."""
        source = genome.source.lower()
        
        if 'sort' in source or 'search' in source or 'binary' in source:
            return PatternType.ALGORITHM
        if 'cache' in source or 'memo' in source or 'optimize' in source:
            return PatternType.OPTIMIZATION
        if 'refactor' in source or 'extract' in source:
            return PatternType.REFACTOR
        if 'try' in source or 'except' in source or 'error' in source:
            return PatternType.ANTIPATTERN
        if 'def ' in source:
            return PatternType.STRUCTURE
        if 'class ' in source:
            return PatternType.STRUCTURE
        
        return PatternType.HEURISTIC
    
    def _extract_tags(self, genome: Genome) -> list[str]:
        """Extract tags from genome."""
        tags = []
        source = genome.source.lower()
        
        # Keyword-based tags
        keywords = {
            'recursive': 'recursion',
            'loop': 'iteration',
            'sort': 'sorting',
            'search': 'searching',
            'cache': 'caching',
            'async': 'async',
            'class': 'oop',
            'dict': 'data-structure',
            'list': 'data-structure',
            'set': 'data-structure',
        }
        
        for keyword, tag in keywords.items():
            if keyword in source:
                tags.append(tag)
        
        return tags
    
    # Similarity computation
    
    def _compute_code_hash(self, source: str) -> str:
        """Compute hash for similarity comparison."""
        # Normalize source
        normalized = self._normalize_for_similarity(source)
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]
    
    def _normalize_for_similarity(self, source: str) -> str:
        """Normalize source for similarity comparison."""
        import re
        
        # Remove comments
        source = re.sub(r'#.*$', '', source, flags=re.MULTILINE)
        source = re.sub(r'""".*?"""', '', source, flags=re.DOTALL)
        source = re.sub(r"'''.'''", '', source, flags=re.DOTALL)
        
        # Normalize whitespace
        source = re.sub(r'\s+', ' ', source)
        
        # Normalize variable names (optional - more sophisticated)
        # source = re.sub(r'\b[a-z]\w*\b', 'VAR', source)
        
        return source.strip().lower()
    
    def _build_similarity_index(self) -> None:
        """Build similarity index for all patterns."""
        pattern_hashes = {}
        
        for pattern in self._patterns.values():
            code_hash = self._compute_code_hash(pattern.source)
            
            # Store hash for later comparison
            if code_hash not in pattern_hashes:
                pattern_hashes[code_hash] = []
            pattern_hashes[code_hash].append(pattern.id)
            
            # Precompute character n-grams
            ngrams = self._compute_ngrams(pattern.source, n=3)
            
            self._similarity_index[pattern.id] = {
                'hash': code_hash,
                'ngrams': ngrams,
            }
    
    def _compute_ngrams(self, text: str, n: int = 3) -> set:
        """Compute character n-grams."""
        normalized = self._normalize_for_similarity(text)
        return set(normalized[i:i+n] for i in range(len(normalized) - n + 1))
    
    def _update_similarity_index(self, pattern: Pattern) -> None:
        """Update similarity index with new pattern."""
        # This would update the index, but for simplicity we clear and rebuild
        self._similarity_index.clear()
    
    # Maintenance
    
    def _evict_old_patterns(self) -> None:
        """Evict old/ineffective patterns when capacity is reached."""
        # Sort by effectiveness and age
        candidates = list(self._patterns.values())
        candidates.sort(key=lambda p: (p.effectiveness, p.last_used))
        
        # Remove bottom 10%
        to_remove = len(candidates) // 10
        for pattern in candidates[:to_remove]:
            self.remove(pattern.id)
    
    def cleanup(self) -> int:
        """Remove stale patterns."""
        if not self.config.get("pattern_aging"):
            return 0
        
        max_age = self.config.get("max_age_days", 365) * 86400
        current_time = time.time()
        
        to_remove = []
        for pattern in self._patterns.values():
            age = current_time - pattern.last_used
            if age > max_age and pattern.use_count < 5:
                to_remove.append(pattern.id)
        
        for pid in to_remove:
            self.remove(pid)
        
        return len(to_remove)
    
    def get_statistics(self) -> dict:
        """Get knowledge base statistics."""
        patterns = list(self._patterns.values())
        
        return {
            "total_patterns": len(patterns),
            "by_type": {
                pt.name: len(self._index_by_type[pt])
                for pt in PatternType
            },
            "by_tag": {
                tag: len(pids)
                for tag, pids in self._index_by_tag.items()
            },
            "avg_effectiveness": np.mean([p.effectiveness for p in patterns]) if patterns else 0,
            "avg_use_count": np.mean([p.use_count for p in patterns]) if patterns else 0,
            "total_uses": sum(p.use_count for p in patterns),
        }
    
    def __len__(self) -> int:
        """Get number of patterns."""
        return len(self._patterns)
    
    def __iter__(self) -> Iterator[Pattern]:
        """Iterate over patterns."""
        return iter(self._patterns.values())
