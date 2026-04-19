"""
Storage Backend Module - Persistence for knowledge base.

Provides pluggable storage backends:
- JSON file storage
- SQLite storage
- In-memory storage (for testing)
"""

from __future__ import annotations
import json
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Iterator
from contextlib import contextmanager

from .base import Pattern


class StorageBackend(ABC):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    def load_all(self) -> list[Pattern]:
        """Load all patterns from storage."""
        pass
    
    @abstractmethod
    def save(self, pattern: Pattern) -> None:
        """Save a pattern to storage."""
        pass
    
    @abstractmethod
    def load(self, pattern_id: str) -> Optional[Pattern]:
        """Load a single pattern by ID."""
        pass
    
    @abstractmethod
    def delete(self, pattern_id: str) -> bool:
        """Delete a pattern from storage."""
        pass
    
    @abstractmethod
    def exists(self, pattern_id: str) -> bool:
        """Check if pattern exists."""
        pass


class JSONStorageBackend(StorageBackend):
    """JSON file-based storage backend."""
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize JSON storage backend."""
        self.storage_dir = storage_dir or Path("knowledge_base")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.storage_dir / "index.json"
        self.patterns_dir = self.storage_dir / "patterns"
        self.patterns_dir.mkdir(exist_ok=True)
        
        # Initialize index
        if not self.index_file.exists():
            self._save_index({})
    
    def _load_index(self) -> dict:
        """Load pattern index."""
        if not self.index_file.exists():
            return {}
        
        with open(self.index_file, 'r') as f:
            return json.load(f)
    
    def _save_index(self, index: dict) -> None:
        """Save pattern index."""
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def load_all(self) -> list[Pattern]:
        """Load all patterns from storage."""
        index = self._load_index()
        patterns = []
        
        for pattern_id in index.keys():
            pattern = self.load(pattern_id)
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def save(self, pattern: Pattern) -> None:
        """Save a pattern to storage."""
        # Save pattern data
        pattern_file = self.patterns_dir / f"{pattern.id}.json"
        with open(pattern_file, 'w') as f:
            json.dump(pattern.to_dict(), f, indent=2)
        
        # Update index
        index = self._load_index()
        index[pattern.id] = {
            "name": pattern.name,
            "type": pattern.pattern_type.name,
            "file": f"{pattern.id}.json",
        }
        self._save_index(index)
    
    def load(self, pattern_id: str) -> Optional[Pattern]:
        """Load a single pattern by ID."""
        pattern_file = self.patterns_dir / f"{pattern_id}.json"
        
        if not pattern_file.exists():
            return None
        
        with open(pattern_file, 'r') as f:
            data = json.load(f)
        
        return Pattern.from_dict(data)
    
    def delete(self, pattern_id: str) -> bool:
        """Delete a pattern from storage."""
        pattern_file = self.patterns_dir / f"{pattern_id}.json"
        
        if pattern_file.exists():
            pattern_file.unlink()
            
            # Update index
            index = self._load_index()
            if pattern_id in index:
                del index[pattern_id]
                self._save_index(index)
            
            return True
        
        return False
    
    def exists(self, pattern_id: str) -> bool:
        """Check if pattern exists."""
        return (self.patterns_dir / f"{pattern_id}.json").exists()


class SQLiteStorageBackend(StorageBackend):
    """SQLite-based storage backend."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize SQLite storage backend."""
        self.db_path = db_path or Path("knowledge_base.db")
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS patterns (
                    id TEXT PRIMARY KEY,
                    data TEXT NOT NULL,
                    created_at REAL,
                    updated_at REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON patterns(created_at)
            """)
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def load_all(self) -> list[Pattern]:
        """Load all patterns from storage."""
        patterns = []
        
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT data FROM patterns")
            for row in cursor:
                data = json.loads(row['data'])
                patterns.append(Pattern.from_dict(data))
        
        return patterns
    
    def save(self, pattern: Pattern) -> None:
        """Save a pattern to storage."""
        import time
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO patterns (id, data, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (
                pattern.id,
                json.dumps(pattern.to_dict()),
                pattern.created_at,
                time.time(),
            ))
            conn.commit()
    
    def load(self, pattern_id: str) -> Optional[Pattern]:
        """Load a single pattern by ID."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT data FROM patterns WHERE id = ?",
                (pattern_id,)
            )
            row = cursor.fetchone()
            
            if row:
                return Pattern.from_dict(json.loads(row['data']))
        
        return None
    
    def delete(self, pattern_id: str) -> bool:
        """Delete a pattern from storage."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM patterns WHERE id = ?",
                (pattern_id,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def exists(self, pattern_id: str) -> bool:
        """Check if pattern exists."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM patterns WHERE id = ?",
                (pattern_id,)
            )
            return cursor.fetchone() is not None


class InMemoryStorageBackend(StorageBackend):
    """In-memory storage for testing."""
    
    def __init__(self):
        """Initialize in-memory storage."""
        self._patterns: dict[str, Pattern] = {}
    
    def load_all(self) -> list[Pattern]:
        """Load all patterns from storage."""
        return list(self._patterns.values())
    
    def save(self, pattern: Pattern) -> None:
        """Save a pattern to storage."""
        self._patterns[pattern.id] = pattern
    
    def load(self, pattern_id: str) -> Optional[Pattern]:
        """Load a single pattern by ID."""
        return self._patterns.get(pattern_id)
    
    def delete(self, pattern_id: str) -> bool:
        """Delete a pattern from storage."""
        if pattern_id in self._patterns:
            del self._patterns[pattern_id]
            return True
        return False
    
    def exists(self, pattern_id: str) -> bool:
        """Check if pattern exists."""
        return pattern_id in self._patterns
    
    def clear(self) -> None:
        """Clear all patterns."""
        self._patterns.clear()
