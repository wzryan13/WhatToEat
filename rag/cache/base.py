# rag/cache/base.py
"""
Cache backend abstractions.

- KeywordCacheBackend: key/value semantics (for exact match, e.g., Redis L1).
- VectorCacheBackend: vector insert/search semantics (for semantic match, e.g., in-memory or future Milvus).
"""
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple


class KeywordCacheBackend(ABC):
    """Abstract base class for keyword-based cache backends (exact match)."""
    
    @abstractmethod
    async def get(self, key: str):
        """Get a value by key."""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: bytes, ttl_seconds: int | None = None) -> bool:
        """Set a value with optional TTL."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value by key."""
        pass
    
    @abstractmethod
    async def clear(self, pattern: str | None = None) -> bool:
        """Clear cache entries matching pattern."""
        pass


class VectorCacheBackend(ABC):
    """Abstract base class for vector-based cache backends (semantic similarity)."""
    
    @abstractmethod
    async def add(
        self,
        key: str,
        embedding: List[float],
        payload: Any,
        ttl_seconds: int | None = None,
        scope: str | None = None,
    ) -> bool:
        """Add a vector with payload to the cache, optionally expiring it.
        
        Args:
            key: Unique cache key
            embedding: Vector embedding for semantic search
            payload: Data to cache
            ttl_seconds: Optional TTL for cache expiration
            scope: Optional scope identifier (e.g., user_id) for isolation
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        embedding: List[float],
        threshold: float,
        scope: str | None = None,
    ) -> Optional[Tuple[Any, float]]:
        """Search for similar vectors, returning (payload, similarity_score) if found.
        
        Args:
            embedding: Query vector embedding
            threshold: Minimum similarity threshold
            scope: Optional scope to filter results (e.g., user_id)
        """
        pass
    
    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cached vectors."""
        pass

