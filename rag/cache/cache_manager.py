# rag/cache/cache_manager.py
"""
两级缓存管理器 — 同时服务 RAG 菜谱检索和 search_agent 餐厅搜索。

缓存策略:
- L1 (精确匹配): Redis, key = hash(query), value = pickle 序列化对象
- L2 (语义匹配): Milvus, 向量相似度搜索近似查询

只缓存检索结果（rerank 前），不缓存 LLM 个性化输出。
"""
import hashlib
import logging
import pickle
from typing import Any, List, Optional

import redis.asyncio as redis
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from rag.cache.base import KeywordCacheBackend, VectorCacheBackend
from rag.cache.backends import MilvusVectorCache, RedisKeywordCache

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages two-level caching for RAG retrieval results.
    
    L1 Cache (Redis - Exact Match):
    - Fast lookup for identical queries
    - Key: hash of the rewritten query
    - Value: serialized list of retrieved documents
    
    L2 Cache (Milvus - Semantic Match):
    - Handles query variations with high similarity
    - Uses vector embeddings for similarity search
    - Falls back when L1 cache misses but similar query exists
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        ttl: int = 3600,
        similarity_threshold: float = 0.92,
        embeddings: Optional[Embeddings] = None,
        l2_enabled: bool = True,
        vector_host: Optional[str] = None,
        vector_port: Optional[int] = None,
        vector_collection: str = "cookhero_retrieval_cache",
        vector_user: Optional[str] = None,
        vector_password: Optional[str] = None,
        vector_secure: bool = False,
    ):
        """
        Initialize the cache manager.
        
        Args:
            redis_host: Redis host address
            redis_port: Redis port
            redis_db: Redis database number
            redis_password: Redis password (if required)
            ttl: Time-to-live for cache entries (seconds)
            similarity_threshold: Minimum similarity for L2 cache matching (0-1)
            embeddings: Embedding model for L2 semantic matching
            l2_enabled: Whether L2 semantic cache is enabled
            vector_host/vector_port: Milvus connection info
            vector_collection: Milvus collection name for cache
            vector_user/vector_password: Optional Milvus credentials
            vector_secure: Whether to use TLS for Milvus
        """
        self.ttl = ttl
        self.similarity_threshold = similarity_threshold
        self.l2_enabled = l2_enabled
        self.embeddings = embeddings
        
        # Initialize Redis connection (L1 cache)
        self.redis_client: Optional[redis.Redis] = None
        self.keyword_cache: Optional[KeywordCacheBackend] = None
        try:
            client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=False,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            self.redis_client = client
            self.keyword_cache = RedisKeywordCache(client)
            logger.info(f"Redis L1 cache connected: {redis_host}:{redis_port}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. L1 cache disabled.")
        
        # Initialize Milvus connection (L2 cache)
        self.vector_cache: Optional[VectorCacheBackend] = None
        self._embedding_dimension: Optional[int] = None
        
        if self.l2_enabled and self.embeddings:
            self._embedding_dimension = self._infer_embedding_dimension()
            if self._embedding_dimension:
                host = vector_host or redis_host
                port = vector_port or 19530
                try:
                    self.vector_cache = MilvusVectorCache(
                        host=host,
                        port=port,
                        collection_name=vector_collection,
                        dimension=self._embedding_dimension,
                        user=vector_user,
                        password=vector_password,
                        secure=vector_secure,
                    )
                    logger.info(f"Milvus L2 cache connected: {host}:{port} (collection={vector_collection})")
                except Exception as exc:
                    logger.warning(f"Failed to initialize Milvus L2 cache: {exc}")
                    self.l2_enabled = False
            else:
                logger.warning("Could not infer embedding dimension. L2 cache disabled.")
                self.l2_enabled = False
        elif self.l2_enabled:
            logger.warning("Embeddings not provided. L2 cache disabled.")
            self.l2_enabled = False
    
    def _compute_hash(self, text: str) -> str:
        """Compute SHA256 hash of a text string."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _get_cache_key(self, data_source: str, query: str, scope: str | None = None) -> str:
        """Generate cache key for retrieval results."""
        query_hash = self._compute_hash(query)
        scope_label = scope or "global"
        return f"rag:retrieval:{data_source}:{scope_label}:{query_hash}"
    
    async def get(
        self,
        data_source: str,
        query: str,
        scope: str | None = None,
    ) -> Optional[Any]:
        """
        从两级缓存获取检索结果。

        Args:
            data_source: 数据源标识（"recipe" 或 "restaurant"）
            query: 查询字符串

        Returns:
            缓存的对象（Document 列表或 POI 列表），未命中则 None
        """
        # Try L1 cache first (exact match)
        if self.keyword_cache:
            try:
                cache_key = self._get_cache_key(data_source, query, scope)
                cached_data = await self.keyword_cache.get(cache_key)
                
                if cached_data:
                    docs = pickle.loads(cached_data)
                    logger.info(f"L1 cache HIT for '{data_source}': {len(docs)} documents")
                    return docs
            except Exception as e:
                logger.warning(f"Error reading L1 cache: {e}")
        
        # Try L2 cache (semantic similarity)
        if self._should_use_l2():
            try:
                query_embedding = self.embeddings.embed_query(query)  # type: ignore
                result = await self.vector_cache.search(  # type: ignore
                    query_embedding,
                    self.similarity_threshold,
                    scope=scope,
                )
                
                if result:
                    cached_data, similarity = result
                    if cached_data:
                        docs = pickle.loads(cached_data)
                        logger.info(f"L2 cache HIT for '{data_source}': similarity={similarity:.4f}, {len(docs)} documents")
                        return docs
            except Exception as e:
                logger.warning(f"Error reading L2 cache: {e}")
        
        logger.debug(f"Cache MISS for '{data_source}'")
        return None
    
    async def set(
        self,
        data_source: str,
        query: str,
        documents: Any,
        scope: str | None = None,
    ) -> bool:
        """
        将检索结果写入两级缓存。

        Args:
            data_source: 数据源标识（"recipe" 或 "restaurant"）
            query: 查询字符串
            documents: 待缓存对象（Document 列表或 POI 列表）

        Returns:
            是否缓存成功
        """
        serialized = pickle.dumps(documents)
        success = True
        
        # Store in L1 cache
        if self.keyword_cache:
            try:
                cache_key = self._get_cache_key(data_source, query, scope)
                stored = await self.keyword_cache.set(cache_key, serialized, ttl_seconds=self.ttl)
                if stored:
                    logger.info(f"L1 cache SET for '{data_source}': {len(documents)} documents (TTL={self.ttl}s)")
                else:
                    success = False
            except Exception as e:
                logger.warning(f"Error writing L1 cache: {e}")
                success = False
        
        # Store in L2 cache
        if self._should_use_l2():
            try:
                query_embedding = self.embeddings.embed_query(query)  # type: ignore
                scoped = scope or "global"
                cache_key = self._compute_hash(f"{data_source}:{scoped}:{query}")
                stored = await self.vector_cache.add(  # type: ignore
                    cache_key,
                    query_embedding,
                    serialized,
                    ttl_seconds=self.ttl,
                    scope=scope,
                )
                if stored:
                    logger.info(f"L2 cache SET for '{data_source}': semantic index updated")
                else:
                    success = False
            except Exception as e:
                logger.warning(f"Error writing L2 cache: {e}")
                success = False
        
        return success
    
    def _infer_embedding_dimension(self) -> Optional[int]:
        """Infer embedding dimension by running a test query."""
        if not self.embeddings:
            return None
        try:
            probe = self.embeddings.embed_query("test query for dimension")
            return len(probe)
        except Exception as exc:
            logger.warning(f"Failed to infer embedding dimension: {exc}")
            return None
    
    def _should_use_l2(self) -> bool:
        """Check if L2 cache should be used."""
        return bool(self.l2_enabled and self.vector_cache and self.embeddings)