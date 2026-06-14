# rag/rag_service.py
"""
RAG 服务编排层 — 统一调度检索流程（纯相关性，不碰画像）。

Pipeline:
1. Cache check (L1 精确 → L2 语义)
2. Query Rewrite (LLM 改写)
3. Metadata Filter (LLM 生成 Milvus expr)
4. Vector Search (top_k=20)
5. Rerank (SiliconFlow cross-encoder, top_k=10)
6. Cache store
7. Post-process (按 parent_id 去重)
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from rag.cache import CacheManager
from rag.pipeline.query_understanding import QueryUnderstandingModule
from rag.pipeline.retrieval import RetrievalOptimizationModule
from rag.pipeline.document_processor import document_processor
from rag.rerankers.siliconflow_reranker import SiliconFlowReranker
from rag.vector_stores.vector_store_factory import get_vector_store

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG 检索服务 — 纯相关性检索，不包含画像个性化逻辑。
    画像相关的硬过滤和偏好重排由 rag_formatter 节点负责。
    """

    def __init__(
        self,
        vector_store,
        embeddings: Embeddings,
        cache: CacheManager | None = None,
    ):
        self.cache = cache
        self.retrieval = RetrievalOptimizationModule(
            vector_store=vector_store,
            score_threshold=settings.RAG_SCORE_THRESHOLD,
        )
        self.query_understanding = QueryUnderstandingModule()
        self.reranker = SiliconFlowReranker()
        self.embeddings = embeddings

    async def search_recipes(
        self,
        query: str,
        metadata_catalog: Optional[dict] = None,
        extra_expr: Optional[str] = None,
        ranker_type: Optional[str] = None,
        ranker_params: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        执行完整的 RAG 检索流程。

        Args:
            query: 用户原始查询
            metadata_catalog: 可用元数据值（用于 LLM 生成过滤表达式）
            extra_expr: 额外的 Milvus 过滤表达式（如 category 粗过滤）
            ranker_type: hybrid ranker 类型，可选 `rrf` 或 `weighted`
            ranker_params: ranker 参数，如 `{\"k\": 60}` 或 `{\"weights\": [0.5, 0.5]}`

        Returns:
            检索并重排序后的文档列表
        """
        # Step 2+3: Query Understanding（单次 LLM：改写 + 元数据过滤表达式）
        rewritten_query, llm_expr = await self.query_understanding.understand(
            query=query,
            metadata_catalog=metadata_catalog,
        )

        # Step 1: Cache check (L1 精确 → L2 语义)
        filter_scope = hashlib.sha256(extra_expr.encode()).hexdigest()[:16] if extra_expr else None
        if self.cache:
            cached = await self.cache.get("recipe", rewritten_query, scope=filter_scope)
            if cached:
                logger.info("RAG 缓存命中: %d 条文档", len(cached))
                return cached

        # 合并 extra_expr 和 llm_expr
        final_expr = self._merge_expressions(extra_expr, llm_expr)

        # Step 4: Vector Search
        docs, scores = await self.retrieval.hybrid_search(
            query=rewritten_query,
            top_k=settings.RAG_TOP_K,
            expr=final_expr,
            ranker_type=ranker_type,
            ranker_params=ranker_params,
        )

        if not docs:
            logger.info("向量检索无结果")
            return []

        # 将检索分数写入 metadata
        for doc, score in zip(docs, scores):
            doc.metadata["retrieval_score"] = score

        # Step 5: Rerank (纯相关性，不注入偏好)
        reranked_docs = await self.reranker.rerank(
            query=rewritten_query,
            documents=docs,
        )

        # 截取 top_k
        reranked_docs = reranked_docs[: settings.RAG_RERANK_TOP_K]

        # Step 7: Post-process (按 parent_id 去重)
        final_docs = await document_processor.post_process_retrieval(reranked_docs)

        # Cache store (rerank 后、LLM 软偏好前)
        if self.cache and final_docs:
            await self.cache.set("recipe", rewritten_query, final_docs, scope=filter_scope)

        logger.info("RAG 检索完成: %d 条结果", len(final_docs))
        return final_docs

    @staticmethod
    def _merge_expressions(expr1: Optional[str], expr2: Optional[str]) -> Optional[str]:
        """合并两个 Milvus 过滤表达式（AND 连接）。"""
        if expr1 and expr2:
            return f"({expr1}) and ({expr2})"
        return expr1 or expr2


# ── 全局实例（延迟初始化） ────────────────────���─────────────

_rag_service: Optional[RAGService] = None


def get_rag_service() -> Optional[RAGService]:
    """获取 RAG 服务单例。"""
    return _rag_service


def init_rag_service() -> Optional[RAGService]:
    """
    初始化 RAG 服务（创建 embedding 模型、连接 Milvus）。
    在 main.py 启动时调用一次。
    """
    global _rag_service

    if not settings.RAG_ENABLED:
        logger.info("RAG 未启用 (RAG_ENABLED=false)")
        return None

    try:
        from rag.embeddings.embedding_factory import get_embedding_model

        # 初始化 embedding 模型
        embeddings = get_embedding_model(settings.EMBEDDING_MODEL)

        vector_store = get_vector_store(
            uri=settings.MILVUS_URI,
            collection_name=settings.MILVUS_COLLECTION,
            embeddings=embeddings,
        )
        logger.info("Milvus hybrid collection 已就绪: %s", settings.MILVUS_COLLECTION)

        cache = None
        try:
            parsed_uri = urlparse(settings.MILVUS_URI)
            cache = CacheManager(
                redis_host=settings.REDIS_HOST,
                redis_port=settings.REDIS_PORT,
                redis_password=settings.REDIS_PASSWORD or None,
                ttl=settings.CACHE_TTL_RECIPE,
                similarity_threshold=settings.CACHE_SIMILARITY_THRESHOLD,
                embeddings=embeddings,
                l2_enabled=settings.CACHE_L2_ENABLED,
                vector_host=parsed_uri.hostname or "127.0.0.1",
                vector_port=parsed_uri.port or 19530,
            )
            logger.info("RAG 缓存初始化成功")
        except Exception as e:
            logger.warning("RAG 缓存初始化失败，将跳过缓存: %s", e)

        _rag_service = RAGService(
            vector_store=vector_store,
            embeddings=embeddings,
            cache=cache,
        )
        logger.info("RAG 服务初始化成功")
        return _rag_service

    except Exception as e:
        logger.error(f"RAG 服务初始化失败: {e}")
        return None
