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

import logging
from typing import List, Optional

from pymilvus import MilvusClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from config.settings import settings
from rag.pipeline.generation import GenerationIntegrationModule
from rag.pipeline.metadata_filter import MetadataFilterExtractor
from rag.pipeline.retrieval import RetrievalOptimizationModule
from rag.pipeline.document_processor import document_processor
from rag.rerankers.siliconflow_reranker import SiliconFlowReranker

logger = logging.getLogger(__name__)


class RAGService:
    """
    RAG 检索服务 — 纯相关性检索，不包含画像个性化逻辑。
    画像相关的硬过滤和偏好重排由 rag_formatter 节点负责。
    """

    def __init__(
        self,
        client: MilvusClient,
        collection_name: str,
        embeddings: Embeddings,
    ):
        self.retrieval = RetrievalOptimizationModule(
            client=client,
            collection_name=collection_name,
            embeddings=embeddings,
            score_threshold=settings.RAG_SCORE_THRESHOLD,
        )
        self.query_rewriter = GenerationIntegrationModule()
        self.metadata_filter = MetadataFilterExtractor()
        self.reranker = SiliconFlowReranker()
        self.embeddings = embeddings

    async def search_recipes(
        self,
        query: str,
        metadata_catalog: Optional[dict] = None,
        extra_expr: Optional[str] = None,
    ) -> List[Document]:
        """
        执行完整的 RAG 检索流程。

        Args:
            query: 用户原始查询
            metadata_catalog: 可用元数据值（用于 LLM 生成过滤表达式）
            extra_expr: 额外的 Milvus 过滤表达式（如 category 粗过滤）

        Returns:
            检索并重排序后的文档列表（纯��关性排序）
        """
        # Step 2: Query Rewrite
        rewritten_query = await self.query_rewriter.rewrite_query(query)

        # Step 3: Metadata Filter (LLM 生成 expr)
        llm_expr = None
        if metadata_catalog:
            llm_expr = await self.metadata_filter.build_filter_expression(
                query=rewritten_query,
                metadata_catalog=metadata_catalog,
            )

        # 合并 extra_expr 和 llm_expr
        final_expr = self._merge_expressions(extra_expr, llm_expr)

        # Step 4: Vector Search
        docs, scores = await self.retrieval.hybrid_search(
            query=rewritten_query,
            top_k=settings.RAG_TOP_K,
            expr=final_expr,
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

        # 连接 Milvus Lite
        client = MilvusClient(uri=settings.MILVUS_URI)
        client.load_collection(settings.MILVUS_COLLECTION)
        logger.info(f"Milvus 集合已加载: {settings.MILVUS_COLLECTION}")

        _rag_service = RAGService(
            client=client,
            collection_name=settings.MILVUS_COLLECTION,
            embeddings=embeddings,
        )
        logger.info("RAG 服务初始化成功")
        return _rag_service

    except Exception as e:
        logger.error(f"RAG 服务初始化失败: {e}")
        return None
