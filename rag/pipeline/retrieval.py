"""混合检索模块 — 基于 langchain_milvus 的 dense + BM25 hybrid search。"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_milvus import Milvus

from config.settings import settings

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """基于 langchain_milvus 的 hybrid retrieval，支持 rrf/weighted 与 expr 过滤。"""

    def __init__(
        self,
        vector_store: Milvus,
        score_threshold: float = 0.0,
    ):
        if not vector_store:
            raise ValueError("vector_store 不能为空")

        self.vector_store = vector_store
        self.score_threshold = score_threshold
        self.default_ranker_type = settings.RAG_RANKER_TYPE
        self.default_ranker_params = self._build_ranker_params(self.default_ranker_type)

        logger.info(
            "检索模块初始化完成, collection=%s ranker_type=%s ranker_params=%s",
            vector_store.collection_name,
            self.default_ranker_type,
            self.default_ranker_params,
        )

    @staticmethod
    def _build_ranker_params(ranker_type: Optional[str]) -> Dict[str, Any]:
        if ranker_type == "weighted":
            return {"weights": settings.RAG_RANKER_WEIGHTS}
        if ranker_type == "rrf":
            return {"k": settings.RAG_RRF_K}
        return {}

    @staticmethod
    def _normalize_ranker_type(ranker_type: Optional[str]) -> Optional[str]:
        normalized = (ranker_type or "").strip().lower() or None
        if normalized not in {None, "rrf", "weighted"}:
            raise ValueError(f"不支持的 ranker_type: {ranker_type}")
        return normalized

    async def hybrid_search(
        self,
        query: str,
        top_k: int,
        score_threshold: Optional[float] = None,
        expr: Optional[str] = None,
        ranker_type: Optional[str] = None,
        ranker_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Document], List[float]]:
        """执行 dense + BM25 混合检索。"""

        threshold = score_threshold if score_threshold is not None else self.score_threshold
        resolved_ranker_type = self._normalize_ranker_type(
            ranker_type or self.default_ranker_type
        )
        resolved_ranker_params = ranker_params or self._build_ranker_params(resolved_ranker_type)
        logger.info(
            "Hybrid 检索 top_k=%d threshold=%s expr=%s ranker_type=%s ranker_params=%s",
            top_k,
            threshold,
            expr,
            resolved_ranker_type,
            resolved_ranker_params,
        )

        results = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            query,
            top_k,
            None,
            expr,
            None,
            ranker_type=resolved_ranker_type,
            ranker_params=resolved_ranker_params,
        )

        docs: List[Document] = []
        scores: List[float] = []
        for doc, score in results:
            if threshold > 0 and score < threshold:
                continue
            docs.append(doc)
            scores.append(score)

        logger.info("Hybrid 检索返回 %d 条文档", len(docs))
        return docs, scores
