# rag/pipeline/retrieval.py
"""混合检索模块 — 结合 dense 向量搜索与 BM25 稀疏搜索。"""

import asyncio
import logging
from typing import List, Tuple, Optional

from langchain_milvus import Milvus
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """
    基于 Milvus 内置混合搜索的检索模块。
    同时使用 dense 向量（语义）和 sparse BM25（关键词）进行检索，
    通过 weighted ranker 融合两路结果。
    """

    def __init__(
        self,
        vectorstore: Milvus,
        score_threshold: float = 0.0,
        ranker_weights: List[float] = None,
    ):
        """
        Args:
            vectorstore: 启用了 BM25 的 Milvus 实例
            score_threshold: 最低分数阈值，低于此分数的结果被过滤
            ranker_weights: [dense_weight, sparse_weight] 固定权重，默认 [0.5, 0.5]
        """
        if not vectorstore:
            raise ValueError("vectorstore 不能为空")

        self.vectorstore = vectorstore
        self.score_threshold = score_threshold
        self.ranker_weights = ranker_weights or [0.5, 0.5]

        logger.info(
            "检索模块初始化完成, 权重: dense=%.2f sparse=%.2f",
            self.ranker_weights[0],
            self.ranker_weights[1],
        )

    async def hybrid_search(
        self,
        query: str,
        top_k: int,
        score_threshold: Optional[float] = None,
        expr: Optional[str] = None,
    ) -> Tuple[List[Document], List[float]]:
        """
        执行混合检索（dense + sparse），使用���定 weighted ranker 融合。

        Args:
            query: 用户查询（通常是 rewrite 后的）
            top_k: 返回文档数量
            score_threshold: 最低分��阈值（覆盖实例默认值）
            expr: Milvus 过滤表达式（如 category 过滤）

        Returns:
            (documents, scores) 元组，两个列表等长
        """
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        logger.info("混合检索 top_k=%d threshold=%s expr=%s", top_k, threshold, expr)

        ranker_params = {"weights": self.ranker_weights, "norm_score": True}

        results = await asyncio.to_thread(
            self.vectorstore.similarity_search_with_score,
            query=query,
            k=top_k,
            fetch_k=int(top_k * 4),
            ranker_type="weighted",
            ranker_params=ranker_params,
            expr=expr,
        )

        docs, scores = [], []
        for doc, score in results:
            docs.append(doc)
            scores.append(score)

        logger.info("混合检索返回 %d 条文档", len(docs))

        # 分数阈值过滤
        if threshold > 0:
            filtered_docs, filtered_scores = [], []
            for doc, score in zip(docs, scores):
                if score >= threshold:
                    filtered_docs.append(doc)
                    filtered_scores.append(score)

            logger.info("分数过滤: %d -> %d (threshold=%s)", len(docs), len(filtered_docs), threshold)
            return filtered_docs, filtered_scores

        return docs, scores
