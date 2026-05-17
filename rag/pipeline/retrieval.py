# rag/pipeline/retrieval.py
"""混合检索模块 — 基于 MilvusClient 的 dense 向量搜索。"""

import asyncio
import logging
from typing import List, Tuple, Optional

from pymilvus import MilvusClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    """
    基于 MilvusClient 的向量检索模块。
    使用 dense 向量（语义相似度）进行检索，支持元数据过滤。

    注意：由于 langchain_milvus 0.3.3 + pymilvus 2.6 存在 ORM 连接 bug��
    此模块直接使用 MilvusClient API，不依赖 langchain wrapper。
    """

    def __init__(
        self,
        client: MilvusClient,
        collection_name: str,
        embeddings: Embeddings,
        score_threshold: float = 0.0,
    ):
        """
        Args:
            client: MilvusClient 实例
            collection_name: 集合名称
            embeddings: embedding 模型（用于将 query 转为向量）
            score_threshold: 最低分数阈值，低于此分数的结果被过滤
        """
        if not client:
            raise ValueError("client 不能为空")

        self.client = client
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.score_threshold = score_threshold

        logger.info("检索模块初始化完成, collection=%s", collection_name)

    async def hybrid_search(
        self,
        query: str,
        top_k: int,
        score_threshold: Optional[float] = None,
        expr: Optional[str] = None,
    ) -> Tuple[List[Document], List[float]]:
        """
        执行向量检索，使用 cosine 相似度。

        Args:
            query: 用户查询（通常是 rewrite 后的）
            top_k: 返回文档数量
            score_threshold: 最低分数阈值（覆盖实例默认值）
            expr: Milvus 过滤表达式（如 category 过滤）

        Returns:
            (documents, scores) 元组，两个列表等长
        """
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        logger.info("向量检索 top_k=%d threshold=%s expr=%s", top_k, threshold, expr)

        # 将 query 转为向量
        query_vector = await asyncio.to_thread(
            self.embeddings.embed_query, query
        )

        # 构建搜索参数
        search_params = {"metric_type": "COSINE"}

        # 执行搜索
        results = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "category", "difficulty", "dish_name",
                           "user_id", "parent_id", "data_source", "source_type"],
            anns_field="dense_vector",
            search_params=search_params,
            filter=expr or "",
        )

        docs, scores = [], []
        if results and results[0]:
            for hit in results[0]:
                entity = hit["entity"]
                metadata = {
                    "category": entity.get("category", ""),
                    "difficulty": entity.get("difficulty", ""),
                    "dish_name": entity.get("dish_name", ""),
                    "user_id": entity.get("user_id", ""),
                    "parent_id": entity.get("parent_id", ""),
                    "data_source": entity.get("data_source", ""),
                    "source_type": entity.get("source_type", ""),
                }
                doc = Document(
                    page_content=entity.get("text", ""),
                    metadata=metadata,
                )
                docs.append(doc)
                scores.append(hit["distance"])

        logger.info("向量检索返回 %d 条文档", len(docs))

        # 分数阈值过滤（cosine 距离越大越相似）
        if threshold > 0:
            filtered_docs, filtered_scores = [], []
            for doc, score in zip(docs, scores):
                if score >= threshold:
                    filtered_docs.append(doc)
                    filtered_scores.append(score)

            logger.info("分数过滤: %d -> %d (threshold=%s)", len(docs), len(filtered_docs), threshold)
            return filtered_docs, filtered_scores

        return docs, scores
