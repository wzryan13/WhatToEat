# rag/rerankers/siliconflow_reranker.py
"""SiliconFlow Cross-Encoder 重排序器 — 纯相关性打分，不注入偏好。"""

import logging
import httpx
from typing import List

from rag.rerankers.base import BaseReranker
from langchain_core.documents import Document
from config.settings import settings

logger = logging.getLogger(__name__)


class SiliconFlowReranker(BaseReranker):
    """
    使用 SiliconFlow Rerank API 进行 cross-encoder 重排序。
    只管语义相关性打分，不注入任何用户偏好。
    """

    def __init__(
        self,
        api_url: str = None,
        api_key: str = None,
        model_name: str = None,
        score_threshold: float = None,
    ):
        self.api_url = api_url or settings.SILICONFLOW_BASE_URL
        self.api_key = api_key or settings.SILICONFLOW_API_KEY
        self.model_name = model_name or settings.SILICONFLOW_MODEL
        self.score_threshold = score_threshold if score_threshold is not None else settings.RAG_RERANK_THRESHOLD
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        使用 SiliconFlow API 对文档进行纯相关性重排序。

        Args:
            query: 用户查询（不含偏好信息）
            documents: 待重排序的文档列表

        Returns:
            过滤并排序后的文档列表（按 rerank_score 降序）
        """
        if not documents:
            return []

        logger.info(f"SiliconFlow 重排序: {len(documents)} 篇文档...")

        doc_contents = [doc.page_content for doc in documents]

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": doc_contents,
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30.0,
                )
                response.raise_for_status()
                api_results = response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"SiliconFlow API HTTP 错误: {e}")
            logger.error(f"响应内容: {e.response.text}")
            return documents  # 降级：返回原始文档
        except Exception as e:
            logger.error(f"SiliconFlow API 调用异常: {e}")
            return documents  # 降级：返回原始文档

        results = api_results.get("results", [])
        if not results:
            logger.warning("SiliconFlow API 返回空结果")
            return []

        # 按 score 过滤并排序
        ranked_docs = []
        for res in results:
            score = res.get("relevance_score", 0.0)
            index = res.get("index")

            if index is not None and index < len(documents):
                logger.debug(
                    f"文档 '{documents[index].metadata.get('dish_name', 'unknown')}' "
                    f"rerank 分数: {score:.4f}"
                )

                if score >= self.score_threshold * 0.9:
                    original_doc = documents[index]
                    original_doc.metadata["rerank_score"] = score
                    ranked_docs.append(original_doc)

        ranked_docs.sort(
            key=lambda doc: doc.metadata.get("rerank_score", 0.0), reverse=True
        )

        logger.info(f"重排序完成: {len(documents)} -> {len(ranked_docs)} 篇")
        return ranked_docs
