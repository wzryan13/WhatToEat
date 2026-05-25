# rag/pipeline/document_processor.py
"""
文档处理器 — 负责 Markdown 分块和检索结果后处理。
分块逻辑：MarkdownHeaderTextSplitter 按 # 和 ## 切分。
post_process_retrieval：按 parent_id 去重后回 PostgreSQL 取完整父文档（small-to-large）。
DATABASE_URL 未配置时自动降级为返回去重 chunk。
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from rag.pipeline.document_repository import get_document_repository

logger = logging.getLogger(__name__)

# 文档所需的元数据键
REQUIRED_METADATA_KEYS = (
    "source",
    "parent_id",
    "dish_name",
    "category",
    "difficulty",
    "is_dish_index",
    "data_source",
    "user_id",
    "source_type",
)


class DocumentProcessor:
    """
    文档处理器：
    - 将 Markdown 文档按标题切分为 chunks
    - 对检索结果按 parent_id 去重，保留最高分
    """

    def __init__(self, headers_to_split_on: List[tuple] | None = None):
        self.headers_to_split_on = headers_to_split_on or [
            ("#", "header_1"),
            ("##", "header_2"),
        ]
        self._splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            strip_headers=False,
        )

    def create_chunks(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
    ) -> List[Document]:
        """
        将文档按 Markdown 标题切分为 chunks。

        Args:
            doc_id: 父文档 ID（存入 chunk metadata 的 parent_id）
            content: 文档内容（Markdown 格式）
            metadata: 基础元数据（会被复制到每个 chunk）

        Returns:
            chunk Document 列表
        """
        chunks: List[Document] = []

        md_chunks = self._splitter.split_text(content)

        for chunk_doc in md_chunks:
            chunk_metadata = self._clone_metadata(metadata, parent_id=doc_id)
            chunks.append(
                Document(
                    id=str(uuid.uuid4()),
                    page_content=chunk_doc.page_content,
                    metadata=chunk_metadata,
                )
            )

        return chunks

    async def post_process_retrieval(
        self,
        retrieved_chunks: List[Document],
    ) -> List[Document]:
        """
        检索后处理：small-to-large 模式。
        先按 parent_id 分组保留最高分，再从 PostgreSQL 取完整父文档。
        DATABASE_URL 未配置时降级为返回去重 chunk。

        Args:
            retrieved_chunks: 向量搜索返回的 chunk 列表

        Returns:
            父文档列表（含完整内容），按分数降序排列
        """
        if not retrieved_chunks:
            return []

        # 按 parent_id 分组，收集最高 retrieval_score / rerank_score
        parent_scores: Dict[str, Dict[str, Any]] = {}

        for chunk in retrieved_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if not parent_id:
                continue

            retrieval_score = chunk.metadata.get("retrieval_score", 0.0)
            rerank_score = chunk.metadata.get("rerank_score")

            if parent_id not in parent_scores:
                parent_scores[parent_id] = {
                    "retrieval_score": retrieval_score,
                    "rerank_score": rerank_score if rerank_score is not None else 0.0,
                    "_chunk": chunk,  # 降级时使用
                }
            else:
                if retrieval_score > parent_scores[parent_id]["retrieval_score"]:
                    parent_scores[parent_id]["retrieval_score"] = retrieval_score
                if rerank_score is not None and rerank_score > parent_scores[parent_id]["rerank_score"]:
                    parent_scores[parent_id]["rerank_score"] = rerank_score

        repo = get_document_repository()

        # 降级路径：DATABASE_URL 未配置，返回去重 chunk
        if repo is None:
            fallback = [scores["_chunk"] for scores in parent_scores.values()]
            fallback.sort(
                key=lambda d: (
                    d.metadata.get("rerank_score", 0.0),
                    d.metadata.get("retrieval_score", 0.0),
                ),
                reverse=True,
            )
            logger.info(
                "后处理(降级): %d chunks -> %d 去重 chunk（无 PostgreSQL）",
                len(retrieved_chunks), len(fallback),
            )
            return fallback

        # 正常路径：从 PostgreSQL 取完整父文档
        parent_ids = list(parent_scores.keys())
        parent_docs = await repo.get_parent_documents(parent_ids)

        final_docs: List[Document] = []
        for parent_id, scores in parent_scores.items():
            if parent_id not in parent_docs:
                logger.warning("父文档未找到，跳过: %s", parent_id)
                continue
            parent_doc = parent_docs[parent_id]
            doc_copy = Document(
                id=parent_doc.id,
                page_content=parent_doc.page_content,
                metadata=parent_doc.metadata.copy(),
            )
            doc_copy.metadata["retrieval_score"] = scores["retrieval_score"]
            doc_copy.metadata["rerank_score"] = scores["rerank_score"]
            final_docs.append(doc_copy)

        final_docs.sort(
            key=lambda d: (
                d.metadata.get("rerank_score", 0.0),
                d.metadata.get("retrieval_score", 0.0),
            ),
            reverse=True,
        )

        logger.info(
            "后处理: %d chunks -> %d 父文档",
            len(retrieved_chunks), len(final_docs),
        )
        return final_docs

    def _clone_metadata(
        self,
        metadata: Dict[str, Any],
        *,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """克隆元数据，可选设置 parent_id。"""
        cloned = {key: metadata.get(key) for key in REQUIRED_METADATA_KEYS}
        if parent_id is not None:
            cloned["parent_id"] = parent_id
        return cloned

    def _create_index_chunk_content(self, index_metadata: Dict[str, Any]) -> str:
        """为菜品索引文档创建 chunk 内容。"""
        content_parts = ["推荐菜,菜谱列表,菜品,食谱,有哪些菜品推荐"]

        source = index_metadata.get("source", "")
        category = index_metadata.get("category", "")
        difficulty = index_metadata.get("difficulty", "")

        if "category" in source and category:
            content_parts.append(f"{category}推荐，")
        elif "difficulty" in source and difficulty:
            content_parts.append(f"{difficulty}难度推荐，")

        content_parts.append("欢迎根据口味挑选合适的菜谱")
        return "".join(content_parts)


# 单例实例
document_processor = DocumentProcessor()
