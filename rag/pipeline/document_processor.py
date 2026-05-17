# rag/pipeline/document_processor.py
"""
文档处理器 — 负责 Markdown 分块和检索结果后处理。
分块逻辑保持原样（MarkdownHeaderTextSplitter 按 # 和 ## 切分）。
post_process_retrieval 简化为按 parent_id 去重，不依赖 PostgreSQL。
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

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
        对检索结果进行后处理：按 parent_id 去重，保留最高分 chunk。

        简化版：不从 PostgreSQL 获取父文档，直接返回去重后的 chunks。

        Args:
            retrieved_chunks: 向量搜索返回的 chunk 列表

        Returns:
            去重后的文档列表（按分数降序）
        """
        if not retrieved_chunks:
            return []

        # 按 parent_id 分组，保留每组最高分的 chunk
        best_by_parent: Dict[str, Document] = {}

        for chunk in retrieved_chunks:
            parent_id = chunk.metadata.get("parent_id", chunk.metadata.get("dish_name", str(id(chunk))))

            rerank_score = chunk.metadata.get("rerank_score", 0.0)
            retrieval_score = chunk.metadata.get("retrieval_score", 0.0)
            current_best_score = rerank_score or retrieval_score

            if parent_id not in best_by_parent:
                best_by_parent[parent_id] = chunk
            else:
                existing = best_by_parent[parent_id]
                existing_score = existing.metadata.get("rerank_score", 0.0) or existing.metadata.get("retrieval_score", 0.0)
                if current_best_score > existing_score:
                    best_by_parent[parent_id] = chunk

        # 按分数降序排列
        final_docs = list(best_by_parent.values())
        final_docs.sort(
            key=lambda d: (
                d.metadata.get("rerank_score", 0.0),
                d.metadata.get("retrieval_score", 0.0),
            ),
            reverse=True,
        )

        logger.info("后处理: %d chunks -> %d 去重文档", len(retrieved_chunks), len(final_docs))
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
