# rag/pipeline/document_repository.py
"""
父文档 PostgreSQL 存储层。
负责在灌入时持久化完整菜谱内容，在检索后回查完整正文（small-to-large 模式）。
"""

import json
import logging
from typing import Dict, List, Optional

from langchain_core.documents import Document

from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentRepository:
    """通过 asyncpg 读写 parent_documents 表。"""

    def __init__(self, dsn: str):
        self.dsn = dsn

    async def _connect(self):
        import asyncpg
        return await asyncpg.connect(self.dsn)

    async def initialize(self) -> None:
        """建表（幂等）。"""
        conn = await self._connect()
        try:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS parent_documents (
                    parent_id   VARCHAR(36)  PRIMARY KEY,
                    content     TEXT         NOT NULL,
                    metadata    JSONB        NOT NULL DEFAULT '{}'::jsonb,
                    data_source VARCHAR(50),
                    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_parent_documents_data_source
                    ON parent_documents(data_source)
                """
            )
        finally:
            await conn.close()

    async def truncate(self) -> None:
        """清空 parent_documents 表（用于 force_rebuild 重灌）。"""
        conn = await self._connect()
        try:
            await conn.execute("TRUNCATE TABLE parent_documents")
        finally:
            await conn.close()

    async def save_parent_documents(self, docs: List[Document]) -> None:
        """
        批量 upsert 父文档（幂等，支持重复灌入）。
        metadata 中所有字段序列化为 JSONB。
        """
        if not docs:
            return

        conn = await self._connect()
        try:
            rows = [
                (
                    doc.id,
                    doc.page_content,
                    json.dumps(doc.metadata, ensure_ascii=False, default=str),
                    doc.metadata.get("data_source"),
                )
                for doc in docs
            ]
            await conn.executemany(
                """
                INSERT INTO parent_documents (parent_id, content, metadata, data_source)
                VALUES ($1, $2, $3::jsonb, $4)
                ON CONFLICT (parent_id)
                DO UPDATE SET
                    content     = EXCLUDED.content,
                    metadata    = EXCLUDED.metadata,
                    data_source = EXCLUDED.data_source
                """,
                rows,
            )
        finally:
            await conn.close()

    async def get_parent_documents(
        self, parent_ids: List[str]
    ) -> Dict[str, Document]:
        """
        按 parent_id 批量取父文档，返回 {parent_id: Document} 映射。
        未找到的 id 不会出现在结果中（调用方负责处理缺失情况）。
        """
        if not parent_ids:
            return {}

        conn = await self._connect()
        try:
            rows = await conn.fetch(
                """
                SELECT parent_id, content, metadata
                FROM parent_documents
                WHERE parent_id = ANY($1::varchar[])
                """,
                parent_ids,
            )
        finally:
            await conn.close()

        result: Dict[str, Document] = {}
        for row in rows:
            try:
                meta = json.loads(row["metadata"]) if row["metadata"] else {}
            except (json.JSONDecodeError, TypeError):
                meta = {}
            result[row["parent_id"]] = Document(
                id=row["parent_id"],
                page_content=row["content"],
                metadata=meta,
            )
        return result


# -----------------------------------------------------------------------
# 单例工厂
# -----------------------------------------------------------------------

_repo: Optional[DocumentRepository] = None
_repo_initialized = False


def get_document_repository() -> Optional[DocumentRepository]:
    """
    返回全局 DocumentRepository 单例。
    DATABASE_URL 未配置时返回 None（调用方应降级处理）。
    """
    global _repo, _repo_initialized
    if _repo_initialized:
        return _repo
    _repo_initialized = True

    dsn = settings.DATABASE_URL
    if not dsn:
        logger.warning(
            "DATABASE_URL 未配置，父文档将不会持久化到 PostgreSQL，"
            "post_process_retrieval 将回退为返回 chunk 片段。"
        )
        _repo = None
        return None

    try:
        import asyncpg  # noqa: F401
    except ImportError:
        logger.warning("asyncpg 未安装，无法使用 PostgreSQL 父文档存储。")
        _repo = None
        return None

    _repo = DocumentRepository(dsn)
    return _repo
