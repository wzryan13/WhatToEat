# rag/vector_stores/vector_store_factory.py
"""Milvus 向量存储工厂 — 创建/连接 Milvus 集合，支持混合搜索（dense + BM25）。"""

import logging
import os
import shutil
from typing import List, Dict, Any

from pymilvus import DataType
from langchain_milvus import Milvus, BM25BuiltInFunction
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# 元数据 scalar schema — 用于 Milvus 精确过滤
METADATA_SCALAR_SCHEMA: Dict[str, Any] = {
    "category": {"dtype": DataType.VARCHAR, "max_length": 128},
    "difficulty": {"dtype": DataType.VARCHAR, "max_length": 64},
    "dish_name": {"dtype": DataType.VARCHAR, "max_length": 256},
    "user_id": {"dtype": DataType.VARCHAR, "max_length": 64},
    "parent_id": {"dtype": DataType.VARCHAR, "max_length": 64},
    "source": {"dtype": DataType.VARCHAR, "max_length": 256},
    "data_source": {"dtype": DataType.VARCHAR, "max_length": 64},
    "source_type": {"dtype": DataType.VARCHAR, "max_length": 64},
    "is_dish_index": {"dtype": DataType.BOOL},
}


def get_vector_store(
    uri: str,
    collection_name: str,
    embeddings: Embeddings,
    chunks: List[Document] = None,
    force_rebuild: bool = False,
) -> Milvus:
    """
    工厂函数：获取 Milvus 向量存储实例。

    使用 Milvus Lite（嵌入式模式），通过 URI 指定本地 .db 文件路径，
    无需 Docker 或外部 Milvus 服务。

    Args:
        uri: Milvus Lite 数据库路径（如 "./milvus_data/milvus.db"）
        collection_name: 集合名称
        embeddings: embedding 模型实例
        chunks: 待索引的文档列表（None 时连接已有集合）
        force_rebuild: 是否强制重建集合（drop_old）

    Returns:
        Milvus 向量存储实例（支持 hybrid search）
    """
    connection_args = {"uri": uri}

    logger.info(f"Milvus Lite 连接: {uri}, 集合: {collection_name}")

    # force_rebuild: 直接删除 db 目录/文件（langchain_milvus 的 drop_old 有 async bug）
    if force_rebuild and os.path.exists(uri):
        logger.warning(f"强制重建: 删除数据库 {uri}")
        if os.path.isdir(uri):
            shutil.rmtree(uri)
        else:
            os.remove(uri)

    if chunks:
        # 有数据要灌入：创建集合并索引
        vector_store = Milvus.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name,
            connection_args=connection_args,
            text_field="text",
            vector_field=["dense", "sparse"],
            builtin_function=BM25BuiltInFunction(),
            metadata_schema=METADATA_SCALAR_SCHEMA,
        )
        logger.info(f"集合 '{collection_name}' 创建成功，索引 {len(chunks)} 个文档")
    else:
        # 无数据：连接已有集合
        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args=connection_args,
            text_field="text",
            vector_field=["dense", "sparse"],
            builtin_function=BM25BuiltInFunction(),
            metadata_schema=METADATA_SCALAR_SCHEMA,
        )
        logger.info(f"连接已有集合: {collection_name}")

    return vector_store
