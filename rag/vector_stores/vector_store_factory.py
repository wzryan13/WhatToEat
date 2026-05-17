# rag/vector_stores/vector_store_factory.py
"""Milvus 向量存储工厂 — 创建/连接 Milvus 集合，支持混合搜索（dense + BM25）。"""

import logging
from typing import List, Dict, Any

from pymilvus import utility, connections, DataType
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
    chunks: List[Document],
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
        chunks: 待索引的文档列表（为空时创建空集合）
        force_rebuild: 是否强制重建集合

    Returns:
        Milvus 向量存储实例（支持 hybrid search）
    """
    connection_args = {"uri": uri}

    logger.info(f"Milvus Lite 连接: {uri}, 集合: {collection_name}")

    # 检查集合是否存在（Milvus Lite 通过 uri 直接连接）
    try:
        connections.connect(alias="default", uri=uri)
        if force_rebuild and utility.has_collection(collection_name, using="default"):
            logger.warning(f"强制重建: 删除集合 {collection_name}")
            utility.drop_collection(collection_name, using="default")

        collection_exists = utility.has_collection(collection_name, using="default")
    finally:
        if connections.has_connection("default"):
            connections.disconnect("default")

    if not collection_exists:
        logger.info(f"集合 '{collection_name}' 不存在，正在创建...")

        if chunks:
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
        else:
            # 用占位文档创建集合以确保 schema 建立
            placeholder_doc = Document(
                page_content="__placeholder__",
                metadata={
                    "category": "__placeholder__",
                    "difficulty": "__placeholder__",
                    "dish_name": "__placeholder__",
                    "user_id": "__placeholder__",
                    "parent_id": "__placeholder__",
                    "source": "__placeholder__",
                    "data_source": "__placeholder__",
                    "source_type": "__placeholder__",
                    "is_dish_index": False,
                },
            )
            vector_store = Milvus.from_documents(
                documents=[placeholder_doc],
                embedding=embeddings,
                collection_name=collection_name,
                connection_args=connection_args,
                text_field="text",
                vector_field=["dense", "sparse"],
                builtin_function=BM25BuiltInFunction(),
                metadata_schema=METADATA_SCALAR_SCHEMA,
            )
            try:
                vector_store.col.delete(expr='text == "__placeholder__"')
                logger.info("占位文档已删除，空集合就绪")
            except Exception as e:
                logger.warning(f"删除占位文档失败: {e}")

        logger.info(f"集合 '{collection_name}' 创建成功")
    else:
        logger.info(f"连接已有集合: {collection_name}")
        vector_store = Milvus(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_args=connection_args,
            text_field="text",
            vector_field=["dense", "sparse"],
            builtin_function=BM25BuiltInFunction(),
            metadata_schema=METADATA_SCALAR_SCHEMA,
        )

    return vector_store
