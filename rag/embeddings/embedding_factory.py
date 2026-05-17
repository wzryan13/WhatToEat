# rag/embeddings/embedding_factory.py
"""Embedding 模型工厂 — 创建 HuggingFace 本地 embedding 实例。"""

import logging
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


def get_embedding_model(model_name: str) -> Embeddings:
    """
    工厂函数：基于模型名称创建本地 embedding 模型。

    Args:
        model_name: HuggingFace 模型名，如 "BAAI/bge-small-zh-v1.5"

    Returns:
        归一化的 embedding 实例（用于余弦相似度计算）
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    logger.info(f"初始化本地 embedding 模型: {model_name}")
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
