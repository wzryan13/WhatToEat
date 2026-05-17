# rag/__init__.py
"""
RAG (Retrieval-Augmented Generation) 模块 — 菜谱检索���强生成。

核心组件:
- Embeddings: 本地 HuggingFace embedding 模型
- Vector Store: Milvus Lite 混合搜索（dense + BM25）
- Retrieval: 混合检索 + 固定权重融合
- Reranker: SiliconFlow Cross-Encoder 纯相关性精排
- Cache: 两级缓存（Redis L1 + Milvus L2）
- Document Processor: Markdown 分块 + 后处理去重
"""

from rag.cache import CacheManager
from rag.pipeline.document_processor import document_processor
from rag.pipeline.retrieval import RetrievalOptimizationModule
from rag.pipeline.generation import GenerationIntegrationModule
from rag.pipeline.metadata_filter import MetadataFilterExtractor
from rag.rerankers.siliconflow_reranker import SiliconFlowReranker

__all__ = [
    "CacheManager",
    "document_processor",
    "RetrievalOptimizationModule",
    "GenerationIntegrationModule",
    "MetadataFilterExtractor",
    "SiliconFlowReranker",
]
