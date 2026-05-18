"""Milvus 向量存储工厂 — 连接 Milvus Standalone/Distributed hybrid collection。"""

import logging
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_milvus import BM25BuiltInFunction, Milvus
from langchain_milvus.vectorstores.milvus import (
    AsyncMilvusClient,
    Collection,
    DEFAULT_MILVUS_CONNECTION,
)
from pymilvus import DataType, MilvusClient, connections, utility

from config.settings import settings

logger = logging.getLogger(__name__)

PRIMARY_FIELD = "id"
TEXT_FIELD = "text"
DENSE_VECTOR_FIELD = "dense_vector"
SPARSE_VECTOR_FIELD = "sparse"
VECTOR_FIELDS = [DENSE_VECTOR_FIELD, SPARSE_VECTOR_FIELD]

DEFAULT_INDEX_PARAMS = [
    {"metric_type": "COSINE", "index_type": "FLAT", "params": {}},
    {"metric_type": "BM25", "index_type": "AUTOINDEX", "params": {}},
]

METADATA_SCALAR_SCHEMA: Dict[str, Any] = {
    "category": {"dtype": DataType.VARCHAR, "max_length": 128},
    "difficulty": {"dtype": DataType.VARCHAR, "max_length": 64},
    "dish_name": {"dtype": DataType.VARCHAR, "max_length": 256},
    "user_id": {"dtype": DataType.VARCHAR, "max_length": 64},
    "parent_id": {"dtype": DataType.VARCHAR, "max_length": 64},
    "source": {"dtype": DataType.VARCHAR, "max_length": 2048},
    "data_source": {"dtype": DataType.VARCHAR, "max_length": 64},
    "source_type": {"dtype": DataType.VARCHAR, "max_length": 64},
    "is_dish_index": {"dtype": DataType.BOOL},
}


class CompatibleMilvus(Milvus):
    """为 langchain_milvus 显式补齐 ORM alias，避免 Collection(using=alias) 失败。"""

    def __init__(
        self,
        embedding_function,
        collection_name: str = "LangChainCollection",
        collection_description: str = "",
        collection_properties: Optional[dict[str, Any]] = None,
        connection_args: Optional[dict[str, Any]] = None,
        consistency_level: str = "Session",
        index_params=None,
        search_params=None,
        drop_old: Optional[bool] = False,
        auto_id: bool = False,
        *,
        primary_field: str = "pk",
        text_field: str = "text",
        vector_field="vector",
        enable_dynamic_field: bool = False,
        metadata_field: Optional[str] = None,
        partition_key_field: Optional[str] = None,
        num_partitions: Optional[int] = None,
        partition_names: Optional[list] = None,
        replica_number: int = 1,
        timeout: Optional[float] = None,
        num_shards: Optional[int] = None,
        vector_schema=None,
        metadata_schema: Optional[dict[str, Any]] = None,
        builtin_function=None,
    ):
        self.default_search_params = {
            "FLAT": {"metric_type": "L2", "params": {}},
            "IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_SQ8": {"metric_type": "L2", "params": {"nprobe": 10}},
            "IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
            "HNSW": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_FLAT": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_SQ": {"metric_type": "L2", "params": {"ef": 10}},
            "RHNSW_PQ": {"metric_type": "L2", "params": {"ef": 10}},
            "IVF_HNSW": {"metric_type": "L2", "params": {"nprobe": 10, "ef": 10}},
            "ANNOY": {"metric_type": "L2", "params": {"search_k": 10}},
            "SCANN": {"metric_type": "L2", "params": {"search_k": 10}},
            "AUTOINDEX": {"metric_type": "L2", "params": {}},
            "GPU_CAGRA": {
                "metric_type": "L2",
                "params": {
                    "itopk_size": 128,
                    "search_width": 4,
                    "min_iterations": 0,
                    "max_iterations": 0,
                    "team_size": 0,
                },
            },
            "GPU_IVF_FLAT": {"metric_type": "L2", "params": {"nprobe": 10}},
            "GPU_IVF_PQ": {"metric_type": "L2", "params": {"nprobe": 10}},
            "GPU_BRUTE_FORCE": {"metric_type": "L2", "params": {"nprobe": 10}},
            "SPARSE_INVERTED_INDEX": {
                "metric_type": "IP",
                "params": {"drop_ratio_build": 0.2},
            },
            "SPARSE_WAND": {"metric_type": "IP", "params": {"drop_ratio_build": 0.2}},
        }

        if not embedding_function and not builtin_function:
            raise ValueError(
                "Either `embedding_function` or `builtin_function` should be provided."
            )

        self.embedding_func = self._from_list(embedding_function)
        self.builtin_func = self._from_list(builtin_function)
        self.collection_name = collection_name
        self.collection_description = collection_description
        self.collection_properties = collection_properties
        self.index_params = index_params
        self.search_params = search_params
        self.consistency_level = consistency_level
        self.auto_id = auto_id
        self.num_partitions = num_partitions
        self._primary_field = primary_field
        self._text_field = text_field

        self._check_vector_field(vector_field, vector_schema)
        if enable_dynamic_field and metadata_field:
            metadata_field = None
        self.enable_dynamic_field = enable_dynamic_field
        self._metadata_field = metadata_field
        self._partition_key_field = partition_key_field
        self.fields: list[str] = []
        self.partition_names = partition_names
        self.replica_number = replica_number
        self.timeout = timeout
        self.num_shards = num_shards
        self.metadata_schema = metadata_schema

        if connection_args is None:
            connection_args = DEFAULT_MILVUS_CONNECTION

        self._connection_args = connection_args
        self._milvus_client = MilvusClient(**connection_args)

        try:
            self._async_milvus_client = AsyncMilvusClient(**connection_args)
        except Exception as exc:  # pragma: no cover
            logger.warning("AsyncMilvusClient 初始化失败，异步 API 不可用: %s", exc)
            self._async_milvus_client = None

        self.alias = self.client._using
        self._ensure_orm_alias()
        self.col = None

        if utility.has_collection(self.collection_name, using=self.alias):
            self.col = Collection(self.collection_name, using=self.alias)
            if self.collection_properties is not None:
                self.col.set_properties(self.collection_properties)

        if drop_old and isinstance(self.col, Collection):
            self.drop()

        self._init(
            partition_names=partition_names,
            replica_number=replica_number,
            timeout=timeout,
        )

    def _ensure_orm_alias(self) -> None:
        if connections.has_connection(self.alias):
            return

        try:
            connections.connect(alias=self.alias, **self._connection_args)
        except Exception:
            if not connections.has_connection(self.alias):
                raise


def _is_milvus_lite_uri(uri: str) -> bool:
    return uri.endswith(".db")


def _get_bm25_function() -> BM25BuiltInFunction:
    return BM25BuiltInFunction(
        input_field_names=TEXT_FIELD,
        output_field_names=SPARSE_VECTOR_FIELD,
    )


def _milvus_common_kwargs(uri: str, collection_name: str) -> Dict[str, Any]:
    return {
        "collection_name": collection_name,
        "connection_args": {"uri": uri},
        "text_field": TEXT_FIELD,
        "primary_field": PRIMARY_FIELD,
        "vector_field": VECTOR_FIELDS,
        "builtin_function": _get_bm25_function(),
        "metadata_schema": METADATA_SCALAR_SCHEMA,
        "index_params": DEFAULT_INDEX_PARAMS,
        "consistency_level": "Strong",
    }


def _existing_collection_fields(client: MilvusClient, collection_name: str) -> set[str]:
    schema = client.describe_collection(collection_name)
    return {field["name"] for field in schema.get("fields", [])}


def _is_hybrid_compatible(client: MilvusClient, collection_name: str) -> bool:
    field_names = _existing_collection_fields(client, collection_name)
    return all(field in field_names for field in [TEXT_FIELD, *VECTOR_FIELDS])


def _dump_collection_documents(
    client: MilvusClient,
    collection_name: str,
) -> List[Document]:
    docs: List[Document] = []
    iterator = client.query_iterator(
        collection_name=collection_name,
        batch_size=512,
        limit=-1,
        filter="",
        output_fields=["*"],
    )

    try:
        while True:
            batch = iterator.next()
            if not batch:
                break

            for row in batch:
                text = row.get(TEXT_FIELD, "")
                metadata = {
                    key: value
                    for key, value in row.items()
                    if key not in {PRIMARY_FIELD, TEXT_FIELD, DENSE_VECTOR_FIELD, SPARSE_VECTOR_FIELD}
                }
                docs.append(Document(page_content=text, metadata=metadata))
    finally:
        iterator.close()

    return docs


def _prepare_remote_collection(
    uri: str,
    collection_name: str,
    force_rebuild: bool,
) -> None:
    client = MilvusClient(uri=uri)
    try:
        if force_rebuild and client.has_collection(collection_name):
            client.drop_collection(collection_name)
            logger.warning("已删除旧集合: %s", collection_name)
    finally:
        client.close()


def _build_vector_store_from_documents(
    uri: str,
    collection_name: str,
    embeddings: Embeddings,
    documents: List[Document],
    force_rebuild: bool,
) -> Milvus:
    _prepare_remote_collection(uri, collection_name, force_rebuild)
    vector_store = CompatibleMilvus.from_documents(
        documents=documents,
        embedding=embeddings,
        **_milvus_common_kwargs(uri, collection_name),
    )
    logger.info("已创建 hybrid collection: %s，文档数=%d", collection_name, len(documents))
    return vector_store


def _connect_existing_vector_store(
    uri: str,
    collection_name: str,
    embeddings: Embeddings,
) -> Milvus:
    vector_store = CompatibleMilvus(
        embedding_function=embeddings,
        **_milvus_common_kwargs(uri, collection_name),
    )
    vector_store.client.load_collection(collection_name=collection_name)
    logger.info("已连接 hybrid collection: %s", collection_name)
    return vector_store


def _validate_hybrid_search(vector_store: Milvus) -> None:
    if vector_store.col is None:
        raise RuntimeError(f"Milvus collection 不存在: {vector_store.collection_name}")

    ranker_type = settings.RAG_RANKER_TYPE
    ranker_params: Dict[str, Any]
    if ranker_type == "weighted":
        ranker_params = {"weights": settings.RAG_RANKER_WEIGHTS}
    else:
        ranker_params = {"k": settings.RAG_RRF_K}

    vector_store.similarity_search_with_score(
        query="测试",
        k=1,
        ranker_type=ranker_type,
        ranker_params=ranker_params,
    )


def get_vector_store(
    uri: str,
    collection_name: str,
    embeddings: Embeddings,
    chunks: Optional[List[Document]] = None,
    force_rebuild: bool = False,
) -> Milvus:
    """获取支持 dense + BM25 hybrid search 的 Milvus 向量存储实例。"""
    if _is_milvus_lite_uri(uri):
        raise RuntimeError(
            "当前配置仍是 Milvus Lite URI。BM25BuiltInFunction hybrid search 需要 "
            "Milvus Standalone 或 Milvus Distributed，请将 MILVUS_URI 改为 "
            "例如 http://127.0.0.1:19530。"
        )

    logger.info("Milvus 连接: %s, 集合: %s", uri, collection_name)

    if chunks is not None:
        return _build_vector_store_from_documents(
            uri=uri,
            collection_name=collection_name,
            embeddings=embeddings,
            documents=chunks,
            force_rebuild=force_rebuild,
        )

    client = MilvusClient(uri=uri)
    try:
        if client.has_collection(collection_name) and not _is_hybrid_compatible(
            client, collection_name
        ):
            client.load_collection(collection_name=collection_name)
            migrated_docs = _dump_collection_documents(client, collection_name)
            client.drop_collection(collection_name)
            logger.warning(
                "检测到旧版 dense-only schema，已重建 hybrid collection: %s",
                collection_name,
            )
            return _build_vector_store_from_documents(
                uri=uri,
                collection_name=collection_name,
                embeddings=embeddings,
                documents=migrated_docs,
                force_rebuild=False,
            )
    finally:
        client.close()

    vector_store = _connect_existing_vector_store(uri, collection_name, embeddings)
    _validate_hybrid_search(vector_store)
    return vector_store
