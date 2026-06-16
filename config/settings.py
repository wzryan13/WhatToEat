import os
from dotenv import load_dotenv

load_dotenv("/Users/wenzhouzhou/PycharmProjects/PythonProject1/.env")


def _parse_float_list(value: str, default: list[float]) -> list[float]:
    raw = (value or "").strip()
    if not raw:
        return default
    try:
        return [float(item.strip()) for item in raw.split(",") if item.strip()]
    except ValueError:
        return default


class Settings:
    # ── API Keys ─────────────────────────────────────────────
    DEEPSEEK_API_KE: str = os.getenv("DEEPSEEK_API_KEY", "")
    AMAP_API_KEY: str = os.getenv("AMAP_MAPS_API_KEY", "")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # ── 模型配置 ──────────────────────────────────────────────
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")
    MODEL_NAME: str = LLM_MODEL

    # ── 查询理解后端（deepseek 云端 / local 本地微调模型）────────
    QU_BACKEND: str = os.getenv("QU_BACKEND", "deepseek")  # "deepseek" | "local"
    QU_LOCAL_BASE_URL: str = os.getenv("QU_LOCAL_BASE_URL", "http://localhost:1234/v1")
    QU_LOCAL_MODEL: str = os.getenv("QU_LOCAL_MODEL", "qu-finetuned")

    # ── 记忆系统 ──────────────────────────────────────────────
    MEMORY_ENABLED: bool = os.getenv("MEMORY_ENABLED", "true").lower() != "false"
    SESSION_TTL_HOURS: int = int(os.getenv("SESSION_TTL_HOURS", "4"))
    DEMO_CHANNEL: str = os.getenv("DEMO_CHANNEL", "cli")
    DEMO_EXTERNAL_ID: str = os.getenv("DEMO_EXTERNAL_ID", "local-cli-user")
    RECENT_HISTORY_N: int = int(os.getenv("RECENT_HISTORY_N", "6"))

    # ── 搜索参数 ──────────────────────────────────────────────
    POI_DETAIL_LIMIT: int = 15  # 并发查POI详情的上限条数（旧流程保留）
    DEFAULT_RADIUS: int = 1000  # 周边搜默认半径（米）
    MAX_CLARIFICATION: int = 10  # 最多追问次数
    MAX_RECOMMENDATIONS: int = 10  # 最终推荐数量上限（6-15家）

    # ── LLM 重试 ──────────────────────────────────────────────
    INTENT_PARSER_MAX_RETRIES: int = 3  # 意图解析LLM最大重试次数

    # ── Search Agent 参数 ─────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = 2   # 搜索最大迭代轮数
    AGENT_POI_PER_KEYWORD: int = 8  # 均分时每个关键词取多少条查详情
    AGENT_MIN_SUFFICIENT: int = 6   # 过滤后达到此数量认为"足够"

    # ── RAG 配置 ──────────────────────────────────────────────
    # Milvus Standalone / Distributed（支持 BM25BuiltInFunction hybrid search）
    MILVUS_URI: str = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
    MILVUS_COLLECTION: str = os.getenv("MILVUS_COLLECTION", "recipe_chunks")

    # Embedding 模型
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")

    # SiliconFlow Reranker
    SILICONFLOW_API_KEY: str = os.getenv("SILICONFLOW_API_KEY", "")
    SILICONFLOW_BASE_URL: str = os.getenv(
        "SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1/rerank"
    )
    SILICONFLOW_MODEL: str = os.getenv("SILICONFLOW_MODEL", "BAAI/bge-reranker-v2-m3")
    RAG_RERANK_THRESHOLD: float = float(os.getenv("RAG_RERANK_THRESHOLD", "0.01"))

    # RAG 检索参数
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "20"))           # 混合检索召回数
    RAG_RERANK_TOP_K: int = int(os.getenv("RAG_RERANK_TOP_K", "10"))  # rerank 后保留数
    RAG_RANKER_TYPE: str = os.getenv("RAG_RANKER_TYPE", "rrf").strip().lower()
    RAG_RANKER_WEIGHTS: list[float] = _parse_float_list(
        os.getenv("RAG_RANKER_WEIGHTS", ""),
        [0.5, 0.5],
    )
    RAG_RRF_K: int = int(os.getenv("RAG_RRF_K", "60"))
    RAG_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))

    # 缓存配置
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    CACHE_TTL_RECIPE: int = int(os.getenv("CACHE_TTL_RECIPE", "86400"))       # 菜谱缓存 24h
    CACHE_TTL_RESTAURANT: int = int(os.getenv("CACHE_TTL_RESTAURANT", "3600"))  # 餐厅缓存 1h
    CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.92"))
    CACHE_L2_ENABLED: bool = os.getenv("CACHE_L2_ENABLED", "true").lower() != "false"

    # RAG 开关
    RAG_ENABLED: bool = os.getenv("RAG_ENABLED", "true").lower() != "false"


settings = Settings()
