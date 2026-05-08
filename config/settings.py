import os
from dotenv import load_dotenv

load_dotenv("/Users/wenzhouzhou/PycharmProjects/PythonProject1/.env")


class Settings:
    # ── API Keys ─────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    AMAP_API_KEY: str = os.getenv("AMAP_API_KEY", os.getenv("GROQ_API_KEY", ""))
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # ── 模型配置 ──────────────────────────────────────────────
    LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
    MODEL_NAME: str = LLM_MODEL

    # ── 记忆系统 ──────────────────────────────────────────────
    MEMORY_ENABLED: bool = os.getenv("MEMORY_ENABLED", "true").lower() != "false"
    SESSION_TTL_HOURS: int = int(os.getenv("SESSION_TTL_HOURS", "4"))
    DEMO_CHANNEL: str = os.getenv("DEMO_CHANNEL", "cli")
    DEMO_EXTERNAL_ID: str = os.getenv("DEMO_EXTERNAL_ID", "local-cli-user")

    # ── 搜索参数 ──────────────────────────────────────────────
    POI_DETAIL_LIMIT: int = 15  # 并发查POI详情的上限条数
    DEFAULT_RADIUS: int = 1000  # 周边搜默认半径（米）
    MAX_CLARIFICATION: int = 10  # 最多追问次数
    MAX_RECOMMENDATIONS: int = 2  # 最终推荐数量上限


settings = Settings()
