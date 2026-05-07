import os
from dotenv import load_dotenv

load_dotenv("/Users/wenzhouzhou/PycharmProjects/PythonProject1/.env")

class Settings:
    # ── API Keys ─────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    AMAP_API_KEY: str = os.getenv("GROQ_API_KEY")

    # ── 模型配置 ──────────────────────────────────────────────
    LLM_MODEL: str = "openai/gpt-oss-120b"

    # ── 搜索参数 ──────────────────────────────────────────────
    POI_DETAIL_LIMIT: int = 15  # 并发查POI详情的上限条数
    DEFAULT_RADIUS: int = 1000  # 周边搜默认半径（米）
    MAX_CLARIFICATION: int = 10  # 最终推荐数量上限
    MAX_RECOMMENDATIONS: int = 2  # 最多追问次数


settings = Settings()