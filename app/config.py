from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",  # 从项目根目录的 .env 文件读
        env_file_encoding="utf-8",
        extra="ignore",  # .env 里有多余变量不报错
    )

    app_env: Literal["dev", "prod"] = "dev" # 区分开发和生产环境
    debug: bool = True
    api_v1_prefix: str = "/api/v1"

    # 注：DATABASE_URL 由中立模块 core/db.py 统一从 config.settings 读取，
    # 后端配置只保留 HTTP / 服务层相关开关。

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
