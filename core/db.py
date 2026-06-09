"""全项目统一的数据库基础设施（中立模块）。

app/（FastAPI）和 memory/（LangGraph 记忆存储）都从这里取 engine / session，
保证全项目只有一套 DB 访问方式。本模块只依赖根 config，不依赖 app 包。
"""
import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from config.settings import settings


class Base(DeclarativeBase):
    pass


def _to_async_url(url: str) -> str:
    """SQLAlchemy async 需要 postgresql+asyncpg:// 驱动前缀。"""
    if url.startswith("postgresql+asyncpg://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


ASYNC_DATABASE_URL: str = _to_async_url(settings.DATABASE_URL)

# DATABASE_URL 未配置时 engine 为 None，记忆系统会降级为内存实现。
engine = (
    create_async_engine(
        ASYNC_DATABASE_URL,
        echo=os.getenv("DB_ECHO", "false").lower() == "true",
        pool_pre_ping=True,
    )
    if ASYNC_DATABASE_URL
    else None
)

SessionLocal = (
    async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )
    if engine is not None
    else None
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """请求级 DB session 依赖：成功提交、异常回滚、最终关闭。

    通过 DI 注入，测试时可用 app.dependency_overrides 替换成 mock/内存库。
    """
    if SessionLocal is None:
        raise RuntimeError("DATABASE_URL 未配置，无法创建数据库会话")
    session = SessionLocal()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
