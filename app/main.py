import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.chat_engine import ChatEngine
from app.config import settings
from app.exceptions import register_exception_handlers
from app.routers import chat, health, items

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动期初始化对话引擎；失败不阻断服务启动，/chat 会返回 503
    engine = ChatEngine()
    try:
        await engine.init()
    except Exception as exc:
        logger.warning("[lifespan] 对话引擎初始化失败，/chat 将返回 503：%s", exc)
    app.state.chat_engine = engine
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="饮食管家 API",
        debug=settings.debug,
        lifespan=lifespan,
    )

    register_exception_handlers(app)

    # /health 不带版本前缀，供存活探针使用
    app.include_router(health.router)
    # 业务路由统一挂在 /api/v1 下
    app.include_router(items.router, prefix=settings.api_v1_prefix)
    app.include_router(chat.router, prefix=settings.api_v1_prefix)

    return app


app = create_app()
