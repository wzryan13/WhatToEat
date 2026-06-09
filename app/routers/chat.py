import json
import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.chat_engine import ChatEngine
from app.exceptions import ServiceUnavailableError
from app.schemas import ChatRequest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


def get_chat_engine(request: Request) -> ChatEngine:
    """从 app.state 取 lifespan 初始化好的对话引擎；未就绪则 503。"""
    engine = getattr(request.app.state, "chat_engine", None)
    if engine is None or not engine.ready:
        raise ServiceUnavailableError(
            "对话引擎未就绪（缺少 langgraph/MCP/RAG 依赖或初始化失败）"
        )
    return engine


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@router.post("")
async def chat(
    payload: ChatRequest,
    engine: ChatEngine = Depends(get_chat_engine),
) -> StreamingResponse:
    async def event_stream():
        try:
            async for event in engine.astream(
                payload.message, payload.thread_id, payload.user_id
            ):
                yield _sse(event)
        except Exception as exc:  # 流已开始，只能以 error 事件告知客户端
            logger.exception("[chat] 流式处理异常: %s", exc)
            yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
