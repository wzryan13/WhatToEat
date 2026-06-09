"""对话引擎：把 LangGraph 主图封装成可被 HTTP/SSE 调用的服务。

所有重依赖（langgraph / langchain / MCP / RAG）都在 init()/astream() 内部懒加载，
保证 import app.chat_engine 本身是轻量的——即使环境没装这些依赖，
app.main 仍可正常 import、/health 仍可用，只是 /chat 会返回 503。
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Any

logger = logging.getLogger(__name__)

# 与 graph.py 注册的节点保持一致，用于过滤 astream_events 的节点状态事件
NODE_NAMES: set[str] = {
    "memory_read",
    "intent_parser",
    "search_agent",
    "rag_agent",
    "clarify",
    "error_output",
    "result_formatter",
    "rag_formatter",
    "memory_write",
}


class ChatEngine:
    """单例对话引擎。生命周期由 FastAPI lifespan 管理。"""

    def __init__(self) -> None:
        self._graph: Any = None
        self._store: Any = None
        self._ready = False

    @property
    def ready(self) -> bool:
        return self._ready

    async def init(self) -> None:
        """启动期初始化：MCP 工具、记忆存储、RAG、编译主图。重依赖在此懒加载。"""
        from graph import build_graph
        from memory.store import get_memory_store, init_memory_store
        from rag.rag_service import init_rag_service
        from tools import init_tools

        await init_tools()
        await init_memory_store()
        init_rag_service()  # 同步函数，失败不影响餐厅链路
        self._graph = build_graph()
        self._store = get_memory_store()
        self._ready = True
        logger.info("[chat_engine] 初始化完成")

    async def astream(
        self,
        message: str,
        thread_id: str | None = None,
        external_id: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        """驱动主图并以事件流形式产出：session / node / result|interrupt / done。

        会话状态由 LangGraph MemorySaver 按 thread_id 持久化：
        - 无 checkpoint：新建 user+session，按完整 initial_state 启动
        - 有 checkpoint 且处于 clarify 中断：Command(resume=message) 恢复
        - 有 checkpoint 普通续聊：只递增 turn_no + 新 user_input
        """
        from langgraph.types import Command

        store = self._store
        graph = self._graph

        config = {"configurable": {"thread_id": thread_id}} if thread_id else None
        snapshot = graph.get_state(config) if config else None
        has_checkpoint = bool(snapshot and getattr(snapshot, "values", None))
        is_interrupted = bool(snapshot and snapshot.next) and "clarify" in (
            snapshot.next or ()
        )

        if has_checkpoint:
            user_id = snapshot.values.get("user_id")
            session_id = snapshot.values.get("session_id")
        else:
            ext = external_id or uuid.uuid4().hex
            user_id = await store.get_or_create_user("api", ext)
            runtime = await store.get_or_create_session(user_id)
            session_id = runtime.session_id
            thread_id = runtime.thread_id
            config = {"configurable": {"thread_id": thread_id}}

        # 先把真实 thread_id 回传给客户端，供后续续聊使用
        yield {"type": "session", "thread_id": thread_id, "user_id": user_id}

        turn_no = await store.next_turn(session_id)
        if is_interrupted:
            payload: Any = Command(resume=message)
        elif has_checkpoint:
            payload = {"user_input": message, "turn_no": turn_no}
        else:
            payload = {
                "user_id": user_id,
                "session_id": session_id,
                "thread_id": thread_id,
                "turn_no": turn_no,
                "user_input": message,
                "conversation_history": [],
            }

        async for event in graph.astream_events(payload, config, version="v2"):
            name = event.get("name", "")
            if name not in NODE_NAMES:
                continue
            kind = event.get("event", "")
            if kind == "on_chain_start":
                yield {"type": "node", "name": name, "status": "running"}
            elif kind == "on_chain_end":
                yield {"type": "node", "name": name, "status": "done"}

        snapshot = graph.get_state(config)
        state = snapshot.values or {}

        interrupt_value = None
        for task in snapshot.tasks:
            if getattr(task, "interrupts", None):
                interrupt_value = task.interrupts[0].value
                break

        if interrupt_value:
            yield {"type": "interrupt", "message": interrupt_value}
        else:
            yield {
                "type": "result",
                "intent": state.get("intent_type", "normal"),
                "response": state.get("response_message", ""),
                "recommendations": state.get("final_recommendations", []) or [],
            }
        yield {"type": "done"}
