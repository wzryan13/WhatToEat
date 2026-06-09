import json

import pytest

from app.main import app
from app.routers.chat import get_chat_engine


class FakeEngine:
    """不依赖 langgraph 的假引擎，用于验证 SSE 管路本身。"""

    ready = True

    async def astream(self, message, thread_id=None, external_id=None):
        yield {"type": "session", "thread_id": "t_test", "user_id": "u_test"}
        yield {"type": "node", "name": "memory_read", "status": "running"}
        yield {"type": "node", "name": "memory_read", "status": "done"}
        yield {
            "type": "result",
            "intent": "recipe",
            "response": f"echo:{message}",
            "recommendations": [],
        }
        yield {"type": "done"}


def _parse_sse(text: str) -> list[dict]:
    return [
        json.loads(line[len("data: ") :])
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


@pytest.mark.asyncio
async def test_chat_sse_stream(client):
    app.dependency_overrides[get_chat_engine] = lambda: FakeEngine()
    try:
        resp = await client.post("/api/v1/chat", json={"message": "教我做番茄炒蛋"})
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        events = _parse_sse(resp.text)
        types = [e["type"] for e in events]
        assert types[0] == "session"
        assert types[-1] == "done"
        assert "node" in types and "result" in types

        result = next(e for e in events if e["type"] == "result")
        assert result["response"] == "echo:教我做番茄炒蛋"
    finally:
        app.dependency_overrides.pop(get_chat_engine, None)


@pytest.mark.asyncio
async def test_chat_unavailable_returns_503(client):
    # 未经 lifespan 初始化（app.state 无 chat_engine）→ 统一 503 错误体
    resp = await client.post("/api/v1/chat", json={"message": "hi"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "service_unavailable"


@pytest.mark.asyncio
async def test_chat_rejects_empty_message(client):
    # 引擎可用时，空 message 应被请求体校验拦下（422），而非 503
    app.dependency_overrides[get_chat_engine] = lambda: FakeEngine()
    try:
        resp = await client.post("/api/v1/chat", json={"message": ""})
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.pop(get_chat_engine, None)
