from __future__ import annotations

import asyncio
import importlib.util
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import settings
from memory.user_profile import (
    memory_context_summary,
    merge_user_profile,
    profile_from_dict,
    profile_to_summary,
    session_from_dict,
)
from models.memory import SessionMemory, UserProfile

logger = logging.getLogger(__name__)


def generate_user_id() -> str:
    return f"u_{uuid.uuid4().hex}"


def generate_session_id(user_id: str) -> str:
    ts = int(datetime.now(timezone.utc).timestamp() * 1000)
    return f"s_{user_id[2:10]}_{ts}"


def generate_thread_id(session_id: str) -> str:
    return f"t_{session_id[2:]}_001"


@dataclass
class SessionRuntime:
    user_id: str
    session_id: str
    thread_id: str


class BaseMemoryStore:
    def __init__(self):
        self._profiles: dict[str, UserProfile] = {}
        self._sessions: dict[str, dict[str, Any]] = {}

    async def initialize(self) -> None:
        return None

    async def get_or_create_user(self, channel: str, external_id: str) -> str:
        stable_key = f"{channel}:{external_id or 'anonymous'}"
        return f"u_{uuid.uuid5(uuid.NAMESPACE_URL, stable_key).hex}"

    async def get_or_create_session(
        self, user_id: str, requested_session_id: str | None = None
    ) -> SessionRuntime:
        if requested_session_id and requested_session_id in self._sessions:
            session = self._sessions[requested_session_id]
            return SessionRuntime(user_id, requested_session_id, session["thread_id"])

        active_session_id = None
        now = datetime.now(timezone.utc)
        for session_id, payload in self._sessions.items():
            if payload["user_id"] != user_id:
                continue
            if payload["expires_at"] <= now:
                continue
            active_session_id = session_id
            break

        session_id = active_session_id or requested_session_id or generate_session_id(user_id)
        payload = self._sessions.setdefault(
            session_id,
            {
                "user_id": user_id,
                "thread_id": generate_thread_id(session_id),
                "turn_no": 0,
                "expires_at": now + timedelta(hours=settings.SESSION_TTL_HOURS),
                "session_memory": SessionMemory(),
            },
        )
        return SessionRuntime(
            user_id=user_id,
            session_id=session_id,
            thread_id=payload["thread_id"],
        )

    async def next_turn(self, session_id: str) -> int:
        payload = self._sessions.setdefault(
            session_id,
            {
                "user_id": "",
                "thread_id": generate_thread_id(session_id),
                "turn_no": 0,
                "expires_at": datetime.now(timezone.utc)
                + timedelta(hours=settings.SESSION_TTL_HOURS),
                "session_memory": SessionMemory(),
            },
        )
        payload["turn_no"] += 1
        return payload["turn_no"]

    async def load_memory_context(self, user_id: str, session_id: str) -> dict[str, Any]:
        profile = self._profiles.get(user_id, UserProfile())
        session = self._sessions.get(session_id, {}).get("session_memory", SessionMemory())
        return {
            "profile": profile.model_dump(exclude_none=True),
            "session": session.model_dump(exclude_none=True),
            "memory_context_summary": memory_context_summary(profile, session),
            "profile_summary_for_rerank": profile_to_summary(profile),
        }

    async def save_session_memory(
        self,
        user_id: str,
        session_id: str,
        memory: SessionMemory,
    ) -> None:
        payload = self._sessions.setdefault(
            session_id,
            {
                "user_id": user_id,
                "thread_id": generate_thread_id(session_id),
                "turn_no": 0,
                "expires_at": datetime.now(timezone.utc)
                + timedelta(hours=settings.SESSION_TTL_HOURS),
                "session_memory": SessionMemory(),
            },
        )
        payload["user_id"] = user_id
        payload["expires_at"] = (
            datetime.fromisoformat(memory.expires_at) if memory.expires_at
            else datetime.now(timezone.utc) + timedelta(hours=settings.SESSION_TTL_HOURS)
        )
        payload["session_memory"] = memory

    async def apply_profile_update(self, user_id: str, candidate: UserProfile) -> None:
        existing = self._profiles.get(user_id, UserProfile())
        self._profiles[user_id] = merge_user_profile(existing, candidate)

    def schedule_profile_update(self, user_id: str, candidate: UserProfile) -> None:
        async def runner() -> None:
            await self.apply_profile_update(user_id, candidate)

        asyncio.create_task(runner())


class AsyncpgMemoryStore(BaseMemoryStore):
    def __init__(self, dsn: str):
        self.dsn = dsn

    async def _connect(self):
        import asyncpg

        return await asyncpg.connect(self.dsn)

    async def initialize(self) -> None:
        conn = await self._connect()
        try:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id VARCHAR(40) PRIMARY KEY,
                    channel VARCHAR(20),
                    external_id VARCHAR(255),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    UNIQUE(channel, external_id)
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id VARCHAR(80) PRIMARY KEY,
                    user_id VARCHAR(40) NOT NULL REFERENCES users(user_id),
                    status VARCHAR(20) NOT NULL DEFAULT 'active',
                    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    last_active_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    ended_at TIMESTAMPTZ,
                    expires_at TIMESTAMPTZ NOT NULL,
                    thread_id VARCHAR(100),
                    turn_no INTEGER NOT NULL DEFAULT 0
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    user_id VARCHAR(40) PRIMARY KEY REFERENCES users(user_id),
                    profile_version INTEGER NOT NULL,
                    profile_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    profile_summary TEXT NOT NULL DEFAULT '',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS session_memories (
                    session_id VARCHAR(80) PRIMARY KEY REFERENCES sessions(session_id),
                    user_id VARCHAR(40) NOT NULL REFERENCES users(user_id),
                    memory_json JSONB NOT NULL DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_user_status ON sessions(user_id, status)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_memories_user_id ON session_memories(user_id)"
            )
            await conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_sessions_user_active ON sessions(user_id) WHERE status = 'active'"
            )
        finally:
            await conn.close()

    async def get_or_create_user(self, channel: str, external_id: str) -> str:
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                SELECT user_id
                FROM users
                WHERE channel = $1 AND external_id = $2
                """,
                channel,
                external_id,
            )
            if row:
                return row["user_id"]

            user_id = generate_user_id()
            await conn.execute(
                """
                INSERT INTO users (user_id, channel, external_id)
                VALUES ($1, $2, $3)
                """,
                user_id,
                channel,
                external_id,
            )
            return user_id
        finally:
            await conn.close()

    async def get_or_create_session(
        self, user_id: str, requested_session_id: str | None = None
    ) -> SessionRuntime:
        conn = await self._connect()
        try:
            async with conn.transaction():
                if requested_session_id:
                    row = await conn.fetchrow(
                        """
                        SELECT session_id, thread_id
                        FROM sessions
                        WHERE session_id = $1 AND user_id = $2
                        FOR UPDATE
                        """,
                        requested_session_id,
                        user_id,
                    )
                    if row:
                        thread_id = row["thread_id"] or generate_thread_id(row["session_id"])
                        if row["thread_id"] is None:
                            await conn.execute(
                                "UPDATE sessions SET thread_id = $2 WHERE session_id = $1",
                                row["session_id"],
                                thread_id,
                            )
                        return SessionRuntime(user_id, row["session_id"], thread_id)

                row = await conn.fetchrow(
                    """
                    SELECT session_id, thread_id
                    FROM sessions
                    WHERE user_id = $1
                      AND status = 'active'
                      AND expires_at > NOW()
                    ORDER BY started_at DESC
                    LIMIT 1
                    FOR UPDATE
                    """,
                    user_id,
                )
                if row:
                    thread_id = row["thread_id"] or generate_thread_id(row["session_id"])
                    if row["thread_id"] is None:
                        await conn.execute(
                            "UPDATE sessions SET thread_id = $2 WHERE session_id = $1",
                            row["session_id"],
                            thread_id,
                        )
                    return SessionRuntime(user_id, row["session_id"], thread_id)

                session_id = generate_session_id(user_id)
                thread_id = generate_thread_id(session_id)
                expires_at = datetime.now(timezone.utc) + timedelta(
                    hours=settings.SESSION_TTL_HOURS
                )
                await conn.execute(
                    """
                    INSERT INTO sessions (
                        session_id, user_id, expires_at, thread_id
                    ) VALUES ($1, $2, $3, $4)
                    """,
                    session_id,
                    user_id,
                    expires_at,
                    thread_id,
                )
                return SessionRuntime(user_id, session_id, thread_id)
        finally:
            await conn.close()

    async def next_turn(self, session_id: str) -> int:
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                UPDATE sessions
                SET turn_no = turn_no + 1,
                    last_active_at = NOW()
                WHERE session_id = $1
                RETURNING turn_no
                """,
                session_id,
            )
            return int(row["turn_no"]) if row else 1
        finally:
            await conn.close()

    async def load_memory_context(self, user_id: str, session_id: str) -> dict[str, Any]:
        conn = await self._connect()
        try:
            profile_row = await conn.fetchrow(
                """
                SELECT profile_json
                FROM user_profiles
                WHERE user_id = $1
                """,
                user_id,
            )
            session_row = await conn.fetchrow(
                """
                SELECT memory_json
                FROM session_memories
                WHERE session_id = $1
                  AND user_id = $2
                  AND expires_at > NOW()
                """,
                session_id,
                user_id,
            )
        finally:
            await conn.close()

        profile = profile_from_dict(
            profile_row["profile_json"] if profile_row is not None else None
        )
        session = session_from_dict(
            session_row["memory_json"] if session_row is not None else None
        )
        return {
            "profile": profile.model_dump(exclude_none=True),
            "session": session.model_dump(exclude_none=True),
            "memory_context_summary": memory_context_summary(profile, session),
            "profile_summary_for_rerank": profile_to_summary(profile),
        }

    async def save_session_memory(
        self,
        user_id: str,
        session_id: str,
        memory: SessionMemory,
    ) -> None:
        conn = await self._connect()
        try:
            expires_at = datetime.fromisoformat(memory.expires_at) if memory.expires_at else (
                datetime.now(timezone.utc) + timedelta(hours=settings.SESSION_TTL_HOURS)
            )
            await conn.execute(
                """
                INSERT INTO session_memories (
                    session_id, user_id, memory_json, updated_at, expires_at
                ) VALUES ($1, $2, $3::jsonb, NOW(), $4)
                ON CONFLICT (session_id)
                DO UPDATE SET
                    memory_json = EXCLUDED.memory_json,
                    updated_at = NOW(),
                    expires_at = EXCLUDED.expires_at
                """,
                session_id,
                user_id,
                memory.model_dump_json(exclude_none=True),
                expires_at,
            )
            await conn.execute(
                """
                UPDATE sessions
                SET last_active_at = NOW(),
                    expires_at = $2
                WHERE session_id = $1
                """,
                session_id,
                expires_at,
            )
        finally:
            await conn.close()

    async def apply_profile_update(self, user_id: str, candidate: UserProfile) -> None:
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                SELECT profile_version, profile_json
                FROM user_profiles
                WHERE user_id = $1
                """,
                user_id,
            )
            existing = profile_from_dict(row["profile_json"] if row else None)
            merged = merge_user_profile(existing, candidate)
            summary = profile_to_summary(merged)
            await conn.execute(
                """
                INSERT INTO user_profiles (
                    user_id, profile_version, profile_json, profile_summary
                ) VALUES ($1, $2, $3::jsonb, $4)
                ON CONFLICT (user_id)
                DO UPDATE SET
                    profile_version = EXCLUDED.profile_version,
                    profile_json = EXCLUDED.profile_json,
                    profile_summary = EXCLUDED.profile_summary,
                    updated_at = NOW()
                """,
                user_id,
                merged.schema_version,
                merged.model_dump_json(exclude_none=True),
                summary,
            )
        finally:
            await conn.close()

    def schedule_profile_update(self, user_id: str, candidate: UserProfile) -> None:
        async def runner() -> None:
            try:
                await self.apply_profile_update(user_id, candidate)
            except Exception as exc:  # pragma: no cover - background path
                logger.exception("[memory] 异步写入用户画像失败: %s", exc)

        asyncio.create_task(runner())


_memory_store: BaseMemoryStore | None = None


async def init_memory_store() -> BaseMemoryStore:
    global _memory_store
    if _memory_store is not None:
        return _memory_store

    if not settings.MEMORY_ENABLED:
        logger.info("[memory] MEMORY_ENABLED=false，使用空实现")
        _memory_store = BaseMemoryStore()
        return _memory_store

    if not settings.DATABASE_URL:
        logger.warning("[memory] 未配置 DATABASE_URL，记忆系统退化为空实现")
        _memory_store = BaseMemoryStore()
        return _memory_store

    if importlib.util.find_spec("asyncpg") is None:
        logger.warning("[memory] 未安装 asyncpg，记忆系统退化为空实现")
        _memory_store = BaseMemoryStore()
        return _memory_store

    store = AsyncpgMemoryStore(settings.DATABASE_URL)
    await store.initialize()
    logger.info("[memory] PostgreSQL 记忆存储初始化完成")
    _memory_store = store
    return _memory_store


def get_memory_store() -> BaseMemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = BaseMemoryStore()
    return _memory_store
