from __future__ import annotations

import importlib.util
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import settings
from memory.user_profile import (
    build_memory_for_intent,
    build_memory_for_rerank,
    now_iso,
    profile_from_dict,
    session_from_dict,
)
from models.memory import MemoryFact, ProfileUpdate, SessionMemory, UserProfile

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


def _fact_values(facts: list[dict] | list[MemoryFact]) -> list[str]:
    values: list[str] = []
    for fact in facts:
        if isinstance(fact, MemoryFact):
            values.append(fact.value)
        elif isinstance(fact, dict) and fact.get("value"):
            values.append(str(fact["value"]))
    return values


def _build_memory_payload(profile: UserProfile, session: SessionMemory) -> dict[str, Any]:
    profile_dict = profile.model_dump(exclude_none=True)
    session_dict = session.model_dump(exclude_none=True)
    intent_data = {
        "profile": profile_dict,
        "session": session_dict,
        "allergies": _fact_values(profile.allergies),
        "food_blacklist": _fact_values(profile.food_blacklist),
        "religious_restrictions": _fact_values(profile.religious_restrictions),
        "disliked_cuisines": _fact_values(profile.disliked_cuisines),
        "active_city": session.active_city,
        "active_location_text": session.active_location_text,
        "active_negative_conditions": session.active_negative_conditions,
    }
    rerank_data = {
        "profile": profile_dict,
        "session": session_dict,
        "allergies": _fact_values(profile.allergies),
        "food_blacklist": _fact_values(profile.food_blacklist),
        "religious_restrictions": _fact_values(profile.religious_restrictions),
        "disliked_cuisines": _fact_values(profile.disliked_cuisines),
        "cuisine_tags": profile.cuisine_tags,
        "spice_tolerance": profile.spice_tolerance.model_dump(exclude_none=True)
        if profile.spice_tolerance
        else None,
        "sweetness": profile.sweetness.model_dump(exclude_none=True)
        if profile.sweetness
        else None,
        "health_goals": _fact_values(profile.health_goals),
        "budget_solo": profile.budget_solo.model_dump(exclude_none=True)
        if profile.budget_solo
        else None,
        "budget_group": profile.budget_group.model_dump(exclude_none=True)
        if profile.budget_group
        else None,
        "active_negative_conditions": session.active_negative_conditions,
    }
    return {
        "memory_for_intent": build_memory_for_intent(profile, session),
        "memory_for_rerank": build_memory_for_rerank(profile, session),
        "memory_for_intent_data": intent_data,
        "memory_for_rerank_data": rerank_data,
    }


def _make_memory_fact(value: str, source: str = "LLM推断", confidence: float = 0.8) -> MemoryFact:
    timestamp = now_iso()
    return MemoryFact(
        value=value,
        source=source,
        confidence=confidence,
        updated_at=timestamp,
        last_seen_at=timestamp,
    )


def _append_unique_fact(facts: list[MemoryFact], value: str, source: str = "LLM推断") -> None:
    if any(fact.value == value for fact in facts):
        return
    facts.append(_make_memory_fact(value=value, source=source))


def _apply_profile_updates(profile: UserProfile, updates: list[ProfileUpdate]) -> UserProfile:
    updated = profile.model_copy(deep=True)

    list_fields = {
        "allergies",
        "food_blacklist",
        "religious_restrictions",
        "disliked_cuisines",
        "health_goals",
        "active_areas",
    }
    scalar_fields = {
        "spice_tolerance",
        "sweetness",
        "home_area",
        "budget_solo",
        "budget_group",
        "default_city",
    }

    for instruction in updates:
        field_parts = instruction.field.split(".")
        root_field = field_parts[0]

        if root_field == "cuisine_tags" and len(field_parts) == 2:
            cuisine = field_parts[1]
            if instruction.action == "add":
                updated.cuisine_tags[cuisine] = instruction.tag_level or "liked"
            elif instruction.action == "remove":
                updated.cuisine_tags.pop(cuisine, None)
            elif instruction.action == "set":
                updated.cuisine_tags[cuisine] = instruction.tag_level or instruction.value or "liked"
            continue

        if root_field in list_fields:
            fact_list = getattr(updated, root_field)
            if instruction.action == "add":
                _append_unique_fact(fact_list, instruction.value)
            elif instruction.action == "remove":
                setattr(updated, root_field, [fact for fact in fact_list if fact.value != instruction.value])
            continue

        if root_field in scalar_fields:
            if instruction.action == "remove":
                setattr(updated, root_field, None)
            elif instruction.action == "set":
                setattr(updated, root_field, _make_memory_fact(instruction.value))

    return updated


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
        now = datetime.now(timezone.utc)
        if requested_session_id and requested_session_id in self._sessions:
            payload = self._sessions[requested_session_id]
            return SessionRuntime(user_id, requested_session_id, payload["thread_id"])

        active_session_id = None
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
        return SessionRuntime(user_id=user_id, session_id=session_id, thread_id=payload["thread_id"])

    async def next_turn(self, session_id: str) -> int:
        payload = self._sessions.setdefault(
            session_id,
            {
                "user_id": "",
                "thread_id": generate_thread_id(session_id),
                "turn_no": 0,
                "expires_at": datetime.now(timezone.utc) + timedelta(hours=settings.SESSION_TTL_HOURS),
                "session_memory": SessionMemory(),
            },
        )
        payload["turn_no"] += 1
        return payload["turn_no"]

    async def load_profile(self, user_id: str) -> UserProfile:
        return self._profiles.get(user_id, UserProfile())

    async def _save_profile(self, user_id: str, profile: UserProfile) -> None:
        self._profiles[user_id] = profile

    async def _load_session_memory(self, session_id: str) -> SessionMemory:
        return self._sessions.get(session_id, {}).get("session_memory", SessionMemory())

    async def load_memory_context(self, user_id: str, session_id: str) -> dict[str, Any]:
        profile = await self.load_profile(user_id)
        session = await self._load_session_memory(session_id)
        return _build_memory_payload(profile, session)

    async def save_session_memory(self, user_id: str, session_id: str, memory: SessionMemory) -> None:
        payload = self._sessions.setdefault(
            session_id,
            {
                "user_id": user_id,
                "thread_id": generate_thread_id(session_id),
                "turn_no": 0,
                "expires_at": datetime.now(timezone.utc) + timedelta(hours=settings.SESSION_TTL_HOURS),
                "session_memory": SessionMemory(),
            },
        )
        payload["user_id"] = user_id
        payload["expires_at"] = (
            datetime.fromisoformat(memory.expires_at)
            if memory.expires_at
            else datetime.now(timezone.utc) + timedelta(hours=settings.SESSION_TTL_HOURS)
        )
        payload["session_memory"] = memory

    async def apply_profile_updates(self, user_id: str, updates: list[ProfileUpdate]) -> None:
        profile = await self.load_profile(user_id)
        updated = _apply_profile_updates(profile, updates)
        await self._save_profile(user_id, updated)


class AsyncpgMemoryStore(BaseMemoryStore):
    def __init__(self, dsn: str):
        super().__init__()
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
                expires_at = datetime.now(timezone.utc) + timedelta(hours=settings.SESSION_TTL_HOURS)
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

    async def load_profile(self, user_id: str) -> UserProfile:
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                SELECT profile_json
                FROM user_profiles
                WHERE user_id = $1
                """,
                user_id,
            )
        finally:
            await conn.close()
        return profile_from_dict(row["profile_json"] if row else None)

    async def _save_profile(self, user_id: str, profile: UserProfile) -> None:
        conn = await self._connect()
        try:
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
                profile.schema_version,
                profile.model_dump_json(exclude_none=True),
                build_memory_for_rerank(profile, SessionMemory()),
            )
        finally:
            await conn.close()

    async def _load_session_memory(self, session_id: str) -> SessionMemory:
        conn = await self._connect()
        try:
            row = await conn.fetchrow(
                """
                SELECT memory_json
                FROM session_memories
                WHERE session_id = $1
                  AND expires_at > NOW()
                """,
                session_id,
            )
        finally:
            await conn.close()
        return session_from_dict(row["memory_json"] if row else None)

    async def load_memory_context(self, user_id: str, session_id: str) -> dict[str, Any]:
        profile = await self.load_profile(user_id)
        session = await self._load_session_memory(session_id)
        return _build_memory_payload(profile, session)

    async def save_session_memory(self, user_id: str, session_id: str, memory: SessionMemory) -> None:
        conn = await self._connect()
        try:
            expires_at = (
                datetime.fromisoformat(memory.expires_at)
                if memory.expires_at
                else datetime.now(timezone.utc) + timedelta(hours=settings.SESSION_TTL_HOURS)
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

    async def apply_profile_updates(self, user_id: str, updates: list[ProfileUpdate]) -> None:
        profile = await self.load_profile(user_id)
        updated = _apply_profile_updates(profile, updates)
        await self._save_profile(user_id, updated)


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
