from __future__ import annotations

import importlib.util
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config.settings import settings
from core.db import SessionLocal
from memory import orm
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


def _as_dict(raw: Any) -> dict | None:
    """JSONB 列正常返回 dict；个别驱动可能返回 JSON 字符串，这里兜底。"""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    return raw


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


class SqlAlchemyMemoryStore(BaseMemoryStore):
    """基于 SQLAlchemy async 的持久化记忆存储（Postgres）。

    public 方法签名与 BaseMemoryStore 完全一致，对 LangGraph 节点 / CLI 透明。
    schema 由 Alembic 管理，本类不负责建表。
    load_memory_context / apply_profile_updates 复用基类实现（内部调用下面被重写的方法）。
    """

    async def initialize(self) -> None:
        return None

    async def get_or_create_user(self, channel: str, external_id: str) -> str:
        async with SessionLocal() as session:
            async with session.begin():
                existing = await session.scalar(
                    select(orm.User.user_id).where(
                        orm.User.channel == channel,
                        orm.User.external_id == external_id,
                    )
                )
                if existing:
                    return existing
                user_id = generate_user_id()
                session.add(
                    orm.User(user_id=user_id, channel=channel, external_id=external_id)
                )
            return user_id

    async def get_or_create_session(
        self, user_id: str, requested_session_id: str | None = None
    ) -> SessionRuntime:
        async with SessionLocal() as session:
            async with session.begin():
                if requested_session_id:
                    row = await session.scalar(
                        select(orm.Session)
                        .where(
                            orm.Session.session_id == requested_session_id,
                            orm.Session.user_id == user_id,
                        )
                        .with_for_update()
                    )
                    if row:
                        thread_id = row.thread_id or generate_thread_id(row.session_id)
                        if row.thread_id is None:
                            row.thread_id = thread_id
                        return SessionRuntime(user_id, row.session_id, thread_id)

                row = await session.scalar(
                    select(orm.Session)
                    .where(
                        orm.Session.user_id == user_id,
                        orm.Session.expires_at > func.now(),
                    )
                    .order_by(orm.Session.started_at.desc())
                    .limit(1)
                    .with_for_update()
                )
                if row:
                    thread_id = row.thread_id or generate_thread_id(row.session_id)
                    if row.thread_id is None:
                        row.thread_id = thread_id
                    return SessionRuntime(user_id, row.session_id, thread_id)

                session_id = generate_session_id(user_id)
                thread_id = generate_thread_id(session_id)
                expires_at = datetime.now(timezone.utc) + timedelta(
                    hours=settings.SESSION_TTL_HOURS
                )
                session.add(
                    orm.Session(
                        session_id=session_id,
                        user_id=user_id,
                        expires_at=expires_at,
                        thread_id=thread_id,
                    )
                )
                return SessionRuntime(user_id, session_id, thread_id)

    async def next_turn(self, session_id: str) -> int:
        async with SessionLocal() as session:
            async with session.begin():
                result = await session.execute(
                    update(orm.Session)
                    .where(orm.Session.session_id == session_id)
                    .values(turn_no=orm.Session.turn_no + 1, last_active_at=func.now())
                    .returning(orm.Session.turn_no)
                )
                turn_no = result.scalar_one_or_none()
            return int(turn_no) if turn_no is not None else 1

    async def load_profile(self, user_id: str) -> UserProfile:
        async with SessionLocal() as session:
            raw = await session.scalar(
                select(orm.UserProfile.profile_json).where(
                    orm.UserProfile.user_id == user_id
                )
            )
        return profile_from_dict(_as_dict(raw))

    async def _save_profile(self, user_id: str, profile: UserProfile) -> None:
        stmt = pg_insert(orm.UserProfile).values(
            user_id=user_id,
            profile_version=profile.schema_version,
            profile_json=profile.model_dump(mode="json", exclude_none=True),
            profile_summary=build_memory_for_rerank(profile, SessionMemory()),
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=[orm.UserProfile.user_id],
            set_={
                "profile_version": stmt.excluded.profile_version,
                "profile_json": stmt.excluded.profile_json,
                "profile_summary": stmt.excluded.profile_summary,
                "updated_at": func.now(),
            },
        )
        async with SessionLocal() as session:
            async with session.begin():
                await session.execute(stmt)

    async def _load_session_memory(self, session_id: str) -> SessionMemory:
        async with SessionLocal() as session:
            raw = await session.scalar(
                select(orm.SessionMemory.memory_json).where(
                    orm.SessionMemory.session_id == session_id,
                    orm.SessionMemory.expires_at > func.now(),
                )
            )
        return session_from_dict(_as_dict(raw))

    async def save_session_memory(
        self, user_id: str, session_id: str, memory: SessionMemory
    ) -> None:
        expires_at = (
            datetime.fromisoformat(memory.expires_at)
            if memory.expires_at
            else datetime.now(timezone.utc) + timedelta(hours=settings.SESSION_TTL_HOURS)
        )
        mem_stmt = pg_insert(orm.SessionMemory).values(
            session_id=session_id,
            user_id=user_id,
            memory_json=memory.model_dump(mode="json", exclude_none=True),
            expires_at=expires_at,
        )
        mem_stmt = mem_stmt.on_conflict_do_update(
            index_elements=[orm.SessionMemory.session_id],
            set_={
                "memory_json": mem_stmt.excluded.memory_json,
                "updated_at": func.now(),
                "expires_at": mem_stmt.excluded.expires_at,
            },
        )
        async with SessionLocal() as session:
            async with session.begin():
                await session.execute(mem_stmt)
                await session.execute(
                    update(orm.Session)
                    .where(orm.Session.session_id == session_id)
                    .values(last_active_at=func.now(), expires_at=expires_at)
                )


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

    store = SqlAlchemyMemoryStore()
    await store.initialize()
    logger.info("[memory] SQLAlchemy 记忆存储初始化完成（schema 由 Alembic 管理）")
    _memory_store = store
    return _memory_store


def get_memory_store() -> BaseMemoryStore:
    global _memory_store
    if _memory_store is None:
        _memory_store = BaseMemoryStore()
    return _memory_store
