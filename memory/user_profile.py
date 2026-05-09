from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from models.memory import MemoryFact, SessionMemory, UserProfile


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    return value.strip()


def dedupe_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned = normalize_text(value)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def _dedupe_facts(facts: Iterable[MemoryFact]) -> list[MemoryFact]:
    merged: dict[str, MemoryFact] = {}
    for fact in facts:
        key = normalize_text(fact.value)
        existing = merged.get(key)
        if existing is None:
            merged[key] = fact.model_copy(deep=True)
            continue

        if fact.confidence > existing.confidence:
            merged[key] = fact.model_copy(deep=True)
            continue

        existing.last_seen_at = fact.last_seen_at
        existing.updated_at = fact.updated_at
    return list(merged.values())


def _migrate_profile_data(data: dict) -> dict:
    version = data.get("schema_version", 1)
    if version < 2:
        for fact_list_key in ("dietary_restrictions", "allergies"):
            for fact in data.get(fact_list_key, []):
                fact.setdefault("seen_count", 1)
                fact.setdefault("first_seen_at", fact.get("updated_at", ""))
        if data.get("default_city") and isinstance(data["default_city"], dict):
            data["default_city"].setdefault("seen_count", 1)
            data["default_city"].setdefault("first_seen_at", data["default_city"].get("updated_at", ""))
    return data


def profile_from_dict(data: dict | None) -> UserProfile:
    if not data:
        return UserProfile()
    migrated = _migrate_profile_data(data)
    return UserProfile.model_validate(migrated)


def session_from_dict(data: dict | None) -> SessionMemory:
    if not data:
        return SessionMemory()
    return SessionMemory.model_validate(data)


def merge_user_profile(existing: UserProfile, candidate: UserProfile) -> UserProfile:
    merged = existing.model_copy(deep=True)
    merged.schema_version = max(existing.schema_version, candidate.schema_version)

    if candidate.default_city:
        if merged.default_city is None:
            merged.default_city = candidate.default_city.model_copy(deep=True)
        elif merged.default_city.value == candidate.default_city.value:
            merged.default_city.last_seen_at = candidate.default_city.last_seen_at
            merged.default_city.updated_at = candidate.default_city.updated_at
        elif candidate.default_city.confidence >= 0.8:
            merged.default_city = candidate.default_city.model_copy(deep=True)

    merged.dietary_restrictions = _dedupe_facts(
        [*merged.dietary_restrictions, *candidate.dietary_restrictions]
    )
    merged.allergies = _dedupe_facts([*merged.allergies, *candidate.allergies])
    return merged


def profile_to_summary(profile: UserProfile) -> str:
    lines: list[str] = []

    if profile.default_city:
        lines.append(f"- 默认城市：{profile.default_city.value}")
    if profile.dietary_restrictions:
        lines.append(
            "- 饮食限制：" + "、".join(fact.value for fact in profile.dietary_restrictions)
        )
    if profile.allergies:
        lines.append("- 过敏信息：" + "、".join(fact.value for fact in profile.allergies))

    if not lines:
        return "暂无长期用户画像。"

    return "用户长期信息：\n" + "\n".join(lines)


def memory_context_summary(profile: UserProfile, session: SessionMemory) -> str:
    sections: list[str] = [profile_to_summary(profile)]

    session_lines: list[str] = []
    if session.active_city:
        session_lines.append(f"- 当前城市：{session.active_city}")
    if session.active_location_text:
        session_lines.append(f"- 当前位置：{session.active_location_text}")
    if session.active_budget_range and (
        session.active_budget_range.min is not None
        or session.active_budget_range.max is not None
    ):
        budget_parts: list[str] = []
        if session.active_budget_range.min is not None:
            budget_parts.append(f"最低{session.active_budget_range.min:g}元")
        if session.active_budget_range.max is not None:
            budget_parts.append(f"最高{session.active_budget_range.max:g}元")
        session_lines.append("- 当前预算：" + "，".join(budget_parts))
    if session.active_negative_conditions:
        session_lines.append(
            "- 当前负向条件：" + "、".join(session.active_negative_conditions)
        )

    if session_lines:
        sections.append("当前会话信息：\n" + "\n".join(session_lines))

    return "\n\n".join(sections)
