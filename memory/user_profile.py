from __future__ import annotations

import copy
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

        existing.last_seen_at = fact.last_seen_at or existing.last_seen_at
        existing.updated_at = fact.updated_at or existing.updated_at
    return list(merged.values())


def _migrate_profile_data(data: dict) -> dict:
    migrated = copy.deepcopy(data)
    version = migrated.get("schema_version", 1)
    if version < 1:
        migrated["schema_version"] = 1

    defaults: dict[str, object] = {
        "allergies": [],
        "food_blacklist": [],
        "religious_restrictions": [],
        "disliked_cuisines": [],
        "cuisine_tags": {},
        "health_goals": [],
        "active_areas": [],
    }
    for key, default_value in defaults.items():
        migrated.setdefault(key, default_value)
    return migrated


def profile_from_dict(data: dict | None) -> UserProfile:
    if not data:
        return UserProfile()
    migrated = _migrate_profile_data(data)
    return UserProfile.model_validate(migrated)


def session_from_dict(data: dict | None) -> SessionMemory:
    if not data:
        return SessionMemory()
    return SessionMemory.model_validate(data)


def build_memory_for_intent(profile: UserProfile, session: SessionMemory) -> str:
    lines: list[str] = []
    if profile.default_city:
        lines.append(f"用户默认城市: {profile.default_city.value}")
    if profile.allergies:
        lines.append("过敏: " + "、".join(f.value for f in profile.allergies))
    if profile.food_blacklist:
        lines.append("不吃的食材: " + "、".join(f.value for f in profile.food_blacklist))
    if profile.religious_restrictions:
        lines.append("饮食限制: " + "、".join(f.value for f in profile.religious_restrictions))
    if profile.disliked_cuisines:
        lines.append("不吃的菜系: " + "、".join(f.value for f in profile.disliked_cuisines))
    if session.active_city:
        lines.append(f"当前城市: {session.active_city}")
    if session.active_location_text:
        lines.append(f"当前位置: {session.active_location_text}")
    if session.active_budget_range:
        if session.active_budget_range.max is not None:
            lines.append(f"人均参考: {session.active_budget_range.max:g}元以内")
        elif session.active_budget_range.min is not None:
            lines.append(f"人均参考: {session.active_budget_range.min:g}元以上")
    return "\n".join(lines) if lines else "暂无记忆信息。"


def build_memory_for_rerank(profile: UserProfile, session: SessionMemory) -> str:
    lines: list[str] = []
    if profile.allergies:
        lines.append("过敏: " + "、".join(f.value for f in profile.allergies))
    if profile.food_blacklist:
        lines.append("不吃: " + "、".join(f.value for f in profile.food_blacklist))
    if profile.disliked_cuisines:
        lines.append("不吃的菜系: " + "、".join(f.value for f in profile.disliked_cuisines))
    if profile.cuisine_tags:
        liked = [k for k, v in profile.cuisine_tags.items() if v == "liked"]
        loved = [k for k, v in profile.cuisine_tags.items() if v == "loved"]
        if loved:
            lines.append("最爱菜系: " + "、".join(loved))
        if liked:
            lines.append("喜欢菜系: " + "、".join(liked))
    if profile.spice_tolerance:
        lines.append(f"辣度接受: {profile.spice_tolerance.value}")
    if profile.sweetness:
        lines.append(f"甜度偏好: {profile.sweetness.value}")
    if profile.health_goals:
        lines.append("健康目标: " + "、".join(f.value for f in profile.health_goals))
    if profile.budget_solo:
        lines.append(f"一人食预算: {profile.budget_solo.value}")
    if profile.budget_group:
        lines.append(f"聚餐预算: {profile.budget_group.value}")
    if session.active_negative_conditions:
        disliked_values = {f.value for f in profile.disliked_cuisines}
        session_negatives = [
            c for c in session.active_negative_conditions
            if c not in disliked_values
        ]
        if session_negatives:
            lines.append("本轮排除: " + "、".join(session_negatives))
    return "\n".join(lines) if lines else "暂无用户偏好。"
