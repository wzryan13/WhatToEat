from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone

from config.settings import settings
from memory.store import get_memory_store
from memory.user_profile import dedupe_strings, now_iso
from models.memory import BudgetRange, MemoryFact, SessionMemory, UserProfile
from models.state import DietState

logger = logging.getLogger(__name__)


def _filters_to_budget_range(filters) -> BudgetRange | None:
    if not filters:
        return None

    getter = getattr(filters, "get", None)
    if callable(getter):
        minimum = getter("price_min")
        maximum = getter("price_max")
    else:
        minimum = getattr(filters, "price_min", None)
        maximum = getattr(filters, "price_max", None)

    if minimum is None and maximum is None:
        return None
    return BudgetRange(min=minimum, max=maximum)


def _extract_allergies(user_input: str) -> list[str]:
    matches = re.findall(r"([\u4e00-\u9fa5A-Za-z]{1,12}过敏)", user_input)
    return dedupe_strings(matches)


def _extract_dietary_restrictions(user_input: str) -> list[str]:
    restrictions: list[str] = []
    restrictions.extend(re.findall(r"(不吃[\u4e00-\u9fa5A-Za-z]{1,12})", user_input))
    restrictions.extend(re.findall(r"(忌口[\u4e00-\u9fa5A-Za-z]{1,12})", user_input))

    for keyword in ("吃素", "素食", "纯素", "清真"):
        if keyword in user_input:
            restrictions.append(keyword)

    return dedupe_strings(restrictions)


def _should_write_default_city(user_input: str, city: str | None) -> bool:
    if not city:
        return False

    stable_signals = (
        "常在",
        "平时在",
        "住在",
        "常驻",
        "工作在",
        "在" + city + "工作",
    )
    return any(signal in user_input for signal in stable_signals)


def _build_profile_candidate(state: DietState) -> UserProfile | None:
    user_input = state.get("user_input", "")
    timestamp = now_iso()

    allergies = [
        MemoryFact(
            value=value,
            confidence=0.99,
            updated_at=timestamp,
            last_seen_at=timestamp,
        )
        for value in _extract_allergies(user_input)
    ]
    restrictions = [
        MemoryFact(
            value=value,
            confidence=0.95,
            updated_at=timestamp,
            last_seen_at=timestamp,
        )
        for value in _extract_dietary_restrictions(user_input)
    ]

    default_city = None
    if _should_write_default_city(user_input, state.get("city")):
        default_city = MemoryFact(
            value=state.get("city", ""),
            confidence=0.8,
            updated_at=timestamp,
            last_seen_at=timestamp,
        )

    if not allergies and not restrictions and default_city is None:
        return None

    return UserProfile(
        default_city=default_city,
        dietary_restrictions=restrictions,
        allergies=allergies,
    )


def _build_session_memory(state: DietState) -> SessionMemory:
    timestamp = datetime.now(timezone.utc)
    expires_at = timestamp + timedelta(hours=settings.SESSION_TTL_HOURS)
    recommendations = state.get("final_recommendations", [])

    return SessionMemory(
        active_city=state.get("city"),
        active_location_text=state.get("location_text"),
        active_budget_range=_filters_to_budget_range(state.get("filters")),
        active_negative_conditions=dedupe_strings(state.get("negative_conditions", [])),
        last_clarification_question=state.get("clarification_message"),
        last_result_summary={
            "recommended_names": [item.get("name") for item in recommendations if item.get("name")],
            "recommended_poi_ids": [item.get("id") for item in recommendations if item.get("id")],
        },
        updated_at=timestamp.isoformat(),
        expires_at=expires_at.isoformat(),
    )


async def memory_write(state: DietState) -> dict:
    user_id = state.get("user_id")
    session_id = state.get("session_id")
    if not user_id or not session_id:
        logger.info("[memory_write] 缺少 user_id/session_id，跳过记忆写入")
        return {}

    store = get_memory_store()
    session_memory = _build_session_memory(state)
    await store.save_session_memory(user_id, session_id, session_memory)

    candidate = _build_profile_candidate(state)
    if candidate is None:
        logger.info("[memory_write] 无高置信用户画像候选")
        return {
            "memory_session": session_memory.model_dump(exclude_none=True),
            "memory_write_candidates": [],
        }

    if candidate.allergies or candidate.dietary_restrictions:
        await store.apply_profile_update(user_id, candidate)
    elif candidate.default_city:
        store.schedule_profile_update(user_id, candidate)

    logger.info("[memory_write] 已更新 SessionMemory，并提交 UserProfile 写入")
    return {
        "memory_session": session_memory.model_dump(exclude_none=True),
        "memory_write_candidates": candidate.model_dump(exclude_none=True),
    }
