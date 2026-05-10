from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from config.prompts import PROFILE_UPDATE_PROMPT
from config.settings import settings
from memory.store import get_memory_store
from memory.user_profile import dedupe_strings
from models.memory import ProfileUpdateDecision, SessionMemory
from models.state import DietState

logger = logging.getLogger(__name__)
llm = ChatAnthropic(model=settings.MODEL_NAME)


def _filters_to_budget_range(filters):
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
    return {"min": minimum, "max": maximum}


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
            "recommended_names": [
                item.get("name") for item in recommendations if item.get("name")
            ],
            "recommended_poi_ids": [
                item.get("id") for item in recommendations if item.get("id")
            ],
        },
        updated_at=timestamp.isoformat(),
        expires_at=expires_at.isoformat(),
    )


async def _async_profile_update(
    user_id: str,
    old_profile: dict,
    conversation_history: list[dict],
    recommendations: list[dict],
) -> None:
    if not conversation_history:
        return

    store = get_memory_store()
    prompt = PROFILE_UPDATE_PROMPT.format(
        old_profile=json.dumps(old_profile, ensure_ascii=False),
        conversation_history=json.dumps(conversation_history, ensure_ascii=False),
        recommendations=json.dumps(recommendations, ensure_ascii=False),
    )
    structured_llm = llm.with_structured_output(ProfileUpdateDecision)

    try:
        result: ProfileUpdateDecision = await structured_llm.ainvoke(
            [
                SystemMessage(content="你负责分析长期用户画像更新。"),
                HumanMessage(content=prompt),
            ]
        )
    except Exception as exc:
        logger.exception("[memory_write] LLM③ 画像更新失败: %s", exc)
        return

    if not result.updates:
        logger.info("[memory_write] LLM③ 判定无需更新画像: %s", result.no_update_reason)
        return

    await store.apply_profile_updates(user_id, result.updates)
    logger.info("[memory_write] LLM③ 已写入 %s 条画像更新", len(result.updates))


async def memory_write(state: DietState) -> dict:
    user_id = state.get("user_id")
    session_id = state.get("session_id")
    if not user_id or not session_id:
        logger.info("[memory_write] 缺少 user_id/session_id，跳过记忆写入")
        return {}

    store = get_memory_store()
    session_memory = _build_session_memory(state)
    await store.save_session_memory(user_id, session_id, session_memory)

    old_profile_obj = await store.load_profile(user_id)
    old_profile = old_profile_obj.model_dump(exclude_none=True)
    history = state.get("conversation_history", [])[-settings.MAX_HISTORY_MESSAGES :]
    recommendations = state.get("final_recommendations", [])

    asyncio.create_task(
        _async_profile_update(
            user_id=user_id,
            old_profile=old_profile,
            conversation_history=history,
            recommendations=recommendations,
        )
    )

    logger.info("[memory_write] SessionMemory 已同步保存，长期画像更新已异步调度")
    return {}
