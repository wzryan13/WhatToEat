import logging

from memory.store import get_memory_store
from models.state import DietState

logger = logging.getLogger(__name__)


async def memory_read(state: DietState) -> dict:
    user_id = state.get("user_id")
    session_id = state.get("session_id")
    if not user_id or not session_id:
        logger.info("[memory_read] 缺少 user_id/session_id，跳过记忆读取")
        return {
            "memory_profile": {},
            "memory_session": {},
            "memory_context_summary": "暂无可用记忆上下文。",
            "profile_summary_for_rerank": "暂无长期用户画像。",
        }

    store = get_memory_store()
    context = await store.load_memory_context(user_id, session_id)
    logger.info("[memory_read] 已载入用户画像和会话记忆")
    return {
        "memory_profile": context["profile"],
        "memory_session": context["session"],
        "memory_context_summary": context["memory_context_summary"],
        "profile_summary_for_rerank": context["profile_summary_for_rerank"],
    }
