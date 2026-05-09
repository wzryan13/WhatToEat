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
            "memory_for_intent": "暂无记忆信息。",
            "memory_for_rerank": "暂无用户偏好。",
            "memory_for_intent_data": {},
            "memory_for_rerank_data": {},
        }

    store = get_memory_store()
    context = await store.load_memory_context(user_id, session_id)
    logger.info("[memory_read] 已载入用户画像和会话记忆")
    return context
