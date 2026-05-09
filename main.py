import asyncio
import logging
from memory import init_memory_store
from memory.store import get_memory_store
from tools import init_tools
from graph import build_graph
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


async def run_conversation():
    await init_tools()
    await init_memory_store()
    app = build_graph()
    memory_store = get_memory_store()
    user_id = await memory_store.get_or_create_user(
        settings.DEMO_CHANNEL,
        settings.DEMO_EXTERNAL_ID,
    )
    runtime = await memory_store.get_or_create_session(user_id)
    config = {"configurable": {"thread_id": runtime.thread_id}}

    print("饮食管家已启动，输入 q 退出\n")

    while True:
        user_input = input("你：").strip()
        if user_input.lower() == "q":
            break
        if not user_input:
            continue

        # 检查当前thread是否处于interrupt状态（等待用户补充位置）
        current_state = app.get_state(config)
        has_checkpoint = bool(current_state and getattr(current_state, "values", None))
        is_interrupted = (
            current_state is not None
            and len(current_state.next) > 0
            and "clarify" in current_state.next
        )
        turn_no = await memory_store.next_turn(runtime.session_id)

        turn_state = {
            "user_id": runtime.user_id,
            "session_id": runtime.session_id,
            "thread_id": runtime.thread_id,
            "turn_no": turn_no,
            "user_input": user_input,
        }

        initial_state = {
            **turn_state,
            "conversation_history": [],
            "clarification_count": 0,
            "clarification_message": None,
            "landmark_resolve_failed": False,
            "result_insufficient": False,
            "landmark_location": None,
            "raw_pois": [],
            "detailed_pois": [],
            "filtered_pois": [],
            "final_recommendations": [],
            "disclaimer_needed": False,
            "disclaimer_message": None,
            "error_message": None,
            "memory_for_intent": "",
            "memory_for_rerank": "",
            "memory_for_intent_data": {},
            "memory_for_rerank_data": {},
            "scene_context": "",
            "mood_factors": [],
            "suggested_cuisines": [],
        }

        ongoing_turn_state = {
            **turn_state,
            "clarification_count": 0,
            "clarification_message": None,
            "landmark_resolve_failed": False,
            "result_insufficient": False,
            "landmark_location": None,
            "raw_pois": [],
            "detailed_pois": [],
            "filtered_pois": [],
            "final_recommendations": [],
            "disclaimer_needed": False,
            "disclaimer_message": None,
            "error_message": None,
            "scene_context": "",
            "mood_factors": [],
            "suggested_cuisines": [],
        }

        if is_interrupted:
            # 处于interrupt状态，resume并传入用户回复
            result = await app.ainvoke(
                turn_state,
                config=config,
            )
        elif has_checkpoint:
            result = await app.ainvoke(ongoing_turn_state, config=config)
        else:
            # 新一轮对话，重新开始
            result = await app.ainvoke(initial_state, config=config)

        response = result.get("response_message", "")
        print(f"\n管家：{response}\n")


if __name__ == "__main__":
    asyncio.run(run_conversation())
