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

        initial_state = {
            "user_id": runtime.user_id,
            "session_id": runtime.session_id,
            "thread_id": runtime.thread_id,
            "turn_no": turn_no,
            "user_input": user_input,
            "conversation_history": [],
        }

        if is_interrupted:
            result = await app.ainvoke(
                {"user_input": user_input},
                config=config,
            )
        elif has_checkpoint:
            result = await app.ainvoke(
                {"user_input": user_input, "turn_no": turn_no},
                config=config,
            )
        else:
            result = await app.ainvoke(initial_state, config=config)

        response = result.get("response_message", "")
        print(f"\n管家：{response}\n")


if __name__ == "__main__":
    asyncio.run(run_conversation())
