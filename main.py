import asyncio
import logging
import uuid
from tools import init_tools
from graph import build_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


async def run_conversation():
    await init_tools()
    app = build_graph()

    # 每个对话session使用固定thread_id，checkpointer据此恢复state
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("饮食管家已启动，输入 q 退出\n")

    while True:
        user_input = input("你：").strip()
        if user_input.lower() == "q":
            break
        if not user_input:
            continue

        initial_state = {
            "user_input": user_input,
            "conversation_history": [],
            "clarification_count": 0,
            "landmark_resolve_failed": False,
            "result_insufficient": False,
            "raw_pois": [],
            "detailed_pois": [],
            "filtered_pois": [],
            "final_recommendations": [],
            "disclaimer_needed": False,
        }

        # 检查当前thread是否处于interrupt状态（等待用户补充位置）
        current_state = app.get_state(config)
        is_interrupted = (
            current_state is not None
            and len(current_state.next) > 0
            and "clarify" in current_state.next
        )

        if is_interrupted:
            # 处于interrupt状态，resume并传入用户回复
            result = await app.ainvoke(
                {"user_input": user_input},
                config=config,
            )
        else:
            # 新一轮对话，重新开始
            result = await app.ainvoke(initial_state, config=config)

        response = result.get("response_message", "")
        print(f"\n管家：{response}\n")


if __name__ == "__main__":
    asyncio.run(run_conversation())