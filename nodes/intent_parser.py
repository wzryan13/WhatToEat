import logging
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

from models.intent import IntentParserOutput, FilterConditions
from models.state import DietState
from config.prompts import INTENT_PARSER_SYSTEM_PROMPT
from config.settings import settings

logger = logging.getLogger(__name__)
llm = ChatDeepSeek(model=settings.MODEL_NAME)


async def intent_parser(state: DietState) -> dict:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M %A")
    conversation_history = state.get("conversation_history", [])[-settings.MAX_HISTORY_MESSAGES :]

    messages = [
        SystemMessage(content=INTENT_PARSER_SYSTEM_PROMPT),
        HumanMessage(content=f"""
当前时间：{current_time}
记忆上下文：{state.get('memory_for_intent', '暂无记忆信息。')}
对话历史：{conversation_history}
最新输入：{state['user_input']}
""")
    ]

    structured_llm = llm.with_structured_output(IntentParserOutput)

    last_error = None
    for attempt in range(settings.INTENT_PARSER_MAX_RETRIES):
        try:
            result: IntentParserOutput = await structured_llm.ainvoke(messages)
            break
        except Exception as e:
            last_error = e
            logger.warning(f"[intent_parser] LLM调用失败(第{attempt + 1}次): {e}")
    else:
        logger.error(f"[intent_parser] LLM {settings.INTENT_PARSER_MAX_RETRIES}次重试均失败")
        return {
            "location_type": state.get("location_type", "none"),
            "location_text": state.get("location_text"),
            "city": state.get("city"),
            "has_contradiction": False,
            "keywords": state.get("keywords") or ["美食"],
            "search_mode": state.get("search_mode", "keyword"),
            "filters": state.get("filters") or FilterConditions(),
            "negative_conditions": state.get("negative_conditions", []),
            "scene_context": state.get("scene_context", ""),
            "mood_factors": state.get("mood_factors", []),
            "suggested_cuisines": state.get("suggested_cuisines", []),
            "current_time": current_time,
            "error_message": f"意图解析失败：{str(last_error)}",
        }

    logger.info(f"[intent_parser] 解析结果: {result}")

    return {
        "intent_type": result.intent_type,
        "location_text": result.location_text,
        "location_type": result.location_type,
        "city": result.city,
        "keywords": result.keywords,
        "search_mode": result.search_mode,
        "filters": result.filters,
        "negative_conditions": result.negative_conditions,
        "has_contradiction": result.has_contradiction,
        "contradiction_message": result.contradiction_message,
        "scene_context": result.scene_context,
        "mood_factors": result.mood_factors,
        "suggested_cuisines": result.suggested_cuisines,
        "current_time": current_time,
        "conversation_history": [
            {"role": "user", "content": state["user_input"]}
        ],
    }
