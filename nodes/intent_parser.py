import logging
from datetime import datetime
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from models.intent import IntentParserOutput, FilterConditions
from models.state import DietState
from config.prompts import INTENT_PARSER_SYSTEM_PROMPT
from config.settings import settings

logger = logging.getLogger(__name__)
llm = ChatAnthropic(model=settings.MODEL_NAME)


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

    try:
        result: IntentParserOutput = await structured_llm.ainvoke(messages)
    except Exception as e:
        logger.error(f"[intent_parser] LLM调用失败: {e}")
        return {
            "location_type": "none",
            "has_contradiction": False,
            "keywords": ["美食"],
            "search_mode": "keyword",
            "filters": FilterConditions(),
            "negative_conditions": [],
            "scene_context": "",
            "mood_factors": [],
            "suggested_cuisines": [],
            "current_time": current_time,
            "error_message": f"意图解析失败：{str(e)}",
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
