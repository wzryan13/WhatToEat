import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from models.state import DietState
from nodes import (
    memory_read,
    intent_parser,
    search_agent,
    rag_agent,
    rag_formatter,
    clarify,
    error_output,
    result_formatter,
    memory_write,
)

logger = logging.getLogger(__name__)


def route_after_intent(state: DietState) -> str:
    """
    意图路由：根据 intent_type 和 location_type 决定��一步。
    - recipe → rag_agent（菜谱检索，不需要位置）
    - 其他 �� 按位置有效性走 search_agent / clarify / error_output
    """
    intent_type = state.get("intent_type", "normal")

    if intent_type in ("recipe", "recommend"):
        return "rag_agent"

    # 餐厅意图：检查位置有效性
    location_type = state.get("location_type", "none")

    if location_type == "invalid":
        return "error_output"
    if location_type in ("none", "relative", "gps"):
        from config.settings import settings
        if state.get("clarification_count", 0) >= settings.MAX_CLARIFICATION:
            return "error_output"
        return "clarify"

    return "search_agent"


def build_graph() -> StateGraph:
    checkpointer = MemorySaver()
    graph = StateGraph(DietState)

    # ── 注册节点 ──
    graph.add_node("memory_read", memory_read)
    graph.add_node("intent_parser", intent_parser)
    graph.add_node("clarify", clarify)
    graph.add_node("error_output", error_output)
    graph.add_node("search_agent", search_agent)
    graph.add_node("result_formatter", result_formatter)
    graph.add_node("rag_agent", rag_agent)
    graph.add_node("rag_formatter", rag_formatter)
    graph.add_node("memory_write", memory_write)

    # ── 入口 ──
    graph.set_entry_point("memory_read")

    graph.add_edge("memory_read", "intent_parser")

    graph.add_conditional_edges(
        "intent_parser",
        route_after_intent,
        {
            "error_output": "error_output",
            "clarify": "clarify",
            "search_agent": "search_agent",
            "rag_agent": "rag_agent",
        }
    )

    graph.add_edge("clarify", "intent_parser")

    # ── search_agent 路径 → 格式化 → 记忆 → 结束 ──
    graph.add_edge("search_agent", "result_formatter")
    graph.add_edge("result_formatter", "memory_write")

    # ── rag_agent 路径 → 个性化格式化 → 记忆 → 结束 ──
    graph.add_edge("rag_agent", "rag_formatter")
    graph.add_edge("rag_formatter", "memory_write")

    graph.add_edge("error_output", "memory_write")
    graph.add_edge("memory_write", END)

    logger.info("[build_graph] Graph构建完成（含 RAG 分支）")
    return graph.compile(checkpointer=checkpointer)
