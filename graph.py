import logging
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from models.state import DietState
from nodes import (
    memory_read,
    intent_parser,
    landmark_resolver,
    keyword_search,
    around_search,
    batch_poi_detail,
    precise_filter,
    llm_rerank,
    clarify,
    error_output,
    result_formatter,
    memory_write,
)

logger = logging.getLogger(__name__)


def route_after_intent(state: DietState) -> str:
    location_type = state.get("location_type", "none")

    # 先判断location是否有效
    if location_type == "invalid":
        return "error_output"
    if location_type in ("none", "relative", "gps"):
        from config.settings import settings
        if state.get("clarification_count", 0) >= settings.MAX_CLARIFICATION:
            return "error_output"
        return "clarify"

    # location有效，再判断搜索模式
    search_mode = state.get("search_mode", "keyword")
    if search_mode == "around":
        return "landmark_resolver"
    return "keyword_search"

def route_after_landmark(state: DietState) -> str:
    if state.get("landmark_resolve_failed"):
        return "keyword_search"
    return "around_search"


def build_graph() -> StateGraph:
    checkpointer = MemorySaver()
    graph = StateGraph(DietState)

    # ── 注册节点 ──
    graph.add_node("memory_read", memory_read)
    graph.add_node("intent_parser", intent_parser)
    graph.add_node("clarify", clarify)
    graph.add_node("error_output", error_output)
    graph.add_node("landmark_resolver", landmark_resolver)
    graph.add_node("keyword_search", keyword_search)
    graph.add_node("around_search", around_search)
    graph.add_node("batch_poi_detail", batch_poi_detail)
    graph.add_node("precise_filter", precise_filter)
    graph.add_node("llm_rerank", llm_rerank)
    graph.add_node("result_formatter", result_formatter)
    graph.add_node("memory_write", memory_write)

    # ── 入口 ──
    graph.set_entry_point("memory_read")

    graph.add_edge("memory_read", "intent_parser")

    # ── intent_parser → info_complement（长记忆查询补充location）→ route_check ──
    # 目前info_complement暂不实现，直接从intent_parser → route_check
    graph.add_conditional_edges(
        "intent_parser",
        route_after_intent,
        {
            "error_output": "error_output",
            "clarify": "clarify",
            "landmark_resolver": "landmark_resolver",
            "keyword_search": "keyword_search",
        }
    )

    # clarify → 回到intent_parser（用户补充后重新解析）
    graph.add_edge("clarify", "intent_parser")

    # ── landmark_resolver → around_search 或降级 keyword_search ──
    graph.add_conditional_edges(
        "landmark_resolver",
        route_after_landmark,
        {
            "around_search":  "around_search",
            "keyword_search": "keyword_search",
        }
    )

    # ── 搜索结果汇聚 → batch_poi_detail ──
    graph.add_edge("keyword_search", "batch_poi_detail")
    graph.add_edge("around_search", "batch_poi_detail")

    # ── POI详情 → 精筛 → rerank → 格式化 ──
    graph.add_edge("batch_poi_detail", "precise_filter")
    graph.add_edge("precise_filter", "llm_rerank")
    graph.add_edge("llm_rerank", "result_formatter")
    graph.add_edge("result_formatter", "memory_write")
    graph.add_edge("error_output", "memory_write")

    # ── 终止 ──
    graph.add_edge("memory_write", END)

    logger.info("[build_graph] Graph构建完成")
    return graph.compile(checkpointer=checkpointer)
