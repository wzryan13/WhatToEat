from typing import Annotated, Optional, TypedDict, Literal
from models.intent import FilterConditions
import operator


class DietState(TypedDict, total=False):

    # ── 身份与执行链 ──
    user_id: str
    session_id: str
    thread_id: str
    turn_no: int

    # ── 原始输入 ──
    user_input: str
    conversation_history: Annotated[list[dict], operator.add]
    conversation_summary: str

    # ── 系统注入 ──
    current_time: str

    # ── 意图解析结果 ──
    intent_type: Literal["normal", "brand", "scene", "time_based", "recipe", "recommend"]
    location_text: Optional[str]
    location_type: Literal["valid", "relative", "gps", "none", "invalid"]
    city: Optional[str]
    keywords: list[str]
    search_mode: Literal["keyword", "around"]
    filters: FilterConditions
    negative_conditions: list[str]
    has_contradiction: bool
    contradiction_message: Optional[str]

    # ── 流程控制 ──
    clarification_count: int
    clarification_message: Optional[str]
    landmark_resolve_failed: bool
    result_insufficient: bool

    # ── 工具调用中间结果 ──
    landmark_location: Optional[str]
    raw_pois: list[dict]
    detailed_pois: list[dict]
    filtered_pois: list[dict]

    # ── 记忆层 ──
    memory_for_intent: str
    memory_for_rerank: str
    memory_for_intent_data: dict
    memory_for_rerank_data: dict

    # ── 场景推理 ──
    scene_context: str
    mood_factors: list[str]
    suggested_cuisines: list[str]
    has_ingredient: bool

    # ── RAG 菜谱检索结果 ──
    rag_query: Optional[str]              # rewrite 后的查询
    rag_documents: list[dict]             # 检索到的菜谱文档列表
    rag_filter_expr: Optional[str]        # 生成的 Milvus 过滤表达式

    # ── 输出 ──
    final_recommendations: list[dict]
    response_message: str
    disclaimer_needed: bool
    disclaimer_message: Optional[str]
    hook_message: Optional[str]
    error_message: Optional[str]
    search_context: Optional[str]
