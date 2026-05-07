from typing import TypedDict, Optional, Literal,Annotated
from models.intent import FilterConditions
import operator


class DietState(TypedDict, total=False):

    # ── 原始输入 ──
    user_input: str
    conversation_history: Annotated[list[dict], operator.add]

    # ── 系统注入 ──
    current_time: str

    # ── 意图解析结果 ──
    intent_type: Literal["normal", "brand", "scene", "time_based"]
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

    # ── 信息补全预留 ──
    # memory_location: Optional[str]
    # memory_user_profile: Optional[dict]
    # user_gps: Optional[str]

    # ── 工具调用中间结果 ──
    landmark_location: Optional[str]
    raw_pois: list[dict]
    detailed_pois: list[dict]
    filtered_pois: list[dict]

    # ── 输出 ──
    final_recommendations: list[dict]
    response_message: str
    disclaimer_needed: bool
    disclaimer_message: Optional[str]
    error_message: Optional[str]