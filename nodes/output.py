import logging
from models.state import DietState
from langgraph.types import interrupt

logger = logging.getLogger(__name__)


async def clarify(state: DietState) -> dict:
    keywords = state.get("keywords", [])
    keywords_str = "、".join(state.get("keywords", [])) if state.get("keywords") else "美食"

    if keywords:
        message = f"您想吃{keywords_str}，请问您在哪个城市或具体位置呢？"
    else:
        message = "请问您目前在哪个城市或具体位置呢？这样我才能为您推荐附近的美食哦。"

    logger.info(f"[clarify] 追问用户，当前追问次数: {state.get('clarification_count', 0) + 1}")

    user_reply = interrupt(message)

    return {
        "user_input": user_reply,
        "clarification_message": message,
        "clarification_count": state.get("clarification_count", 0) + 1,
        "response_message": message,
        "conversation_history": [
            {"role": "assistant", "content": message},
            {"role": "user", "content": user_reply},
        ],
    }


async def error_output(state: DietState) -> dict:
    location_text = state.get("location_text", "")
    location_type = state.get("location_type", "none")

    if location_type == "invalid":
        msg = f"抱歉，无法识别‘{location_text}’这个位置，请提供有效的城市或地区。"
    else:
        msg = "抱歉，未能获取到您的位置信息，无法为您推荐餐厅，请告知您所在的城市或区域。"

    logger.info(f"[error_output] 错误类型: {location_type}")
    return {
        "response_message": msg,
        "error_message": msg,
        "conversation_history": [{"role": "assistant", "content": msg}],
    }


async def result_formatter(state: DietState) -> dict:
    recs = state.get("final_recommendations", [])

    if not recs:
        if state.get("result_insufficient"):
            msg = "抱歉，根据您的条件没有找到符合的餐厅，建议放宽筛选条件再试试。"
        else:
            msg = "抱歉，暂时没有找到合适的餐厅，请换个关键词或位置试试。"
        return {
            "response_message": msg,
            "conversation_history": [{"role": "assistant", "content": msg}],
        }

    poi_map = {
        poi["id"]: poi
        for poi in (state.get("filtered_pois") or state.get("detailed_pois", []))
    }

    lines = ["为您推荐以下餐厅：\n"]
    for i, rec in enumerate(recs, 1):
        poi = poi_map.get(rec.get("id"), {})
        name = rec.get("name") or poi.get("name", "")
        cost = poi.get("cost", "未知")
        rating = poi.get("rating", "暂无")
        address = poi.get("address", "")
        open_time = poi.get("open_time", "")
        reason = rec.get("reason", "")
        hook = rec.get("hook", "")
        is_open = rec.get("is_open")

        open_status = ""
        if is_open is False:
            open_status = " ⚠️ 当前可能未营业"
        elif is_open is True:
            open_status = " ✅ 当前营业中"

        lines.append(f"{i}. {name}{open_status}")
        lines.append(f"   📍 {address}")
        lines.append(f"   💰 人均：{cost}元  ⭐ 评分：{rating}")
        if open_time:
            lines.append(f"   🕐 营业时间：{open_time}")
        combined_reason = f"{reason} {hook}".strip()
        lines.append(f"   💬 {combined_reason}")
        lines.append("")

    if state.get("landmark_resolve_failed"):
        lines.append("（提示：未能精确定位您的位置，已按城市范围为您搜索）")
    if state.get("has_contradiction"):
        lines.append(f"（提示：{state.get('contradiction_message', '')}）")
    if state.get("disclaimer_needed"):
        lines.append(f"（{state.get('disclaimer_message', '')}）")
    if state.get("result_insufficient"):
        lines.append("（提示：符合全部条件的餐厅较少，以上为最接近的推荐）")

    response = "\n".join(lines)
    logger.info(f"[result_formatter] 输出 {len(recs)} 条推荐")
    return {
        "response_message": response,
        "conversation_history": [{"role": "assistant", "content": response}],
    }
