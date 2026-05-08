import asyncio
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from models.state import DietState
from models.rerank import LLMRerankOutput
from config.prompts import RERANK_SYSTEM_PROMPT
from config.settings import settings
from tools import _tools

logger = logging.getLogger(__name__)
llm = ChatAnthropic(model=settings.MODEL_NAME)


async def batch_poi_detail(state: DietState) -> dict:
    pois = state.get("raw_pois", [])[:settings.POI_DETAIL_LIMIT]

    tasks = [
        _tools["search_detail"].ainvoke({"id": poi["id"]})
        for poi in pois
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    detailed = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"[batch_poi_detail] 单次POI详情失败: {result}")
            continue
        parsed = _parse_detail(result)
        if parsed:
            detailed.append(parsed)

    logger.info(f"[batch_poi_detail] 成功获取 {len(detailed)} 条详情")
    return {"detailed_pois": detailed}


async def precise_filter(state: DietState) -> dict:
    filters = state["filters"]
    passed = []

    for poi in state.get("detailed_pois", []):
        # 人均过滤
        cost = poi.get("cost")
        if cost:
            try:
                cost_val = float(cost)
                if filters.get("price_max") and cost_val > filters["price_max"]:
                    continue
                if filters.get("price_min") and cost_val < filters["price_min"]:
                    continue
            except (ValueError, TypeError):
                pass

        # 评分过滤
        rating = poi.get("rating")
        if filters.get("min_rating") and rating:
            try:
                if float(rating) < filters["min_rating"]:
                    continue
            except (ValueError, TypeError):
                pass

        passed.append(poi)

    insufficient = len(passed) < 2
    logger.info(f"[precise_filter] 过滤后剩余 {len(passed)} 条，不足={insufficient}")
    return {
        "filtered_pois": passed,
        "result_insufficient": insufficient,
    }


async def llm_rerank(state: DietState) -> dict:
    source = state.get("filtered_pois") or state.get("detailed_pois", [])

    if not source:
        logger.warning("[llm_rerank] 候选列表为空，跳过 LLM 推理，直接返回空推荐")
        return {
            "final_recommendations": [],
            "disclaimer_needed": True,
            "disclaimer_message": "抱歉，没有找到符合您要求的餐厅，要不要换个条件试试？",
        }

    poi_info = [
        {
            "id": poi.get("id"),
            "name": poi.get("name"),
            "type": poi.get("type"),
            "address": poi.get("address"),
            "cost": poi.get("cost"),
            "rating": poi.get("rating"),
            "open_time": poi.get("open_time"),
            "opentime2": poi.get("opentime2"),
            "business_area": poi.get("business_area"),
        }
        for poi in source
    ]

    import json
    system_prompt = RERANK_SYSTEM_PROMPT.format(
        current_time=state.get("current_time", ""),
        negative_conditions=state.get("negative_conditions", []),
        user_input=state["user_input"],
        profile_summary=state.get("profile_summary_for_rerank", "暂无长期用户画像。"),
        max_recommendations=settings.MAX_RECOMMENDATIONS,
    )

    structured_llm = llm.with_structured_output(LLMRerankOutput)

    try:
        result: LLMRerankOutput = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"餐厅列表：{json.dumps(poi_info, ensure_ascii=False)}")
        ])
    except Exception as e:
        logger.error(f"[llm_rerank] LLM调用失败: {e}")
        return {
            "final_recommendations": [],
            "disclaimer_needed": False,
            "disclaimer_message": None,
        }

    logger.info(f"[llm_rerank] 推荐 {len(result.recommendations)} 家")
    return {
        "final_recommendations": [r.model_dump() for r in result.recommendations],
        "disclaimer_needed": bool(result.disclaimer),
        "disclaimer_message": result.disclaimer,
    }


def _parse_detail(result) -> dict | None:
    """解析POI详情返回结果"""
    import json
    try:
        if isinstance(result, list) and result:
            text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
            return json.loads(text)
        elif isinstance(result, dict):
            return result
        elif isinstance(result, str):
            return json.loads(result)
    except Exception as e:
        logger.warning(f"[_parse_detail] 解析失败: {e}")
    return None
