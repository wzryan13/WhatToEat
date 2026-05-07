import asyncio
import logging
from models.state import DietState
from tools import _tools

logger = logging.getLogger(__name__)


async def landmark_resolver(state: DietState) -> dict:
    try:
        result = await _tools["geo"].ainvoke({
            "address": state["location_text"],
            "city": state["city"] or ""
        })
        location = result['return'][0]["location"]
        logger.info(f"[landmark_resolver] 解析经纬度成功: {location}")
        return {
            "landmark_location": location,
            "landmark_resolve_failed": False,
        }
    except Exception as e:
        logger.warning(f"[landmark_resolver] 解析失败，降级关键词搜: {e}")
        return {
            "landmark_location": None,
            "landmark_resolve_failed": True,
        }


async def keyword_search(state: DietState) -> dict:
    tasks = [
        _tools["text_search"].ainvoke({
            "keywords": kw,
            "city": state["city"] or ""
        })
        for kw in state["keywords"]    #并发数量由keyword里的数量决定
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_ids = set()
    merged = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"[keyword_search] 单次搜索失败: {result}")
            continue
        # 解析返回结构
        pois = _parse_pois(result)
        for poi in pois:
            if poi["id"] not in seen_ids:
                seen_ids.add(poi["id"])
                merged.append(poi)

    logger.info(f"[keyword_search] 合并去重后共 {len(merged)} 条")
    return {"raw_pois": merged}


async def around_search(state: DietState) -> dict:
    tasks = [
        _tools["around_search"].ainvoke({
            "keywords": kw,
            "location": state["landmark_location"],
            "radius": str(state["filters"].get("radius") or 1000)
        })
        for kw in state["keywords"]
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    seen_ids = set()
    merged = []
    for result in results:
        if isinstance(result, Exception):
            logger.warning(f"[around_search] 单次搜索失败: {result}")
            continue
        pois = _parse_pois(result)
        for poi in pois:
            if poi["id"] not in seen_ids:
                seen_ids.add(poi["id"])
                merged.append(poi)

    logger.info(f"[around_search] 合并去重后共 {len(merged)} 条")
    return {"raw_pois": merged}


def _parse_pois(result) -> list:
    """解析工具返回结果，兼容不同返回格式"""
    import json
    try:
        if isinstance(result, list) and result:
            text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
            parsed = json.loads(text)
            return parsed.get("pois", [])
        elif isinstance(result, dict):
            return result.get("pois", [])
        elif isinstance(result, str):
            parsed = json.loads(result)
            return parsed.get("pois", [])
    except Exception as e:
        logger.warning(f"[_parse_pois] 解析失败: {e}")
    return []