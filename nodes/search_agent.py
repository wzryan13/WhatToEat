import asyncio
import json
import logging
import re
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek

import redis.asyncio as aioredis

from config.prompts import RERANK_SYSTEM_PROMPT
from config.settings import settings
from models.intent import FilterConditions
from models.rerank import LLMRerankOutput
from models.state import DietState
from tools import _tools

logger = logging.getLogger(__name__)
llm = ChatDeepSeek(model=settings.MODEL_NAME)

# ── 超时与并发控制 ───────────────────────────────────────
# 单位：秒
TIMEOUT_GEO = 20.0       # 地标解析正常 ~2.5s，留 2 倍余量
TIMEOUT_SEARCH = 60.0   # around/text 搜索正常 ~9s，留 3s 余量切尾巴
TIMEOUT_DETAIL = 30    # 详情正常 ~3s，留 2.6 倍余量
DETAIL_CONCURRENCY = 8 # 高德个人认证 key QPS=5，按 P25 RT≈2s 推算上限

# ── Geo 缓存 ────────────────────────────────────────────────
GEO_CACHE_TTL = 604800  # 7 days

_geo_cache_client: aioredis.Redis | None = None
_geo_cache_initialized = False


def _get_geo_cache() -> aioredis.Redis | None:
    global _geo_cache_client, _geo_cache_initialized
    if not _geo_cache_initialized:
        _geo_cache_initialized = True
        try:
            _geo_cache_client = aioredis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD or None,
                db=0,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
        except Exception as e:
            logger.warning(f"[search_agent] geo缓存初始化失败: {e}")
    return _geo_cache_client


# ── 工具函数 ──────────────────────────────────────────────


def _parse_pois(result) -> list:
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
        logger.warning(f"[search_agent] POI解析失败: {e}")
    return []


def _parse_detail(result) -> dict | None:
    try:
        if isinstance(result, list) and result:
            text = result[0].get("text", "") if isinstance(result[0], dict) else str(result[0])
            return json.loads(text)
        elif isinstance(result, dict):
            return result
        elif isinstance(result, str):
            return json.loads(result)
    except Exception as e:
        logger.warning(f"[search_agent] 详情解析失败: {e}")
    return None


def _brand_dedup(pois: list) -> list:
    """连锁店品牌去重：去掉括号里的分店名，同品牌只保留第一条"""
    seen_brands = set()
    result = []
    for poi in pois:
        name = poi.get("name", "")
        brand = re.sub(r'[（(][^)）]*[)）]', '', name).strip()
        if brand not in seen_brands:
            seen_brands.add(brand)
            result.append(poi)
    return result


def _even_split(pois_by_keyword: dict[str, list], per_keyword: int) -> list:
    """按关键词均分取 POI，跨关键词 ID 去重"""
    result = []
    seen_ids = set()
    for kw, pois in pois_by_keyword.items():
        count = 0
        for poi in pois:
            if count >= per_keyword:
                break
            poi_id = poi.get("id")
            if poi_id and poi_id not in seen_ids:
                seen_ids.add(poi_id)
                result.append(poi)
                count += 1
    return result


def _precise_filter(pois: list, filters) -> list:
    """硬约束过滤：价格、评分"""
    price_max = filters.get("price_max")
    price_min = filters.get("price_min")
    min_rating = filters.get("min_rating")

    passed = []
    for poi in pois:
        cost = poi.get("cost")
        if cost:
            try:
                cost_val = float(cost)
                if price_max is not None and cost_val > price_max:
                    continue
                if price_min is not None and cost_val < price_min:
                    continue
            except (ValueError, TypeError):
                pass

        rating = poi.get("rating")
        if min_rating is not None and rating:
            try:
                if float(rating) < min_rating:
                    continue
            except (ValueError, TypeError):
                pass

        passed.append(poi)
    return passed


def _is_diverse(pois: list) -> bool:
    """检查品类多样性：某个中类占比超过 70% 则不够多样"""
    if len(pois) < 4:
        return True

    categories = []
    for poi in pois:
        poi_type = poi.get("type", "")
        parts = poi_type.split(";")
        if len(parts) >= 2:
            categories.append(parts[1].strip())

    if not categories:
        return True

    counter = Counter(categories)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(categories) < 0.7


# ── 搜索管道 ──────────────────────────────────────────────

async def _geo_resolve(address: str, city: str) -> tuple[str | None, str | None]:
    cache = _get_geo_cache()
    cache_key = f"geo:{city}:{address}"
    if cache:
        try:
            cached = await cache.get(cache_key)
            if cached:
                logger.info(f"[search_agent] geo缓存命中: {address} → {cached}")
                return cached, None
        except Exception as e:
            logger.warning(f"[search_agent] geo缓存读取失败: {e}")

    try:
        result = await asyncio.wait_for(
            _tools["geo"].ainvoke({
                "address": address,
                "city": city or "",
            }),
            timeout=TIMEOUT_GEO,
        )

        # 解包 ainvoke 返回: [{"type": "text", "text": "{...}"}]
        if isinstance(result, list) and result and isinstance(result[0], dict):
            text = result[0].get("text", "")
        elif isinstance(result, str):
            text = result
        else:
            logger.warning(f"[search_agent] 地标解析返回未知类型: {type(result).__name__}")
            return None, f"unexpected_result_type:{type(result).__name__}"

        parsed = json.loads(text)
        items = parsed.get("return", [])

        if not items:
            logger.warning(f"[search_agent] 地标解析无结果: {address}, raw={parsed}")
            return None, "empty_results"

        location = items[0].get("location")
        if location:
            if cache:
                try:
                    await cache.setex(cache_key, GEO_CACHE_TTL, location)
                except Exception as e:
                    logger.warning(f"[search_agent] geo缓存写入失败: {e}")
            logger.info(f"[search_agent] 地标解析成功: {address} → {location}")
            return location, None
        else:
            logger.warning(f"[search_agent] 地标解析无 location 字段: {items[0]}")
            return None, "missing_location"

    except asyncio.TimeoutError:
        logger.warning(f"[search_agent] 地标解析超时({TIMEOUT_GEO}s): {address}")
        return None, "tool_timeout"
    except asyncio.CancelledError as e:
        logger.warning(f"[search_agent] 地标解析调用被取消: {e}")
        return None, "tool_cancelled"
    except Exception as e:
        logger.warning(f"[search_agent] 地标解析失败: {e}")
        return None, f"tool_error:{type(e).__name__}"

async def _search_and_detail(
    keywords: list[str],
    city: str,
    filters,
    search_type: str,
    location: str | None,
    radius: int,
) -> list[dict]:
    """搜索 → 品牌去重 → 均分 → 查详情 → 硬约束过滤"""

    # 1. 并发搜索（每个 keyword 独立超时，超时就空着不拖整批）
    async def _search_one(kw: str):
        try:
            if search_type == "around" and location:
                return await asyncio.wait_for(
                    _tools["around_search"].ainvoke({
                        "keywords": kw,
                        "location": location,
                        "radius": str(radius),
                    }),
                    timeout=TIMEOUT_SEARCH,
                )
            else:
                return await asyncio.wait_for(
                    _tools["text_search"].ainvoke({
                        "keywords": kw,
                        "city": city or "",
                    }),
                    timeout=TIMEOUT_SEARCH,
                )
        except asyncio.TimeoutError:
            logger.warning(f"[search_agent] 搜索'{kw}'超时({TIMEOUT_SEARCH}s)")
            return None
        except Exception as e:
            logger.warning(f"[search_agent] 搜索'{kw}'失败: {e}")
            return None

    results = await asyncio.gather(*[_search_one(kw) for kw in keywords])

    # 2. 按关键词分组 + 品牌去重
    pois_by_keyword: dict[str, list] = {}
    for kw, result in zip(keywords, results):
        if result is None:
            pois_by_keyword[kw] = []
            continue
        pois_by_keyword[kw] = _brand_dedup(_parse_pois(result))

    # 3. 按关键词均分
    selected = _even_split(pois_by_keyword, settings.AGENT_POI_PER_KEYWORD)
    if not selected:
        return []

    logger.info(
        f"[search_agent] {len(keywords)}个关键词 → 去重均分后{len(selected)}条 → 查详情"
    )

    # 4. 详情段：限流 + 单条超时 + as_completed 边收边过滤 + 累计够数提前停
    sem = asyncio.Semaphore(DETAIL_CONCURRENCY)

    async def _fetch_one(poi: dict):
        async with sem:
            try:
                raw = await asyncio.wait_for(
                    _tools["search_detail"].ainvoke({"id": poi["id"]}),
                    timeout=TIMEOUT_DETAIL,
                )
                return _parse_detail(raw)
            except asyncio.TimeoutError:
                logger.warning(
                    f"[search_agent] POI详情超时({TIMEOUT_DETAIL}s): id={poi.get('id')}"
                )
                return None
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning(f"[search_agent] POI详情失败: {e}")
                return None

    detail_tasks = [asyncio.create_task(_fetch_one(p)) for p in selected]

    passed: list[dict] = []
    detailed_count = 0
    try:
        for coro in asyncio.as_completed(detail_tasks):
            parsed = await coro
            if parsed is None:
                continue
            detailed_count += 1
            # 边收边过滤：单条走 _precise_filter 即可复用价格/评分硬约束
            if _precise_filter([parsed], filters):
                passed.append(parsed)
                if len(passed) >= settings.AGENT_MIN_SUFFICIENT:
                    break
    finally:
        # 提前停或异常都要把剩下的任务 cancel 掉，避免泄漏
        for t in detail_tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*detail_tasks, return_exceptions=True)

    logger.info(
        f"[search_agent] 详情{detailed_count}条 → 过滤后{len(passed)}条"
        f"（计划查{len(selected)}条，{'提前停' if len(passed) >= settings.AGENT_MIN_SUFFICIENT else '全部完成'}）"
    )
    return passed


# ── LLM 排序 ─────────────────────────────────────────────


async def _llm_rerank(
    state: DietState,
    pois: list[dict],
    search_context: str,
) -> LLMRerankOutput:
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
        for poi in pois
    ]

    system_prompt = RERANK_SYSTEM_PROMPT.format(
        current_time=state.get("current_time", ""),
        negative_conditions=state.get("negative_conditions", []),
        user_input=state["user_input"],
        memory_for_rerank=state.get("memory_for_rerank", "暂无用户偏好。"),
        max_recommendations=settings.MAX_RECOMMENDATIONS,
        search_context=search_context,
        scene_context=state.get("scene_context", ""),
    )

    structured_llm = llm.with_structured_output(LLMRerankOutput)

    try:
        result: LLMRerankOutput = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"餐厅列表：{json.dumps(poi_info, ensure_ascii=False)}"
            ),
        ])
        logger.info(f"[search_agent] LLM推荐 {len(result.recommendations)} 家")
        return result
    except Exception as e:
        logger.error(f"[search_agent] LLM rerank失败: {e}")
        return LLMRerankOutput(
            recommendations=[],
            disclaimer="推荐服务暂时不可用，请稍后再试",
        )


# ── 主节点 ────────────────────────────────────────────────


async def search_agent(state: DietState) -> dict:
    keywords = state.get("keywords", [])
    city = state.get("city", "")
    search_mode = state.get("search_mode", "keyword")
    location_text = state.get("location_text", "")
    filters = state.get("filters") or FilterConditions()
    radius = (filters.get("radius") or settings.DEFAULT_RADIUS)

    all_pois: list[dict] = []
    search_actions: list[str] = []
    landmark_location: str | None = None
    landmark_resolve_failed = False
    geo_failure_reason: str | None = None

    # ── Phase 1: 地标解析 ──
    if search_mode == "around" and location_text:
        landmark_location, geo_failure_reason = await _geo_resolve(location_text, city)
        if not landmark_location:
            landmark_resolve_failed = True
            search_mode = "keyword"
            reason_text = f"（{geo_failure_reason}）" if geo_failure_reason else ""
            search_actions.append(f"地标解析失败{reason_text}，降级为关键词搜索")

    # ── Phase 2: 迭代搜索 ──
    first_batch_size = min(3, len(keywords))
    keyword_queue = list(keywords[first_batch_size:])
    current_batch = keywords[:first_batch_size]

    for iteration in range(settings.AGENT_MAX_ITERATIONS):
        if not current_batch:
            break

        pois = await _search_and_detail(
            keywords=current_batch,
            city=city,
            filters=filters,
            search_type=search_mode,
            location=landmark_location,
            radius=radius,
        )

        # 合并去重
        seen_ids = {p.get("id") for p in all_pois}
        new_count = 0
        for poi in pois:
            pid = poi.get("id")
            if pid and pid not in seen_ids:
                all_pois.append(poi)
                seen_ids.add(pid)
                new_count += 1

        search_actions.append(
            f"第{iteration + 1}轮搜索关键词{current_batch}，新增{new_count}条"
        )

        # 够了就停
        if len(all_pois) >= settings.AGENT_MIN_SUFFICIENT and _is_diverse(all_pois):
            break

        # 不够或不够多样 → 准备下一轮
        if len(all_pois) < settings.AGENT_MIN_SUFFICIENT:
            if keyword_queue:
                current_batch = keyword_queue[:2]
                keyword_queue = keyword_queue[2:]
                search_actions.append("结果不足，追加关键词搜索")
            elif search_mode == "around" and radius < 5000:
                radius = min(radius * 2, 5000)
                current_batch = keywords[:2]
                search_actions.append(f"结果不足，扩大搜索半径至{radius}米")
            else:
                break
        elif not _is_diverse(all_pois):
            if keyword_queue:
                current_batch = keyword_queue[:2]
                keyword_queue = keyword_queue[2:]
                search_actions.append("品类单一，追加不同类型关键词")
            else:
                break
        else:
            break

    search_context = "；".join(search_actions)
    logger.info(f"[search_agent] 搜索完成: {search_context}")

    # ── Phase 3: 无结果快速返回 ──
    if not all_pois:
        msg = "抱歉，没有找到符合您要求的餐厅，要不要换个条件试试？"
        return {
            "final_recommendations": [],
            "filtered_pois": [],
            "detailed_pois": [],
            "result_insufficient": True,
            "disclaimer_needed": True,
            "disclaimer_message": msg,
            "hook_message": None,
            "landmark_resolve_failed": landmark_resolve_failed,
            "landmark_location": landmark_location,
            "search_mode": search_mode,
            "geo_failure_reason": geo_failure_reason,
            "search_context": search_context,
        }

    # ── Phase 4: LLM 排序推荐 ──
    rerank_result = await _llm_rerank(state, all_pois, search_context)

    return {
        "final_recommendations": [
            r.model_dump() for r in rerank_result.recommendations
        ],
        "filtered_pois": all_pois,
        "detailed_pois": all_pois,
        "result_insufficient": len(all_pois) < settings.AGENT_MIN_SUFFICIENT,
        "disclaimer_needed": bool(rerank_result.disclaimer),
        "disclaimer_message": rerank_result.disclaimer,
        "hook_message": rerank_result.hook,
        "landmark_resolve_failed": landmark_resolve_failed,
        "landmark_location": landmark_location,
        "search_mode": search_mode,
        "geo_failure_reason": geo_failure_reason,
        "search_context": search_context,
    }
