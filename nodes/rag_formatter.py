# nodes/rag_formatter.py
"""
RAG 个性化 + 输出节点。

职责:
4. 硬约束过滤 — 扫描文档内容，排除含过敏/黑名单食材的菜谱
5. 软偏好 LLM 重排 — cuisine_tags、health_goals、spice_tolerance 影响排序 + 推荐理由
6. 格式化输出 top 6-8 卡片
"""

import json
import logging
from typing import List

from pydantic import BaseModel, Field, model_validator
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_deepseek import ChatDeepSeek

from config.settings import settings
from models.state import DietState

logger = logging.getLogger(__name__)

llm = ChatDeepSeek(model=settings.MODEL_NAME)


# ── Pydantic 输出模型 ──────────────────────────────────────


class RecipeRecommendation(BaseModel):
    dish_name: str = Field(description="菜名")
    reason: str = Field(description="个性化推荐理由（结合用户画像）")
    rank: int = Field(description="推荐排名，1 为最推荐")


class RAGFormatterOutput(BaseModel):
    response_message: str = Field(
        description="给用户的开场白（2句话内），说明找到了哪些菜谱、各有什么特点，语气自然亲切"
    )
    recommendations: List[RecipeRecommendation] = Field(
        description="个性化排序后的菜谱推荐列表（6-8 道），dish_name 必须与输入菜谱名完全一致"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_json_strings(cls, data):
        """兼容 LLM 把 recommendations 列表整体或单项序列化成 JSON 字符串的情况。"""
        if not isinstance(data, dict):
            return data
        v = data.get("recommendations")
        if isinstance(v, str):
            try:
                data["recommendations"] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                data["recommendations"] = []
        elif isinstance(v, list):
            coerced = []
            for item in v:
                if isinstance(item, str):
                    try:
                        coerced.append(json.loads(item))
                    except (json.JSONDecodeError, ValueError):
                        continue
                else:
                    coerced.append(item)
            data["recommendations"] = coerced
        return data


# ── Prompt ──────────────────────────────────────────────────

RAG_FORMATTER_SYSTEM_PROMPT = """你是一个个性化菜谱推荐助手。

你的任务是：
1. 根据用户的**偏好**（喜好菜系、辣度、健康目标等）对菜谱重新排序
2. 为每道推荐菜谱生成个性化推荐理由
3. 生成一句自然的开场白

注意：含有过敏/禁忌食材的菜谱已经在上游被过滤掉了，你收到的都是安全的菜谱。

【用户画像】
{user_profile}

【偏好规则（软约束，影响排序）】
- 喜欢的菜系: {liked_cuisines}
- 辣度偏好: {spice_tolerance}
- 健康目标: {health_goals}
- 匹配偏好的菜谱排名靠前，但不排除不匹配的

【输出字段说明】
- response_message: 开场白，2 句话以内，说明找到了哪些菜谱、各有什么特点，语气自然轻松，不要出现"以下是"这类机械表达
- recommendations: 按偏好排序的菜谱列表，每道菜含 dish_name / reason / rank
  - reason: 结合用户画像的个性化推荐理由，一句话，说明为什么推荐这道菜、适合什么场景
  - dish_name 必须与输入菜谱名完全一致，不要修改
"""


# ── 节点函数 ────────────────────────────────────────────────


async def rag_formatter(state: DietState) -> dict:
    """
    RAG 个性化 + 格式化节点。
    接收 rag_documents，应用画像过滤/排序，输出最终推荐。
    """
    rag_documents = state.get("rag_documents", [])

    if not rag_documents:
        return {
            "response_message": "抱歉，没有找到相关的菜谱，要不要换个关键词试试？",
            "final_recommendations": [],
            "conversation_history": [{
                "role": "assistant",
                "content": "response_message",
                "tool_summary": "菜谱检索无结果",
            }],
        }

    # 获取用户画像数据
    profile_data = state.get("memory_for_rerank_data") or {}
    allergies = _extract_values(profile_data.get("allergies", []))
    food_blacklist = _extract_values(profile_data.get("food_blacklist", []))

    # ── 硬约束过滤（代码预过滤，不依赖 LLM） ──
    banned_keywords = set(allergies + food_blacklist)
    if banned_keywords:
        filtered_documents = []
        for doc in rag_documents:
            content = doc.get("content", "")
            dish_name = doc.get("metadata", {}).get("dish_name", "")
            text_to_check = content + dish_name
            if not any(kw in text_to_check for kw in banned_keywords):
                filtered_documents.append(doc)
            else:
                excluded_kw = [kw for kw in banned_keywords if kw in text_to_check]
                logger.info(f"[rag_formatter] 硬过滤排除: {dish_name} (含: {excluded_kw})")
        logger.info(f"[rag_formatter] 硬过滤: {len(rag_documents)} -> {len(filtered_documents)}")
        rag_documents = filtered_documents

    if not rag_documents:
        return {
            "response_message": "抱歉，根据你的饮食限制，暂时没有找到合适的菜谱。要不要换个关键词试试？",
            "final_recommendations": [],
            "conversation_history": [{
                "role": "assistant",
                "content": "response_message",
                "tool_summary": "菜谱检索无结果",
            }],
        }

    intent_type = state.get("intent_type", "recipe")

    # ── recipe：直接返回检索结果，不调 LLM ──
    if intent_type == "recipe":
        recs = _docs_to_recommendations(rag_documents)
        msg = "为你找到以下菜谱："
        return {
            "response_message": msg,
            "final_recommendations": recs,
            "conversation_history": [{
                "role": "assistant",
                "content": msg,
                "tool_summary": _build_recipe_summary(state, recs),
            }],
        }

    # ── recommend：调 LLM 做软偏好排序 + 推荐理由 ──
    liked_cuisines = _extract_cuisine_tags(profile_data.get("cuisine_tags", {}))
    spice_tolerance = profile_data.get("spice_tolerance", {})
    spice_str = spice_tolerance.get("value", "未知") if isinstance(spice_tolerance, dict) else str(spice_tolerance)
    health_goals = _extract_values(profile_data.get("health_goals", []))

    recipes_text = _format_recipes_for_llm(rag_documents)
    user_profile = state.get("memory_for_rerank", "暂无用户偏好信息。")

    system_prompt = RAG_FORMATTER_SYSTEM_PROMPT.format(
        user_profile=user_profile,
        liked_cuisines="、".join(liked_cuisines) if liked_cuisines else "无特别偏好",
        spice_tolerance=spice_str,
        health_goals="、".join(health_goals) if health_goals else "无",
    )

    try:
        structured_llm = llm.with_structured_output(RAGFormatterOutput)
        result: RAGFormatterOutput = await structured_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"以下是检索到的菜谱:\n\n{recipes_text}"),
        ])

        logger.info(f"[rag_formatter] 推荐 {len(result.recommendations)} 道菜")

        doc_index = {
            doc.get("metadata", {}).get("dish_name", ""): doc
            for doc in rag_documents
        }

        final = []
        for r in result.recommendations:
            matched_doc = doc_index.get(r.dish_name, {})
            raw_meta = matched_doc.get("metadata", {})
            final.append({
                "dish_name": r.dish_name,
                "reason": r.reason,
                "content": matched_doc.get("content", ""),
                "category": raw_meta.get("category", ""),
                "difficulty": raw_meta.get("difficulty", ""),
            })

        return {
            "response_message": result.response_message,
            "final_recommendations": final,
            "conversation_history": [{
                "role": "assistant",
                "content": result.response_message,
                "tool_summary": _build_recipe_summary(state, final),
            }],
        }

    except Exception as e:
        logger.error(f"[rag_formatter] LLM 调用失败: {e}")
        recs = _docs_to_recommendations(rag_documents)
        msg = "为你找到以下菜谱："
        return {
            "response_message": msg,
            "final_recommendations": recs,
            "conversation_history": [{
                "role": "assistant",
                "content": msg,
                "tool_summary": _build_recipe_summary(state, recs),
            }],
        }


def _build_recipe_summary(state: DietState, recs: list) -> str:
    """压缩菜谱推荐结果"""
    intent_type = state.get("intent_type", "recipe")
    query = state.get("rag_query", "") or "、".join(state.get("keywords", []))

    if intent_type == "recipe":
        # 精确搜索：用户问的是具体菜的做法
        names = [r.get("dish_name", "") for r in recs[:3]]
        return f"查找'{query}'菜谱，找到{len(recs)}个：{'、'.join(names)}"
    else:
        # recommend：推荐类
        names = [r.get("dish_name", "") for r in recs[:5]]
        categories = list({r.get("category", "") for r in recs if r.get("category")})
        parts = [f"推荐{len(recs)}道菜谱"]
        if categories:
            parts.append(f"涵盖{'、'.join(categories[:3])}")
        parts.append(f"包括{'、'.join(names)}")
        return "，".join(parts)

# ── 辅助函数 ────────────────────────────────────────────────


def _docs_to_recommendations(rag_documents: list) -> list:
    """将 rag_documents 转为统一的 final_recommendations 结构。"""
    recs = []
    for doc in rag_documents:
        meta = doc.get("metadata", {})
        recs.append({
            "dish_name": meta.get("dish_name", ""),
            "reason": "",
            "content": doc.get("content", ""),
            "category": meta.get("category", ""),
            "difficulty": meta.get("difficulty", ""),
        })
    return recs


def _extract_values(memory_facts: list) -> List[str]:
    """从 MemoryFact 列表中提取 value 字段。"""
    values = []
    for fact in memory_facts:
        if isinstance(fact, dict):
            v = fact.get("value", "")
        else:
            v = str(fact)
        if v:
            values.append(v)
    return values


def _extract_cuisine_tags(cuisine_tags: dict) -> List[str]:
    """提取喜欢/最爱的菜系。"""
    liked = []
    for cuisine, level in cuisine_tags.items():
        if level in ("liked", "loved"):
            liked.append(cuisine)
    return liked


def _format_recipes_for_llm(rag_documents: list) -> str:
    """将菜谱文档格式化为 LLM 可读的文本。"""
    parts = []
    for i, doc in enumerate(rag_documents, 1):
        metadata = doc.get("metadata", {})
        dish_name = metadata.get("dish_name", f"菜谱{i}")
        category = metadata.get("category", "未分类")
        difficulty = metadata.get("difficulty", "未知")
        content = doc.get("content", "")

        content_preview = content[:1000] + "..." if len(content) > 1000 else content

        parts.append(
            f"### {i}. {dish_name}\n"
            f"- 分类: {category}\n"
            f"- 难度: {difficulty}\n"
            f"- 内容:\n{content_preview}\n"
        )
    return "\n".join(parts)
