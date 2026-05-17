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

from pydantic import BaseModel, Field
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
    recommendations: List[RecipeRecommendation] = Field(
        description="个性化排序后的菜谱推荐列表（6-8 道）"
    )
    response_message: str = Field(
        description="给用户的完整回复文本（包含菜谱信息和推荐理由）"
    )


# ── Prompt ──────────────────────────────────────────────────

RAG_FORMATTER_SYSTEM_PROMPT = """你是一个个性化菜谱推荐助手。

你的任务是：
1. 根据用户画像中的**过敏/黑名单**信息，排除含有禁忌食材的菜谱
2. 根据用户的**偏好**（喜好菜系、辣度、健康目标等）对剩余菜谱重新排序
3. 为每道推荐菜谱生成个性化推荐理由
4. 生成完整的回复文本

【用户画像】
{user_profile}

【排除规则（硬约束）】
- 过敏食材: {allergies}
- 不吃的食材: {food_blacklist}
- 如果菜谱内容中包含以上任何食材，必须排除，不能推荐

【偏好规则（软约束，影响排序）】
- 喜欢的菜系: {liked_cuisines}
- 辣度偏好: {spice_tolerance}
- 健康目标: {health_goals}
- 匹配偏好的菜谱排名靠前，但不排除不匹配的

【输出要求】
- 推荐 6-8 道菜（如果过滤后不足 6 道则全部推荐）
- 每道菜附上简短的个性化推荐理由
- 回复文本要自然友好，像朋友推荐一样
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
        }

    # 获取用户画像数据
    profile_data = state.get("memory_for_rerank_data") or {}
    allergies = _extract_values(profile_data.get("allergies", []))
    food_blacklist = _extract_values(profile_data.get("food_blacklist", []))
    liked_cuisines = _extract_cuisine_tags(profile_data.get("cuisine_tags", {}))
    spice_tolerance = profile_data.get("spice_tolerance", {})
    spice_str = spice_tolerance.get("value", "未知") if isinstance(spice_tolerance, dict) else str(spice_tolerance)
    health_goals = _extract_values(profile_data.get("health_goals", []))

    # 构建菜谱信息文本
    recipes_text = _format_recipes_for_llm(rag_documents)

    # 构建用户画像摘要
    user_profile = state.get("memory_for_rerank", "暂无用户偏好信息。")

    system_prompt = RAG_FORMATTER_SYSTEM_PROMPT.format(
        user_profile=user_profile,
        allergies="、".join(allergies) if allergies else "无",
        food_blacklist="、".join(food_blacklist) if food_blacklist else "无",
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

        return {
            "response_message": result.response_message,
            "final_recommendations": [r.model_dump() for r in result.recommendations],
        }

    except Exception as e:
        logger.error(f"[rag_formatter] LLM 调用失败: {e}")
        # 降级：直接返回检索结果
        fallback_msg = _fallback_format(rag_documents)
        return {
            "response_message": fallback_msg,
            "final_recommendations": [],
        }


# ── 辅助函数 ────────────────────────────────────────────────


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

        # 截取前 500 字避免 token 爆炸
        content_preview = content[:500] + "..." if len(content) > 500 else content

        parts.append(
            f"### {i}. {dish_name}\n"
            f"- 分类: {category}\n"
            f"- 难度: {difficulty}\n"
            f"- 内容:\n{content_preview}\n"
        )
    return "\n".join(parts)


def _fallback_format(rag_documents: list) -> str:
    """降级格式化：不依赖 LLM，直接返回菜谱列表。"""
    lines = ["为你找到以下菜谱：\n"]
    for i, doc in enumerate(rag_documents[:8], 1):
        metadata = doc.get("metadata", {})
        dish_name = metadata.get("dish_name", f"菜谱{i}")
        category = metadata.get("category", "")
        lines.append(f"{i}. **{dish_name}**" + (f" ({category})" if category else ""))
    lines.append("\n需要查看具体哪道菜的做法吗？")
    return "\n".join(lines)
