# nodes/rag_agent.py
"""
RAG 检索节点 — 纯相关性检索，不碰用户画像。

职责:
1. 从 state 取查询信息
2. 可选：从画像取 disliked_cuisines 构建 category 粗过滤 expr
3. 调用 rag_service.search_recipes()
4. 返回 rag_documents（top_k=10 相关性排序文档列表）
"""

import logging
from models.state import DietState
from rag.rag_service import get_rag_service

logger = logging.getLogger(__name__)


async def rag_agent(state: DietState) -> dict:
    """
    RAG 检索节点：执行菜谱检索流程。
    只管相关性，画像个性化在 rag_formatter 中完成。
    """
    service = get_rag_service()

    if not service:
        logger.error("RAG 服务未初始化")
        return {
            "rag_documents": [],
            "rag_query": None,
            "error_message": "菜谱检索服务暂不可用，请稍后再试。",
        }

    # 构建查询：优先用 keywords 组合，fallback 到 user_input
    keywords = state.get("keywords", [])
    user_input = state.get("user_input", "")
    query = " ".join(keywords) if keywords else user_input

    # 可选：从画像取 disliked_cuisines 构建粗过滤 expr
    extra_expr = _build_category_filter(state)

    logger.info(f"[rag_agent] 开始检索, query='{query}', expr={extra_expr}")

    # 调用 RAG 服务
    docs = await service.search_recipes(
        query=query,
        extra_expr=extra_expr,
    )

    # 将 Document 对象转为 dict 方便下游使用
    rag_documents = []
    for doc in docs:
        rag_documents.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "rerank_score": doc.metadata.get("rerank_score", 0.0),
            "retrieval_score": doc.metadata.get("retrieval_score", 0.0),
        })

    logger.info(f"[rag_agent] 检索完成, 返回 {len(rag_documents)} 条菜谱")

    return {
        "rag_documents": rag_documents,
        "rag_query": query,
        "rag_filter_expr": extra_expr,
    }


def _build_category_filter(state: DietState) -> str | None:
    """
    从用户画像中提取 disliked_cuisines，构建 category 级粗过滤 expr。
    这是检索阶段唯一使用画像的地方（仅排除不喜欢的菜系）。
    """
    profile_data = state.get("memory_for_rerank_data") or {}
    disliked = profile_data.get("disliked_cuisines", [])

    if not disliked:
        return None

    # 构建 Milvus expr: category != "烧烤" and category != "美式"
    conditions = []
    for cuisine in disliked:
        # MemoryFact 格式可能是 dict 或 str
        value = cuisine.get("value", cuisine) if isinstance(cuisine, dict) else cuisine
        if value:
            conditions.append(f'category != "{value}"')

    if not conditions:
        return None

    expr = " and ".join(conditions)
    logger.info(f"[rag_agent] 画像粗过滤 expr: {expr}")
    return expr
