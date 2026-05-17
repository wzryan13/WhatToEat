# rag/pipeline/generation.py
"""查询改写模块 — 使用 LLM 将模糊查询优化为语义搜索友好的句子。"""

import logging

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

from config.settings import settings

logger = logging.getLogger(__name__)


class QueryRewriteOutput(BaseModel):
    """查询改写的结构化输出"""
    rewritten_query: str = Field(description="重写后的自然语言查询语句")


REWRITE_PROMPT_TEMPLATE = """
你是食谱数据库的智能搜索助手。你的任务是将用户的输入优化为一个**清晰、自然且完整**的句子，以便进行语义搜索。

**准则：**
1.  **仅限自然语言：** 不要输出关键词堆砌。重写后的查询必须是一个语法完整的句子或自然的问句。
2.  **严禁幻觉：** 除非用户明确提及，否则不要添加具体的形容词（如"简单的"、"快速的"、"健康的"、"辣的"）。
3.  **澄清但不设限：** 如果查询很模糊（例如"我饿了"），将其重写为请求食物推荐的通用但清晰的句子。
4.  **扩展概念：** 对于推荐类查询，可以适当扩展相关概念以提高检索效果。例如"荤素搭配"可以扩展为"既有肉类又有蔬菜的菜品"。
5.  **保持语气：** 保持礼貌和对话感，与原查询的语言风格相匹配。

**示例：**
-   "我想做点吃的" -> "你能推荐一些适合我做的食谱吗？"
-   "今晚吃啥？" -> "今晚晚餐有什么好的食谱推荐吗？"
-   "有鸡蛋和西红柿，能做什么" -> "用鸡蛋和西红柿可以做什么菜？"
-   "红烧肉做法" -> "如何制作红烧肉？"
-   "来点甜的" -> "给我看一些关于甜点或甜食的食谱。"
-   "有什么荤素搭配的家常菜？" -> "有哪些既有肉类又有蔬菜的家常菜？"

原始查询: {query}
"""

REWRITE_PROMPT = ChatPromptTemplate.from_template(REWRITE_PROMPT_TEMPLATE)


class GenerationIntegrationModule:
    """查询改写模块 — 使用 LLM 将模糊用户输入转为语义搜索友好的查询。"""

    def __init__(self):
        self._llm = ChatDeepSeek(
            model=settings.MODEL_NAME,
            temperature=0.0,
        )

    async def rewrite_query(self, query: str) -> str:
        """
        使用 LLM 将模糊查询改写为更精确的搜索查询。

        Args:
            query: 用户原始输入

        Returns:
            改写后的查询字符串
        """
        structured_llm = self._llm.with_structured_output(QueryRewriteOutput)
        prompt = REWRITE_PROMPT.format_prompt(query=query)

        try:
            result: QueryRewriteOutput = await structured_llm.ainvoke(
                list(prompt.messages)
            )
            rewritten = result.rewritten_query.strip()

            if rewritten != query:
                logger.info(f"查询改写: '{query}' -> '{rewritten}'")
            else:
                logger.info(f"查询无需改写: '{query}'")

            return rewritten
        except Exception as e:
            logger.warning(f"查询改写失败，使用原始查询: {e}")
            return query
