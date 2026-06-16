# rag/pipeline/query_understanding.py
"""
RAG 查询理解模块 — 单次 LLM 调用同时完成「查询改写」+「元数据过滤表达式生成」。

合并前是两次串行 LLM 调用（generation.rewrite_query → metadata_filter.build_filter_expression），
metadata_filter 依赖 rewrite 的输出，形成 ~2s 的串行等待。
合并为一次 structured_output 调用，输出 {rewritten_query, expr}，省掉一次往返。

prompt = 原「改写」prompt + 原「过滤」prompt 逐字拼接（规则保持不变），
唯一改动：输出契约从原来的纯文本 JSON 改为 structured output。
expr 输出字符串 "NONE" 表示不生成过滤（与原版一致）。
"""

import logging
import re
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

from config.settings import settings

logger = logging.getLogger(__name__)


# ── 结构化输出模型 ──────────────────────────────────────────


class QueryUnderstandingOutput(BaseModel):
    """查询理解的结构化输出：改写后的查询 + 过滤表达式（"NONE" 表示不过滤）。"""

    rewritten_query: str = Field(
        description="重写后的自然语言查询语句"
    )
    expr: str = Field(
        description="可直接用于 Milvus 的过滤表达式；无法确定任何可靠过滤条件时输出字符串 NONE"
    )


# ── Prompt（= 原改写 prompt + 原过滤 prompt 逐字拼接） ──────────


QUERY_UNDERSTANDING_PROMPT = ChatPromptTemplate.from_template(
    """你是食谱数据库的查询理解助手。给定用户查询，你要**同时**完成两件事，并输出结构化结果 rewritten_query 与 expr。

================ 任务一：查询改写（rewritten_query）================

你是食谱数据库的智能搜索助手。你的任务是将用户的输入优化为一个**清晰、自然且完整**的句子，以便进行语义搜索。

**准则：**
1.  **仅限自然语言：** 不要输出关键词堆砌。重写后的查询必须是一个语法完整的句子或自然的问句（例如"我该如何制作……"或"有哪些……"）。
2.  **严禁幻觉：** 除非用户明确提及，否则不要添加具体的形容词（如"简单的"、"快速的"、"健康的"、"辣的"）。
3.  **澄清但不设限：** 如果查询很模糊（例如"我饿了"），将其重写为请求食物推荐的通用但清晰的句子，除非用户指定，否则不要假设是午餐还是晚餐。
4.  **扩展概念：** 对于推荐类查询，可以适当扩展相关概念以提高检索效果。例如"荤素搭配"可以扩展为"既有肉类又有蔬菜的菜品"。
5.  **保持语气：** 保持礼貌和对话感，与原查询的语言风格相匹配。

**示例：**

-   Original: "我想做点吃的"
    -> Rewritten: "你能推荐一些适合我做的食谱吗？"
    *（解释：将模糊的愿望转化为清晰的推荐请求，没有擅自假设是"晚餐"或"简单"的菜。）*

-   Original: "今晚吃啥？"
    -> Rewritten: "今晚晚餐有什么好的食谱推荐吗？"
    *（解释："今晚"暗示了晚餐场景，将其转化为自然的问句。）*

-   Original: "有鸡蛋和西红柿，能做什么"
    -> Rewritten: "用鸡蛋和西红柿可以做什么菜？"
    *（解释：澄清了利用特定食材烹饪的意图，保留了问句格式。）*

-   Original: "红烧肉做法"
    -> Rewritten: "如何制作红烧肉？"
    *（解释：微调语法使其成为完整的句子，意图保持不变。）*

-   Original: "来点甜的"
    -> Rewritten: "给我看一些关于甜点或甜食的食谱。"
    *（解释：将"甜的"扩展为自然的语义类别"甜点"，并组成完整句子。）*

-   Original: "有什么荤素搭配的家常菜？"
    -> Rewritten: "有哪些既有肉类又有蔬菜的家常菜？"
    *（解释：将"荤素搭配"扩展为"既有肉类又有蔬菜"，使语义更清晰，便于检索。）*

-   Original: "推荐几道清淡的菜"
    -> Rewritten: "有哪些口味清淡、不油腻的菜品？"
    *（解释：将"清淡"扩展为"口味清淡、不油腻"，提高检索准确性。）*

只输出1句重写后的查询，禁止添加前缀/后缀/解释/Markdown/项目符号/标题，禁止多行，仅返回重写后的自然语言查询:

================ 任务二：元数据过滤表达式（expr）================

你的任务是：**根据用户查询，判断是否可以生成一个可直接用于 Milvus `expr` 参数的布尔过滤表达式**。
**只有在条件明确、无歧义、不会明显损害召回的情况下，才允许生成过滤表达式；否则必须放弃过滤。**

【最高优先级原则】

- 元数据过滤是**精确约束**，不是语义理解或推理
- 只有当用户**明确表达**了可直接映射到元数据字段的条件时，才生成过滤
- 任何不确定、模糊、需要推断的情况，**一律不生成过滤**

【允许使用的字段（严格限制）】

你 **只能** 使用以下 metadata 字段：
- `category`
- `dish_name`
- `difficulty`

禁止使用任何未列出的字段。
**字段值必须严格来自【可用元数据取值】，禁止猜测、扩展或改写。**

【字段使用规则】

1. category  
- 仅在用户明确指定菜系或菜品大类时使用  
- 示例：川菜、家常菜、凉菜  
- 不得从口味、场景、食材等信息中推断 category  

2. dish_name  
- 仅在用户明确提及具体菜名时使用  
- 允许使用 `LIKE` / `ILIKE` 进行模糊匹配  
- 不得从食材或描述中推断菜名  

3. difficulty（高风险字段）  
- 仅在用户**明确提到难度要求**时使用  
  - 如：简单 / 新手 / 困难 / 复杂  
- 未明确提及难度，一律禁止使用该字段  

【逻辑组合规则】

- 仅在**所有条件都高度确定**时才使用 `AND`
- 若多个条件存在确定性差异，只保留**最确定的条件**
- 可使用 `OR / NOT`，但需确保不会扩大歧义
- 必要时使用括号明确优先级

禁止输出任何解释、注释、Markdown、换行或多余文本。

【Milvus 过滤表达式参考】
{reference_material}

【可用元数据取值】
{metadata_schema}

================ 当前查询 ================
{query}

================ 输出（强约束） ================
请输出结构化结果：
- rewritten_query：改写后的自然语言查询语句
- expr：可直接用于 Milvus 的过滤表达式；当无法确定任何可靠过滤条件时，输出字符串 "NONE"
"""
)


REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
REFERENCE_FILES = ("operators.md",)


def _load_reference_material() -> str:
    """加载 Milvus 操作符参考文档。"""
    sections = []
    for filename in REFERENCE_FILES:
        path = REFERENCE_DIR / filename
        try:
            sections.append(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning("参考文件不存在: %s", path)
        except Exception as exc:
            logger.warning("读取参考文件 %s 失败: %s", path, exc)
    return "\n\n".join(sections)


def _summarize_metadata(metadata_catalog: dict) -> str:
    """将元数据目录格式化为 prompt 可读文本。"""
    lines = []
    for source, metadata in metadata_catalog.items():
        lines.append(f"来源: {source}")
        for key, values in metadata.items():
            sample = "、".join(values)
            lines.append(f"- {key} (共{len(values)}个): {sample}")
    return "\n".join(lines)


def _clean_expression(raw_text: Optional[str]) -> Optional[str]:
    """清理 LLM 输出的过滤表达式：去代码块、去引号、NONE→None。"""
    if not raw_text:
        return None
    text = raw_text.strip()

    fence_pattern = r"```(?:[a-zA-Z0-9_+-]+)?\s*([\s\S]*?)```"
    match = re.search(fence_pattern, text)
    if match:
        text = match.group(1).strip()

    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = text[1:-1].strip()

    if text.upper() == "NONE" or not text:
        return None
    return text


# ── 模块 ────────────────────────────────────────────────────


class QueryUnderstandingModule:
    """单次 LLM 调用完成查询改写 + 元数据过滤表达式生成。"""

    def __init__(self):
        self._llm = ChatDeepSeek(model=settings.MODEL_NAME, temperature=0.0)
        self.reference_material = _load_reference_material()

    async def understand(
        self,
        query: str,
        metadata_catalog: Optional[dict] = None,
    ) -> Tuple[str, Optional[str]]:
        """
        Args:
            query: 用户原始查询
            metadata_catalog: 可用元数据值，格式 {"来源": {"category": [...], "difficulty": [...]}}

        Returns:
            (rewritten_query, filter_expr) —— filter_expr 为 None 表示不过滤；
            任一失败时逐字段兜底（rewritten→原 query，expr→None）
        """
        metadata_schema = (
            _summarize_metadata(metadata_catalog)
            if metadata_catalog
            else "（无可用元数据，expr 一律输出 NONE）"
        )

        structured_llm = self._llm.with_structured_output(QueryUnderstandingOutput)
        prompt = QUERY_UNDERSTANDING_PROMPT.format_prompt(
            query=query,
            reference_material=self.reference_material,
            metadata_schema=metadata_schema,
        )

        try:
            result: QueryUnderstandingOutput = await structured_llm.ainvoke(
                list(prompt.messages)
            )
        except Exception as exc:
            logger.warning("查询理解失败，逐字段兜底: %s", exc)
            return query, None

        # 逐字段兜底
        rewritten = (result.rewritten_query or "").strip() or query
        filter_expr = _clean_expression(result.expr)

        if rewritten != query:
            logger.info("查询改写: '%s' -> '%s'", query, rewritten)
        logger.info("生成元数据过滤表达式: %s", filter_expr or "NONE")

        return rewritten, filter_expr
