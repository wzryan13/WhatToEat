# rag/pipeline/query_understanding.py
"""
RAG 查询理解模块 — 单次 LLM 调用同时完成「查询改写」+「元数据过滤表达式生成」。

合并前是两次串行 LLM 调用（generation.rewrite_query → metadata_filter.build_filter_expression），
metadata_filter 依赖 rewrite 的输出，形成 ~2s 的串行等待。
合并为一次 structured_output 调用，输出 {rewritten_query, filter_expr}，省掉一次往返。

两块规则在 prompt 里分节隔开：
- 改写：自然语言、不堆砌关键词、不幻觉
- 过滤：精确约束，只用 category/dish_name/difficulty，不确定就返回 null（弃权）
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, Field, model_validator
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from openai import AsyncOpenAI

from config.settings import settings

logger = logging.getLogger(__name__)


# student（本地微调模型）专用：训练时 prompt 末尾的 JSON 格式指令（与 finetune 训练一致）
JSON_DIRECTIVE = (
    "\n\n【输出格式·严格】只输出一个 JSON 对象，形如 "
    '{"rewritten_query": "改写后的完整句子", "filter_expr": "<Milvus 过滤表达式>"}，'
    "无法可靠生成过滤时 filter_expr 取 null。不要输出任何解释、复述或多余文字。"
)


def _parse_json(text: str) -> dict:
    """从模型输出里提取 JSON 对象（容错 markdown fence / 前后多余文本）。"""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return {}


# ── 结构化输出模型 ──────────────────────────────────────────


class QueryUnderstandingOutput(BaseModel):
    """查询理解的结构化输出：改写后的查询 + 可选的过滤表达式。"""

    rewritten_query: str = Field(
        description="重写后的自然语言查询语句，语法完整、不堆砌关键词"
    )
    filter_expr: Optional[str] = Field(
        default=None,
        description="可直接用于 Milvus expr 参数的过滤表达式；"
                    "无法确定可靠过滤条件时返回 null，这是预期且正确的行为，不要硬编",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_none(cls, data):
        """把 LLM 可能输出的 'NONE'/'none'/'' 统一成 None。"""
        if not isinstance(data, dict):
            return data
        v = data.get("filter_expr")
        if isinstance(v, str) and v.strip().upper() in ("NONE", ""):
            data["filter_expr"] = None
        return data


# ── Prompt ──────────────────────────────────────────────────


QUERY_UNDERSTANDING_PROMPT = ChatPromptTemplate.from_template(
    """你是食谱数据库的查询理解助手。给定用户查询，你要**同时**完成两件事，并输出结构化结果：
（1）把查询改写为适合语义检索的自然句子；（2）判断能否生成一个 Milvus 元数据过滤表达式。

================ 任务一：查询改写（rewritten_query）================

将用户输入优化为一个**清晰、自然且完整**的句子，以便进行语义搜索。

准则：
1. 仅限自然语言：不要输出关键词堆砌，必须是语法完整的句子或自然问句。
2. 严禁幻觉：除非用户明确提及，否则不要添加具体形容词（如"简单的""快速的""健康的""辣的"）。
3. 澄清但不设限：查询很模糊时（如"我饿了"），改写为请求食物推荐的通用但清晰的句子。
4. 扩展概念：推荐类查询可适当扩展相关概念（"荤素搭配"→"既有肉类又有蔬菜的菜品"）。
5. 保持语气：与原查询的语言风格相匹配。

示例：
- "我想做点吃的" -> "你能推荐一些适合我做的食谱吗？"
- "红烧肉做法" -> "如何制作红烧肉？"
- "有鸡蛋和西红柿，能做什么" -> "用鸡蛋和西红柿可以做什么菜？"

================ 任务二：元数据过滤表达式（filter_expr）================

**只有在条件明确、无歧义、不会明显损害召回时，才生成过滤表达式；否则必须返回 null。**

【最高优先级原则】
- 元数据过滤是精确约束，不是语义理解或推理。
- 只有当用户明确表达了可直接映射到元数据字段的条件时，才生成过滤。
- 任何不确定、模糊、需要推断的情况，一律返回 null。

【允许使用的字段（严格限制）】
只能使用：`category`、`dish_name`、`difficulty`。禁止任何未列出的字段。
字段值必须严格来自【可用元数据取值】，禁止猜测、扩展或改写。

【字段使用规则】
1. category：仅当用户明确指定菜系/菜品大类时使用；只能从可用取值中选；不得从口味/场景/食材推断。
2. dish_name：仅当用户明确提及具体菜名时使用；允许 LIKE 模糊匹配；不得从食材或描述推断。
3. difficulty（高风险）：仅当用户明确提到难度（简单/新手/困难/复杂）时使用；未明确提及一律禁止。

【逻辑组合】仅在所有条件都高度确定时才用 AND；可用 OR/NOT 但勿扩大歧义；必要时用括号。

【Milvus 过滤表达式参考】
{reference_material}

【可用元数据取值】
{metadata_schema}

【弃权示范（重要）】
- 查询 "今晚做什么菜好呢" → 无任何明确可映射字段 → filter_expr = null
- 查询 "推荐点好吃的" → 模糊 → filter_expr = null

================ 当前查询 ================
{query}

================ 输出 ================
输出 rewritten_query（字符串）与 filter_expr（字符串或 null）。
filter_expr 无法可靠生成时必须为 null，不要为了填字段而硬编。
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
        self.backend = settings.QU_BACKEND
        self.reference_material = _load_reference_material()
        if self.backend == "local":
            # 本地微调模型（OpenAI 兼容 endpoint，如 LM Studio server）
            self._client = AsyncOpenAI(
                base_url=settings.QU_LOCAL_BASE_URL, api_key="lm-studio"
            )
        else:
            self._llm = ChatDeepSeek(model=settings.MODEL_NAME, temperature=0.0)

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
            (rewritten_query, filter_expr) —— 任一失败时逐字段兜底（rewritten→原 query，expr→None）
        """
        metadata_schema = (
            _summarize_metadata(metadata_catalog)
            if metadata_catalog
            else "（无可用元数据，filter_expr 一律返回 null）"
        )

        if self.backend == "local":
            return await self._understand_local(query, metadata_schema)

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
        filter_expr = _clean_expression(result.filter_expr)

        if rewritten != query:
            logger.info("查询改写: '%s' -> '%s'", query, rewritten)
        logger.info("生成元数据过滤表达式: %s", filter_expr or "NONE")

        return rewritten, filter_expr

    async def _understand_local(
        self, query: str, metadata_schema: str
    ) -> Tuple[str, Optional[str]]:
        """本地微调模型后端：用训练时的 prompt（+JSON 指令）调本地 endpoint，解析 JSON。"""
        prompt = QUERY_UNDERSTANDING_PROMPT.format_messages(
            query=query,
            reference_material=self.reference_material,
            metadata_schema=metadata_schema,
        )
        content = prompt[0].content + JSON_DIRECTIVE
        try:
            resp = await self._client.chat.completions.create(
                model=settings.QU_LOCAL_MODEL,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                max_tokens=256,
            )
            data = _parse_json(resp.choices[0].message.content or "")
        except Exception as exc:
            logger.warning("本地查询理解失败，逐字段兜底: %s", exc)
            return query, None

        rewritten = (data.get("rewritten_query") or "").strip() or query
        filter_expr = _clean_expression(data.get("filter_expr"))
        if rewritten != query:
            logger.info("查询改写(local): '%s' -> '%s'", query, rewritten)
        logger.info("生成元数据过滤表达式(local): %s", filter_expr or "NONE")
        return rewritten, filter_expr
