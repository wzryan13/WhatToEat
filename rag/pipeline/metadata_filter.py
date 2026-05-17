# rag/pipeline/metadata_filter.py
"""
LLM 驱动的元数据过滤表达式生成器。
根据用户查询 + 可用元数据值，生成 Milvus expr 过滤表达式。
仅用于 category/difficulty/dish_name 级别的粗过滤。
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

from config.settings import settings

logger = logging.getLogger(__name__)


FILTER_EXPRESSION_PROMPT = ChatPromptTemplate.from_template(
    """
你是「Milvus 元数据过滤表达式生成器」。

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
- 可使用 `OR / NOT`，但需确保不会扩大歧义
- 必要时使用括号明确优先级

【Milvus 过滤表达式参考】
{reference_material}

【可用元数据取值】
{metadata_schema}

当前查询：
{query}

【输出格式（强约束）】

你 **必须且只能** 输出以下 JSON：

- 当可以生成过滤表达式时：
{{"expr": "<可直接用于 Milvus 的过滤表达式>"}}

- 当无法确定任何可靠过滤条件时：
{{"expr": "NONE"}}
"""
)


REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
REFERENCE_FILES = ("operators.md",)


def extract_first_valid_json(text: str) -> dict:
    """从 LLM 输出文本中提取第一个有效的 JSON 对象。"""
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown 代码块中提取
    fence_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    match = re.search(fence_pattern, text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试匹配花括号包裹的内容
    brace_pattern = r"\{[^{}]*\}"
    match = re.search(brace_pattern, text)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"expr": "NONE"}


class MetadataFilterExtractor:
    """LLM 驱动的 Milvus 元数据过滤表达式生成器。"""

    def __init__(self):
        self._llm = ChatDeepSeek(
            model=settings.MODEL_NAME,
            temperature=0.0,
        )
        self.reference_material = self._load_reference_material()

    async def build_filter_expression(
        self,
        query: str,
        metadata_catalog: Dict[str, Dict[str, List[str]]],
    ) -> Optional[str]:
        """
        根据查询生成 Milvus 过滤表达式。

        Args:
            query: 用户查询（通常是 rewrite 后的）
            metadata_catalog: 可用元数据值字典，格式:
                {"来源名": {"category": ["川菜", "家常菜", ...], "difficulty": [...]}}

        Returns:
            Milvus 过滤表达式字符串，或 None（无法生成可靠过滤时）
        """
        if not metadata_catalog:
            return None

        metadata_schema = self._summarize_metadata(metadata_catalog)

        try:
            prompt = FILTER_EXPRESSION_PROMPT.format_prompt(
                query=query,
                reference_material=self.reference_material,
                metadata_schema=metadata_schema,
            )
            response = await self._llm.ainvoke(list(prompt.messages))
            content = response.content.strip()

            result = extract_first_valid_json(content)
            raw = result.get("expr", "")

            expression = self._clean_expression(raw)
            logger.info("生成元数据过滤表达式: %s", expression or "NONE")
            return expression
        except Exception as exc:
            logger.warning("元数据过滤表达式生成失败: %s", exc)
            return None

    def _load_reference_material(self) -> str:
        """加载 Milvus 操作符参考文档。"""
        sections: List[str] = []
        for filename in REFERENCE_FILES:
            path = REFERENCE_DIR / filename
            try:
                sections.append(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                logger.warning("参考文件不存在: %s", path)
            except Exception as exc:
                logger.warning("读取参考文件 %s 失败: %s", path, exc)
        return "\n\n".join(sections)

    @staticmethod
    def _summarize_metadata(metadata_catalog: Dict[str, Dict[str, List[str]]]) -> str:
        """将元数据目录格式化为 prompt 可读的文本。"""
        lines = []
        for source, metadata in metadata_catalog.items():
            lines.append(f"来源: {source}")
            for key, values in metadata.items():
                sample = "、".join(values)
                lines.append(f"- {key} (共{len(values)}个): {sample}")
        return "\n".join(lines)

    @staticmethod
    def _clean_expression(raw_text: str) -> Optional[str]:
        """清理 LLM 输出的表达式文本。"""
        text = raw_text.strip()

        # 去除 markdown 代码块
        fence_pattern = r"```(?:[a-zA-Z0-9_+-]+)?\s*([\s\S]*?)```"
        match = re.search(fence_pattern, text)
        if match:
            text = match.group(1).strip()

        # 去除多余引号
        if text.startswith('"') and text.endswith('"') and len(text) >= 2:
            text = text[1:-1].strip()

        if text.upper() == "NONE" or not text:
            return None

        return text
