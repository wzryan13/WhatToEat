import json

from pydantic import BaseModel, Field, model_validator
from typing import Optional


class Recommendation(BaseModel):
    id: str = Field(description="餐厅POI ID")
    name: str = Field(description="餐厅名称")
    category: str = Field(description="所属品类，使用用户搜索的关键词分类，如麻辣烫、冒菜")
    reason: str = Field(description="推荐理由，一句话")
    is_open: Optional[bool] = Field(
        None,
        description="当前是否营业，无法判断时为None"
    )


class LLMRerankOutput(BaseModel):
    recommendations: list[Recommendation] = Field(
        description="推荐餐厅列表，最多5家，按推荐度排序"
    )
    disclaimer: Optional[str] = Field(
        None,
        description="需要附加的免责提示，如忌口类推荐的不确定性说明"
    )
    hook: Optional[str] = Field(
        None,
        description="对话钩子，整体推荐后自然试探用户的一个偏好维度"
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
