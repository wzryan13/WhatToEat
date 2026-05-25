import json

from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal


class FilterConditions(BaseModel):
    price_max: Optional[float] = Field(None, description="人均上限（元）", ge=0)
    price_min: Optional[float] = Field(None, description="人均下限（元）", ge=0)
    radius: Optional[int] = Field(None, description="周边搜索半径（米）", ge=0)
    open_time: Optional[str] = Field(
        None,
        description="就餐时段",
        examples=["早餐", "午餐", "晚餐", "夜宵", "下午茶"]
    )
    min_rating: Optional[float] = Field(None, description="最低评分", ge=0, le=5)

    def get(self, key: str, default=None):
        return getattr(self, key, default)


class IntentParserOutput(BaseModel):
    intent_type: Literal["normal", "brand", "scene", "time_based", "recipe", "recommend"] = Field(
        description="意图类型: normal(餐厅), brand(品牌), scene(场景), time_based(时段), recipe(菜谱搜索), recommend(菜谱推荐)"
    )
    location_text: Optional[str] = Field(
        None,
        description="原始位置文本，如成都万象城"
    )
    location_type: Literal["valid", "relative", "gps", "none", "invalid"] = Field(
        description="位置类型"
    )
    city: Optional[str] = Field(
        None,
        description="仅城市名，如成都、北京"
    )
    keywords: list[str] = Field(
        description="高德搜索关键词列表，3-6个，多角度扩展"
    )
    search_mode: Literal["keyword", "around"] = Field(
        description="搜索模式，around仅当用户明确指定距离范围时使用"
    )
    filters: FilterConditions = Field(
        default_factory=FilterConditions,
        description="过滤条件"
    )
    negative_conditions: list[str] = Field(
        default_factory=list,
        description="负向条件列表，如不辣、不吃牛肉"
    )
    has_contradiction: bool = Field(
        default=False,
        description="是否存在矛盾条件，如人均10以下的米其林"
    )
    contradiction_message: Optional[str] = Field(
        None,
        description="矛盾条件说明"
    )
    scene_context: str = Field(
        default="",
        description="动态场景描述，如晚餐、放松场景、需要口味刺激"
    )
    mood_factors: list[str] = Field(
        default_factory=list,
        description="情绪因素，如疲劳、需要安慰食物"
    )
    suggested_cuisines: list[str] = Field(
        default_factory=list,
        description="推荐搜索的具体菜品或菜系"
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_json_strings(cls, data):
        """兼容 LLM 把嵌套对象/列表序列化成 JSON 字符串的情况。"""
        if not isinstance(data, dict):
            return data
        for key in ("filters", "negative_conditions", "mood_factors", "suggested_cuisines", "keywords"):
            v = data.get(key)
            if isinstance(v, str):
                try:
                    data[key] = json.loads(v)
                except (json.JSONDecodeError, ValueError):
                    pass
        return data
