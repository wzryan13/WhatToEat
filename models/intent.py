from pydantic import BaseModel, Field
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
    intent_type: Literal["normal", "brand", "scene", "time_based"] = Field(
        description="意图类型"
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
