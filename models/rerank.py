from pydantic import BaseModel, Field
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
