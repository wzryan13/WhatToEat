from pydantic import BaseModel, Field
from typing import Optional


class POIBasic(BaseModel):
    """关键词搜/周边搜返回的基础字段"""
    id: str
    name: str
    address: Optional[str] = None
    typecode: Optional[str] = None


class POIDetail(BaseModel):
    """POI详情完整字段"""
    id: str
    name: str
    location: Optional[str] = Field(None, description="经纬度，格式：longitude,latitude")
    address: Optional[str] = None
    business_area: Optional[str] = None
    city: Optional[str] = None
    type: Optional[str] = Field(None, description="类型，格式：大类;中类;小类")
    cost: Optional[str] = Field(None, description="人均消费（元）")
    opentime2: Optional[str] = Field(None, description="完整营业时间，含星期和节假日")
    open_time: Optional[str] = Field(None, description="简化营业时间")
    rating: Optional[str] = Field(None, description="评分")
    photos: Optional[dict] = None
    meal_ordering: Optional[str] = None