from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ItemCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=1024)


class ItemRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None
    created_at: datetime


class ErrorResponse(BaseModel):
    """全局异常 handler 的统一返回结构。"""

    code: str
    message: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    # 续聊时回传上一轮 SSE session 事件里的 thread_id
    thread_id: str | None = None
    # 可选的外部用户标识；缺省则按匿名用户处理
    user_id: str | None = None
