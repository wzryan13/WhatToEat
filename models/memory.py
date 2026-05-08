from __future__ import annotations

from pydantic import BaseModel, Field


class MemoryFact(BaseModel):
    value: str = Field(description="记忆值")
    source: str = Field(default="explicit_user_input", description="来源")
    confidence: float = Field(default=0.9, ge=0, le=1, description="置信度")
    updated_at: str = Field(default="", description="最后更新时间")
    last_seen_at: str = Field(default="", description="最后一次看到该信息的时间")


class BudgetRange(BaseModel):
    min: float | None = None
    max: float | None = None


class UserProfile(BaseModel):
    schema_version: int = 1
    default_city: MemoryFact | None = None
    dietary_restrictions: list[MemoryFact] = Field(default_factory=list)
    allergies: list[MemoryFact] = Field(default_factory=list)


class SessionMemory(BaseModel):
    active_city: str | None = None
    active_location_text: str | None = None
    active_budget_range: BudgetRange | None = None
    active_negative_conditions: list[str] = Field(default_factory=list)
    last_clarification_question: str | None = None
    last_result_summary: dict = Field(default_factory=dict)
    updated_at: str = ""
    expires_at: str = ""
