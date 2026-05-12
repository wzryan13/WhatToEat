from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class MemoryFact(BaseModel):
    value: str
    source: str = ""
    confidence: float = Field(default=0.5, ge=0, le=1)
    updated_at: str = ""
    last_seen_at: str = ""


class BudgetRange(BaseModel):
    min: float | None = None
    max: float | None = None


class UserProfile(BaseModel):
    schema_version: int = 1

    allergies: list[MemoryFact] = Field(default_factory=list)
    food_blacklist: list[MemoryFact] = Field(default_factory=list)
    religious_restrictions: list[MemoryFact] = Field(default_factory=list)
    disliked_cuisines: list[MemoryFact] = Field(default_factory=list)

    spice_tolerance: MemoryFact | None = None
    sweetness: MemoryFact | None = None
    cuisine_tags: dict[str, str] = Field(default_factory=dict)
    health_goals: list[MemoryFact] = Field(default_factory=list)

    home_area: MemoryFact | None = None
    budget_solo: MemoryFact | None = None
    budget_group: MemoryFact | None = None
    active_areas: list[MemoryFact] = Field(default_factory=list)
    default_city: MemoryFact | None = None


class SessionMemory(BaseModel):
    active_city: str | None = None
    active_location_text: str | None = None
    active_budget_range: BudgetRange | None = None
    active_negative_conditions: list[str] = Field(default_factory=list)
    last_clarification_question: str | None = None
    last_result_summary: dict = Field(default_factory=dict)
    updated_at: str = ""
    expires_at: str = ""


class ProfileUpdate(BaseModel):
    field: str = Field(
        description="要更新的画像字段。可选值：allergies, food_blacklist, "
        "religious_restrictions, disliked_cuisines, spice_tolerance, sweetness, "
        "cuisine_tags.{菜系名}, health_goals, home_area, budget_solo, budget_group, default_city"
    )
    action: Literal["add", "remove", "set"] = Field(
        description="操作类型。add: 向列表字段追加新值; remove: 从列表字段移除; set: 设置/覆盖标量字段的值"
    )
    value: str = Field(
        description="具体的值，如'花生'、'微辣'、'日料'"
    )
    tag_level: str = Field(
        default="",
        description="仅 cuisine_tags 字段使用，可选 'liked' 或 'loved'"
    )
    reason: str = Field(
        default="",
        description="更新理由，简述为什么判断这是长期偏好而非临时状态"
    )


class ProfileUpdateDecision(BaseModel):
    updates: list[ProfileUpdate] = Field(default_factory=list)
    no_update_reason: str = ""
