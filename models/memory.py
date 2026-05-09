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
    field: str
    action: Literal["add", "remove", "set"]
    value: str
    tag_level: str = ""
    reason: str = ""


class ProfileUpdateDecision(BaseModel):
    updates: list[ProfileUpdate] = Field(default_factory=list)
    no_update_reason: str = ""
