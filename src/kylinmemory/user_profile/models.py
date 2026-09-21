"""Domain models; no Kylin or storage implementation details live here."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .schema import ALLOWED_PATHS, Lifecycle, STORED_PATHS


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


ProfileScalar = str | int | float | bool
ProfileValue = ProfileScalar | list[ProfileScalar]


class InteractionMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: Annotated[str, Field(min_length=1, max_length=50_000)]
    source: Literal["conversation", "l1", "l2"] = "conversation"
    source_ref: Annotated[str, Field(max_length=512)] = ""


class Evidence(BaseModel):
    quote: Annotated[str, Field(min_length=1, max_length=2_000)]
    observed_at: datetime = Field(default_factory=utc_now)


class ProfileEntry(BaseModel):
    value: ProfileValue
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    source: Literal["onboarding", "explicit", "inferred"]
    lifecycle: Lifecycle
    evidence: list[Evidence] = Field(default_factory=list, max_length=10)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    archived: bool = False


class OnboardingState(BaseModel):
    completed: bool = False
    next_question_index: Annotated[int, Field(ge=0, le=3)] = 0


class UserProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    user_id: Annotated[str, Field(min_length=1, max_length=256)]
    onboarding: OnboardingState = Field(default_factory=OnboardingState)
    entries: dict[str, ProfileEntry] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("entries")
    @classmethod
    def validate_entry_paths(cls, entries: dict[str, ProfileEntry]) -> dict[str, ProfileEntry]:
        unknown = set(entries) - STORED_PATHS
        if unknown:
            raise ValueError(f"unknown profile paths: {sorted(unknown)}")
        return entries


class ProfileUpdate(BaseModel):
    action: Literal["upsert", "delete"] = "upsert"
    path: str
    value: ProfileValue | None = None
    confidence: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    explicit: bool = False
    evidence_quote: Annotated[str, Field(max_length=2_000)] = ""

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        if path not in ALLOWED_PATHS:
            raise ValueError(f"unsupported profile path: {path}")
        return path


class ExtractionBatch(BaseModel):
    updates: list[ProfileUpdate] = Field(default_factory=list, max_length=50)


class ApplyResult(BaseModel):
    applied: list[str] = Field(default_factory=list)
    deleted: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    rejected: list[str] = Field(default_factory=list)


class OnboardingPrompt(BaseModel):
    completed: bool
    field: str | None = None
    question: str | None = None
