from enum import StrEnum
from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class SafetyVerdict(StrEnum):
    safe = "safe"
    suspicious = "suspicious"
    malicious = "malicious"


class Intent(StrEnum):
    rag = "rag"
    scheduling = "scheduling"
    small_talk = "small_talk"


class SourceType(StrEnum):
    resume = "resume"
    code = "code"
    readme = "readme"
    changelog = "changelog"
    adr = "adr"
    devlog = "devlog"
    contribution_scope = "contribution-scope"
    notebook_output = "notebook-output"
    unknown = "unknown"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=128)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


class ChatEvent(BaseModel):
    type: Literal["token", "done", "error", "meta"]
    data: str
    session_id: str | None = None
    grounded: bool | None = None
    available_slots: list[str] | None = None


class GuardResult(BaseModel):
    safety: SafetyVerdict
    intent: Intent
    keywords: str = Field(default="", max_length=800)
    source_filter: SourceType | None = None
    refusal_reason: str | None = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    source_type: SourceType = SourceType.unknown
    repo_name: str | None = None
    file_path: str | None = None
    title: str | None = None
    metadata: dict[str, str | int | float | bool | None | list[float]] = Field(default_factory=dict)


class BookingRequest(BaseModel):
    preferred_time: str
    attendee_name: str
    attendee_email: str
    notes: str | None = None


class PersonaState(TypedDict, total=False):
    raw_input: str
    session_id: str | None
    conversation_history: list[dict[str, str]]
    guard: GuardResult
    route: str
    chunks: list[RetrievedChunk]
    answer: str
    correction_pending: bool
    available_slots: list[str]
