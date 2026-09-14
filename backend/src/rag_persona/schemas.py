from enum import StrEnum
from typing import Literal, TypedDict

from pydantic import BaseModel, EmailStr, Field


class SafetyVerdict(StrEnum):
    safe = "safe"
    suspicious = "suspicious"
    malicious = "malicious"


class Intent(StrEnum):
    rag = "rag"
    scheduling = "scheduling"
    small_talk = "small_talk"
    end_call = "end_call"


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


class BookingStage(StrEnum):
    """Explicit position in the voice booking conversation.

    Replaces inferring state by string-matching the assistant's previous turn.
    """

    idle = "idle"
    offering = "offering"
    awaiting_name = "awaiting_name"
    awaiting_email = "awaiting_email"
    confirmed = "confirmed"


class Turn(BaseModel):
    """One prior message, as replayed by the client.

    Validated rather than trusted: the history is spliced straight into the model
    request, so an unconstrained `role` would let a caller inject arbitrary system
    or tool turns into the conversation.
    """

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=128)
    conversation_history: list[Turn] = Field(default_factory=list, max_length=20)


class ChatEvent(BaseModel):
    type: Literal["token", "done", "error", "meta", "correction"]
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
    attendee_name: str = Field(min_length=1, max_length=120)
    attendee_email: EmailStr
    notes: str | None = Field(default=None, max_length=2000)


class PersonaState(TypedDict, total=False):
    # Per-turn input
    raw_input: str
    session_id: str | None
    conversation_history: list[dict[str, str]]
    mode: Literal["chat", "voice"]
    customer_number: str

    # Pipeline output
    guard: GuardResult
    route: str
    chunks: list[RetrievedChunk]
    query_vector: list[float]
    answer: str
    grounded: bool
    retry_count: int

    # Voice booking flow (explicit state machine; see nodes/calcom.py)
    booking_stage: BookingStage
    booking_slots: list[str]
    booking_slot_index: int
    booking_name: str
    booking_email: str

    # Chat-mode booking UI
    available_slots: list[str]
