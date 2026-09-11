"""Shared fakes and whole-graph drivers.

Nothing in this suite touches the network. Groq, Qdrant, the embedding model and
Cal.com are all replaced by hand-written fakes that record what they were asked
for, so the assertions can be about behaviour ("Cal.com was fetched once") rather
than about mock call signatures.

pytest-asyncio is not installed in this environment, so async code is driven
explicitly with ``asyncio.run`` from ordinary sync tests. That keeps every test
free of event-loop configuration and works whether or not the plugin shows up
later.
"""

import asyncio
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from typing import Any

import pytest

from rag_persona.config import Settings
from rag_persona.graph import build_graph
from rag_persona.schemas import PersonaState, RetrievedChunk, SourceType

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeEmbeddings:
    """A constant vector: retrieval ranking is the store's job, not the test's."""

    def embed_one(self, text: str) -> list[float]:
        return [0.1] * 384


class FakeStore:
    """Returns exactly `limit` chunks, so the caller's requested breadth is observable."""

    def __init__(self) -> None:
        self.search_limits: list[int] = []

    def repo_names(self) -> list[str]:
        return ["AI_Persona"]

    def search(self, **kwargs: Any) -> list[RetrievedChunk]:
        limit = int(kwargs["limit"])
        self.search_limits.append(limit)
        return [
            RetrievedChunk(
                chunk_id=f"c{index}",
                text=f"Evidence chunk {index}.",
                score=0.9,
                source_type=SourceType.resume,
            )
            for index in range(limit)
        ]


class GradingGroq:
    """Streams one fixed answer per generation; grader verdicts are scripted up front."""

    def __init__(self, grades: Iterable[bool]) -> None:
        self.grades = list(grades)
        self.generation_calls = 0
        self.chunks_in_prompt: list[int] = []
        self.max_tokens_seen: list[int | None] = []

    async def stream_completion(
        self, model: str, system: str, user: str, history: Any = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        self.generation_calls += 1
        self.max_tokens_seen.append(max_tokens)
        self.chunks_in_prompt.append(user.count("<CHUNK"))
        for token in ("He ", "studied ", "at ", "IIT ", "Roorkee."):
            yield token

    async def json_completion(
        self, model: str, system: str, user: str, history: Any = None
    ) -> dict[str, Any]:
        if "ANSWER" in user:
            return {"grounded": self.grades.pop(0) if self.grades else True}
        return {"safety": "safe", "intent": "rag", "keywords": "education degree"}


class SchedulingGroq:
    """Only the opening turn reads as scheduling to the guard.

    Every later booking turn ("yes that works", "john dot carter at gmail dot com")
    classifies as `rag`, so the router has to keep the turn in the booking flow from
    `booking_stage` alone.
    """

    async def stream_completion(
        self, model: str, system: str, user: str, history: Any = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        yield "unused"

    async def json_completion(
        self, model: str, system: str, user: str, history: Any = None
    ) -> dict[str, Any]:
        intent = "scheduling" if "book a meeting" in user.lower() else "rag"
        return {"safety": "safe", "intent": intent, "keywords": "x"}


# Fixed timestamps, never `now()`: Tue 09:00, Tue 10:30, Wed 14:00 UTC.
OFFERED_SLOTS = [
    "2026-09-08T09:00:00Z",
    "2026-09-08T10:30:00Z",
    "2026-09-09T14:00:00Z",
]


class FakeCalcom:
    configured = True

    def __init__(self) -> None:
        self.fetches = 0
        self.bookings: list[Any] = []

    async def get_available_slots(self) -> dict[str, Any]:
        self.fetches += 1
        # Availability deliberately shifts after the first fetch. If the node refetched
        # per turn, every slot index would shift by one and the caller would be booked
        # into a slot they never heard offered.
        times = OFFERED_SLOTS if self.fetches == 1 else ["2026-09-30T23:00:00Z", *OFFERED_SLOTS]
        return {"data": {"slots": {"2026-09-08": [{"time": time} for time in times]}}}

    async def create_booking(self, request: Any) -> dict[str, Any]:
        self.bookings.append(request)
        return {"id": "bk_1"}


# ---------------------------------------------------------------------------
# Graph drivers
# ---------------------------------------------------------------------------


@dataclass
class Turn:
    text: str
    route: str
    stage: Any
    slot_index: int
    answer: str


@dataclass
class GraphRun:
    events: list[dict[str, Any]] = field(default_factory=list)
    final: dict[str, Any] = field(default_factory=dict)
    turns: list[Turn] = field(default_factory=list)
    persisted: dict[str, Any] = field(default_factory=dict)

    def event_types(self) -> list[str]:
        return [event["type"] for event in self.events]


def _base_state(text: str, mode: str) -> PersonaState:
    return {
        "raw_input": text,
        "mode": mode,
        "conversation_history": [],
        "answer": "",
        "chunks": [],
        "route": "",
        "grounded": True,
        "retry_count": 0,
        "available_slots": [],
    }


async def _stream_turn(graph: Any, state: PersonaState, thread_id: str) -> GraphRun:
    run = GraphRun()
    config = {"configurable": {"thread_id": thread_id}}
    async for stream_mode, chunk in graph.astream(
        state, config=config, stream_mode=["custom", "values"]
    ):
        if stream_mode == "values":
            run.final = chunk
        else:
            run.events.append(chunk)
    return run


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# `_env_file=None` plus explicit values keeps the developer's real .env out of the
# tests: settings passed here outrank both the file and the process environment.
@pytest.fixture(scope="module")
def rag_settings() -> Settings:
    return Settings(
        _env_file=None,
        groq_api_key="x",
        qdrant_url="http://localhost:6333",
        max_retrieval_candidates=8,
        max_context_chunks=5,
    )


@pytest.fixture(scope="module")
def booking_settings() -> Settings:
    return Settings(
        _env_file=None,
        groq_api_key="x",
        calcom_api_key="k",
        calcom_event_type_id="1",
        calcom_username="tejasv",
    )


@pytest.fixture(scope="module")
def ungrounded_then_grounded(rag_settings: Settings) -> tuple[GraphRun, GradingGroq]:
    """One chat turn whose first answer fails the grounding check and is retried."""
    groq = GradingGroq(grades=[False, True])
    graph = build_graph(
        settings=rag_settings,
        groq=groq,
        embeddings=FakeEmbeddings(),
        store=FakeStore(),
        calcom=None,
    )
    run = asyncio.run(_stream_turn(graph, _base_state("Where did he study?", "chat"), "retry-1"))
    return run, groq


@pytest.fixture(scope="module")
def grounded_first_time(rag_settings: Settings) -> tuple[GraphRun, GradingGroq]:
    """The same turn, with the grader accepting the first answer."""
    groq = GradingGroq(grades=[True])
    graph = build_graph(
        settings=rag_settings,
        groq=groq,
        embeddings=FakeEmbeddings(),
        store=FakeStore(),
        calcom=None,
    )
    run = asyncio.run(_stream_turn(graph, _base_state("Where did he study?", "chat"), "happy-1"))
    return run, groq


BOOKING_TURNS = [
    "I'd like to book a meeting",
    "no that doesn't work",
    "yes that works",
    "my name is John Carter",
    "not an email",
    "john dot carter at gmail dot com",
]


async def _drive_booking_call(graph: Any, thread_id: str) -> GraphRun:
    run = GraphRun()
    config = {"configurable": {"thread_id": thread_id}}
    for text in BOOKING_TURNS:
        # A fresh input state each time, exactly as the API layer builds it per webhook
        # request: everything the booking flow remembers has to come from the checkpointer.
        turn = await _stream_turn(graph, _base_state(text, "voice"), thread_id)
        run.turns.append(
            Turn(
                text=text,
                route=str(turn.final.get("route", "")),
                stage=turn.final.get("booking_stage"),
                slot_index=int(turn.final.get("booking_slot_index", -1)),
                answer=str(turn.final.get("answer", "")),
            )
        )
    run.persisted = dict((await graph.aget_state(config)).values)
    return run


@pytest.fixture(scope="module")
def booking_call(booking_settings: Settings) -> tuple[GraphRun, FakeCalcom]:
    """A six-turn voice booking call, each turn a separate `astream` invocation."""
    calcom = FakeCalcom()
    graph = build_graph(
        settings=booking_settings,
        groq=SchedulingGroq(),
        embeddings=None,
        store=None,
        calcom=calcom,
    )
    run = asyncio.run(_drive_booking_call(graph, "call_abc123"))
    return run, calcom
