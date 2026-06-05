import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, StreamingResponse

from rag_persona.config import Settings, get_settings
from rag_persona.graph import build_graph
from rag_persona.nodes.generator import stream_generator_node
from rag_persona.nodes.grader import grade_answer
from rag_persona.schemas import BookingRequest, ChatEvent, ChatRequest, PersonaState
from rag_persona.services.calcom import CalComClient
from rag_persona.services.embeddings import EmbeddingService
from rag_persona.services.groq_client import GroqClient
from rag_persona.services.qdrant_store import QdrantStore


def try_build_services(settings: Settings) -> dict[str, Any]:
    groq = GroqClient(settings) if settings.groq_api_key else None
    embeddings = EmbeddingService(settings) if settings.qdrant_url else None
    store = QdrantStore(settings) if settings.qdrant_url else None
    calcom = CalComClient(settings) if settings.calcom_api_key else None
    return {"groq": groq, "embeddings": embeddings, "store": store, "calcom": calcom}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    services = try_build_services(settings)
    app.state.settings = settings
    app.state.services = services
    app.state.graph = build_graph(settings=settings, **services)
    yield


app = FastAPI(
    title="RAG Persona Backend",
    version="0.1.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def sse(event: ChatEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "timestamp": datetime.now(UTC).isoformat()}


@app.get("/warm")
async def warm() -> dict[str, str]:
    services = app.state.services
    if services.get("embeddings") is not None:
        services["embeddings"].embed_one("warmup")
    return {"status": "warm"}


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        initial_state: PersonaState = {
            "raw_input": request.message,
            "session_id": request.session_id,
            "conversation_history": request.conversation_history,
        }
        state: PersonaState = await app.state.graph.ainvoke(initial_state)
        yield sse(ChatEvent(type="meta", data=json.dumps({"route": state.get("route", "rag")})))

        answer_parts: list[str] = []
        async for token in stream_generator_node(
            state=state,
            settings=app.state.settings,
            groq=app.state.services.get("groq"),
        ):
            answer_parts.append(token)
            yield sse(ChatEvent(type="token", data=token))

        answer = "".join(answer_parts)
        grounded = await grade_answer(
            state=state,
            answer=answer,
            settings=app.state.settings,
            groq=app.state.services.get("groq"),
        )

        yield sse(
            ChatEvent(
                type="done",
                data="",
                session_id=request.session_id or state.get("session_id"),
                grounded=grounded,
                available_slots=state.get("available_slots"),
            )
        )

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/book")
async def book_slot(request: BookingRequest) -> dict[str, object]:
    calcom = app.state.services.get("calcom")
    if calcom is None or not calcom.configured:
        return {"status": "error", "message": "Cal.com client is not configured"}
    try:
        result = await calcom.create_booking(request)
        return {"status": "success", "booking": result}
    except Exception as e:
        return {"status": "error", "message": str(e)}

