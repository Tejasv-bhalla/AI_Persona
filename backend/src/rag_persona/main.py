import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any

import logging
from fastapi import FastAPI, Request, Header, HTTPException, status
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
from rag_persona.voice.vapi_adapter import parse_vapi_request, format_vapi_response_stream

logger = logging.getLogger(__name__)



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
async def health() -> dict[str, Any]:
    settings = getattr(app.state, "settings", None)
    vapi_id = settings.vapi_assistant_id if settings else None
    return {
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "voice": "configured" if vapi_id else "not_configured",
        "vapi_assistant_id": vapi_id or None,
    }


@app.get("/warm")
async def warm() -> dict[str, str]:
    services = app.state.services
    if services.get("embeddings") is not None:
        services["embeddings"].embed_one("warmup")
    return {"status": "warm"}


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        try:
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
        except Exception as e:
            error_msg = "The Groq API free-tier limit was reached. Please try again in a few minutes."
            if "rate_limit" not in str(e).lower() and "429" not in str(e):
                error_msg = f"An unexpected error occurred: {str(e)}"
            yield sse(ChatEvent(type="error", data=error_msg))

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


@app.post("/voice")
@app.post("/voice/chat/completions")
async def voice_endpoint(request: Request) -> StreamingResponse:
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    raw_input, history = parse_vapi_request(payload)

    initial_state: PersonaState = {
        "raw_input": raw_input,
        "conversation_history": history,
        "mode": "voice",
    }

    state: PersonaState = await app.state.graph.ainvoke(initial_state)

    token_stream = stream_generator_node(
        state=state,
        settings=app.state.settings,
        groq=app.state.services.get("groq"),
    )

    async def cache_accumulator() -> AsyncIterator[str]:
        full_tokens = []
        async for token in token_stream:
            full_tokens.append(token)
            yield token

        # After streaming completes, cache the full response if it is a new generation
        if "answer" not in state and state.get("mode") == "voice":
            guard = state.get("guard")
            keywords = guard.keywords if guard else None
            chunks = state.get("chunks", [])
            query_vector = state.get("query_vector")
            full_answer = "".join(full_tokens)
            if keywords and chunks and full_answer:
                from rag_persona.voice.response_cache import set_cached_response
                set_cached_response(
                    key=keywords,
                    chunks=chunks,
                    answer=full_answer,
                    vector=query_vector,
                )

    vapi_stream = format_vapi_response_stream(cache_accumulator())

    return StreamingResponse(
        vapi_stream,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/vapi-webhook")
async def vapi_webhook(
    request: Request,
    x_vapi_secret: str | None = Header(default=None, alias="x-vapi-secret"),
) -> dict[str, str]:
    settings = getattr(app.state, "settings", None)
    expected_secret = settings.vapi_webhook_secret if settings else ""

    if expected_secret and x_vapi_secret != expected_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid x-vapi-secret header",
        )

    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON payload")

    message = payload.get("message", {})
    event_type = message.get("type")
    call_id = message.get("call", {}).get("id")

    if event_type == "call-started":
        logger.info(f"Vapi Call Started: {call_id} at {message.get('timestamp')}")
    elif event_type == "call-ended":
        logger.info(
            f"Vapi Call Ended: {call_id}. Duration: {message.get('duration')}s. "
            f"End reason: {message.get('endedReason')}"
        )
    elif event_type == "transcript":
        logger.info(f"Vapi Call {call_id} Transcript: {message.get('transcript')}")
    else:
        logger.info(f"Vapi webhook received event: {event_type} for call: {call_id}")

    return {"status": "ok"}

