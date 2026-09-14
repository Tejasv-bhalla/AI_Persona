import hmac
import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse, StreamingResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from rag_persona.config import Settings, get_settings
from rag_persona.graph import build_graph
from rag_persona.schemas import BookingRequest, ChatEvent, ChatRequest, PersonaState
from rag_persona.services.calcom import CalComClient
from rag_persona.services.embeddings import EmbeddingService
from rag_persona.services.groq_client import GroqClient
from rag_persona.services.qdrant_store import QdrantStore
from rag_persona.voice.response_cache import set_cached_response
from rag_persona.voice.vapi_adapter import format_vapi_response_stream, parse_vapi_request

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Never surfaced with exception detail: clients get this, the traceback goes to the log.
GENERIC_ERROR_MESSAGE = "An unexpected error occurred. Please try again."


def client_ip(request: Request) -> str:
    """Rate-limit key. Render sits behind a proxy, so prefer the leftmost X-Forwarded-For hop."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        leftmost = forwarded.split(",", 1)[0].strip()
        if leftmost:
            return leftmost
    return get_remote_address(request)


limiter = Limiter(key_func=client_ip)


def try_build_services(settings: Settings) -> dict[str, Any]:
    groq = GroqClient(settings) if settings.groq_api_key else None
    embeddings = EmbeddingService(settings) if settings.qdrant_url else None
    store = QdrantStore(settings) if settings.qdrant_url else None
    calcom = CalComClient(settings) if settings.calcom_api_key else None
    return {"groq": groq, "embeddings": embeddings, "store": store, "calcom": calcom}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    services = try_build_services(settings)
    app.state.settings = settings
    app.state.services = services
    app.state.graph = build_graph(settings=settings, **services)

    if not settings.vapi_webhook_secret:
        logger.warning(
            "vapi_webhook_secret is not configured: /voice endpoints are unauthenticated."
        )
    if not settings.eval_api_key:
        logger.warning("eval_api_key is not configured: /chat/eval is disabled (404).")

    yield

    calcom: CalComClient | None = services.get("calcom")
    if calcom is not None:
        await calcom.aclose()


settings = get_settings()

app = FastAPI(
    title="RAG Persona Backend",
    version="0.1.0",
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)

app.state.limiter = limiter
# slowapi's handler is typed against its own concrete exception rather than Exception,
# which is narrower than Starlette's handler protocol allows.
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def sse(event: ChatEvent) -> str:
    return f"data: {event.model_dump_json()}\n\n"


def turn_state(**overrides: Any) -> PersonaState:
    """Build the graph input for one turn.

    The checkpointer keeps every key that is not overwritten, which is what carries
    booking progress across turns of a call. Per-turn keys must therefore be reset
    explicitly, or a stale answer from the previous turn leaks into this one.
    """
    state: PersonaState = {
        "answer": "",
        "chunks": [],
        "route": "",
        "grounded": True,
        "retry_count": 0,
        "available_slots": [],
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


def history_dicts(payload: ChatRequest) -> list[dict[str, str]]:
    """Validated turns back to plain dicts, the shape the nodes and Groq SDK expect."""
    return [turn.model_dump() for turn in payload.conversation_history]


def graph_config(thread_id: str | None) -> dict[str, Any]:
    """Scope checkpointed state to one conversation; anonymous turns get their own."""
    return {"configurable": {"thread_id": thread_id or f"anon-{uuid4().hex}"}}


def verify_vapi_secret(provided: str | None) -> None:
    """Require the shared Vapi secret when one is configured; allow through otherwise (dev)."""
    app_settings: Settings | None = getattr(app.state, "settings", None)
    expected = app_settings.vapi_webhook_secret if app_settings else ""
    if not expected:
        return
    if provided is None or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid x-vapi-secret header",
        )


def verify_eval_key(provided: str | None) -> None:
    """Eval-only endpoint: hidden entirely unless a key is configured."""
    app_settings: Settings | None = getattr(app.state, "settings", None)
    expected = app_settings.eval_api_key if app_settings else ""
    if not expected:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    if provided is None or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid x-eval-key header",
        )


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
@limiter.limit(settings.rate_limit_chat)
async def chat(request: Request, payload: ChatRequest) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        final: PersonaState = {}
        route_sent = False
        try:
            stream = app.state.graph.astream(
                turn_state(
                    raw_input=payload.message,
                    session_id=payload.session_id,
                    conversation_history=history_dicts(payload),
                ),
                config=graph_config(payload.session_id),
                stream_mode=["custom", "values"],
            )
            async for mode, chunk in stream:
                if mode == "values":
                    final = chunk
                    if not route_sent and chunk.get("route"):
                        route_sent = True
                        yield sse(
                            ChatEvent(type="meta", data=json.dumps({"route": chunk["route"]}))
                        )
                    continue
                # A `correction` chunk means the grounding check failed and a fresh
                # answer is now streaming; the client replaces what it already rendered.
                yield sse(ChatEvent(type=chunk["type"], data=chunk["text"]))

            yield sse(
                ChatEvent(
                    type="done",
                    data="",
                    session_id=payload.session_id or final.get("session_id"),
                    grounded=final.get("grounded", True),
                    available_slots=final.get("available_slots"),
                )
            )
        except Exception as e:
            logger.exception("Unhandled error while streaming /chat response")
            error_msg = GENERIC_ERROR_MESSAGE
            if "rate_limit" in str(e).lower() or "429" in str(e):
                error_msg = (
                    "The Groq API free-tier limit was reached. "
                    "Please try again in a few minutes."
                )
            yield sse(ChatEvent(type="error", data=error_msg))

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/chat/eval")
@limiter.limit(settings.rate_limit_chat)
async def chat_eval(
    request: Request,
    payload: ChatRequest,
    x_eval_key: str | None = Header(default=None, alias="x-eval-key"),
) -> dict[str, Any]:
    verify_eval_key(x_eval_key)
    try:
        final: PersonaState = {}
        stream = app.state.graph.astream(
            turn_state(
                raw_input=payload.message,
                session_id=payload.session_id,
                conversation_history=history_dicts(payload),
            ),
            config=graph_config(payload.session_id),
            stream_mode="values",
        )
        async for chunk in stream:
            final = chunk

        return {
            "response": final.get("answer", ""),
            "retrieved_contexts": [chunk.text for chunk in final.get("chunks", [])],
            "grounded": final.get("grounded", True),
        }
    except Exception as e:
        logger.exception("Unhandled error in /chat/eval")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=GENERIC_ERROR_MESSAGE,
        ) from e


@app.post("/book")
@limiter.limit(settings.rate_limit_book)
async def book_slot(request: Request, payload: BookingRequest) -> dict[str, object]:
    calcom = app.state.services.get("calcom")
    if calcom is None or not calcom.configured:
        return {"status": "error", "message": "Cal.com client is not configured"}
    try:
        result = await calcom.create_booking(payload)
        return {"status": "success", "booking": result}
    except Exception:
        logger.exception("Cal.com booking failed")
        return {"status": "error", "message": GENERIC_ERROR_MESSAGE}


@app.post("/voice")
@app.post("/voice/chat/completions")
@limiter.limit(settings.rate_limit_voice)
async def voice_endpoint(
    request: Request,
    x_vapi_secret: str | None = Header(default=None, alias="x-vapi-secret"),
) -> StreamingResponse:
    verify_vapi_secret(x_vapi_secret)

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from e

    raw_input, history = parse_vapi_request(payload)

    message_obj = payload.get("message", {})
    call_obj = message_obj.get("call", {}) if isinstance(message_obj, dict) else {}
    if not call_obj and isinstance(payload, dict):
        call_obj = payload.get("call", {})

    call_id = ""
    customer_number = ""
    if isinstance(call_obj, dict):
        call_id = str(call_obj.get("id") or "")
        if isinstance(call_obj.get("customer"), dict):
            customer_number = call_obj["customer"].get("number", "")

    app_settings: Settings = app.state.settings

    async def token_stream() -> AsyncIterator[str]:
        final: PersonaState = {}
        try:
            # Keying the thread on the Vapi call id is what lets the booking flow
            # remember which slot was offered and whose name it already collected.
            stream = app.state.graph.astream(
                turn_state(
                    raw_input=raw_input,
                    conversation_history=history,
                    mode="voice",
                    customer_number=customer_number,
                ),
                config=graph_config(call_id or customer_number or None),
                stream_mode=["custom", "values"],
            )
            async for mode, chunk in stream:
                if mode == "values":
                    final = chunk
                else:
                    yield chunk["text"]
        except Exception as e:
            logger.exception("Error occurred during voice session streaming")
            error_msg = (
                "I'm having a bit of trouble connecting to my brain right now, "
                "but please ask again in a moment."
            )
            if "rate_limit" in str(e).lower() or "429" in str(e):
                error_msg = (
                    "I am experiencing a high volume of requests right now. "
                    "Could you please repeat that in a few seconds?"
                )
            yield error_msg
            return

        guard = final.get("guard")
        # Re-writing an entry that was itself a cache hit is harmless: it refreshes
        # the TTL and its LRU recency, which is what we want for a popular question.
        if final.get("route") == "rag" and guard and guard.keywords and final.get("answer"):
            set_cached_response(
                key=guard.keywords,
                chunks=final.get("chunks", []),
                answer=final["answer"],
                vector=final.get("query_vector"),
                max_entries=app_settings.voice_cache_max_entries,
            )

    return StreamingResponse(
        format_vapi_response_stream(token_stream()),
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
    verify_vapi_secret(x_vapi_secret)

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid JSON payload") from e

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
