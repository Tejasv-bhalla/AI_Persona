from langgraph.config import get_stream_writer

from rag_persona.config import Settings
from rag_persona.prompts import GENERATOR_SYSTEM_PROMPT, VOICE_GENERATOR_SYSTEM_PROMPT
from rag_persona.schemas import PersonaState, RetrievedChunk
from rag_persona.services.groq_client import GroqClient

MAX_HISTORY_TURNS = 6


def build_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No retrieved context."

    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        source = " | ".join(
            value
            for value in [chunk.source_type.value, chunk.repo_name, chunk.file_path, chunk.title]
            if value
        )
        blocks.append(f'<CHUNK index="{index}" source="{source}">\n{chunk.text}\n</CHUNK>')
    return "\n\n".join(blocks)


def build_generation_prompt(state: PersonaState) -> str:
    guard = state["guard"]
    return f"""<SYSTEM-CONTEXT>
Persona: Tejasv Bhalla, IIT Roorkee. Answer as a grounded portfolio persona.
</SYSTEM-CONTEXT>

<RETRIEVED-CONTEXT>
{build_context(state.get("chunks", []))}
</RETRIEVED-CONTEXT>

<QUERY>
{guard.keywords}
</QUERY>
"""


def fallback_answer(state: PersonaState) -> str:
    """Canned reply for every route that does not call the generation model."""
    route = state.get("route")
    mode = state.get("mode", "chat")

    if route == "refusal":
        if mode == "voice":
            return (
                "I'm specifically here to help with questions about Tejasv's "
                "background and to help schedule time with him."
            )
        guard = state.get("guard")
        reason = (guard.refusal_reason if guard else None) or "I cannot help with that request."
        return f"I can’t help with that. {reason}"

    if route == "end_call":
        return (
            "Great, thanks for your time. Tejasv looks forward to connecting "
            "with you. Have a great day."
        )

    if route == "small_talk":
        if mode == "voice":
            return (
                "Hello! I'm Tejasv's AI representative. I can tell you about his "
                "background, experience, or help you schedule a call."
            )
        return (
            "Hey, I’m Tejasv’s RAG-grounded persona. Ask me about my projects, "
            "experience, or technical decisions."
        )

    if route == "scheduling":
        if mode == "voice":
            return (
                "I can help you schedule a call with Tejasv. "
                "Let me check the next available slots."
            )
        return (
            "I can help schedule a call, but the booking endpoint needs name, "
            "email, and a start time."
        )

    if mode == "voice":
        return (
            "I don't have that specific information, but Tejasv would be happy "
            "to discuss it directly."
        )
    return (
        "I don’t have enough retrieved knowledge to answer that from the indexed "
        "source base yet."
    )


async def generate_node(
    state: PersonaState,
    settings: Settings,
    groq: GroqClient | None,
) -> PersonaState:
    """Produce the answer, emitting it token-by-token on the graph's custom stream.

    Runs inside the graph so the grounding grader can send it round a second time.
    A re-run after a failed grounding check emits `correction` events instead of
    `token` events, so the client can replace the answer it already rendered.
    """
    writer = get_stream_writer()
    event_type = "correction" if state.get("retry_count", 0) else "token"

    # Already answered upstream: a canned route reply, or a voice cache hit.
    existing = state.get("answer")
    if existing:
        writer({"type": event_type, "text": existing})
        return {**state, "answer": existing}

    if groq is None or state.get("route") != "rag":
        answer = fallback_answer(state)
        writer({"type": event_type, "text": answer})
        return {**state, "answer": answer}

    mode = state.get("mode", "chat")
    history = state.get("conversation_history", [])[-MAX_HISTORY_TURNS:]

    parts: list[str] = []
    async for token in groq.stream_completion(
        model=settings.groq_voice_model if mode == "voice" else settings.groq_generation_model,
        system=VOICE_GENERATOR_SYSTEM_PROMPT if mode == "voice" else GENERATOR_SYSTEM_PROMPT,
        user=build_generation_prompt(state),
        history=history,
        max_tokens=(
            settings.max_output_tokens_voice if mode == "voice"
            else settings.max_output_tokens_chat
        ),
    ):
        parts.append(token)
        writer({"type": event_type, "text": token})

    return {**state, "answer": "".join(parts)}
