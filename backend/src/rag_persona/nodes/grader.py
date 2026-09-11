import logging

from rag_persona.config import Settings
from rag_persona.prompts import GRADER_PROMPT
from rag_persona.schemas import PersonaState, RetrievedChunk
from rag_persona.services.groq_client import GroqClient

logger = logging.getLogger(__name__)

# One re-grounding attempt. A second adds latency and rarely changes the verdict.
MAX_GROUNDING_RETRIES = 1


def build_grader_context(chunks: list[RetrievedChunk]) -> str:
    """Format chunks for the grader.

    Source metadata (repo, file path, title) is stripped: the grader only judges
    whether the answer's claims appear in the text, so the metadata is wasted tokens.
    """
    if not chunks:
        return "No retrieved context."
    return "\n\n".join(
        f'<CHUNK index="{index}">\n{chunk.text}\n</CHUNK>'
        for index, chunk in enumerate(chunks, start=1)
    )


async def grade_answer(
    state: PersonaState,
    answer: str,
    settings: Settings,
    groq: GroqClient | None,
) -> bool:
    """True when every substantive claim in `answer` is supported by the retrieved chunks."""
    if groq is None or state.get("route") != "rag":
        return True

    try:
        result = await groq.json_completion(
            model=settings.groq_grader_model,
            system=GRADER_PROMPT,
            user=(
                f"<CONTEXT>\n{build_grader_context(state.get('chunks', []))}\n</CONTEXT>\n"
                f"<ANSWER>\n{answer}\n</ANSWER>"
            ),
        )
        return bool(result.get("grounded", False))
    except Exception:
        # Fail open: a grader outage must not block an answer the user is waiting on.
        logger.exception("Hallucination grader failed")
        return True


async def grade_node(
    state: PersonaState,
    settings: Settings,
    groq: GroqClient | None,
) -> PersonaState:
    """Check the answer against its sources and decide whether to regenerate.

    Voice is exempt: the caller has already heard the answer by the time a verdict
    lands, so a correction pass would only add latency to the next turn.
    """
    if state.get("mode") == "voice" or state.get("route") != "rag" or not state.get("chunks"):
        return {**state, "grounded": True}

    grounded = await grade_answer(
        state=state,
        answer=state.get("answer", ""),
        settings=settings,
        groq=groq,
    )
    if grounded:
        return {**state, "grounded": True}

    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_GROUNDING_RETRIES:
        # Out of attempts: keep the answer, but tell the client it is unverified.
        return {**state, "grounded": False}

    # Clearing `answer` is what makes the generator produce a fresh one on the retry pass.
    logger.info("Answer failed grounding check; retrying retrieval (attempt %d)", retry_count + 1)
    return {**state, "grounded": False, "retry_count": retry_count + 1, "answer": ""}


def route_after_grade(state: PersonaState) -> str:
    """Retry only when the grader both failed the answer and cleared it for regeneration."""
    if state.get("grounded") is False and not state.get("answer"):
        return "retry"
    return "end"
