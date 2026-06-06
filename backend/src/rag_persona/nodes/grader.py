import logging

from rag_persona.config import Settings
from rag_persona.prompts import GRADER_PROMPT
from rag_persona.schemas import PersonaState, RetrievedChunk
from rag_persona.services.groq_client import GroqClient

logger = logging.getLogger(__name__)


def build_grader_context(chunks: list[RetrievedChunk]) -> str:
    """
    Format retrieved chunks for the grounding grader.
    Strips out all source metadata (file paths, repo names) to optimize input tokens,
    passing only the raw chunk index and text.
    """
    if not chunks:
        return "No retrieved context."

    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        blocks.append(f'<CHUNK index="{index}">\n{chunk.text}\n</CHUNK>')
    return "\n\n".join(blocks)


async def grade_answer(
    state: PersonaState,
    answer: str,
    settings: Settings,
    groq: GroqClient | None,
) -> bool:
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
        logger.exception("Hallucination grader failed")
        return True
