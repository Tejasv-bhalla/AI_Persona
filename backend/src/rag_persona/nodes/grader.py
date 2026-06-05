import logging

from rag_persona.config import Settings
from rag_persona.nodes.generator import build_context
from rag_persona.prompts import GRADER_PROMPT
from rag_persona.schemas import PersonaState
from rag_persona.services.groq_client import GroqClient

logger = logging.getLogger(__name__)


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
                f"<CONTEXT>\n{build_context(state.get('chunks', []))}\n</CONTEXT>\n"
                f"<ANSWER>\n{answer}\n</ANSWER>"
            ),
        )
        return bool(result.get("grounded", False))
    except Exception:
        logger.exception("Hallucination grader failed")
        return True
