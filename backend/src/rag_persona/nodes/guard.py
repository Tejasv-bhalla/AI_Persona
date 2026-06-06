from rag_persona.config import Settings
from rag_persona.prompts import GUARD_PROMPT, VOICE_GUARD_PROMPT
from rag_persona.schemas import GuardResult, Intent, PersonaState, SafetyVerdict, SourceType
from rag_persona.services.groq_client import GroqClient


def fallback_guard(message: str, mode: str = "chat") -> GuardResult:
    lowered = message.lower()
    
    if mode == "chat":
        malicious_markers = [
            "ignore previous",
            "system prompt",
            "developer message",
            "jailbreak",
            "reveal secrets",
        ]
        if any(marker in lowered for marker in malicious_markers):
            return GuardResult(
                safety=SafetyVerdict.malicious,
                intent=Intent.rag,
                keywords="",
                refusal_reason="The request attempts to override or extract protected instructions.",
            )

    if any(marker in lowered for marker in ["goodbye", "bye", "talk later", "have a good day"]):
        intent = Intent.end_call
    elif any(marker in lowered for marker in ["book", "schedule", "calendar", "meeting", "call"]):
        intent = Intent.scheduling
    elif any(marker in lowered for marker in ["hi", "hello", "hey", "thanks"]):
        intent = Intent.small_talk
    else:
        intent = Intent.rag

    return GuardResult(safety=SafetyVerdict.safe, intent=intent, keywords=message[:800])


async def guard_node(
    state: PersonaState,
    settings: Settings,
    groq: GroqClient | None,
) -> PersonaState:
    raw_input = state["raw_input"]
    mode = state.get("mode", "chat")
    
    if groq is None:
        guard = fallback_guard(raw_input, mode)
    else:
        if mode == "voice":
            result = await groq.json_completion(
                model=settings.groq_guard_model,
                system=VOICE_GUARD_PROMPT,
                user=f"<USER-INPUT>{raw_input}</USER-INPUT>",
            )
            guard = GuardResult(
                safety=SafetyVerdict.safe,
                intent=Intent(str(result.get("intent", Intent.rag.value))),
                keywords=str(result.get("keywords", ""))[:800],
                source_filter=(
                    SourceType(str(result["source_filter"]))
                    if result.get("source_filter") and result.get("source_filter") != "unknown"
                    else None
                ),
                refusal_reason=None,
            )
        else:
            result = await groq.json_completion(
                model=settings.groq_guard_model,
                system=GUARD_PROMPT,
                user=f"<UNTRUSTED-USER-INPUT>{raw_input}</UNTRUSTED-USER-INPUT>",
            )
            guard = GuardResult(
                safety=SafetyVerdict(str(result.get("safety", SafetyVerdict.safe.value))),
                intent=Intent(str(result.get("intent", Intent.rag.value))),
                keywords=str(result.get("keywords", ""))[:800],
                source_filter=(
                    SourceType(str(result["source_filter"]))
                    if result.get("source_filter") and result.get("source_filter") != "unknown"
                    else None
                ),
                refusal_reason=(
                    str(result["refusal_reason"]) if result.get("refusal_reason") else None
                ),
            )

    return {**state, "guard": guard}

