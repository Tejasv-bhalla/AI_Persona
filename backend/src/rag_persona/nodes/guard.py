import re

from rag_persona.config import Settings
from rag_persona.prompts import GUARD_PROMPT, VOICE_GUARD_PROMPT
from rag_persona.schemas import GuardResult, Intent, PersonaState, SafetyVerdict, SourceType
from rag_persona.services.groq_client import GroqClient

# Word-boundary matched, not substring: "hi" must not fire inside "his"/"this",
# "book" inside "notebook", or "call" inside "recall".
_END_CALL_RE = re.compile(r"\b(goodbye|bye|talk later|have a good day)\b")
_SCHEDULING_RE = re.compile(r"\b(book|booking|schedule|calendar|meeting|call)\b")
_SMALL_TALK_RE = re.compile(r"\b(hi|hello|hey|thanks)\b")

# Vocabularies for the fast path below. Deliberately small: a turn only skips the
# classifier when EVERY word it contains is in one of these sets, so anything
# carrying real content falls through to the model.
_TOKEN_RE = re.compile(r"[a-z']+")
_GREETING_WORDS = frozenset({
    "hi", "hello", "hey", "hiya", "yo", "sup", "howdy", "greetings",
    "thanks", "thank", "you", "so", "much", "there",
    "good", "morning", "afternoon", "evening",
    "whats", "up", "how", "are", "doing",
})
_SIGN_OFF_WORDS = frozenset({
    "bye", "goodbye", "byebye", "cya", "later", "talk", "to", "soon",
    "have", "a", "good", "great", "day", "night",
    "thanks", "thank", "you", "all", "thats", "it", "im", "done", "cheers",
})


def fast_path_guard(message: str) -> GuardResult | None:
    """Classify greetings and sign-offs without calling the model.

    The guard LLM is a full round trip and measured ~37% of time-to-first-token.
    Greetings and sign-offs need no retrieval and no distilled keywords, so the
    model adds latency and nothing else. Returns None for anything ambiguous, which
    keeps real questions on the model path where keyword distillation and the
    source filter genuinely matter.
    """
    tokens = _TOKEN_RE.findall(message.lower())
    # A question mark means a real question however short, e.g. "you there?".
    if not tokens or len(tokens) > 6 or "?" in message:
        return None

    # Sign-off first: "thanks, that's all" is a farewell, while a bare "thanks" is
    # small talk. Only the sign-off vocabulary covers the closing words.
    if set(tokens) <= _SIGN_OFF_WORDS and not set(tokens) <= _GREETING_WORDS:
        return GuardResult(safety=SafetyVerdict.safe, intent=Intent.end_call, keywords="")
    if set(tokens) <= _GREETING_WORDS:
        return GuardResult(safety=SafetyVerdict.safe, intent=Intent.small_talk, keywords="")
    return None


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
                refusal_reason=(
                    "The request attempts to override or extract protected instructions."
                ),
            )

    if _END_CALL_RE.search(lowered):
        intent = Intent.end_call
    elif _SCHEDULING_RE.search(lowered):
        intent = Intent.scheduling
    elif _SMALL_TALK_RE.search(lowered):
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
    
    # 1. Run hardcoded keyword check first as a mandatory failsafe
    lowered = raw_input.lower()
    malicious_markers = [
        "ignore previous",
        "ignore all previous",
        "system prompt",
        "developer message",
        "jailbreak",
        "reveal secrets",
        "forget all instructions",
        "forget previous",
        "reveal your instructions",
        "you are no longer",
        "new instructions",
        "stop simulating",
    ]
    if mode == "chat" and any(marker in lowered for marker in malicious_markers):
        guard = GuardResult(
            safety=SafetyVerdict.malicious,
            intent=Intent.rag,
            keywords="",
            refusal_reason="The request attempts to override or extract protected instructions.",
        )
        return {**state, "guard": guard}

    # Deterministic short-circuit before the model call.
    fast = fast_path_guard(raw_input)
    if fast is not None:
        return {**state, "guard": fast}

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

