from rag_persona.schemas import BookingStage, Intent, PersonaState, SafetyVerdict

# While the booking machine is mid-conversation the caller is answering *our* question,
# so the utterance must go back to the scheduling node whatever the guard made of it.
ACTIVE_BOOKING_STAGES = frozenset(
    {BookingStage.offering, BookingStage.awaiting_name, BookingStage.awaiting_email}
)


def route_from_state(state: PersonaState) -> str:
    guard = state["guard"]
    if guard.safety in (SafetyVerdict.malicious, SafetyVerdict.suspicious):
        return "refusal"

    # e.g. "john at gmail dot com" reads as a factual question to the guard; without this
    # it would be classified `rag` and sent to retrieval, silently dropping the booking.
    if state.get("booking_stage") in ACTIVE_BOOKING_STAGES:
        return "scheduling"

    if guard.intent == Intent.scheduling:
        return "scheduling"
    if guard.intent == Intent.small_talk:
        return "small_talk"
    if guard.intent == Intent.end_call:
        return "end_call"
    return "rag"


async def router_node(state: PersonaState) -> PersonaState:
    return {**state, "route": route_from_state(state)}
