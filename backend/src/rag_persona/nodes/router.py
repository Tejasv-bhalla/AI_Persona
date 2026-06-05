from rag_persona.schemas import Intent, PersonaState, SafetyVerdict


def route_from_state(state: PersonaState) -> str:
    guard = state["guard"]
    if guard.safety == SafetyVerdict.malicious:
        return "refusal"
    if guard.intent == Intent.scheduling:
        return "scheduling"
    if guard.intent == Intent.small_talk:
        return "small_talk"
    return "rag"


async def router_node(state: PersonaState) -> PersonaState:
    return {**state, "route": route_from_state(state)}

