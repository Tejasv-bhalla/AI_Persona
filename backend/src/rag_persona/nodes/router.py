from rag_persona.schemas import Intent, PersonaState, SafetyVerdict


def route_from_state(state: PersonaState) -> str:
    guard = state["guard"]
    if guard.safety == SafetyVerdict.malicious:
        return "refusal"
        
    # Check if we are in voice mode and currently expecting an email response for scheduling
    mode = state.get("mode", "chat")
    if mode == "voice":
        history = state.get("conversation_history", [])
        if history:
            last_assistant_msg = ""
            for msg in reversed(history):
                if msg.get("role") == "assistant":
                    last_assistant_msg = msg.get("content", "")
                    break
            if last_assistant_msg and (
                "email address" in last_assistant_msg.lower() 
                or "what email" in last_assistant_msg.lower() 
                or "send the invitation to" in last_assistant_msg.lower() 
                or "send the invite to" in last_assistant_msg.lower()
                or "can i get your name" in last_assistant_msg.lower()
                or "does that work for you" in last_assistant_msg.lower()
                or "how about" in last_assistant_msg.lower()
            ):
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

