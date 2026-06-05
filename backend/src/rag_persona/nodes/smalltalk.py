import random

from rag_persona.schemas import PersonaState


def smalltalk_node(state: PersonaState) -> PersonaState:
    greetings = [
        "Hi! I'm Tejasv's AI persona. Ask me about his projects, experience, or skills — or book a call directly.",
        "Hello! Happy to tell you about Tejasv's background and work. What would you like to know?",
    ]
    answer = random.choice(greetings)
    return {**state, "answer": answer}
