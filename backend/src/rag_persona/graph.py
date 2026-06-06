from collections.abc import Awaitable, Callable
from functools import partial

from langgraph.graph import END, StateGraph

from rag_persona.config import Settings
from rag_persona.nodes.calcom import calcom_node
from rag_persona.nodes.guard import guard_node
from rag_persona.nodes.retrieval import retrieval_node
from rag_persona.nodes.router import route_from_state, router_node
from rag_persona.nodes.smalltalk import smalltalk_node
from rag_persona.schemas import PersonaState
from rag_persona.services.calcom import CalComClient
from rag_persona.services.embeddings import EmbeddingService
from rag_persona.services.groq_client import GroqClient
from rag_persona.services.qdrant_store import QdrantStore

Node = Callable[[PersonaState], Awaitable[PersonaState]]


def build_graph(
    settings: Settings,
    groq: GroqClient | None,
    embeddings: EmbeddingService | None,
    store: QdrantStore | None,
    calcom: CalComClient | None = None,
):
    graph = StateGraph(PersonaState)

    graph.add_node("guard", partial(guard_node, settings=settings, groq=groq))
    graph.add_node("router", router_node)
    graph.add_node(
        "retrieval",
        partial(retrieval_node, settings=settings, embeddings=embeddings, store=store),
    )
    graph.add_node("calcom", partial(calcom_node, settings=settings, calcom=calcom))
    graph.add_node("smalltalk", smalltalk_node)

    graph.set_entry_point("guard")
    graph.add_edge("guard", "router")
    graph.add_conditional_edges(
        "router",
        route_from_state,
        {
            "rag": "retrieval",
            "scheduling": "calcom",
            "small_talk": "smalltalk",
            "refusal": END,
            "end_call": END,
        },
    )
    graph.add_edge("retrieval", END)
    graph.add_edge("calcom", END)
    graph.add_edge("smalltalk", END)
    return graph.compile()

