from functools import partial

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from rag_persona.config import Settings
from rag_persona.nodes.calcom import calcom_node
from rag_persona.nodes.generator import generate_node
from rag_persona.nodes.grader import grade_node, route_after_grade
from rag_persona.nodes.guard import guard_node
from rag_persona.nodes.retrieval import retrieval_node
from rag_persona.nodes.router import route_from_state, router_node
from rag_persona.nodes.smalltalk import smalltalk_node
from rag_persona.schemas import PersonaState
from rag_persona.services.calcom import CalComClient
from rag_persona.services.embeddings import EmbeddingService
from rag_persona.services.groq_client import GroqClient
from rag_persona.services.qdrant_store import QdrantStore

# Types the checkpointer is permitted to revive. Anything not listed here is refused,
# which is what keeps a compromised checkpoint store from instantiating arbitrary classes.
CHECKPOINT_TYPES = [
    ("rag_persona.schemas", name)
    for name in (
        "BookingStage",
        "GuardResult",
        "Intent",
        "RetrievedChunk",
        "SafetyVerdict",
        "SourceType",
    )
]


def build_graph(
    settings: Settings,
    groq: GroqClient | None,
    embeddings: EmbeddingService | None,
    store: QdrantStore | None,
    calcom: CalComClient | None = None,
) -> CompiledStateGraph[PersonaState, None, PersonaState, PersonaState]:
    """Compile the persona pipeline.

        guard -> router -> {retrieval | calcom | smalltalk | (canned reply)}
                        -> generate -> grade -> retrieval (once, if ungrounded)
                                             -> END

    Generation lives inside the graph rather than in the API layer so the grounding
    grader can send a failed answer back through retrieval. `generate` streams its
    tokens out over the graph's custom stream channel, so routing it through the
    graph costs nothing in time-to-first-token.
    """
    graph = StateGraph(PersonaState)

    graph.add_node("guard", partial(guard_node, settings=settings, groq=groq))
    graph.add_node("router", router_node)
    graph.add_node(
        "retrieval",
        partial(retrieval_node, settings=settings, embeddings=embeddings, store=store),
    )
    graph.add_node("calcom", partial(calcom_node, settings=settings, calcom=calcom))
    graph.add_node("smalltalk", smalltalk_node)
    graph.add_node("generate", partial(generate_node, settings=settings, groq=groq))
    graph.add_node("grade", partial(grade_node, settings=settings, groq=groq))

    graph.set_entry_point("guard")
    graph.add_edge("guard", "router")
    graph.add_conditional_edges(
        "router",
        route_from_state,
        {
            "rag": "retrieval",
            "scheduling": "calcom",
            "small_talk": "smalltalk",
            # Refusals and sign-offs still go through `generate`, which emits the
            # canned reply on the same stream channel as a model-generated one.
            "refusal": "generate",
            "end_call": "generate",
        },
    )
    graph.add_edge("retrieval", "generate")
    graph.add_edge("calcom", "generate")
    graph.add_edge("smalltalk", "generate")
    graph.add_edge("generate", "grade")
    graph.add_conditional_edges("grade", route_after_grade, {"retry": "retrieval", "end": END})

    # Carries booking state across turns of a call, keyed by thread_id. In-process
    # only: it is lost on restart and not shared between instances. Fine on a single
    # free-tier container; a multi-instance deploy needs a shared checkpointer.
    #
    # The serializer is given an explicit allowlist rather than the permissive
    # default, so only this project's own state types can be revived from a
    # checkpoint. LangGraph will start refusing unregistered types in a later release.
    checkpointer = MemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES))
    return graph.compile(checkpointer=checkpointer)
