from rag_persona.config import Settings
from rag_persona.ingestion.bm25 import encode_sparse_query
from rag_persona.schemas import PersonaState, RetrievedChunk
from rag_persona.services.embeddings import EmbeddingService
from rag_persona.services.qdrant_store import QdrantStore
from rag_persona.services.reranker import rerank_by_cosine


def extract_repo_filter(text: str) -> str | None:
    lowered = text.lower()
    repos = {
        "audio-emotion-classification": ["audio-emotion-classification", "emotion classification", "audio emotion"],
        "Credit-Default-Prediction": ["credit-default-prediction", "credit default", "credit-default"],
        "JHealth": ["jhealth", "jilo health", "jilo-health"],
        "Stock-Market-Prediction": ["stock-market-prediction", "stock market", "stock-market"],
        "TalentScoutBot": ["talentscoutbot", "talentscout", "talent scout"],
        "Shramik.ai": ["shramik.ai", "shramik", "shramik-ai"],
    }
    for repo_name, aliases in repos.items():
        if any(alias in lowered for alias in aliases):
            return repo_name
    return None


async def retrieval_node(
    state: PersonaState,
    settings: Settings,
    embeddings: EmbeddingService | None,
    store: QdrantStore | None,
) -> PersonaState:
    if embeddings is None or store is None:
        return {**state, "chunks": []}

    guard = state["guard"]
    raw_input = state.get("raw_input", "")
    repo_filter = extract_repo_filter(raw_input) or extract_repo_filter(guard.keywords)

    query_vector = embeddings.embed_one(guard.keywords)
    try:
        candidates = store.search(
            query_vector=query_vector,
            sparse_query=encode_sparse_query(guard.keywords),
            source_filter=guard.source_filter,
            repo_filter=repo_filter,
            limit=settings.max_retrieval_candidates,
        )
    except Exception:
        return {**state, "chunks": []}

    candidate_vectors: dict[str, list[float]] = {}
    for candidate in candidates:
        vector = candidate.metadata.get("dense_vector")
        if isinstance(vector, list):
            candidate_vectors[candidate.chunk_id] = [float(item) for item in vector]

    chunks: list[RetrievedChunk]
    if candidate_vectors:
        chunks = rerank_by_cosine(
            query_vector=query_vector,
            chunks=candidates,
            candidate_vectors=candidate_vectors,
            limit=settings.max_context_chunks,
        )
    else:
        chunks = candidates[: settings.max_context_chunks]

    return {**state, "chunks": chunks}
