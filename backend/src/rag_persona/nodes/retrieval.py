import asyncio
from rag_persona.config import Settings
from rag_persona.ingestion.bm25 import encode_sparse_query
from rag_persona.schemas import PersonaState, RetrievedChunk
from rag_persona.services.embeddings import EmbeddingService
from rag_persona.services.qdrant_store import QdrantStore
from rag_persona.voice.response_cache import get_cached_response, set_cached_response


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
    mode = state.get("mode", "chat")

    raw_input = state.get("raw_input", "")
    repo_filter = extract_repo_filter(raw_input) or extract_repo_filter(guard.keywords)

    # Offload CPU-bound ONNX embedding inference to a thread pool to avoid blocking the event loop
    query_vector = await asyncio.to_thread(embeddings.embed_one, guard.keywords)
    state["query_vector"] = query_vector

    # Cache lookup for voice mode
    if mode == "voice" and guard.keywords:
        cached = get_cached_response(
            guard.keywords,
            vector=query_vector,
            ttl_seconds=settings.voice_cache_ttl_seconds,
        )
        if cached:
            new_state = {**state, "chunks": cached["chunks"]}
            if cached.get("answer"):
                new_state["answer"] = cached["answer"]
            return new_state
    
    # Restrict retrieval limit in voice mode
    limit = 5 if mode == "voice" else settings.max_retrieval_candidates
    try:
        candidates = store.search(
            query_vector=query_vector,
            sparse_query=encode_sparse_query(guard.keywords),
            source_filter=guard.source_filter,
            repo_filter=repo_filter,
            limit=limit,
        )
    except Exception:
        return {**state, "chunks": []}

    # Return the Qdrant retrieved candidates directly.
    # This preserves the sparse search RRF ranks and avoids redundant cosine similarity calculations.
    chunks = candidates[: settings.max_context_chunks]

    return {**state, "chunks": chunks}


