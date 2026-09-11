import asyncio
import re

from rag_persona.config import Settings
from rag_persona.ingestion.bm25 import encode_sparse_query
from rag_persona.schemas import PersonaState
from rag_persona.services.embeddings import EmbeddingService
from rag_persona.services.qdrant_store import QdrantStore
from rag_persona.voice.response_cache import get_cached_response

# Extra breadth granted to a retry after a failed grounding check.
RETRY_EXTRA_CANDIDATES = 4
RETRY_EXTRA_CHUNKS = 2


def repo_tokens(repo_name: str) -> set[str]:
    """Split a repo name into lowercase word tokens.

    Handles both delimiters and camelCase, so "Stock-Market-Prediction" yields
    {stock, market, prediction} and "TalentScoutBot" yields {talent, scout, bot}.
    Single characters are dropped as noise ("LegalX" -> {legal}).
    """
    spaced = re.sub(r"[-_.]+", " ", repo_name)
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", spaced)
    spaced = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", spaced)
    return {token for token in spaced.lower().split() if len(token) > 1}


def _distinctive_tokens(token_map: dict[str, set[str]]) -> set[str]:
    """Tokens that occur in exactly one repo name, and so identify it on their own.

    Derived from the corpus rather than hardcoded: "shramik" names one repo and is
    decisive, while "prediction" and "ai" name several and are not.
    """
    counts: dict[str, int] = {}
    for tokens in token_map.values():
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
    return {token for token, count in counts.items() if count == 1}


def extract_repo_filter(text: str, store: QdrantStore | None) -> str | None:
    """Resolve a repo name mentioned in free text, e.g. "the stock market project".

    Matches on the full name, on a decisive single token, or on two or more
    overlapping tokens. Ambiguous mentions ("prediction") return None so the
    search stays unfiltered rather than being narrowed to the wrong repo.
    """
    if store is None or not text:
        return None

    repo_names = store.repo_names()
    if not repo_names:
        return None

    lowered = text.lower()
    query_tokens = set(re.findall(r"[a-z0-9]+", lowered))
    token_map = {name: repo_tokens(name) for name in repo_names}
    distinctive = _distinctive_tokens(token_map)

    scored: list[tuple[tuple[int, int], str]] = []
    for name, tokens in token_map.items():
        # Whole-name mentions ("talentscoutbot", "audio-emotion-classification").
        aliases = {name.lower(), re.sub(r"[-_.]+", " ", name.lower())}
        matched_aliases = [alias for alias in aliases if alias in lowered]
        if matched_aliases:
            # Longest full-name match wins, so a short repo name cannot shadow a longer one.
            score = (3, len(max(matched_aliases, key=len)))
        else:
            overlap = tokens & query_tokens
            if not overlap:
                continue
            decisive = overlap & distinctive
            if not decisive and len(overlap) < 2:
                continue
            score = (2 if decisive else 1, len(overlap))
        scored.append((score, name))

    if not scored:
        return None
    scored.sort(reverse=True)
    # A tie means the text names two repos equally well ("the persona bot"). Prefer an
    # unfiltered search over guessing which one the caller meant.
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


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
    repo_filter = extract_repo_filter(raw_input, store) or extract_repo_filter(
        guard.keywords, store
    )
    if not repo_filter and state.get("conversation_history"):
        for turn in reversed(state["conversation_history"]):
            if turn.get("role") == "user":
                repo_filter = extract_repo_filter(turn.get("content", ""), store)
                if repo_filter:
                    break

    search_text = guard.keywords or raw_input
    # Set by the grader when it rejects an answer and sends it back round.
    is_retry = state.get("retry_count", 0) > 0

    # Offload CPU-bound ONNX embedding inference to a thread pool to avoid blocking the event loop
    query_vector = await asyncio.to_thread(embeddings.embed_one, search_text)

    # Cache lookup for voice mode. Skipped on a retry: the cached answer is the one
    # that just failed the grounding check.
    if mode == "voice" and guard.keywords and not is_retry:
        cached = get_cached_response(
            guard.keywords,
            vector=query_vector,
            ttl_seconds=settings.voice_cache_ttl_seconds,
        )
        if cached:
            new_state: PersonaState = {
                **state,
                "chunks": cached["chunks"],
                "query_vector": query_vector,
            }
            if cached.get("answer"):
                new_state["answer"] = cached["answer"]
            return new_state

    # Voice trades recall for latency. A retry widens the pool, since a narrow one
    # is the likeliest reason the first answer could not be grounded.
    limit = 5 if mode == "voice" else settings.max_retrieval_candidates
    if is_retry:
        limit += RETRY_EXTRA_CANDIDATES
    try:
        candidates = store.search(
            query_vector=query_vector,
            sparse_query=encode_sparse_query(search_text),
            source_filter=guard.source_filter,
            repo_filter=repo_filter,
            limit=limit,
        )
        # Classifier keywords can be poor or off-topic; fall back to the user's own words once
        if len(candidates) < 2 and raw_input and search_text != raw_input:
            fallback_vector = await asyncio.to_thread(embeddings.embed_one, raw_input)
            fallback = store.search(
                query_vector=fallback_vector,
                sparse_query=encode_sparse_query(raw_input),
                source_filter=guard.source_filter,
                repo_filter=repo_filter,
                limit=limit,
            )
            if len(fallback) > len(candidates):
                candidates = fallback
    except Exception:
        return {**state, "chunks": [], "query_vector": query_vector}

    # Return the Qdrant candidates directly: this preserves the server-side RRF
    # fusion ranks. There is no second-stage reranker (see README tradeoffs).
    max_chunks = settings.max_context_chunks + (RETRY_EXTRA_CHUNKS if is_retry else 0)
    chunks = candidates[:max_chunks]

    return {**state, "chunks": chunks, "query_vector": query_vector}
