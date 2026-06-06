import time
import math
from typing import Any

# Global dictionary to store cached search results
# Format:
# {
#     "normalized_keyword_string": {
#         "chunks": list of RetrievedChunk,
#         "answer": str | None,
#         "vector": list[float] | None,
#         "cached_at": float (timestamp),
#         "hit_count": int
#     }
# }
_CACHE: dict[str, dict[str, Any]] = {}


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Calculate the cosine similarity between two numeric vectors.
    """
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot_prod = sum(x * y for x, y in zip(v1, v2))
    norm_v1 = math.sqrt(sum(x * x for x in v1))
    norm_v2 = math.sqrt(sum(y * y for y in v2))
    if norm_v1 == 0.0 or norm_v2 == 0.0:
        return 0.0
    return dot_prod / (norm_v1 * norm_v2)


def get_cached_response(
    key: str,
    vector: list[float] | None = None,
    ttl_seconds: int = 3600,
    similarity_threshold: float = 0.95,
) -> dict[str, Any] | None:
    """
    Retrieve cached search results and answers using semantic matching if a vector is provided.
    Otherwise falls back to exact match on normalized key.
    Applies lazy TTL expiration.
    """
    normalized_key = key.strip().lower()
    current_time = time.time()

    # 1. Semantic Match
    if vector:
        best_entry = None
        best_key = None
        highest_sim = -1.0
        
        # Make a copy of keys to avoid modification during iteration
        for k, entry in list(_CACHE.items()):
            # Lazy expiration check
            cached_at = entry.get("cached_at", 0.0)
            if current_time - cached_at > ttl_seconds:
                _CACHE.pop(k, None)
                continue

            cached_vector = entry.get("vector")
            if cached_vector:
                sim = cosine_similarity(vector, cached_vector)
                if sim > highest_sim:
                    highest_sim = sim
                    best_entry = entry
                    best_key = k

        if highest_sim >= similarity_threshold and best_entry:
            best_entry["hit_count"] = best_entry.get("hit_count", 0) + 1
            return best_entry

    # 2. Exact Match Fallback
    if not normalized_key:
        return None

    entry = _CACHE.get(normalized_key)
    if not entry:
        return None

    # Check expiration
    cached_at = entry.get("cached_at", 0.0)
    if current_time - cached_at > ttl_seconds:
        _CACHE.pop(normalized_key, None)
        return None

    entry["hit_count"] = entry.get("hit_count", 0) + 1
    return entry


def set_cached_response(
    key: str,
    chunks: list[Any],
    answer: str | None = None,
    vector: list[float] | None = None,
) -> None:
    """
    Store search results in the cache for the given keywords key, along with optional
    generated answer and query vector.
    """
    normalized_key = key.strip().lower()
    if not normalized_key:
        return

    _CACHE[normalized_key] = {
        "chunks": chunks,
        "answer": answer,
        "vector": vector,
        "cached_at": time.time(),
        "hit_count": 0,
    }


def clear_cache() -> None:
    """
    Clear all entries from the cache.
    """
    _CACHE.clear()
