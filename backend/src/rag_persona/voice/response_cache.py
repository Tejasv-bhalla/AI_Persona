import time
from collections import OrderedDict
from typing import Any

import numpy as np

# In-process cache only: it lives in this worker's memory, so it is lost on restart and is
# not shared across instances or workers. That is deliberate -- an external cache is not
# worth the dependency or the round-trip latency at this traffic level.
#
# Insertion order is LRU order (least recently used first). Format:
# {
#     "normalized_keyword_string": {
#         "chunks": list of RetrievedChunk,
#         "answer": str | None,
#         "vector": list[float] | None,
#         "unit_vector": np.ndarray | None,  # pre-normalized, for vectorized scoring
#         "cached_at": float (timestamp),
#         "hit_count": int
#     }
# }
_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()

# Mirrors settings.voice_cache_max_entries; the caller passes the configured value.
DEFAULT_MAX_ENTRIES = 256


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Calculate the cosine similarity between two numeric vectors.
    """
    if len(v1) != len(v2) or not v1:
        return 0.0
    a = np.asarray(v1, dtype=np.float32)
    b = np.asarray(v2, dtype=np.float32)
    denominator = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def _to_unit_vector(vector: list[float] | None) -> np.ndarray | None:
    """Normalize once at write time so lookups are a plain dot product."""
    if not vector:
        return None
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        return None
    return arr / norm


def _hit(key: str) -> dict[str, Any]:
    """Record a hit and mark the entry most recently used."""
    entry = _CACHE[key]
    entry["hit_count"] = entry.get("hit_count", 0) + 1
    _CACHE.move_to_end(key)
    return entry


def _expire(current_time: float, ttl_seconds: int) -> None:
    for key, entry in list(_CACHE.items()):
        if current_time - entry.get("cached_at", 0.0) > ttl_seconds:
            del _CACHE[key]


def get_cached_response(
    key: str,
    vector: list[float] | None = None,
    ttl_seconds: int = 3600,
    similarity_threshold: float = 0.88,
) -> dict[str, Any] | None:
    """
    Retrieve cached search results and answers using semantic matching if a vector is provided.
    Otherwise falls back to exact match on normalized key.
    Applies lazy TTL expiration.
    """
    normalized_key = key.strip().lower()
    _expire(time.time(), ttl_seconds)

    # 1. Semantic Match
    query = _to_unit_vector(vector)
    if query is not None:
        candidate_keys = [
            k
            for k, entry in _CACHE.items()
            if entry.get("unit_vector") is not None
            and entry["unit_vector"].shape == query.shape
        ]
        if candidate_keys:
            # One matrix-vector product scores every entry at once.
            matrix = np.stack([_CACHE[k]["unit_vector"] for k in candidate_keys])
            similarities = matrix @ query
            best = int(np.argmax(similarities))
            if float(similarities[best]) >= similarity_threshold:
                return _hit(candidate_keys[best])

    # 2. Exact Match Fallback
    if not normalized_key or normalized_key not in _CACHE:
        return None

    return _hit(normalized_key)


def set_cached_response(
    key: str,
    chunks: list[Any],
    answer: str | None = None,
    vector: list[float] | None = None,
    max_entries: int = DEFAULT_MAX_ENTRIES,
) -> None:
    """
    Store search results in the cache for the given keywords key, along with optional
    generated answer and query vector. Evicts the least recently used entry when full.
    """
    normalized_key = key.strip().lower()
    if not normalized_key:
        return

    # Re-insert rather than overwrite so the refreshed entry becomes most recently used.
    _CACHE.pop(normalized_key, None)
    _CACHE[normalized_key] = {
        "chunks": chunks,
        "answer": answer,
        "vector": vector,
        "unit_vector": _to_unit_vector(vector),
        "cached_at": time.time(),
        "hit_count": 0,
    }

    while _CACHE and len(_CACHE) > max_entries:
        _CACHE.popitem(last=False)


def clear_cache() -> None:
    """
    Clear all entries from the cache.
    """
    _CACHE.clear()
