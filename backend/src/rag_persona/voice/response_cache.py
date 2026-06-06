import time
from typing import Any

# Global dictionary to store cached search results
# Format:
# {
#     "normalized_keyword_string": {
#         "chunks": list of RetrievedChunk,
#         "cached_at": float (timestamp),
#         "hit_count": int
#     }
# }
_CACHE: dict[str, dict[str, Any]] = {}


def get_cached_response(key: str, ttl_seconds: int = 3600) -> dict[str, Any] | None:
    """
    Retrieve cached search results for a normalized keyword key.
    Applies lazy TTL expiration.
    """
    normalized_key = key.strip().lower()
    if not normalized_key:
        return None

    entry = _CACHE.get(normalized_key)
    if not entry:
        return None

    # Check if the cache entry has expired
    cached_at = entry.get("cached_at", 0.0)
    if time.time() - cached_at > ttl_seconds:
        _CACHE.pop(normalized_key, None)
        return None

    entry["hit_count"] = entry.get("hit_count", 0) + 1
    return entry


def set_cached_response(key: str, chunks: list[Any]) -> None:
    """
    Store search results in the cache for the given keywords key.
    """
    normalized_key = key.strip().lower()
    if not normalized_key:
        return

    _CACHE[normalized_key] = {
        "chunks": chunks,
        "cached_at": time.time(),
        "hit_count": 0,
    }


def clear_cache() -> None:
    """
    Clear all entries from the cache.
    """
    _CACHE.clear()
