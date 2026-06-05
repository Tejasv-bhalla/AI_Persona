import numpy as np

from rag_persona.schemas import RetrievedChunk


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


def rerank_by_cosine(
    query_vector: list[float],
    chunks: list[RetrievedChunk],
    candidate_vectors: dict[str, list[float]],
    limit: int,
) -> list[RetrievedChunk]:
    query = np.asarray(query_vector, dtype=np.float32)
    scored: list[RetrievedChunk] = []

    for chunk in chunks:
        vector = candidate_vectors.get(chunk.chunk_id)
        if vector is None:
            scored.append(chunk)
            continue
        score = cosine_similarity(query, np.asarray(vector, dtype=np.float32))
        scored.append(chunk.model_copy(update={"score": score}))

    return sorted(scored, key=lambda chunk: chunk.score, reverse=True)[:limit]

