import math
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_persona.config import Settings
from rag_persona.ingestion.bm25 import SparseVector
from rag_persona.schemas import RetrievedChunk, SourceType


class QdrantStore:
    def __init__(self, settings: Settings) -> None:
        if settings.qdrant_url is None:
            raise RuntimeError("QDRANT_URL is required")
        self.settings = settings
        self._repo_names: list[str] | None = None
        self.client = QdrantClient(
            url=str(settings.qdrant_url),
            api_key=settings.qdrant_api_key or None,
            # qdrant-client types timeout as int and rounds it up internally; do the
            # same here so the float setting matches the declared signature.
            timeout=math.ceil(settings.request_timeout_seconds),
        )

    def ensure_collection(self) -> None:
        collections = self.client.get_collections().collections
        if any(item.name == self.settings.qdrant_collection for item in collections):
            return

        self.client.create_collection(
            collection_name=self.settings.qdrant_collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.settings.embedding_dimensions,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                )
            },
        )
        self.client.create_payload_index(
            collection_name=self.settings.qdrant_collection,
            field_name="source_type",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.settings.qdrant_collection,
            field_name="repo_name",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def reset_collection(self) -> None:
        collections = self.client.get_collections().collections
        if any(item.name == self.settings.qdrant_collection for item in collections):
            self.client.delete_collection(self.settings.qdrant_collection)
        self.ensure_collection()

    def upsert_chunks(
        self,
        chunks: list[dict[str, object]],
        vectors: list[list[float]],
        sparse_vectors: list[SparseVector] | None = None,
        batch_size: int = 100,
    ) -> None:
        sparse_vectors = sparse_vectors or [SparseVector(indices=[], values=[]) for _ in chunks]
        for start in range(0, len(chunks), batch_size):
            points: list[models.PointStruct] = []
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]
            batch_sparse = sparse_vectors[start : start + batch_size]
            for chunk, vector, sparse_vector in zip(
                batch_chunks,
                batch_vectors,
                batch_sparse,
                strict=True,
            ):
                chunk_id = str(chunk["chunk_id"])
                points.append(
                    models.PointStruct(
                        id=str(uuid5(NAMESPACE_URL, chunk_id)),
                        vector={
                            "dense": vector,
                            "bm25": models.SparseVector(
                                indices=sparse_vector.indices,
                                values=sparse_vector.values,
                            ),
                        },
                        payload=chunk,
                    )
                )
            self.client.upsert(collection_name=self.settings.qdrant_collection, points=points)

    def repo_names(self) -> list[str]:
        """Distinct repo_name values in the collection, cached after the first successful call."""
        if self._repo_names is not None:
            return self._repo_names
        names: set[str] = set()
        try:
            offset = None
            while True:
                points, offset = self.client.scroll(
                    collection_name=self.settings.qdrant_collection,
                    limit=512,
                    offset=offset,
                    with_payload=["repo_name"],
                    with_vectors=False,
                )
                for point in points:
                    name = (point.payload or {}).get("repo_name")
                    if isinstance(name, str) and name:
                        names.add(name)
                if offset is None:
                    break
        except Exception:
            return []
        self._repo_names = sorted(names)
        return self._repo_names

    def delete_by_repo(self, repo_name: str, dry_run: bool = False) -> int:
        """Delete every point belonging to a repo. Returns the number of points matched."""
        selector = models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="repo_name",
                        match=models.MatchValue(value=repo_name),
                    )
                ]
            )
        )
        matched = self.client.count(
            collection_name=self.settings.qdrant_collection,
            count_filter=selector.filter,
            exact=True,
        ).count
        if dry_run or matched == 0:
            return int(matched)
        self.client.delete(
            collection_name=self.settings.qdrant_collection,
            points_selector=selector,
        )
        return int(matched)

    def search(
        self,
        query_vector: list[float],
        source_filter: SourceType | None,
        limit: int,
        sparse_query: SparseVector | None = None,
        repo_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        must_conditions: list[models.Condition] = []
        if source_filter and source_filter != SourceType.unknown:
            must_conditions.append(
                models.FieldCondition(
                    key="source_type",
                    match=models.MatchValue(value=source_filter.value),
                )
            )
        if repo_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="repo_name",
                    match=models.MatchValue(value=repo_filter),
                )
            )

        query_filter = models.Filter(must=must_conditions) if must_conditions else None

        if sparse_query and sparse_query.indices:
            response = self.client.query_points(
                collection_name=self.settings.qdrant_collection,
                prefetch=[
                    models.Prefetch(
                        query=query_vector,
                        using="dense",
                        filter=query_filter,
                        limit=limit,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=sparse_query.indices,
                            values=sparse_query.values,
                        ),
                        using="bm25",
                        filter=query_filter,
                        limit=limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results = response.points
        else:
            response = self.client.query_points(
                collection_name=self.settings.qdrant_collection,
                query=query_vector,
                using="dense",
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            results = response.points

        chunks: list[RetrievedChunk] = []
        for result in results:
            payload = result.payload or {}
            chunks.append(
                RetrievedChunk(
                    chunk_id=str(payload.get("chunk_id", result.id)),
                    text=str(payload.get("text", "")),
                    score=float(result.score),
                    source_type=SourceType(
                        str(payload.get("source_type", SourceType.unknown.value))
                    ),
                    repo_name=(
                        payload.get("repo_name")
                        if isinstance(payload.get("repo_name"), str)
                        else None
                    ),
                    file_path=(
                        payload.get("file_path")
                        if isinstance(payload.get("file_path"), str)
                        else None
                    ),
                    title=payload.get("title") if isinstance(payload.get("title"), str) else None,
                    metadata={key: value for key, value in payload.items() if key != "text"},
                )
            )
        return chunks
