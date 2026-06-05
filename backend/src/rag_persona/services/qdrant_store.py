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
        self.client = QdrantClient(
            url=str(settings.qdrant_url),
            api_key=settings.qdrant_api_key or None,
            timeout=settings.request_timeout_seconds,
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

    def point_exists(self, chunk_id: str) -> bool:
        """Check whether a point with the stable chunk_id already exists in the collection."""
        point_id = str(uuid5(NAMESPACE_URL, str(chunk_id)))
        try:
            # qdrant-client exposes get_point which returns None or raises on missing
            self.client.get_point(collection_name=self.settings.qdrant_collection, point_id=point_id)
            return True
        except Exception:
            return False

    def delete_by_repo(self, repo_name: str, dry_run: bool = False) -> int:
        """Delete points matching repo_name. Returns number of points deleted (best effort)."""
        from qdrant_client.http import models as _models

        filt = _models.Filter(must=[_models.FieldCondition(key="repo_name", match=_models.MatchValue(value=repo_name))])
        # preview count
        count = self.client.count(collection_name=self.settings.qdrant_collection, filter=filt)
        to_delete = count.count if hasattr(count, "count") else 0
        if dry_run or to_delete == 0:
            return int(to_delete)
        # perform delete
        self.client.delete(collection_name=self.settings.qdrant_collection, filter=filt)
        return int(to_delete)

    def search(
        self,
        query_vector: list[float],
        source_filter: SourceType | None,
        limit: int,
        sparse_query: SparseVector | None = None,
        repo_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        must_conditions = []
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
                with_vectors=True,
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
                with_vectors=True,
            )
            results = response.points

        chunks: list[RetrievedChunk] = []
        for result in results:
            payload = result.payload or {}
            vector = result.vector
            dense_vector: list[float] | None = None
            if isinstance(vector, dict) and isinstance(vector.get("dense"), list):
                dense_vector = [float(item) for item in vector["dense"]]
            if dense_vector is not None:
                payload = {**payload, "dense_vector": dense_vector}
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
