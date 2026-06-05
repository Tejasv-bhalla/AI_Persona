from collections.abc import Iterable
from functools import cached_property

import numpy as np
from fastembed import TextEmbedding

from rag_persona.config import Settings


class EmbeddingService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def model(self) -> TextEmbedding:
        return TextEmbedding(model_name=self.settings.embedding_model)

    def embed_one(self, text: str) -> list[float]:
        vector = next(self.model.embed([text]))
        return vector.astype(np.float32).tolist()

    def embed_many(self, texts: Iterable[str]) -> list[list[float]]:
        return [vector.astype(np.float32).tolist() for vector in self.model.embed(list(texts))]

