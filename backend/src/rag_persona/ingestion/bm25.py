import math
import re
from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b

TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]


class BM25Encoder:
    def __init__(self, documents: list[str], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        tokenized = [tokenize(document) for document in documents]
        self.average_length = sum(len(tokens) for tokens in tokenized) / max(len(tokenized), 1)
        document_frequency: Counter[str] = Counter()

        for tokens in tokenized:
            document_frequency.update(set(tokens))

        self.idf: dict[str, float] = {}
        total_documents = max(len(tokenized), 1)
        for token, frequency in document_frequency.items():
            self.idf[token] = math.log(1 + (total_documents - frequency + 0.5) / (frequency + 0.5))

    def encode_document(self, text: str) -> SparseVector:
        tokens = tokenize(text)
        counts = Counter(tokens)
        document_length = max(len(tokens), 1)
        indices: list[int] = []
        values: list[float] = []

        for token, frequency in counts.items():
            idf = self.idf.get(token)
            if idf is None:
                continue
            index = stable_token_index(token)
            denominator = frequency + self.k1 * (
                1 - self.b + self.b * document_length / max(self.average_length, 1)
            )
            score = idf * (frequency * (self.k1 + 1)) / denominator
            indices.append(index)
            values.append(float(score))

        return SparseVector(indices=indices, values=values)


def stable_token_index(token: str) -> int:
    digest = blake2b(token.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big", signed=False)


def encode_sparse_query(text: str) -> SparseVector:
    """Encode a query as raw term frequencies.

    Query weights are deliberately unweighted: `BM25Encoder.encode_document` already bakes
    IDF and length normalisation into the document vector, so the dot product of a raw-tf
    query with a document vector is the standard BM25 score. Applying IDF here as well
    would double-count it.
    """
    counts = Counter(tokenize(text))
    return SparseVector(
        indices=[stable_token_index(token) for token in counts],
        values=[float(frequency) for frequency in counts.values()],
    )
