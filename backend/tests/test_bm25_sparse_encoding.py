"""The hand-rolled BM25 sparse encoder used for Qdrant's lexical half of hybrid search."""

from rag_persona.ingestion.bm25 import (
    BM25Encoder,
    SparseVector,
    encode_sparse_query,
    stable_token_index,
)

CORPUS = [
    "the quick brown fox jumps over the dog",
    "the lazy dog sleeps all day",
    "the cat and the dog are friends",
    "the bird flies above the roof",
]


def dot_product(document: SparseVector, query: SparseVector) -> float:
    weights = dict(zip(document.indices, document.values, strict=True))
    return sum(weights.get(index, 0.0) * value
               for index, value in zip(query.indices, query.values, strict=True))


def test_a_rare_term_outweighs_a_term_in_every_document() -> None:
    encoder = BM25Encoder(CORPUS)
    assert encoder.idf["fox"] > encoder.idf["dog"] > encoder.idf["the"]


def test_every_encoded_term_gets_exactly_one_index_and_one_weight() -> None:
    encoder = BM25Encoder(CORPUS)
    vector = encoder.encode_document(CORPUS[0])
    assert len(vector.indices) == len(vector.values)
    assert len(set(vector.indices)) == len(vector.indices)


def test_terms_absent_from_the_corpus_are_dropped_rather_than_scored() -> None:
    """An unseen token has no IDF, and Qdrant would never match it anyway."""
    encoder = BM25Encoder(CORPUS)
    assert encoder.encode_document("xylophone unicycle").indices == []


def test_token_indices_are_stable_across_calls_and_encoder_instances() -> None:
    """Indices are the wire format: a drift would silently orphan the whole collection."""
    assert stable_token_index("retrieval") == stable_token_index("retrieval")
    assert stable_token_index("retrieval") != stable_token_index("retrievai")
    assert encode_sparse_query("qdrant").indices == [stable_token_index("qdrant")]


def test_a_document_containing_the_query_terms_outscores_one_that_does_not() -> None:
    """Why the query side stays raw term frequency: IDF is already in the document
    vector, so their dot product is the BM25 score. Adding IDF here would square it."""
    encoder = BM25Encoder(CORPUS)
    query = encode_sparse_query("quick fox")
    relevant = dot_product(encoder.encode_document(CORPUS[0]), query)
    irrelevant = dot_product(encoder.encode_document(CORPUS[1]), query)
    assert relevant > irrelevant


def test_repeating_a_query_term_weights_it_more_heavily() -> None:
    query = encode_sparse_query("dog dog cat")
    weights = dict(zip(query.indices, query.values, strict=True))
    assert weights[stable_token_index("dog")] == 2.0
    assert weights[stable_token_index("cat")] == 1.0
