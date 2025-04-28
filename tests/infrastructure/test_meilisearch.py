import pytest

from src.domain.dataclasses.dataclasses import (
    SearchRequestDataClass,
    VectorisedDocument,
)
from src.infrastructure.vectorstores.meilisearch import MeiliVectorStore
from tests.fakes import FakeMeiliIndex, fake_results


@pytest.fixture
def service() -> MeiliVectorStore:
    return MeiliVectorStore(
        index=FakeMeiliIndex(),  # type: ignore
        embedder_name="test_embedder",  # type: ignore
    )


def test_add_texts(service: MeiliVectorStore) -> None:
    vector_data = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]]
    document = VectorisedDocument(
        id="1",
        url="http://localhost:8000/docs",
        chunk="a chunk of text",
        vector=vector_data,
    )

    expected = {
        "id": "1",
        "url": document.url,
        "chunk": document.chunk,
        "_vectors": {service.embedder_name: document.vector},
    }

    service.add_texts([document])
    actual = service.index.get_document("1")

    assert actual == expected


def test_hybrid_search(service: MeiliVectorStore) -> None:
    search_query = SearchRequestDataClass(query="test query", limit=5)

    result = service.hybrid_search(
        query=search_query,
        vector=[0.0, 0.0, 0.0],
    )
    assert result == fake_results
