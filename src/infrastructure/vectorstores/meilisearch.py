import re
from functools import lru_cache
from typing import Any

import meilisearch
from meilisearch.errors import MeilisearchError
from meilisearch.index import Index

from src.conf.settings import get_settings
from src.domain.dataclasses.dataclasses import (
    SearchRequestDataClass,
    VectorisedDocument,
)
from src.exceptions.exceptions import VectorDatabaseError
from src.infrastructure.vectorstores.base import VectorStoreABC


def get_meilisearch_client() -> meilisearch.Client:
    settings = get_settings()
    return meilisearch.Client(
        url=settings.meilisearch_url,
        api_key=settings.meili_master_key.get_secret_value(),
    )


class MeiliVectorStore(VectorStoreABC):
    def __init__(self, index: Index, embedder_name: str) -> None:
        self.index = index
        self.embedder_name = embedder_name

    def _sanitise_identifier(self, raw_value: str, max_bytes: int = 511) -> str:
        cleaned = raw_value.lower()

        cleaned = re.sub(r"[^a-z0-9_-]", "_", cleaned)

        if len(cleaned.encode("utf-8")) > max_bytes - 9:
            cleaned = cleaned[: max_bytes - 9]

        return f"{cleaned}"

    def add_texts(self, documents: list[VectorisedDocument]) -> None:
        try:
            documents_as_dict = self._convert_documents_to_dict(documents)

            self.index.add_documents(documents_as_dict)
        except MeilisearchError as e:
            message = "error adding documents to vector store"
            raise VectorDatabaseError(message=message) from e

    def _convert_documents_to_dict(
        self,
        documents: list[VectorisedDocument],
    ) -> list[dict[str, str | None | dict[str, list[float]]]]:
        return [
            {
                "id": self._sanitise_identifier(doc.id),
                "_vectors": {self.embedder_name: doc.vector},
                "chunk": doc.chunk,
                "url": doc.url,
            }
            for doc in documents
        ]

    def hybrid_search(
        self,
        query: SearchRequestDataClass,
        vector: list[float],
    ) -> dict[str, Any]:
        params = {
            "vector": vector,
            "limit": query.limit,
            "hybrid": {"embedder": self.embedder_name, "semanticRatio": 0.7},
        }
        try:
            return self.index.search(query=query.query, opt_params=params)
        except MeilisearchError as e:
            message = "error executing hybrid search"
            raise VectorDatabaseError(message=message) from e


@lru_cache
def get_vectorstore() -> MeiliVectorStore:
    """Return a wrapper around Meilisearch vector store."""
    client = get_meilisearch_client()
    index_name = "documents"
    api_settings = get_settings()
    settings = {
        "embedders": {
            f"{api_settings.embedder_name}": {
                "source": "userProvided",
                "dimensions": 1024,
            },
        },
    }

    # Ensure index exists
    if index_name not in [idx.uid for idx in client.get_indexes()["results"]]:
        client.create_index(index_name, {"primaryKey": "id"})

    index = client.index(index_name)
    index.update_settings(body=settings)

    return MeiliVectorStore(index=index, embedder_name=api_settings.embedder_name)
