from abc import ABC, abstractmethod
from typing import Any

from src.domain.dataclasses.dataclasses import (
    SearchRequestDataClass,
    VectorisedDocument,
)


class VectorStoreABC(ABC):
    @abstractmethod
    def add_texts(
        self,
        documents: list[VectorisedDocument],
    ) -> None:
        """Add a list of texts (with optional embeddings and metadata) to the vector store.

        Returns a list of IDs assigned to the inserted documents.
        """

    @abstractmethod
    def hybrid_search(
        self,
        query: SearchRequestDataClass,
        vector: list[float],
    ) -> dict[str, Any]:
        """Perform a semantic search against the vectorstore.

        Returns an object containing revelvant objects .
        """
