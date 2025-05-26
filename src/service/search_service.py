from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import LangChainException

from src.domain.chunk import chunk_paragraphs
from src.domain.dataclasses.dataclasses import (
    Document,
    SearchRequestDataClass,
    SimilarityRequestDataClass,
    VectorisedDocument,
)
from src.exceptions.exceptions import (
    EmbedderError,
)
from src.infrastructure.llms.base import LLMABC
from src.infrastructure.logger import setup_logger
from src.infrastructure.vectorstores.base import VectorStoreABC

logger = setup_logger(name="logger")


class SearchService:
    def __init__(
        self,
        embedder: Embeddings,
        vectorstore: VectorStoreABC,
        llm: LLMABC,
    ) -> None:
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm

    def index_documents(self, documents: list[Document]) -> None:
        """Index a document: create embeddings + store them."""
        try:
            vectorized_documents: list[VectorisedDocument] = []
            for document in documents:
                chunked_text = chunk_paragraphs(document.body)
                embeddings = self.embedder.embed_documents(chunked_text)

                for i, (embedding, chunk) in enumerate(
                    zip(embeddings, chunked_text, strict=True),
                ):
                    vectorized_documents.append(
                        VectorisedDocument(
                            id=f"{document.id}::{i}",
                            vector=embedding,
                            chunk=chunk,
                            url=document.url,
                        ),
                    )
            self.vectorstore.add_texts(vectorized_documents)

        except LangChainException as e:
            error_message = (
                "Failed to index documents. "
                "Check if the embedder is configured correctly."
            )
            raise EmbedderError(message=error_message) from e

    def semantic_search(self, request: SearchRequestDataClass) -> dict[str, Any]:
        embedded_query = self.embedder.embed_query(request.query)
        return self.vectorstore.hybrid_search(query=request, vector=embedded_query)

    def conversational_search(self, request: SearchRequestDataClass) -> dict[str, Any]:
        keywords = self.llm.extract_keywords(request.query)

        logger.info("Keywords: %s", keywords)

        embedded_query = self.embedder.embed_query(keywords)

        cleaned_query = SearchRequestDataClass(
            query=keywords,
            limit=request.limit,
        )
        results = self.vectorstore.hybrid_search(
            query=cleaned_query,
            vector=embedded_query,
        )

        summary = self.llm.summarise(request.query, results)

        return {"summary": summary, "sources": results["hits"]}

    def similar_search(self, request: SimilarityRequestDataClass) -> dict[str, Any]:
        return self.vectorstore.similarity_search(
            request,
        )
