from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import LangChainException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.domain.chunk import chunk_paragraphs
from src.domain.dataclasses.dataclasses import (
    Document,
    SearchRequestDataClass,
    SimilarityRequestDataClass,
    VectorisedDocument,
)
from src.exceptions.exceptions import (
    ConversationalSearchError,
    EmbedderError,
    IndexingError,
    SemanticSearchError,
    SimilarSearchError,
    VectorDatabaseError,
)
from src.infrastructure.logger import setup_logger
from src.infrastructure.vectorstores.base import VectorStoreABC

logger = setup_logger(name="logger")


class SearchService:
    def __init__(
        self,
        embedder: Embeddings,
        vectorstore: VectorStoreABC,
        llm: BaseChatModel,
    ) -> None:
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm
        # --- Prompt for keyword extraction ---
        self.keyword_prompt = ChatPromptTemplate.from_messages(  # type: ignore
            [
                (
                    "system",
                    """
                "You are an assistant that extracts keywords from user queries.\n"
                "Your task is to identify named entities and core topics from the query "
                "and return them as a simple, comma-separated string (without punctuation).\n\n"
                "**Example:**\n"
                "Input: 'Tell me about the royal family's trip to Scotland.'\n"
                "Output: king, queen, royal family, Scotland"
                    """,
                ),
                ("human", "Search query: {query}"),
            ],
        )

        # --- Prompt for summarization ---
        self.summarise_prompt = ChatPromptTemplate.from_messages(  # type: ignore
            [
                (
                    "system",
                    """
                You are a summarisation assistant.
                Your task is to summarise search results based on the original query.
                You must not use any external knowledge or data.
                You summarise information from news articles in the manner of a journalist. 
                You must only use the context passed to you in the prompt. 

                If you can't answer - You should say so. 
                You must not use your training data. 
            """,
                ),
                ("human", "query: {original_query}"),
                ("human", "Search results: {results}"),
            ],
        )

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

        except VectorDatabaseError as e:
            error_message = (
                "Index failed due to a vector database error. "
                "Please check vector indexing and MeiliSearch configuration."
            )
            raise IndexingError(message=error_message) from e

    def semantic_search(self, request: SearchRequestDataClass) -> dict[str, Any]:
        try:
            embedded_query = self.embedder.embed_query(request.query)
            return self.vectorstore.hybrid_search(query=request, vector=embedded_query)

        except LangChainException as e:
            error_message = (
                "Failed to perform semantic search. "
                "Check if the LLM is configured correctly."
            )
            raise SemanticSearchError(message=error_message) from e

        except VectorDatabaseError as e:
            error_message = (
                "Semantic search failed due to a vector database error. "
                "Please check vector db configuration."
            )
            raise SemanticSearchError(message=error_message) from e

    def conversational_search(self, request: SearchRequestDataClass) -> dict[str, Any]:
        try:
            keyword_chain = self.keyword_prompt | self.llm | StrOutputParser()  # type: ignore
            keywords = keyword_chain.invoke({"query": request.query})  # type: ignore

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

            summarise_chain = self.summarise_prompt | self.llm | StrOutputParser()  # type: ignore
            summary = summarise_chain.invoke(  # type: ignore
                {
                    "original_query": request.query,
                    "results": results["hits"],
                },
            )

            return {"summary": summary, "sources": results["hits"]}
        except LangChainException as e:
            error_message = (
                "Failed to perform conversational search. "
                "Check if the LLM is configured correctly."
            )
            raise ConversationalSearchError(message=error_message) from e

        except VectorDatabaseError as e:
            error_message = (
                "Semantic search failed due to a vector database error. "
                "Please check vector indexing and MeiliSearch configuration."
            )
            raise ConversationalSearchError(message=error_message) from e

    def similar_search(self, request: SimilarityRequestDataClass) -> dict[str, Any]:
        try:
            return self.vectorstore.similarity_search(
                request,
            )

        except VectorDatabaseError as e:
            error_message = (
                "Similarity search failed due to a vector database error. "
                "Please check vector indexing and MeiliSearch configuration."
            )
            raise SimilarSearchError(message=error_message) from e
