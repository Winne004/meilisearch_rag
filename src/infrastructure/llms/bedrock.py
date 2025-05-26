from typing import Any, override

from langchain_aws import BedrockEmbeddings
from langchain_core.exceptions import LangChainException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.conf.settings import get_settings
from src.exceptions.exceptions import KeywordExtractionError, SummarisationError
from src.infrastructure.llms.base import LLMABC


class LangchainLLM(LLMABC):
    def __init__(self, chat_model: BaseChatModel) -> None:
        self.llm = chat_model
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
        self.summary_prompt = self.summarise_prompt = ChatPromptTemplate.from_messages(  # type: ignore
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

    @override
    def extract_keywords(self, query: str) -> str:
        try:
            chain = self.keyword_prompt | self.llm | StrOutputParser()  # type: ignore
            return chain.invoke({"query": query})  # type: ignore
        except LangChainException as e:
            raise KeywordExtractionError("Keyword extraction failed") from e

    @override
    def summarise(
        self,
        query: str,
        results: dict[str, Any],
    ) -> str:
        try:
            summarise_chain = self.summarise_prompt | self.llm | StrOutputParser()  # type: ignore
            return summarise_chain.invoke(  # type: ignore
                {
                    "original_query": query,
                    "results": results["hits"],
                },
            )
        except LangChainException as e:
            raise SummarisationError("Summarisation failed") from e


def get_embedder() -> BedrockEmbeddings:
    settings = get_settings()
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        aws_access_key_id=settings.aws_access_key_id.get_secret_value(),  # type: ignore
        aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),  # type: ignore
        region_name=settings.region,
    )
