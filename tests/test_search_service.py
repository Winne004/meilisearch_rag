import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from src.domain.dataclasses.dataclasses import Document, SearchRequestDataClass
from src.exceptions.exceptions import (
    ConversationalSearchError,
    EmbedderError,
    SemanticSearchError,
)
from src.infrastructure.llms.bedrock import LangchainLLM
from src.service.search_service import SearchService
from tests.fakes import (
    FailingEmbedder,
    FailingVectorStore,
    FakeEmbedder,
    FakeFailingLangchainLLM,
    FakeLangchainLLM,
    FakeVectorStore,
    fake_results,
)


@pytest.fixture
def service() -> SearchService:
    first_msg = AIMessage(content="test")
    second_msg = AIMessage(content="summary of documents")
    return SearchService(
        embedder=FakeEmbedder(),
        vectorstore=FakeVectorStore(),
        llm=LangchainLLM(FakeMessagesListChatModel(responses=[first_msg, second_msg])),
    )


@pytest.fixture
def failing_embedder_service() -> SearchService:
    return SearchService(
        embedder=FailingEmbedder(),
        vectorstore=FailingVectorStore(),
        llm=FakeLangchainLLM(),
    )


@pytest.fixture
def failing_llm_service() -> SearchService:
    return SearchService(
        embedder=FakeEmbedder(),
        vectorstore=FakeVectorStore(),
        llm=FakeFailingLangchainLLM(),
    )


@pytest.fixture
def failing_vector_db_service() -> SearchService:
    return SearchService(
        embedder=FakeEmbedder(),
        vectorstore=FailingVectorStore(),
        llm=FakeLangchainLLM(),
    )


def test_index_documents(service: SearchService) -> None:
    doc = Document(id="1", body="Some body", url="http://example.com")
    service.index_documents([doc])

    vectorstore: FakeVectorStore = service.vectorstore  # type: ignore
    assert len(vectorstore.texts) > 0
    assert isinstance(vectorstore.texts[0].vector, list)


def test_semantic_search(service: SearchService):
    request = SearchRequestDataClass(query="hello", limit=1)
    result = service.semantic_search(request)
    assert result == fake_results


def test_conversational_search(service: SearchService):
    request = SearchRequestDataClass(query="What is AI?", limit=1)
    result = service.conversational_search(request)

    assert "summary" in result
    assert "sources" in result
    assert result["summary"] == "summary of documents"


def test_similarity_search(service: SearchService):
    from src.domain.dataclasses.dataclasses import SimilarityRequestDataClass

    request = SimilarityRequestDataClass(id="1", limit=1)
    result = service.similar_search(request)

    assert result == fake_results


def test_index_documents_fails_with_embedder_error():
    service = SearchService(FailingEmbedder(), FailingVectorStore(), FakeLangchainLLM())
    doc = Document(id="1", body="Fail", url="http://fail.com")

    with pytest.raises(EmbedderError):
        service.index_documents([doc])


def test_semantic_search_fails_with_error(failing_vector_db_service: SearchService):
    request = SearchRequestDataClass(query="bad query", limit=1)

    with pytest.raises(SemanticSearchError):
        failing_vector_db_service.semantic_search(request)


def test_conversational_search_fails_with_llm_error(failing_llm_service: SearchService):
    request = SearchRequestDataClass(query="What's up?", limit=1)

    with pytest.raises(ConversationalSearchError):
        failing_llm_service.conversational_search(request)
