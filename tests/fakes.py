from collections.abc import Mapping
from typing import (
    Any,
    NoReturn,
    override,
)

from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import LangChainException
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.outputs import (
    ChatResult,
)
from langchain_core.runnables import RunnableConfig

from src.domain.dataclasses.dataclasses import (
    SearchRequestDataClass,
    SimilarityRequestDataClass,
    VectorisedDocument,
)
from src.exceptions.exceptions import SemanticSearchError, SimilarSearchError
from src.infrastructure.vectorstores.base import VectorStoreABC

fake_results = {
    "hits": [
        {
            "id": "https___example_org_accessibility___5_a1b2c3d4",
            "chuck": "All form fields follow a logical tab order for seamless navigation. Each field includes appropriate 'label' and 'id' tags to improve accessibility and data entry.\n\nLinks are written to be meaningful out of context, with descriptive titles added where helpful.\n\nThe site uses responsive typography. You can increase or decrease font sizes via your browser settings. Visit digitalaccesshelp.org for assistance.",
            "url": "https://example.org/accessibility/",
        },
        {
            "id": "https___example_org_accessibility___5",
            "chuck": "All form fields follow a logical tab order for seamless navigation. Each field includes appropriate 'label' and 'id' tags to improve accessibility and data entry.\n\nLinks are written to be meaningful out of context, with descriptive titles added where helpful.\n\nThe site uses responsive typography. You can increase or decrease font sizes via your browser settings. Visit digitalaccesshelp.org for assistance.",
            "url": "https://example.org/accessibility/",
        },
        {
            "id": "https___example_org_accessibility___7_e5f6g7h8",
            "chuck": "We’ve ensured that all visual elements meet accessibility contrast standards. A high-contrast mode is available through the navigation bar.\n\nWe provide documents in downloadable PDF format. Free PDF software is available at https://downloads.example.com/pdfreader.\n\nYour personal data is only used to respond to your inquiry and is deleted afterward. We do not share your data with third parties.",
            "url": "https://example.org/accessibility/",
        },
        {
            "id": "https___example_org_accessibility___7",
            "chuck": "We’ve ensured that all visual elements meet accessibility contrast standards. A high-contrast mode is available through the navigation bar.\n\nWe provide documents in downloadable PDF format. Free PDF software is available at https://downloads.example.com/pdfreader.\n\nYour personal data is only used to respond to your inquiry and is deleted afterward. We do not share your data with third parties.",
            "url": "https://example.org/accessibility/",
        },
        {
            "id": "https___example_org_team___2_x1y2z3a4",
            "chuck": "We support advanced services and equipment not covered by standard health programs, helping to enhance care environments for patients, families, and staff across all departments.\n\nWe also coordinate an outstanding volunteer team whose contributions improve the quality of care and experiences across our organization.\n\nWe’ll only contact you regarding your request, and your information will be securely deleted afterward. We do not distribute your personal data.",
            "url": "https://example.org/team/",
        },
    ],
    "query": "string",
    "processingTimeMs": 8,
    "limit": 5,
    "offset": 0,
    "estimatedTotalHits": 174,
    "semanticHitCount": 5,
}


class FakeEmbedder(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(i)] * 3 for i, _ in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeVectorStore(VectorStoreABC):
    def __init__(self) -> None:
        self.texts: list[VectorisedDocument] = []
        self.last_query = None
        self.last_vector = None

    def add_texts(self, documents: list[VectorisedDocument]) -> None:
        self.texts.extend(documents)

    def hybrid_search(
        self,
        query: SearchRequestDataClass,
        vector: list[float],
    ) -> dict[str, Any]:
        self.last_query = query
        self.last_vector = vector
        return fake_results

    def similarity_search(
        self,
        request: SimilarityRequestDataClass,
    ) -> dict[str, Any]:
        return fake_results


class FailingEmbedder(Embeddings):
    def embed_documents(self, texts: list[str]) -> NoReturn:  # noqa: ARG002
        raise LangChainException("embedding failed")

    def embed_query(self, text: str) -> NoReturn:  # noqa: ARG002
        raise LangChainException("query embedding failed")


class FailingVectorStore(VectorStoreABC):
    def add_texts(self, documents: list[VectorisedDocument]) -> NoReturn:
        raise Exception("add_texts failed")

    def hybrid_search(
        self,
        query: SearchRequestDataClass,
        vector: list[float],
    ) -> NoReturn:
        raise SemanticSearchError

    def similarity_search(
        self,
        request: SimilarityRequestDataClass,
    ) -> NoReturn:
        raise SimilarSearchError("similarity_search failed")


class FailingLLM(BaseChatModel):
    @override
    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> BaseMessage:
        raise LangChainException("LLM invoke failed")

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        ChatResult(
            generations=[],
            llm_output={},
        )
        raise LangChainException("LLM generate failed")

    @property
    def _llm_type(self) -> str:
        return "failing_llm"


class FakeMeiliIndex:
    """A fake MeiliSearch index for testing purposes."""

    def __init__(self) -> None:
        self.documents = {}

    def add_documents(
        self,
        documents: list[dict[str, dict[str, list[float]] | str]],
    ) -> None:
        for doc in documents:
            doc_id = doc.get("id")
            if doc_id is None:
                raise ValueError("Document must have an 'id' field.")
            self.documents[doc_id] = doc

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self.documents.get(doc_id)

    def search(
        self,
        query: SearchRequestDataClass,
        opt_params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return fake_results
