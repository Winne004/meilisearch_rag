from src.infrastructure.llms.bedrock import get_bederock_llm, get_embedder
from src.infrastructure.vectorstores.meilisearch import get_vectorstore
from src.service.search_service import SearchService


def get_search_service() -> SearchService:
    embedder = get_embedder()
    vectorstore = get_vectorstore()
    llm = get_bederock_llm()
    return SearchService(embedder, vectorstore, llm)
