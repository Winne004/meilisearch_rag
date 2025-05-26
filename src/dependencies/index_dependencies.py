from src.infrastructure.llms.bedrock import LangchainLLM, get_embedder
from src.infrastructure.llms.factory import get_langchain_base_chat_model
from src.infrastructure.vectorstores.meilisearch import get_vectorstore
from src.service.search_service import SearchService


def get_search_service() -> SearchService:
    embedder = get_embedder()
    vectorstore = get_vectorstore()
    llm = LangchainLLM(chat_model=get_langchain_base_chat_model(llm_name="Bedrock"))
    return SearchService(embedder, vectorstore, llm)
