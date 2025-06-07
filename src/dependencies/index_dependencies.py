from src.conf.settings import get_settings
from src.infrastructure.llms.bedrock import LangchainLLM, get_embedder
from src.infrastructure.llms.factory import get_langchain_base_chat_model
from src.infrastructure.vectorstores.meilisearch import get_vectorstore
from src.service.search_service import SearchService


def get_dependencies() -> SearchService:
    settings = get_settings()

    embedder = get_embedder(
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        model_id=settings.model_id,
        region=settings.region,
    )

    vectorstore = get_vectorstore(settings.meilisearch_url, settings.meili_master_key)
    llm = LangchainLLM(
        chat_model=get_langchain_base_chat_model(
            provider=settings.provider,
            model_id=settings.model_id,
            aws_access_key_id=settings.aws_access_key_id,
            aws_secret_access_key=settings.aws_secret_access_key,
            region=settings.region,
            llm_name=settings.model_provider,
        ),
    )

    return SearchService(embedder, vectorstore, llm)
