from langchain_aws import BedrockEmbeddings, ChatBedrockConverse

from src.conf.settings import get_settings


def get_bederock_llm() -> ChatBedrockConverse:
    settings = get_settings()
    return ChatBedrockConverse(
        provider=settings.provider,
        model=settings.model_id,
        aws_access_key_id=settings.aws_access_key_id.get_secret_value(),  # type: ignore
        aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),  # type: ignore
        region_name=settings.region,
    )


def get_embedder() -> BedrockEmbeddings:
    settings = get_settings()
    return BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        aws_access_key_id=settings.aws_access_key_id.get_secret_value(),  # type: ignore
        aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),  # type: ignore
        region_name=settings.region,
    )
