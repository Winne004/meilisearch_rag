from typing import Literal

from langchain_aws import ChatBedrockConverse
from langchain_core.language_models.chat_models import BaseChatModel

from src.conf.settings import get_settings


def get_langchain_base_chat_model(
    llm_name: Literal["Bedrock"],
) -> type["BaseChatModel"]:
    settings = get_settings()
    if llm_name == "Bedrock":
        return ChatBedrockConverse(
            provider=settings.provider,
            model=settings.model_id,
            aws_access_key_id=settings.aws_access_key_id.get_secret_value(),  # type: ignore
            aws_secret_access_key=settings.aws_secret_access_key.get_secret_value(),  # type: ignore
            region_name=settings.region,
        )

    raise ValueError(f"Unsupported LLM name: {llm_name}. Supported: 'Bedrock'.")
