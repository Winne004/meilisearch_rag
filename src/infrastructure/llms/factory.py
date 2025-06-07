from typing import Literal

from langchain_aws import ChatBedrockConverse
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr


def get_langchain_base_chat_model(
    provider: Literal["Amazon"],
    model_id: str,
    aws_access_key_id: SecretStr,
    aws_secret_access_key: SecretStr,
    region: str,
    llm_name: Literal["Bedrock"],
) -> BaseChatModel:
    if llm_name == "Bedrock":
        return ChatBedrockConverse(
            provider=provider,
            model=model_id,
            aws_access_key_id=aws_access_key_id.get_secret_value(),  # type: ignore
            aws_secret_access_key=aws_secret_access_key.get_secret_value(),  # type: ignore
            region_name=region,
        )

    raise ValueError(f"Unsupported LLM name: {llm_name}. Supported: 'Bedrock'.")
