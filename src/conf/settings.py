import os
from functools import lru_cache
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

env_file = os.getenv("ENV_FILE", ".env")  # fallback to .env if ENV_FILE is not set


class Settings(BaseSettings):
    provider: Literal["Amazon"] = "Amazon"
    model_id: str
    embedder_name: str
    meilisearch_url: str
    meili_master_key: SecretStr
    aws_access_key_id: SecretStr
    aws_secret_access_key: SecretStr
    region: str
    model_provider: Literal["Bedrock"] = "Bedrock"
    model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="UTF-8")

    @classmethod
    def settings_customise_sources(  # type: ignore
        cls,
        settings_cls: PydanticBaseSettingsSource,
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[
        PydanticBaseSettingsSource,
        PydanticBaseSettingsSource,
        PydanticBaseSettingsSource,
        PydanticBaseSettingsSource,
    ]:
        return (
            dotenv_settings,  # 1st priority: .env file
            env_settings,  # 2nd priority: system env vars
            init_settings,  # 3rd priority: directly passed args
            file_secret_settings,  # 4th: /var/run/secrets etc.
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore
