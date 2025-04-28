import os
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

env_file = os.getenv("ENV_FILE", ".env")  # fallback to .env if ENV_FILE is not set


class Settings(BaseSettings):
    provider: str = "Amazon"
    model_id: str
    embedder_name: str
    meilisearch_url: str
    meili_master_key: SecretStr
    aws_access_key_id: SecretStr
    aws_secret_access_key: SecretStr
    region: str
    model_config = SettingsConfigDict(env_file=env_file, env_file_encoding="UTF-8")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            dotenv_settings,  # 1st priority: .env file
            env_settings,  # 2nd priority: system env vars
            init_settings,  # 3rd priority: directly passed args
            file_secret_settings,  # 4th: /var/run/secrets etc.
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore
