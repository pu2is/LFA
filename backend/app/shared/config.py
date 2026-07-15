from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "LFA Backend"
    environment: str = "development"

    database_url: str = "postgresql+psycopg://lfa:lfa_password@localhost:5432/lfa"

    redis_url: str = "redis://localhost:6379/0"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:3b"
    ollama_embed_model: str = "bge-m3"
    # Wider than Ollama's runtime default (2048-4096): the 3-stage initial
    # labeling flow (ADR-0001 D3) accumulates conversation history across
    # calls, which can otherwise get silently truncated mid-flow.
    ollama_num_ctx: int = 8192

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()