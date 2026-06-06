from functools import lru_cache

from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    allowed_origins: str = "http://localhost:5173"
    dry_run: bool = False

    groq_api_key: str = Field(default="", repr=False)
    groq_guard_model: str = "llama-3.1-8b-instant"
    groq_generation_model: str = "llama-3.3-70b-versatile"
    groq_grader_model: str = "llama-3.3-70b-versatile"
    groq_voice_model: str = "llama-3.1-8b-instant"

    qdrant_url: HttpUrl | None = None
    qdrant_api_key: str = Field(default="", repr=False)
    qdrant_collection: str = "tejasv_knowledge_base"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimensions: int = 384

    calcom_api_key: str = Field(default="", repr=False)
    calcom_event_type_id: str = ""
    calcom_username: str = ""
    github_token: str = Field(default="", repr=False)
    github_username: str = ""

    max_retrieval_candidates: int = 8
    max_context_chunks: int = 3
    request_timeout_seconds: float = 20.0

    vapi_api_key: str = Field(default="", repr=False)
    vapi_phone_number_id: str = ""
    vapi_assistant_id: str = ""
    vapi_webhook_secret: str = Field(default="", repr=False)
    voice_max_response_words: int = 80
    voice_cache_ttl_seconds: int = 3600

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
