from functools import lru_cache

from pydantic import Field, HttpUrl, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "local"
    allowed_origins: str = "http://localhost:5173"
    dry_run: bool = False
    repo_blocklist: list[str] = []

    groq_api_key: str = Field(default="", repr=False)
    # Every role uses a non-reasoning model. gpt-oss variants emit 37-87 reasoning
    # chunks before any content, which cost ~1.8s of dead air before the first token
    # and, measured against the golden set, led the model to embellish answers with
    # world knowledge absent from the retrieved context - the one thing a grounded
    # persona must not do. They also starve the JSON classifiers: at a 200-token cap
    # gpt-oss-20b returned empty output on 2 of 5 guard prompts where qwen answered
    # all 5 using 60 tokens. gpt-oss-20b is kept only as a cross-family fallback.
    groq_guard_model: str = "qwen/qwen3.8-27b"
    groq_generation_model: str = "qwen/qwen3.8-27b"
    groq_grader_model: str = "qwen/qwen3.8-27b"
    groq_voice_model: str = "qwen/qwen3.8-27b"
    # Used when the primary generation model is unavailable (rate limit / outage).
    groq_fallback_model: str = "openai/gpt-oss-20b"

    # Groq's free tier allows 1000 output tokens per minute and admits a request
    # against its max_tokens, not its actual output. One chat turn makes three
    # calls (guard, generation, grader), so these must sum below that ceiling or
    # the whole turn is rejected. Some models also refuse outright when no cap is
    # set, because their default max output alone exceeds the limit.
    max_output_tokens_json: int = 200
    max_output_tokens_chat: int = 500
    max_output_tokens_voice: int = 300

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
    max_context_chunks: int = 5
    request_timeout_seconds: float = 20.0

    vapi_api_key: str = Field(default="", repr=False)
    vapi_phone_number_id: str = ""
    vapi_assistant_id: str = ""
    vapi_webhook_secret: str = Field(default="", repr=False)
    voice_max_response_words: int = 80
    voice_cache_ttl_seconds: int = 3600
    voice_cache_max_entries: int = 256

    # Shared secret required by /chat/eval. Empty disables the endpoint entirely.
    eval_api_key: str = Field(default="", repr=False)

    rate_limit_chat: str = "20/minute"
    rate_limit_voice: str = "60/minute"
    rate_limit_book: str = "5/hour"

    @field_validator("qdrant_url", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        """Treat a blank QDRANT_URL as unset.

        `.env.example` ships the key with an empty value, so without this a fresh
        `cp .env.example .env` fails URL validation and the app cannot boot at all,
        instead of starting up and degrading gracefully as designed.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
