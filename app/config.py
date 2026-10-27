"""Application settings, read from environment variables and the project's .env file."""

from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.errors import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # NAMS
    memory_api_key: SecretStr | None = None
    memory_endpoint: str = "https://memory.neo4jlabs.com/v1"
    memory_workspace_id: str | None = None
    memory_timeout_seconds: float = 30.0

    # Chat model
    llm_provider: Literal["openai"] = "openai"
    llm_api_key: SecretStr | None = None
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None

    # Embeddings (separate from the chat model)
    embedding_model: str = "text-embedding-3-small"
    embedding_api_key: SecretStr | None = None
    embedding_base_url: str | None = None

    # ChromaDB and retrieval
    chroma_dir: str = "data/chroma"
    chroma_collection: str = "documents"
    chunk_size: int = 800
    chunk_overlap: int = 100
    max_upload_mb: float = 10
    retrieval_top_k: int = 4
    retrieval_max_distance: float = 0.75

    # Chat behaviour
    history_messages: int = 6
    memory_entity_limit: int = 5
    extraction_wait_seconds: float = 0

    @field_validator(
        "memory_api_key",
        "memory_workspace_id",
        "llm_api_key",
        "llm_base_url",
        "embedding_api_key",
        "embedding_base_url",
        mode="before",
    )
    @classmethod
    def _blank_means_unset(cls, value):
        # "LLM_API_KEY=" in a .env file should behave like "not set".
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @model_validator(mode="after")
    def _check_chunking(self) -> "Settings":
        if self.chunk_size < 50:
            raise ValueError("CHUNK_SIZE must be at least 50 characters")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be >= 0 and smaller than CHUNK_SIZE")
        return self

    # --- derived values ---------------------------------------------------

    @property
    def chroma_path(self) -> Path:
        """Chroma directory; a relative CHROMA_DIR is resolved against the project root."""
        path = Path(self.chroma_dir)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    # --- credential checks (called when a feature is used, not at import) ---

    def require_memory_key(self) -> str:
        if self.memory_api_key is None:
            raise ConfigError(
                "MEMORY_API_KEY is not set. Add your NAMS API key to the .env file."
            )
        return self.memory_api_key.get_secret_value()

    def require_llm_key(self) -> str:
        if self.llm_api_key is None:
            raise ConfigError("LLM_API_KEY is not set. Add your chat-model API key to the .env file.")
        return self.llm_api_key.get_secret_value()

    def require_embedding_key(self) -> str:
        if self.embedding_api_key is not None:
            return self.embedding_api_key.get_secret_value()
        # Reusing the chat key is only safe when both clients talk to the default OpenAI host.
        if self.llm_api_key is not None and not self.llm_base_url and not self.embedding_base_url:
            return self.llm_api_key.get_secret_value()
        raise ConfigError(
            "EMBEDDING_API_KEY is not set. Add it to the .env file "
            "(it can only fall back to LLM_API_KEY when no *_BASE_URL is configured)."
        )
