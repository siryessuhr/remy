from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings, loaded from environment variables."""

    # Database
    DATABASE_URL: str

    # LLM provider configuration
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-5.6-sol"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_EMBEDDING_DIMENSIONS: int = 512

    OLLAMA_BASE_URL: str | None = None
    OLLAMA_MODEL: str | None = None
    OLLAMA_EMBEDDING_MODEL: str | None = None
    OLLAMA_EMBEDDING_DIMENSIONS: int = 512

    # LangSmith tracing
    LANGSMITH_API_KEY: str
    LANGSMITH_ENDPOINT: str
    LANGSMITH_PROJECT: str
    LANGSMITH_TRACING: bool = True

    # App / FastAPI
    APP_HOST: str
    APP_PORT: int
    DEV: bool

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=True)


settings = Settings()
