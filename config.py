from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/email_manager"

    # AI Providers
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    gemini_api_key: str = ""
    openai_api_key: str = ""

    # Embedding Service
    ollama_embedding_url: str = "http://18.224.56.11:11434/api/embeddings"
    embedding_model: str = "nomic-embed-text"

    # App Settings
    default_timezone: str = "Asia/Karachi"
    email_fetch_limit: int = 8
    bulk_threshold: int = 1000
    ai_confidence_threshold: float = 0.7
    max_body_chars: int = 4000
    semantic_search_limit: int = 8

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
