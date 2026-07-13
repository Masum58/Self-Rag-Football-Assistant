"""
WHAT: Central application configuration, loaded from environment variables (.env file).
WHY: Keeps all secrets/config in one place instead of scattering os.getenv() calls
     across the codebase. Makes it easy to swap values per environment (dev/prod).
OUTPUT: A singleton `settings` object importable anywhere in the app.
CALLED FROM: main.py, memory/short_term.py, memory/long_term.py, vectorstore/pinecone_client.py, graph/nodes.py
WHY CALLED: Any module that needs a connection string or API key imports `settings`
            instead of reading os.environ directly.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Redis (short-term memory)
    redis_url: str

    # PostgreSQL (long-term memory)
    postgres_dsn: str

    # Pinecone (vector store) — new serverless API only needs api_key for auth;
    # cloud/region are used when creating the index (no more "environment" concept)
    pinecone_api_key: str
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_index_name: str = "self-rag-index"

    # Groq (LLM for generation + grading)
    groq_api_key: str

    # Tavily (web search fallback, for hybrid mode)
    tavily_api_key: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


# Singleton instance — import this everywhere, don't re-instantiate Settings()
settings = Settings()