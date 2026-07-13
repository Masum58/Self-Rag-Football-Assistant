"""
WHAT: Quick sanity check script to confirm all settings load correctly from .env
WHY: Verifies Pydantic Settings picks up Redis, Postgres, Pinecone, Groq config
     before we build actual connection logic.
OUTPUT: Prints each loaded value (API keys truncated for safety) to the terminal.
CALLED FROM: Run manually via `python test_config.py`
WHY CALLED: Manual verification step during setup, not part of the app itself.
"""

from app.core.config import settings

print("REDIS_URL       :", settings.redis_url)
print("POSTGRES_DSN    :", settings.postgres_dsn)
print("PINECONE_KEY    :", settings.pinecone_api_key[:8] + "...")
print("PINECONE_CLOUD  :", settings.pinecone_cloud)
print("PINECONE_REGION :", settings.pinecone_region)
print("PINECONE_INDEX  :", settings.pinecone_index_name)
print("GROQ_KEY        :", settings.groq_api_key[:8] + "...")
print("\n✅ All settings loaded successfully!")