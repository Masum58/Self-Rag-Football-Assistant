"""
WHAT: Short-term memory layer using Redis — stores the raw back-and-forth
      of the CURRENT conversation (unlike long_term.py, which stores
      durable facts/summaries that survive forever).
WHY: The Self-RAG graph needs to remember "what did we just talk about"
     within a session, without cluttering Postgres with temporary chat turns.
     Redis is fast and supports auto-expiry (TTL), so old sessions clean
     themselves up automatically.
OUTPUT: `add_message()` appends a turn; `get_history()` fetches the whole
        conversation so far; `clear_session()` wipes it early if needed.
CALLED FROM: app/graph/nodes.py (to read recent context before generating,
             and to save each new turn after generating)
WHY CALLED: Gives the graph short-term conversational context (like human
            working memory), separate from long-term facts.

সহজ ভাষায় (Bengali):
Long-term memory ছিল "স্থায়ী তথ্য" (যেমন favorite_team, কখনো মুছবে না)।
এটা তার উল্টো — "আজকের চ্যাট" মনে রাখে, একটা নির্দিষ্ট সময় (TTL) পরে বা
session শেষ হলে নিজে থেকেই মুছে যায়। Redis-এ আমরা প্রতিটা session-এর
কথোপকথন একটা "list" হিসেবে রাখব — user বলল, bot বলল, user বলল... এভাবে
ক্রমানুসারে।
"""

import json
import redis.asyncio as redis

from app.core.config import settings

# TTL (Time To Live) — এই সময় পর Redis নিজে থেকেই key মুছে ফেলবে।
# ৩০ মিনিট = 1800 সেকেন্ড। মানে ৩০ মিনিট কথা না হলে session ভুলে যাবে।
SESSION_TTL_SECONDS = 1800

# Redis client — connection pool নিজেই ভেতরে ভেতরে ম্যানেজ করে, তাই
# আমাদের db.py এর মতো আলাদা করে pool বানানোর দরকার নেই।
_redis_client: redis.Redis | None = None


def get_redis_client() -> redis.Redis:
    """
    WHAT: Returns a singleton Redis client, creating it on first use.
    WHY: Avoids opening a new Redis connection every time a function in
         this file is called — reuses one client across the whole app.
    OUTPUT: An async Redis client instance.
    CALLED FROM: add_message(), get_history(), clear_session() (below).
    WHY CALLED: Internal helper — other files should not need to call this
                directly, just use the functions below.

    সহজ ভাষায়: এটা internal helper, সরাসরি call করার দরকার নেই।
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.redis_url, decode_responses=True)
    return _redis_client


def _session_key(session_id: str) -> str:
    """
    WHAT: Builds the Redis key name for a given session.
    WHY: Keeps key naming consistent everywhere (e.g. "chat_history:abc123"),
         so we don't accidentally use different formats in different places.
    OUTPUT: A string like "chat_history:<session_id>".
    CALLED FROM: add_message(), get_history(), clear_session() (below).
    WHY CALLED: Single source of truth for how session keys are named.

    Example:
        _session_key("abc123") → "chat_history:abc123"
    """
    return f"chat_history:{session_id}"


async def add_message(session_id: str, role: str, content: str) -> None:
    """
    WHAT: Appends one message (user or assistant turn) to a session's history.
    WHY: Every time the user says something, or the bot replies, we push it
         onto the Redis list — building up the conversation in order.
    OUTPUT: None.
    CALLED FROM: app/graph/nodes.py — once for the user's message, once for
                 the assistant's reply, each turn.
    WHY CALLED: Keeps a running record of "what's been said so far" in this
                session, so the next turn can use it as context.

    সহজ ভাষায়: প্রতিটা turn (user বলল / bot বলল) এখানে যোগ হবে। প্রতিবার
    call করার সময় TTL রিসেট হয়ে যায় — মানে active conversation চলতে থাকলে
    session কখনো expire হবে না, শুধু ৩০ মিনিট চুপ থাকলে মুছে যাবে।

    Example:
        await add_message("session_abc", "user", "আমার নাম Masum")
        await add_message("session_abc", "assistant", "নমস্কার Masum!")
    """
    client = get_redis_client()
    key = _session_key(session_id)
    message = json.dumps({"role": role, "content": content})

    await client.rpush(key, message)          # লিস্টের শেষে নতুন message যোগ করা
    await client.expire(key, SESSION_TTL_SECONDS)  # TTL রিসেট/সেট করা


async def get_history(session_id: str) -> list[dict]:
    """
    WHAT: Fetches the full conversation history for a session, in order.
    WHY: The graph needs to see recent turns before generating a reply,
         so the LLM has short-term context (like "what did they just ask").
    OUTPUT: List of dicts like [{"role": "user", "content": "..."}, ...],
            oldest message first. Empty list if session doesn't exist/expired.
    CALLED FROM: app/graph/nodes.py, before generation.
    WHY CALLED: Injects recent conversational context into the prompt.

    Example:
        history = await get_history("session_abc")
        # → [{"role": "user", "content": "আমার নাম Masum"},
        #    {"role": "assistant", "content": "নমস্কার Masum!"}]
    """
    client = get_redis_client()
    key = _session_key(session_id)
    raw_messages = await client.lrange(key, 0, -1)  # 0 থেকে -1 মানে "সবগুলো"
    return [json.loads(m) for m in raw_messages]


async def clear_session(session_id: str) -> None:
    """
    WHAT: Deletes a session's history immediately (before TTL expiry).
    WHY: Useful if the user explicitly wants to start a fresh conversation,
         rather than waiting 30 minutes for auto-expiry.
    OUTPUT: None.
    CALLED FROM: A "reset conversation" button/endpoint, if we add one later.
    WHY CALLED: Manual cleanup, on demand.

    Example:
        await clear_session("session_abc")
    """
    client = get_redis_client()
    key = _session_key(session_id)
    await client.delete(key)