"""
WHAT: Long-term memory layer built on LangGraph's AsyncPostgresStore.
WHY: Replaces our earlier hand-written SQL (memory_facts/memory_summaries)
     with LangGraph's own persistent Store — same put/get/search API as
     InMemoryStore, but backed by Postgres so memories survive restarts,
     plus built-in semantic search (search by meaning, not exact match).
OUTPUT: `init_store()` / `close_store()` manage the store's lifecycle.
        `save_memory()`, `search_memory()`, `get_memory()`, `list_memories()`
        are the functions the rest of the app calls to read/write memories.
CALLED FROM: app/main.py (init_store/close_store on startup/shutdown)
             app/graph/nodes.py (save_memory/search_memory during the
             Self-RAG graph's generation step)
WHY CALLED: Gives the graph a durable, semantically-searchable memory of
            each user across sessions.

সহজ ভাষায় (Bengali):
এই ফাইলটা মূলত ৪টা কাজ করে — মনে রাখা (save), সার্চ করে বের করা (search),
নির্দিষ্ট একটা জিনিস খুঁজে বের করা (get), আর সব দেখা (list)। ঠিক যেমন আপনার
দেখানো store.put()/store.get()/store.search() কোড, শুধু ডেটা এখন RAM এর বদলে
Postgres এ থাকবে (restart দিলেও হারাবে না) আর embedding OpenAI এর বদলে MiniLM
(লোকাল, ফ্রি) দিয়ে হবে।
"""

from sentence_transformers import SentenceTransformer
from langgraph.store.postgres.aio import AsyncPostgresStore

from app.core.config import settings

# ---------------------------------------------------------------------------
# Local embedding function (MiniLM) — no OpenAI needed, matches prior projects
# ---------------------------------------------------------------------------

_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIMS = 384  # all-MiniLM-L6-v2 output size


def _embed(texts: list[str]) -> list[list[float]]:
    """
    WHAT: Turns a list of strings into a list of embedding vectors.
    WHY: LangGraph's store `index` config accepts a plain function like this
         to compute embeddings for whatever gets stored, enabling semantic
         search without needing OpenAI.
    OUTPUT: List of float vectors, one per input string, each length 384.
    CALLED FROM: AsyncPostgresStore internally, whenever memories are saved
                 or a search query needs to be embedded.
    WHY CALLED: This is the "embed" function passed into the store's index config.

    সহজ ভাষায়: টেক্সটকে সংখ্যার list (vector) এ রূপান্তর করে, যাতে "similar meaning"
    এর টেক্সট গুলো কাছাকাছি সংখ্যায় পড়ে। আপনি নিজে এটা কখনো call করবেন না —
    store নিজে থেকেই ভেতরে ভেতরে এটা ব্যবহার করবে।

    Example:
        _embed(["User likes pizza", "User is from Bangladesh"])
        # → [[0.021, -0.114, ...], [0.087, 0.003, ...]]  (each list has 384 numbers)
    """
    vectors = _model.encode(texts, convert_to_numpy=True)
    return vectors.tolist()


# ---------------------------------------------------------------------------
# Store lifecycle (mirrors the pattern used in core/db.py for asyncpg)
# ---------------------------------------------------------------------------

_store_cm = None  # holds the async context manager so we can close it later
_store: AsyncPostgresStore | None = None


async def init_store() -> None:
    """
    WHAT: Opens the AsyncPostgresStore and runs setup() once to create its tables.
    WHY: Must run before any save/search call, same as connect_to_db() for asyncpg.
    OUTPUT: None (sets the module-level _store variable).
    CALLED FROM: app/main.py lifespan startup.
    WHY CALLED: Ensures the store's own tables exist and connection is ready
                before the first request arrives.

    সহজ ভাষায়: অ্যাপ চালু হওয়ার সময় একবার চলবে — memory store এর "দরজা খোলার"
    মতো। এটা না চললে save_memory/search_memory কোনোটাই কাজ করবে না
    (RuntimeError আসবে)।

    Example (app/main.py এর ভেতরে, lifespan startup এ):
        await init_store()
    """
    global _store_cm, _store
    _store_cm = AsyncPostgresStore.from_conn_string(
        settings.postgres_dsn,
        index={
            "dims": EMBEDDING_DIMS,
            "embed": _embed,
            "fields": ["data"],  # embed the "data" field of whatever dict we store
        },
    )
    _store = await _store_cm.__aenter__()
    await _store.setup()  # creates the store's tables (idempotent, safe to re-run)


async def close_store() -> None:
    """
    WHAT: Closes the AsyncPostgresStore connection cleanly.
    WHY: Mirrors close_db() — releases resources on shutdown.
    OUTPUT: None.
    CALLED FROM: app/main.py lifespan shutdown.
    WHY CALLED: Clean shutdown, no dangling connections.

    সহজ ভাষায়: অ্যাপ বন্ধ হওয়ার সময় "দরজা বন্ধ করা" — connection ছেড়ে দেওয়া।

    Example (app/main.py এর ভেতরে, lifespan shutdown এ):
        await close_store()
    """
    global _store_cm, _store
    if _store_cm is not None:
        await _store_cm.__aexit__(None, None, None)
        _store_cm = None
        _store = None


def _get_store() -> AsyncPostgresStore:
    """
    WHAT: Returns the active store instance, or raises if not initialized.
    WHY: Internal helper so every public function below doesn't repeat the
         same None-check.
    OUTPUT: The AsyncPostgresStore instance.
    CALLED FROM: save_memory, search_memory, get_memory, list_memories (below).
    WHY CALLED: Guards against calling memory functions before init_store() runs.

    সহজ ভাষায়: এটা internal helper, বাইরে থেকে সরাসরি call করার দরকার নেই।
    """
    if _store is None:
        raise RuntimeError(
            "Memory store is not initialized. Did you forget to call "
            "init_store() on app startup?"
        )
    return _store


# ---------------------------------------------------------------------------
# Public memory functions — same shape as the store.put/get/search you showed
# ---------------------------------------------------------------------------

async def save_memory(user_id: str, key: str, data: str) -> None:
    """
    WHAT: Saves one memory item under a user's namespace.
    WHY: Mirrors `store.put(namespace, key, {"data": ...})` from your example
         code, but scoped per user_id so each user's memories stay separate.
    OUTPUT: None.
    CALLED FROM: app/graph/nodes.py, whenever the graph decides to remember
                 something new about the user.
    WHY CALLED: Persists a fact/preference/summary for later semantic recall.

    সহজ ভাষায়: একটা নতুন তথ্য মনে রাখতে চাইলে এটা call করবেন। "key" হলো এই
    memory-র ইউনিক নাম (একই key দিয়ে আবার save করলে পুরনোটা overwrite হয়ে যাবে)।

    Example:
        await save_memory("masum", "favorite_team", "Brazil")
        await save_memory("masum", "language_pref", "Bengali mixed with English")
    """
    store = _get_store()
    namespace = ("memories", user_id)
    await store.aput(namespace, key, {"data": data})


async def search_memory(user_id: str, query: str, limit: int = 3) -> list[str]:
    """
    WHAT: Finds the most semantically relevant memories for a user, given a
          natural-language query.
    WHY: Mirrors `store.search(namespace, query=..., limit=...)` from your
         example — lets the graph pull in "what do we know that's relevant
         to this question" rather than every memory ever saved.
    OUTPUT: List of memory text strings, most relevant first.
    CALLED FROM: app/graph/nodes.py, before generation, to inject relevant
                 long-term context into the prompt.
    WHY CALLED: Gives the LLM personalized, relevant context without
                overflowing it with every fact ever stored.

    সহজ ভাষায়: exact শব্দ না মিললেও "meaning" কাছাকাছি হলে খুঁজে বের করে দেবে।
    যেমন "team" শব্দ না থাকলেও "which country does the user support" জিজ্ঞেস
    করলে "favorite_team: Brazil" এই memory-টা খুঁজে পাবে।

    Example:
        results = await search_memory("masum", "what does the user like to watch?")
        # → ["User likes watching Brazil football matches", "User enjoys World Cup"]
    """
    store = _get_store()
    namespace = ("memories", user_id)
    items = await store.asearch(namespace, query=query, limit=limit)
    return [item.value["data"] for item in items]


async def get_memory(user_id: str, key: str) -> str | None:
    """
    WHAT: Fetches one specific memory by its exact key.
    WHY: Mirrors `store.get(namespace, key)` — for when you know exactly
         which memory you want (not a semantic search).
    OUTPUT: The memory text, or None if that key doesn't exist.
    CALLED FROM: app/graph/nodes.py or routers, when a specific fact is needed.
    WHY CALLED: Direct lookup, faster than semantic search when the key is known.

    সহজ ভাষায়: search_memory এর মতো "আন্দাজ করে খোঁজা" না, এটা exact key দিয়ে
    সরাসরি বের করে — অনেক দ্রুত, কারণ semantic similarity calculate করতে হয় না।

    Example:
        team = await get_memory("masum", "favorite_team")
        # → "Brazil"  (or None if never saved)
    """
    store = _get_store()
    namespace = ("memories", user_id)
    item = await store.aget(namespace, key)
    return item.value["data"] if item else None


async def list_memories(user_id: str) -> list[str]:
    """
    WHAT: Lists every memory saved for a user, regardless of relevance.
    WHY: Mirrors `store.search(namespace)` with no query — useful for
         debugging or showing a user everything the system remembers.
    OUTPUT: List of all memory text strings for that user.
    CALLED FROM: A debug/admin route, or manual testing.
    WHY CALLED: Full visibility into what's stored per user.

    সহজ ভাষায়: এই user সম্পর্কে system এখন পর্যন্ত যা যা মনে রেখেছে, সব একসাথে
    দেখতে চাইলে এটা ব্যবহার করবেন — testing/debugging এর সময় কাজে লাগে।

    Example:
        all_memories = await list_memories("masum")
        # → ["favorite_team: Brazil", "language_pref: Bengali mixed with English"]
    """
    store = _get_store()
    namespace = ("memories", user_id)
    items = await store.asearch(namespace)
    return [item.value["data"] for item in items]