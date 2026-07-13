"""
WHAT: Manages a single asyncpg connection pool for PostgreSQL.
WHY: Opening a new DB connection on every request is slow. A pool keeps a set of
     ready-to-use connections open, so requests just borrow/return one instantly.
     We use raw asyncpg (no ORM) to keep full control over SQL, matching the
     pattern used in prior projects.
OUTPUT: `get_pool()` returns the active pool; `connect_to_db()` / `close_db()`
        open/close it once at app startup/shutdown.
CALLED FROM: app/main.py (on FastAPI startup/shutdown events)
             app/memory/long_term.py (to run queries against memory_facts / memory_summaries)
WHY CALLED: Any code that needs to read/write Postgres asks this module for a
            connection instead of creating its own.
"""

import asyncpg
from app.core.config import settings

# Module-level pool reference. Starts as None until connect_to_db() runs.
_pool: asyncpg.Pool | None = None


async def connect_to_db() -> None:
    """
    WHAT: Creates the asyncpg connection pool.
    WHY: Called once, when FastAPI starts up, so the pool is ready before
         any request needs it.
    OUTPUT: None (sets the module-level _pool variable).
    CALLED FROM: app/main.py lifespan startup.
    WHY CALLED: Ensures we don't open a fresh connection per-request.
    """
    global _pool
    _pool = await asyncpg.create_pool(
        dsn=settings.postgres_dsn,
        min_size=1,
        max_size=10,
    )


async def close_db() -> None:
    """
    WHAT: Closes the asyncpg connection pool gracefully.
    WHY: Called once, when FastAPI shuts down, so connections aren't left dangling.
    OUTPUT: None.
    CALLED FROM: app/main.py lifespan shutdown.
    WHY CALLED: Clean shutdown — releases DB resources back to Postgres.
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    """
    WHAT: Returns the active connection pool.
    WHY: Other modules (like memory/long_term.py) need a handle to the pool
         to run queries, without knowing how/when it was created.
    OUTPUT: The asyncpg.Pool instance.
    CALLED FROM: app/memory/long_term.py functions that run SQL queries.
    WHY CALLED: To acquire a connection via `async with get_pool().acquire() as conn:`
    """
    if _pool is None:
        raise RuntimeError(
            "Database pool is not initialized. Did you forget to call connect_to_db() "
            "on app startup?"
        )
    return _pool