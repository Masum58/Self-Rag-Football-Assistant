"""
WHAT: A simple health-check endpoint.
WHY: Gives us a quick way to confirm the FastAPI app is running AND that the
     Postgres connection pool actually works (by running a trivial query).
OUTPUT: JSON response with app status and db status.
CALLED FROM: Browser/curl hitting GET /health
WHY CALLED: Manual verification during setup; later useful for uptime monitoring.
"""

from fastapi import APIRouter
from app.core.db import get_pool

router = APIRouter()


@router.get("/health")
async def health_check():
    """
    WHAT: Checks FastAPI app status and runs `SELECT 1` against Postgres.
    WHY: `SELECT 1` is the simplest possible query — if it succeeds, the pool,
         the network path to Docker, and the credentials are all correct.
    OUTPUT: {"status": "ok", "database": "connected"} on success.
    CALLED FROM: GET /health
    WHY CALLED: Confirms db.py's connection pool actually works end-to-end.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        result = await conn.fetchval("SELECT 1")

    return {
        "status": "ok",
        "database": "connected" if result == 1 else "unexpected_response",
    }