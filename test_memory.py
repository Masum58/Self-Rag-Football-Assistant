"""
WHAT: Manual test script for long-term memory (save + search + get).
WHY: Lets us verify AsyncPostgresStore works end-to-end before wiring it
     into the actual FastAPI routes/graph.
OUTPUT: Prints each step's result to the terminal.
CALLED FROM: Run manually via `python test_memory.py`
WHY CALLED: Quick manual sanity check, not part of the app itself.

চালানোর নিয়ম: venv activate করা অবস্থায়, project root থেকে চালান:
    python test_memory.py
"""

import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.memory.long_term import init_store, close_store, save_memory, search_memory, get_memory

async def main():
    print("Store চালু হচ্ছে...")
    await init_store()

    print("\n--- কিছু memory save করছি ---")
    await save_memory("test_user", "favorite_team", "User's favorite football team is Brazil")
    await save_memory("test_user", "language_pref", "User prefers Bengali mixed with English technical terms")
    await save_memory("test_user", "location", "User is based in Bangladesh")
    print("✅ ৩টা memory save হয়ে গেছে")

    print("\n--- Exact key দিয়ে get করছি ---")
    team = await get_memory("test_user", "favorite_team")
    print("favorite_team:", team)

    print("\n--- Semantic search করছি ---")
    results = await search_memory("test_user", "which country does the user support in football?")
    print("Search results:")
    for r in results:
        print(" -", r)

    print("\n--- Store বন্ধ করছি ---")
    await close_store()
    print("✅ সব ঠিকমতো চলেছে!")


if __name__ == "__main__":
    asyncio.run(main())