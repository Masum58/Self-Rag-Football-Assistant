"""
WHAT: Manual test script for short-term memory (Redis conversation history).
WHY: Verifies add_message/get_history works before wiring it into the graph.
OUTPUT: Prints each step's result to the terminal.
CALLED FROM: Run manually via `python test_short_term.py`

চালানোর নিয়ম: venv activate করা অবস্থায়, project root থেকে চালান:
    python test_short_term.py
"""

from app.memory.short_term import add_message, get_history, clear_session, get_redis_client


async def main():
    session_id = "test_session_1"

    print("--- আগের history (যদি থাকে) মুছে ফেলছি ---")
    await clear_session(session_id)

    print("\n--- কিছু message যোগ করছি ---")
    await add_message(session_id, "user", "আমার নাম Masum")
    await add_message(session_id, "assistant", "নমস্কার Masum! কী সাহায্য করতে পারি?")
    await add_message(session_id, "user", "Self-RAG নিয়ে একটা প্রশ্ন ছিল")
    print("✅ ৩টা message যোগ হয়ে গেছে")

    print("\n--- পুরো history fetch করছি ---")
    history = await get_history(session_id)
    for msg in history:
        print(f"  [{msg['role']}] {msg['content']}")

    print("\n✅ সব ঠিকমতো চলেছে!")

    # Redis connection explicitly বন্ধ করছি, নাহলে script শেষে
    # Windows-এ একটা harmless "Event loop is closed" warning আসে
    client = get_redis_client()
    await client.aclose()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())