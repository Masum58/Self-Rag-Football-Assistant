"""
WHAT: Manual end-to-end test for the compiled Self-RAG graph.
WHY: Verifies the whole pipeline — memory loading, retrieval, grading,
     generation, self-correction, and saving — works together correctly.
OUTPUT: Prints the final answer and some internal state for inspection.
CALLED FROM: Run manually via `python test_graph.py`

চালানোর নিয়ম: venv activate করা অবস্থায়, project root থেকে চালান:
    python test_graph.py
"""

import asyncio
import sys

# Windows-এ psycopg async mode এর জন্য SelectorEventLoop লাগবে
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from app.memory.long_term import init_store, close_store
from app.memory.short_term import get_redis_client
from app.graph.build_graph import build_graph


async def main():
    print("Memory store চালু হচ্ছে...")
    await init_store()

    print("Graph বানানো হচ্ছে...")
    graph = build_graph()

    initial_state = {
        "user_id": "test_user",
        "session_id": "test_session_graph",
        "question": "who scored the hand of god goal?",
        "chat_history": [],
        "long_term_context": [],
        "retrieved_docs": [],
        "documents_relevant": False,
        "generation": "",
        "generation_grounded": False,
        "generation_useful": False,
        "retrieval_retry_count": 0,
        "generation_retry_count": 0,
    }

    print(f"\n🔍 প্রশ্ন: {initial_state['question']}")
    print("Graph চালানো হচ্ছে (retrieve → grade → generate → grade)...\n")

    result = await graph.ainvoke(initial_state)

    print("--- Retrieved Documents ---")
    for doc in result["retrieved_docs"]:
        print(" -", doc)

    print("\n--- Grading Results ---")
    print("Documents relevant:", result["documents_relevant"])
    print("Generation grounded:", result["generation_grounded"])
    print("Generation useful:", result["generation_useful"])
    print("Retrieval retries used:", result["retrieval_retry_count"])
    print("Generation retries used:", result["generation_retry_count"])

    print("\n--- Final Answer ---")
    print(result["generation"])

    await close_store()

    # Redis connection explicitly বন্ধ করছি, নাহলে script শেষে
    # Windows-এ একটা harmless "Event loop is closed" warning আসে
    redis_client = get_redis_client()
    await redis_client.aclose()

    print("\n✅ Graph test সম্পূর্ণ!")


if __name__ == "__main__":
    asyncio.run(main())