"""
WHAT: The /chat endpoint — the public entrypoint for talking to the
      Self-RAG system over HTTP.
WHY: Everything we built (memory, vectorstore, graph) needs a way for the
     outside world (Postman, curl, a frontend) to actually use it.
OUTPUT: POST /chat accepts a question, runs the full Self-RAG graph, and
        returns the final answer.
CALLED FROM: External clients (Postman, curl, frontend apps).
WHY CALLED: This IS the product — the rest of the app supports this endpoint.

সহজ ভাষায় (Bengali):
এতদিন যা যা বানিয়েছি (memory, vectorstore, graph), সবকিছু ভেতরে ভেতরে
কাজ করছিল, কিন্তু বাইরে থেকে কেউ এটা ব্যবহার করতে পারছিল না। এই
ফাইলটাই সেই "দরজা"।

এই ভার্সনে নতুন যোগ হয়েছে: error handling + logging। আগে কোনো node fail
করলে (Redis down, Groq timeout, ইত্যাদি) পুরো request একটা raw, অস্পষ্ট
"500 Internal Server Error" দিয়ে crash করত। এখন প্রতিটা সম্ভাব্য ব্যর্থতা
আলাদা করে ধরা হয়, ইউজারকে বোধগম্য একটা error message দেওয়া হয়, আর
terminal-এ ঠিক কী হয়েছে সেটা log হয়ে থাকে (debug করা সহজ হয়)।
"""

import time
from fastapi import APIRouter, HTTPException
import redis.exceptions
import asyncpg

from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.graph.build_graph import build_graph
from app.core.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Graph একবারই compile করে রাখছি (module load হওয়ার সময়), প্রতি request
# এ নতুন করে বানানো costly এবং অপ্রয়োজনীয়।
_graph = build_graph()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    WHAT: Runs the full Self-RAG graph for one question and returns the answer.
    WHY: This is the single entrypoint that ties memory + retrieval +
         grading + generation together into one HTTP call.
    OUTPUT: ChatResponse with the final answer and grading flags.
    CALLED FROM: POST /chat (external clients)
    WHY CALLED: Public API for the Self-RAG system.

    সহজ ভাষায়: request থেকে user_id/session_id/question নিয়ে graph এর
    initial state বানাই, graph চালাই। এখন পুরো কাজটা try/except এ মোড়ানো —
    যা কিছু ভুল হোক না কেন (Redis বন্ধ, Postgres বন্ধ, Groq/Tavily fail),
    ইউজার একটা readable error message পাবে, "500 Internal Server Error"
    এর বদলে — আর প্রতিটা ঘটনা log এ থেকে যাবে।

    Example request body:
        {
            "user_id": "masum",
            "session_id": "session_abc123",
            "question": "who has won the most world cups?"
        }
    """
    start_time = time.monotonic()
    logger.info(f"[{request.session_id}] Question from {request.user_id}: {request.question!r}")

    initial_state = {
        "user_id": request.user_id,
        "session_id": request.session_id,
        "question": request.question,
        "chat_history": [],
        "long_term_context": [],
        "retrieved_docs": [],
        "web_search_results": [],
        "documents_relevant": False,
        "generation": "",
        "generation_grounded": False,
        "generation_useful": False,
        "retrieval_retry_count": 0,
        "generation_retry_count": 0,
    }

    try:
        result = await _graph.ainvoke(initial_state)

    except (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError) as e:
        # সহজ ভাষায়: Redis (short-term memory) এর সাথে connect করতে না
        # পারলে এখানে ধরা পড়বে — সাধারণত Docker container বন্ধ থাকলে হয়।
        logger.error(f"[{request.session_id}] Redis connection failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Short-term memory (Redis) is unavailable right now. "
                   "Please check that the Redis container is running and try again.",
        )

    except (asyncpg.PostgresError, ConnectionRefusedError, OSError) as e:
        # সহজ ভাষায়: Postgres (long-term memory) এর সাথে সমস্যা হলে এখানে ধরা পড়বে।
        logger.error(f"[{request.session_id}] Database error: {e}")
        raise HTTPException(
            status_code=503,
            detail="Long-term memory (database) is unavailable right now. "
                   "Please check that the Postgres container is running and try again.",
        )

    except Exception as e:
        # সহজ ভাষায়: বাকি যেকোনো অপ্রত্যাশিত error (Groq API fail, Tavily fail,
        # JSON parsing error ইত্যাদি) এখানে catch হবে — যাতে সার্ভার crash না করে
        # raw traceback ইউজারকে না দেখিয়ে, একটা সাধারণ ক্ষমা-চাওয়া message দেয়।
        logger.error(f"[{request.session_id}] Unexpected error in graph execution: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Something went wrong while generating a response. Please try again.",
        )

    elapsed = time.monotonic() - start_time
    logger.info(
        f"[{request.session_id}] Done in {elapsed:.2f}s — "
        f"relevant={result['documents_relevant']}, "
        f"grounded={result['generation_grounded']}, "
        f"useful={result['generation_useful']}"
    )

    return ChatResponse(
        answer=result["generation"],
        documents_relevant=result["documents_relevant"],
        generation_grounded=result["generation_grounded"],
        generation_useful=result["generation_useful"],
    )