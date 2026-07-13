"""
WHAT: The FastAPI application entrypoint.
WHY: This is the single file that ties everything together — it creates the
     app, wires up routers, and manages startup/shutdown of shared resources
     like the Postgres connection pool and the long-term memory store.
OUTPUT: A running FastAPI app (`uvicorn app.main:app`).
CALLED FROM: Run via `uvicorn app.main:app --reload` from the project root.
WHY CALLED: This is the actual server process the user starts.

সহজ ভাষায় (Bengali):
এই ফাইলটাই আপনার পুরো অ্যাপের "মেইন সুইচ"। এখানে ৩টা কাজ হয় —
(১) FastAPI app বানানো, (২) app চালু/বন্ধ হওয়ার সময় Postgres pool আর
memory store খোলা/বন্ধ করা, (৩) routers (যেমন health check) app-এর সাথে
যুক্ত করা। `uvicorn app.main:app --reload` চালালে এই ফাইলটাই run হয়।
"""

from contextlib import asynccontextmanager
# ↑ সহজ ভাষায়: এটা একটা Python built-in টুল, যা দিয়ে "শুরুতে এটা করো,
# শেষে ওটা করো" — এই ধরনের startup/shutdown ব্লক লেখা যায়।
# Example: with asynccontextmanager, আমরা lifespan() ফাংশনে
# "yield" এর আগেরটুকু startup, পরেরটুকু shutdown হিসেবে ধরা হয়।

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# ↑ সহজ ভাষায়: CORSMiddleware ছাড়া browser থেকে চলা কোনো ওয়েবসাইট/UI
# (যেমন আমাদের React chat UI) আমাদের FastAPI server-কে call করতে পারবে না —
# browser নিরাপত্তার কারণে block করে দেয়। এটা যোগ করলে allow করা হবে।
# ↑ সহজ ভাষায়: FastAPI ফ্রেমওয়ার্কের মূল ক্লাস। এটা দিয়েই আমাদের
# ওয়েব অ্যাপ (server) বানানো হয়।
# Example: app = FastAPI() → একটা খালি FastAPI app তৈরি হলো।

from app.core.db import connect_to_db, close_db
# ↑ সহজ ভাষায়: আগে বানানো Postgres connection pool খোলা/বন্ধ করার ফাংশন।
# connect_to_db() → pool খোলে (app চালুর সময়)
# close_db()      → pool বন্ধ করে (app বন্ধের সময়)

from app.core.logging_config import setup_logging, get_logger
# ↑ সহজ ভাষায়: প্রতিটা module এ consistent logging ব্যবহারের জন্য।
# setup_logging() → app চালুর সময় একবার logging format ঠিক করে
# get_logger()    → এই ফাইলের নিজস্ব logger বানানোর জন্য

logger = get_logger(__name__)

from app.memory.long_term import init_store, close_store
# ↑ সহজ ভাষায়: long-term memory store (AsyncPostgresStore) খোলা/বন্ধ করার ফাংশন।
# init_store()  → memory store চালু করে, table বানায় (app চালুর সময়)
# close_store() → memory store বন্ধ করে (app বন্ধের সময়)

from app.routers import health, chat
# ↑ সহজ ভাষায়: /health এর পাশাপাশি এখন /chat endpoint ও import করছি।


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    WHAT: FastAPI's modern startup/shutdown hook (replaces old @app.on_event).
    WHY: We need the Postgres pool AND the long-term memory store open BEFORE
         the first request arrives, and closed cleanly when the server stops.
    OUTPUT: Yields control back to FastAPI while the app runs; runs cleanup after.
    CALLED FROM: FastAPI itself, automatically, when the app starts/stops.
    WHY CALLED: Ensures connect_to_db()/init_store() run once at boot, and
                close_db()/close_store() run once at exit.

    সহজ ভাষায়: এই ফাংশনটা FastAPI নিজে থেকেই কল করে — আপনাকে ম্যানুয়ালি
    কল করতে হয় না। `yield` এর আগে যা লেখা আছে, সেটা "app চালু হওয়ার সময়"
    একবার চলে। `yield` এর পরে যা লেখা, সেটা "app বন্ধ হওয়ার সময়" চলে।

    Example (মানসিক ছবি):
        server চালু হলো  → connect_to_db(), init_store() রান হলো
        ... server চলতে থাকল, request আসতে থাকল ...
        Ctrl+C দিয়ে বন্ধ করলেন → close_store(), close_db() রান হলো
    """
    # Startup — server চালু হওয়ার সাথে সাথে এই লাইনগুলো একবার চলবে
    setup_logging()          # সবার আগে logging configure করছি
    logger.info("Starting Self-RAG FastAPI...")
    await connect_to_db()   # Postgres connection pool খুলছি
    await init_store()      # Long-term memory store খুলছি + table বানাচ্ছি
    logger.info("Startup complete — Postgres pool and memory store ready.")
    yield                   # ← এখান থেকে server আসল request handle করা শুরু করে
    # Shutdown — server বন্ধ হওয়ার সময় (Ctrl+C ইত্যাদি) এই লাইনগুলো চলবে
    logger.info("Shutting down...")
    await close_store()     # memory store বন্ধ করছি
    await close_db()        # Postgres connection pool বন্ধ করছি


app = FastAPI(title="Self-RAG FastAPI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Development-এ সব origin allow করছি; production-এ নির্দিষ্ট domain দেওয়া ভালো
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ↑ সহজ ভাষায়: এখন যেকোনো browser-based frontend (localhost এ চলা React
# artifact সহ) আমাদের /chat, /health endpoint call করতে পারবে।
# ↑ সহজ ভাষায়: এখানেই আসল FastAPI app বানানো হলো।
# title="Self-RAG FastAPI" → শুধু docs page (Swagger UI) এ নাম হিসেবে দেখাবে
# lifespan=lifespan → উপরে বানানো startup/shutdown লজিকটা এই app-এর
#                     সাথে যুক্ত করে দিলাম
# Example: browser এ http://127.0.0.1:8000/docs খুললে এই title-টাই দেখবেন।

# Routers
# Routers
app.include_router(health.router)
app.include_router(chat.router)
# ↑ সহজ ভাষায়: /chat endpoint টাও এখন app-এর সাথে যুক্ত হলো।
# Example: এখন POST http://127.0.0.1:8000/chat কল করলে Self-RAG graph চলবে।