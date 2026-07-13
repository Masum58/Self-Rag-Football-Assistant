# Self-RAG FastAPI

A **Self-RAG** (Self-Reflective Retrieval-Augmented Generation) chatbot built with **LangGraph** and **FastAPI** — it doesn't just retrieve and generate blindly like plain RAG, it grades its own retrieval and generation at every step, retries when something looks wrong, and falls back to live web search or clearly-labeled general knowledge when its own knowledge base doesn't have the answer.

Built as a learning/portfolio project around football & World Cup trivia, but the architecture (memory, grading, retry loops, hybrid fallback) generalizes to any RAG use case.

---

## ✨ Features

- **Self-RAG core loop** — retrieve → grade relevance (ISREL) → generate → grade groundedness & usefulness (ISSUP/ISUSE), with automatic retries when a step fails its own quality check
- **Smart Intent Routing** — a new `route_query` node acts as an intent classifier. It instantly routes casual chat or personal memory questions directly to generation, skipping expensive retrieval/search loops entirely to ensure speed and prevent hallucinations.
- **Short-term memory** — Redis-backed conversation history per session (TTL-based auto-expiry)
- **Long-term memory** — PostgreSQL + `pgvector`, using LangGraph's `AsyncPostgresStore` for semantic search over remembered facts (survives across sessions)
- **Hybrid answering** — trusted sources (memory, chat history, retrieved docs) are tried first; live **web search (Tavily)** is used as a fallback; if nothing has the answer, the model uses general knowledge but **transparently labels it** as `[General Knowledge]`
- **Document ingestion pipeline** — real `.txt` file loading + chunking (`RecursiveCharacterTextSplitter`) before embedding into Pinecone, not just hardcoded strings
- **Production-minded API layer** — structured logging, graceful error handling (Redis/Postgres/LLM failures return clean HTTP errors instead of crashing), CORS-enabled for browser clients
- **Latency-optimized** — a small/fast Groq model (`llama-3.1-8b-instant`) handles grading steps, while a larger model (`openai/gpt-oss-120b`) is reserved for final answer generation; parallelized memory reads; cached Pinecone index handle
- **Premium React Frontend** — a standalone HTML+React Single Page App (SPA) featuring a dark glassmorphism UI, markdown rendering, streaming typing animation, persistent local storage for multiple chat sessions, and live grading badges per response.

---

## 🏗️ Architecture

```
                         ┌───────────────┐
                         │ load_context  │  (Redis + Postgres, in parallel)
                         └───────┬───────┘
                                 ▼
                         ┌───────────────┐
                         │  route_query  │──(Skip Retrieval)──┐
                         └───────┬───────┘                    │
                                 ▼                            │
                         ┌───────────────┐                    │
                    ┌────│   retrieve    │◄────┐              │
                    │    └───────┬───────┘     │ retry        │
                    │            ▼             │              │
                    │    ┌───────────────┐     │              │
                    │    │grade_documents│─────┘              │
                    │    └───────┬───────┘                    │
                    │            │ relevant / retries x       │
                    │            ▼                            │
                    │    ┌───────────────┐                    │
                    └───►│  web_search   │                    │
                         └───────┬───────┘                    │
                                 ▼                            │
                         ┌───────────────┐                    │
                    ┌────│   generate    │◄───────────────────┘
                    │    └───────┬───────┘     │ retry
                    │            ▼             │
                    │    ┌────────────────┐    │
                    │    │grade_generation│────┘
                    │    └───────┬────────┘
                    │            │ grounded + useful
                    │            ▼
                    │    ┌──────────────────────┐
                    │    │extract_and_save_facts│ (writes to long-term memory)
                    │    └──────────┬───────────┘
                    │               ▼
                    │        ┌───────────┐
                    └────────│ save_turn │ (writes to short-term memory)
                             └───────────┘
```

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| API | FastAPI |
| LLM | Groq (`openai/gpt-oss-120b` for generation, `llama-3.1-8b-instant` for grading) |
| Vector store | Pinecone (serverless) |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, local, free) |
| Short-term memory | Redis |
| Long-term memory | PostgreSQL + `pgvector`, via LangGraph `AsyncPostgresStore` |
| Web search fallback | Tavily |
| Containerization | Docker Compose (Redis + Postgres) |
| Frontend (test UI) | Standalone HTML + React (CDN, no build step) |

---

## 📁 Project Structure

```
self-rag-fastapi/
├── app/
│   ├── main.py                  # FastAPI entrypoint, lifespan, CORS
│   ├── core/
│   │   ├── config.py             # Pydantic Settings (env vars)
│   │   ├── db.py                 # asyncpg connection pool
│   │   └── logging_config.py     # centralized logging setup
│   ├── routers/
│   │   ├── chat.py               # POST /chat
│   │   └── health.py             # GET /health
│   ├── memory/
│   │   ├── short_term.py         # Redis conversation history
│   │   └── long_term.py          # AsyncPostgresStore semantic memory
│   ├── vectorstore/
│   │   └── pinecone_client.py    # Pinecone init, upsert, query
│   ├── ingestion/
│   │   └── build_index.py        # loads + chunks + uploads sample docs
│   ├── graph/
│   │   ├── state.py               # GraphState schema
│   │   ├── nodes.py               # all node functions (the "brain")
│   │   └── build_graph.py         # wires nodes into a LangGraph StateGraph
│   ├── schemas/
│   │   └── chat_schema.py        # Pydantic request/response models
│   └── data/sample_docs/         # sample .txt files for ingestion
├── frontend/
│   └── self_rag_chat_tester.html # standalone test UI
├── db/migrations/                # (legacy, superseded by AsyncPostgresStore)
├── docker-compose.yml
├── requirements.txt
└── .env                          # not committed — see .env.example
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11
- Docker Desktop
- API keys: [Groq](https://console.groq.com), [Pinecone](https://app.pinecone.io), [Tavily](https://app.tavily.com)

### 1. Clone & set up the environment
```bash
git clone <your-repo-url>
cd self-rag-fastapi
py -3.11 -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

### 2. Configure environment variables
Copy `.env.example` to `.env` and fill in your keys:
```env
REDIS_URL=redis://localhost:6379/0
POSTGRES_DSN=postgresql://selfrag_user:selfrag_pass@localhost:5432/selfrag_db

PINECONE_API_KEY=your_pinecone_key
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_INDEX_NAME=self-rag-index

GROQ_API_KEY=your_groq_key
TAVILY_API_KEY=your_tavily_key
```

### 3. Start Redis & Postgres
```bash
docker compose up -d
docker compose ps
```

### 4. Ingest sample documents into Pinecone
```bash
python -m app.ingestion.build_index
```

### 5. Run the server
```bash
uvicorn app.main:app --reload
```

Visit `http://127.0.0.1:8000/docs` for the interactive Swagger UI, or open `frontend/self_rag_chat_tester.html` directly in your browser for the chat test UI.

---

## 🔌 API

### `POST /chat`
```json
{
  "user_id": "masum",
  "session_id": "session_abc123",
  "question": "who has won the most world cups?"
}
```

Response:
```json
{
  "answer": "Brazil has won the most FIFA World Cups, with five titles (1958, 1962, 1970, 1994, and 2002).",
  "documents_relevant": true,
  "generation_grounded": true,
  "generation_useful": true
}
```

### `GET /health`
Returns `{"status": "ok", "database": "connected"}` — verifies the Postgres pool is alive.

---

## 🗺️ Roadmap / Not Yet Done

- [ ] API authentication (currently open — any `user_id` can be used by anyone)
- [ ] Golden dataset + evaluation pipeline (accuracy/groundedness metrics before further deployment)
- [ ] Rate limiting
- [ ] Deduplication of ingested documents in Pinecone

---

## 📄 License

Personal/portfolio project — license not yet decided.