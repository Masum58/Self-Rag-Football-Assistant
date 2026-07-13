"""
WHAT: Pinecone connection setup + helper functions to upload (upsert) and
      query vectors.
WHY: This is our RAG vectorstore — where document chunks live as embeddings,
     so the Self-RAG graph can retrieve relevant context for a user's query.
OUTPUT: `get_index()` returns a ready-to-use Pinecone index handle.
        `upsert_documents()` uploads text+embeddings; `query_similar()`
        searches for the most relevant chunks.
CALLED FROM: app/ingestion/build_index.py (to upload sample docs)
             app/graph/nodes.py (the "retrieve" node, to fetch context)
WHY CALLED: Keeps all Pinecone-specific code in one place, so the graph
            and ingestion script never talk to the Pinecone SDK directly.

সহজ ভাষায় (Bengali):
Pinecone হলো আমাদের "ডকুমেন্ট লাইব্রেরি" — কিন্তু বই এর বদলে এখানে টেক্সটের
embedding (সংখ্যার vector) জমা থাকে। যখন কেউ প্রশ্ন করে, আমরা প্রশ্নটাকেও
vector-এ বদলে Pinecone-কে জিজ্ঞেস করি "এই vector-এর কাছাকাছি আর কোন কোন
vector আছে?" — সেগুলোই সবচেয়ে relevant ডকুমেন্ট।
"""

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from app.core.config import settings

# একই MiniLM model ব্যবহার করছি (long_term.py এর মতো) — সামঞ্জস্যপূর্ণ থাকার জন্য
_model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_DIMS = 384

_pc: Pinecone | None = None
_index_cache = None  # index handle cache — একবার তৈরি হলে আর re-check করা লাগবে না


def _get_client() -> Pinecone:
    """
    WHAT: Returns a singleton Pinecone client.
    WHY: Avoids re-authenticating every time we need to talk to Pinecone.
    OUTPUT: A Pinecone client instance.
    CALLED FROM: get_index() (below).
    WHY CALLED: Internal helper.

    সহজ ভাষায়: internal helper, সরাসরি call করার দরকার নেই।
    """
    global _pc
    if _pc is None:
        _pc = Pinecone(api_key=settings.pinecone_api_key)
    return _pc


def get_index():
    """
    WHAT: Returns a handle to our Pinecone index, creating it first if it
          doesn't exist yet.
    WHY: Every upsert/query needs an index handle. Auto-creating it means
         we don't need a separate manual "create index" step.
    OUTPUT: A Pinecone Index object, ready for upsert()/query() calls.
    CALLED FROM: upsert_documents(), query_similar() (below).
    WHY CALLED: Single source of truth for "which index are we using".

    সহজ ভাষায়: প্রথমবার চালানোর সময় Pinecone dashboard-এ গিয়ে ম্যানুয়ালি
    index বানানোর দরকার নেই — এই ফাংশনটাই চেক করে, না থাকলে বানিয়ে দেয়।

    Latency ফিক্স: আগে প্রতিটা request-এ `client.list_indexes()` কল হতো —
    একটা network round-trip, যেটা অপ্রয়োজনীয় কারণ index একবার তৈরি
    হয়ে গেলে সেটা মুছে না যাওয়া পর্যন্ত আর চেক করার দরকার নেই। এখন এই
    ফলাফল module-level ভ্যারিয়েবলে cache করা হচ্ছে, তাই প্রথম call-এর পর
    থেকে সরাসরি cached handle ফেরত যাবে, কোনো extra network call ছাড়াই।
    """
    global _index_cache
    if _index_cache is not None:
        return _index_cache

    client = _get_client()
    existing_indexes = [idx["name"] for idx in client.list_indexes()]

    if settings.pinecone_index_name not in existing_indexes:
        client.create_index(
            name=settings.pinecone_index_name,
            dimension=EMBEDDING_DIMS,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

    _index_cache = client.Index(settings.pinecone_index_name)
    return _index_cache


def upsert_documents(documents: list[dict]) -> None:
    """
    WHAT: Embeds and uploads a batch of documents into Pinecone.
    WHY: Turns raw text into searchable vectors, tagged with an id and
         metadata (so we can retrieve the original text later — Pinecone
         only stores vectors, not readable text, unless we put it in metadata).
    OUTPUT: None.
    CALLED FROM: app/ingestion/build_index.py — run once (or whenever new
                 documents are added) to populate the vectorstore.
    WHY CALLED: This is how documents get INTO Pinecone in the first place.

    Example:
        upsert_documents([
            {"id": "doc1", "text": "Brazil won the World Cup 5 times."},
            {"id": "doc2", "text": "The 2026 World Cup is hosted by USA, Canada, Mexico."},
        ])
    """
    index = get_index()
    texts = [doc["text"] for doc in documents]
    embeddings = _model.encode(texts, convert_to_numpy=True).tolist()

    vectors = [
        {
            "id": doc["id"],
            "values": embedding,
            "metadata": {"text": doc["text"]},  # আসল টেক্সট এখানে রাখছি, পরে পড়ার জন্য
        }
        for doc, embedding in zip(documents, embeddings)
    ]

    index.upsert(vectors=vectors)


def query_similar(query_text: str, top_k: int = 3) -> list[dict]:
    """
    WHAT: Finds the top_k most similar documents to a query, by meaning.
    WHY: This is the actual "retrieve" step of Self-RAG — given a user's
         question, find the most relevant chunks of context.
    OUTPUT: List of dicts like [{"text": "...", "score": 0.87}, ...],
            sorted by relevance (highest score first).
    CALLED FROM: app/graph/nodes.py — the "retrieve" node.
    WHY CALLED: Core retrieval step before the LLM generates an answer.

    Example:
        results = query_similar("who has won the most World Cups?")
        # → [{"text": "Brazil won the World Cup 5 times.", "score": 0.91}, ...]
    """
    index = get_index()
    query_embedding = _model.encode([query_text], convert_to_numpy=True).tolist()[0]

    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    return [
        {"text": match["metadata"]["text"], "score": match["score"]}
        for match in results["matches"]
    ]