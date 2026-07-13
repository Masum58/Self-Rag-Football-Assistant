"""
WHAT: Manual test script for Pinecone retrieval (query_similar).
WHY: Confirms the vectorstore returns relevant football documents for
     natural-language questions, before wiring it into the Self-RAG graph.
OUTPUT: Prints matched documents with their similarity scores.
CALLED FROM: Run manually via `python test_retrieval.py`
"""

from app.vectorstore.pinecone_client import query_similar

TEST_QUERIES = [
    "who has won the most world cups?",
    "where is the 2026 world cup being held?",
    "did messi ever win a world cup?",
]


def main():
    for query in TEST_QUERIES:
        print(f"\n🔍 Query: {query}")
        results = query_similar(query, top_k=2)
        for r in results:
            print(f"   [score={r['score']:.3f}] {r['text']}")


if __name__ == "__main__":
    main()