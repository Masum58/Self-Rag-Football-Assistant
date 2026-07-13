"""
WHAT: Defines the "state" that flows through every node of our Self-RAG
      LangGraph — i.e. the shared data structure every node reads from and
      writes to.
WHY: LangGraph nodes are just functions that take state in and return
     partial state updates. We need one consistent schema so every node
     agrees on what fields exist and what they mean.
OUTPUT: A TypedDict (`GraphState`) used to type-hint and validate state
        across the graph.
CALLED FROM: app/graph/nodes.py (every node function's input/output type)
             app/graph/build_graph.py (when constructing the StateGraph)
WHY CALLED: Single source of truth for "what data exists at each step of
            the Self-RAG flow".

সহজ ভাষায় (Bengali):
GraphState-কে ভাবুন একটা "ট্রে" হিসেবে, যেটা প্রতিটা node (retrieve, grade,
generate...) এর হাত দিয়ে যায়। প্রতিটা node ট্রে থেকে কিছু জিনিস পড়ে, আর
কিছু জিনিস নতুন করে বসিয়ে পরের node-এর কাছে পাঠিয়ে দেয়। এই ফাইলে আমরা
ঠিক করছি ট্রে-তে কী কী "স্লট" থাকবে।
"""

from typing import TypedDict, Optional


class GraphState(TypedDict):
    # --- ইউজার শনাক্তকরণ (কার সাথে কথা হচ্ছে) ---
    user_id: str
    # সহজ ভাষায়: কোন user এই conversation চালাচ্ছে — long-term memory
    # (Postgres) থেকে facts আনতে এই id লাগবে।
    # Example: "masum"

    session_id: str
    # সহজ ভাষায়: এই নির্দিষ্ট চ্যাট সেশনের id — short-term memory (Redis)
    # থেকে এই সেশনের history আনতে লাগবে।
    # Example: "session_abc123"

    # --- মূল প্রশ্ন এবং তার ইতিহাস ---
    question: str
    # সহজ ভাষায়: ইউজার এখন যা জিজ্ঞেস করেছে।
    # Example: "who has won the most world cups?"

    chat_history: list[dict]
    # সহজ ভাষায়: Redis থেকে আনা আগের কথোপকথন (short-term memory)।
    # Example: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]

    long_term_context: list[str]
    # সহজ ভাষায়: Postgres/PostgresStore থেকে semantic search করে আনা
    # relevant memory (long-term memory)।
    # Example: ["User's favorite football team is Brazil"]

    # --- Routing ---
    needs_retrieval: Optional[bool]
    # সহজ ভাষায়: ইউজারের প্রশ্নটি কি শুধু casual chat নাকি কোনো factual information দরকার?

    # --- Retrieval (Pinecone থেকে) ---
    retrieved_docs: list[str]
    # সহজ ভাষায়: Pinecone থেকে retrieve করা document text গুলো।
    # Example: ["Brazil has won the FIFA World Cup a record 5 times..."]

    web_search_results: list[str]
    # সহজ ভাষায়: Pinecone/memory/history কোথাও উত্তর না পেলে, Tavily দিয়ে
    # করা live web search-এর ফলাফল। Hybrid mode এর "৪র্থ source"।
    # Example: ["France won the 2018 FIFA World Cup, defeating Croatia 4-2."]

    # --- Self-RAG grading ফলাফল ---
    documents_relevant: Optional[bool]
    # সহজ ভাষায়: ISREL check এর ফলাফল — retrieved_docs আসলেই প্রশ্নের
    # সাথে relevant কিনা (grade_documents node এটা সেট করবে)।

    generation: str
    # সহজ ভাষায়: LLM এর তৈরি করা answer।

    generation_grounded: Optional[bool]
    # সহজ ভাষায়: ISSUP check এর ফলাফল — generation টা আসলেই retrieved_docs
    # দিয়ে supported (hallucination না) কিনা।

    generation_useful: Optional[bool]
    # সহজ ভাষায়: ISUSE check এর ফলাফল — generation টা আসলেই প্রশ্নের
    # উত্তর দিয়েছে কিনা (নাকি topic থেকে সরে গেছে)।

    # --- Loop/retry নিয়ন্ত্রণ (infinite loop আটকানোর জন্য) ---
    retrieval_retry_count: int
    # সহজ ভাষায়: retrieve → grade_documents loop কতবার চলেছে, সেটার counter।
    # একটা max limit (যেমন ৩) দিয়ে আটকাবো, নাহলে infinite loop হতে পারে।

    generation_retry_count: int
    # সহজ ভাষায়: generate → grade_generation loop কতবার চলেছে, সেটার counter।