"""
WHAT: Wires all the node functions from nodes.py into an actual LangGraph
      StateGraph, with conditional edges that implement Self-RAG's
      self-correcting loops.
WHY: Individual node functions alone don't do anything — this file defines
     the ORDER they run in, and the DECISIONS that route between them
     (e.g. "if documents aren't relevant, go back and retrieve again").
OUTPUT: `build_graph()` returns a compiled, runnable LangGraph app.
CALLED FROM: app/routers/chat.py (to actually run the graph on a user's
             question)
WHY CALLED: This compiled graph IS the Self-RAG system — everything else
            was preparation for this.

সহজ ভাষায় (Bengali):
nodes.py এ আমরা আলাদা আলাদা "স্টেশন" বানিয়েছিলাম (retrieve, grade,
generate...)। এই ফাইলে আমরা ঠিক করছি স্টেশনগুলো কোন ক্রমে চলবে, আর
কোথায় কোথায় "যদি এমন হয়, তাহলে ওই স্টেশনে ফিরে যাও" — এই লুপগুলো বসবে।
এটাই আসল Self-RAG behavior তৈরি করে।
"""

from langgraph.graph import StateGraph, END

from app.graph.state import GraphState
from app.graph.nodes import (
    load_context,
    retrieve,
    grade_documents,
    web_search,
    generate,
    grade_generation,
    extract_and_save_facts,
    save_turn,
    MAX_RETRIEVAL_RETRIES,
    MAX_GENERATION_RETRIES,
)


# ---------------------------------------------------------------------------
# Decision functions — এগুলো কোনো state বদলায় না, শুধু "পরে কোন node এ
# যাবে" সেটা একটা string হিসেবে ফেরত দেয়। এগুলোই conditional edge এ ব্যবহার হবে।
# ---------------------------------------------------------------------------

def decide_after_grading_documents(state: GraphState) -> str:
    """
    WHAT: Decides whether to proceed to generate() or loop back to retrieve().
    WHY: If documents weren't relevant, retrying with a fresh retrieval
         might work better than generating from bad context. But we cap
         retries so it can't loop forever.
    OUTPUT: "generate" or "retrieve" — the name of the next node to run.
    CALLED FROM: LangGraph itself, via add_conditional_edges() below.
    WHY CALLED: Implements the ISREL retry loop.

    সহজ ভাষায়: ডকুমেন্ট relevant হলে generate এ যাও। relevant না হলে,
    এখনো retry বাকি থাকলে আবার retrieve করো। retry শেষ হয়ে গেলে বাধ্য
    হয়ে generate এ যাও (যা আছে তা দিয়েই উত্তর দেওয়ার চেষ্টা করো)।

    Example:
        documents_relevant=False, retrieval_retry_count=1, MAX=2
        → "retrieve" (আরেকবার চেষ্টা করার সুযোগ আছে)

        documents_relevant=False, retrieval_retry_count=2, MAX=2
        → "generate" (retry শেষ, যা আছে তা দিয়েই এগোও)
    """
    if state["documents_relevant"]:
        return "generate"
    if state["retrieval_retry_count"] < MAX_RETRIEVAL_RETRIES:
        return "retrieve"
    return "web_search"  # retry ফুরিয়ে গেছে, Tavily দিয়ে web search করো


def decide_after_grading_generation(state: GraphState) -> str:
    """
    WHAT: Decides whether the generation is good enough to save & return,
          or whether to retry generation (or retrieval, if docs seem to
          be the real problem).
    WHY: This is Self-RAG's core self-correction — a hallucinated or
          unhelpful answer shouldn't be handed back to the user.
    OUTPUT: "extract_and_save_facts" (accept), "generate" (retry generation),
            or "retrieve" (retry retrieval, as a last resort).
    CALLED FROM: LangGraph itself, via add_conditional_edges() below.
    WHY CALLED: Implements the ISSUP + ISUSE retry loop.

    সহজ ভাষায়: answer যদি grounded (hallucination না) এবং useful দুটোই
    হয়, তাহলে গ্রহণ করে নাও (fact extraction + save turn এর দিকে এগোও)। না হলে, retry বাকি থাকলে আবার generate করো।
    যদি generation বারবার fail করে (retry শেষ), তাহলে ধরে নাও আসলে
    retrieved docs-ই যথেষ্ট ভালো ছিল না — retrieve থেকে আবার শুরু করো,
    যদি সেই retry-ও বাকি থাকে। সব retry শেষ হয়ে গেলে যা আছে তা দিয়েই
    মেনে নাও (infinite loop এড়াতে)।
    """
    if state["generation_grounded"] and state["generation_useful"]:
        return "extract_and_save_facts"

    if state["generation_retry_count"] < MAX_GENERATION_RETRIES:
        return "generate"

    if state["retrieval_retry_count"] < MAX_RETRIEVAL_RETRIES:
        return "retrieve"

    return "extract_and_save_facts"  # সব retry ফুরিয়ে গেছে, যা আছে তা দিয়েই মেনে নাও


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """
    WHAT: Constructs and compiles the full Self-RAG StateGraph.
    WHY: This is the one function the rest of the app calls to get a
         ready-to-run graph — it hides all the wiring details.
    OUTPUT: A compiled LangGraph app with an `.ainvoke(initial_state)` method.
    CALLED FROM: app/routers/chat.py, typically once at startup (the
                 compiled graph can be reused across requests).
    WHY CALLED: Produces the runnable Self-RAG system.

    সহজ ভাষায়: এই ফাংশনটা call করলেই পুরো Self-RAG গ্রাফ রেডি হয়ে যাবে,
    ব্যবহার করার জন্য। এটা normally app চালুর সময় একবার কল হবে, বারবার
    না (graph বানানো একটু costly, reuse করাই ভালো)।

    Example:
        graph = build_graph()
        result = await graph.ainvoke({
            "user_id": "masum",
            "session_id": "session_abc",
            "question": "who has won the most world cups?",
            "chat_history": [],
            "long_term_context": [],
            "retrieved_docs": [],
            "documents_relevant": False,
            "generation": "",
            "generation_grounded": False,
            "generation_useful": False,
            "retrieval_retry_count": 0,
            "generation_retry_count": 0,
        })
        print(result["generation"])
    """
    graph = StateGraph(GraphState)

    # --- Node গুলো যোগ করা ---
    graph.add_node("load_context", load_context)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_documents", grade_documents)
    graph.add_node("web_search", web_search)
    graph.add_node("generate", generate)
    graph.add_node("grade_generation", grade_generation)
    graph.add_node("extract_and_save_facts", extract_and_save_facts)
    graph.add_node("save_turn", save_turn)

    # --- সরল (unconditional) edge — সবসময় এই ক্রমেই চলবে ---
    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "retrieve")
    graph.add_edge("retrieve", "grade_documents")
    graph.add_edge("web_search", "generate")  # web search শেষ হলে generate এ যাও
    graph.add_edge("generate", "grade_generation")
    graph.add_edge("extract_and_save_facts", "save_turn")
    graph.add_edge("save_turn", END)

    # --- Conditional edge — decision function অনুযায়ী branch হবে ---
    graph.add_conditional_edges(
        "grade_documents",
        decide_after_grading_documents,
        {
            "generate": "generate",
            "retrieve": "retrieve",
            "web_search": "web_search",
        },
    )

    graph.add_conditional_edges(
        "grade_generation",
        decide_after_grading_generation,
        {
            "extract_and_save_facts": "extract_and_save_facts",
            "generate": "generate",
            "retrieve": "retrieve",
        },
    )

    return graph.compile()