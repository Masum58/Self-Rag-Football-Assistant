"""
WHAT: The actual Self-RAG logic — each function here is one "node" in our
      LangGraph graph. A node reads some fields from GraphState, does one
      job, and returns the fields it changed.
WHY: This is where retrieval, grading, and generation actually happen —
     everything before this (state.py, pinecone_client.py, memory files)
     was just plumbing. This file is the "brain".
OUTPUT: Each function returns a partial dict that LangGraph merges into
        the overall GraphState.
CALLED FROM: app/graph/build_graph.py (wires these functions into a graph
             with edges between them)
WHY CALLED: build_graph.py needs actual functions to attach to each node
            name in the graph.

সহজ ভাষায় (Bengali):
এই ফাইলে কয়েকটা ফাংশন আছে, প্রতিটা একটা করে "স্টেশন" — request একটার পর
একটা স্টেশন দিয়ে যাবে (build_graph.py পরে ঠিক করবে কোন স্টেশনের পর
কোনটা)। প্রতিটা ফাংশন GraphState থেকে দরকারি জিনিস পড়ে, নিজের কাজ
শেষে যা বদলেছে সেটুকু ফেরত দেয়।
"""

import asyncio
import json
from langchain_groq import ChatGroq
from tavily import TavilyClient

from app.core.config import settings
from app.graph.state import GraphState
from app.vectorstore.pinecone_client import query_similar
from app.memory.short_term import get_history, add_message
from app.memory.long_term import search_memory, save_memory

# Groq LLM (বড় মডেল) — শুধু generate() এ ব্যবহার হবে, যেখানে answer quality
# সবচেয়ে গুরুত্বপূর্ণ। openai/gpt-oss-120b ব্যবহার করছি কারণ
# llama-3.3-70b-versatile deprecated হয়ে গেছে।
llm = ChatGroq(model="openai/gpt-oss-120b", api_key=settings.groq_api_key, temperature=0)

# Grading LLM (ছোট, দ্রুত মডেল) — grade_documents, grade_generation,
# extract_and_save_facts এর মতো simple yes/no classification কাজে ব্যবহার হবে।
# llama-3.1-8b-instant Groq-এ অনেক গুণ দ্রুত (500+ tokens/sec), আর এই সহজ
# কাজগুলোর জন্য বড় মডেল লাগে না। এটাই মূল latency fix।
grading_llm = ChatGroq(model="llama-3.1-8b-instant", api_key=settings.groq_api_key, temperature=0)

# Tavily client — web search fallback (Pinecone-এ কিছু না পেলে ব্যবহার হবে)।
tavily = TavilyClient(api_key=settings.tavily_api_key)

MAX_RETRIEVAL_RETRIES = 2
MAX_GENERATION_RETRIES = 2


# ---------------------------------------------------------------------------
# Node 0: load_context — memory থেকে দরকারি context জোগাড় করা
# ---------------------------------------------------------------------------

async def load_context(state: GraphState) -> dict:
    """
    WHAT: Loads short-term chat history (Redis) and long-term relevant
          memories (Postgres semantic search) before anything else happens.
    WHY: The graph needs conversational + personal context ready before
         retrieval/generation, so answers feel personalized and continuous.
    OUTPUT: {"chat_history": [...], "long_term_context": [...]}
    CALLED FROM: build_graph.py — the very first node in the graph.
    WHY CALLED: Sets up context that later nodes (generate) will use.

    সহজ ভাষায়: এটাই graph এর প্রথম node। এখানে Redis থেকে "এতক্ষণ কী কথা
    হয়েছে" আর Postgres থেকে "এই user সম্পর্কে কী কী প্রাসঙ্গিক তথ্য জানি"
    — দুটোই টেনে আনা হয়।

    Latency ফিক্স: আগে এই দুটো call একে একে (sequential) হতো। এখন
    `asyncio.gather()` দিয়ে দুটোই একসাথে (parallel) চলে, যাতে মোট সময়
    লাগে দুটোর মধ্যে যেটা বেশি সময় নেয় সেটার সমান, দুটোর যোগফল না।
    """
    history, relevant_memories = await asyncio.gather(
        get_history(state["session_id"]),
        search_memory(state["user_id"], state["question"], limit=3),
    )

    return {
        "chat_history": history,
        "long_term_context": relevant_memories,
    }


# ---------------------------------------------------------------------------
# Node 0.5: route_query — Intent classification
# ---------------------------------------------------------------------------

async def route_query(state: GraphState) -> dict:
    """
    WHAT: Decides if the user's message needs external facts (retrieval) or
          if it is just casual conversation / asks about the user themselves.
    WHY: Running retrieval and web search for "hi" or "thanks" is slow and
         causes hallucinations.
    """
    prompt = f"""You are a routing assistant for a Football Assistant chatbot.
Analyze the user's message and decide if it requires searching an external knowledge base or the web for factual football information.

Message: "{state['question']}"

Rules:
- If the message is casual chat (e.g., "hi", "wow great", "thanks", "hello") -> NO retrieval needed.
- If the message asks about the user themselves or the conversation history (e.g., "what is my name?", "how do you know") -> NO retrieval needed (it will be answered from memory).
- If the message asks a factual football question (e.g., "who won the world cup?") -> YES retrieval needed.

Respond with ONLY a JSON object: {{"needs_retrieval": true}} or {{"needs_retrieval": false}}"""

    response = await grading_llm.ainvoke(prompt)
    try:
        result = json.loads(response.content.strip())
        needs_retrieval = result.get("needs_retrieval", True)
    except:
        needs_retrieval = True

    return {"needs_retrieval": needs_retrieval}


# ---------------------------------------------------------------------------
# Node 1: retrieve — Pinecone থেকে ডকুমেন্ট খুঁজে আনা
# ---------------------------------------------------------------------------

async def retrieve(state: GraphState) -> dict:
    """
    WHAT: Queries Pinecone for documents relevant to the current question.
    WHY: This is the "R" (Retrieval) in RAG — before generating an answer,
         find supporting context.
    OUTPUT: {"retrieved_docs": [...]}
    CALLED FROM: build_graph.py — first retrieval step, and again if
                 grade_documents decides the docs weren't relevant enough.
    WHY CALLED: Supplies context for the generate() node.

    সহজ ভাষায়: প্রশ্নটা নিয়ে Pinecone-এ গিয়ে সবচেয়ে কাছাকাছি অর্থের
    ডকুমেন্ট খুঁজে আনা।
    """
    results = query_similar(state["question"], top_k=3)
    docs = [r["text"] for r in results]
    return {"retrieved_docs": docs, "web_search_results": []}


# ---------------------------------------------------------------------------
# Node 2: grade_documents — ISREL check (retrieved docs কি relevant?)
# ---------------------------------------------------------------------------

async def grade_documents(state: GraphState) -> dict:
    """
    WHAT: Asks the LLM to judge whether the retrieved docs actually help
          answer the question.
    WHY: This is Self-RAG's ISREL check — plain RAG would use retrieved
         docs blindly, even if they're irrelevant. We verify first.
    OUTPUT: {"documents_relevant": bool, "retrieval_retry_count": int}
    CALLED FROM: build_graph.py, right after retrieve().
    WHY CALLED: Decides whether to proceed to generate() or loop back to
                retrieve() again (handled by a conditional edge later).

    সহজ ভাষায়: retrieve() যে ডকুমেন্ট এনেছে, সেগুলো আসলেই প্রশ্নের কাজে
    লাগবে কিনা তা LLM কে জিজ্ঞেস করে যাচাই করা। "হ্যাঁ/না" উত্তর নিয়ে
    documents_relevant ঠিক করি।

    Latency ফিক্স: এখন এই grading ছোট/দ্রুত মডেল (grading_llm) দিয়ে হচ্ছে,
    বড় generate() মডেলের বদলে — এই simple yes/no সিদ্ধান্তে বড় মডেল লাগে না।
    """
    docs_text = "\n".join(f"- {d}" for d in state["retrieved_docs"])

    prompt = f"""You are grading whether retrieved documents are relevant to a question.

Question: {state['question']}

Retrieved documents:
{docs_text}

Are these documents relevant enough to help answer the question?
Respond with ONLY a JSON object, no other text: {{"relevant": true}} or {{"relevant": false}}"""

    response = await grading_llm.ainvoke(prompt)
    result = json.loads(response.content.strip())

    return {
        "documents_relevant": result["relevant"],
        "retrieval_retry_count": state.get("retrieval_retry_count", 0) + 1,
        "web_search_results": state.get("web_search_results", []),
    }


# ---------------------------------------------------------------------------
# Node 3: generate — Groq দিয়ে আসল answer তৈরি করা
# ---------------------------------------------------------------------------

async def generate(state: GraphState) -> dict:
    """
    WHAT: Generates an answer using the question, retrieved docs, chat
          history, and long-term memory as context.
    WHY: This is the "G" (Generation) in RAG — produces the actual reply.
    OUTPUT: {"generation": str}
    CALLED FROM: build_graph.py, after documents are confirmed relevant.
    WHY CALLED: Produces the answer that grade_generation will then check.

    সহজ ভাষায়: সব context (retrieved docs + memory + chat history + web
    search) একসাথে করে LLM কে দিয়ে আসল উত্তর লেখানো। এখানে বড় মডেল
    (llm) ই থাকছে, কারণ চূড়ান্ত answer-এর quality সবচেয়ে গুরুত্বপূর্ণ।
    """
    docs_text = "\n".join(f"- {d}" for d in state["retrieved_docs"])
    memory_text = "\n".join(f"- {m}" for m in state.get("long_term_context", []))
    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in state.get("chat_history", [])
    )
    web_text = "\n".join(f"- {r}" for r in state.get("web_search_results", []))

    prompt = f"""Answer the user's question using whichever of the following FOUR sources
is actually relevant. All four are valid sources of truth — do not favor one over another.

1. What we remember about this user (long-term memory):
{memory_text or "(nothing remembered yet)"}

2. This conversation so far (recent chat history):
{history_text or "(no prior messages in this session)"}

3. Retrieved reference documents (knowledge base):
{docs_text or "(no documents retrieved)"}

4. Live web search results (fetched in real-time):
{web_text or "(no web search was performed)"}

Question: {state['question']}

Instructions:
- If the question is about the user themselves (name, preferences, something they said
  earlier), look in sources 1 and 2 FIRST — that is where personal info lives.
- If sources 3 or 4 contain a direct answer, use them.
- If NONE of the four sources answer the question, use your own general knowledge
  BUT you MUST begin that part of your answer with the prefix:
  "[General Knowledge] "
  Example: "[General Knowledge] France won the 2018 FIFA World Cup, defeating Croatia 4-2."
- Never say "I don't have that information" — always try to help, with the appropriate
  label if you are drawing on general knowledge instead of the sources above.
- Answer clearly and concisely."""

    response = await llm.ainvoke(prompt)
    return {"generation": response.content.strip()}


# ---------------------------------------------------------------------------
# Node 3b: web_search — Tavily দিয়ে live internet search (Pinecone fallback)
# ---------------------------------------------------------------------------

async def web_search(state: GraphState) -> dict:
    """
    WHAT: Performs a live web search using Tavily when Pinecone retrieval
          fails to find relevant documents after max retries.
    WHY: Pinecone শুধু ingested document জানে — real-time বা out-of-scope
         প্রশ্নের জন্য (যেমন "who won the 2026 World Cup") web search দরকার।
    OUTPUT: {"web_search_results": [...], "retrieved_docs": [...]}
    CALLED FROM: build_graph.py — conditional edge from grade_documents,
                 যখন documents_relevant=False এবং retry শেষ।
    WHY CALLED: Fills web_search_results so generate() has live context.

    সহজ ভাষায়: Pinecone-এ কিছু না পেলে এই node Tavily API দিয়ে internet
    থেকে সরাসরি তথ্য খুঁজে আনে। এটা "শেষ চেষ্টা" — যদি এখানেও না পাওয়া
    যায়, generate() general knowledge + explicit label দিয়ে উত্তর দেবে।
    """
    try:
        response = tavily.search(
            query=state["question"],
            search_depth="basic",
            max_results=3,
        )
        results = [
            r["content"]
            for r in response.get("results", [])
            if r.get("content")
        ]
    except Exception:
        results = []

    return {
        "web_search_results": results,
        # web search এর পর retrieved_docs unchanged রাখি (generate() উভয় দেখবে)
        "retrieved_docs": state.get("retrieved_docs", []),
    }


# ---------------------------------------------------------------------------
# Node 4: grade_generation — ISSUP + ISUSE check
# ---------------------------------------------------------------------------

async def grade_generation(state: GraphState) -> dict:
    """
    WHAT: Asks the LLM to check two things about its own generation:
          (1) is it grounded in the available context (not hallucinated)?
          (2) does it actually answer the question (is it useful)?
    WHY: This is Self-RAG's ISSUP + ISUSE checks — the self-correcting
         part that plain RAG doesn't do.
    OUTPUT: {"generation_grounded": bool, "generation_useful": bool,
             "generation_retry_count": int}
    CALLED FROM: build_graph.py, right after generate().
    WHY CALLED: Decides whether to accept the answer or loop back to
                generate() (or even retrieve()) again.

    সহজ ভাষায়: LLM নিজেই নিজের লেখা answer চেক করে — "আমি কি বানিয়ে
    বলেছি (hallucination), নাকি context থেকেই বলেছি?" আর "আমি কি আসলেই
    প্রশ্নের জবাব দিয়েছি?"

    Latency ফিক্স: এই grading ও এখন দ্রুত মডেল (grading_llm) দিয়ে হচ্ছে।
    """
    # Casual chat bypass: no need to grade "hi" or "thanks" against strict context
    if state.get("needs_retrieval") is False:
        return {
            "generation_grounded": True,
            "generation_useful": True,
            "generation_retry_count": state.get("generation_retry_count", 0),
        }

    docs_text = "\n".join(f"- {d}" for d in state["retrieved_docs"])
    memory_text = "\n".join(f"- {m}" for m in state.get("long_term_context", []))
    history_text = "\n".join(
        f"{m['role']}: {m['content']}" for m in state.get("chat_history", [])
    )
    web_text = "\n".join(f"- {r}" for r in state.get("web_search_results", []))

    prompt = f"""Check the following generated answer against two criteria.

Available context (any of these FOUR sources count as valid grounding):
1. Long-term memory about the user: {memory_text or "(none)"}
2. Recent conversation history: {history_text or "(none)"}
3. Retrieved reference documents: {docs_text or "(none)"}
4. Live web search results: {web_text or "(none)"}

Question: {state['question']}

Generated answer: {state['generation']}

Grounding rules (read carefully):
- "grounded" = TRUE if the answer is supported by ANY of the four sources above.
- "grounded" = TRUE also if the answer visibly starts with the prefix "[General Knowledge]"
  — this means the model transparently disclosed it used general knowledge, which we
  explicitly allow and treat as honest, non-hallucinated output.
- "grounded" = FALSE only if the answer makes factual claims that appear in NONE of
  the four sources above AND does NOT carry the "[General Knowledge]" prefix.
  Unlabeled general knowledge = potential hallucination = not grounded.

Useful rules:
- "useful" = TRUE if the answer actually addresses the question asked.
- "useful" = FALSE if the answer is off-topic or is a refusal like "I don't know"
  when information was clearly available in the sources.

Respond with ONLY a JSON object, no other text:
{{"grounded": true, "useful": true}}"""

    response = await grading_llm.ainvoke(prompt)
    result = json.loads(response.content.strip())

    return {
        "generation_grounded": result["grounded"],
        "generation_useful": result["useful"],
        "generation_retry_count": state.get("generation_retry_count", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node 5: extract_and_save_facts — এই turn থেকে নতুন personal fact বের করে
#          long-term memory (Postgres) এ সেভ করা
# ---------------------------------------------------------------------------

async def extract_and_save_facts(state: GraphState) -> dict:
    """
    WHAT: Asks the LLM whether the user's message contains a durable personal
          fact (name, preference, etc.), and if so, saves it via save_memory().
    WHY: Without this, long-term memory (Postgres) never actually gets
         written to — search_memory() would always return empty results,
         even after a user says "my name is masum".
    OUTPUT: {} (no state fields change, just a side effect — writes to Postgres)
    CALLED FROM: build_graph.py, after a good generation is accepted, before
                 save_turn.
    WHY CALLED: This is what makes long-term memory ACTUALLY persistent.

    সহজ ভাষায়: প্রতিটা turn শেষে LLM কে জিজ্ঞেস করা হয় "এই কথায় কি user
    সম্পর্কে কোনো স্থায়ী তথ্য (নাম, পছন্দ ইত্যাদি) আছে?" থাকলে সেটা
    Postgres-এ সেভ করে রাখা হয়, যাতে ভবিষ্যতে অন্য session-এও মনে থাকে।

    Latency ফিক্স: এটাও দ্রুত মডেল (grading_llm) দিয়ে হচ্ছে এখন।

    Example:
        user বলল "my name is masum"
        → LLM detect করবে এটা একটা fact
        → save_memory("masum", "user_name", "The user's name is Masum") কল হবে
    """
    prompt = f"""Does the following user message contain a durable personal fact worth
remembering long-term (e.g. their name, a preference, where they're from, what they like)?
Ignore questions, greetings, or anything that isn't a fact ABOUT the user.

User message: {state['question']}

Respond with ONLY a JSON object, no other text:
{{"has_fact": true, "fact_key": "user_name", "fact_text": "The user's name is Masum"}}
or, if there's no durable fact:
{{"has_fact": false, "fact_key": "", "fact_text": ""}}"""

    response = await grading_llm.ainvoke(prompt)
    result = json.loads(response.content.strip())

    if result.get("has_fact"):
        await save_memory(state["user_id"], result["fact_key"], result["fact_text"])

    return {}


# ---------------------------------------------------------------------------
# Node 6: save_turn — এই turn টা memory তে সেভ করে রাখা
# ---------------------------------------------------------------------------

async def save_turn(state: GraphState) -> dict:
    """
    WHAT: Saves the user's question and the final answer into short-term
          memory (Redis), so the next turn in this session has context.
    WHY: Without this, chat_history would never grow — every turn would
         look like the first turn.
    OUTPUT: {} (no state fields change, just a side effect)
    CALLED FROM: build_graph.py — the last node, after a good generation
                 is accepted.
    WHY CALLED: Keeps short-term memory up to date for the next turn.

    সহজ ভাষায়: graph এর একদম শেষ ধাপ — এই turn এর প্রশ্ন আর উত্তর Redis-এ
    সেভ করে রাখা, যাতে পরের বার একই session-এ কথা বললে বট মনে রাখে।
    """
    await add_message(state["session_id"], "user", state["question"])
    await add_message(state["session_id"], "assistant", state["generation"])
    return {}