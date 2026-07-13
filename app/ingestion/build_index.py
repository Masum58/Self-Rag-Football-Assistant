"""
WHAT: Loads real text files from app/data/sample_docs/, splits them into
      chunks, and uploads those chunks into Pinecone.
WHY: Real documents are too long to embed as one single vector (the
      embedding would blur together too many different topics). Splitting
      into smaller, focused chunks means each vector represents one
      coherent idea, making retrieval much more accurate.
OUTPUT: Prints how many chunks were created and uploaded.
CALLED FROM: Run manually via `python -m app.ingestion.build_index`
WHY CALLED: One-time (or occasional) data-loading step.

সহজ ভাষায় (Bengali):
আগে আমরা হাতে লেখা ছোট ছোট বাক্য সরাসরি Pinecone-এ পাঠিয়েছিলাম। এবার
আসল pipeline বানাচ্ছি — (১) real .txt ফাইল থেকে টেক্সট load করা,
(২) সেই বড় টেক্সটকে ছোট ছোট chunk-এ ভাগ করা, (৩) প্রতিটা chunk আলাদা
ভাবে Pinecone-এ পাঠানো।
"""

from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vectorstore.pinecone_client import upsert_documents

SAMPLE_DOCS_DIR = Path("app/data/sample_docs")


def load_and_chunk_documents() -> list[dict]:
    """
    WHAT: Loads every .txt file in sample_docs/, splits each into chunks.
    WHY: Separates "loading" (get raw text out of a file) from "chunking"
         (break that text into embedding-friendly pieces) — two distinct
         steps, kept as two distinct tool calls (TextLoader, then splitter).
    OUTPUT: List of dicts like [{"id": "...", "text": "..."}, ...] — ready
            to hand to upsert_documents().
    CALLED FROM: main() (below).
    WHY CALLED: Prepares chunked data before uploading to Pinecone.

    সহজ ভাষায়: sample_docs ফোল্ডারে যত .txt ফাইল আছে, প্রতিটা খুলে পড়ে
    (TextLoader), তারপর ভেঙে ছোট ছোট অংশ বানায় (RecursiveCharacterTextSplitter)।

    Example:
        chunks = load_and_chunk_documents()
        # → [{"id": "world_cup_history_chunk_0", "text": "FIFA World Cup History..."},
        #    {"id": "world_cup_history_chunk_1", "text": "Brazil holds the record..."}, ...]
    """
    # RecursiveCharacterTextSplitter: বড় টেক্সটকে ছোট chunk এ ভাঙে, কিন্তু
    # চেষ্টা করে বাক্য/paragraph এর মাঝখানে না কেটে, যতটা সম্ভব অর্থপূর্ণ
    # জায়গায় (paragraph break, sentence end) কাটতে।
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,      # প্রতিটা chunk প্রায় ৪০০ ক্যারেক্টার
        chunk_overlap=50,    # পাশাপাশি chunk এ ৫০ ক্যারেক্টার overlap, যাতে
                             # কোনো বাক্যের context হারিয়ে না যায়
    )

    all_chunks = []

    txt_files = list(SAMPLE_DOCS_DIR.glob("*.txt"))
    print(f"{len(txt_files)}টা .txt ফাইল পাওয়া গেছে {SAMPLE_DOCS_DIR}-এ")

    for file_path in txt_files:
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents = loader.load()  # ফাইল থেকে raw text বের করা (Document object হিসেবে)

        chunks = splitter.split_documents(documents)  # raw text কে ছোট chunk এ ভাঙা

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{file_path.stem}_chunk_{i}",
                "text": chunk.page_content,
            })

        print(f"  - {file_path.name}: {len(chunks)}টা chunk তৈরি হলো")

    return all_chunks


def main():
    chunks = load_and_chunk_documents()
    print(f"\nমোট {len(chunks)}টা chunk Pinecone-এ upload করছি...")
    upsert_documents(chunks)
    print("✅ Upload সম্পূর্ণ!")


if __name__ == "__main__":
    main()