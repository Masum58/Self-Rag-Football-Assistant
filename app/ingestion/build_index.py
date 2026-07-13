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
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vectorstore.pinecone_client import upsert_documents

SAMPLE_DOCS_DIR = Path("app/data/sample_docs")


def process_single_file(file_path: Path, splitter=None) -> list[dict]:
    """
    WHAT: Processes a single file, selects the right loader, and chunks it.
    """
    if splitter is None:
        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    loader = None
    ext = file_path.suffix.lower()

    # ফাইলের ধরন অনুযায়ী লোডার সিলেক্ট করা
    if ext == ".txt":
        loader = TextLoader(str(file_path), encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(str(file_path))
    elif ext == ".csv":
        loader = CSVLoader(str(file_path))
    else:
        print(f"  - {file_path.name}: Unsupported file type, skipping.")
        return []

    file_chunks = []
    try:
        documents = loader.load()  # ফাইল থেকে raw text বের করা
        chunks = splitter.split_documents(documents)  # raw text কে ছোট chunk এ ভাঙা

        for i, chunk in enumerate(chunks):
            file_chunks.append({
                "id": f"{file_path.stem}_chunk_{i}",
                "text": chunk.page_content,
            })

        print(f"  - {file_path.name}: {len(chunks)}টা chunk তৈরি হলো")
    except Exception as e:
        print(f"  - {file_path.name}: Error loading file - {e}")

    return file_chunks


def load_and_chunk_documents() -> list[dict]:
    """
    WHAT: Loads every file in sample_docs/, splits each into chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
    )

    all_chunks = []

    # সব ধরনের ফাইল খোঁজার জন্য rglob("*") ব্যবহার করছি
    all_files = [f for f in SAMPLE_DOCS_DIR.rglob("*") if f.is_file()]
    print(f"মোট {len(all_files)}টা ফাইল পাওয়া গেছে {SAMPLE_DOCS_DIR}-এ")

    for file_path in all_files:
        all_chunks.extend(process_single_file(file_path, splitter))

    return all_chunks


def main():
    chunks = load_and_chunk_documents()
    print(f"\nমোট {len(chunks)}টা chunk Pinecone-এ upload করছি...")
    upsert_documents(chunks)
    print("✅ Upload সম্পূর্ণ!")


if __name__ == "__main__":
    main()