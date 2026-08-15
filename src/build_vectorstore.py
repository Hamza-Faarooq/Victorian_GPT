"""
Stage 4a — Build the ChromaDB vector store used for retrieval.

Embeds every cleaned novel with a sentence-transformers model and indexes
the chunks into a persistent ChromaDB collection, one file at a time to
keep memory usage low.

Usage:
    python src/build_vectorstore.py
    python src/build_vectorstore.py --query "What happens to the ship Demeter in the storm?"
"""

import argparse
from pathlib import Path

from config import CLEANED_DIR, EMBEDDING_MODEL, RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE, VECTORSTORE_DIR


def build_vectorstore(
    cleaned_dir: Path,
    persist_dir: Path,
    embedding_model: str = EMBEDDING_MODEL,
    chunk_size: int = RAG_CHUNK_SIZE,
    chunk_overlap: int = RAG_CHUNK_OVERLAP,
):
    """Embed and index every cleaned novel into a persistent Chroma collection."""
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    persist_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    vectorstore = Chroma(persist_directory=str(persist_dir), embedding_function=embeddings)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    books = sorted(p for p in cleaned_dir.iterdir() if p.suffix == ".txt")
    total_chunks = 0

    print(f"\nIndexing {len(books)} manuscripts...\n")
    for i, path in enumerate(books):
        print(f"[{i + 1}/{len(books)}] Embedding: {path.name}...")

        text = path.read_text(encoding="utf-8")
        split_docs = text_splitter.split_text(text)
        sources = [{"source": path.name} for _ in split_docs]

        vectorstore.add_texts(texts=split_docs, metadatas=sources)
        total_chunks += len(split_docs)

    print(f"\n{total_chunks} total chunks embedded into ChromaDB at: {persist_dir}")
    return vectorstore


def test_query(vectorstore, query: str, k: int = 2) -> None:
    print(f"\nTesting retrieval for: '{query}'")
    docs = vectorstore.similarity_search(query, k=k)
    for i, doc in enumerate(docs):
        print(f"\n--- RETRIEVED CHUNK {i + 1} (Source: {doc.metadata['source']}) ---")
        print(doc.page_content)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the VictorianGPT RAG vector store.")
    parser.add_argument("--cleaned-dir", type=Path, default=CLEANED_DIR)
    parser.add_argument("--persist-dir", type=Path, default=VECTORSTORE_DIR)
    parser.add_argument("--embedding-model", type=str, default=EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=RAG_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=RAG_CHUNK_OVERLAP)
    parser.add_argument(
        "--query",
        type=str,
        default="What happens to the ship Demeter in the storm?",
        help="Sample query to test retrieval with after indexing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vs = build_vectorstore(
        args.cleaned_dir,
        args.persist_dir,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    test_query(vs, args.query)
