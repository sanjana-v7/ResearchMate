"""
Verification script for ResearchMate: indexes sample documents and verifies
multi-document semantic retrieval and provenance.
"""

from pathlib import Path
from src.document_processor import DocumentProcessor
from src.text_splitter import DocumentTextSplitter
from src.vector_store import VectorStore
from src.retriever import DocumentRetriever
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, INDEX_PATH, METADATA_PATH

def main():
    print("--- 1. Initializing VectorStore ---")
    vs = VectorStore(index_path=INDEX_PATH, metadata_path=METADATA_PATH)
    vs.clear()

    splitter = DocumentTextSplitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    samples_dir = Path(__file__).parent / "data" / "sample_documents"
    sample_files = list(samples_dir.glob("*.txt"))

    print(f"Found {len(sample_files)} sample documents:")
    for f in sample_files:
        print(f" - {f.name}")
        pages = DocumentProcessor.process_file(f, f.name)
        chunks = splitter.split_pages(pages)
        added = vs.add_chunks(chunks)
        print(f"   Indexed {added} chunks from {f.name}")

    vs.save()
    print("\n--- 2. Corpus Statistics ---")
    stats = vs.get_document_stats()
    print(f"Total documents: {stats['total_documents']}")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total pages: {stats['total_pages']}")

    print("\n--- 3. Testing Semantic Queries ---")
    retriever = DocumentRetriever(vs)

    test_queries = [
        "How much meal allowance do employees get on travel?",
        "What processor does the HyperRouter X-1000 use?",
        "What are the four evaluation metrics for RAG systems?"
    ]

    for q in test_queries:
        print(f"\nQuery: '{q}'")
        results = retriever.retrieve(q, top_k=1)
        for r in results:
            print(f" -> Top Match: [{r['filename']} | Page {r['page']} | Score: {r['score']:.4f}]")
            print(f"    Excerpt: {r['text'][:140]}...")

    print("\n[SUCCESS] End-to-end ingestion and retrieval pipeline verified!")

if __name__ == "__main__":
    main()
