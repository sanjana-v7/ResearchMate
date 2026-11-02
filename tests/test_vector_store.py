"""
Tests for VectorStore: FAISS indexing, metadata synchronization,
similarity search, duplicate detection, document removal, and persistence.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np

from src.text_splitter import TextChunk
from src.vector_store import VectorStore


@pytest.fixture
def sample_chunks():
    """Provides sample chunks from multiple documents."""
    return [
        TextChunk(
            chunk_id="policy_p1_c0",
            filename="company_policy.pdf",
            page=1,
            chunk_index=0,
            text="Employees are eligible for 20 days of paid annual leave per year.",
            doc_hash="hash_policy_123"
        ),
        TextChunk(
            chunk_id="policy_p2_c0",
            filename="company_policy.pdf",
            page=2,
            chunk_index=0,
            text="Health insurance covers inpatient treatments up to $50,000 annually.",
            doc_hash="hash_policy_123"
        ),
        TextChunk(
            chunk_id="manual_p1_c0",
            filename="product_manual.txt",
            page=1,
            chunk_index=0,
            text="To reset the router, hold the red button on the back for 10 seconds.",
            doc_hash="hash_manual_456"
        )
    ]


class TestVectorStore:

    def test_indexing_and_total_count(self, sample_chunks):
        """Verify adding chunks correctly populates FAISS index and metadata."""
        vs = VectorStore()
        vs.clear()

        count = vs.add_chunks(sample_chunks)
        assert count == 3
        assert vs.total_vectors == 3
        assert len(vs.chunks_metadata) == 3

    def test_similarity_search_ranking(self, sample_chunks):
        """Verify semantic similarity search returns the most relevant passage."""
        vs = VectorStore()
        vs.clear()
        vs.add_chunks(sample_chunks)

        # Search for leave / vacation
        results = vs.similarity_search("How many vacation days do I get?", top_k=1)
        assert len(results) == 1
        assert results[0]["filename"] == "company_policy.pdf"
        assert results[0]["page"] == 1
        assert "paid annual leave" in results[0]["text"]
        assert results[0]["score"] > 0.4  # High semantic similarity

        # Search for router reset
        router_results = vs.similarity_search("How to reboot device?", top_k=1)
        assert len(router_results) == 1
        assert router_results[0]["filename"] == "product_manual.txt"
        assert "reset the router" in router_results[0]["text"]

    def test_search_with_filename_filter(self, sample_chunks):
        """Verify similarity search can filter results to a specific document."""
        vs = VectorStore()
        vs.clear()
        vs.add_chunks(sample_chunks)

        # Search with filter for product_manual.txt
        results = vs.similarity_search(
            "medical insurance hospital",
            top_k=2,
            filter_filename="product_manual.txt"
        )
        assert len(results) <= 1
        if results:
            assert results[0]["filename"] == "product_manual.txt"

    def test_duplicate_detection(self, sample_chunks):
        """Verify detection of duplicate content hash and duplicate filenames."""
        vs = VectorStore()
        vs.clear()
        vs.add_chunks(sample_chunks)

        # Test duplicate hash
        is_dup_hash, reason = vs.is_duplicate("new_name.pdf", "hash_policy_123")
        assert is_dup_hash is True
        assert "matching content hash" in reason

        # Test duplicate filename
        is_dup_name, reason2 = vs.is_duplicate("company_policy.pdf", "unique_hash_999")
        assert is_dup_name is True
        assert "already indexed" in reason2

        # Test non-duplicate
        is_dup_none, _ = vs.is_duplicate("fresh_doc.pdf", "brand_new_hash_000")
        assert is_dup_none is False

    def test_document_removal_and_rebuilding(self, sample_chunks):
        """Verify removing a document purges its chunks and rebuilds the FAISS index."""
        vs = VectorStore()
        vs.clear()
        vs.add_chunks(sample_chunks)
        assert vs.total_vectors == 3

        # Remove company_policy.pdf (which has 2 chunks)
        removed = vs.remove_document("company_policy.pdf")
        assert removed is True
        assert vs.total_vectors == 1
        assert len(vs.chunks_metadata) == 1
        assert vs.chunks_metadata[0]["filename"] == "product_manual.txt"

        # Search should no longer return company_policy.pdf
        results = vs.similarity_search("annual leave", top_k=3)
        for r in results:
            assert r["filename"] != "company_policy.pdf"

    def test_document_stats(self, sample_chunks):
        """Verify calculation of aggregate corpus statistics."""
        vs = VectorStore()
        vs.clear()
        vs.add_chunks(sample_chunks)

        stats = vs.get_document_stats()
        assert stats["total_documents"] == 2
        assert stats["total_chunks"] == 3
        assert stats["total_pages"] == 3  # company_policy has 2 pages, product_manual has 1

    def test_persistence_save_and_load(self, sample_chunks):
        """Verify index and metadata serialization to disk and reload."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_index = Path(tmp_dir) / "test_index.bin"
            tmp_meta = Path(tmp_dir) / "test_meta.json"

            vs1 = VectorStore()
            vs1.clear()
            vs1.add_chunks(sample_chunks)
            vs1.save(index_path=tmp_index, metadata_path=tmp_meta)

            assert tmp_index.exists()
            assert tmp_meta.exists()

            vs2 = VectorStore()
            loaded = vs2.load(index_path=tmp_index, metadata_path=tmp_meta)
            assert loaded is True
            assert vs2.total_vectors == 3
            assert len(vs2.chunks_metadata) == 3
            assert vs2.chunks_metadata[0]["chunk_id"] == "policy_p1_c0"
