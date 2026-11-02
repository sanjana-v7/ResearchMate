"""
Tests for DocumentTextSplitter: recursive character splitting, metadata preservation, and chunk filtering.
"""

import pytest
from src.document_processor import PageContent
from src.text_splitter import DocumentTextSplitter, TextChunk


class TestTextSplitter:

    def test_chunking_and_metadata_preservation(self):
        """Verify text is chunked while maintaining page and document provenance."""
        # Create sample multi-page document content
        long_text_p1 = "Paragraph 1 about system requirements. " * 30  # ~1170 chars
        short_text_p2 = "Page 2 concluding remarks on architecture."

        pages = [
            PageContent(
                filename="system_spec.pdf",
                page=1,
                text=long_text_p1,
                doc_hash="hash_p123"
            ),
            PageContent(
                filename="system_spec.pdf",
                page=2,
                text=short_text_p2,
                doc_hash="hash_p123"
            )
        ]

        splitter = DocumentTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = splitter.split_pages(pages)

        assert len(chunks) > 1
        # Check first page chunks
        p1_chunks = [c for c in chunks if c.page == 1]
        assert len(p1_chunks) >= 3
        for c in p1_chunks:
            assert c.filename == "system_spec.pdf"
            assert c.doc_hash == "hash_p123"
            assert "p1_c" in c.chunk_id
            assert len(c.text) > 0

        # Check second page chunk
        p2_chunks = [c for c in chunks if c.page == 2]
        assert len(p2_chunks) == 1
        assert p2_chunks[0].page == 2
        assert "Page 2 concluding remarks" in p2_chunks[0].text
        assert "p2_c0" in p2_chunks[0].chunk_id

    def test_empty_chunks_removal(self):
        """Verify that empty or whitespace-only pages/chunks are ignored."""
        pages = [
            PageContent(
                filename="empty_doc.txt",
                page=1,
                text="   \n\n   \t  ",
                doc_hash="hash_empty"
            ),
            PageContent(
                filename="valid_doc.txt",
                page=1,
                text="Valid content for chunking.",
                doc_hash="hash_valid"
            )
        ]

        splitter = DocumentTextSplitter(chunk_size=800, chunk_overlap=150)
        chunks = splitter.split_pages(pages)

        assert len(chunks) == 1
        assert chunks[0].filename == "valid_doc.txt"
        assert chunks[0].text == "Valid content for chunking."

    def test_chunk_serialization(self):
        """Verify to_dict and from_dict roundtrip."""
        chunk = TextChunk(
            chunk_id="doc1_p1_c0",
            filename="policy.pdf",
            page=1,
            chunk_index=0,
            text="Employees must submit travel expenses within 30 days.",
            doc_hash="abc12345"
        )
        chunk_dict = chunk.to_dict()
        assert chunk_dict["chunk_id"] == "doc1_p1_c0"
        assert chunk_dict["page"] == 1

        reconstructed = TextChunk.from_dict(chunk_dict)
        assert reconstructed == chunk
