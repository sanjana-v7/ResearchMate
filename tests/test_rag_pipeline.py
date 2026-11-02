"""
Tests for RAGPipeline: retrieval formatting, prompt construction,
citation correctness, missing information handling, and API resilience.
"""

from unittest.mock import MagicMock, patch
import pytest

from src.text_splitter import TextChunk
from src.vector_store import VectorStore
from src.retriever import DocumentRetriever
from src.rag_pipeline import RAGPipeline, RAGResponse, SYSTEM_INSTRUCTION


@pytest.fixture
def multi_doc_vector_store():
    """Populates vector store with chunks from distinct documents to test multi-doc RAG."""
    vs = VectorStore()
    vs.clear()
    chunks = [
        TextChunk(
            chunk_id="travel_p1_c0",
            filename="travel_policy.pdf",
            page=1,
            chunk_index=0,
            text="Meals during domestic business travel are reimbursed up to $75 per day.",
            doc_hash="hash_travel"
        ),
        TextChunk(
            chunk_id="travel_p2_c0",
            filename="travel_policy.pdf",
            page=2,
            chunk_index=0,
            text="All flight bookings must be economy class for flights under 6 hours.",
            doc_hash="hash_travel"
        ),
        TextChunk(
            chunk_id="security_p1_c0",
            filename="security_guidelines.txt",
            page=1,
            chunk_index=0,
            text="Passwords must be at least 14 characters long and changed every 90 days.",
            doc_hash="hash_sec"
        )
    ]
    vs.add_chunks(chunks)
    return vs


class TestRAGPipeline:

    def test_retriever_formatting(self, multi_doc_vector_store):
        """Verify context formatter produces clean, structured markdown with source and page."""
        retriever = DocumentRetriever(multi_doc_vector_store)
        results = retriever.retrieve("daily meal reimbursement limit", top_k=1)
        assert len(results) == 1

        formatted = retriever.format_context_for_prompt(results)
        assert "Source Document: travel_policy.pdf" in formatted
        assert "Page: 1" in formatted
        assert "$75 per day" in formatted

    def test_missing_context_when_empty(self):
        """Verify response indicates no documents indexed when store is empty."""
        empty_vs = VectorStore()
        empty_vs.clear()
        retriever = DocumentRetriever(empty_vs)
        pipeline = RAGPipeline(retriever=retriever, api_key="dummy_key")

        response = pipeline.answer_question("What is the reimbursement limit?")
        assert response.has_context is False
        assert "No documents have been indexed yet" in response.answer

    def test_missing_api_key_handling(self, multi_doc_vector_store):
        """Verify graceful error when API key is missing."""
        retriever = DocumentRetriever(multi_doc_vector_store)
        pipeline = RAGPipeline(retriever=retriever, api_key="")

        response = pipeline.answer_question("What is the password policy?")
        assert response.success is False
        assert "Gemini API key is not configured" in response.answer

    def test_prompt_construction(self, multi_doc_vector_store):
        """Verify prompt properly binds history, retrieved passages, and user question."""
        retriever = DocumentRetriever(multi_doc_vector_store)
        pipeline = RAGPipeline(retriever=retriever, api_key="test_key")

        history = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]
        prompt = pipeline._build_prompt(
            question="What is the meal limit?",
            context="[Passage 1] travel_policy.pdf Page 1: $75 per day",
            chat_history=history
        )

        assert "PREVIOUS CONVERSATION TURNS:" in prompt
        assert "Hi" in prompt
        assert "RETRIEVED CONTEXT FROM UPLOADED DOCUMENTS:" in prompt
        assert "$75 per day" in prompt
        assert "USER QUESTION:\nWhat is the meal limit?" in prompt

    def test_mocked_gemini_generation_with_citations(self, multi_doc_vector_store):
        """Verify complete RAG flow with simulated Gemini API response including citation validation."""
        retriever = DocumentRetriever(multi_doc_vector_store)
        pipeline = RAGPipeline(retriever=retriever, api_key="mock_key")

        mock_answer_text = (
            "Domestic meal expenses are reimbursed up to $75 per day [Document: travel_policy.pdf, Page: 1]."
        )

        # Mock the client.models.generate_content call
        mock_response = MagicMock()
        mock_response.text = mock_answer_text

        with patch.object(pipeline.client.models, "generate_content", return_value=mock_response):
            rag_res = pipeline.answer_question("How much can I expense for meals?")

            assert rag_res.success is True
            assert "[Document: travel_policy.pdf, Page: 1]" in rag_res.answer
            assert len(rag_res.sources) > 0
            assert rag_res.sources[0]["filename"] == "travel_policy.pdf"
            assert rag_res.sources[0]["page"] == 1

    def test_multi_document_retrieval(self, multi_doc_vector_store):
        """Verify questions touching multiple topics retrieve passages from multiple documents."""
        retriever = DocumentRetriever(multi_doc_vector_store)
        results = retriever.retrieve("travel flight rules and security password requirements", top_k=3)

        filenames = {r["filename"] for r in results}
        # Both travel_policy.pdf and security_guidelines.txt should be retrieved
        assert "travel_policy.pdf" in filenames
        assert "security_guidelines.txt" in filenames
