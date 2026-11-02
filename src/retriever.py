"""
ResearchMate - Retriever Module
Handles semantic retrieval, candidate reranking, threshold filtering,
and context assembly for the RAG pipeline.
"""

from typing import List, Dict, Any, Optional
from src.vector_store import VectorStore
from config import DEFAULT_TOP_K, SIMILARITY_SCORE_THRESHOLD


class DocumentRetriever:
    """Retrieves and formats relevant document passages based on semantic similarity."""

    def __init__(
        self,
        vector_store: VectorStore,
        default_top_k: int = DEFAULT_TOP_K,
        score_threshold: float = SIMILARITY_SCORE_THRESHOLD
    ):
        self.vector_store = vector_store
        self.default_top_k = default_top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_filename: Optional[str] = None,
        apply_threshold: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant document chunks for a query.

        Args:
            query: The question or query string.
            top_k: Optional override for number of passages.
            filter_filename: Optional document filter.
            apply_threshold: Whether to strictly discard passages below score_threshold.

        Returns:
            List of retrieved chunks with metadata and similarity scores.
        """
        k = top_k if top_k is not None else self.default_top_k
        results = self.vector_store.similarity_search(
            query=query,
            top_k=k,
            filter_filename=filter_filename
        )

        if apply_threshold and self.score_threshold > 0:
            filtered = [r for r in results if r.get("score", 0.0) >= self.score_threshold]
            return filtered

        return results

    @staticmethod
    def format_context_for_prompt(retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Format retrieved passages into a structured, unambiguous prompt context
        with explicit source tags and page numbers.

        Args:
            retrieved_chunks: List of retrieved chunk dictionaries.

        Returns:
            Formatted context string.
        """
        if not retrieved_chunks:
            return "No relevant context found in uploaded documents."

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            fname = chunk.get("filename", "Unknown Document")
            page = chunk.get("page", 1)
            text = chunk.get("text", "").strip()
            score = chunk.get("score", 0.0)

            block = (
                f"--- [Passage {i}] ---\n"
                f"Source Document: {fname}\n"
                f"Page: {page}\n"
                f"Relevance Score: {score:.3f}\n"
                f"Content:\n{text}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)
