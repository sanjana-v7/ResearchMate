"""
ResearchMate - RAG Pipeline Module
Orchestrates semantic retrieval, grounded prompt construction, Gemini API generation,
and citation verification using the official google-genai SDK.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import os

from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.retriever import DocumentRetriever
from config import (
    DEFAULT_GEMINI_MODEL,
    FALLBACK_GEMINI_MODEL,
    DEFAULT_TEMPERATURE,
    DEFAULT_MAX_OUTPUT_TOKENS,
    get_gemini_api_key
)


@dataclass
class RAGResponse:
    """Encapsulates the grounded answer, citations, and execution status."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error_message: Optional[str] = None
    has_context: bool = True


SYSTEM_INSTRUCTION = """You are ResearchMate, an accurate, analytical, and honest AI Document Assistant.
Your primary directive is to answer user queries using ONLY the provided document excerpts (retrieved context).

RULES AND CONSTRAINTS:
1. STRICT GROUNDING: Answer strictly and exclusively using facts, figures, and statements found in the RETRIEVED CONTEXT. Never introduce external knowledge or assumptions.
2. CITATIONS: Whenever stating a fact or claim from the context, include an explicit citation in the format: [Document: <filename>, Page: <page>].
3. NO FABRICATION: Never fabricate or hallucinate citations or content. Every cited document and page must correspond exactly to the passages in the context.
4. MISSING INFORMATION: If the provided context does not contain enough information to answer the question, clearly state: "I could not find information regarding this in the uploaded documents." Do not speculate or make up an answer.
5. CONFLICTING INFORMATION: If different documents or pages contradict each other, explicitly highlight the discrepancy (e.g., "[Document A, Page 2] states X, whereas [Document B, Page 4] states Y").
6. DISTINGUISH FACTS FROM INFERENCE: If the question requires synthesis or logical deduction, clearly separate directly stated facts from logical deductions.
7. TONE & STYLE: Maintain a professional, concise, and structured tone. Use markdown headings or bullet points where helpful.
"""


class RAGPipeline:
    """End-to-end RAG orchestrator coordinating retrieval, prompt assembly, and Gemini inference."""

    def __init__(
        self,
        retriever: DocumentRetriever,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL
    ):
        self.retriever = retriever
        self.api_key = api_key or get_gemini_api_key()
        self.model_name = model_name
        self.client: Optional[genai.Client] = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Google GenAI client if an API key is available."""
        if self.api_key and self.api_key.strip():
            try:
                self.client = genai.Client(api_key=self.api_key.strip())
            except Exception:
                self.client = None

    def update_api_key(self, new_key: str):
        """Update API key and reinitialize client."""
        self.api_key = new_key.strip()
        self._initialize_client()

    def answer_question(
        self,
        question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 5,
        filter_filename: Optional[str] = None
    ) -> RAGResponse:
        """
        Execute full RAG pipeline for a user question.

        Args:
            question: User's natural language question.
            chat_history: Optional list of previous conversation turns [{'role': 'user'|'assistant', 'content': str}].
            top_k: Number of chunks to retrieve.
            filter_filename: Optional document filter.

        Returns:
            RAGResponse containing grounded answer and verified sources.
        """
        if not question or not question.strip():
            return RAGResponse(
                answer="Please enter a valid question.",
                sources=[],
                success=False,
                error_message="Empty question string."
            )

        # Ensure API key is configured
        if not self.api_key:
            return RAGResponse(
                answer="Gemini API key is not configured. Please enter your GEMINI_API_KEY in the sidebar or set it in your .env / Streamlit secrets.",
                sources=[],
                success=False,
                error_message="Missing GEMINI_API_KEY"
            )

        if self.client is None:
            self._initialize_client()
            if self.client is None:
                return RAGResponse(
                    answer="Could not initialize Gemini API client. Please verify your GEMINI_API_KEY.",
                    sources=[],
                    success=False,
                    error_message="Client initialization failed"
                )

        # 1. Semantic Retrieval
        retrieved_chunks = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            filter_filename=filter_filename
        )

        # Check if vector store is completely empty
        if not retrieved_chunks:
            return RAGResponse(
                answer="No documents have been indexed yet. Please upload documents in the sidebar first.",
                sources=[],
                success=True,
                has_context=False
            )

        # 2. Context Construction
        formatted_context = self.retriever.format_context_for_prompt(retrieved_chunks)

        # 3. Prompt Assembly (with conversational context if available)
        prompt_content = self._build_prompt(question, formatted_context, chat_history)

        # 4. Generate Content via Gemini SDK
        try:
            config = types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=DEFAULT_TEMPERATURE,
                max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt_content,
                config=config
            )

            generated_text = response.text if response and response.text else "No response generated."

            return RAGResponse(
                answer=generated_text,
                sources=retrieved_chunks,
                success=True,
                has_context=True
            )

        except APIError as e:
            # Handle Gemini API errors (rate limits, invalid key, quota, etc.)
            error_str = str(e)
            user_friendly_msg = f"Gemini API Error: {error_str}"
            if "RESOURCE_EXHAUSTED" in error_str or "429" in error_str:
                user_friendly_msg = "Gemini API rate limit or quota exceeded. Please wait a moment and try again."
            elif "INVALID_ARGUMENT" in error_str or "API_KEY_INVALID" in error_str:
                user_friendly_msg = "The provided Gemini API Key is invalid. Please check your credentials."

            return RAGResponse(
                answer=user_friendly_msg,
                sources=retrieved_chunks,
                success=False,
                error_message=error_str,
                has_context=True
            )
        except Exception as e:
            return RAGResponse(
                answer=f"An unexpected error occurred during answer generation: {str(e)}",
                sources=retrieved_chunks,
                success=False,
                error_message=str(e),
                has_context=True
            )

    def _build_prompt(
        self,
        question: str,
        context: str,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Construct the prompt incorporating conversational history and retrieved context."""
        history_section = ""
        if chat_history and len(chat_history) > 0:
            history_lines = []
            # Keep recent conversation turns for context
            for turn in chat_history[-6:]:
                role = "User" if turn.get("role") == "user" else "Assistant"
                content = turn.get("content", "").strip()
                history_lines.append(f"{role}: {content}")
            history_section = "PREVIOUS CONVERSATION TURNS:\n" + "\n".join(history_lines) + "\n\n"

        prompt = (
            f"{history_section}"
            f"RETRIEVED CONTEXT FROM UPLOADED DOCUMENTS:\n"
            f"{context}\n\n"
            f"USER QUESTION:\n"
            f"{question}\n\n"
            f"Please provide a grounded answer with specific citations [Document: <filename>, Page: <page>] "
            f"based strictly on the retrieved context above."
        )
        return prompt
