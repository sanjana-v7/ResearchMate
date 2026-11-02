"""
ResearchMate - Summarizer Module
Generates strictly grounded, structured summaries for uploaded documents.
Covers Main Topic, Key Points, Important Information, Dates & Figures, and Conclusions.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.vector_store import VectorStore
from config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_TEMPERATURE,
    get_gemini_api_key
)


@dataclass
class SummaryResult:
    """Stores generated document summary and status."""
    document_name: str
    content: str
    success: bool = True
    error_message: Optional[str] = None


SUMMARIZATION_SYSTEM_PROMPT = """You are ResearchMate Document Summarizer.
Your goal is to produce an accurate, comprehensive, and strictly grounded summary of the provided document text.

CRITICAL RULES:
1. ONLY summarize information explicitly present in the document. Do not invent facts, extrapolate, or bring outside knowledge.
2. Structure the summary cleanly with the following markdown headers:
   ### 1. Main Topic
   A concise overview of the subject matter, purpose, and scope of the document.
   
   ### 2. Key Points
   Bullet points covering the core arguments, policies, guidelines, or findings.
   
   ### 3. Important Information & Operational Details
   Specific procedures, responsibilities, technical specifications, or notable rules.
   
   ### 4. Important Dates, Numbers & Figures
   Any dates, statistics, percentages, deadlines, financial amounts, or quantitative benchmarks found in the text.
   (If none exist, explicitly state: "No specific dates or figures mentioned.")
   
   ### 5. Conclusions & Takeaways
   Final conclusions, recommendations, or closing outcomes directly supported by the text.
"""


class DocumentSummarizer:
    """Generates structured summaries of single or multiple documents using Gemini."""

    def __init__(
        self,
        vector_store: VectorStore,
        api_key: Optional[str] = None,
        model_name: str = DEFAULT_GEMINI_MODEL
    ):
        self.vector_store = vector_store
        self.api_key = api_key or get_gemini_api_key()
        self.model_name = model_name
        self.client: Optional[genai.Client] = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize the Google GenAI client."""
        if self.api_key and self.api_key.strip():
            try:
                self.client = genai.Client(api_key=self.api_key.strip())
            except Exception:
                self.client = None

    def update_api_key(self, new_key: str):
        """Update API key and reinitialize client."""
        self.api_key = new_key.strip()
        self._initialize_client()

    def summarize_document(self, filename: Optional[str] = None) -> SummaryResult:
        """
        Summarize a specific document or all indexed documents.

        Args:
            filename: Name of the document to summarize, or None to summarize the entire corpus.

        Returns:
            SummaryResult containing structured markdown summary.
        """
        # Ensure API client is configured
        if not self.api_key:
            return SummaryResult(
                document_name=filename or "All Documents",
                content="",
                success=False,
                error_message="Gemini API Key is not configured. Please set GEMINI_API_KEY."
            )

        if self.client is None:
            self._initialize_client()
            if self.client is None:
                return SummaryResult(
                    document_name=filename or "All Documents",
                    content="",
                    success=False,
                    error_message="Could not initialize Gemini API client."
                )

        # Collect chunks for the target document(s)
        matching_chunks = [
            meta for meta in self.vector_store.chunks_metadata
            if filename is None or meta.get("filename") == filename
        ]

        if not matching_chunks:
            target_desc = f"document '{filename}'" if filename else "any uploaded documents"
            return SummaryResult(
                document_name=filename or "All Documents",
                content="",
                success=False,
                error_message=f"No text chunks found for {target_desc}."
            )

        # Combine text content (preserving page structure)
        # Sort chunks by page and chunk index to ensure sequential reading order
        matching_chunks.sort(key=lambda c: (c.get("filename", ""), c.get("page", 1), c.get("chunk_index", 0)))

        text_sections = []
        current_page = None
        for chunk in matching_chunks:
            page = chunk.get("page", 1)
            fname = chunk.get("filename", "Doc")
            if page != current_page:
                text_sections.append(f"\n--- [{fname} - Page {page}] ---\n")
                current_page = page
            text_sections.append(chunk.get("text", ""))

        full_document_text = "\n".join(text_sections)

        # Truncate if exceptionally huge to stay comfortably within context window (e.g., 100,000 chars)
        max_chars = 120000
        if len(full_document_text) > max_chars:
            full_document_text = full_document_text[:max_chars] + "\n\n[... Document truncated for summarization context limit ...]"

        prompt = (
            f"Please generate a comprehensive, structured summary for the following document text "
            f"according to the specified 5-section format.\n\n"
            f"DOCUMENT: {filename or 'Combined Uploaded Documents'}\n\n"
            f"TEXT CONTENT:\n{full_document_text}"
        )

        try:
            config = types.GenerateContentConfig(
                system_instruction=SUMMARIZATION_SYSTEM_PROMPT,
                temperature=DEFAULT_TEMPERATURE,
                max_output_tokens=3000
            )

            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config
                )
            except APIError as api_err:
                if ("404" in str(api_err) or "NOT_FOUND" in str(api_err)) and self.model_name != "gemini-3.8-flash":
                    response = self.client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt,
                        config=config
                    )
                    self.model_name = "gemini-3.8-flash"
                else:
                    raise api_err

            summary_text = response.text if response and response.text else "No summary could be generated."

            return SummaryResult(
                document_name=filename or "All Documents",
                content=summary_text,
                success=True
            )

        except APIError as e:
            return SummaryResult(
                document_name=filename or "All Documents",
                content="",
                success=False,
                error_message=f"Gemini API Error during summarization: {str(e)}"
            )
        except Exception as e:
            return SummaryResult(
                document_name=filename or "All Documents",
                content="",
                success=False,
                error_message=f"Unexpected error during summarization: {str(e)}"
            )
