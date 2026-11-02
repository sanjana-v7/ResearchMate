"""
ResearchMate - Text Splitter Module
Implements semantic chunking using LangChain's RecursiveCharacterTextSplitter.
Preserves page-level metadata and generates unique chunk IDs.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.document_processor import PageContent
from config import DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP


@dataclass
class TextChunk:
    """Represents a text chunk with complete provenance metadata."""
    chunk_id: str
    filename: str
    page: int
    chunk_index: int
    text: str
    doc_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk into a dictionary representation for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextChunk":
        """Reconstruct TextChunk from dictionary."""
        return cls(**data)


class DocumentTextSplitter:
    """Splits document pages into overlapping semantic chunks with metadata preservation."""

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

    def split_pages(self, pages: List[PageContent]) -> List[TextChunk]:
        """
        Split a list of PageContent objects into TextChunks.
        Preserves original filename, page number, and document hash.
        Removes empty or whitespace-only chunks.

        Args:
            pages: List of PageContent objects extracted from documents.

        Returns:
            List of TextChunk objects with complete metadata.
        """
        chunks: List[TextChunk] = []

        for page_obj in pages:
            raw_text = page_obj.text.strip()
            if not raw_text:
                continue

            # Split text of this specific page
            page_splits = self.splitter.split_text(raw_text)

            chunk_idx = 0
            for split_text in page_splits:
                cleaned_text = split_text.strip()
                if not cleaned_text:
                    continue

                # Generate clean unique chunk id
                safe_filename = "".join(c if c.isalnum() else "_" for c in page_obj.filename)
                chunk_id = f"{safe_filename}_p{page_obj.page}_c{chunk_idx}"

                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        filename=page_obj.filename,
                        page=page_obj.page,
                        chunk_index=chunk_idx,
                        text=cleaned_text,
                        doc_hash=page_obj.doc_hash
                    )
                )
                chunk_idx += 1

        return chunks
