"""
ResearchMate - Document Processor Module
Handles extraction of raw text from PDF, TXT, and DOCX files while preserving
document metadata (filenames, page numbers, content hashes). Includes robust
detection of empty files, corrupted documents, and image-only/scanned documents.
"""

import hashlib
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union, BinaryIO

import pymupdf as fitz
import docx


class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass


class EmptyFileError(DocumentProcessingError):
    """Raised when an uploaded file contains 0 bytes."""
    pass


class CorruptedFileError(DocumentProcessingError):
    """Raised when a file cannot be parsed due to corruption or format violations."""
    pass


class NoTextExtractedError(DocumentProcessingError):
    """Raised when a document contains no extractable text (e.g., scanned/image-only PDF)."""
    pass


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when an uploaded file extension is not supported."""
    pass


@dataclass
class PageContent:
    """Represents text extracted from a specific page/section of a document."""
    filename: str
    page: int
    text: str
    doc_hash: str


def compute_file_hash(file_bytes: bytes) -> str:
    """Compute SHA-256 hash of raw file bytes for duplicate upload detection."""
    return hashlib.sha256(file_bytes).hexdigest()


class DocumentProcessor:
    """Extracts text and metadata from PDF, TXT, and DOCX files."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

    @classmethod
    def is_supported(cls, filename: str) -> bool:
        """Check if filename has a supported extension."""
        ext = Path(filename).suffix.lower()
        return ext in cls.SUPPORTED_EXTENSIONS

    @classmethod
    def process_file(
        cls,
        file_input: Union[str, Path, BinaryIO, bytes],
        filename: str
    ) -> List[PageContent]:
        """
        Process a file from disk path, BytesIO, or raw bytes.

        Args:
            file_input: Path to file or bytes/stream.
            filename: Original file name with extension.

        Returns:
            List of PageContent objects representing extracted pages/sections.

        Raises:
            EmptyFileError: If file is 0 bytes.
            UnsupportedFileTypeError: If file extension is not supported.
            CorruptedFileError: If file cannot be read or parsed.
            NoTextExtractedError: If file has no extractable text.
        """
        ext = Path(filename).suffix.lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"File format '{ext}' is not supported. Supported formats: {', '.join(cls.SUPPORTED_EXTENSIONS)}"
            )

        # Read bytes
        if isinstance(file_input, (str, Path)):
            path = Path(file_input)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {path}")
            with open(path, "rb") as f:
                file_bytes = f.read()
        elif isinstance(file_input, bytes):
            file_bytes = file_input
        elif hasattr(file_input, "read"):
            file_bytes = file_input.read()
            # If the stream pointer has moved, try to seek back if possible
            if hasattr(file_input, "seek"):
                try:
                    file_input.seek(0)
                except Exception:
                    pass
        else:
            raise ValueError("Invalid file_input type provided.")

        if len(file_bytes) == 0:
            raise EmptyFileError(f"Uploaded file '{filename}' is empty (0 bytes).")

        doc_hash = compute_file_hash(file_bytes)

        if ext == ".pdf":
            pages = cls._process_pdf(file_bytes, filename, doc_hash)
        elif ext == ".txt":
            pages = cls._process_txt(file_bytes, filename, doc_hash)
        elif ext == ".docx":
            pages = cls._process_docx(file_bytes, filename, doc_hash)
        else:
            raise UnsupportedFileTypeError(f"Unsupported extension: {ext}")

        # Check total extractable text across all pages
        total_text = "".join(p.text for p in pages).strip()
        if not total_text:
            raise NoTextExtractedError(
                f"Document '{filename}' contains no extractable text. "
                "If this is a scanned PDF or contains only images, OCR is required."
            )

        return pages

    @classmethod
    def _process_pdf(cls, file_bytes: bytes, filename: str, doc_hash: str) -> List[PageContent]:
        """Extract text from PDF preserving 1-indexed page numbers."""
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except Exception as e:
            raise CorruptedFileError(f"Failed to parse PDF '{filename}'. File may be corrupted or invalid: {str(e)}")

        pages: List[PageContent] = []
        try:
            for page_idx in range(len(doc)):
                page = doc[page_idx]
                page_text = page.get_text("text").strip()
                # Record page even if empty to preserve page structure, but we filter when chunking
                pages.append(
                    PageContent(
                        filename=filename,
                        page=page_idx + 1,  # 1-indexed page numbers
                        text=page_text,
                        doc_hash=doc_hash
                    )
                )
        except Exception as e:
            raise CorruptedFileError(f"Error reading pages from PDF '{filename}': {str(e)}")
        finally:
            doc.close()

        return pages

    @classmethod
    def _process_txt(cls, file_bytes: bytes, filename: str, doc_hash: str) -> List[PageContent]:
        """Extract text from TXT file with encoding detection fallback."""
        text = None
        for encoding in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
            try:
                text = file_bytes.decode(encoding)
                break
            except UnicodeDecodeError:
                continue

        if text is None:
            raise CorruptedFileError(f"Unable to decode text file '{filename}' with supported encodings.")

        text = text.strip()
        return [
            PageContent(
                filename=filename,
                page=1,
                text=text,
                doc_hash=doc_hash
            )
        ]

    @classmethod
    def _process_docx(cls, file_bytes: bytes, filename: str, doc_hash: str) -> List[PageContent]:
        """Extract text and tables from DOCX document."""
        try:
            doc_stream = io.BytesIO(file_bytes)
            doc = docx.Document(doc_stream)
        except Exception as e:
            raise CorruptedFileError(f"Failed to parse DOCX '{filename}'. File may be corrupted: {str(e)}")

        text_parts: List[str] = []

        # Extract paragraphs
        for para in doc.paragraphs:
            para_text = para.text.strip()
            if para_text:
                text_parts.append(para_text)

        # Extract tables
        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    text_parts.append(" | ".join(row_cells))

        full_text = "\n\n".join(text_parts).strip()

        # In Word DOCX, hard page boundaries are dynamic based on font/margins.
        # We deliver the document structured with page 1 as the primary container.
        return [
            PageContent(
                filename=filename,
                page=1,
                text=full_text,
                doc_hash=doc_hash
            )
        ]
