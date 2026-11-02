"""
Tests for DocumentProcessor: PDF, TXT, DOCX extraction, error handling, and metadata preservation.
"""

import io
import pytest
from pathlib import Path
import pymupdf as fitz
import docx

from src.document_processor import (
    DocumentProcessor,
    PageContent,
    EmptyFileError,
    CorruptedFileError,
    NoTextExtractedError,
    UnsupportedFileTypeError,
    compute_file_hash
)


def create_dummy_pdf(pages_text: list) -> bytes:
    """Helper to generate a real valid multi-page PDF in memory."""
    doc = fitz.open()
    for text in pages_text:
        page = doc.new_page()
        page.insert_text((50, 50), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_dummy_docx(paragraphs: list, tables_data: list = None) -> bytes:
    """Helper to generate a real valid DOCX in memory."""
    doc = docx.Document()
    for p in paragraphs:
        doc.add_paragraph(p)
    if tables_data:
        table = doc.add_table(rows=len(tables_data), cols=len(tables_data[0]))
        for r_idx, row in enumerate(tables_data):
            for c_idx, val in enumerate(row):
                table.cell(r_idx, c_idx).text = val
    stream = io.BytesIO()
    doc.save(stream)
    return stream.getvalue()


class TestDocumentProcessor:

    def test_pdf_extraction_multi_page(self):
        """Verify multi-page PDF extraction and 1-indexed page number preservation."""
        page_texts = [
            "This is Page 1 containing information on ResearchMate architecture.",
            "This is Page 2 detailing the FAISS vector database and cosine similarity."
        ]
        pdf_bytes = create_dummy_pdf(page_texts)

        pages = DocumentProcessor.process_file(pdf_bytes, "test_doc.pdf")
        assert len(pages) == 2
        assert pages[0].filename == "test_doc.pdf"
        assert pages[0].page == 1
        assert "Page 1" in pages[0].text
        assert pages[1].page == 2
        assert "Page 2" in pages[1].text
        assert pages[0].doc_hash == compute_file_hash(pdf_bytes)

    def test_txt_extraction_utf8(self):
        """Verify TXT file extraction with UTF-8 encoding."""
        sample_text = "ResearchMate is an AI document assistant designed for general-purpose RAG."
        txt_bytes = sample_text.encode("utf-8")

        pages = DocumentProcessor.process_file(txt_bytes, "notes.txt")
        assert len(pages) == 1
        assert pages[0].filename == "notes.txt"
        assert pages[0].page == 1
        assert sample_text in pages[0].text
        assert pages[0].doc_hash == compute_file_hash(txt_bytes)

    def test_txt_extraction_latin1_fallback(self):
        """Verify TXT fallback to Latin-1 encoding for non-UTF8 characters."""
        sample_text = "Café policies and résumé guidelines for employees."
        txt_bytes = sample_text.encode("latin-1")

        pages = DocumentProcessor.process_file(txt_bytes, "policy.txt")
        assert len(pages) == 1
        assert "policies" in pages[0].text

    def test_docx_extraction_paragraphs_and_tables(self):
        """Verify DOCX extraction handles paragraphs and embedded tables."""
        paragraphs = ["Company Employee Handbook 2026.", "Section 1: Working Hours and Remote Policies."]
        tables = [["Policy", "Allowance"], ["Remote Stipend", "$500 annually"]]
        docx_bytes = create_dummy_docx(paragraphs, tables)

        pages = DocumentProcessor.process_file(docx_bytes, "handbook.docx")
        assert len(pages) == 1
        assert "Company Employee Handbook" in pages[0].text
        assert "Remote Stipend | $500 annually" in pages[0].text
        assert pages[0].page == 1

    def test_empty_file_error(self):
        """Verify EmptyFileError is raised when file contains 0 bytes."""
        with pytest.raises(EmptyFileError):
            DocumentProcessor.process_file(b"", "empty.pdf")

    def test_unsupported_file_type_error(self):
        """Verify UnsupportedFileTypeError for extensions other than pdf, txt, docx."""
        with pytest.raises(UnsupportedFileTypeError):
            DocumentProcessor.process_file(b"some content", "data.csv")

    def test_corrupted_pdf_error(self):
        """Verify CorruptedFileError is raised on damaged PDF headers."""
        corrupted_bytes = b"%PDF-1.4 completely invalid stream data that cannot be parsed"
        with pytest.raises(CorruptedFileError):
            DocumentProcessor.process_file(corrupted_bytes, "corrupt.pdf")

    def test_no_text_extracted_error(self):
        """Verify NoTextExtractedError is raised for blank or non-text files."""
        # Create a PDF with blank pages (no text)
        doc = fitz.open()
        doc.new_page()  # empty page
        blank_pdf = doc.tobytes()
        doc.close()

        with pytest.raises(NoTextExtractedError):
            DocumentProcessor.process_file(blank_pdf, "blank.pdf")
