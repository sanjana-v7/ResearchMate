"""
ResearchMate - Vector Store Module
Manages FAISS index and chunk metadata mapping.
Supports incremental multi-document indexing, duplicate detection,
individual document removal with index rebuilding, and persistence.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import faiss

from src.text_splitter import TextChunk
from src.embeddings import EmbeddingManager
from config import INDEX_PATH, METADATA_PATH, EMBEDDING_DIMENSION


class VectorStore:
    """FAISS-backed vector database with synchronized metadata and document lifecycle operations."""

    def __init__(
        self,
        dimension: int = EMBEDDING_DIMENSION,
        index_path: Optional[Path] = INDEX_PATH,
        metadata_path: Optional[Path] = METADATA_PATH
    ):
        self.dimension = dimension
        self.index_path = Path(index_path) if index_path else None
        self.metadata_path = Path(metadata_path) if metadata_path else None
        self.embedding_manager = EmbeddingManager()

        # Initialize empty IndexFlatIP (Inner Product = Cosine Similarity for normalized vectors)
        self.index: faiss.Index = faiss.IndexFlatIP(self.dimension)
        # List of metadata dicts corresponding 1:1 with vector IDs (0 to N-1)
        self.chunks_metadata: List[Dict[str, Any]] = []

    @property
    def total_vectors(self) -> int:
        """Return total number of vectors in FAISS index."""
        return self.index.ntotal if self.index else 0

    def get_indexed_documents(self) -> List[str]:
        """Return unique list of filenames currently stored in the index."""
        seen = []
        for meta in self.chunks_metadata:
            fname = meta.get("filename")
            if fname and fname not in seen:
                seen.append(fname)
        return seen

    def get_document_hashes(self) -> Set[str]:
        """Return set of document hashes currently stored in the index."""
        return {meta.get("doc_hash") for meta in self.chunks_metadata if meta.get("doc_hash")}

    def is_duplicate(self, filename: str, doc_hash: str) -> Tuple[bool, str]:
        """
        Check if a document is already present in the index.

        Returns:
            Tuple of (is_duplicate: bool, reason: str)
        """
        current_hashes = self.get_document_hashes()
        if doc_hash in current_hashes:
            return True, f"Document with matching content hash already indexed."

        current_files = self.get_indexed_documents()
        if filename in current_files:
            return True, f"Document named '{filename}' is already indexed."

        return False, ""

    def add_chunks(self, chunks: List[TextChunk]) -> int:
        """
        Embed and index a list of text chunks.

        Args:
            chunks: List of TextChunk objects to be indexed.

        Returns:
            Number of chunks successfully added.
        """
        if not chunks:
            return 0

        # Extract texts for embedding
        texts = [c.text for c in chunks]
        embeddings = self.embedding_manager.embed_texts(texts)

        # Add vectors to FAISS index
        self.index.add(embeddings)

        # Append metadata
        for chunk in chunks:
            self.chunks_metadata.append(chunk.to_dict())

        return len(chunks)

    def similarity_search(
        self,
        query: str,
        top_k: int = 5,
        filter_filename: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute semantic similarity search against the indexed chunks.

        Args:
            query: User's query string.
            top_k: Number of most relevant results to return.
            filter_filename: Optional filter to restrict search to a single document.

        Returns:
            List of dictionaries containing chunk metadata and similarity 'score'.
        """
        if self.total_vectors == 0:
            return []

        # Embed query vector
        query_vector = self.embedding_manager.embed_query(query)

        # If filtering by filename, search more candidates to account for filtering
        search_k = min(self.total_vectors, top_k * 3 if filter_filename else top_k)
        if search_k <= 0:
            return []

        scores, indices = self.index.search(query_vector, search_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks_metadata):
                continue

            meta = self.chunks_metadata[idx].copy()
            # If filename filter is specified, skip non-matching documents
            if filter_filename and meta.get("filename") != filter_filename:
                continue

            meta["score"] = float(score)
            results.append(meta)

            if len(results) >= top_k:
                break

        return results

    def remove_document(self, filename: str) -> bool:
        """
        Remove all chunks associated with a specific filename and rebuild the index.

        Args:
            filename: Name of the document to remove.

        Returns:
            True if document was found and removed, False otherwise.
        """
        remaining_chunks = [
            meta for meta in self.chunks_metadata if meta.get("filename") != filename
        ]

        if len(remaining_chunks) == len(self.chunks_metadata):
            return False  # Document was not present

        # Rebuild index from remaining chunks
        self.clear()
        if remaining_chunks:
            # Reconstruct TextChunk objects
            chunk_objs = [TextChunk.from_dict(m) for m in remaining_chunks]
            self.add_chunks(chunk_objs)

        return True

    def clear(self):
        """Reset the vector index and metadata to an empty state."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks_metadata = []

    def get_document_stats(self) -> Dict[str, Any]:
        """
        Compute statistics across the indexed knowledge base.

        Returns:
            Dict containing total_documents, total_chunks, total_pages, and per-document breakdown.
        """
        doc_details: Dict[str, Dict[str, Any]] = {}

        for meta in self.chunks_metadata:
            fname = meta.get("filename", "Unknown")
            page = meta.get("page", 1)

            if fname not in doc_details:
                doc_details[fname] = {
                    "chunks": 0,
                    "pages": set()
                }

            doc_details[fname]["chunks"] += 1
            doc_details[fname]["pages"].add(page)

        doc_summary = []
        total_pages = 0
        for fname, stats in doc_details.items():
            page_count = len(stats["pages"])
            total_pages += page_count
            doc_summary.append({
                "filename": fname,
                "chunks": stats["chunks"],
                "pages": page_count
            })

        return {
            "total_documents": len(doc_summary),
            "total_chunks": len(self.chunks_metadata),
            "total_pages": total_pages,
            "documents": doc_summary
        }

    def save(self, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None):
        """Persist FAISS index and metadata to disk."""
        target_index = Path(index_path or self.index_path)
        target_meta = Path(metadata_path or self.metadata_path)

        target_index.parent.mkdir(parents=True, exist_ok=True)
        target_meta.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(target_index))
        with open(target_meta, "w", encoding="utf-8") as f:
            json.dump(self.chunks_metadata, f, indent=2, ensure_ascii=False)

    def load(self, index_path: Optional[Path] = None, metadata_path: Optional[Path] = None) -> bool:
        """
        Load FAISS index and metadata from disk if available.

        Returns:
            True if successfully loaded, False otherwise.
        """
        target_index = Path(index_path or self.index_path)
        target_meta = Path(metadata_path or self.metadata_path)

        if not target_index.exists() or not target_meta.exists():
            return False

        try:
            self.index = faiss.read_index(str(target_index))
            with open(target_meta, "r", encoding="utf-8") as f:
                self.chunks_metadata = json.load(f)
            return True
        except Exception:
            self.clear()
            return False
