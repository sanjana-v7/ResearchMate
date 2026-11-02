"""
ResearchMate - Embeddings Module
Wraps Sentence Transformers (all-MiniLM-L6-v2) for generating normalized
dense vector representations of text chunks and user queries.
"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION


class EmbeddingManager:
    """Singleton-style embedding manager using SentenceTransformer."""

    _instance = None
    _model = None

    def __new__(cls, model_name: str = EMBEDDING_MODEL_NAME):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """Load the SentenceTransformer model onto CPU/device."""
        if self._model is None:
            # We load the specified sentence transformer model
            self._model = SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        return EMBEDDING_DIMENSION

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate L2-normalized embeddings for a list of text strings.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size for model inference.

        Returns:
            2D numpy array of shape (len(texts), EMBEDDING_DIMENSION) with dtype float32.
        """
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)

        # normalize_embeddings=True ensures inner product equals cosine similarity
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate L2-normalized embedding for a single search query.

        Args:
            query: User's query string.

        Returns:
            2D numpy array of shape (1, EMBEDDING_DIMENSION) with dtype float32.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return np.zeros((1, self.dimension), dtype=np.float32)

        embedding = self._model.encode(
            [cleaned_query],
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.astype(np.float32)
