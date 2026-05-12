"""Text embedder for the caption side of the hybrid index.

Uses sentence-transformers to load Qwen/Qwen3-Embedding-0.6B (or any
sentence-transformers compatible model via TEN_TEXT_EMBED_MODEL).
"""
from __future__ import annotations

import threading

import numpy as np

from .config import CONFIG


class TextEmbedder:
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or CONFIG.text_embed_model
        self._model = None
        self._lock = threading.Lock()
        self._dim: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_id, device=CONFIG.device)
            self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self._ensure_loaded()
        vecs = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        # Qwen3-Embedding wants a task instruction prefix for queries.
        prompt = (
            "Instruct: Given a search query, retrieve video clip captions that match.\nQuery: "
            + text
        )
        return self.embed_passages([prompt])[0]
