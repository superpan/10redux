"""Cross-encoder reranker.

Sits between the bi-encoder retrieval (Qwen3-Embedding cosine search) and the
final result cut. The bi-encoder is fast but coarse — it embeds query and
document independently and matches by cosine similarity. A cross-encoder
sees the (query, document) pair jointly and produces a more accurate score,
at the cost of running the model once per candidate.

Workflow at query time:
  1. Bi-encoder fetches top-K candidates (K = `reranker_top_k`, default 100).
  2. For each candidate, build a (query, candidate_text) pair.
  3. Cross-encoder scores the pairs in one batched forward pass.
  4. Sort by cross-encoder score, return the top `limit`.

Default model: Qwen3-Reranker-0.6B (pairs with our Qwen3-Embedding-0.6B).
Switch via TEN_RERANKER_MODEL.
"""
from __future__ import annotations

import os
import threading
from typing import Protocol

from .config import CONFIG


class RerankerProtocol(Protocol):
    def score(self, query: str, candidate_texts: list[str]) -> list[float]: ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder wrapper.

    Works with any encoder-decoder reranker exposed via the CrossEncoder API,
    including Qwen3-Reranker, BGE-Reranker-v2, and friends.
    """

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or CONFIG.reranker_model
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_id, device=CONFIG.device)

    def score(self, query: str, candidate_texts: list[str]) -> list[float]:
        if not candidate_texts:
            return []
        self._ensure_loaded()
        pairs = [(query, t) for t in candidate_texts]
        scores = self._model.predict(pairs, show_progress_bar=False)
        return [float(s) for s in scores]


def make_reranker() -> RerankerProtocol | None:
    """Return a reranker per TEN_RERANKER_BACKEND, or None if disabled."""
    backend = os.environ.get("TEN_RERANKER_BACKEND", CONFIG.reranker_backend).lower()
    if backend in ("", "none", "off", "disabled", "false", "0"):
        return None
    if backend in ("crossencoder", "cross-encoder", "qwen3", "bge"):
        return CrossEncoderReranker()
    raise ValueError(
        f"Unknown TEN_RERANKER_BACKEND: {backend!r} "
        "(expected 'none' | 'crossencoder')"
    )
