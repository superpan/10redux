"""Hybrid search: text-side and visual-side retrieval, fused via RRF.

- Text query  -> embed with TextEmbedder, search caption collection.
- Image query -> sample frames, embed with VideoEmbedder, search visual collection.
- Both       -> RRF-merge the two ranked lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embed_audio import AudioEmbedderProtocol, make_audio_embedder
from .embed_text import TextEmbedder
from .embed_video import VideoEmbedder
from .rerank import RerankerProtocol, make_reranker
from .store import Store
from .video import Clip, sample_clip_frames
from .config import CONFIG


_UNSET = object()


@dataclass
class Hit:
    clip_id: str
    score: float
    payload: dict
    sources: list[str]  # ["text"], ["visual"], or ["text","visual"]


def _rrf(
    rank_lists: list[list[tuple[str, dict]]],
    source_names: list[str] | None = None,
    k: int = 60,
) -> list[Hit]:
    """Reciprocal Rank Fusion. Each input list is [(clip_id, payload), ...] in rank order."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    sources: dict[str, list[str]] = {}
    names = source_names or ["text", "visual", "audio"]
    for li, ranked in enumerate(rank_lists):
        src = names[li] if li < len(names) else f"src{li}"
        for rank, (cid, payload) in enumerate(ranked):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            payloads[cid] = payload
            sources.setdefault(cid, []).append(src)
    return [
        Hit(clip_id=cid, score=score, payload=payloads[cid], sources=sources[cid])
        for cid, score in sorted(scores.items(), key=lambda kv: -kv[1])
    ]


class Searcher:
    def __init__(self) -> None:
        self.store = Store()
        self.text_embedder = TextEmbedder()
        self.video_embedder = VideoEmbedder()
        # Lazy: reranker / CLAP are only loaded if their env var is set.
        self._reranker: RerankerProtocol | None | object = _UNSET
        self._audio_embedder: AudioEmbedderProtocol | None | object = _UNSET

    @property
    def reranker(self) -> RerankerProtocol | None:
        if self._reranker is _UNSET:
            self._reranker = make_reranker()
        return self._reranker  # type: ignore[return-value]

    @property
    def audio_embedder(self) -> AudioEmbedderProtocol | None:
        if self._audio_embedder is _UNSET:
            self._audio_embedder = make_audio_embedder()
        return self._audio_embedder  # type: ignore[return-value]

    def search(
        self,
        text: str | None = None,
        image_path: Path | None = None,
        video_path: Path | None = None,
        limit: int = 20,
        per_source: int | None = None,
    ) -> list[Hit]:
        # per_source is the per-collection fetch depth. Must be at least `limit`,
        # otherwise rankings deeper than 50 disappear. Default to max(limit, 50).
        # When a reranker is enabled, also fetch at least reranker_top_k so the
        # reranker has a meaningful candidate pool to rescore.
        if per_source is None:
            per_source = max(50, limit)
        elif per_source < limit:
            per_source = limit
        if self.reranker is not None and text:
            per_source = max(per_source, CONFIG.reranker_top_k)
        rank_lists: list[list[tuple[str, dict]]] = []

        active_sources: list[str] = []

        if text:
            qv = self.text_embedder.embed_query(text)
            text_hits = self.store.search_text(qv, limit=per_source)
            ranked_text = [(h.payload["clip_id"], h.payload) for h in text_hits]
            # Cross-encoder rerank rescues coarse bi-encoder ordering for the text side.
            if self.reranker is not None:
                ranked_text = self._rerank(text, ranked_text)
            rank_lists.append(ranked_text)
            active_sources.append("text")

        if image_path is not None or video_path is not None:
            frames = self._frames_from_query(image_path, video_path)
            vv = self.video_embedder.embed([frames])[0]
            vis_hits = self.store.search_visual(vv, limit=per_source)
            rank_lists.append([(h.payload["clip_id"], h.payload) for h in vis_hits])
            active_sources.append("visual")

        # CLAP audio: used as a *bounded post-fusion reranker*, not a peer source.
        # Equal-weight 3-way RRF (the obvious integration) regresses retrieval
        # because CLAP's text encoder is trained for audio alignment, not general
        # semantic retrieval — adding it via RRF injects rank noise across the
        # whole list. As a reranker scoped to the top-K candidates with a score
        # floor, it can promote audio-relevant clips but never push the right
        # video out of the top-K.
        audio_score_map: dict[str, float] = {}
        if text and self.audio_embedder is not None:
            try:
                qa = self.audio_embedder.embed_text([text])[0]
                # Pull a wide audio pool so the rerank top-K from text+visual is
                # likely to overlap it. Top-K from audio alone almost never
                # intersects top-K from text+visual.
                audio_hits = self.store.search_audio(qa, limit=per_source)
                audio_score_map = {h.payload["clip_id"]: float(h.score) for h in audio_hits}
            except Exception:
                pass

        if not rank_lists:
            raise ValueError("Provide at least one of: text, image_path, video_path")

        if len(rank_lists) == 1:
            hits = [
                Hit(
                    clip_id=cid,
                    score=1.0 / (i + 1),
                    payload=payload,
                    sources=[active_sources[0]],
                )
                for i, (cid, payload) in enumerate(rank_lists[0][:limit])
            ]
        else:
            hits = _rrf(rank_lists, source_names=active_sources)

        if audio_score_map:
            hits = self._audio_rerank(hits, audio_score_map)
        return hits[:limit]

    def _audio_rerank(self, hits: list[Hit], audio_scores: dict[str, float]) -> list[Hit]:
        """Apply CLAP-audio reranking to the top-K candidates only.

        Adds `audio_weight * max(0, audio_cos - threshold)` to each candidate's
        score, then resorts. Items below the head are untouched, so audio can
        only reorder within candidates the visual+text channels already
        agreed are plausible.
        """
        k = CONFIG.audio_rerank_top_k
        head, tail = hits[:k], hits[k:]
        if not head:
            return hits
        thr = CONFIG.audio_rerank_threshold
        w = CONFIG.audio_rerank_weight
        for h in head:
            cos = audio_scores.get(h.clip_id, 0.0)
            if cos > thr:
                h.score += w * (cos - thr)
                if "audio" not in h.sources:
                    h.sources.append("audio")
        head.sort(key=lambda h: -h.score)
        return head + tail

    def _rerank(
        self, query: str, ranked: list[tuple[str, dict]]
    ) -> list[tuple[str, dict]]:
        top_k = CONFIG.reranker_top_k
        head, tail = ranked[:top_k], ranked[top_k:]
        # Use the same text the bi-encoder embedded: caption (+transcript if any).
        texts = [
            f"{p['caption']}\n{p.get('transcript', '')}".strip() for _, p in head
        ]
        scores = self.reranker.score(query, texts)
        ordered = sorted(zip(head, scores), key=lambda x: -x[1])
        return [item for item, _s in ordered] + tail

    def _frames_from_query(self, image_path: Path | None, video_path: Path | None) -> np.ndarray:
        if image_path is not None:
            from PIL import Image as PILImage

            img = np.asarray(PILImage.open(image_path).convert("RGB"))
            # Tile single frame across the temporal dimension; V-JEPA still produces a sane vector.
            from .video import _resize_frame  # type: ignore[attr-defined]

            f = _resize_frame(img, CONFIG.frame_resize)
            return np.stack([f] * CONFIG.frames_per_clip, axis=0)
        assert video_path is not None
        from .video import probe

        meta = probe(video_path)
        clip = Clip(video_path=video_path, t_start=0.0, t_end=min(meta.duration, CONFIG.clip_seconds))
        return sample_clip_frames(clip, CONFIG.frames_per_clip, CONFIG.frame_resize)
