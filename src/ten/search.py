"""Hybrid search: text-side and visual-side retrieval, fused via RRF.

- Text query  -> embed with TextEmbedder, search caption collection.
- Image query -> sample frames, embed with VideoEmbedder, search visual collection.
- Both       -> RRF-merge the two ranked lists.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .embed_text import TextEmbedder
from .embed_video import VideoEmbedder
from .store import Store
from .video import Clip, sample_clip_frames
from .config import CONFIG


@dataclass
class Hit:
    clip_id: str
    score: float
    payload: dict
    sources: list[str]  # ["text"], ["visual"], or ["text","visual"]


def _rrf(rank_lists: list[list[tuple[str, dict]]], k: int = 60) -> list[Hit]:
    """Reciprocal Rank Fusion. Each input list is [(clip_id, payload), ...] in rank order."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}
    sources: dict[str, list[str]] = {}
    source_names = ["text", "visual"]
    for li, ranked in enumerate(rank_lists):
        src = source_names[li] if li < len(source_names) else f"src{li}"
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
        if per_source is None:
            per_source = max(50, limit)
        elif per_source < limit:
            per_source = limit
        rank_lists: list[list[tuple[str, dict]]] = []

        if text:
            qv = self.text_embedder.embed_query(text)
            text_hits = self.store.search_text(qv, limit=per_source)
            rank_lists.append([(h.payload["clip_id"], h.payload) for h in text_hits])

        if image_path is not None or video_path is not None:
            frames = self._frames_from_query(image_path, video_path)
            vv = self.video_embedder.embed([frames])[0]
            vis_hits = self.store.search_visual(vv, limit=per_source)
            rank_lists.append([(h.payload["clip_id"], h.payload) for h in vis_hits])

        if not rank_lists:
            raise ValueError("Provide at least one of: text, image_path, video_path")

        if len(rank_lists) == 1:
            return [
                Hit(
                    clip_id=cid,
                    score=1.0 / (i + 1),
                    payload=payload,
                    sources=["text" if text else "visual"],
                )
                for i, (cid, payload) in enumerate(rank_lists[0][:limit])
            ]

        return _rrf(rank_lists)[:limit]

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
