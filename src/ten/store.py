"""Qdrant store: up to three collections sharing payloads (visual + text + audio).

Each clip has a stable string id (blake2b hex). We store it as the Qdrant
point id (UUID-shaped). Payload is identical across collections so a hit in
any side can render the same UI card.

The `ten_audio` collection is only created when CLAP embeddings are enabled.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from .config import CONFIG


@dataclass
class ClipPayload:
    clip_id: str
    video_path: str
    video_name: str
    t_start: float
    t_end: float
    duration: float
    caption: str
    thumb_path: str
    transcript: str = ""
    library: str = ""  # parent dir name; lets the UI/API scope search to one library
    # Abstract caption of the audio content from the audio LM ("a woman gives
    # a monologue", "instrumental music with guitar"). Empty when ASR-only path
    # is used or no audio LM is configured. Concatenated into the embedded text
    # at ingest time so search hits on it naturally.
    audio_caption: str = ""

    def to_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "video_path": self.video_path,
            "video_name": self.video_name,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "duration": self.duration,
            "caption": self.caption,
            "thumb_path": self.thumb_path,
            "transcript": self.transcript,
            "library": self.library,
            "audio_caption": self.audio_caption,
        }


def clip_id_to_uuid(clip_id: str) -> str:
    # Deterministic UUID from the clip hash so the same clip maps to the same point.
    return str(uuid.UUID(clip_id[:32]))


class Store:
    def __init__(self) -> None:
        self.client = QdrantClient(url=CONFIG.qdrant_url, api_key=CONFIG.qdrant_api_key)

    def ensure_collections(
        self,
        visual_dim: int,
        text_dim: int,
        audio_dim: int | None = None,
    ) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if CONFIG.visual_collection not in existing:
            self.client.create_collection(
                collection_name=CONFIG.visual_collection,
                vectors_config=qm.VectorParams(size=visual_dim, distance=qm.Distance.COSINE),
            )
        if CONFIG.text_collection not in existing:
            self.client.create_collection(
                collection_name=CONFIG.text_collection,
                vectors_config=qm.VectorParams(size=text_dim, distance=qm.Distance.COSINE),
            )
        if audio_dim is not None and CONFIG.audio_collection not in existing:
            self.client.create_collection(
                collection_name=CONFIG.audio_collection,
                vectors_config=qm.VectorParams(size=audio_dim, distance=qm.Distance.COSINE),
            )
        # Index video_path + library on every active collection so we can filter
        # by source path or by library (parent dir name; lets `?library=personal`
        # scope the search to just one ingest folder).
        cols = [CONFIG.visual_collection, CONFIG.text_collection]
        if audio_dim is not None:
            cols.append(CONFIG.audio_collection)
        for col in cols:
            for field in ("video_path", "library"):
                try:
                    self.client.create_payload_index(
                        collection_name=col,
                        field_name=field,
                        field_schema=qm.PayloadSchemaType.KEYWORD,
                    )
                except Exception:
                    pass

    def upsert(
        self,
        payloads: list[ClipPayload],
        visual_vecs: np.ndarray,
        text_vecs: np.ndarray,
        audio_vecs: np.ndarray | None = None,
    ) -> None:
        ids = [clip_id_to_uuid(p.clip_id) for p in payloads]
        payload_dicts = [p.to_dict() for p in payloads]
        self.client.upsert(
            collection_name=CONFIG.visual_collection,
            points=qm.Batch(ids=ids, vectors=visual_vecs.tolist(), payloads=payload_dicts),
        )
        self.client.upsert(
            collection_name=CONFIG.text_collection,
            points=qm.Batch(ids=ids, vectors=text_vecs.tolist(), payloads=payload_dicts),
        )
        if audio_vecs is not None:
            self.client.upsert(
                collection_name=CONFIG.audio_collection,
                points=qm.Batch(ids=ids, vectors=audio_vecs.tolist(), payloads=payload_dicts),
            )

    def has_clip(self, clip_id: str) -> bool:
        try:
            res = self.client.retrieve(
                collection_name=CONFIG.visual_collection,
                ids=[clip_id_to_uuid(clip_id)],
                with_payload=False,
                with_vectors=False,
            )
            return len(res) > 0
        except Exception:
            return False

    def _library_filter(self, library: str | None) -> qm.Filter | None:
        if not library:
            return None
        return qm.Filter(
            must=[qm.FieldCondition(key="library", match=qm.MatchValue(value=library))]
        )

    def search_visual(
        self, vec: np.ndarray, limit: int = 50, library: str | None = None
    ) -> list[qm.ScoredPoint]:
        return self.client.query_points(
            collection_name=CONFIG.visual_collection,
            query=vec.tolist(),
            limit=limit,
            query_filter=self._library_filter(library),
            with_payload=True,
        ).points

    def search_text(
        self, vec: np.ndarray, limit: int = 50, library: str | None = None
    ) -> list[qm.ScoredPoint]:
        return self.client.query_points(
            collection_name=CONFIG.text_collection,
            query=vec.tolist(),
            limit=limit,
            query_filter=self._library_filter(library),
            with_payload=True,
        ).points

    def search_audio(
        self, vec: np.ndarray, limit: int = 50, library: str | None = None
    ) -> list[qm.ScoredPoint]:
        return self.client.query_points(
            collection_name=CONFIG.audio_collection,
            query=vec.tolist(),
            limit=limit,
            query_filter=self._library_filter(library),
            with_payload=True,
        ).points

    def list_libraries(self) -> list[str]:
        """Distinct `library` values currently in the index, sorted alphabetically.

        Used by the UI/API to populate the library dropdown without hardcoding.
        """
        seen: set[str] = set()
        offset = None
        while True:
            pts, offset = self.client.scroll(
                collection_name=CONFIG.visual_collection,
                limit=1024,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in pts:
                lib = (p.payload or {}).get("library")
                if lib:
                    seen.add(lib)
            if offset is None:
                break
        return sorted(seen)

    def get(self, clip_id: str) -> dict | None:
        res = self.client.retrieve(
            collection_name=CONFIG.visual_collection,
            ids=[clip_id_to_uuid(clip_id)],
            with_payload=True,
        )
        return res[0].payload if res else None

    def list_clips(self, video_name: str | None = None, limit: int = 100) -> list[dict]:
        """Enumerate clip payloads, optionally filtered by exact video filename.

        Backed by Qdrant's scroll API. video_name isn't a payload-indexed field
        (only video_path is), but for small collections (<10K) this filter is fine.
        """
        flt = None
        if video_name:
            flt = qm.Filter(
                must=[qm.FieldCondition(key="video_name", match=qm.MatchValue(value=video_name))]
            )
        points, _ = self.client.scroll(
            collection_name=CONFIG.visual_collection,
            scroll_filter=flt,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return [p.payload for p in points]

    def stats(self) -> dict:
        out = {}
        for col in (CONFIG.visual_collection, CONFIG.text_collection):
            try:
                info = self.client.get_collection(col)
                out[col] = {"points": info.points_count, "status": str(info.status)}
            except Exception as e:
                out[col] = {"error": str(e)}
        return out
