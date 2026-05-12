"""FastAPI backend.

Endpoints:
  GET  /health
  GET  /stats
  GET  /search?q=...&limit=20            -> text search
  POST /search/image  (file upload)      -> image-as-query
  GET  /clip/{clip_id}                   -> payload
  GET  /clip/{clip_id}/summary           -> on-demand long summary (Qwen3-VL)
  GET  /thumb/{clip_id}.jpg              -> JPEG thumbnail
  GET  /clip/{clip_id}/stream            -> HTTP Range stream of original video
"""
from __future__ import annotations

import io
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .caption import CaptionerProtocol, make_captioner
from .config import CONFIG
from .search import Searcher
from .store import Store
from .video import Clip, sample_clip_frames

app = FastAPI(title="ten")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_searcher: Searcher | None = None
_captioner: CaptionerProtocol | None = None
_store: Store | None = None
_summary_cache: dict[str, str] = {}


def searcher() -> Searcher:
    global _searcher
    if _searcher is None:
        _searcher = Searcher()
    return _searcher


def captioner() -> CaptionerProtocol:
    global _captioner
    if _captioner is None:
        _captioner = make_captioner()
    return _captioner


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/stats")
def stats() -> dict:
    return store().stats()


@app.get("/search")
def api_search(q: str = Query(..., min_length=1), limit: int = 20) -> dict:
    hits = searcher().search(text=q, limit=limit)
    return {"query": q, "hits": [_hit_to_json(h) for h in hits]}


@app.post("/search/image")
async def api_search_image(file: UploadFile = File(...), limit: int = 20) -> dict:
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    tmp = CONFIG.data_dir / "_query_uploads"
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / (file.filename or "upload.jpg")
    img.save(path, format="JPEG")
    hits = searcher().search(image_path=path, limit=limit)
    return {"query": f"<image:{file.filename}>", "hits": [_hit_to_json(h) for h in hits]}


@app.get("/clip/{clip_id}")
def api_clip(clip_id: str) -> dict:
    payload = store().get(clip_id)
    if not payload:
        raise HTTPException(404, "clip not found")
    return payload


@app.get("/clip/{clip_id}/summary")
def api_summary(clip_id: str) -> dict:
    if clip_id in _summary_cache:
        return {"clip_id": clip_id, "summary": _summary_cache[clip_id], "cached": True}
    payload = store().get(clip_id)
    if not payload:
        raise HTTPException(404, "clip not found")
    clip = Clip(
        video_path=Path(payload["video_path"]),
        t_start=float(payload["t_start"]),
        t_end=float(payload["t_end"]),
    )
    frames = sample_clip_frames(clip, CONFIG.frames_per_clip, CONFIG.frame_resize)
    summary = captioner().summarize(frames)
    _summary_cache[clip_id] = summary
    return {"clip_id": clip_id, "summary": summary, "cached": False}


@app.get("/thumb/{clip_id}.jpg")
def api_thumb(clip_id: str) -> FileResponse:
    path = CONFIG.thumbs_dir / f"{clip_id}.jpg"
    if not path.exists():
        # Lazy-generate from the stored payload if missing.
        payload = store().get(clip_id)
        if not payload:
            raise HTTPException(404, "clip not found")
        from .video import write_thumbnail

        write_thumbnail(
            Clip(
                video_path=Path(payload["video_path"]),
                t_start=float(payload["t_start"]),
                t_end=float(payload["t_end"]),
            ),
            path,
        )
    return FileResponse(path, media_type="image/jpeg")


@app.get("/clip/{clip_id}/stream")
def api_stream(clip_id: str, request: Request) -> Response:
    payload = store().get(clip_id)
    if not payload:
        raise HTTPException(404, "clip not found")
    return _range_response(Path(payload["video_path"]), request)


def _hit_to_json(h) -> dict:
    return {
        "clip_id": h.clip_id,
        "score": h.score,
        "sources": h.sources,
        "payload": h.payload,
    }


def _range_response(path: Path, request: Request) -> Response:
    """Serve a file with HTTP Range support so <video> can seek without buffering the whole file."""
    if not path.exists():
        raise HTTPException(404, "video file missing on disk")
    file_size = path.stat().st_size
    range_header = request.headers.get("range")
    media_type = "video/mp4"  # browsers tolerate this for most container types

    if range_header is None:
        return FileResponse(path, media_type=media_type)

    # bytes=start-end
    try:
        units, rng = range_header.split("=", 1)
        if units.strip() != "bytes":
            raise ValueError
        start_s, end_s = rng.split("-", 1)
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except Exception:
        raise HTTPException(416, "invalid Range")

    end = min(end, file_size - 1)
    length = end - start + 1
    chunk = 1024 * 256

    def iter_file():
        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                buf = f.read(min(chunk, remaining))
                if not buf:
                    break
                remaining -= len(buf)
                yield buf

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    return StreamingResponse(iter_file(), status_code=206, media_type=media_type, headers=headers)


# Serve the built React frontend if present (mounted last so /api routes win).
_ui_dist = Path(__file__).resolve().parent.parent.parent / "ui" / "dist"
if _ui_dist.exists():
    app.mount("/", StaticFiles(directory=_ui_dist, html=True), name="ui")
