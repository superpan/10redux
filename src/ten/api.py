"""FastAPI backend.

Endpoints:
  GET  /health
  GET  /stats
  GET  /search?q=...&limit=20                       -> text search
  POST /search/image  (file upload)                 -> image-as-query
  GET  /clip/{clip_id}                              -> payload
  GET  /clip/{clip_id}/summary?dimension=narrative  -> on-demand summary
  GET  /summary?video=...&t_start=...&t_end=...     -> arbitrary-range summary
  POST /summary  {"spans":[{...}],"dimension":"..."} -> multi-span summary
  GET  /thumb/{clip_id}.jpg                         -> JPEG thumbnail
  GET  /clip/{clip_id}/stream                       -> HTTP Range stream
"""
from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

from .config import CONFIG
from .search import Searcher
from .store import Store
from .summarize import DIMENSIONS, Span, Summarizer, make_summarizer
from .video import Clip

# MCP server is imported here so its lifespan can be wired into FastAPI.
# Falling back to None lets the rest of the API keep working if mcp is missing
# or fails to import (e.g. partial install).
try:
    from .mcp_server import mcp as _mcp_server
except Exception as _mcp_import_err:
    _mcp_server = None
    logging.getLogger("ten.api").warning(
        "MCP server import failed; /mcp routes disabled: %s", _mcp_import_err
    )


@asynccontextmanager
async def _lifespan(app: FastAPI):  # type: ignore[no-redef]
    """FastMCP's streamable_http transport runs its session manager inside a
    task group; without an active lifespan the first /mcp request blows up
    with "Task group is not initialized". Chain it into FastAPI's lifespan.
    """
    if _mcp_server is None:
        yield
        return
    async with _mcp_server.session_manager.run():
        yield


app = FastAPI(title="ten", lifespan=_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_searcher: Searcher | None = None
_summarizer: Summarizer | None = None
_store: Store | None = None


def searcher() -> Searcher:
    global _searcher
    if _searcher is None:
        _searcher = Searcher()
    return _searcher


def summarizer() -> Summarizer:
    global _summarizer
    if _summarizer is None:
        _summarizer = make_summarizer()
    return _summarizer


def store() -> Store:
    global _store
    if _store is None:
        _store = Store()
    return _store


class SpanIn(BaseModel):
    video: str = Field(..., description="Absolute or repo-relative path to the video file")
    t_start: float = Field(..., ge=0)
    t_end: float = Field(..., gt=0)


class SummaryIn(BaseModel):
    spans: list[SpanIn] = Field(..., min_length=1)
    dimension: str = "narrative"
    max_frames: int = Field(32, ge=1, le=128)
    max_tokens: int = Field(256, ge=16, le=1024)


def _validate_dimension(dimension: str) -> str:
    if dimension not in DIMENSIONS:
        raise HTTPException(400, f"unknown dimension; expected one of {list(DIMENSIONS)}")
    return dimension


def _spans_from_clip_id(clip_id: str) -> list[Span]:
    payload = store().get(clip_id)
    if not payload:
        raise HTTPException(404, "clip not found")
    return [
        Span(
            video_path=Path(payload["video_path"]),
            t_start=float(payload["t_start"]),
            t_end=float(payload["t_end"]),
        )
    ]


@app.get("/health")
def health() -> dict:
    return {"ok": True}


@app.get("/stats")
def stats() -> dict:
    return store().stats()


@app.get("/search")
def api_search(
    q: str = Query(..., min_length=1),
    limit: int = 20,
    library: str | None = Query(None, description="Filter to a single library (e.g. 'personal')."),
) -> dict:
    hits = searcher().search(text=q, limit=limit, library=library)
    return {"query": q, "library": library, "hits": [_hit_to_json(h) for h in hits]}


@app.post("/search/image")
async def api_search_image(
    file: UploadFile = File(...),
    limit: int = 20,
    library: str | None = Query(None),
) -> dict:
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    tmp = CONFIG.data_dir / "_query_uploads"
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / (file.filename or "upload.jpg")
    img.save(path, format="JPEG")
    hits = searcher().search(image_path=path, limit=limit, library=library)
    return {"query": f"<image:{file.filename}>", "library": library, "hits": [_hit_to_json(h) for h in hits]}


@app.get("/libraries")
def api_libraries() -> dict:
    return {"libraries": store().list_libraries()}


@app.get("/clip/{clip_id}")
def api_clip(clip_id: str) -> dict:
    payload = store().get(clip_id)
    if not payload:
        raise HTTPException(404, "clip not found")
    return payload


@app.get("/clip/{clip_id}/summary")
def api_clip_summary(clip_id: str, dimension: str = "narrative") -> dict:
    dim = _validate_dimension(dimension)
    spans = _spans_from_clip_id(clip_id)
    text = summarizer().summarize(spans, dimension=dim)
    return {"clip_id": clip_id, "dimension": dim, "summary": text}


@app.get("/summary")
def api_summary_range(
    video: str = Query(..., description="Path to the video file"),
    t_start: float = Query(..., ge=0),
    t_end: float = Query(..., gt=0),
    dimension: str = "narrative",
    max_frames: int = Query(32, ge=1, le=128),
    max_tokens: int = Query(256, ge=16, le=1024),
) -> dict:
    dim = _validate_dimension(dimension)
    if t_end <= t_start:
        raise HTTPException(400, "t_end must be greater than t_start")
    span = Span(video_path=Path(video), t_start=t_start, t_end=t_end)
    text = summarizer().summarize(
        [span], dimension=dim, max_frames=max_frames, max_tokens=max_tokens
    )
    return {"dimension": dim, "summary": text, "spans": [span.cache_key()]}


@app.post("/summary")
def api_summary_multi(req: SummaryIn) -> dict:
    dim = _validate_dimension(req.dimension)
    spans = [
        Span(video_path=Path(s.video), t_start=s.t_start, t_end=s.t_end) for s in req.spans
    ]
    text = summarizer().summarize(
        spans, dimension=dim, max_frames=req.max_frames, max_tokens=req.max_tokens
    )
    return {"dimension": dim, "summary": text, "spans": [s.cache_key() for s in spans]}


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
    # `no-cache` = must revalidate before using cached copy. Pairs with the
    # Last-Modified header FileResponse sets so the browser issues an
    # If-Modified-Since and the server replies 304 if the thumb is unchanged.
    # Without this, Safari and Chrome aggressively cache JPEGs and never
    # notice when we regenerate a thumb (e.g., post-rotation backfill).
    return FileResponse(
        path,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-cache, must-revalidate"},
    )


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


# MCP server — exposes search / inspect_clip / list_clips / list_libraries /
# summarize_clip / index_status as MCP tools at /mcp/. Mount before the static
# UI so its routes win over the catch-all `/` mount below. Lifespan wiring
# (see `_lifespan` near the top) is what keeps the streamable_http session
# manager alive across requests.
if _mcp_server is not None:
    app.mount("/mcp", _mcp_server.streamable_http_app())


# Serve the built React frontend if present (mounted last so /api routes win).
# Next.js static export goes to ui/out (not ui/dist). Keep the dist fallback for
# anyone still on the old Vite build.
_repo_root = Path(__file__).resolve().parent.parent.parent
for _candidate in ("out", "dist"):
    _ui_dir = _repo_root / "ui" / _candidate
    if _ui_dir.exists():
        app.mount("/", StaticFiles(directory=_ui_dir, html=True), name="ui")
        break
