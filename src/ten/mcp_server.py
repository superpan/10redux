"""MCP server exposing ten's search/clip/summary primitives to remote agents.

Mounted on the existing FastAPI app at `/mcp` (see `api.py`), so a Claude
client elsewhere on the tailnet adds the server with:

    claude mcp add ten --transport http --url http://spark-297f:8765/mcp/

The transport is **streamable HTTP** (the current MCP transport — single
endpoint, supersedes the old HTTP+SSE split). No auth at the protocol
level; the implicit guard is whatever fronts the FastAPI app (Tailscale ACL,
reverse proxy, etc.).

Tool surface mirrors the CLI / REST primitives so an agent that drives ten
through MCP has the same affordances a human does:

- `search`          — text query against the index, scope-able via `library`
- `inspect_clip`    — full payload (caption, transcript, paths, timestamps)
- `list_clips`      — enumerate clips, optionally filtered by video filename
- `list_libraries`  — distinct `library` tags currently in the index
- `summarize_clip`  — re-decode the clip and ask Qwen3-VL for a paragraph
- `index_status`    — Qdrant collection point counts

Tool results that reference clip data return *URLs* (thumb / clip stream)
instead of binary payloads — the client can fetch via normal HTTP using
the same FastAPI endpoints the UI already uses.
"""
from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from .search import Searcher
from .store import Store
from .summarize import Span, make_summarizer

mcp = FastMCP(
    "ten",
    instructions=(
        "Open-weight video search over ~10s clips. Use `search` for "
        "natural-language queries; the returned clip_ids feed into "
        "`inspect_clip`, `summarize_clip`, and the thumb / stream URLs."
    ),
    # FastMCP defaults its internal route to "/mcp"; we mount the resulting
    # ASGI app on FastAPI at "/mcp", which would double-prefix to /mcp/mcp.
    # Setting this to "/" makes the effective URL just /mcp/ on the parent.
    streamable_http_path="/",
)

# Lazy singletons. FastMCP runs tools in a thread pool, so first-call latency
# (model loads) hits the calling tool, not server start. Same behaviour as the
# REST endpoints.
_store: Store | None = None
_searcher: Searcher | None = None


def _ensure() -> tuple[Store, Searcher]:
    global _store, _searcher
    if _store is None:
        _store = Store()
    if _searcher is None:
        _searcher = Searcher()
    return _store, _searcher


@mcp.tool()
def search(query: str, limit: int = 10, library: str | None = None) -> list[dict]:
    """Search the video index by natural-language text.

    Returns up to `limit` hits, each with clip_id, score, sources (which
    retrieval channels contributed), the captioner's caption, ASR transcript
    if any, video filename, time window, and URLs for thumb + stream.

    Pass `library` (e.g. "personal") to scope the search to one ingest folder.
    Use `list_libraries` to see what's available.
    """
    _, searcher = _ensure()
    hits = searcher.search(text=query, limit=limit, library=library)
    return [
        {
            "clip_id": h.clip_id,
            "score": round(float(h.score), 4),
            "sources": list(h.sources),
            "library": h.payload.get("library", ""),
            "video_name": h.payload.get("video_name", ""),
            "t_start": float(h.payload["t_start"]),
            "t_end": float(h.payload["t_end"]),
            "caption": h.payload.get("caption", ""),
            "transcript": (h.payload.get("transcript") or "")[:240],
            "thumb_url": f"/thumb/{h.clip_id}.jpg",
            "clip_url": f"/clip/{h.clip_id}/stream",
        }
        for h in hits
    ]


@mcp.tool()
def inspect_clip(clip_id: str) -> dict:
    """Return the full Qdrant payload for a clip (caption, transcript,
    paths, timestamps, library, thumb path)."""
    store, _ = _ensure()
    payload = store.get(clip_id)
    if not payload:
        return {"error": "clip not found", "clip_id": clip_id}
    return payload


@mcp.tool()
def list_clips(video_name: str | None = None, limit: int = 25) -> list[dict]:
    """Enumerate indexed clips. With `video_name` (exact filename match)
    returns that video's clips in time order — useful for browsing a single
    video. Without it, returns up to `limit` clips from the index."""
    store, _ = _ensure()
    return store.list_clips(video_name=video_name, limit=limit)


@mcp.tool()
def list_libraries() -> list[str]:
    """Distinct `library` tags currently in the index. Each tag is the
    parent directory name of the videos ingested into it."""
    store, _ = _ensure()
    return store.list_libraries()


@mcp.tool()
def summarize_clip(clip_id: str, dimension: str = "narrative") -> str:
    """Re-decode the clip and ask Qwen3-VL for a paragraph-length summary.
    `dimension` is one of: narrative, visual, motion, aesthetic."""
    store, _ = _ensure()
    payload = store.get(clip_id)
    if not payload:
        return f"clip not found: {clip_id}"
    span = Span(
        video_path=Path(payload["video_path"]),
        t_start=float(payload["t_start"]),
        t_end=float(payload["t_end"]),
    )
    return make_summarizer().summarize([span], dimension=dimension)


@mcp.tool()
def index_status() -> dict:
    """Qdrant collection point counts and per-library distribution."""
    store, _ = _ensure()
    from collections import Counter

    counts = store.stats()
    libs: Counter[str] = Counter()
    offset = None
    while True:
        pts, offset = store.client.scroll(
            collection_name="ten_visual",
            limit=1024,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in pts:
            libs[(p.payload or {}).get("library") or "?"] += 1
        if offset is None:
            break
    return {"collections": counts, "by_library": dict(libs.most_common())}
