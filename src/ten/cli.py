"""`ten` CLI — index, search, serve."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .config import CONFIG

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Open-weight video search.")
console = Console()


@app.command()
def index(
    folder: Path = typer.Argument(..., exists=True, file_okay=False, dir_okay=True),
    force: bool = typer.Option(False, "--force", help="Re-embed clips even if already indexed."),
    max_videos: Optional[int] = typer.Option(None, "--max-videos", help="Limit videos for testing."),
    asr: bool = typer.Option(
        False,
        "--asr",
        help="Enable Whisper transcripts (large-v3). Adds ~real-time per clip on GPU.",
    ),
    clap: bool = typer.Option(
        False,
        "--clap",
        help="Enable LAION CLAP audio embeddings (third Qdrant collection: ten_audio).",
    ),
) -> None:
    """Walk FOLDER recursively, chunk videos, embed, and upsert to Qdrant."""
    import os

    if asr and os.environ.get("TEN_ASR_BACKEND", "").lower() in ("", "none"):
        os.environ["TEN_ASR_BACKEND"] = "whisper"
    if clap and os.environ.get("TEN_CLAP_BACKEND", "").lower() in ("", "none"):
        os.environ["TEN_CLAP_BACKEND"] = "clap"
    from .ingest import ingest_folder

    ingest_folder(folder.resolve(), force=force, max_videos=max_videos)


@app.command()
def search(
    query: Optional[str] = typer.Argument(
        None,
        help="Natural-language query. Optional if --image/--video is given.",
    ),
    limit: int = typer.Option(20, "--limit", "-n"),
    image: Optional[Path] = typer.Option(None, "--image", help="Image-as-query (RRF-fused with text if both given)."),
    video: Optional[Path] = typer.Option(None, "--video", help="Video-as-query (RRF-fused with text if both given)."),
) -> None:
    """Search the index. Provide a text query, --image, --video, or any combination."""
    if not any([query, image, video]):
        console.print(
            "[red]provide at least one of:[/red] QUERY  |  --image PATH  |  --video PATH"
        )
        raise typer.Exit(2)
    from .search import Searcher

    s = Searcher()
    hits = s.search(text=query, image_path=image, video_path=video, limit=limit)

    title_parts: list[str] = []
    if query:
        title_parts.append(f'"{query}"')
    if image:
        title_parts.append(f"image:{image.name}")
    if video:
        title_parts.append(f"video:{video.name}")
    title = " + ".join(title_parts) + f" — top {len(hits)}"
    table = Table(title=title, show_lines=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("id", style="dim", no_wrap=True)
    table.add_column("score", justify="right")
    table.add_column("src", justify="center")
    table.add_column("video")
    table.add_column("t")
    table.add_column("caption", overflow="fold")
    for i, h in enumerate(hits, 1):
        p = h.payload
        table.add_row(
            str(i),
            p["clip_id"],
            f"{h.score:.3f}",
            "+".join(h.sources),
            p["video_name"],
            f"{p['t_start']:.1f}-{p['t_end']:.1f}s",
            p["caption"],
        )
    console.print(table)


@app.command()
def serve(
    host: str = typer.Option(CONFIG.host, "--host"),
    port: int = typer.Option(CONFIG.port, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the FastAPI server (and serve the Next.js UI if built into ui/out)."""
    import uvicorn

    uvicorn.run("ten.api:app", host=host, port=port, reload=reload, log_level="info")


@app.command()
def clip(clip_id: str = typer.Argument(..., help="Clip id from `ten search` or `ten clips`.")) -> None:
    """Show the full payload for a single clip (path, timestamps, caption, thumb)."""
    from .store import Store

    payload = Store().get(clip_id)
    if not payload:
        console.print("[red]clip not found[/red]")
        raise typer.Exit(1)
    table = Table(show_header=False, box=None)
    table.add_column("", style="dim")
    table.add_column("")
    for k in ("clip_id", "video_name", "video_path", "t_start", "t_end", "duration", "thumb_path"):
        table.add_row(k, str(payload.get(k, "—")))
    console.print(table)
    console.print()
    console.print("[bold]caption[/bold]")
    console.print(payload.get("caption", "—"))
    transcript = payload.get("transcript", "")
    if transcript:
        console.print()
        console.print("[bold]transcript[/bold]")
        console.print(transcript)


@app.command()
def clips(
    video: Optional[str] = typer.Option(
        None, "--video", help="Filter by exact video filename (e.g. BigBuckBunny_480p.mov)."
    ),
    limit: int = typer.Option(25, "--limit", "-n"),
) -> None:
    """Enumerate indexed clips. Defaults to a 25-clip sample; use --video to filter."""
    from .store import Store

    rows = Store().list_clips(video_name=video, limit=limit)
    if not rows:
        msg = f"no clips found for video={video!r}" if video else "index is empty"
        console.print(f"[yellow]{msg}[/yellow]")
        return
    title = f"clips (video={video})" if video else f"clips — first {len(rows)}"
    table = Table(title=title, show_lines=False)
    table.add_column("id", style="dim", no_wrap=True)
    table.add_column("video")
    table.add_column("t")
    table.add_column("caption", overflow="fold")
    for p in rows:
        cap = p.get("caption", "")
        if len(cap) > 100:
            cap = cap[:100] + "…"
        table.add_row(
            p["clip_id"],
            p["video_name"],
            f"{p['t_start']:.1f}-{p['t_end']:.1f}s",
            cap,
        )
    console.print(table)


@app.command()
def status() -> None:
    """Print Qdrant collection stats and resolved config."""
    from .store import Store

    table = Table(title="ten — config")
    table.add_column("key")
    table.add_column("value")
    for k in [
        "data_dir",
        "qdrant_url",
        "vjepa_model",
        "vlm_model",
        "text_embed_model",
        "asr_backend",
        "asr_model",
        "asr_language",
        "clip_seconds",
        "frames_per_clip",
        "device",
        "dtype",
        "batch_size",
    ]:
        table.add_row(k, str(getattr(CONFIG, k)))
    console.print(table)

    try:
        st = Store().stats()
        t2 = Table(title="qdrant")
        t2.add_column("collection")
        t2.add_column("info")
        for c, info in st.items():
            t2.add_row(c, str(info))
        console.print(t2)
    except Exception as e:
        console.print(f"[red]qdrant unreachable:[/red] {e}")


@app.command()
def summary(
    clip_id: Optional[str] = typer.Argument(
        None,
        help="Single clip id (copy from `ten search`). Mutually exclusive with --range/--clip-ids.",
    ),
    range_: Optional[str] = typer.Option(
        None,
        "--range",
        "-r",
        metavar="VIDEO:T_START-T_END",
        help="Arbitrary time range, e.g. ./videos/foo.mp4:60-180",
    ),
    clip_ids: Optional[str] = typer.Option(
        None,
        "--clip-ids",
        help="Comma-separated clip ids — summarize them as one multi-span input.",
    ),
    dimension: str = typer.Option(
        "narrative",
        "--dimension",
        "-d",
        help="narrative | visual | motion | aesthetic",
    ),
    max_frames: int = typer.Option(32, "--max-frames"),
    max_tokens: int = typer.Option(256, "--max-tokens"),
) -> None:
    """Summarize a clip, a time range, or a set of clips along a chosen dimension."""
    from .store import Store
    from .summarize import DIMENSIONS, Span, make_summarizer

    given = sum(x is not None for x in (clip_id, range_, clip_ids))
    if given != 1:
        console.print(
            "[red]provide exactly one of:[/red] CLIP_ID  |  --range VIDEO:T0-T1  |  --clip-ids id1,id2,..."
        )
        raise typer.Exit(2)
    if dimension not in DIMENSIONS:
        console.print(f"[red]unknown dimension {dimension!r}; expected one of {DIMENSIONS}[/red]")
        raise typer.Exit(2)

    store = Store()
    spans: list[Span] = []
    if clip_id:
        payload = store.get(clip_id)
        if not payload:
            console.print("[red]clip not found[/red]")
            raise typer.Exit(1)
        spans.append(
            Span(
                video_path=Path(payload["video_path"]),
                t_start=float(payload["t_start"]),
                t_end=float(payload["t_end"]),
            )
        )
    elif clip_ids:
        for cid in [c.strip() for c in clip_ids.split(",") if c.strip()]:
            payload = store.get(cid)
            if not payload:
                console.print(f"[red]clip not found: {cid}[/red]")
                raise typer.Exit(1)
            spans.append(
                Span(
                    video_path=Path(payload["video_path"]),
                    t_start=float(payload["t_start"]),
                    t_end=float(payload["t_end"]),
                )
            )
    else:
        assert range_ is not None
        try:
            video_path, range_str = range_.rsplit(":", 1)
            t0_str, t1_str = range_str.split("-", 1)
            t0, t1 = float(t0_str), float(t1_str)
        except Exception:
            console.print("[red]bad --range; expected VIDEO:T_START-T_END[/red]")
            raise typer.Exit(2)
        spans.append(Span(video_path=Path(video_path), t_start=t0, t_end=t1))

    text = make_summarizer().summarize(
        spans, dimension=dimension, max_frames=max_frames, max_tokens=max_tokens
    )
    console.print(text)


if __name__ == "__main__":
    app()
