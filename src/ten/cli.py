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
) -> None:
    """Walk FOLDER recursively, chunk videos, embed, and upsert to Qdrant."""
    from .ingest import ingest_folder

    ingest_folder(folder.resolve(), force=force, max_videos=max_videos)


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language query."),
    limit: int = typer.Option(20, "--limit", "-n"),
    image: Optional[Path] = typer.Option(None, "--image", help="Optional image-as-query (RRF-fused)."),
    video: Optional[Path] = typer.Option(None, "--video", help="Optional video-as-query (RRF-fused)."),
) -> None:
    """Search the index. Combine --image/--video with the text query for hybrid retrieval."""
    from .search import Searcher

    s = Searcher()
    hits = s.search(text=query, image_path=image, video_path=video, limit=limit)

    table = Table(title=f'"{query}" — top {len(hits)}', show_lines=False)
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
    """Run the FastAPI server (and serve the React UI if built into ui/dist)."""
    import uvicorn

    uvicorn.run("ten.api:app", host=host, port=port, reload=reload, log_level="info")


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
