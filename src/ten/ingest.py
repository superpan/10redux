"""Ingest orchestrator: walk a folder of videos, chunk, embed, caption, upsert.

Resumable: skips clips whose id already exists in Qdrant.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .caption import CaptionerProtocol, make_captioner
from .config import CONFIG
from .embed_text import TextEmbedder
from .embed_video import VideoEmbedder
from .store import ClipPayload, Store
from .video import (
    Clip,
    enumerate_clips,
    probe,
    sample_clip_frames,
    walk_videos,
    write_thumbnail,
)

console = Console()


def _thumb_path(clip_id: str) -> Path:
    return CONFIG.thumbs_dir / f"{clip_id}.jpg"


def ingest_folder(root: Path, force: bool = False, max_videos: int | None = None) -> dict:
    CONFIG.ensure_dirs()
    videos = walk_videos(root)
    if max_videos:
        videos = videos[:max_videos]
    if not videos:
        console.print(f"[yellow]No videos found under {root}[/yellow]")
        return {"videos": 0, "clips": 0, "skipped": 0}

    console.print(f"[bold]Found {len(videos)} videos[/bold] under {root}")

    # Lazy-load models on first clip; pre-load store metadata immediately.
    video_embedder = VideoEmbedder()
    text_embedder = TextEmbedder()
    captioner = make_captioner()
    store = Store()

    # Sniff dims (forces model load) so we can create collections up-front.
    visual_dim = video_embedder.dim
    text_dim = text_embedder.dim
    store.ensure_collections(visual_dim=visual_dim, text_dim=text_dim)

    # Plan clips up-front for accurate progress.
    plan: list[Clip] = []
    for vp in videos:
        try:
            meta = probe(vp)
        except Exception as e:
            console.print(f"[red]probe failed:[/red] {vp}: {e}")
            continue
        plan.extend(enumerate_clips(meta, CONFIG.clip_seconds, CONFIG.clip_overlap))

    console.print(f"[bold]Planned {len(plan)} clips[/bold] (clip={CONFIG.clip_seconds}s, overlap={CONFIG.clip_overlap}s)")

    skipped = 0
    processed = 0
    batch: list[tuple[Clip, np.ndarray]] = []

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )

    with progress:
        task = progress.add_task("ingesting", total=len(plan))
        for clip in plan:
            if not force and store.has_clip(clip.clip_id):
                skipped += 1
                progress.advance(task)
                continue
            try:
                frames = sample_clip_frames(clip, CONFIG.frames_per_clip, CONFIG.frame_resize)
            except Exception as e:
                console.print(f"[red]decode failed:[/red] {clip.video_path} {clip.t_start:.1f}-{clip.t_end:.1f}: {e}")
                progress.advance(task)
                continue
            batch.append((clip, frames))
            if len(batch) >= CONFIG.batch_size:
                processed += _flush(batch, video_embedder, text_embedder, captioner, store)
                batch.clear()
            progress.advance(task)
        if batch:
            processed += _flush(batch, video_embedder, text_embedder, captioner, store)

    console.print(
        f"[green]Done.[/green] processed={processed} skipped={skipped} total={len(plan)}"
    )
    return {"videos": len(videos), "clips": len(plan), "processed": processed, "skipped": skipped}


def _flush(
    batch: list[tuple[Clip, np.ndarray]],
    video_embedder: VideoEmbedder,
    text_embedder: TextEmbedder,
    captioner: CaptionerProtocol,
    store: Store,
) -> int:
    clips = [b[0] for b in batch]
    frames_list = [b[1] for b in batch]

    # Visual embeddings (batched).
    visual_vecs = video_embedder.embed(frames_list)

    # Captions (sequential; VLM dominated by attention).
    captions = captioner.caption_batch(frames_list)

    # Text embeddings (batched).
    text_vecs = text_embedder.embed_passages(captions)

    # Thumbnails + payloads.
    payloads: list[ClipPayload] = []
    for clip, caption in zip(clips, captions, strict=True):
        thumb = _thumb_path(clip.clip_id)
        if not thumb.exists():
            try:
                write_thumbnail(clip, thumb)
            except Exception:
                pass
        payloads.append(
            ClipPayload(
                clip_id=clip.clip_id,
                video_path=str(clip.video_path.resolve()),
                video_name=clip.video_path.name,
                t_start=clip.t_start,
                t_end=clip.t_end,
                duration=clip.t_end - clip.t_start,
                caption=caption,
                thumb_path=str(thumb.resolve()),
            )
        )

    store.upsert(payloads, visual_vecs, text_vecs)
    return len(payloads)
