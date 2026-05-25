"""Re-extract thumbnails for clips whose source videos have a display-rotation tag.

The original write_thumbnail did not apply the container's rotation, so any
iPhone-portrait clip ingested before that fix has a sideways JPEG on disk.
This script walks the index, identifies clips whose source video has a
non-zero rotation, and regenerates only those thumbnails.

Captions and visual embeddings are unaffected by this backfill — they were
generated from the same sideways frames at ingest time and need a `--force`
re-ingest to fix. Run that separately if you care about retrieval quality
on portrait-rotated content (see `make reindex FOLDER=...`).

Run:
  uv run python tools/backfill_thumbnails.py
  uv run python tools/backfill_thumbnails.py --library personal     # scope to one library
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ten.config import CONFIG
from ten.video import Clip, _video_rotation, write_thumbnail


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default=None, help="Only backfill thumbnails for this library tag.")
    parser.add_argument("--dry-run", action="store_true", help="Print plan, don't rewrite JPEGs.")
    args = parser.parse_args()

    client = QdrantClient(url=CONFIG.qdrant_url, api_key=CONFIG.qdrant_api_key)

    scroll_filter = None
    if args.library:
        scroll_filter = qm.Filter(
            must=[qm.FieldCondition(key="library", match=qm.MatchValue(value=args.library))]
        )

    # Pass 1: identify which source videos have non-zero rotation.
    video_rotation: dict[str, int] = {}
    clips_by_video: dict[str, list[dict]] = defaultdict(list)
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=CONFIG.visual_collection,
            scroll_filter=scroll_filter,
            limit=1024,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in pts:
            payload = p.payload or {}
            vp = payload.get("video_path")
            if not vp:
                continue
            if vp not in video_rotation:
                video_rotation[vp] = _video_rotation(vp)
            clips_by_video[vp].append(payload)
        if offset is None:
            break

    rotated_videos = {vp for vp, r in video_rotation.items() if r}
    rotated_clips = sum(len(clips_by_video[vp]) for vp in rotated_videos)
    total_videos = len(video_rotation)
    total_clips = sum(len(v) for v in clips_by_video.values())
    print(f"scanned: {total_videos} videos, {total_clips} clips")
    print(f"  rotated videos: {len(rotated_videos)}  ->  {rotated_clips} clips to re-thumb")
    if not rotated_clips:
        print("nothing to do.")
        return 0

    if args.dry_run:
        for vp in list(rotated_videos)[:8]:
            print(f"  would re-thumb {len(clips_by_video[vp])} clips from {Path(vp).name} (rot={video_rotation[vp]})")
        return 0

    # Pass 2: regenerate thumbnails for affected clips.
    done = errors = 0
    for vp in sorted(rotated_videos):
        for payload in clips_by_video[vp]:
            try:
                clip = Clip(
                    video_path=Path(vp),
                    t_start=float(payload["t_start"]),
                    t_end=float(payload["t_end"]),
                )
                write_thumbnail(clip, Path(payload["thumb_path"]))
                done += 1
            except Exception as e:
                print(f"  ERROR {payload.get('clip_id')[:12]} ({Path(vp).name}): {e}")
                errors += 1
        if done % 50 == 0 and done:
            print(f"  ... {done}/{rotated_clips}")
    print(f"done. regenerated={done} errors={errors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
