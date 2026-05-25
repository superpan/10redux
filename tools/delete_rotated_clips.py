"""Delete Qdrant points whose source video has a non-zero display rotation.

Used after the `apply display-matrix rotation when decoding frames` change:
clips ingested before that fix have captions and embeddings produced from
sideways frames. Deleting their points lets `make index ...` re-process
just those clips (the resume-aware `has_clip` check leaves correctly
oriented clips alone).

Run:
  uv run python tools/delete_rotated_clips.py --library personal --dry-run
  uv run python tools/delete_rotated_clips.py --library personal
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ten.config import CONFIG
from ten.video import _video_rotation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", default=None,
                        help="Restrict scan to one library (e.g. 'personal').")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report counts, do not delete.")
    args = parser.parse_args()

    client = QdrantClient(url=CONFIG.qdrant_url, api_key=CONFIG.qdrant_api_key)
    scroll_filter = None
    if args.library:
        scroll_filter = qm.Filter(
            must=[qm.FieldCondition(key="library", match=qm.MatchValue(value=args.library))]
        )

    # Walk ten_visual to discover (point_id, source_video) per clip.
    point_ids_by_video: dict[str, list[str]] = defaultdict(list)
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
            vp = (p.payload or {}).get("video_path")
            if not vp:
                continue
            point_ids_by_video[vp].append(p.id)
        if offset is None:
            break

    # Classify each video by rotation. Cached, so cheap on repeat runs.
    rotated_videos = [vp for vp in point_ids_by_video if _video_rotation(vp)]
    to_delete: list[str] = []
    for vp in rotated_videos:
        to_delete.extend(point_ids_by_video[vp])

    total_videos = len(point_ids_by_video)
    total_clips = sum(len(v) for v in point_ids_by_video.values())
    print(f"scanned: {total_videos} videos / {total_clips} clips")
    print(f"  rotated videos: {len(rotated_videos)}  ->  {len(to_delete)} clips to delete")

    if not to_delete or args.dry_run:
        return 0

    collections = [CONFIG.visual_collection, CONFIG.text_collection, CONFIG.audio_collection]
    existing = {c.name for c in client.get_collections().collections}
    for col in collections:
        if col not in existing:
            continue
        # Qdrant deletes are no-ops for unknown ids, so passing all ids to all
        # collections is safe even if audio doesn't have the point.
        client.delete(collection_name=col, points_selector=qm.PointIdsList(points=to_delete))
        print(f"  deleted from {col}")
    print(f"done. {len(to_delete)} point ids removed; next ingest will re-process them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
