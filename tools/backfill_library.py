"""One-time migration: add `library` to every point's payload.

`library` is the parent directory of the video file (`videos/personal/foo.mp4`
→ `personal`). New ingests set this at upsert time; this script backfills
existing points that were indexed before the field existed.

Run:
  uv run python tools/backfill_library.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from ten.config import CONFIG

COLLECTIONS = [CONFIG.visual_collection, CONFIG.text_collection, CONFIG.audio_collection]


def main() -> int:
    client = QdrantClient(url=CONFIG.qdrant_url, api_key=CONFIG.qdrant_api_key)
    existing = {c.name for c in client.get_collections().collections}

    overall = Counter()
    for col in COLLECTIONS:
        if col not in existing:
            print(f"  {col}: skip (collection not present)")
            continue
        seen = updated = skipped = 0
        # Create the payload index up-front so it's ready for the new field.
        try:
            client.create_payload_index(
                collection_name=col,
                field_name="library",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
        except Exception:
            pass

        # Scroll the whole collection. Update one point at a time to keep memory
        # bounded; for ≤100K points this finishes in a few minutes.
        offset = None
        while True:
            pts, offset = client.scroll(
                collection_name=col,
                limit=512,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in pts:
                seen += 1
                payload = p.payload or {}
                video_path = payload.get("video_path")
                if not video_path:
                    skipped += 1
                    continue
                # Derive library from the parent directory of the video file.
                library = Path(video_path).parent.name
                if payload.get("library") == library:
                    skipped += 1
                    continue
                client.set_payload(
                    collection_name=col,
                    payload={"library": library},
                    points=[p.id],
                )
                updated += 1
                overall[library] += 1
            if offset is None:
                break

        print(f"  {col}: seen={seen} updated={updated} skipped={skipped}")

    print()
    print("library distribution after backfill (visual collection writes only counted once per id):")
    # Approximate: print the cumulative counter divided by 3 (writes happen per collection)
    # Cleaner: re-scroll visual collection only.
    seen_per_lib: Counter[str] = Counter()
    if CONFIG.visual_collection in existing:
        offset = None
        while True:
            pts, offset = client.scroll(
                collection_name=CONFIG.visual_collection,
                limit=1024,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in pts:
                lib = (p.payload or {}).get("library") or "<none>"
                seen_per_lib[lib] += 1
            if offset is None:
                break
    total = sum(seen_per_lib.values())
    for lib, n in seen_per_lib.most_common():
        print(f"  {lib:25s} {n:5d}  ({100*n/total:.1f}%)" if total else f"  {lib:25s} {n:5d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
