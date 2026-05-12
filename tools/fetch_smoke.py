"""Download a small, diverse, public-domain video set for smoke-testing the pipeline.

Sources mixed across Blender's open-movie host (download.blender.org) and the
Internet Archive — both are CC-BY / public domain and don't require auth.
URLs were probed live; Internet Archive items can throttle to 503 under burst,
so we retry with backoff and a UA header.

Run:
    uv run python tools/fetch_smoke.py
or:
    make fetch-smoke
"""
from __future__ import annotations

import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

DEST = Path("videos/smoke")
UA = "Mozilla/5.0 (compatible; ten-smoke/0.1)"


@dataclass(frozen=True)
class Video:
    url: str
    name: str
    approx_mb: int
    description: str


VIDEOS: list[Video] = [
    Video(
        url="https://download.blender.org/durian/trailer/sintel_trailer-720p.mp4",
        name="Sintel_trailer.mp4",
        approx_mb=8,
        description="1 min, fantasy trailer (edge case: very short clip)",
    ),
    Video(
        url="https://download.blender.org/peach/bigbuckbunny_movies/big_buck_bunny_480p_h264.mov",
        name="BigBuckBunny_480p.mov",
        approx_mb=249,
        description="10 min, animated, animals/forest (Blender Foundation, CC-BY)",
    ),
    Video(
        url="https://archive.org/download/CC_1916_07_10_TheVagabond/CC_1916_07_10_TheVagabond_512kb.mp4",
        name="ChaplinVagabond_1916.mp4",
        approx_mb=150,
        description="26 min, B&W silent live-action (Charlie Chaplin, public domain)",
    ),
    Video(
        url="https://download.blender.org/demo/movies/Sintel.2010.1080p.mkv",
        name="Sintel_1080p.mkv",
        approx_mb=1172,
        description="15 min, fantasy with dragon (mkv container test, ~1.2 GB)",
    ),
]


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def fetch(v: Video, dest: Path, retries: int = 4) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  already have {dest.name} ({_human(dest.stat().st_size)}), skipping")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            print(f"  [attempt {attempt}/{retries}] downloading {dest.name}")
            req = urllib.request.Request(v.url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                total = int(r.headers.get("Content-Length") or 0)
                last_pct = -1
                with tmp.open("wb") as f:
                    read = 0
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        read += len(chunk)
                        if total:
                            pct = int(read * 100 / total)
                            if pct != last_pct and pct % 10 == 0:
                                print(f"    {pct}%  ({_human(read)} / {_human(total)})", flush=True)
                                last_pct = pct
            tmp.rename(dest)
            print(f"  done: {_human(dest.stat().st_size)}")
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
            wait = 2**attempt
            print(f"  attempt {attempt} failed: {e}; retrying in {wait}s", file=sys.stderr)
            tmp.unlink(missing_ok=True)
            time.sleep(wait)
    raise RuntimeError(f"giving up on {v.url}")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {len(VIDEOS)} smoke-test videos to {DEST.resolve()}")
    print(f"Total approx: ~{sum(v.approx_mb for v in VIDEOS)} MB\n")
    failed: list[str] = []
    for v in VIDEOS:
        print(f"[{v.name}] — {v.description}")
        try:
            fetch(v, DEST / v.name)
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)
            failed.append(v.name)
        print()
    if failed:
        print(f"WARNING: failed to fetch {len(failed)} video(s): {failed}", file=sys.stderr)
        # Don't fail hard — partial set is still useful.
    print("Now run:  make qdrant-up && make index FOLDER=./videos/smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
