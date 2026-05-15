"""Download QVHighlights val-split videos for long-form moment-retrieval eval.

QVHighlights only ships annotations + pre-computed features — raw videos must be
re-downloaded from YouTube. Each `vid` in the annotation file encodes a
150-second window from a longer YouTube video as `<youtube_id>_<start>_<end>`.

Source:
  https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_val_release.jsonl

Output:
  data/qvhighlights_val.jsonl                — annotations (mirrored locally)
  videos/qvhighlights/<vid>.mp4              — one 150s clip per unique vid
  videos/qvhighlights/_failures.jsonl        — dead/blocked YouTube IDs (resumable)

Expect ~10-20% loss to dead/region-locked/copyright-struck IDs. Runtime is
hours, not minutes — kick off in the background.

Run:
  make fetch-qvhighlights
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlopen

ANNOT_URL = "https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_val_release.jsonl"
ANNOT_LOCAL = Path("data/qvhighlights_val.jsonl")
VIDEO_DIR = Path("videos/qvhighlights")
FAIL_LOG = VIDEO_DIR / "_failures.jsonl"

MAX_WORKERS = 3  # YouTube throttles aggressively past this


def fetch_annotations() -> list[dict]:
    ANNOT_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if not ANNOT_LOCAL.exists():
        print(f"downloading annotations -> {ANNOT_LOCAL}")
        with urlopen(ANNOT_URL) as resp, ANNOT_LOCAL.open("wb") as f:
            shutil.copyfileobj(resp, f)
    items = []
    with ANNOT_LOCAL.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"loaded {len(items)} val queries from {ANNOT_LOCAL}")
    return items


def parse_vid(vid: str) -> tuple[str, float, float]:
    """`abc123_60.0_210.0` -> ("abc123", 60.0, 210.0). YouTube IDs may contain
    underscores, so split from the right twice."""
    rest, end_s = vid.rsplit("_", 1)
    yt_id, start_s = rest.rsplit("_", 1)
    return yt_id, float(start_s), float(end_s)


def already_failed(vid: str) -> bool:
    if not FAIL_LOG.exists():
        return False
    with FAIL_LOG.open() as f:
        for line in f:
            try:
                if json.loads(line).get("vid") == vid:
                    return True
            except json.JSONDecodeError:
                continue
    return False


def log_failure(vid: str, reason: str) -> None:
    FAIL_LOG.parent.mkdir(parents=True, exist_ok=True)
    with FAIL_LOG.open("a") as f:
        f.write(json.dumps({"vid": vid, "reason": reason, "ts": time.time()}) + "\n")


def download_one(vid: str) -> tuple[str, str]:
    """Returns (vid, status) where status is one of 'ok', 'skip', or an error code."""
    target = VIDEO_DIR / f"{vid}.mp4"
    if target.exists() and target.stat().st_size > 0:
        return vid, "skip"

    yt_id, start, end = parse_vid(vid)
    url = f"https://www.youtube.com/watch?v={yt_id}"

    # --download-sections lets us fetch just the 150s window we need.
    # --force-keyframes-at-cuts gives accurate trims at slight reencode cost.
    cmd = [
        "uv", "run", "--with", "yt-dlp", "yt-dlp",
        "--quiet", "--no-warnings",
        "--no-playlist",
        "--format", "mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "--merge-output-format", "mp4",
        "--download-sections", f"*{start}-{end}",
        "--force-keyframes-at-cuts",
        "-o", str(target),
        "--socket-timeout", "30",
        "--retries", "2",
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        log_failure(vid, "timeout")
        return vid, "timeout"
    if result.returncode == 0 and target.exists() and target.stat().st_size > 0:
        return vid, "ok"
    # Strip the verbose ERROR: prefix yt-dlp uses
    err = (result.stderr or result.stdout or "").strip().splitlines()
    msg = err[-1][:200] if err else f"rc={result.returncode}"
    log_failure(vid, msg)
    # Clean up any partial file
    if target.exists():
        target.unlink()
    return vid, f"fail: {msg}"


def main() -> int:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    items = fetch_annotations()
    vids = sorted({item["vid"] for item in items})
    print(f"unique videos to fetch: {len(vids)}")

    existing = sum(1 for v in vids if (VIDEO_DIR / f"{v}.mp4").exists())
    print(f"  already on disk: {existing}")

    failed_before = sum(1 for v in vids if already_failed(v))
    print(f"  previously failed (will retry once): {failed_before}")

    # Filter to work queue, retry previously-failed once.
    work = [v for v in vids if not (VIDEO_DIR / f"{v}.mp4").exists()]
    print(f"  to download this run: {len(work)}")
    print()

    if not work:
        print("nothing to do.")
        return 0

    ok = fail = 0
    start_ts = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download_one, v): v for v in work}
        for i, fut in enumerate(as_completed(futures), 1):
            vid, status = fut.result()
            if status == "ok":
                ok += 1
            elif status == "skip":
                pass
            else:
                fail += 1
            if i % 25 == 0 or i == len(work):
                elapsed = time.time() - start_ts
                rate = i / elapsed if elapsed else 0
                eta = (len(work) - i) / rate if rate else 0
                print(
                    f"  [{i:5d}/{len(work)}] ok={ok} fail={fail} "
                    f"rate={rate:.2f}/s eta={eta/60:.1f}min",
                    flush=True,
                )

    on_disk = sum(1 for v in vids if (VIDEO_DIR / f"{v}.mp4").exists())
    print()
    print(f"done. {on_disk}/{len(vids)} videos on disk ({on_disk/len(vids)*100:.1f}%)")
    print(f"failures logged at {FAIL_LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
