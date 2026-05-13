"""Download MSR-VTT (videos + 1K-A test split) for retrieval evaluation.

Source: https://huggingface.co/datasets/friedrichor/MSR-VTT
  - MSRVTT_Videos.zip   (~2.2 GB, 10K mp4s)
  - msrvtt_test_1k.json (~340 KB, the standard 1K-A test split: 1 caption/video)

Output:
  videos/msrvtt/<video_id>.mp4    (1K test videos only — matches 1K-A protocol)
  data/msrvtt_test_1k.json        (test split copied here)

Indexing only the 1K test videos keeps retrieval comparable to published
1K-A numbers; indexing all 10K would change the candidate pool.

Run:
  make fetch-msrvtt
"""
from __future__ import annotations

import json
import shutil
import sys
import zipfile
from pathlib import Path

REPO_ID = "friedrichor/MSR-VTT"
VIDEO_ZIP = "MSRVTT_Videos.zip"
TEST_JSON = "msrvtt_test_1k.json"

VIDEO_DIR = Path("videos/msrvtt")
DATA_DIR = Path("data")


def main() -> int:
    from huggingface_hub import hf_hub_download

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)

    test_json_target = DATA_DIR / TEST_JSON
    if test_json_target.exists():
        print(f"have {test_json_target}, skipping")
    else:
        print(f"downloading {TEST_JSON}")
        path = hf_hub_download(repo_id=REPO_ID, filename=TEST_JSON, repo_type="dataset")
        shutil.copy(path, test_json_target)
        print(f"  -> {test_json_target} ({test_json_target.stat().st_size:,} bytes)")

    # Test-split filter: keep only the 1K videos referenced by the test JSON.
    with test_json_target.open() as f:
        test_items = json.load(f)
    wanted = {item["video"] for item in test_items}  # e.g. {"video7020.mp4", ...}
    print(f"test split has {len(wanted)} unique videos")

    sample = next(VIDEO_DIR.glob("*.mp4"), None)
    if sample is not None:
        n = sum(1 for _ in VIDEO_DIR.glob("*.mp4"))
        if n >= len(wanted):
            print(f"have {n} videos in {VIDEO_DIR}, skipping zip download")
            return 0
        print(f"only {n}/{len(wanted)} videos present; re-extracting")

    print(f"downloading {VIDEO_ZIP} (~2.2 GB)")
    zip_path = hf_hub_download(repo_id=REPO_ID, filename=VIDEO_ZIP, repo_type="dataset")
    print(f"  cached at {zip_path}")
    print(f"unpacking 1K test videos into {VIDEO_DIR}")
    extracted = 0
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.namelist():
            if not m.lower().endswith(".mp4"):
                continue
            name = Path(m).name
            if name not in wanted:
                continue
            target = VIDEO_DIR / name
            if target.exists():
                continue
            with zf.open(m) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted += 1
    n = sum(1 for _ in VIDEO_DIR.glob("*.mp4"))
    print(f"extracted {extracted} new files; {n}/{len(wanted)} test videos now on disk.")
    if n < len(wanted):
        missing = sorted(wanted - {p.name for p in VIDEO_DIR.glob("*.mp4")})
        print(f"WARNING: {len(missing)} videos not found in zip (first 5: {missing[:5]})",
              file=sys.stderr)
    print()
    print("Next:")
    print("  TEN_CLIP_SECONDS=60 TEN_CLIP_OVERLAP=0 \\")
    print("    make index FOLDER=./videos/msrvtt        # one clip per video for retrieval eval")
    print("  make eval                                  # compute Recall@K + MRR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
