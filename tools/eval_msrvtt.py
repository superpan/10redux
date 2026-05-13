"""Text-to-video retrieval eval against the MSR-VTT 1K-A test split.

Protocol: for each (caption, ground_truth_video_id) pair, run text search,
restrict results to MSR-VTT videos (filter by filename prefix), find the rank
of the ground-truth video, and report the standard metrics:
  - Recall@1, Recall@5, Recall@10
  - Median rank (MdR), Mean rank (MnR)

Assumes ingest used `TEN_CLIP_SECONDS` large enough that each MSR-VTT video
becomes a single clip (otherwise we aggregate by best clip per video).

Run:
  make eval
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

DEFAULT_TEST = Path("data/msrvtt_test_1k.json")
DEFAULT_LIMIT = 2000  # over-fetch so the namespace filter still has 1K msrvtt videos available
DEFAULT_NAMESPACE = "/videos/msrvtt/"
RESULTS_DIR = Path("data/eval")


def load_test(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"unexpected schema in {path}; expected JSON list")
    return data


def best_rank_for_video(
    hits: list, target_video_id: str, namespace_path: str | None
) -> int | None:
    """Walk hit list; first hit whose video file matches target_video_id wins.

    If `namespace_path` is set (e.g. "/videos/msrvtt/"), only hits whose
    `video_path` contains that substring are counted toward the rank — this
    matches the MSR-VTT 1K-A candidate-pool protocol and excludes any
    unrelated clips (e.g. an earlier smoke ingest) that happen to share the
    Qdrant index.
    """
    target_filenames = {f"{target_video_id}.mp4", f"{target_video_id}.MP4"}
    seen_videos: set[str] = set()
    for h in hits:
        path = h.payload.get("video_path", "")
        if namespace_path and namespace_path not in path:
            continue
        name = h.payload.get("video_name", "")
        if name in seen_videos:
            continue
        seen_videos.add(name)
        if name in target_filenames:
            return len(seen_videos)
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--test-json", type=Path, default=DEFAULT_TEST)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument(
        "--namespace",
        type=str,
        default=DEFAULT_NAMESPACE,
        help="Path substring required of any candidate (default isolates the msrvtt set).",
    )
    p.add_argument("--max-queries", type=int, default=None, help="Cap for quick smoke runs.")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write per-query results JSON here (default: data/eval/msrvtt_<ts>.json)",
    )
    args = p.parse_args()

    if not args.test_json.exists():
        print(f"missing {args.test_json}; run `make fetch-msrvtt` first", file=sys.stderr)
        return 1

    from ten.search import Searcher  # noqa: E402  (defer heavy imports)

    items = load_test(args.test_json)
    if args.max_queries:
        items = items[: args.max_queries]
    print(f"running {len(items)} queries against the index "
          f"(limit={args.limit}, namespace={args.namespace!r})")

    searcher = Searcher()
    ranks: list[int] = []
    misses: list[dict] = []
    per_query: list[dict] = []
    t0 = time.time()
    for i, item in enumerate(items, 1):
        caption = item["caption"]
        target = item["video_id"]
        hits = searcher.search(text=caption, limit=args.limit)
        rank = best_rank_for_video(hits, target, namespace_path=args.namespace)
        per_query.append(
            {
                "video_id": target,
                "caption": caption,
                "rank": rank,
                "top_video": hits[0].payload.get("video_name") if hits else None,
            }
        )
        if rank is None:
            misses.append({"video_id": target, "caption": caption})
        else:
            ranks.append(rank)
        if i % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{i}/{len(items)}]  {elapsed:.0f}s  hit-rate={len(ranks)/i:.3f}")

    n = len(items)
    found = len(ranks)
    r1 = sum(1 for r in ranks if r <= 1) / n
    r5 = sum(1 for r in ranks if r <= 5) / n
    r10 = sum(1 for r in ranks if r <= 10) / n
    mdr = statistics.median(ranks) if ranks else float("inf")
    mnr = statistics.fmean(ranks) if ranks else float("inf")

    print()
    print(f"== MSR-VTT 1K-A text -> video retrieval ==")
    print(f"  queries     : {n}")
    print(f"  found       : {found}  ({found/n:.3f})")
    print(f"  Recall@1    : {r1:.3f}")
    print(f"  Recall@5    : {r5:.3f}")
    print(f"  Recall@10   : {r10:.3f}")
    print(f"  Median rank : {mdr}")
    print(f"  Mean rank   : {mnr:.1f}")
    print(f"  wall time   : {time.time() - t0:.0f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = args.out or RESULTS_DIR / "msrvtt_latest.json"
    with out.open("w") as f:
        json.dump(
            {
                "n_queries": n,
                "n_found": found,
                "recall@1": r1,
                "recall@5": r5,
                "recall@10": r10,
                "median_rank": mdr,
                "mean_rank": mnr,
                "limit": args.limit,
                "config": {
                    "vlm_backend": os.environ.get("TEN_VLM_BACKEND", "transformers"),
                    "vjepa_model": os.environ.get(
                        "TEN_VJEPA_MODEL", "facebook/vjepa2-vitl-fpc16-256-ssv2"
                    ),
                    "text_embed_model": os.environ.get(
                        "TEN_TEXT_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B"
                    ),
                    "vlm_model": os.environ.get("TEN_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct"),
                },
                "per_query": per_query,
                "misses": misses[:50],
            },
            f,
            indent=2,
        )
    print(f"  results     : {out}")

    plot_path = out.with_suffix(".png")
    try:
        _plot(ranks, n, plot_path)
        print(f"  plot        : {plot_path}")
    except ImportError:
        print("  plot        : (skipped — install dev extras: `uv sync --extra dev`)")
    return 0


def _plot(ranks: list[int], n: int, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = list(range(1, 51))
    recall_at_k = [sum(1 for r in ranks if r <= k) / n for k in ks]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(ks, recall_at_k, marker="o", markersize=3)
    for k_marker in (1, 5, 10):
        v = recall_at_k[k_marker - 1]
        ax.axvline(k_marker, color="0.85", linestyle=":", linewidth=0.8)
        ax.annotate(f"R@{k_marker}={v:.3f}", (k_marker, v),
                    textcoords="offset points", xytext=(6, -10), fontsize=8)
    ax.set_xlabel("K")
    ax.set_ylabel("Recall@K")
    ax.set_title("MSR-VTT 1K-A: Recall@K curve")
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    bins = [1, 2, 3, 6, 11, 21, 51, 101, 201, 501, 1001]
    ax.hist(ranks, bins=bins, edgecolor="black")
    ax.set_xscale("log")
    ax.set_xlabel("Rank of ground-truth video (log scale)")
    ax.set_ylabel("Count")
    ax.set_title(f"Rank distribution (n={len(ranks)} found / {n} queries)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
