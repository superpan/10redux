"""Open-set moment retrieval eval on QVHighlights.

For each query in `highlight_*_release.jsonl`:
  - run text search against the index, restricted to the qvhighlights namespace
  - walk hits in rank order; find the first hit whose source video matches
    the GT `vid` (or `vid` prefix) and whose [t_start, t_end] overlaps a
    `relevant_window` either with IoU >= threshold or at all
  - record that hit's rank; aggregate Recall@K at multiple IoU thresholds.

Why "open-set": our system retrieves across the whole index, not within a
known target video. That's harder than the official QVHighlights protocol
but it's what a real video-search product does, and it's the protocol that
lets CLAP audio embeddings actually pay off (or not).

Our clips are fixed 10s windows. Tight IoU thresholds (0.5, 0.7) penalize
long ground-truth moments (a 60s relevant_window can never exceed IoU 0.17
with a 10s clip). We report a `any_overlap` metric too so the
"found the right region" signal survives the IoU gauntlet.

Run CLAP ablation by toggling the env var:
  TEN_CLAP_BACKEND=clap  uv run python tools/eval_qvhighlights.py --tag clap_on
  TEN_CLAP_BACKEND=none  uv run python tools/eval_qvhighlights.py --tag clap_off

Run:
  make eval-qvh
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

DEFAULT_TEST = Path("data/qvhighlights_pilot.jsonl")
DEFAULT_LIMIT = 500
DEFAULT_NAMESPACE = "/videos/qvh_pilot/"
RESULTS_DIR = Path("data/eval")
IOU_THRESHOLDS = (0.3, 0.5, 0.7)
RECALL_KS = (1, 5, 10, 20)


def load_test(path: Path) -> list[dict]:
    items = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def iou(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    inter = max(0.0, min(a_end, b_end) - max(a_start, b_start))
    if inter <= 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start)
    return inter / union if union > 0 else 0.0


def best_window_iou(t_start: float, t_end: float, windows: list[list[float]]) -> float:
    return max((iou(t_start, t_end, w[0], w[1]) for w in windows), default=0.0)


def any_overlap(t_start: float, t_end: float, windows: list[list[float]]) -> bool:
    return any(max(0.0, min(t_end, w[1]) - max(t_start, w[0])) > 0 for w in windows)


def rank_for(
    hits, target_vid: str, windows: list[list[float]], namespace: str
) -> dict:
    """Find earliest rank meeting each criterion. Returns dict of {criterion: rank or None}."""
    out: dict = {"any_overlap": None, "top_iou": 0.0, "top_vid_match_rank": None}
    for thr in IOU_THRESHOLDS:
        out[f"iou>={thr}"] = None

    rank = 0
    for h in hits:
        path = h.payload.get("video_path", "")
        if namespace not in path:
            continue
        rank += 1
        vid = Path(path).stem
        if vid != target_vid:
            continue
        if out["top_vid_match_rank"] is None:
            out["top_vid_match_rank"] = rank
        t0 = float(h.payload["t_start"])
        t1 = float(h.payload["t_end"])
        if out["any_overlap"] is None and any_overlap(t0, t1, windows):
            out["any_overlap"] = rank
        clip_iou = best_window_iou(t0, t1, windows)
        out["top_iou"] = max(out["top_iou"], clip_iou)
        for thr in IOU_THRESHOLDS:
            if out[f"iou>={thr}"] is None and clip_iou >= thr:
                out[f"iou>={thr}"] = rank
        # Early exit: all criteria satisfied at K<=max(RECALL_KS) and at the
        # tightest threshold means no further hits matter.
        if all(out[f"iou>={thr}"] is not None for thr in IOU_THRESHOLDS) and out["any_overlap"] is not None:
            break
    return out


def recall_at(ranks: list[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / max(1, len(ranks))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--test-jsonl", type=Path, default=DEFAULT_TEST)
    p.add_argument("--namespace", type=str, default=DEFAULT_NAMESPACE)
    p.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    p.add_argument("--max-queries", type=int, default=None)
    p.add_argument("--tag", type=str, default=None,
                   help="Tag appended to output filename, e.g. 'clap_on'")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if not args.test_jsonl.exists():
        print(f"missing {args.test_jsonl}", file=sys.stderr)
        return 1

    from ten.search import Searcher

    items = load_test(args.test_jsonl)
    if args.max_queries:
        items = items[: args.max_queries]
    clap_active = os.environ.get("TEN_CLAP_BACKEND", "none").lower() not in ("", "none", "off", "disabled", "false", "0")
    print(f"running {len(items)} queries  limit={args.limit}  namespace={args.namespace!r}  CLAP={'on' if clap_active else 'off'}")

    searcher = Searcher()
    per_query = []
    ranks_by_crit: dict[str, list] = {
        "any_overlap": [],
        "top_vid_match_rank": [],
        **{f"iou>={t}": [] for t in IOU_THRESHOLDS},
    }
    ious: list[float] = []
    t0 = time.time()
    for i, item in enumerate(items, 1):
        q = item["query"]
        vid = item["vid"]
        windows = item["relevant_windows"]
        hits = searcher.search(text=q, limit=args.limit)
        r = rank_for(hits, vid, windows, args.namespace)
        per_query.append({"qid": item["qid"], "vid": vid, "query": q, **r})
        for k, v in r.items():
            if k == "top_iou":
                ious.append(v)
            else:
                ranks_by_crit[k].append(v)
        if i % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{i}/{len(items)}]  {elapsed:.0f}s  any_R@10={recall_at(ranks_by_crit['any_overlap'], 10):.3f}")

    n = len(items)
    summary = {"n_queries": n, "limit": args.limit, "clap": "on" if clap_active else "off"}
    print()
    print("== QVHighlights open-set moment retrieval ==")
    print(f"  CLAP: {'on' if clap_active else 'off'}")
    print(f"  queries: {n}")
    for crit in ranks_by_crit:
        row = {f"R@{k}": recall_at(ranks_by_crit[crit], k) for k in RECALL_KS}
        summary[crit] = row
        cells = "  ".join(f"{k}={v:.3f}" for k, v in row.items())
        print(f"  {crit:20s}  {cells}")
    summary["mean_top_iou"] = statistics.fmean(ious) if ious else 0.0
    summary["median_top_iou"] = statistics.median(ious) if ious else 0.0
    print(f"  top_iou_mean        {summary['mean_top_iou']:.3f}")
    print(f"  top_iou_median      {summary['median_top_iou']:.3f}")
    print(f"  wall time           {time.time() - t0:.0f}s")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("clap_on" if clap_active else "clap_off")
    out = args.out or RESULTS_DIR / f"qvhighlights_{tag}.json"
    with out.open("w") as f:
        json.dump({"summary": summary, "per_query": per_query}, f, indent=2)
    print(f"  results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
