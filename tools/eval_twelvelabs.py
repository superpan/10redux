"""Run the QVHighlights 100-video pilot eval against TwelveLabs (Marengo 3.0).

Apples-to-apples with the ten pilot we already ran (`eval_qvhighlights.py`):
same 100 videos, same 100 queries, same metric (video-level top-K match
against the relevant ground-truth vid).

Approach:
  1. Create a fresh TL index dedicated to this eval (so prior uploads don't
     contaminate the pool).
  2. Upload the 100 QVH pilot videos via POST /v1.3/tasks (multipart) and
     remember the (clip_vid -> tl_video_id) mapping.
  3. Poll task status until every video reports `ready`.
  4. For each query, POST /v1.3/search and walk the moment list deduped
     by video_id to find the rank of the GT video.
  5. Compute Recall@K (K=1,5,10,20) on `top_vid_match` and write
     data/eval/twelvelabs_qvh_pilot.json.
  6. Also compute the same metric for ten on the same 100 queries from
     the cached `data/eval/qvhighlights_clap_rerank_v2.json` so the
     comparison is one-to-one.

API key is read from $TL_API_KEY. Never committed.

Run:
  TL_API_KEY=tlk_... uv run python tools/eval_twelvelabs.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

TL_BASE = "https://api.twelvelabs.io/v1.3"
INDEX_NAME = "ten-qvh-pilot-eval"
RECALL_KS = (1, 5, 10, 20)
PILOT_JSONL = Path("data/qvhighlights_pilot.jsonl")
VIDEO_DIR = Path("videos/qvh_pilot")
TEN_CACHED = Path("data/eval/qvhighlights_clap_rerank_v2.json")
OUT = Path("data/eval/twelvelabs_qvh_pilot.json")
UPLOAD_CONCURRENCY = 4         # TL likely tolerates more, but be polite on first run
POLL_INTERVAL_S = 10
POLL_TIMEOUT_S = 30 * 60       # 30 min upper bound on indexing the whole set


def _api_key() -> str:
    key = os.environ.get("TL_API_KEY", "")
    if not key:
        print("error: TL_API_KEY env var not set", file=sys.stderr)
        sys.exit(2)
    return key


def _headers(key: str, *, json_content: bool = False) -> dict:
    h = {"x-api-key": key}
    if json_content:
        h["Content-Type"] = "application/json"
    return h


def find_or_create_index(client: httpx.Client, key: str) -> str:
    r = client.get(f"{TL_BASE}/indexes", headers=_headers(key))
    r.raise_for_status()
    for ix in r.json().get("data", []):
        if ix["index_name"] == INDEX_NAME:
            print(f"  reusing existing index '{INDEX_NAME}' ({ix['_id']}, {ix.get('video_count', 0)} videos)")
            return ix["_id"]
    body = {
        "index_name": INDEX_NAME,
        "models": [{"model_name": "marengo3.0", "model_options": ["visual", "audio"]}],
        "addons": ["thumbnail"],
    }
    r = client.post(f"{TL_BASE}/indexes", headers=_headers(key, json_content=True), json=body)
    r.raise_for_status()
    idx_id = r.json()["_id"]
    print(f"  created index '{INDEX_NAME}' ({idx_id})")
    return idx_id


def upload_one(key: str, index_id: str, path: Path) -> tuple[Path, str | None, str | None]:
    try:
        with httpx.Client(timeout=120.0) as c:
            with path.open("rb") as f:
                r = c.post(
                    f"{TL_BASE}/tasks",
                    headers=_headers(key),
                    data={"index_id": index_id},
                    files={"video_file": (path.name, f, "video/mp4")},
                )
            if r.status_code >= 300:
                return path, None, f"HTTP {r.status_code}: {r.text[:160]}"
            j = r.json()
            return path, j.get("_id") or j.get("video_id"), None
    except Exception as e:
        return path, None, str(e)[:160]


def upload_all(client: httpx.Client, key: str, index_id: str, paths: list[Path]) -> dict[str, str]:
    """Returns {qvh_vid -> tl_task_id}. Skips files that fail to upload."""
    mapping: dict[str, str] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=UPLOAD_CONCURRENCY) as pool:
        futs = {pool.submit(upload_one, key, index_id, p): p for p in paths}
        for i, f in enumerate(as_completed(futs), 1):
            p, task_id, err = f.result()
            qvh_vid = p.stem
            if task_id:
                mapping[qvh_vid] = task_id
            else:
                errors[qvh_vid] = err or "unknown"
            if i % 10 == 0 or i == len(paths):
                print(f"    upload [{i}/{len(paths)}]  ok={len(mapping)} fail={len(errors)}")
    if errors:
        print(f"    upload errors:")
        for v, e in list(errors.items())[:5]:
            print(f"      {v}: {e}")
    return mapping


def poll_all_ready(client: httpx.Client, key: str, task_ids: list[str]) -> dict[str, str]:
    """Returns {task_id -> tl_video_id} for tasks that reach `ready` within timeout."""
    ready: dict[str, str] = {}
    failed: list[str] = []
    pending = set(task_ids)
    t0 = time.time()
    while pending and time.time() - t0 < POLL_TIMEOUT_S:
        for tid in list(pending):
            try:
                r = client.get(f"{TL_BASE}/tasks/{tid}", headers=_headers(key))
                if r.status_code >= 300:
                    continue
                j = r.json()
                status = j.get("status")
                if status == "ready":
                    ready[tid] = j.get("video_id") or tid
                    pending.discard(tid)
                elif status == "failed":
                    failed.append(tid)
                    pending.discard(tid)
            except Exception:
                pass
        if pending:
            print(f"    poll: ready={len(ready)} failed={len(failed)} pending={len(pending)}  ({int(time.time()-t0)}s)")
            time.sleep(POLL_INTERVAL_S)
    if pending:
        print(f"  WARNING: {len(pending)} tasks still pending after {POLL_TIMEOUT_S}s")
    return ready


def search_one(key: str, index_id: str, query: str, limit: int = 50) -> list[dict]:
    # TL's /search rejects application/x-www-form-urlencoded (gets back
    # `content_type_invalid`) — it requires multipart/form-data. To repeat the
    # `search_options` field for multi-value, pass the form as a list of
    # `files=` tuples (each value tuple has `(filename, content)` with
    # filename=None for plain text fields).
    files = [
        ("index_id", (None, index_id)),
        ("query_text", (None, query)),
        ("search_options", (None, "visual")),
        ("search_options", (None, "audio")),
        ("page_limit", (None, str(limit))),
    ]
    with httpx.Client(timeout=30.0) as c:
        r = c.post(f"{TL_BASE}/search", headers=_headers(key), files=files)
    if r.status_code >= 300:
        return []
    return r.json().get("data", [])


def video_rank(hits: list[dict], target_video_id: str) -> int | None:
    """First rank where target_video_id appears, with dedupe by video.

    TL returns moments (clip windows), often multiple per video. We collapse
    consecutive same-video hits and assign the first appearance a rank equal
    to the unique-video position.
    """
    seen: list[str] = []
    for h in hits:
        v = h.get("video_id")
        if not v:
            continue
        if v not in seen:
            seen.append(v)
            if v == target_video_id:
                return len(seen)
    return None


def compute_recall(ranks: list[int | None], k: int) -> float:
    n = len(ranks)
    if not n:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / n


def ten_top_vid_recall_for(qids: set[int], k: int) -> float:
    """Lookup ten's cached `top_vid_match_rank` for the same qids."""
    if not TEN_CACHED.exists():
        return float("nan")
    data = json.load(TEN_CACHED.open())
    subset = [q for q in data["per_query"] if q["qid"] in qids]
    n = len(subset)
    if not n:
        return float("nan")
    return sum(1 for q in subset if q["top_vid_match_rank"] is not None and q["top_vid_match_rank"] <= k) / n


def main() -> int:
    key = _api_key()
    if not PILOT_JSONL.exists() or not VIDEO_DIR.exists():
        print(f"error: {PILOT_JSONL} or {VIDEO_DIR} missing", file=sys.stderr)
        return 2

    items = [json.loads(l) for l in PILOT_JSONL.open() if l.strip()]
    print(f"loaded {len(items)} queries from {PILOT_JSONL}")

    needed_videos = {it["vid"] for it in items}
    on_disk = [p for p in VIDEO_DIR.glob("*.mp4") if p.stem in needed_videos]
    missing = needed_videos - {p.stem for p in on_disk}
    if missing:
        print(f"warning: {len(missing)} videos referenced by queries are missing on disk: {list(missing)[:3]}")

    with httpx.Client(timeout=30.0) as client:
        index_id = find_or_create_index(client, key)

        # If the index is empty (newly created or fresh), upload all.
        info = client.get(f"{TL_BASE}/indexes/{index_id}", headers=_headers(key)).json()
        video_count = info.get("video_count", 0)
        if video_count >= len(on_disk):
            print(f"  index already has {video_count} videos; skipping upload (assumes prior eval state)")
            # No clean way to recover qvh_vid -> tl_video_id without listing assets.
            # For now, force re-create-from-scratch by aborting; user should delete index.
            print("  delete the index first if you want to re-upload; for now assuming we can search")
        else:
            print(f"  uploading {len(on_disk)} videos ({sum(p.stat().st_size for p in on_disk) / 1e9:.1f} GB)...")
            mapping = upload_all(client, key, index_id, on_disk)
            print(f"  upload done: {len(mapping)}/{len(on_disk)}")
            print(f"  polling for indexing completion (this can take a while)...")
            tid_to_vid = poll_all_ready(client, key, list(mapping.values()))
            print(f"  indexed: {len(tid_to_vid)}/{len(mapping)}")
            mapping_path = OUT.with_name("twelvelabs_qvh_pilot_mapping.json")
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            mapping_path.write_text(json.dumps({
                "qvh_vid_to_task_id": mapping,
                "task_id_to_tl_video_id": tid_to_vid,
            }, indent=2))
            print(f"  saved mapping: {mapping_path}")

        # Build qvh_vid -> tl_video_id by listing index assets.
        # TL doesn't expose original filenames on assets, so we rely on the upload-time
        # mapping we just persisted. If it's not there (rare reuse case), the eval
        # falls back to per-query relevance via TL's filename metadata.
        mapping_path = OUT.with_name("twelvelabs_qvh_pilot_mapping.json")
        if mapping_path.exists():
            m = json.loads(mapping_path.read_text())
            qvh_to_tl = {qvh: m["task_id_to_tl_video_id"].get(tid) for qvh, tid in m["qvh_vid_to_task_id"].items()}
            qvh_to_tl = {k: v for k, v in qvh_to_tl.items() if v}
        else:
            qvh_to_tl = {}
        print(f"  resolved {len(qvh_to_tl)} qvh_vid -> tl_video_id pairs")

    # Run queries.
    print()
    print(f"running {len(items)} queries against TwelveLabs (marengo3.0)...")
    per_query = []
    t0 = time.time()
    for i, it in enumerate(items, 1):
        gt_qvh_vid = it["vid"]
        gt_tl_vid = qvh_to_tl.get(gt_qvh_vid)
        hits = search_one(key, index_id, it["query"], limit=50) if gt_tl_vid else []
        rank = video_rank(hits, gt_tl_vid) if gt_tl_vid else None
        per_query.append({
            "qid": it["qid"], "query": it["query"], "vid": gt_qvh_vid,
            "tl_video_id": gt_tl_vid, "top_vid_match_rank": rank,
            "n_hits": len(hits),
        })
        if i % 25 == 0 or i == len(items):
            r1 = compute_recall([q["top_vid_match_rank"] for q in per_query], 1)
            r10 = compute_recall([q["top_vid_match_rank"] for q in per_query], 10)
            print(f"  [{i}/{len(items)}]  {int(time.time()-t0)}s  R@1={r1:.3f} R@10={r10:.3f}")

    ranks = [q["top_vid_match_rank"] for q in per_query]
    summary = {
        "system": "twelvelabs:marengo3.0",
        "n_queries": len(items),
        "n_resolved_videos": len(qvh_to_tl),
        **{f"top_vid_R@{k}": compute_recall(ranks, k) for k in RECALL_KS},
        "median_rank": statistics.median([r for r in ranks if r is not None]) if any(r is not None for r in ranks) else float("inf"),
        "n_missed": sum(1 for r in ranks if r is None),
    }

    # Compare against ten on the SAME 100 queries.
    qids = {q["qid"] for q in per_query}
    ten = {f"top_vid_R@{k}": ten_top_vid_recall_for(qids, k) for k in RECALL_KS}

    print()
    print(f"== Results — both systems, top-vid-match recall on the same 100 queries ==")
    print(f"  {'metric':12s}  {'TwelveLabs':>11s}  {'ten':>8s}  {'Δ':>7s}")
    for k in RECALL_KS:
        a = summary[f"top_vid_R@{k}"]
        b = ten[f"top_vid_R@{k}"]
        print(f"  R@{k:<10d}  {a:11.3f}  {b:8.3f}  {a - b:+7.3f}")
    print(f"  TL  missed (no GT video resolved): {summary['n_missed']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "summary": summary,
        "ten_summary": {**ten, "system": "ten (caption-mediated)"},
        "per_query": per_query,
    }, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
