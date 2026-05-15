"""Micro-benchmarks for each stage of the pipeline.

Times each model call / index lookup in isolation with warm-up + N=50 samples,
reports P50/P95/P99 + mean. Uses the existing smoke videos for inputs so you
can re-run without setup beyond `make fetch-smoke` and `make qdrant-up`.

Run:
  uv run python tools/bench.py                      # all stages
  uv run python tools/bench.py --stage ingest       # ingest stages only
  uv run python tools/bench.py --stage query        # query stages only
  uv run python tools/bench.py --n 100              # more samples
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

console = Console()


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    if not s:
        return float("nan")
    idx = int(p / 100 * (len(s) - 1) + 0.5)
    return s[min(max(idx, 0), len(s) - 1)]


def time_stage(label: str, fn: Callable[[], None], n: int = 50, warmup: int = 3) -> dict:
    """Run fn warmup+n times, capture timings in ms."""
    for _ in range(warmup):
        try:
            fn()
        except Exception as e:
            return {"label": label, "error": str(e)}
    times_ms: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as e:
            return {"label": label, "error": str(e)}
        times_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "label": label,
        "n": n,
        "mean": statistics.fmean(times_ms),
        "p50": _percentile(times_ms, 50),
        "p95": _percentile(times_ms, 95),
        "p99": _percentile(times_ms, 99),
        "min": min(times_ms),
        "max": max(times_ms),
    }


def render(title: str, results: list[dict]) -> None:
    t = Table(title=title, show_lines=False)
    t.add_column("stage")
    t.add_column("n", justify="right")
    t.add_column("p50 ms", justify="right")
    t.add_column("p95 ms", justify="right")
    t.add_column("p99 ms", justify="right")
    t.add_column("mean ms", justify="right")
    for r in results:
        if "error" in r:
            t.add_row(r["label"], "[red]err[/red]", r["error"][:40], "", "", "")
            continue
        t.add_row(
            r["label"],
            str(r["n"]),
            f"{r['p50']:7.1f}",
            f"{r['p95']:7.1f}",
            f"{r['p99']:7.1f}",
            f"{r['mean']:7.1f}",
        )
    console.print(t)


def bench_ingest(n: int) -> list[dict]:
    """Time per-stage costs on a representative clip from the smoke set."""
    import os

    from ten.audio import extract_audio
    from ten.caption import make_captioner
    from ten.embed_audio import CLAP_SAMPLE_RATE, CLAPEmbedder
    from ten.embed_text import TextEmbedder
    from ten.embed_video import VideoEmbedder
    from ten.config import CONFIG
    from ten.video import Clip, sample_clip_frames

    sample = Path("videos/smoke/Sintel_1080p.mkv")
    if not sample.exists():
        console.print(f"[red]bench needs {sample}; run `make fetch-smoke`[/red]")
        return []

    clip = Clip(video_path=sample, t_start=120.0, t_end=130.0)
    frames = sample_clip_frames(clip, CONFIG.frames_per_clip, CONFIG.frame_resize)

    results: list[dict] = []

    results.append(time_stage(
        "video frame sample (10 s, 8 frames)",
        lambda: sample_clip_frames(clip, CONFIG.frames_per_clip, CONFIG.frame_resize),
        n=n,
    ))

    asr_wav = Path("/tmp/ten_bench_asr.wav")
    results.append(time_stage(
        "ffmpeg audio extract (16 kHz)",
        lambda: extract_audio(clip, asr_wav, sample_rate=16000),
        n=n,
    ))

    clap_wav = Path("/tmp/ten_bench_clap.wav")
    results.append(time_stage(
        "ffmpeg audio extract (48 kHz)",
        lambda: extract_audio(clip, clap_wav, sample_rate=CLAP_SAMPLE_RATE),
        n=n,
    ))

    ve = VideoEmbedder()
    ve.embed([frames])  # warm
    results.append(time_stage(
        "V-JEPA 2 visual embed (1 clip)",
        lambda: ve.embed([frames]),
        n=n,
    ))

    te = TextEmbedder()
    te.embed_passages(["warm"])  # warm
    cap_text = "A bearded man in a dimly lit, rustic workshop holds a large weapon."
    results.append(time_stage(
        "Qwen3-Embedding text passage (1 string)",
        lambda: te.embed_passages([cap_text]),
        n=n,
    ))

    if os.environ.get("TEN_VLM_BACKEND", "").lower() == "vllm":
        cap = make_captioner()
        cap.caption(frames)  # warm
        results.append(time_stage(
            "Qwen3-VL caption via vLLM (1 clip)",
            lambda: cap.caption(frames),
            n=max(10, n // 5),
        ))
    else:
        results.append({"label": "Qwen3-VL caption (skipped — set TEN_VLM_BACKEND=vllm)", "error": "skipped"})

    if os.environ.get("TEN_ASR_BACKEND", "").lower() == "whisper":
        from ten.asr import TransformersWhisperTranscriber
        tr = TransformersWhisperTranscriber()
        tr.transcribe(asr_wav)  # warm
        results.append(time_stage(
            "Whisper-large-v3 transcribe (10 s clip)",
            lambda: tr.transcribe(asr_wav),
            n=max(10, n // 5),
        ))
    else:
        results.append({"label": "Whisper ASR (skipped — set TEN_ASR_BACKEND=whisper)", "error": "skipped"})

    ce = CLAPEmbedder()
    ce.embed([clap_wav])  # warm
    results.append(time_stage(
        "CLAP audio embed (10 s clip)",
        lambda: ce.embed([clap_wav]),
        n=n,
    ))

    return results


def bench_query(n: int) -> list[dict]:
    """Time per-stage costs of a text search."""
    import os

    from ten.embed_text import TextEmbedder
    from ten.embed_audio import CLAPEmbedder
    from ten.search import Searcher
    from ten.store import Store

    queries = [
        "a person riding a bicycle in the rain",
        "a dragon breathing fire",
        "two children playing in a park",
        "a woman cooking in a kitchen",
        "soldiers marching in formation",
    ]
    q = queries[0]

    results: list[dict] = []

    te = TextEmbedder()
    te.embed_query("warm")
    results.append(time_stage(
        "Qwen3-Embedding query (1 query)",
        lambda: te.embed_query(q),
        n=n,
    ))

    ce = CLAPEmbedder()
    ce.embed_text(["warm"])
    results.append(time_stage(
        "CLAP text query embed (1 query)",
        lambda: ce.embed_text([q]),
        n=n,
    ))

    store = Store()
    qv = te.embed_query(q)
    store.search_text(qv, limit=100)  # warm
    results.append(time_stage(
        "Qdrant text-collection search (limit=100)",
        lambda: store.search_text(qv, limit=100),
        n=n,
    ))

    s = Searcher()
    s.search(text=q, limit=10)  # warm
    results.append(time_stage(
        "search() bi-encoder only (limit=10)",
        lambda: s.search(text=q, limit=10),
        n=n,
    ))

    if os.environ.get("TEN_RERANKER_BACKEND", "").lower() == "crossencoder":
        _ = s.reranker  # property loads the reranker
        s.search(text=q, limit=10)  # warm with rerank
        results.append(time_stage(
            "search() + rerank top_k=100 (limit=10)",
            lambda: s.search(text=q, limit=10),
            n=max(10, n // 3),
        ))
    else:
        results.append({"label": "search() + rerank (skipped — set TEN_RERANKER_BACKEND=crossencoder)", "error": "skipped"})

    return results


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("ingest", "query", "all"), default="all")
    p.add_argument("--n", type=int, default=50, help="samples per stage")
    args = p.parse_args()

    if args.stage in ("ingest", "all"):
        console.print("\n[bold]Ingest stages[/bold] (per-clip work)")
        render("Ingest stages", bench_ingest(args.n))

    if args.stage in ("query", "all"):
        console.print("\n[bold]Query stages[/bold] (per-query work)")
        render("Query stages", bench_query(args.n))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
