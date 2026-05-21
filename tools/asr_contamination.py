"""Classify the transcripts currently in `ten_text` as real speech vs Whisper-failure modes.

Run after an `--asr` ingest to estimate how much of the text collection is
hallucinated (music_glyph / repetition_loop / filler / near_empty) vs real.
This headroom motivates whether to enable VAD pre-gating.

Categories:
- empty           : null / blank transcript (ASR not invoked or returned nothing)
- music_glyph     : contains `¶` (Whisper transcribes instrumentals as note glyphs)
- repetition_loop : the top trigram (or bigram for short texts) repeats heavily
- filler          : 1–2 words, all in a fixed filler set ("uh", "hmm", "so", …)
- near_empty      : < 3 chars
- non_alpha       : no alphanumeric character (symbol-only)
- real            : everything else

Run:
  uv run python tools/asr_contamination.py
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from qdrant_client import QdrantClient

from ten.config import CONFIG

FILLERS = {
    "uh", "huh", "hmm", "um", "oh", "mhm", "mm", "hi", "ah", "eh", "mmm",
    "yeah", "ok", "okay", "so",
}

WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
ALPHA_NUM_RE = re.compile(r"[^\W_]", re.UNICODE)


def classify(t: str | None) -> str:
    if t is None:
        return "empty"
    s = t.strip()
    if not s:
        return "empty"
    if "¶" in s:
        return "music_glyph"
    if len(s) < 3:
        return "near_empty"
    if not ALPHA_NUM_RE.search(s):
        return "non_alpha"
    words = WORD_RE.findall(s.lower())
    if not words:
        return "non_alpha"
    if 1 <= len(words) <= 2 and all(w in FILLERS for w in words):
        return "filler"
    if len(words) >= 8:
        ngram_n = 3 if len(words) >= 12 else 2
        ngrams = list(zip(*[words[i:] for i in range(ngram_n)]))
        if ngrams:
            top_count = Counter(ngrams).most_common(1)[0][1]
            ratio = top_count / max(1, len(ngrams))
            if (ngram_n == 3 and top_count >= 4) or (ngram_n == 2 and ratio > 0.5):
                return "repetition_loop"
    return "real"


def main() -> int:
    client = QdrantClient(url=CONFIG.qdrant_url, api_key=CONFIG.qdrant_api_key)
    ns_buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    offset = None
    while True:
        pts, offset = client.scroll(
            collection_name=CONFIG.text_collection,
            limit=1024,
            offset=offset,
            with_payload=True,
        )
        for p in pts:
            ns = p.payload.get("video_path", "").rsplit("/", 2)[-2] or "?"
            cls = classify(p.payload.get("transcript"))
            ns_buckets[ns][cls] += 1
        if offset is None:
            break

    cats = ["empty", "near_empty", "music_glyph", "non_alpha", "filler", "repetition_loop", "real"]
    print(f"{'namespace':14s} " + " ".join(f"{c[:8]:>8s}" for c in cats) + "   total")
    print("-" * 96)
    for ns, b in sorted(ns_buckets.items()):
        total = sum(b.values())
        row = " ".join(f"{b[c]:>8d}" for c in cats)
        print(f"{ns:14s} {row}   {total}")

    overall = defaultdict(int)
    for b in ns_buckets.values():
        for c, n in b.items():
            overall[c] += n
    nonempty = sum(n for c, n in overall.items() if c != "empty")
    garbage = sum(overall[c] for c in cats if c not in ("real", "empty"))
    real = overall["real"]
    print()
    print(f"non-empty: {nonempty}   real: {real} ({100*real/max(1,nonempty):.1f}%)   "
          f"garbage: {garbage} ({100*garbage/max(1,nonempty):.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
