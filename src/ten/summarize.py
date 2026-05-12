"""Multi-span, multi-dimension video summarization.

Decoupled from the ingest-side `Captioner`. Takes:
  - one or more `Span(video_path, t_start, t_end)` — any duration, any number
  - a `dimension` selecting the prompt (narrative / visual / motion / aesthetic)

Internally allocates a frame budget across spans proportionally to duration,
re-decodes the frames, and calls the underlying VLM once with the dimension
prompt. Results are cached in-process by (sorted spans, dimension).

Adding a new dimension is a one-line addition to DIMENSION_PROMPTS.
Adding a new modality (e.g. dialogue from ASR) means a new dimension whose
implementation can fall back to a different code path.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .caption import CaptionerProtocol, make_captioner
from .config import CONFIG
from .video import Clip, sample_clip_frames


Dimension = Literal["narrative", "visual", "motion", "aesthetic"]


DIMENSION_PROMPTS: dict[str, str] = {
    "narrative": (
        "Summarize what happens across this footage in 3-5 sentences. "
        "Cover: who/what appears, the setting, what changes over time, and any "
        "notable events. Be factual and specific. Avoid speculation."
    ),
    "visual": (
        "Describe the visual content of this footage in 3-5 sentences. "
        "Cover: notable objects, people, environments, colors, and composition. "
        "Do not narrate plot — only what is visible."
    ),
    "motion": (
        "Describe what moves in this footage and how. "
        "Cover: subject motion (gestures, locomotion, fights, dances), camera "
        "moves (pan, tilt, dolly, zoom), and any cuts. 3-5 sentences."
    ),
    "aesthetic": (
        "Describe the visual style and mood of this footage in 3-5 sentences. "
        "Cover: lighting, color palette, framing, era / genre cues, and overall "
        "feel. Do not narrate plot."
    ),
}

DIMENSIONS: tuple[str, ...] = tuple(DIMENSION_PROMPTS)


@dataclass(frozen=True)
class Span:
    video_path: Path
    t_start: float
    t_end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.t_end - self.t_start)

    def cache_key(self) -> tuple:
        return (str(self.video_path.resolve()), round(self.t_start, 3), round(self.t_end, 3))


def _allocate_frames(spans: list[Span], max_frames: int) -> list[int]:
    """Distribute `max_frames` across spans proportionally to duration.

    Floors at 2 frames per span when budget allows, so very short spans still
    contribute a temporal sample.
    """
    if not spans:
        return []
    n = len(spans)
    if max_frames < n:
        # Fewer frames than spans — give one each (clipped down later if needed).
        return [1] * n

    floor = 2 if max_frames >= 2 * n else 1
    base = [floor] * n
    remaining = max_frames - sum(base)
    total_dur = sum(s.duration for s in spans) or 1.0
    if remaining > 0:
        for i, s in enumerate(spans):
            base[i] += round(remaining * (s.duration / total_dur))
    # Trim if rounding overshot.
    while sum(base) > max_frames:
        i = base.index(max(base))
        base[i] -= 1
    return base


def _gather_frames(spans: list[Span], counts: list[int], resize: int) -> np.ndarray:
    out: list[np.ndarray] = []
    for span, n in zip(spans, counts):
        if n <= 0:
            continue
        clip = Clip(video_path=span.video_path, t_start=span.t_start, t_end=span.t_end)
        out.append(sample_clip_frames(clip, n, resize))
    if not out:
        raise ValueError("No frames sampled — empty spans?")
    return np.concatenate(out, axis=0)


class Summarizer:
    def __init__(self, captioner: CaptionerProtocol | None = None) -> None:
        self._cap = captioner or make_captioner()
        self._cache: dict[tuple, str] = {}
        self._lock = threading.Lock()

    def summarize(
        self,
        spans: list[Span] | tuple[Span, ...],
        dimension: Dimension = "narrative",
        max_frames: int = 32,
        max_tokens: int = 256,
    ) -> str:
        if not spans:
            raise ValueError("spans must be non-empty")
        if dimension not in DIMENSION_PROMPTS:
            raise ValueError(
                f"unknown dimension {dimension!r}; expected one of {DIMENSIONS}"
            )

        # Sort + tuple-ify so cache key is stable across call orderings.
        spans_sorted = sorted(spans, key=lambda s: s.cache_key())
        key = (tuple(s.cache_key() for s in spans_sorted), dimension, max_frames, max_tokens)

        cached = self._cache.get(key)
        if cached is not None:
            return cached

        counts = _allocate_frames(spans_sorted, max_frames)
        frames = _gather_frames(spans_sorted, counts, CONFIG.frame_resize)
        prompt = DIMENSION_PROMPTS[dimension]
        text = self._cap.generate(frames, prompt, max_tokens=max_tokens)

        with self._lock:
            self._cache[key] = text
        return text


def make_summarizer() -> Summarizer:
    return Summarizer()
