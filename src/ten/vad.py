"""Voice Activity Detection — pre-gate on Whisper to skip non-speech clips.

Whisper hallucinates on music, wind, and near-silence (see EVAL.md § ASR
caveats). Across our index ~11% of audio-bearing clips end up with garbage
transcripts (`¶¶¶`, repetition loops, single-word fillers); on outdoor /
music-heavy content the rate is ~35%. Running Silero VAD over the WAV
*before* invoking Whisper turns those clips into empty transcripts instead
of hallucinated noise.

Silero VAD is ~2 MB ONNX, runs on CPU at ~1 ms / sec of audio, so it doesn't
contend with the GPU pipeline. Lazy-loaded behind a lock like other model
wrappers in this codebase. Opt-in via `TEN_VAD_BACKEND=silero`.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Protocol

from .config import CONFIG


class VADProtocol(Protocol):
    def speech_fraction(self, audio_path: Path) -> float: ...


class SileroVAD:
    """Silero VAD via the `silero-vad` pip package (ONNX, CPU-friendly).

    `speech_fraction` returns the share of the input audio that VAD marks
    as speech, in [0, 1]. The caller decides the threshold for "enough
    speech to bother transcribing" (see CONFIG.vad_min_speech_fraction).
    """

    def __init__(self, sampling_rate: int = 16000, threshold: float = 0.5) -> None:
        self.sampling_rate = sampling_rate
        self.threshold = threshold
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from silero_vad import load_silero_vad

            self._model = load_silero_vad(onnx=True)

    def speech_fraction(self, audio_path: Path) -> float:
        self._ensure_loaded()
        import numpy as np
        import soundfile as sf
        import torch
        from silero_vad import get_speech_timestamps

        # Load via soundfile to dodge silero-vad's torchaudio dependency
        # (torchaudio 2.11 outsources audio I/O to a separate torchcodec pkg).
        audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != self.sampling_rate:
            ratio = self.sampling_rate / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len, dtype=np.float32),
                np.arange(len(audio), dtype=np.float32),
                audio,
            ).astype(np.float32)
        if len(audio) == 0:
            return 0.0
        wav = torch.from_numpy(audio)
        segs = get_speech_timestamps(
            wav,
            self._model,
            sampling_rate=self.sampling_rate,
            threshold=self.threshold,
        )
        if not segs:
            return 0.0
        speech = sum(s["end"] - s["start"] for s in segs)
        return float(speech) / float(len(wav))


def make_vad() -> VADProtocol | None:
    backend = os.environ.get("TEN_VAD_BACKEND", CONFIG.vad_backend).lower()
    if backend in ("", "none", "off", "disabled", "false", "0"):
        return None
    if backend == "silero":
        return SileroVAD()
    raise ValueError(
        f"Unknown TEN_VAD_BACKEND: {backend!r} (expected 'none' or 'silero')"
    )
