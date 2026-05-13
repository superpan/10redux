"""ASR layer — runs Whisper-family models to produce transcripts per clip.

Two backends, both lazy-loaded so processes that don't enable ASR pay nothing:

- "whisper" (default): HF transformers + the project's existing torch CUDA.
  Works on aarch64 / Blackwell where CTranslate2 wheels lack CUDA support.
- "fasterwhisper": faster-whisper (CTranslate2). Faster on x86_64 where
  prebuilt wheels include CUDA. On aarch64 it falls back to CPU and is slow.

Enable via `TEN_ASR_BACKEND=whisper` (env) or `ten index --asr` (CLI).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Protocol

from .config import CONFIG


class TranscriberProtocol(Protocol):
    def transcribe(self, audio_path: Path) -> str: ...


# Map short Whisper names to canonical HF model ids (transformers needs the full
# id; faster-whisper accepts either).
_WHISPER_HF_ID = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo",
}


def _resolve_whisper_id(model_id: str) -> str:
    return _WHISPER_HF_ID.get(model_id, model_id)


# ---------------------------------------------------------------------------
# Transformers backend (default — uses the project's existing torch CUDA)
# ---------------------------------------------------------------------------


class TransformersWhisperTranscriber:
    """HF transformers Whisper via the ASR pipeline.

    Slightly slower than faster-whisper on x86, but the only backend that
    GPU-accelerates on the Blackwell aarch64 stack (CTranslate2 wheels there
    are CPU-only).
    """

    def __init__(
        self,
        model_id: str | None = None,
        language: str | None = None,
    ) -> None:
        self.model_id = _resolve_whisper_id(model_id or CONFIG.asr_model)
        self.language = language if language is not None else CONFIG.asr_language
        self._pipe = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._pipe is not None:
            return
        with self._lock:
            if self._pipe is not None:
                return
            import torch
            from transformers import pipeline

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(CONFIG.dtype, torch.bfloat16)
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_id,
                dtype=dtype,
                device=CONFIG.device if CONFIG.device.startswith("cuda") else -1,
            )

    def transcribe(self, audio_path: Path) -> str:
        self._ensure_loaded()
        gen_kwargs: dict = {}
        if self.language:
            gen_kwargs["language"] = self.language
        result = self._pipe(
            str(audio_path),
            chunk_length_s=30,
            return_timestamps=False,
            generate_kwargs=gen_kwargs,
        )
        text = result["text"] if isinstance(result, dict) else result
        return (text or "").strip()


# ---------------------------------------------------------------------------
# faster-whisper backend (x86 / GPU CTranslate2 only)
# ---------------------------------------------------------------------------


class FasterWhisperTranscriber:
    """faster-whisper (CTranslate2 build of Whisper).

    Multilingual, robust to music/noise, ~5× faster than HF Whisper at the
    same model size. CTranslate2 picks up CUDA automatically; on systems
    without GPU support, falls back to CPU at much lower throughput.
    """

    def __init__(
        self,
        model_id: str | None = None,
        language: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self.model_id = model_id or CONFIG.asr_model
        self.language = language if language is not None else CONFIG.asr_language
        self.compute_type = compute_type or CONFIG.asr_compute_type
        self._model = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from faster_whisper import WhisperModel

            device = "cuda" if CONFIG.device.startswith("cuda") else "cpu"
            self._model = WhisperModel(
                self.model_id,
                device=device,
                compute_type=self.compute_type,
            )

    def transcribe(self, audio_path: Path) -> str:
        self._ensure_loaded()
        segments, _info = self._model.transcribe(
            str(audio_path),
            language=self.language,  # None = auto-detect
            beam_size=5,
            vad_filter=True,  # skip pure silence so we don't hallucinate text
        )
        return " ".join(s.text.strip() for s in segments).strip()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_transcriber() -> TranscriberProtocol | None:
    """Return a transcriber per TEN_ASR_BACKEND, or None if ASR is disabled."""
    backend = os.environ.get("TEN_ASR_BACKEND", CONFIG.asr_backend).lower()
    if backend in ("", "none", "off", "disabled", "false", "0"):
        return None
    if backend in ("whisper", "transformers"):
        return TransformersWhisperTranscriber()
    if backend in ("fasterwhisper", "faster-whisper"):
        return FasterWhisperTranscriber()
    raise ValueError(
        f"Unknown TEN_ASR_BACKEND: {backend!r} "
        "(expected 'none' | 'whisper' | 'fasterwhisper')"
    )
