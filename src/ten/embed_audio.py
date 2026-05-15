"""Audio embeddings via LAION CLAP.

Adds an audio modality alongside V-JEPA visual embeddings and Qwen3-Embedding
text embeddings. CLAP has matched audio + text encoders trained contrastively,
so a text query can search audio content directly: "applause", "music with
heavy bass", "dog barking" — things ASR transcripts can't capture.

Lives in its own Qdrant collection (`ten_audio`). Search-side fusion with
text + visual collections happens in `Searcher` via RRF.

Default model: `laion/clap-htsat-fused` (handles variable-length audio
better than the unfused variant).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import CONFIG


CLAP_SAMPLE_RATE = 48_000  # what LAION CLAP was trained on


class AudioEmbedderProtocol(Protocol):
    @property
    def dim(self) -> int: ...
    def embed(self, audio_paths: list[Path]) -> np.ndarray: ...
    def embed_text(self, texts: list[str]) -> np.ndarray: ...


class CLAPEmbedder:
    """LAION CLAP wrapper. Audio and text encoders share a 512-d projection space."""

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or CONFIG.clap_model
        self._model = None
        self._processor = None
        self._dim: int | None = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import ClapModel, ClapProcessor

            # Always float32: CLAP is ~150M params (cheap), and its audio
            # processor returns float32 mel spectrograms — running the model
            # in bf16 hits a dtype mismatch in the audio branch's first conv.
            self._processor = ClapProcessor.from_pretrained(self.model_id)
            self._model = ClapModel.from_pretrained(self.model_id, dtype=torch.float32)
            self._model.eval().to(CONFIG.device)

    def _load_wav(self, path: Path) -> np.ndarray:
        import soundfile as sf

        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)  # downmix
        if sr != CLAP_SAMPLE_RATE:
            # Should not happen if extract_audio used sample_rate=48000, but
            # guard anyway. Resample with a quick linear-interp fallback.
            ratio = CLAP_SAMPLE_RATE / sr
            new_len = int(len(audio) * ratio)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len, dtype=np.float32),
                np.arange(len(audio), dtype=np.float32),
                audio,
            )
        return audio.astype(np.float32)

    @staticmethod
    def _to_tensor(out):
        # transformers 5.x get_{audio,text}_features returns BaseModelOutputWithPooling
        # whose pooler_output is the joint-space projected embedding. 4.x returned
        # the projected tensor directly.
        import torch

        if isinstance(out, torch.Tensor):
            return out
        for attr in ("pooler_output", "audio_embeds", "text_embeds"):
            if hasattr(out, attr):
                v = getattr(out, attr)
                if isinstance(v, torch.Tensor):
                    return v
        raise RuntimeError(f"Unexpected CLAP output type: {type(out).__name__}")

    @property
    def dim(self) -> int:
        if self._dim is not None:
            return self._dim
        self._ensure_loaded()
        import torch

        with torch.no_grad():
            silence = np.zeros(CLAP_SAMPLE_RATE, dtype=np.float32)
            inputs = self._processor(
                audio=[silence], sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt"
            ).to(CONFIG.device)
            v = self._to_tensor(self._model.get_audio_features(**inputs))
            self._dim = int(v.shape[-1])
        return self._dim

    def embed(self, audio_paths: list[Path]) -> np.ndarray:
        """Encode a list of WAV files to L2-normalized audio vectors (B, D)."""
        import torch

        self._ensure_loaded()
        audios = [self._load_wav(p) for p in audio_paths]
        with torch.no_grad():
            inputs = self._processor(
                audio=audios, sampling_rate=CLAP_SAMPLE_RATE, return_tensors="pt", padding=True
            ).to(CONFIG.device)
            feats = self._to_tensor(self._model.get_audio_features(**inputs))
            feats = torch.nn.functional.normalize(feats.float(), p=2, dim=-1)
        return feats.cpu().numpy()

    def embed_text(self, texts: list[str]) -> np.ndarray:
        """Encode text queries to L2-normalized vectors in the shared CLAP space."""
        import torch

        self._ensure_loaded()
        with torch.no_grad():
            inputs = self._processor(
                text=texts, return_tensors="pt", padding=True
            ).to(CONFIG.device)
            feats = self._to_tensor(self._model.get_text_features(**inputs))
            feats = torch.nn.functional.normalize(feats.float(), p=2, dim=-1)
        return feats.cpu().numpy()


def make_audio_embedder() -> AudioEmbedderProtocol | None:
    backend = os.environ.get("TEN_CLAP_BACKEND", CONFIG.clap_backend).lower()
    if backend in ("", "none", "off", "disabled", "false", "0"):
        return None
    if backend in ("clap", "laion-clap"):
        return CLAPEmbedder()
    raise ValueError(
        f"Unknown TEN_CLAP_BACKEND: {backend!r} (expected 'none' or 'clap')"
    )
