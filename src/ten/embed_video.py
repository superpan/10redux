"""V-JEPA 2 video encoder wrapper.

Produces a single L2-normalized vector per clip by mean-pooling the
encoder's spatio-temporal tokens. Loads lazily on first use.
"""
from __future__ import annotations

import threading
from typing import Sequence

import numpy as np
import torch

from .config import CONFIG


_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class VideoEmbedder:
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or CONFIG.vjepa_model
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
            from transformers import AutoModel, AutoVideoProcessor

            dtype = _DTYPE_MAP.get(CONFIG.dtype, torch.bfloat16)
            self._processor = AutoVideoProcessor.from_pretrained(self.model_id)
            self._model = AutoModel.from_pretrained(self.model_id, dtype=dtype)
            self._model.eval().to(CONFIG.device)

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        if self._dim is None:
            with torch.no_grad():
                dummy = np.zeros((CONFIG.frames_per_clip, CONFIG.frame_resize, CONFIG.frame_resize, 3), dtype=np.uint8)
                v = self.embed([dummy])
                self._dim = v.shape[-1]
        return self._dim

    @torch.no_grad()
    def embed(self, clips_frames: Sequence[np.ndarray]) -> np.ndarray:
        """clips_frames: list of (T, H, W, 3) uint8 arrays. Returns (B, D) float32, L2-normalized."""
        self._ensure_loaded()
        # AutoVideoProcessor accepts list[list[ndarray]] (one list of frames per video).
        videos = [list(c) for c in clips_frames]
        inputs = self._processor(videos=videos, return_tensors="pt")
        inputs = {k: v.to(CONFIG.device) for k, v in inputs.items()}

        # V-JEPA2 returns last_hidden_state of shape (B, N_tokens, D).
        out = self._model(**inputs)
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            # Some checkpoints expose `pooler_output` or just the bare tensor.
            hidden = out[0] if isinstance(out, tuple) else out
        pooled = hidden.mean(dim=1)
        pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
        return pooled.cpu().numpy()
