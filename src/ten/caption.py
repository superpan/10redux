"""Qwen3-VL captioner — backend-agnostic interface.

Two backends, picked via TEN_VLM_BACKEND:
  - "transformers" (default): in-process HF model. Zero extra infra.
  - "vllm": HTTP client to a vLLM OpenAI-compatible server (much faster ingest).

The CaptionerProtocol is what the rest of the pipeline (ingest, api, cli) imports.
Use `make_captioner()` to get the right implementation.
"""
from __future__ import annotations

import base64
import io
import os
import threading
from typing import Protocol, Sequence

import httpx
import numpy as np
import torch
from PIL import Image

from .config import CONFIG


_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

CAPTION_PROMPT = (
    "Describe this short video clip in one or two concise sentences. "
    "Focus on visible actions, objects, people, and scene. "
    "Be specific (concrete nouns, observable verbs). Avoid speculation."
)

SUMMARY_PROMPT = (
    "Summarize this video clip in 3-5 sentences. "
    "Cover: what is happening, who or what appears, the setting, and any notable changes over time. "
    "Be factual and specific."
)


class CaptionerProtocol(Protocol):
    def caption(self, frames: np.ndarray) -> str: ...
    def summarize(self, frames: np.ndarray) -> str: ...
    def caption_batch(self, clips_frames: Sequence[np.ndarray]) -> list[str]: ...


def _frames_to_pil(frames: np.ndarray) -> list[Image.Image]:
    return [Image.fromarray(f) for f in frames]


# ---------------------------------------------------------------------------
# Transformers backend
# ---------------------------------------------------------------------------


class TransformersCaptioner:
    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = model_id or CONFIG.vlm_model
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            from transformers import AutoModelForImageTextToText, AutoProcessor

            dtype = _DTYPE_MAP.get(CONFIG.dtype, torch.bfloat16)
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id, torch_dtype=dtype, device_map=CONFIG.device
            )
            self._model.eval()

    @torch.no_grad()
    def _generate(self, frames: np.ndarray, prompt: str, max_new_tokens: int) -> str:
        self._ensure_loaded()
        pil_frames = _frames_to_pil(frames)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], videos=[pil_frames], return_tensors="pt", padding=True
        ).to(CONFIG.device)
        gen = self._model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0
        )
        in_len = inputs["input_ids"].shape[1]
        out_tokens = gen[:, in_len:]
        return self._processor.batch_decode(
            out_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def caption(self, frames: np.ndarray) -> str:
        return self._generate(frames, CAPTION_PROMPT, max_new_tokens=96)

    def summarize(self, frames: np.ndarray) -> str:
        return self._generate(frames, SUMMARY_PROMPT, max_new_tokens=256)

    def caption_batch(self, clips_frames: Sequence[np.ndarray]) -> list[str]:
        return [self.caption(f) for f in clips_frames]


# ---------------------------------------------------------------------------
# vLLM backend (OpenAI-compatible HTTP)
# ---------------------------------------------------------------------------


def _frames_to_data_urls(frames: np.ndarray, jpeg_quality: int = 80) -> list[str]:
    urls: list[str] = []
    for f in frames:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="JPEG", quality=jpeg_quality)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        urls.append(f"data:image/jpeg;base64,{b64}")
    return urls


class VLLMCaptioner:
    """Client for a running vLLM OpenAI-compatible server.

    Multiplexes ingest by sending each clip as a chat completion with the clip's
    frames packed as multiple image_url blocks (the standard OpenAI multi-image
    payload). vLLM's continuous batching keeps the GPU saturated across requests.
    """

    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.url = (url or os.environ.get("TEN_VLLM_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.environ.get("TEN_VLLM_MODEL", CONFIG.vlm_model)
        self.api_key = api_key or os.environ.get("TEN_VLLM_API_KEY", "EMPTY")
        self.max_concurrency = int(
            max_concurrency or os.environ.get("TEN_VLLM_CONCURRENCY", "8")
        )
        self._client = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))

    def _chat(self, frames: np.ndarray, prompt: str, max_tokens: int) -> str:
        content: list[dict] = [
            {"type": "image_url", "image_url": {"url": u}} for u in _frames_to_data_urls(frames)
        ]
        content.append({"type": "text", "text": prompt})
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = self._client.post(f"{self.url}/chat/completions", json=body, headers=headers)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    def caption(self, frames: np.ndarray) -> str:
        return self._chat(frames, CAPTION_PROMPT, max_tokens=96)

    def summarize(self, frames: np.ndarray) -> str:
        return self._chat(frames, SUMMARY_PROMPT, max_tokens=256)

    def caption_batch(self, clips_frames: Sequence[np.ndarray]) -> list[str]:
        # Fan out concurrently — vLLM does the actual batching server-side via continuous batching.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
            return list(ex.map(self.caption, clips_frames))


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_captioner() -> CaptionerProtocol:
    backend = os.environ.get("TEN_VLM_BACKEND", "transformers").lower()
    if backend == "vllm":
        return VLLMCaptioner()
    if backend == "transformers":
        return TransformersCaptioner()
    raise ValueError(f"Unknown TEN_VLM_BACKEND: {backend!r} (expected 'transformers' or 'vllm')")


# Backwards-compat shim: old code imported `Captioner` directly.
Captioner = TransformersCaptioner
