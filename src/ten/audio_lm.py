"""Audio language-model wrapper — replaces Whisper + CLAP + Silero VAD with one model.

Architectural intent
--------------------

ten today runs three chained audio encoders:

  Whisper       -> literal speech transcript (words spoken)
  CLAP          -> audio<->text contrastive (semantic match), reranker only
  Silero VAD    -> energy-based gate so Whisper doesn't hallucinate on music

The QVHighlights eval against TwelveLabs Marengo 3.0 (see EVAL.md) showed
3 of the 9 R@1 gap queries are *abstract speech-act semantics* —
"monologue", "sing", "talking and eating" — that the chained encoders
can't surface. Marengo wins them because joint training learned the
audio -> abstract-description association directly.

ten is already architected around the captioner pattern:
  Qwen3-VL  -> visual caption -> Qwen3-Embedding -> ten_text -> search

This module makes audio symmetric:
  Voxtral (or MOSS-Audio) -> audio caption + transcript ->  concatenated into
  the visual caption text -> Qwen3-Embedding -> ten_text -> search

Net effect on the index: `ten_audio` collection goes away, `ten_text` gets
better fill for audio-cue queries, retrieval is one channel not three.

Backend notes
-------------

`voxtral` (default) loads `mistralai/Voxtral-Mini-3B-2507` via the official
`VoxtralForConditionalGeneration` class in transformers ≥ 5.x. Apache 2.0,
clean HF Hub packaging, mature deployment story.

`moss` was the original recommendation (Qwen3 backbone alignment with the
rest of ten) but the OpenMOSS-Team/MOSS-Audio-4B-Instruct Hub snapshot
ships `configuration_moss_audio.py` + `processing_moss_audio.py` but
*not* `modeling_moss_audio.py` (which lives only in their GitHub repo).
`AutoModel*.from_pretrained` therefore can't instantiate it. The MOSS
backend stays stubbed until OpenMOSS either lands the modeling file in
the Hub snapshot or registers the class via `auto_map`. The fallback
recipe (vendor `modeling_moss_audio.py` from their `src/` into ten) is
intentionally not taken here — it'd drag in their full requirements tree.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Protocol

from .config import CONFIG


class AudioLMProtocol(Protocol):
    """Produces both an utterance transcript and a free-form audio caption.

    Returning empty strings is normal and meaningful — empty transcript means
    no speech detected, empty caption means no notable audio. Callers should
    `if (transcript or caption):` before doing anything with the result.
    """

    def transcribe(self, audio_path: Path) -> str: ...
    def caption(self, audio_path: Path) -> str: ...
    def describe(self, audio_path: Path) -> tuple[str, str]: ...


# ---------------------------------------------------------------------------
# Voxtral (Mistral) — Apache 2.0, official transformers class
# ---------------------------------------------------------------------------


class VoxtralAudioLM:
    """Wraps `mistralai/Voxtral-Mini-3B-2507` (default) via the official
    `VoxtralForConditionalGeneration` class.

    Lazy-loaded behind a lock. Uses the model's chat template for the audio
    caption path and the dedicated `apply_transcription_request` helper for
    the transcript path.
    """

    def __init__(
        self,
        model_id: str | None = None,
        caption_prompt: str | None = None,
        transcribe_language: str | None = None,
    ) -> None:
        self.model_id = model_id or CONFIG.audio_lm_model
        self.caption_prompt = caption_prompt or CONFIG.audio_lm_caption_prompt
        # If None, processor.apply_transcription_request will auto-detect.
        self.transcribe_language = transcribe_language or CONFIG.audio_lm_transcribe_language
        self._model = None
        self._processor = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoProcessor, VoxtralForConditionalGeneration

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(CONFIG.dtype, torch.bfloat16)
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            # `low_cpu_mem_usage=True` skips PyTorch's caching_allocator_warmup,
            # which probes a large contiguous block on the GPU and OOMs on
            # GB10 unified memory even when plenty is free.
            self._model = VoxtralForConditionalGeneration.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            ).to(CONFIG.device)

    def _decode(self, inputs, max_new_tokens: int) -> str:
        import torch

        # Voxtral expects bf16 inputs and returns generated_ids that include
        # the prompt; slice past inputs.input_ids.shape[1] to get only the
        # newly generated tokens.
        dtype = next(self._model.parameters()).dtype
        inputs = inputs.to(self._model.device, dtype=dtype)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        new_tokens = out[:, inputs.input_ids.shape[1]:]
        decoded = self._processor.batch_decode(new_tokens, skip_special_tokens=True)
        return decoded[0].strip()

    def transcribe(self, audio_path: Path) -> str:
        self._ensure_loaded()
        inputs = self._processor.apply_transcription_request(
            language=self.transcribe_language,
            audio=str(audio_path),
            model_id=self.model_id,
        )
        return self._decode(inputs, CONFIG.audio_lm_max_new_tokens)

    def caption(self, audio_path: Path) -> str:
        self._ensure_loaded()
        conversation = [{
            "role": "user",
            "content": [
                {"type": "audio", "path": str(audio_path)},
                {"type": "text", "text": self.caption_prompt},
            ],
        }]
        inputs = self._processor.apply_chat_template(conversation)
        return self._decode(inputs, CONFIG.audio_lm_max_new_tokens)

    def describe(self, audio_path: Path) -> tuple[str, str]:
        # Two prompts means two forward passes today. A follow-up should fuse
        # them in one call once a multi-task prompt is validated.
        return self.transcribe(audio_path), self.caption(audio_path)


# ---------------------------------------------------------------------------
# MOSS-Audio — stubbed; see backend notes in the module docstring
# ---------------------------------------------------------------------------


class MOSSAudioLMStub:
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "MOSS-Audio backend is stubbed: the Hub snapshot at "
            "OpenMOSS-Team/MOSS-Audio-4B-Instruct ships configuration_moss_audio.py "
            "and processing_moss_audio.py but NOT modeling_moss_audio.py, so "
            "from_pretrained can't instantiate. Use TEN_AUDIO_LM_BACKEND=voxtral "
            "or vendor MOSS-Audio's src/modeling_moss_audio.py into ten. "
            "See src/ten/audio_lm.py module docstring."
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_audio_lm() -> AudioLMProtocol | None:
    """Return an audio-LM wrapper per TEN_AUDIO_LM_BACKEND, or None if disabled.

    When this returns a non-None instance, the ingest pipeline should bypass
    the legacy ASR / VAD / CLAP path entirely — the audio LM replaces all
    three. See `ingest.py` for the dispatch.
    """
    backend = os.environ.get("TEN_AUDIO_LM_BACKEND", CONFIG.audio_lm_backend).lower()
    if backend in ("", "none", "off", "disabled", "false", "0"):
        return None
    if backend in ("voxtral",):
        return VoxtralAudioLM()
    if backend in ("moss", "mossaudio", "moss-audio"):
        return MOSSAudioLMStub()
    raise ValueError(
        f"Unknown TEN_AUDIO_LM_BACKEND: {backend!r} "
        "(expected 'none' | 'voxtral' | 'moss')"
    )
