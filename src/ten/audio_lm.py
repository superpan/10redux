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
  MOSS-Audio (or Voxtral)  -> audio caption + transcript ->  concatenated into
  the visual caption text  ->  Qwen3-Embedding -> ten_text  -> search

Net effect on the index: `ten_audio` collection goes away, `ten_text` gets
better fill for audio-cue queries, retrieval is one channel not three.

Status
------

This is the **branch sketch**. The asr.py / embed_audio.py / vad.py modules
are left in place so the old path keeps working when `TEN_AUDIO_LM_BACKEND`
is unset. Real cleanup (deleting those modules + the `ten_audio` collection
+ the audio-rerank knobs in config + the rerank logic in Searcher) follows
once we have eval numbers showing the new path is at least as good.
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
    def describe(self, audio_path: Path) -> tuple[str, str]:
        """Single audio decode that returns (transcript, caption).

        Default implementation calls the two separately; subclasses can override
        to batch or fuse in one forward pass when the model supports it.
        """
        return self.transcribe(audio_path), self.caption(audio_path)


# ---------------------------------------------------------------------------
# MOSS-Audio (OpenMOSS) — Qwen3 backbone, audio understanding + ASR + captioning
# ---------------------------------------------------------------------------


_DEFAULT_TRANSCRIBE_PROMPT = (
    "Transcribe the speech in this audio. If there is no speech, respond with "
    "exactly an empty string and nothing else."
)
_DEFAULT_CAPTION_PROMPT = (
    "In one factual sentence, describe the audio content: the speech style "
    "(monologue, conversation, singing), any music, and any prominent ambient "
    "or environmental sounds. Do NOT transcribe specific words. If the audio "
    "is silent or has nothing notable, respond with exactly an empty string."
)


class MOSSAudioLM:
    """Wraps `OpenMOSS-Team/MOSS-Audio-4B-Instruct` (default) or 8B variant.

    Lazy-loaded behind a lock so server start stays fast. CUDA / bf16 by
    default to match the rest of ten's model wrappers.

    Prompt-driven: the same model runs transcribe vs. caption by changing
    the system instruction. The two-call default is the safe baseline;
    `describe()` can be overridden to a single forward pass once the
    model's chat template is wired up (the upstream `infer.py` is the
    reference — see the repo README for the exact prompt format).
    """

    def __init__(
        self,
        model_id: str | None = None,
        transcribe_prompt: str | None = None,
        caption_prompt: str | None = None,
    ) -> None:
        self.model_id = model_id or CONFIG.audio_lm_model
        self.transcribe_prompt = transcribe_prompt or CONFIG.audio_lm_transcribe_prompt
        self.caption_prompt = caption_prompt or CONFIG.audio_lm_caption_prompt
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
            from transformers import AutoModelForCausalLM, AutoProcessor

            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            dtype = dtype_map.get(CONFIG.dtype, torch.bfloat16)
            self._processor = AutoProcessor.from_pretrained(
                self.model_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id, dtype=dtype, trust_remote_code=True
            )
            self._model.eval().to(CONFIG.device)

    def _generate(self, audio_path: Path, prompt: str) -> str:
        """Run a single prompt against an audio file and return the text.

        The exact tensor-construction call depends on MOSS-Audio's processor
        API (which is custom — `trust_remote_code=True` above). The shape
        below follows the common pattern of multimodal processors in
        transformers: pass audio + a chat-template message, decode the
        generated continuation. Adapt to match the upstream `infer.py` once
        the model is downloaded for a smoke test.
        """
        import torch

        self._ensure_loaded()
        messages = [
            {"role": "user", "content": [
                {"type": "audio", "audio": str(audio_path)},
                {"type": "text", "text": prompt},
            ]}
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(CONFIG.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=CONFIG.audio_lm_max_new_tokens,
                do_sample=False,
            )
        prompt_len = inputs["input_ids"].shape[-1]
        new_tokens = out[0, prompt_len:]
        return self._processor.decode(new_tokens, skip_special_tokens=True).strip()

    def transcribe(self, audio_path: Path) -> str:
        return self._generate(audio_path, self.transcribe_prompt)

    def caption(self, audio_path: Path) -> str:
        return self._generate(audio_path, self.caption_prompt)

    def describe(self, audio_path: Path) -> tuple[str, str]:
        # Two prompts means two forward passes today. A follow-up should fuse
        # them in one call once the chat template confirms the model accepts a
        # multi-instruction prompt that yields both outputs in one decode.
        return self.transcribe(audio_path), self.caption(audio_path)


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
    if backend in ("moss", "mossaudio", "moss-audio"):
        return MOSSAudioLM()
    # Voxtral hook left here for the follow-up (3B Apache 2.0, 30-min context).
    # Implementation mirrors MOSSAudioLM but with Mistral's processor API.
    if backend in ("voxtral",):
        raise NotImplementedError(
            "Voxtral backend not yet wired; see audio_lm.py header for the plan."
        )
    raise ValueError(
        f"Unknown TEN_AUDIO_LM_BACKEND: {backend!r} "
        "(expected 'none' | 'moss' | 'voxtral')"
    )
