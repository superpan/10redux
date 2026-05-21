from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class Config:
    # XDG-style default so `uv tool install .` works from any directory.
    # Override via TEN_DATA_DIR=. to keep the previous project-relative behavior.
    data_dir: Path = field(
        default_factory=lambda: Path(
            _env("TEN_DATA_DIR", str(Path.home() / ".local" / "share" / "ten"))
        ).resolve()
    )
    qdrant_url: str = field(default_factory=lambda: _env("TEN_QDRANT_URL", "http://localhost:6333"))
    qdrant_api_key: str | None = field(default_factory=lambda: os.environ.get("TEN_QDRANT_API_KEY"))

    visual_collection: str = "ten_visual"
    text_collection: str = "ten_text"
    audio_collection: str = "ten_audio"

    vjepa_model: str = field(
        default_factory=lambda: _env("TEN_VJEPA_MODEL", "facebook/vjepa2-vitl-fpc16-256-ssv2")
    )
    vlm_model: str = field(
        default_factory=lambda: _env("TEN_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    )
    text_embed_model: str = field(
        default_factory=lambda: _env("TEN_TEXT_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    )

    # ASR (off by default — opt-in via TEN_ASR_BACKEND=whisper or `ten index --asr`).
    # When enabled, each clip gets a Whisper transcript that is concatenated
    # with the visual caption before the text embedding step.
    asr_backend: str = field(default_factory=lambda: _env("TEN_ASR_BACKEND", "none"))
    asr_model: str = field(default_factory=lambda: _env("TEN_ASR_MODEL", "large-v3"))
    asr_language: str | None = field(
        default_factory=lambda: os.environ.get("TEN_ASR_LANGUAGE") or None
    )
    asr_compute_type: str = field(
        default_factory=lambda: _env("TEN_ASR_COMPUTE_TYPE", "float16")
    )

    # Voice Activity Detection — pre-gate on Whisper. Off by default; opt-in via
    # TEN_VAD_BACKEND=silero or `ten index --vad`. Has no effect without ASR on.
    # Kills the Whisper-on-music (`¶¶¶`), Whisper-on-wind, and Whisper-on-silence
    # hallucination failure modes by skipping transcription on clips with little
    # detected speech. Contamination measured before VAD: ~11% of audio-bearing
    # clips, up to ~35% on outdoor / music-heavy content.
    vad_backend: str = field(default_factory=lambda: _env("TEN_VAD_BACKEND", "none"))
    vad_min_speech_fraction: float = float(_env("TEN_VAD_MIN_SPEECH_FRACTION", "0.10"))

    # Reranker (off by default — opt-in via TEN_RERANKER_BACKEND=crossencoder).
    # When enabled, the bi-encoder's per_source results are re-scored by a
    # cross-encoder before the limit cut.
    reranker_backend: str = field(
        default_factory=lambda: _env("TEN_RERANKER_BACKEND", "none")
    )
    reranker_model: str = field(
        default_factory=lambda: _env("TEN_RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
    )
    reranker_top_k: int = int(_env("TEN_RERANKER_TOP_K", "100"))

    # CLAP audio embeddings (off by default — opt-in via TEN_CLAP_BACKEND=clap).
    # Adds a third Qdrant collection (ten_audio). Search uses CLAP as a bounded
    # post-fusion reranker (not a peer RRF source — equal-weight 3-way RRF
    # regressed retrieval in the QVHighlights pilot because CLAP's text encoder
    # is trained for audio alignment, not general semantics).
    clap_backend: str = field(default_factory=lambda: _env("TEN_CLAP_BACKEND", "none"))
    clap_model: str = field(
        default_factory=lambda: _env("TEN_CLAP_MODEL", "laion/clap-htsat-fused")
    )
    audio_rerank_top_k: int = int(_env("TEN_AUDIO_RERANK_TOP_K", "30"))
    audio_rerank_threshold: float = float(_env("TEN_AUDIO_RERANK_THRESHOLD", "0.30"))
    audio_rerank_weight: float = float(_env("TEN_AUDIO_RERANK_WEIGHT", "0.03"))

    clip_seconds: float = float(_env("TEN_CLIP_SECONDS", "10"))
    clip_overlap: float = float(_env("TEN_CLIP_OVERLAP", "1"))
    # 8 frames matches V-JEPA 2's spatial-temporal budget well enough and
    # halves Qwen3-VL prefill cost. Empirically (MSR-VTT 1K-A): R@1 0.343 -> 0.338,
    # R@5/10 essentially unchanged, ingest -14%. Bump to 16 for marginally better
    # captions on motion-heavy footage.
    frames_per_clip: int = int(_env("TEN_FRAMES_PER_CLIP", "8"))
    frame_resize: int = int(_env("TEN_FRAME_RESIZE", "256"))

    device: str = field(default_factory=lambda: _env("TEN_DEVICE", "cuda"))
    dtype: str = field(default_factory=lambda: _env("TEN_DTYPE", "bfloat16"))
    batch_size: int = int(_env("TEN_BATCH_SIZE", "4"))

    host: str = field(default_factory=lambda: _env("TEN_HOST", "127.0.0.1"))
    port: int = int(_env("TEN_PORT", "8765"))

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.thumbs_dir.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
