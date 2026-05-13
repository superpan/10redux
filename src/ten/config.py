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

    vjepa_model: str = field(
        default_factory=lambda: _env("TEN_VJEPA_MODEL", "facebook/vjepa2-vitl-fpc16-256-ssv2")
    )
    vlm_model: str = field(
        default_factory=lambda: _env("TEN_VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    )
    text_embed_model: str = field(
        default_factory=lambda: _env("TEN_TEXT_EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
    )

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
