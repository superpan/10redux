"""Audio extraction from a video clip window.

Used by the ASR layer at ingest time. We shell out to ffmpeg here (rather than
PyAV) because ffmpeg's `-ss / -t` flags + `-ac 1 -ar 16000` produce exactly the
mono 16 kHz wav that Whisper-family models want, with one process call.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .video import Clip


def extract_audio(clip: Clip, dest: Path, sample_rate: int = 16000) -> bool:
    """Extract `clip`'s audio range to `dest` as `sample_rate` mono WAV.

    Whisper wants 16 kHz; CLAP wants 48 kHz. Pass the right sample_rate so we
    don't have to resample in Python.

    Returns True on success, False if the source has no audio stream or ffmpeg
    failed (treated as "no audio available").
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH; install it (`make ffmpeg`)")
    duration = clip.t_end - clip.t_start
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{clip.t_start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(clip.video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        str(dest),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    except subprocess.CalledProcessError:
        return False
    except subprocess.TimeoutExpired:
        return False
    # ffmpeg writes ~80 bytes of WAV header even when there is no audio data.
    if not dest.exists() or dest.stat().st_size < 200:
        return False
    return True
