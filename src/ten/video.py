"""Video probing, clip enumeration, and uniform frame sampling via PyAV.

Why PyAV (not decord): PyAV ships aarch64 wheels and binds libav directly, so it
works on DGX Spark (Grace Blackwell) where decord has no prebuilt wheel.
"""
from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
from PIL import Image

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".ts", ".wmv"}


from functools import lru_cache


@lru_cache(maxsize=8192)
def _video_rotation(video_path: str) -> int:
    """Return display rotation in CCW degrees (0/90/180/270) for a video.

    iPhone (and many other phone) videos store sensor-orientation pixels with a
    display-matrix side-data tag specifying the rotation needed to display them
    correctly. PyAV's `frame.to_ndarray()` ignores this tag and returns raw
    landscape pixels, so we need to read the tag and apply `np.rot90` to frames
    ourselves. Without this every iPhone-portrait clip would be ingested
    sideways (wrong captions, wrong embeddings, wrong thumbnails).

    Probed via ffprobe (~50–200 ms once per file, cached). Returns 0 if no
    rotation tag is present (already-correct codec orientation).
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream_side_data=rotation",
                "-of", "default=nw=1:nk=1",
                video_path,
            ],
            capture_output=True, text=True, timeout=5,
        )
        line = (result.stdout or "").strip().splitlines()
        rot = int(line[0]) if line and line[0].lstrip("-").isdigit() else 0
    except Exception:
        return 0
    # Normalize to [0, 360) and snap to nearest 90.
    return ((rot % 360) // 90) * 90


def _apply_rotation(img: np.ndarray, rotation_ccw_deg: int) -> np.ndarray:
    """Apply CCW rotation in 90° steps. `np.rot90(k=1)` is CCW by 90°."""
    if not rotation_ccw_deg:
        return img
    return np.rot90(img, k=rotation_ccw_deg // 90)


@dataclass(frozen=True)
class VideoMeta:
    path: Path
    duration: float
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class Clip:
    video_path: Path
    t_start: float
    t_end: float

    @property
    def clip_id(self) -> str:
        # Stable hash of absolute path + window. Used as Qdrant point id (int via uuid5-style).
        h = hashlib.blake2b(
            f"{self.video_path.resolve()}|{self.t_start:.3f}|{self.t_end:.3f}".encode(),
            digest_size=16,
        ).hexdigest()
        return h


def probe(path: Path) -> VideoMeta:
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError(f"No video stream in {path}")
        duration = float(container.duration / av.time_base) if container.duration else 0.0
        if duration == 0.0 and stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        width = stream.codec_context.width
        height = stream.codec_context.height
    return VideoMeta(path=path, duration=duration, fps=fps, width=width, height=height)


def enumerate_clips(meta: VideoMeta, clip_seconds: float, overlap: float) -> list[Clip]:
    if meta.duration <= 0:
        return []
    step = max(0.5, clip_seconds - overlap)
    n = max(1, math.ceil((meta.duration - overlap) / step))
    clips: list[Clip] = []
    for i in range(n):
        t0 = i * step
        t1 = min(meta.duration, t0 + clip_seconds)
        if t1 - t0 < 1.0:
            break
        clips.append(Clip(video_path=meta.path, t_start=t0, t_end=t1))
    return clips


def walk_videos(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            out.append(p)
    return sorted(out)


def sample_clip_frames(clip: Clip, num_frames: int, resize: int) -> np.ndarray:
    """Decode `num_frames` uniformly spaced RGB frames from the clip window.

    Frames are display-rotated per the container's display-matrix tag, so the
    captioner and visual encoder see upright pixels regardless of how the
    camera held the sensor (cf. `_video_rotation`).

    Returns array shaped (num_frames, H, W, 3) uint8.
    """
    target_times = np.linspace(clip.t_start, clip.t_end, num_frames + 1)[:num_frames]
    frames: list[np.ndarray] = []
    rotation = _video_rotation(str(clip.video_path.resolve()))

    with av.open(str(clip.video_path)) as container:
        stream = next(s for s in container.streams if s.type == "video")
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base) if stream.time_base else 1 / 1000

        # Seek slightly before start, then read sequentially.
        seek_pts = int(max(0, clip.t_start - 0.5) / time_base)
        try:
            container.seek(seek_pts, stream=stream, any_frame=False, backward=True)
        except av.AVError:
            container.seek(0)

        target_idx = 0
        last_frame = None
        for frame in container.decode(stream):
            t = float(frame.pts * time_base) if frame.pts is not None else 0.0
            while target_idx < num_frames and t >= target_times[target_idx]:
                use = frame if last_frame is None else last_frame
                img = use.to_ndarray(format="rgb24")
                img = _apply_rotation(img, rotation)
                frames.append(_resize_frame(img, resize))
                target_idx += 1
            last_frame = frame
            if target_idx >= num_frames or t > clip.t_end:
                break

    # Pad by repeating last frame if decoding fell short (very short clips).
    if not frames:
        raise RuntimeError(f"No frames decoded for {clip.video_path} {clip.t_start:.2f}-{clip.t_end:.2f}")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames, axis=0)


def _resize_frame(img: np.ndarray, short_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if min(h, w) == short_side:
        return img
    scale = short_side / min(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    pil = Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR)
    # Center-crop to short_side x short_side
    left = (new_w - short_side) // 2
    top = (new_h - short_side) // 2
    pil = pil.crop((left, top, left + short_side, top + short_side))
    return np.asarray(pil)


def write_thumbnail(clip: Clip, out_path: Path, resize: int = 320) -> None:
    """Grab the middle frame of the clip and save as JPEG.

    Applies the container's display rotation (cf. `_video_rotation`) so
    portrait phone footage doesn't land sideways.
    """
    t_mid = (clip.t_start + clip.t_end) / 2
    rotation = _video_rotation(str(clip.video_path.resolve()))
    with av.open(str(clip.video_path)) as container:
        stream = next(s for s in container.streams if s.type == "video")
        stream.thread_type = "AUTO"
        time_base = float(stream.time_base) if stream.time_base else 1 / 1000
        try:
            container.seek(int(max(0, t_mid - 0.5) / time_base), stream=stream, backward=True)
        except av.AVError:
            container.seek(0)
        for frame in container.decode(stream):
            t = float(frame.pts * time_base) if frame.pts is not None else 0.0
            if t >= t_mid:
                img = frame.to_ndarray(format="rgb24")
                img = _apply_rotation(img, rotation)
                pil = Image.fromarray(_resize_frame(img, resize))
                out_path.parent.mkdir(parents=True, exist_ok=True)
                pil.save(out_path, format="JPEG", quality=82)
                return
    raise RuntimeError(f"Could not extract thumbnail at t={t_mid:.2f}s from {clip.video_path}")


def has_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
