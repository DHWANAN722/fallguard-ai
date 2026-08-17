"""
video.py — Frame sampling and annotated-clip writing for uploaded video.

A 30 s clip at 30 fps is 900 frames; running BlazePose over all of them inside a
Streamlit request would take minutes and time the container out. So frames are
sampled at a target analysis rate (default 6 fps) which is dense enough to
resolve a fall — falls take 0.4-0.8 s from loss of balance to impact — while
keeping a 30 s clip to ~180 inferences.

Timestamps are kept in *real seconds* rather than frame indices so the pelvis
descent velocity in ``infer.py`` stays physically meaningful regardless of the
source frame rate or the sampling stride.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np


def probe(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("could not open video — the codec may be unsupported")
    info = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)) or 25.0,
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    info["duration"] = info["frames"] / info["fps"] if info["fps"] else 0.0
    cap.release()
    return info


def iter_frames(path: str, target_fps: float = 6.0, max_frames: int = 300):
    """Yield ``(frame_index, timestamp_seconds, bgr_frame)`` at ~target_fps."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("could not open video — the codec may be unsupported")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    stride = max(1, int(round(src_fps / max(target_fps, 0.1))))

    idx = emitted = 0
    try:
        while emitted < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                yield idx, idx / src_fps, frame
                emitted += 1
            idx += 1
    finally:
        cap.release()


#: fourccs a browser's <video> element can actually decode. `mp4v` is
#: MPEG-4 Part 2 — OpenCV will happily write it, but Chrome and Safari will not
#: play it, which would render an inert black player in the dashboard. So the
#: codec that succeeded is reported back and the caller decides.
BROWSER_SAFE = {"avc1", "H264"}


def write_gif(
    frames: list[np.ndarray],
    fps: float = 6.0,
    max_width: int = 420,
) -> str | None:
    """Encode annotated frames to a looping GIF.

    A GIF is used for the *primary* on-page playback rather than the MP4
    because a browser plays it automatically, forever, with no controls and no
    click required — the annotated feed simply runs, which is what a monitoring
    wall would look like. ``st.video`` by contrast renders a paused player with
    a play button, so the reviewer sees a still frame unless they interact.

    Frames are downscaled and colour-quantised because GIF is a 256-colour
    format with no interframe compression; at full 540x960 a 14-frame clip is
    several megabytes, and this repository is served to every visitor.
    """
    if not frames:
        return None

    try:
        from PIL import Image
    except ImportError:                                # pragma: no cover
        return None

    h, w = frames[0].shape[:2]
    if w > max_width:
        scale = max_width / w
        size = (max_width, int(round(h * scale)))
    else:
        size = (w, h)

    pil = []
    for f in frames:
        img = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        if size != (w, h):
            img = img.resize(size, Image.LANCZOS)
        # ADAPTIVE keeps the neon skeleton colours distinct against the
        # photographic background; the default web palette muddies them.
        pil.append(img.convert("P", palette=Image.ADAPTIVE, colors=128))

    path = os.path.join(tempfile.gettempdir(), f"fallguard_{os.getpid()}.gif")
    pil[0].save(
        path, save_all=True, append_images=pil[1:],
        duration=int(1000 / max(fps, 0.1)), loop=0, optimize=True,
        disposal=2,
    )
    return path if os.path.getsize(path) > 512 else None


def _encode_h264(frames: list[np.ndarray], fps: float, path: str) -> bool:
    """Encode to real H.264 via the ffmpeg binary bundled with imageio-ffmpeg.

    OpenCV's wheels ship no H.264 encoder on Streamlit Cloud — `avc1`/`H264`
    fail and it falls back to `mp4v` (MPEG-4 Part 2), which OpenCV writes
    happily and no browser will play. The result is an inert black player.
    imageio-ffmpeg bundles a static ffmpeg with libx264, so this path produces
    a file `st.video` can actually play, with `faststart` so it begins before
    the whole file has buffered.
    """
    import subprocess

    import imageio_ffmpeg

    h, w = frames[0].shape[:2]
    w -= w % 2                       # libx264 requires even dimensions
    h -= h % 2

    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        # yuv420p is what browsers decode; libx264 would otherwise pick a
        # chroma format Safari refuses
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", path,
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.PIPE)
        for f in frames:
            proc.stdin.write(np.ascontiguousarray(f[:h, :w]).tobytes())
        proc.stdin.close()
        proc.wait(timeout=120)
        return proc.returncode == 0 and os.path.getsize(path) > 1024
    except Exception:                                  # pragma: no cover
        return False


def write_annotated(
    frames: list[np.ndarray],
    fps: float = 6.0,
) -> tuple[str, str] | None:
    """Encode annotated frames to MP4.

    Returns ``(path, codec)``, or ``None`` if every encoder failed. Check
    ``codec in BROWSER_SAFE`` before handing the file to ``st.video`` — serving
    a file the browser cannot decode is worse than showing stills.

    H.264 via bundled ffmpeg is tried first; the OpenCV writers remain as a
    fallback so the app still works if imageio-ffmpeg is unavailable.
    """
    if not frames:
        return None

    path = os.path.join(tempfile.gettempdir(), f"fallguard_{os.getpid()}_h264.mp4")
    if _encode_h264(frames, fps, path):
        return path, "avc1"

    h, w = frames[0].shape[:2]
    w -= w % 2
    h -= h % 2
    for fourcc in ("avc1", "H264", "mp4v"):
        p = os.path.join(tempfile.gettempdir(),
                         f"fallguard_{os.getpid()}_{fourcc}.mp4")
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if not vw.isOpened():
            vw.release()
            continue
        for f in frames:
            vw.write(f[:h, :w])
        vw.release()
        if os.path.exists(p) and os.path.getsize(p) > 1024:
            return p, fourcc
    return None
