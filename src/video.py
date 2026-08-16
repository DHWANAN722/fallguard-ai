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


def write_annotated(
    frames: list[np.ndarray],
    fps: float = 6.0,
) -> tuple[str, str] | None:
    """Encode annotated frames to MP4.

    Returns ``(path, fourcc)``, or ``None`` if no encoder worked at all. Check
    ``fourcc in BROWSER_SAFE`` before handing the file to ``st.video``; OpenCV
    wheels on Streamlit Cloud do not always ship an H.264 encoder, and silently
    serving an unplayable file is worse than showing stills.
    """
    if not frames:
        return None

    h, w = frames[0].shape[:2]
    # even dimensions are required by most H.264 encoders
    w -= w % 2
    h -= h % 2

    for fourcc in ("avc1", "H264", "mp4v"):
        path = os.path.join(tempfile.gettempdir(),
                            f"fallguard_{os.getpid()}_{fourcc}.mp4")
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
        if not vw.isOpened():
            vw.release()
            continue
        for f in frames:
            vw.write(f[:h, :w])
        vw.release()
        if os.path.exists(path) and os.path.getsize(path) > 1024:
            return path, fourcc
    return None
