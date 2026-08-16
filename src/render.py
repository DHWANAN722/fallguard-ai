"""
render.py — Turn a 33-landmark skeleton into a fixed-size tensor for the CNN,
and into a neon overlay for the dashboard.

Two distinct renderers live here:

``render_cnn``      64x64x3 float tensor, the CNN's input. Channels are split
                    by anatomical group so the convolutional filters can learn
                    limb-configuration patterns rather than a tangle of lines:
                        ch0 = torso + head      ch1 = arms      ch2 = legs
                    Crucially the skeleton is rendered **in frame coordinates,
                    not re-centred**, so vertical position in the room — the
                    single strongest fall cue — survives into the tensor.

``draw_overlay``    Cyberpunk neon skeleton drawn onto a real BGR video frame
                    for the Streamlit dashboard.
"""

from __future__ import annotations

import cv2
import numpy as np

from .skeleton import BONES, N_LANDMARKS

IMG_SIZE = 64

#: BGR neon palette, indexed by bone group (torso, arms, legs, head).
NEON_BGR = {
    0: (255, 247, 0),      # cyan      torso
    1: (255, 0, 200),      # magenta   arms
    2: (140, 255, 0),      # spring    legs
    3: (0, 210, 255),      # amber     head
}


def render_cnn(
    P: np.ndarray,
    vis: np.ndarray | None = None,
    size: int = IMG_SIZE,
    vis_thresh: float = 0.35,
) -> np.ndarray:
    """Rasterise one skeleton to a ``(size, size, 3)`` float32 tensor in [0, 1].

    Landmarks below ``vis_thresh`` visibility are omitted, so occlusion is
    represented as genuinely missing structure rather than as a wrong bone —
    this is what lets the CNN stay calibrated when furniture hides the legs.
    """
    if vis is None:
        vis = np.ones(N_LANDMARKS, dtype=np.float32)

    # one contiguous buffer per channel — cv2 cannot draw into a strided
    # `img[:, :, c]` view, and stacking at the end is cheap
    chans = [np.zeros((size, size), dtype=np.uint8) for _ in range(3)]

    # frame coords → pixel coords, keeping absolute position in the room
    pt = (np.clip(P, 0.0, 1.0) * (size - 1)).astype(np.int32)
    xy = [(int(p[0]), int(p[1])) for p in pt]

    # 2 px strokes, not 1 px. The network's stem convolution is stride-2, and a
    # 1 px antialiased line can be annihilated by that downsample; 2 px
    # guarantees every bone survives into the first feature map.
    for a, b, group in BONES:
        if vis[a] < vis_thresh or vis[b] < vis_thresh:
            continue
        ch = {0: 0, 3: 0, 1: 1, 2: 2}[group]     # head folds into the torso channel
        cv2.line(chans[ch], xy[a], xy[b], 255, 2, cv2.LINE_AA)

    # joints get a dot in every channel so the network can localise
    # articulation points independently of which limb they belong to
    for i in range(N_LANDMARKS):
        if vis[i] < vis_thresh:
            continue
        for c in range(3):
            cv2.circle(chans[c], xy[i], 1, 90, -1)

    return np.stack(chans, axis=-1).astype(np.float32) / 255.0


def render_batch(P: np.ndarray, V: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Vectorised wrapper: ``(N,33,2)`` + ``(N,33)`` → ``(N,size,size,3)``."""
    out = np.zeros((len(P), size, size, 3), dtype=np.float32)
    for i in range(len(P)):
        out[i] = render_cnn(P[i], V[i], size)
    return out


def draw_overlay(
    frame: np.ndarray,
    P: np.ndarray,
    vis: np.ndarray | None = None,
    vis_thresh: float = 0.35,
    accent: tuple[int, int, int] | None = None,
    glow: bool = True,
) -> np.ndarray:
    """Draw a glowing neon skeleton onto a BGR frame.

    The glow is a genuine two-pass bloom: a thick, heavily blurred stroke is
    screen-blended underneath a crisp core stroke, which reads as emissive
    rather than merely thick.
    """
    out = frame.copy()
    h, w = out.shape[:2]
    if vis is None:
        vis = np.ones(N_LANDMARKS, dtype=np.float32)

    pt = (P * np.array([w, h])).astype(np.int32)
    xy = [(int(p[0]), int(p[1])) for p in pt]
    thick = max(1, int(round(min(h, w) / 320)))

    if glow:
        halo = np.zeros_like(out)
        for a, b, group in BONES:
            if vis[a] < vis_thresh or vis[b] < vis_thresh:
                continue
            col = accent or NEON_BGR[group]
            cv2.line(halo, xy[a], xy[b], col, thick * 6, cv2.LINE_AA)
        halo = cv2.GaussianBlur(halo, (0, 0), sigmaX=thick * 5)
        # screen blend: 1-(1-a)(1-b) — brightens without clipping to flat white
        a = out.astype(np.float32) / 255.0
        b = halo.astype(np.float32) / 255.0
        out = ((1.0 - (1.0 - a) * (1.0 - b * 0.85)) * 255).astype(np.uint8)

    for a_i, b_i, group in BONES:
        if vis[a_i] < vis_thresh or vis[b_i] < vis_thresh:
            continue
        col = accent or NEON_BGR[group]
        cv2.line(out, xy[a_i], xy[b_i], col, thick * 2, cv2.LINE_AA)

    for i in range(N_LANDMARKS):
        if vis[i] < vis_thresh:
            continue
        cv2.circle(out, xy[i], thick * 2 + 1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, xy[i], thick * 2 + 1, (30, 30, 30), 1, cv2.LINE_AA)

    return out
