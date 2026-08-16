"""
pose.py — MediaPipe Pose (BlazePose) wrapper.

Isolated behind a tiny interface so the rest of the system depends on
``(landmarks, visibility)`` arrays rather than on MediaPipe itself. That is
what lets the identical feature/render/classify path serve both the procedural
training corpus and real camera frames, and it means swapping in YOLOv8-Pose or
OpenPose later touches only this file.

Version note
------------
MediaPipe 0.10.x is pinned deliberately. The 1.x line removed the
``mp.solutions.pose`` API in favour of Tasks, which fetches its model bundle
from the network at first call — an avoidable cold-start failure mode on
Streamlit Cloud. The 0.10 wheels ship the BlazePose weights inside the package,
so the dashboard has no runtime download at all.
"""

from __future__ import annotations

import numpy as np

_POSE = None


def _mp():
    import mediapipe as mp
    return mp


def get_pose(static: bool = True, complexity: int = 1, min_conf: float = 0.5):
    """Return a cached MediaPipe ``Pose`` instance.

    MediaPipe graphs are expensive to construct (~1 s) and are not thread-safe,
    so exactly one is built per process and reused. Streamlit reruns the script
    on every interaction, which would otherwise rebuild it constantly.
    """
    global _POSE
    if _POSE is None:
        mp = _mp()
        _POSE = mp.solutions.pose.Pose(
            static_image_mode=static,
            model_complexity=complexity,
            enable_segmentation=False,
            min_detection_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
    return _POSE


def aspect_correct(P: np.ndarray, width: int, height: int) -> np.ndarray:
    """Rescale x so that one unit of x equals one unit of y.

    THIS IS NOT COSMETIC. MediaPipe normalises x against the image *width* and
    y against the image *height*, independently. On a square image that is
    harmless, but on a 9:16 phone photo the x axis is stretched by 1/0.45 ≈ 2.2
    relative to y — so every angle, every bounding-box ratio and every limb
    direction computed from raw landmarks is wrong.

    Concretely, a standing subject in a portrait photo measured a
    height/width ratio of 1.38 where the training corpus (built in a square
    frame) puts standing at 3.01 ± 1.3. The subject was landing far outside the
    training distribution purely because of the photo's shape, and was being
    classified as Walking.

    The x axis is scaled about the frame centre, which keeps the *vertical*
    coordinate untouched — important, because pelvis height is interpreted as a
    genuine "how high in the room" and must stay a fraction of frame height.
    """
    P = P.copy()
    s = float(width) / float(height)
    P[:, 0] = (P[:, 0] - 0.5) * s + 0.5
    return P


def aspect_uncorrect(P: np.ndarray, width: int, height: int) -> np.ndarray:
    """Inverse of ``aspect_correct`` — back to MediaPipe's raw normalisation.

    Needed whenever landmarks are drawn *onto the original photograph*: the
    overlay has to line up with the pixels, and the corrected coordinates
    deliberately do not. Analysis uses corrected coordinates; rendering on a
    real frame uses these.
    """
    P = P.copy()
    s = float(width) / float(height)
    P[:, 0] = (P[:, 0] - 0.5) / s + 0.5
    return P


def estimate(frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Run pose estimation on one BGR frame.

    Returns
    -------
    ``(landmarks (33,2) float32, visibility (33,) float32)`` with the landmarks
    already **aspect-corrected** (see ``aspect_correct``), or ``None`` when no
    person is detected.
    """
    import cv2

    pose = get_pose()
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res = pose.process(rgb)

    if not res.pose_landmarks:
        return None

    h, w = frame_bgr.shape[:2]
    lm = res.pose_landmarks.landmark
    P = np.array([[p.x, p.y] for p in lm], dtype=np.float32)
    V = np.array([p.visibility for p in lm], dtype=np.float32)
    return aspect_correct(P, w, h), V


def available() -> tuple[bool, str]:
    """Probe MediaPipe once so the dashboard can fail loudly but gracefully."""
    try:
        import mediapipe as mp
        if not hasattr(mp, "solutions"):
            return False, (
                f"mediapipe {mp.__version__} has no `solutions` API — "
                "pin mediapipe==0.10.18 (see requirements.txt)"
            )
        get_pose()
        return True, mp.__version__
    except Exception as exc:                       # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}"
