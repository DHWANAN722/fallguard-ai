"""
features.py — Geometric feature extraction from a BlazePose skeleton.

The feature vector has three blocks, in this order:

    [0    : 66 )   body-normalised landmark coordinates
                   (re-centred on the pelvis, scaled by torso length, so they
                   describe *posture* independently of where the subject is
                   standing or how far from the camera they are)
    [66   : 99 )   per-landmark visibility, straight from the pose estimator
    [99   : 99+K)  engineered clinical descriptors (torso angle, aspect ratio,
                   pelvis height, knee/hip flexion, leg split, ...)

The engineered block is what makes the model auditable: every one of those
numbers is a quantity a clinician or caregiver can reason about, which is why
the dashboard surfaces four of them directly in the "biomechanical evidence"
panel next to the network's prediction.
"""

from __future__ import annotations

import numpy as np

from .skeleton import (
    L_ANKLE, L_EAR, L_ELBOW, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, N_LANDMARKS,
    NOSE, R_ANKLE, R_EAR, R_ELBOW, R_HIP, R_KNEE, R_SHOULDER, R_WRIST,
)

EPS = 1e-6

ENGINEERED_NAMES = [
    "torso_angle_from_vertical",
    "torso_angle_signed",
    "principal_axis_angle",
    "bbox_aspect_h_over_w",
    "bbox_height",
    "bbox_width",
    "centroid_y",
    "pelvis_y",
    "shoulder_y",
    "nose_y",
    "ankle_y",
    "pelvis_above_ankle",
    "shoulder_above_pelvis",
    "nose_above_pelvis",
    "body_height_span",
    "knee_flex_left",
    "knee_flex_right",
    "knee_flex_mean",
    "hip_flex_left",
    "hip_flex_right",
    "hip_flex_asymmetry",
    "ankle_horizontal_split",
    "leg_verticality",
    "shoulder_w_over_torso",
    "elongation_ratio",
    "mean_visibility",
    "lower_body_visibility",
]
N_ENGINEERED = len(ENGINEERED_NAMES)
N_FEATURES = N_LANDMARKS * 2 + N_LANDMARKS + N_ENGINEERED   # 66 + 33 + 27 = 126

FEATURE_NAMES = (
    [f"{ax}_{i}" for i in range(N_LANDMARKS) for ax in ("x", "y")]
    + [f"vis_{i}" for i in range(N_LANDMARKS)]
    + ENGINEERED_NAMES
)


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at joint ``b`` formed by ``a-b-c``, in degrees."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < EPS or n2 < EPS:
        return 180.0
    cosv = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosv)))


def engineered(P: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Compute the K clinical descriptors for one skeleton."""
    sho = (P[L_SHOULDER] + P[R_SHOULDER]) / 2.0
    pel = (P[L_HIP] + P[R_HIP]) / 2.0
    ank = (P[L_ANKLE] + P[R_ANKLE]) / 2.0

    torso_vec = sho - pel                       # points "up" the body
    torso_len = float(np.linalg.norm(torso_vec)) + EPS

    # angle of the trunk away from vertical: 0° upright, 90° horizontal
    torso_abs = float(np.degrees(np.arctan2(abs(torso_vec[0]), abs(torso_vec[1]))))
    torso_signed = float(np.degrees(np.arctan2(torso_vec[0], -torso_vec[1])))

    # principal axis of the whole point cloud — robust when the trunk is
    # occluded, and the classic descriptor in the fall-detection literature
    C = P - P.mean(axis=0)
    try:
        _, S, Vt = np.linalg.svd(C, full_matrices=False)
        axis = Vt[0]
        principal = float(np.degrees(np.arctan2(abs(axis[0]), abs(axis[1]))))
        elong = float(S[0] / (S[1] + EPS))
    except np.linalg.LinAlgError:          # pragma: no cover - degenerate input
        principal, elong = torso_abs, 1.0

    x0, y0 = P[:, 0].min(), P[:, 1].min()
    x1, y1 = P[:, 0].max(), P[:, 1].max()
    bw, bh = float(x1 - x0), float(y1 - y0)
    aspect = bh / (bw + EPS)

    knee_l = _angle(P[L_HIP], P[L_KNEE], P[L_ANKLE])
    knee_r = _angle(P[R_HIP], P[R_KNEE], P[R_ANKLE])
    hip_l = _angle(P[L_SHOULDER], P[L_HIP], P[L_KNEE])
    hip_r = _angle(P[R_SHOULDER], P[R_HIP], P[R_KNEE])

    # how far apart the feet are horizontally, normalised by trunk length —
    # the primary walking-vs-standing cue
    split = float(abs(P[L_ANKLE, 0] - P[R_ANKLE, 0]) / torso_len)

    # are the legs still stacked under the pelvis? separates bending from falling
    leg_vec = ank - pel
    leg_vertical = float(abs(leg_vec[1]) / (np.linalg.norm(leg_vec) + EPS))

    sho_w = float(np.linalg.norm(P[L_SHOULDER] - P[R_SHOULDER]))

    lower = [L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_HIP, R_HIP]

    return np.array([
        torso_abs / 90.0,
        torso_signed / 180.0,
        principal / 90.0,
        np.clip(aspect, 0, 8) / 8.0,
        bh, bw,
        float(P[:, 1].mean()),
        float(pel[1]), float(sho[1]), float(P[NOSE, 1]), float(ank[1]),
        float(ank[1] - pel[1]),                 # +ve when the pelvis is above the feet
        float(pel[1] - sho[1]),
        float(pel[1] - P[NOSE, 1]),
        bh,
        knee_l / 180.0, knee_r / 180.0, (knee_l + knee_r) / 360.0,
        hip_l / 180.0, hip_r / 180.0, abs(hip_l - hip_r) / 180.0,
        np.clip(split, 0, 4) / 4.0,
        leg_vertical,
        float(sho_w / torso_len),
        float(np.clip(elong, 0, 12) / 12.0),
        float(vis.mean()),
        float(vis[lower].mean()),
    ], dtype=np.float32)


def extract(P: np.ndarray, vis: np.ndarray) -> np.ndarray:
    """Full ``(N_FEATURES,)`` feature vector for one skeleton."""
    pel = (P[L_HIP] + P[R_HIP]) / 2.0
    sho = (P[L_SHOULDER] + P[R_SHOULDER]) / 2.0
    scale = float(np.linalg.norm(sho - pel)) + EPS

    norm = ((P - pel) / scale).astype(np.float32).reshape(-1)
    return np.concatenate([norm, vis.astype(np.float32), engineered(P, vis)])


def extract_batch(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """``(N,33,2)`` + ``(N,33)`` → ``(N, N_FEATURES)``."""
    out = np.zeros((len(P), N_FEATURES), dtype=np.float32)
    for i in range(len(P)):
        out[i] = extract(P[i], V[i])
    return out


# --------------------------------------------------------------------------
# interpretable descriptors surfaced in the dashboard
# --------------------------------------------------------------------------
def clinical_summary(P: np.ndarray, vis: np.ndarray) -> dict:
    """The four numbers the caregiver-facing "biomechanical evidence" panel shows."""
    e = engineered(P, vis)
    names = {n: i for i, n in enumerate(ENGINEERED_NAMES)}
    return {
        "torso_angle": float(e[names["torso_angle_from_vertical"]] * 90.0),
        "aspect_ratio": float(e[names["bbox_aspect_h_over_w"]] * 8.0),
        "pelvis_height": float(1.0 - e[names["pelvis_y"]]),   # 1.0 = top of frame
        "leg_verticality": float(e[names["leg_verticality"]]),
        "mean_visibility": float(e[names["mean_visibility"]]),
    }


def biomechanical_fall_score(P: np.ndarray, vis: np.ndarray) -> float:
    """A transparent, rule-based fall likelihood in [0, 1].

    This deliberately does **not** use the neural network. It is the second
    opinion in the dashboard's two-tier alert logic: a fall is only escalated
    to EMERGENCY when the CNN and this independent biomechanical rule agree.
    Requiring agreement is what suppresses the bending-over false alarm, since
    bending scores high on trunk angle but low on pelvis descent.

    The four terms are the descriptors used throughout the fall-detection
    literature: trunk inclination, bounding-box aspect ratio, pelvis height in
    frame, and loss of leg verticality.
    """
    c = clinical_summary(P, vis)
    trunk = np.clip((c["torso_angle"] - 45.0) / 35.0, 0, 1)          # >45° tilting
    aspect = np.clip((1.8 - c["aspect_ratio"]) / 1.0, 0, 1)          # <1.8 lying down
    low = np.clip((0.38 - c["pelvis_height"]) / 0.22, 0, 1)          # pelvis near floor
    legs = np.clip((0.75 - c["leg_verticality"]) / 0.45, 0, 1)       # legs not under body

    score = 0.30 * trunk + 0.22 * aspect + 0.33 * low + 0.15 * legs
    return float(np.clip(score, 0.0, 1.0))
