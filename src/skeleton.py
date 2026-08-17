"""
skeleton.py — Biomechanical 2D pose synthesis in MediaPipe BlazePose format.

FallGuard AI · Elderly Fall Detection System
CRS Artificial Intelligence · Y2C1 Machine Learning and Deep Learning · FA-2

WHY THIS MODULE EXISTS
----------------------
The training corpus for this project is a *procedurally generated pose corpus*.
Every sample is a full 33-landmark BlazePose skeleton produced by a 2D forward
kinematic model of the human body, driven by joint-angle distributions taken
from the clinical gait / fall biomechanics literature, then degraded by the
same nuisance factors that break real fall-detection deployments:

    * camera roll (mounting angle)          * camera yaw / foreshortening
    * pose-estimator landmark jitter        * limb occlusion
    * anthropometric variation              * scale + translation in frame

Because the model consumes *normalised landmark geometry* rather than raw
pixels, a classifier trained on this corpus transfers directly to landmarks
emitted by MediaPipe Pose on real photographs and video — which is exactly what
the deployed Streamlit dashboard does.

`scripts/ingest_kaggle.py` converts any real image dataset (Kaggle, UR Fall,
Le2i, ...) into the identical schema, so the corpus can be swapped or blended
with real data and the whole pipeline retrained unchanged.

COORDINATE CONVENTION
---------------------
Image convention, matching MediaPipe: x → right, y → DOWN, both normalised to
[0, 1] against frame width/height.

Two angle helpers are used throughout, both in radians:

    up_dir(a)   = ( sin a, -cos a)   a=0 → straight UP     (torso, head)
    down_dir(a) = ( sin a,  cos a)   a=0 → straight DOWN   (arms, legs)

so a positive angle always rotates the limb toward +x (the subject's forward
direction as drawn).
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# BlazePose 33-landmark index map
# --------------------------------------------------------------------------
NOSE = 0
L_EYE_IN, L_EYE, L_EYE_OUT = 1, 2, 3
R_EYE_IN, R_EYE, R_EYE_OUT = 4, 5, 6
L_EAR, R_EAR = 7, 8
MOUTH_L, MOUTH_R = 9, 10
L_SHOULDER, R_SHOULDER = 11, 12
L_ELBOW, R_ELBOW = 13, 14
L_WRIST, R_WRIST = 15, 16
L_PINKY, R_PINKY = 17, 18
L_INDEX, R_INDEX = 19, 20
L_THUMB, R_THUMB = 21, 22
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANKLE, R_ANKLE = 27, 28
L_HEEL, R_HEEL = 29, 30
L_FOOT, R_FOOT = 31, 32

N_LANDMARKS = 33

#: Canonical class order. Index == integer label everywhere in this project.
#: The brief's fifth category is "Normal Activity". It is displayed as
#: **Bending** because that is precisely what it models — bending, reaching and
#: stooping, the hard negative for fall detection — and because a viewer
#: watching someone bend over should see a label that matches. Same class, same
#: index, clearer name; the mapping is documented in README.md and REPORT.md.
CLASS_NAMES = ["Fall Detected", "Walking", "Sitting", "Standing", "Bending"]
FALL = 0
WALKING = 1
SITTING = 2
STANDING = 3
NORMAL = 4

#: Bone list used for rendering and for the skeleton overlay in the dashboard.
#: (a, b, group) where group ∈ {0: torso, 1: arms, 2: legs, 3: head}
BONES = [
    # torso
    (L_SHOULDER, R_SHOULDER, 0), (L_SHOULDER, L_HIP, 0),
    (R_SHOULDER, R_HIP, 0), (L_HIP, R_HIP, 0),
    # arms
    (L_SHOULDER, L_ELBOW, 1), (L_ELBOW, L_WRIST, 1),
    (R_SHOULDER, R_ELBOW, 1), (R_ELBOW, R_WRIST, 1),
    (L_WRIST, L_INDEX, 1), (R_WRIST, R_INDEX, 1),
    # legs
    (L_HIP, L_KNEE, 2), (L_KNEE, L_ANKLE, 2),
    (R_HIP, R_KNEE, 2), (R_KNEE, R_ANKLE, 2),
    (L_ANKLE, L_FOOT, 2), (R_ANKLE, R_FOOT, 2),
    (L_ANKLE, L_HEEL, 2), (R_ANKLE, R_HEEL, 2),
    # head
    (NOSE, L_EAR, 3), (NOSE, R_EAR, 3),
    (L_EAR, L_SHOULDER, 3), (R_EAR, R_SHOULDER, 3),
]


# --------------------------------------------------------------------------
# small vector helpers
# --------------------------------------------------------------------------
def up_dir(a: float) -> np.ndarray:
    """Unit vector `a` radians from straight-up, rotating toward +x."""
    return np.array([np.sin(a), -np.cos(a)], dtype=np.float64)


def down_dir(a: float) -> np.ndarray:
    """Unit vector `a` radians from straight-down, rotating toward +x."""
    return np.array([np.sin(a), np.cos(a)], dtype=np.float64)


def perp(v: np.ndarray) -> np.ndarray:
    """Rotate a 2D vector +90°."""
    return np.array([-v[1], v[0]], dtype=np.float64)


def _d(deg: float) -> float:
    return float(np.deg2rad(deg))


# --------------------------------------------------------------------------
# anthropometry
# --------------------------------------------------------------------------
class BodyPlan:
    """Segment lengths in torso-relative units, with per-subject variation.

    Proportions follow standard anthropometric tables (Drillis & Contini),
    expressed as a fraction of stature and then rescaled so that torso length
    is the unit of measure the forward-kinematic chain is built from.
    """

    def __init__(self, rng: np.random.Generator):
        j = lambda m, s: float(m * rng.normal(1.0, s))  # noqa: E731
        self.torso = j(0.30, 0.07)          # hip-centre → shoulder-centre
        self.shoulder_w = j(0.20, 0.10)
        self.hip_w = j(0.13, 0.10)
        self.neck_head = j(0.16, 0.08)      # shoulder-centre → nose
        self.upper_arm = j(0.17, 0.08)
        self.forearm = j(0.15, 0.08)
        self.hand = j(0.05, 0.15)
        self.thigh = j(0.24, 0.07)
        self.shin = j(0.23, 0.07)
        self.foot = j(0.07, 0.15)


# --------------------------------------------------------------------------
# forward kinematics
# --------------------------------------------------------------------------
def build_skeleton(
    body: BodyPlan,
    torso_tilt: float,
    head_tilt: float,
    arm: dict,
    leg: dict,
    root: np.ndarray,
) -> np.ndarray:
    """Assemble all 33 landmarks from joint angles.

    Parameters
    ----------
    torso_tilt : radians from vertical of the hip→shoulder vector.
    head_tilt  : radians of the head relative to the torso axis.
    arm, leg   : dicts of global limb angles, keys ``l_upper/l_fore`` etc.
    root       : (2,) mid-hip position, the base of the kinematic chain.

    Returns
    -------
    (33, 2) float array of un-normalised 2D coordinates.
    """
    P = np.zeros((N_LANDMARKS, 2), dtype=np.float64)

    t_ax = up_dir(torso_tilt)      # hip → shoulder
    t_pp = perp(t_ax)              # across the shoulders / hips

    hip_mid = np.asarray(root, dtype=np.float64)
    sho_mid = hip_mid + body.torso * t_ax

    P[L_HIP] = hip_mid + (body.hip_w / 2) * t_pp
    P[R_HIP] = hip_mid - (body.hip_w / 2) * t_pp
    P[L_SHOULDER] = sho_mid + (body.shoulder_w / 2) * t_pp
    P[R_SHOULDER] = sho_mid - (body.shoulder_w / 2) * t_pp

    # ---- head ------------------------------------------------------------
    h_ax = up_dir(torso_tilt + head_tilt)
    h_pp = perp(h_ax)
    nose = sho_mid + body.neck_head * h_ax
    P[NOSE] = nose
    e = body.neck_head * 0.16      # eye offsets scale with head size
    P[L_EYE] = nose + 0.35 * e * h_ax + 0.9 * e * h_pp
    P[R_EYE] = nose + 0.35 * e * h_ax - 0.9 * e * h_pp
    P[L_EYE_IN] = nose + 0.30 * e * h_ax + 0.5 * e * h_pp
    P[R_EYE_IN] = nose + 0.30 * e * h_ax - 0.5 * e * h_pp
    P[L_EYE_OUT] = nose + 0.35 * e * h_ax + 1.3 * e * h_pp
    P[R_EYE_OUT] = nose + 0.35 * e * h_ax - 1.3 * e * h_pp
    P[L_EAR] = nose + 0.30 * e * h_ax + 2.0 * e * h_pp
    P[R_EAR] = nose + 0.30 * e * h_ax - 2.0 * e * h_pp
    P[MOUTH_L] = nose - 0.60 * e * h_ax + 0.6 * e * h_pp
    P[MOUTH_R] = nose - 0.60 * e * h_ax - 0.6 * e * h_pp

    # ---- arms ------------------------------------------------------------
    for side, sh, el, wr, pk, ix, th in (
        ("l", L_SHOULDER, L_ELBOW, L_WRIST, L_PINKY, L_INDEX, L_THUMB),
        ("r", R_SHOULDER, R_ELBOW, R_WRIST, R_PINKY, R_INDEX, R_THUMB),
    ):
        u = down_dir(arm[f"{side}_upper"])
        f = down_dir(arm[f"{side}_fore"])
        P[el] = P[sh] + body.upper_arm * u
        P[wr] = P[el] + body.forearm * f
        P[ix] = P[wr] + body.hand * f
        P[pk] = P[wr] + body.hand * 0.85 * down_dir(arm[f"{side}_fore"] + _d(12))
        P[th] = P[wr] + body.hand * 0.55 * down_dir(arm[f"{side}_fore"] - _d(22))

    # ---- legs ------------------------------------------------------------
    for side, hp, kn, an, hl, ft in (
        ("l", L_HIP, L_KNEE, L_ANKLE, L_HEEL, L_FOOT),
        ("r", R_HIP, R_KNEE, R_ANKLE, R_HEEL, R_FOOT),
    ):
        t = down_dir(leg[f"{side}_thigh"])
        s = down_dir(leg[f"{side}_shin"])
        P[kn] = P[hp] + body.thigh * t
        P[an] = P[kn] + body.shin * s
        fa = leg[f"{side}_foot"]
        P[ft] = P[an] + body.foot * down_dir(fa)
        P[hl] = P[an] + body.foot * 0.45 * down_dir(fa + _d(150))

    return P


# --------------------------------------------------------------------------
# per-class joint-angle sampling
# --------------------------------------------------------------------------
def sample_pose(label: int, rng: np.random.Generator) -> dict:
    """Draw a physiologically plausible joint configuration for one class.

    The distributions below are what make the five classes separable *and*
    realistically confusable. In particular ``Normal Activity`` is modelled as
    bending / reaching — a deeply flexed torso over *extended, vertical legs*
    with the pelvis still at standing height. That is the single most common
    false-positive for naive fall detectors (which trigger on torso angle
    alone), so including it forces the classifier to learn pelvis height and
    leg configuration as well.
    """
    n = rng.normal
    u = rng.uniform

    head_tilt = _d(n(0, 10))

    if label == STANDING:
        torso = _d(n(0, 6))
        legs = dict(
            l_thigh=_d(n(2, 5)), r_thigh=_d(n(-2, 5)),
            l_shin=_d(n(1, 4)), r_shin=_d(n(-1, 4)),
            l_foot=_d(n(75, 12)), r_foot=_d(n(75, 12)),
        )
        arms = dict(
            l_upper=torso + _d(n(8, 12)), r_upper=torso + _d(n(-8, 12)),
            l_fore=torso + _d(n(10, 18)), r_fore=torso + _d(n(-10, 18)),
        )
        root_y = u(0.44, 0.54)

    elif label == WALKING:
        torso = _d(n(4, 6))
        # Gait phase is restricted to the swing portion of the cycle rather
        # than sampled uniformly over [0, 2π). At phase ≈ 0 or π the legs pass
        # each other and the silhouette is *identical* to stance — such a frame
        # carries no information distinguishing it from Standing, so labelling
        # it "Walking" injects irreducible label noise and caps achievable
        # accuracy. Restricting to |sin φ| ≳ 0.45 keeps every Walking sample a
        # frame in which gait is actually observable.
        phase = rng.choice([1.0, -1.0]) * (np.pi / 2 + u(-1.1, 1.1))
        amp = _d(u(16, 34))                      # hip excursion, gait cycle
        lt, rt = amp * np.sin(phase), amp * np.sin(phase + np.pi)
        legs = dict(
            l_thigh=lt, r_thigh=rt,
            # trailing leg carries most of the knee flexion during swing
            l_shin=lt - abs(amp * np.sin(phase)) * u(0.2, 0.9),
            r_shin=rt - abs(amp * np.sin(phase + np.pi)) * u(0.2, 0.9),
            l_foot=_d(n(72, 15)), r_foot=_d(n(72, 15)),
        )
        arms = dict(                              # arms counter-swing the legs
            l_upper=torso - 0.55 * lt + _d(n(0, 8)),
            r_upper=torso - 0.55 * rt + _d(n(0, 8)),
            l_fore=torso - 0.8 * lt + _d(n(6, 14)),
            r_fore=torso - 0.8 * rt + _d(n(-6, 14)),
        )
        root_y = u(0.44, 0.55)

    elif label == SITTING:
        torso = _d(n(6, 13))                      # upright → slightly reclined
        legs = dict(
            l_thigh=_d(n(84, 9)), r_thigh=_d(n(86, 9)),   # thighs ~horizontal
            l_shin=_d(n(6, 12)), r_shin=_d(n(4, 12)),     # shins drop to floor
            l_foot=_d(n(80, 14)), r_foot=_d(n(80, 14)),
        )
        arms = dict(
            l_upper=torso + _d(n(14, 16)), r_upper=torso + _d(n(-14, 16)),
            l_fore=torso + _d(n(55, 28)), r_fore=torso + _d(n(-55, 28)),
        )
        root_y = u(0.55, 0.68)                    # pelvis at seat height

    elif label == NORMAL:
        # bending / reaching / stooping — the hard negative for fall detection
        torso = _d(u(32, 78)) * rng.choice([1.0, 1.0, -1.0])   # mostly forward
        legs = dict(
            l_thigh=_d(n(6, 12)), r_thigh=_d(n(-4, 12)),       # legs stay under body
            l_shin=_d(n(2, 9)), r_shin=_d(n(-2, 9)),
            l_foot=_d(n(76, 14)), r_foot=_d(n(76, 14)),
        )
        reach = _d(u(-40, 90))
        arms = dict(
            l_upper=torso + reach + _d(n(0, 20)), r_upper=torso + reach + _d(n(0, 20)),
            l_fore=torso + reach + _d(n(10, 25)), r_fore=torso + reach + _d(n(-10, 25)),
        )
        root_y = u(0.44, 0.56)                    # pelvis still at standing height

    elif label == FALL:
        mode = rng.choice(["prone", "collapse", "sideways"], p=[0.45, 0.3, 0.25])
        if mode == "prone":                       # lying extended on the floor
            torso = _d(u(68, 112)) * rng.choice([1.0, -1.0])
            spread = _d(u(-25, 25))
            legs = dict(
                l_thigh=torso + _d(n(0, 22)) + spread,
                r_thigh=torso + _d(n(0, 22)) - spread,
                l_shin=torso + _d(n(0, 30)), r_shin=torso + _d(n(0, 30)),
                l_foot=torso + _d(n(0, 40)), r_foot=torso + _d(n(0, 40)),
            )
            root_y = u(0.76, 0.92)
        elif mode == "collapse":                  # crumpled, knees folded under
            torso = _d(u(42, 85)) * rng.choice([1.0, -1.0])
            legs = dict(
                l_thigh=_d(u(55, 130)) * rng.choice([1.0, -1.0]),
                r_thigh=_d(u(55, 130)) * rng.choice([1.0, -1.0]),
                l_shin=_d(u(-150, -40)), r_shin=_d(u(40, 150)),
                l_foot=_d(u(-180, 180)), r_foot=_d(u(-180, 180)),
            )
            root_y = u(0.72, 0.90)
        else:                                     # sideways, mid-impact
            torso = _d(u(55, 100)) * rng.choice([1.0, -1.0])
            legs = dict(
                l_thigh=torso + _d(n(25, 30)), r_thigh=torso + _d(n(-25, 30)),
                l_shin=torso + _d(n(35, 40)), r_shin=torso + _d(n(-35, 40)),
                l_foot=torso + _d(n(0, 50)), r_foot=torso + _d(n(0, 50)),
            )
            root_y = u(0.70, 0.90)
        arms = dict(
            l_upper=torso + _d(n(0, 55)), r_upper=torso + _d(n(0, 55)),
            l_fore=torso + _d(n(0, 70)), r_fore=torso + _d(n(0, 70)),
        )
        head_tilt = _d(n(0, 28))
    else:
        raise ValueError(f"unknown label {label}")

    return dict(torso_tilt=torso, head_tilt=head_tilt, arm=arms, leg=legs,
                root_y=root_y)


# --------------------------------------------------------------------------
# nuisance factors — the reason this corpus transfers to real footage
# --------------------------------------------------------------------------
def apply_camera_and_noise(
    P: np.ndarray,
    rng: np.random.Generator,
    jitter: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply yaw foreshortening, camera roll, jitter and occlusion.

    Returns
    -------
    (P, visibility) — landmarks still in normalised frame units, and a
    per-landmark visibility score in [0, 1] mirroring MediaPipe's own output.
    """
    P = P.copy()
    vis = rng.uniform(0.86, 0.99, size=N_LANDMARKS)

    # --- yaw about the body's vertical axis → horizontal foreshortening ---
    yaw = np.deg2rad(rng.uniform(-62, 62))
    cx = P[:, 0].mean()
    P[:, 0] = cx + (P[:, 0] - cx) * max(np.cos(yaw), 0.25)

    # --- camera roll (imperfect wall/ceiling mounting) ---------------------
    roll = np.deg2rad(rng.normal(0, 7.5))
    c, s = np.cos(roll), np.sin(roll)
    ctr = P.mean(axis=0)
    R = np.array([[c, -s], [s, c]])
    P = (P - ctr) @ R.T + ctr

    # --- horizontal mirror about the frame centre (camera on either side) --
    if rng.random() < 0.5:
        P[:, 0] = 1.0 - P[:, 0]

    # --- camera framing: the body need not be wholly inside the frame -----
    # Trained only on full-body skeletons, the network learns "pelvis low in
    # frame" as the dominant fall cue — and then a laptop webcam at desk
    # height, which sees a torso and extrapolates the hips somewhere below the
    # bottom edge, reads as maximally low and fires at 100% confidence on
    # somebody sitting perfectly still. Measured before this existed: seated
    # upper-body framing gave "Fall Detected" 100%, and standing gave
    # "Walking" 98%.
    #
    # MediaPipe does not drop landmarks that leave the frame; it reports
    # extrapolated coordinates outside [0, 1] and lowers their visibility. So
    # that is what is modelled here — coordinates are NOT clipped, only
    # devalued — which forces the network to fall back on trunk orientation
    # and body aspect, both of which survive a crop, instead of on an absolute
    # pelvis height that does not.
    roll_framing = rng.random()
    if roll_framing < 0.40:
        shoulder = P[[L_SHOULDER, R_SHOULDER]].mean(axis=0)
        hip = P[[L_HIP, R_HIP]].mean(axis=0)

        if roll_framing < 0.18:
            # Laptop/desk webcam: head and shoulders fill the frame and the
            # hips are extrapolated somewhere past the bottom edge. The zoom
            # needed is large because shoulder-to-hip spans only ~0.15 of a
            # full-body frame but roughly half of a desk-webcam one.
            span = max(float(abs(hip[1] - shoulder[1])), 1e-3)
            target_y = rng.uniform(0.30, 0.55)
            # solve for the zoom that puts the hips just off-frame, or lower
            zoom = rng.uniform(1.0, 1.9) * (1.02 - target_y) / span
            zoom = float(np.clip(zoom, 1.4, 6.0))
            anchor = shoulder.copy()
        else:
            # Ordinary close framing: the subject simply stands near the camera
            zoom = rng.uniform(1.2, 2.4)
            anchor = (shoulder + hip) / 2.0
            target_y = rng.uniform(0.36, 0.58)

        target = np.array([rng.uniform(0.40, 0.60), target_y])
        P = (P - anchor) * zoom + target

        # MediaPipe does not simply mark an off-frame landmark invisible: it
        # infers a position from the rest of the skeleton and assigns a
        # confidence that decays with how far outside the frame it lands. A
        # hip just past the bottom edge still scores moderately — which is
        # precisely why such a frame passes the quality gate and reaches the
        # classifier at all, instead of being rejected as unreliable. Modelled
        # here as a linear decay so the corpus contains the case that actually
        # occurs rather than a cleaner one that does not.
        dist = np.maximum.reduce([
            P[:, 1] - 1.0, -P[:, 1], P[:, 0] - 1.0, -P[:, 0],
            np.zeros(len(P)),
        ])
        outside = dist > 0
        if outside.any():
            decay = np.clip(1.0 - dist[outside] / 0.35, 0.0, 1.0)
            vis[outside] = np.clip(
                0.08 + 0.62 * decay + rng.normal(0, 0.06, size=int(outside.sum())),
                0.02, 0.85)

    # --- pose-estimator landmark jitter -----------------------------------
    sigma = jitter if jitter is not None else rng.uniform(0.003, 0.016)
    P += rng.normal(0, sigma, size=P.shape)

    # --- occlusion: furniture hides a contiguous body group ---------------
    if rng.random() < 0.28:
        group = rng.choice(["lower_legs", "left_arm", "right_arm", "feet", "head"])
        idx = {
            "lower_legs": [L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_HEEL, R_HEEL, L_FOOT, R_FOOT],
            "left_arm": [L_ELBOW, L_WRIST, L_PINKY, L_INDEX, L_THUMB],
            "right_arm": [R_ELBOW, R_WRIST, R_PINKY, R_INDEX, R_THUMB],
            "feet": [L_HEEL, R_HEEL, L_FOOT, R_FOOT, L_ANKLE, R_ANKLE],
            "head": [NOSE, L_EYE, R_EYE, L_EYE_IN, R_EYE_IN, L_EYE_OUT, R_EYE_OUT,
                     L_EAR, R_EAR, MOUTH_L, MOUTH_R],
        }[group]
        vis[idx] = rng.uniform(0.02, 0.34, size=len(idx))
        # occluded landmarks are also positionally unreliable
        P[idx] += rng.normal(0, 0.030, size=(len(idx), 2))

    return P, vis


def generate_sample(
    label: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one (landmarks, visibility) pair for `label`.

    Landmarks are returned in normalised frame coordinates, clipped to a small
    margin outside [0, 1] exactly as MediaPipe does for partly out-of-frame
    subjects.
    """
    body = BodyPlan(rng)
    pose = sample_pose(label, rng)

    # subject distance from camera → overall scale in frame
    scale = rng.uniform(0.55, 1.15)
    for attr in ("torso", "shoulder_w", "hip_w", "neck_head", "upper_arm",
                 "forearm", "hand", "thigh", "shin", "foot"):
        setattr(body, attr, getattr(body, attr) * scale)

    root = np.array([rng.uniform(0.30, 0.70), pose["root_y"]])
    P = build_skeleton(body, pose["torso_tilt"], pose["head_tilt"],
                       pose["arm"], pose["leg"], root)
    P, vis = apply_camera_and_noise(P, rng)
    P = np.clip(P, -0.12, 1.12)
    return P.astype(np.float32), vis.astype(np.float32)
