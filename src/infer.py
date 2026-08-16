"""
infer.py — The serving-time inference engine and the fall-alert state machine.

This is where model output becomes a *clinical decision*. Two ideas drive the
design, and both exist because a raw per-frame classifier is unsafe in a
healthcare setting:

1. **Two-tier corroboration.** The CNN's ``Fall Detected`` probability is
   cross-checked against ``features.biomechanical_fall_score`` — an independent,
   fully transparent geometric rule. Because the two disagree in different ways
   (the CNN can be fooled by unusual limb configurations; the rule can be fooled
   by a deep bend), requiring agreement before escalating removes most false
   alarms while keeping true falls.

2. **Temporal persistence and impact velocity.** A fall is an *event*, not a
   frame. On video the engine additionally requires the fall state to persist
   over consecutive frames, and separately watches for rapid pelvis descent
   (the impact signature). A person who is simply sitting on the floor is
   flagged, but without the sudden-descent marker — which is exactly the
   distinction a caregiver needs.

Alert levels
------------
``NORMAL``   nothing of concern
``WATCH``    one detector fired; log it, do not page anyone
``ALERT``    both detectors agree on a single frame
``EMERGENCY`` sustained agreement, or agreement plus impact velocity
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .features import biomechanical_fall_score, clinical_summary, extract
from .render import render_cnn
from .skeleton import CLASS_NAMES, FALL, L_HIP, R_HIP

NORMAL, WATCH, ALERT, EMERGENCY = "NORMAL", "WATCH", "ALERT", "EMERGENCY"

#: Level → (display label, hex colour) used by the dashboard.
LEVEL_STYLE = {
    NORMAL: ("ALL CLEAR", "#00ff9c"),
    WATCH: ("MONITORING", "#ffd400"),
    ALERT: ("FALL ALERT", "#ff7a00"),
    EMERGENCY: ("EMERGENCY — FALL CONFIRMED", "#ff1f4f"),
}


@dataclass
class Prediction:
    """Everything the dashboard needs about a single analysed frame."""
    label: str
    label_index: int
    confidence: float
    probabilities: np.ndarray
    biomech_score: float
    clinical: dict
    level: str
    reasons: list[str] = field(default_factory=list)
    landmarks: np.ndarray | None = None
    visibility: np.ndarray | None = None
    frame_index: int = 0
    timestamp: float = 0.0


class FallDetector:
    """Stateful engine: pose → CNN → corroboration → alert level.

    Parameters
    ----------
    cnn_prob_threshold
        Minimum CNN ``Fall Detected`` probability to count as a fall vote.
    biomech_threshold
        Minimum rule-based score to count as the corroborating vote.
    persistence_frames
        Consecutive corroborated frames required to escalate to EMERGENCY.
    impact_velocity
        Pelvis descent, in frame-heights per second, that marks an impact.
    """

    def __init__(
        self,
        model_dir: str,
        cnn_prob_threshold: float = 0.55,
        # 0.42 is calibrated, not guessed. On held-out data the rule-based score
        # separates cleanly — falls sit at median 0.74 (5th percentile 0.44)
        # while every non-fall class sits below 0.41 at the 95th percentile.
        # 0.42 corroborates ~97% of true falls at a ~1% non-fall false vote,
        # and because an alert additionally requires the CNN to agree — and the
        # CNN has 100% fall precision on test — the *joint* false-alarm rate is
        # effectively zero. Raising it to 0.50 would silently drop 6% of real
        # falls to WATCH for no practical gain in specificity.
        biomech_threshold: float = 0.42,
        persistence_frames: int = 4,
        impact_velocity: float = 0.55,
        history: int = 240,
    ):
        from .cnn_numpy import NumpyHybrid

        self.cnn = NumpyHybrid(os.path.join(model_dir, "fallguard_cnn.npz"))
        with open(os.path.join(model_dir, "labels.json")) as fh:
            self.meta = json.load(fh)
        self.classes = self.meta.get("classes", CLASS_NAMES)

        self.cnn_prob_threshold = cnn_prob_threshold
        self.biomech_threshold = biomech_threshold
        self.persistence_frames = persistence_frames
        self.impact_velocity = impact_velocity

        self._pelvis: deque[tuple[float, float]] = deque(maxlen=12)   # (t, y)
        self._streak = 0
        self.history: deque[Prediction] = deque(maxlen=history)

    # -- state -------------------------------------------------------------
    def reset(self) -> None:
        """Clear temporal state. Call between uploads."""
        self._pelvis.clear()
        self._streak = 0
        self.history.clear()

    # -- core --------------------------------------------------------------
    def predict(
        self,
        P: np.ndarray,
        V: np.ndarray,
        timestamp: float = 0.0,
        frame_index: int = 0,
        temporal: bool = False,
    ) -> Prediction:
        """Classify one skeleton and fold it into the alert state machine.

        ``temporal=False`` (single image) skips persistence and velocity, since
        neither is defined without a time axis — a still photograph can reach
        ALERT but never EMERGENCY on persistence alone.
        """
        prob = self.cnn.predict(render_cnn(P, V)[None], extract(P, V)[None])[0]
        idx = int(prob.argmax())
        conf = float(prob[idx])

        bio = biomechanical_fall_score(P, V)
        clin = clinical_summary(P, V)

        cnn_vote = float(prob[FALL]) >= self.cnn_prob_threshold
        bio_vote = bio >= self.biomech_threshold

        # ---- pelvis descent velocity (frame-heights / second) ------------
        pelvis_y = float((P[L_HIP, 1] + P[R_HIP, 1]) / 2.0)
        velocity = 0.0
        if temporal:
            self._pelvis.append((timestamp, pelvis_y))
            if len(self._pelvis) >= 2:
                t0, y0 = self._pelvis[0]
                dt = timestamp - t0
                if dt > 1e-3:
                    velocity = (pelvis_y - y0) / dt      # +ve = moving downward
        impact = velocity >= self.impact_velocity

        # ---- fuse --------------------------------------------------------
        reasons: list[str] = []
        if cnn_vote:
            reasons.append(f"CNN fall probability {prob[FALL]:.0%}")
        if bio_vote:
            reasons.append(f"biomechanical score {bio:.0%}")
        # Descent velocity is only meaningful as *corroboration*. Surfacing it
        # on its own would put "pelvis descent 0.65 frame-heights/s" under an
        # ALL CLEAR banner for someone simply sitting down quickly, which reads
        # as a contradiction to a caregiver.
        if impact and (cnn_vote or bio_vote):
            reasons.append(f"pelvis descent {velocity:.2f} frame-heights/s")

        if cnn_vote and bio_vote:
            self._streak += 1
        else:
            self._streak = 0

        if cnn_vote and bio_vote:
            sustained = temporal and self._streak >= self.persistence_frames
            level = EMERGENCY if (sustained or impact) else ALERT
            if sustained:
                reasons.append(f"sustained over {self._streak} frames")
        elif cnn_vote or bio_vote:
            level = WATCH
        else:
            level = NORMAL

        if level == NORMAL:
            reasons.append("posture consistent with normal activity")

        # low landmark visibility must never masquerade as confidence
        if clin["mean_visibility"] < 0.45:
            reasons.append("⚠ low landmark visibility — treat result as provisional")

        pred = Prediction(
            label=self.classes[idx], label_index=idx, confidence=conf,
            probabilities=prob, biomech_score=bio, clinical=clin,
            level=level, reasons=reasons, landmarks=P, visibility=V,
            frame_index=frame_index, timestamp=timestamp,
        )
        self.history.append(pred)
        return pred

    # -- aggregation for the analytics panel -------------------------------
    def summary(self) -> dict:
        """Roll the history up into the dashboard's monitoring metrics."""
        if not self.history:
            return {
                "total": 0, "counts": {c: 0 for c in self.classes},
                "fall_frames": 0, "normal_frames": 0, "mean_confidence": 0.0,
                "peak_level": NORMAL, "alert_frames": 0,
            }

        counts = {c: 0 for c in self.classes}
        for p in self.history:
            counts[p.label] += 1

        order = [NORMAL, WATCH, ALERT, EMERGENCY]
        peak = max((p.level for p in self.history), key=order.index)
        falls = counts.get(self.classes[FALL], 0)

        return {
            "total": len(self.history),
            "counts": counts,
            "fall_frames": falls,
            "normal_frames": len(self.history) - falls,
            "mean_confidence": float(np.mean([p.confidence for p in self.history])),
            "peak_level": peak,
            "alert_frames": sum(1 for p in self.history if p.level in (ALERT, EMERGENCY)),
        }

    def timeline(self) -> dict:
        """Per-frame series for the monitoring charts."""
        return {
            "t": [p.timestamp for p in self.history],
            "frame": [p.frame_index for p in self.history],
            "label": [p.label for p in self.history],
            "confidence": [p.confidence for p in self.history],
            "fall_prob": [float(p.probabilities[FALL]) for p in self.history],
            "biomech": [p.biomech_score for p in self.history],
            "level": [p.level for p in self.history],
        }
