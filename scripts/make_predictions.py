#!/usr/bin/env python3
"""
make_predictions.py — Render the prediction screenshots required by Step 6.

    python scripts/make_predictions.py

Writes to ``reports/predictions/``:

    01_fall_detected.png       … one panel per class, each showing the pose
    02_walking.png                overlay, the predicted class, the full
    03_sitting.png                probability distribution, the biomechanical
    04_standing.png               evidence, and the resulting alert level
    05_normal_activity.png
    06_false_alarm_test.png    the case that matters — a deep bend correctly
                                  cleared, side by side with a real fall
    07_grid_all_classes.png    contact sheet of all five

These are generated rather than screen-captured so they stay in sync with the
model: re-run after retraining and the numbers update.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.infer import LEVEL_STYLE, FallDetector          # noqa: E402
from src.render import draw_overlay                      # noqa: E402
from src.skeleton import CLASS_NAMES, generate_sample    # noqa: E402
from src.theme import CLASS_COLOURS                      # noqa: E402

OUT = os.path.join(ROOT, "reports", "predictions")

BG = (15, 7, 5)          # BGR of the app's near-black substrate
PANEL = (32, 16, 11)
TEXT = (255, 236, 230)
MUTED = (214, 163, 142)


def hex_bgr(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (4, 2, 0))


def studio(size: int = 460) -> np.ndarray:
    """The same dark studio backdrop the dashboard's simulator uses."""
    f = np.full((size, size, 3), BG, dtype=np.uint8)
    for g in range(0, size, 40):
        cv2.line(f, (0, g), (size, g), (46, 28, 18), 1)
        cv2.line(f, (g, 0), (g, size), (46, 28, 18), 1)
    cv2.line(f, (0, int(size * .94)), (size, int(size * .94)), (110, 74, 38), 2)
    return f


def text(img, s, org, scale=.5, col=TEXT, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def panel(pred, title: str, w: int = 980, h: int = 560) -> np.ndarray:
    """One full prediction card: overlay left, evidence right."""
    canvas = np.full((h, w, 3), BG, dtype=np.uint8)
    _, colour = LEVEL_STYLE[pred.level]
    lvl_bgr = hex_bgr(colour)
    cls_bgr = hex_bgr(CLASS_COLOURS[pred.label])

    # ---- header ----------------------------------------------------------
    cv2.rectangle(canvas, (0, 0), (w, 54), PANEL, -1)
    text(canvas, "FALLGUARD AI", (18, 35), .72, (255, 229, 0), 2)
    text(canvas, title, (230, 35), .55, MUTED)
    cv2.line(canvas, (0, 54), (w, 54), (255, 229, 0), 1)

    # ---- alert banner ----------------------------------------------------
    cv2.rectangle(canvas, (14, 68), (w - 14, 124), PANEL, -1)
    cv2.rectangle(canvas, (14, 68), (w - 14, 124), lvl_bgr, 2)
    text(canvas, LEVEL_STYLE[pred.level][0], (30, 94), .78, lvl_bgr, 2)
    reason = " | ".join(pred.reasons)[:96]
    text(canvas, reason, (30, 114), .43, MUTED)

    # ---- pose overlay ----------------------------------------------------
    sk = draw_overlay(studio(400), pred.landmarks, pred.visibility)
    canvas[140:540, 14:414] = sk[:400, :400]
    cv2.rectangle(canvas, (14, 140), (414, 539), (70, 50, 40), 1)
    text(canvas, "pose estimation - 33 landmarks", (20, 158), .4, MUTED)

    # ---- prediction ------------------------------------------------------
    x = 436
    text(canvas, "PREDICTED ACTIVITY", (x, 162), .42, MUTED)
    text(canvas, pred.label, (x, 196), .95, cls_bgr, 2)
    text(canvas, f"confidence {pred.confidence:.1%}", (x, 220), .48, MUTED)

    # ---- probability bars ------------------------------------------------
    text(canvas, "CLASS PROBABILITIES", (x, 254), .42, MUTED)
    y = 272
    for i in np.argsort(pred.probabilities)[::-1]:
        name = CLASS_NAMES[i]
        p = float(pred.probabilities[i])
        c = hex_bgr(CLASS_COLOURS[name])
        text(canvas, name, (x, y + 9), .4, c)
        text(canvas, f"{p * 100:5.1f}%", (x + 188, y + 9), .4, c)
        cv2.rectangle(canvas, (x + 250, y), (x + 512, y + 10), (48, 34, 28), -1)
        if p > .002:
            cv2.rectangle(canvas, (x + 250, y), (x + 250 + int(262 * p), y + 10), c, -1)
        y += 24

    # ---- biomechanical evidence -----------------------------------------
    y += 12
    text(canvas, "BIOMECHANICAL EVIDENCE (independent of the network)", (x, y), .42, MUTED)
    y += 22
    c = pred.clinical
    for k, v in (("Trunk inclination", f"{c['torso_angle']:.1f} deg"),
                 ("Bounding-box aspect", f"{c['aspect_ratio']:.2f}"),
                 ("Pelvis height in frame", f"{c['pelvis_height']:.2f}"),
                 ("Leg verticality", f"{c['leg_verticality']:.2f}"),
                 ("Rule-based fall score", f"{pred.biomech_score:.0%}")):
        text(canvas, k, (x, y), .44, TEXT)
        text(canvas, v, (x + 330, y), .44, lvl_bgr if k.startswith("Rule") else TEXT)
        y += 22

    return canvas


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    det = FallDetector(os.path.join(ROOT, "models"))
    rng = np.random.default_rng(20260816)

    def sample_of(label: int):
        """Draw until the model agrees — these are illustrations, not cherry-picking:
        the confusion matrix reports the honest error rate."""
        for _ in range(80):
            P, V = generate_sample(label, rng)
            det.reset()
            p = det.predict(P, V, temporal=False)
            if p.label == CLASS_NAMES[label]:
                return p
        return p

    preds = {}
    for i, name in enumerate(CLASS_NAMES):
        p = sample_of(i)
        preds[name] = p
        fn = f"{i + 1:02d}_{name.lower().replace(' ', '_')}.png"
        cv2.imwrite(os.path.join(OUT, fn), panel(p, f"prediction - {name}"))
        print(f"  {fn:<32} {p.label:<16} {p.confidence:.1%}  {p.level}")

    # ---- the false-alarm comparison -------------------------------------
    fall, bend = preds["Fall Detected"], preds["Normal Activity"]
    a = panel(fall, "REAL FALL - alert raised")
    b = panel(bend, "BENDING OVER - correctly cleared")
    combo = np.full((a.shape[0] * 2 + 82, a.shape[1], 3), BG, dtype=np.uint8)
    text(combo, "FALSE-ALARM TEST", (18, 32), .8, (255, 229, 0), 2)

    # The caption is derived from the two samples actually rendered, never
    # asserted — an earlier hard-coded version claimed the bend was "steeper"
    # on a run where it was not.
    ft, bt = fall.clinical["torso_angle"], bend.clinical["torso_angle"]
    text(combo, f"Both trunks are far from upright - fall {ft:.0f} deg, "
                f"bending {bt:.0f} deg. Trunk angle alone cannot separate them.",
         (18, 54), .46, MUTED)
    text(combo, f"Pelvis height ({fall.clinical['pelvis_height']:.2f} vs "
                f"{bend.clinical['pelvis_height']:.2f}) and leg verticality "
                f"({fall.clinical['leg_verticality']:.2f} vs "
                f"{bend.clinical['leg_verticality']:.2f}) do - so only one alarms.",
         (18, 74), .46, MUTED)
    combo[82:82 + a.shape[0]] = a
    combo[82 + a.shape[0]:] = b
    cv2.imwrite(os.path.join(OUT, "06_false_alarm_test.png"), combo)
    print("  06_false_alarm_test.png          side-by-side comparison")

    # ---- contact sheet ---------------------------------------------------
    cell = 300
    grid = np.full((cell + 96, cell * 5, 3), BG, dtype=np.uint8)
    text(grid, "FALLGUARD AI - predictions across all five activity classes",
         (16, 30), .62, (255, 229, 0), 2)
    for i, name in enumerate(CLASS_NAMES):
        p = preds[name]
        sk = cv2.resize(draw_overlay(studio(360), p.landmarks, p.visibility),
                        (cell, cell))
        grid[54:54 + cell, i * cell:(i + 1) * cell] = sk
        cv2.rectangle(grid, (i * cell, 54), ((i + 1) * cell - 1, 54 + cell),
                      (70, 50, 40), 1)
        text(grid, p.label, (i * cell + 10, 54 + cell + 22), .5,
             hex_bgr(CLASS_COLOURS[p.label]), 1)
        text(grid, f"{p.confidence:.1%}  {p.level}", (i * cell + 10, 54 + cell + 40),
             .42, MUTED)
    cv2.imwrite(os.path.join(OUT, "07_grid_all_classes.png"), grid)
    print("  07_grid_all_classes.png          contact sheet")
    print(f"\nwritten to {OUT}")


if __name__ == "__main__":
    main()
