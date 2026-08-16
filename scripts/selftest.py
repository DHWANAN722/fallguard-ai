#!/usr/bin/env python3
"""
selftest.py — End-to-end verification of every component.

    python scripts/selftest.py

Checks the things that would silently produce a wrong answer rather than an
exception: that the NumPy runtime still matches Keras, that the alert logic
raises no false alarms on benign postures, that the reported metrics match the
model actually on disk, and that video I/O round-trips.

Exits non-zero on any failure, so it can gate a commit.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PASS, FAIL = "  PASS", "  FAIL"
failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{PASS if cond else FAIL}  {name}{'  — ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


def main() -> None:
    print("=" * 74)
    print("FallGuard AI — self test")
    print("=" * 74)

    # ---------------------------------------------------------------- files
    print("\n[1] artefacts")
    for rel in ("models/fallguard_cnn.npz", "models/labels.json",
                "reports/metrics.json", "reports/confusion_matrix.png",
                "reports/training_curves.png", "reports/model_comparison.png",
                "reports/per_class_metrics.png",
                "reports/classification_report.txt",
                "app.py", "requirements.txt", "README.md", "REPORT.md",
                "VIDEO_SCRIPT.md", "notebooks/FallGuard_Training.ipynb",
                "reports/predictions/01_fall_detected.png",
                "reports/predictions/06_false_alarm_test.png",
                "reports/predictions/07_grid_all_classes.png"):
        check(rel, os.path.exists(os.path.join(ROOT, rel)))

    # ------------------------------------------------------------- geometry
    print("\n[2] skeleton geometry")
    from src.features import biomechanical_fall_score, clinical_summary
    from src.skeleton import CLASS_NAMES, FALL, generate_sample

    rng = np.random.default_rng(0)
    stats = {}
    for lab, name in enumerate(CLASS_NAMES):
        pel, tor = [], []
        for _ in range(300):
            P, V = generate_sample(lab, rng)
            c = clinical_summary(P, V)
            pel.append(c["pelvis_height"])
            tor.append(c["torso_angle"])
        stats[name] = (float(np.mean(pel)), float(np.mean(tor)))

    fall_pelvis = stats["Fall Detected"][0]
    check("fall pelvis is lowest in frame",
          all(fall_pelvis < v[0] for k, v in stats.items() if k != "Fall Detected"),
          f"fall={fall_pelvis:.3f}")
    check("bending trunk angle overlaps a fall's (the hard case exists)",
          stats["Normal Activity"][1] > 35,
          f"bending trunk={stats['Normal Activity'][1]:.1f}°")

    # ------------------------------------------------------------ inference
    print("\n[3] model + inference")
    from src.infer import ALERT, EMERGENCY, NORMAL, WATCH, FallDetector

    det = FallDetector(os.path.join(ROOT, "models"))
    check("model loads", det.cnn is not None)
    check("5 classes", len(det.classes) == 5, str(det.classes))

    rng = np.random.default_rng(99)
    t0 = time.time()
    n = 200
    for _ in range(n):
        det.predict(*generate_sample(0, rng))
    ms = (time.time() - t0) / n * 1000
    check("latency under 20 ms/frame", ms < 20, f"{ms:.1f} ms")

    # ------------------------------------------------------------ behaviour
    print("\n[4] alert behaviour — false alarms are the metric that matters")
    rng = np.random.default_rng(2026)
    per_class = {}
    for lab, name in enumerate(CLASS_NAMES):
        det.reset()
        levels = {}
        correct = 0
        for _ in range(400):
            p = det.predict(*generate_sample(lab, rng))
            levels[p.level] = levels.get(p.level, 0) + 1
            correct += (p.label == name)
        per_class[name] = (correct / 400, levels)

    for name, (acc, levels) in per_class.items():
        bad = levels.get(ALERT, 0) + levels.get(EMERGENCY, 0)
        if name == "Fall Detected":
            check("fall reaches ALERT >=90% of frames", bad >= 360,
                  f"{bad}/400 · acc {acc:.3f}")
        else:
            check(f"no false alarm on {name}", bad == 0,
                  f"{bad} false alerts · acc {acc:.3f}")

    print("\n[5] temporal escalation")
    det.reset()
    seq = [3] * 4 + [1] * 5 + [0] * 10
    levels = []
    for i, lab in enumerate(seq):
        p = det.predict(*generate_sample(lab, rng), timestamp=i / 6.0,
                        frame_index=i, temporal=True)
        levels.append(p.level)
    check("no alert during the benign prefix",
          all(l in (NORMAL, WATCH) for l in levels[:9]))
    check("escalates to EMERGENCY after the fall", EMERGENCY in levels[9:])
    s = det.summary()
    check("summary counts are consistent",
          s["total"] == len(seq) and s["fall_frames"] + s["non_fall_frames"] == s["total"],
          str({k: s[k] for k in ("total", "fall_frames", "non_fall_frames")}))
    # regression guard: `normal_activity_frames` must be the CLASS count, not
    # "everything that is not a fall" — the prefix here is Standing/Walking
    check("'Normal activity' counts the class, not all non-falls",
          s["normal_activity_frames"] == s["counts"]["Normal Activity"]
          and s["normal_activity_frames"] < s["non_fall_frames"],
          f"normal_activity={s['normal_activity_frames']} "
          f"non_fall={s['non_fall_frames']}")

    # regression guard: history must not silently truncate a long clip
    det.reset()
    for i in range(600):
        det.predict(*generate_sample(3, rng), timestamp=i / 6.0, frame_index=i,
                    temporal=True)
    check("history holds a long clip without truncating",
          det.summary()["total"] == 600, str(det.summary()["total"]))

    # ----------------------------------------------------- reported metrics
    print("\n[6] reported metrics match the model on disk")
    with open(os.path.join(ROOT, "reports", "metrics.json")) as fh:
        m = json.load(fh)
    deployed = next(k for k in m["models"] if "CNN" in k)
    reported = m["models"][deployed]["accuracy"]
    check("comparison table is single-split (no validation row leaked in)",
          not any("valid" in k.lower() for k in m["models"]),
          ", ".join(m["models"]))
    check("validation metrics reported separately", "validation" in m,
          f"val acc {m['validation']['accuracy']:.4f}" if "validation" in m else "missing")
    if "validation" in m:
        check("validation and test agree within 2 points",
              abs(m["validation"]["accuracy"] - reported) < 0.02,
              f"val {m['validation']['accuracy']:.4f} vs test {reported:.4f}")

    data = os.path.join(ROOT, "data", "fallguard_dataset.npz")
    if os.path.exists(data):
        from src import dataset as ds
        from src.features import extract_batch
        from src.render import render_batch

        corpus = ds.load(data)
        P, V, y = ds.split_arrays(corpus, "test")
        pred = det.cnn.predict(render_batch(P, V), extract_batch(P, V)).argmax(1)
        actual = float((pred == y).mean())
        check("recomputed test accuracy matches metrics.json",
              abs(actual - reported) < 1e-6,
              f"recomputed {actual:.4f} vs reported {reported:.4f}")

        fall_recall = float((pred[y == FALL] == FALL).mean())
        fall_prec = float((y[pred == FALL] == FALL).mean())
        check("fall recall == 1.000", fall_recall == 1.0, f"{fall_recall:.4f}")
        check("fall precision == 1.000", fall_prec == 1.0, f"{fall_prec:.4f}")
        check("deployed model beats both baselines",
              all(reported >= v["accuracy"] for k, v in m["models"].items()),
              f"{deployed} {reported:.4f}")
    else:
        print("       (corpus absent — skipping recomputation)")

    # --------------------------------------------------------------- video
    print("\n[7] video I/O")
    import cv2

    from src import video as vid
    from src.render import draw_overlay

    frames = [draw_overlay(np.full((240, 320, 3), 18, np.uint8),
                           *generate_sample(i % 5, rng)) for i in range(10)]
    r = vid.write_annotated(frames, fps=6)
    check("annotated clip encodes", r is not None,
          f"codec {r[1]}, browser-safe {r[1] in vid.BROWSER_SAFE}" if r else "")
    if r:
        info = vid.probe(r[0])
        check("clip round-trips", info["frames"] > 0, str(info["frames"]) + " frames")
        check("frame iteration works",
              len(list(vid.iter_frames(r[0], 6, 50))) > 0)

    # ---------------------------------------------------------------- pose
    print("\n[8] pose estimation")
    from src import pose
    ok, msg = pose.available()
    check("MediaPipe available", ok, msg)
    if ok:
        blank = np.full((360, 480, 3), 40, np.uint8)
        check("no-person frame returns None gracefully",
              pose.estimate(blank) is None)

    # ---------------------------------------------------------------- app
    print("\n[9] streamlit app renders")
    try:
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(os.path.join(ROOT, "app.py"),
                               default_timeout=300).run()
        check("app renders with no exception", not at.exception,
              at.exception[0].value if at.exception else "")
        check("four tabs present", len(at.tabs) == 4, str(len(at.tabs)))
        at.button[0].click().run()
        check("simulation runs with no exception", not at.exception,
              at.exception[0].value if at.exception else "")
    except ImportError:
        print("       (streamlit testing harness unavailable — skipped)")

    # --------------------------------------------------------------- done
    print("\n" + "=" * 74)
    if failures:
        print(f"FAILED — {len(failures)} check(s):")
        for f in failures:
            print(f"   · {f}")
        sys.exit(1)
    print("ALL CHECKS PASSED")
    print("=" * 74)


if __name__ == "__main__":
    main()
