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

    # ------------------------------------------------------ committed samples
    # Regression guard. Bundled samples are useless if .gitignore quietly drops
    # them: the app works locally, the deployed copy shows no button, and the
    # commit message says otherwise. This caught `!assets/*.mp4` failing to
    # reach assets/samples/.
    print("[0] every bundled sample is tracked by git")
    import subprocess
    sdir = os.path.join(ROOT, "assets", "samples")
    if os.path.isdir(sdir):
        try:
            tracked = set(subprocess.check_output(
                ["git", "ls-files", "assets/samples"], cwd=ROOT,
                text=True).split())
            for f in sorted(os.listdir(sdir)):
                rel = f"assets/samples/{f}"
                check(f"tracked: {f}", rel in tracked,
                      "" if rel in tracked else "on disk but NOT committed")
        except Exception as exc:                       # pragma: no cover
            print(f"       (git unavailable: {exc})")

    # ---------------------------------------------------------------- files
    print("\n[1] artefacts")
    for rel in ("models/fallguard_cnn.npz", "models/labels.json",
                "models/fallguard_web.json", "assets/live_monitor.js",
                "assets/live_ui.js", "src/webcam.py",
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

    # Pelvis height is only comparable across classes when the pelvis is
    # actually IN the frame. Since the corpus gained camera-framing
    # augmentation, ~15% of skeletons have hips extrapolated off the bottom
    # edge, and a standing person cropped at the waist registers a lower
    # pelvis than someone genuinely on the floor. Averaging the two
    # populations together compares a geometric fact with a framing artefact,
    # so the in-frame samples are the ones measured here.
    rng = np.random.default_rng(0)
    stats = {}
    for lab, name in enumerate(CLASS_NAMES):
        pel, tor = [], []
        for _ in range(300):
            P, V = generate_sample(lab, rng)
            c = clinical_summary(P, V)
            if c["pelvis_height"] > 0.02:          # pelvis genuinely in frame
                pel.append(c["pelvis_height"])
            tor.append(c["torso_angle"])
        stats[name] = (float(np.mean(pel)), float(np.mean(tor)))

    fall_pelvis = stats["Fall Detected"][0]
    check("fall pelvis is lowest in frame (in-frame samples)",
          all(fall_pelvis < v[0] for k, v in stats.items() if k != "Fall Detected"),
          f"fall={fall_pelvis:.3f}")
    check("bending trunk angle overlaps a fall's (the hard case exists)",
          stats["Bending"][1] > 35,
          f"bending trunk={stats['Bending'][1]:.1f}°")

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
    # Sensitivity is reported separately for fully-framed and partially-framed
    # skeletons. Merging them hides the trade-off being made: a fall filmed so
    # close that the hips are off-screen is genuinely harder, and holding the
    # combined number to the full-body bar would mean either weakening the
    # off-frame guard that killed the seated false alarm, or quietly lowering
    # the bar and calling it a pass.
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
        if name == "Fall Detected":
            continue
        bad = levels.get(ALERT, 0) + levels.get(EMERGENCY, 0)
        if name == "Bending":
            # Not asserted at exactly zero, because it is not exactly zero and
            # saying otherwise would be a lie the tests enforce. The residual
            # ~1 in 2 800 is a generated crouch so extreme that the pelvis is
            # on the floor and the body is wider than tall — geometrically a
            # fall, and it scores exactly ON the threshold. Driving it to zero
            # costs 8 points of fall sensitivity, which is the wrong trade in
            # a safety system.
            check("false alarms on Bending stay under 0.5%", bad <= 2,
                  f"{bad}/400 · acc {acc:.3f}")
        else:
            check(f"no false alarm on {name}", bad == 0,
                  f"{bad} false alerts · acc {acc:.3f}")

    # Fall sensitivity gets its own, larger sample and is reported split by
    # framing. Only ~3% of falls land off-frame, so inside a 400-sample run
    # that subgroup is ~12 cases — far too few to hold to a threshold without
    # the result being decided by noise. Merging the groups instead would hide
    # the trade-off: a fall filmed close enough that the hips leave the frame
    # is genuinely harder, and reporting one blended number would let a real
    # weakness pass unnoticed behind the easy majority.
    full, part = [0, 0], [0, 0]
    for _ in range(1200):
        P, V = generate_sample(FALL, rng)
        det.reset()
        p = det.predict(P, V, timestamp=0.0, temporal=False)
        b = full if p.clinical["pelvis_height"] > 0.02 else part
        b[1] += 1
        b[0] += p.level in (ALERT, EMERGENCY)

    check("fall reaches ALERT >=80% of fully-framed frames",
          full[0] >= 0.80 * max(full[1], 1),
          f"{full[0]}/{full[1]} = {full[0]/max(full[1],1):.1%} "
          "(measured 82.6% over 2,907)")

    # Reported, not asserted. Setting a threshold this subgroup can clear would
    # mean picking a number low enough to be meaningless, which reads as a
    # passing test while hiding a real weakness. When the crop puts the hips
    # off-screen, the extrapolated hip corrupts the trunk-angle measurement as
    # well as pelvis height, so even posture shape stops being trustworthy —
    # a limit of the framing, not of the thresholds. ~3% of falls; see README.
    print(f"       (falls with hips off-frame reach ALERT "
          f"{part[0]}/{part[1]} = {part[0]/max(part[1],1):.1%} — intrinsic to "
          "the framing, documented as a known limitation)")

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
    # regression guard: `bending_frames` must be the CLASS count, not
    # "everything that is not a fall" — the prefix here is Standing/Walking
    check("'Bending' counts the class, not all non-falls",
          s["bending_frames"] == s["counts"]["Bending"]
          and s["bending_frames"] < s["non_fall_frames"],
          f"bending={s['bending_frames']} "
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

        # Recall is split by framing for the same reason the alert rate is:
        # since the corpus gained camera-framing augmentation the test set
        # contains falls cropped at the waist, which are materially harder.
        # A single blended recall would quietly absorb that.
        in_frame = np.array([clinical_summary(P[i], V[i])["pelvis_height"] > 0.02
                             for i in range(len(P))])
        is_fall = y == FALL
        rec_all = float((pred[is_fall] == FALL).mean())
        rec_full = float((pred[is_fall & in_frame] == FALL).mean())
        fall_prec = float((y[pred == FALL] == FALL).mean())

        check("fall recall >= 0.99 on fully-framed test samples",
              rec_full >= 0.99,
              f"{rec_full:.4f}  (n={int((is_fall & in_frame).sum())})")
        check("fall recall >= 0.98 overall", rec_all >= 0.98, f"{rec_all:.4f}")
        check("fall precision == 1.000 (no false fall is ever reported)",
              fall_prec == 1.0, f"{fall_prec:.4f}")
        if (is_fall & ~in_frame).any():
            rec_part = float((pred[is_fall & ~in_frame] == FALL).mean())
            print(f"       (falls with hips off-frame: recall {rec_part:.4f} "
                  f"over n={int((is_fall & ~in_frame).sum())} — known "
                  "limitation, documented in README)")
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

    g = vid.write_gif(frames, fps=6)
    check("looping GIF encodes", g is not None,
          f"{os.path.getsize(g)/1024:.0f} KB" if g else "")
    if g:
        from PIL import Image
        im = Image.open(g)
        # loop == 0 means "repeat forever"; anything else stops after N plays
        check("GIF loops forever", im.info.get("loop") == 0,
              f"loop={im.info.get('loop')}, {im.n_frames} frames")

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
        check("five tabs present", len(at.tabs) == 5, str(len(at.tabs)))
        at.button[0].click().run()
        check("simulation runs with no exception", not at.exception,
              at.exception[0].value if at.exception else "")
    except ImportError:
        print("       (streamlit testing harness unavailable — skipped)")

    # ------------------------------------------------ live browser component
    # The real-time tab runs the model in JavaScript, so the usual Python
    # checks cover none of it. These verify the payload is complete and, where
    # Node is available, actually execute the ported network and compare it
    # against Python rather than trusting that it works.
    print("\n[10] in-browser live monitoring")
    import re
    import subprocess

    from src import webcam

    ok_web, msg_web = webcam.available()
    check("exported browser model present", ok_web, msg_web)

    if ok_web:
        spec_txt, core_js, ui_js = webcam._payload()
        spec = json.loads(spec_txt)

        check("golden cases carry a rendered tensor for the arithmetic check",
              all("tensor_u8_b64" in g and "features" in g for g in spec["golden"]),
              f"{len(spec['golden'])} cases")
        check("golden cases are genuinely ambiguous (a weak test proves nothing)",
              max(sorted(g["expected"])[-1] - sorted(g["expected"])[-2]
                  for g in spec["golden"]) < 0.95,
              "max top-2 margin "
              f"{max(sorted(g['expected'])[-1] - sorted(g['expected'])[-2] for g in spec['golden']):.3f}")

        # every element the UI addresses must exist in the component markup
        ids = set(re.findall(r'\$\("([A-Za-z0-9_]+)"\)', ui_js))
        missing = sorted(i for i in ids if f'id="{i}"' not in webcam.BODY)
        check("every DOM id referenced by the UI exists", not missing,
              f"{len(ids)} ids" if not missing else f"missing {missing}")

        check("the canvas rasteriser is gone (it did not match OpenCV)",
              "RCTX" not in core_js and "createElement" not in core_js)

        # the port itself, executed
        try:
            node = subprocess.run(["node", "--version"], capture_output=True,
                                  text=True, timeout=20)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            node = None
        if node and node.returncode == 0:
            head = ui_js.split("/* ---------------------------------------"
                               "--------------------- alert state */")[0]
            head = head.replace(
                "const $ = (id) => document.getElementById(id);", "")
            bundle = os.path.join(ROOT, "reports", "_port_check.mjs")
            with open(bundle, "w") as fh:
                fh.write(f"const SPEC={spec_txt};\n{core_js}\n{head}\n"
                         "export {selfTest};\n")
            driver = (
                "globalThis.atob=(b)=>Buffer.from(b,'base64').toString('binary');"
                f"const m=await import('file://{bundle}');"
                "const r=m.selfTest();"
                "console.log(JSON.stringify(r));"
                "process.exit(r.ok?0:1);")
            res = subprocess.run(["node", "--input-type=module", "-e", driver],
                                 capture_output=True, text=True, timeout=180)
            os.remove(bundle)
            tail = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else "{}"
            try:
                r = json.loads(tail)
            except json.JSONDecodeError:
                r = {}
            check("JavaScript port reproduces Python", res.returncode == 0,
                  f"arithmetic Δ {r.get('arith', float('nan')):.2e}, "
                  f"raster Δ {r.get('raster', float('nan')):.5f}"
                  if r else (res.stderr.strip().splitlines() or ["no output"])[-1])
        else:
            print("       (node unavailable — port executed only in the browser)")

    # ------------------------------------------------- deployment integrity
    # This section exists because the deployed app died with an ImportError on
    # `import cv2` while every file in this repository stayed byte-identical.
    # mediapipe declares `opencv-contrib-python` with no upper bound, OpenCV
    # 5.0.0.93 was published, the container was rebuilt, and pip helpfully took
    # the new major version — whose GUI build needs a libGL that Streamlit
    # Cloud does not have.
    #
    # Pinning the direct requirements was never enough to prevent that: the
    # danger lives in what those pins *drag in*. So the guard checks the
    # transitive edges, which is where the bug actually was.
    print("\n[11] deployment integrity — an app that runs today should run in a month")
    import re

    req_path = os.path.join(ROOT, "requirements.txt")
    with open(req_path) as fh:
        req_lines = [ln.split("#")[0].strip() for ln in fh]
    reqs = [ln for ln in req_lines if ln]
    pinned = {}
    for ln in reqs:
        mt = re.match(r"^([A-Za-z0-9._-]+)\s*==\s*([^\s;]+)$", ln)
        if mt:
            pinned[mt.group(1).lower().replace("_", "-")] = mt.group(2)

    check("every direct requirement is pinned with ==",
          len(pinned) == len(reqs),
          f"{len(pinned)}/{len(reqs)} pinned"
          + ("" if len(pinned) == len(reqs)
             else f" — loose: {[r for r in reqs if '==' not in r]}"))

    pkgs_path = os.path.join(ROOT, "packages.txt")
    check("packages.txt exists (apt deps for Streamlit Cloud)",
          os.path.exists(pkgs_path))
    if os.path.exists(pkgs_path):
        # Parsed EXACTLY as Streamlit Cloud parses it: every non-empty line is
        # handed to `apt-get install` verbatim. It does NOT strip `#` comments.
        #
        # The first version of this file carried a long explanatory header and
        # took the deployment down harder than the bug it was fixing — apt tried
        # to install packages named "Every", "OpenCV", "headless" and failed,
        # which aborted the whole install step. An earlier version of THIS CHECK
        # did `ln.split("#")[0]`, so it was more permissive than the real
        # consumer and reported a cheerful PASS on a file that could not work.
        #
        # A test that models the consumer more leniently than the consumer is
        # worse than no test: it converts a loud failure into a false sense of
        # safety. So the rule here is the actual rule — bare package names only.
        with open(pkgs_path) as fh:
            apt_lines = [ln.rstrip("\n") for ln in fh if ln.strip()]
        apt = set(apt_lines)

        bad = [ln for ln in apt_lines
               if not re.fullmatch(r"[a-z0-9][a-z0-9+._-]*", ln)]
        check("packages.txt contains only bare apt package names",
              not bad,
              "no comments, no blank-line cruft" if not bad
              else f"apt would try to install these literally: {bad[:6]}")

        # a GUI OpenCV needs both; without them `import cv2` aborts at startup
        for lib in ("libgl1", "libglib2.0-0"):
            check(f"apt package '{lib}' declared", lib in apt,
                  "" if lib in apt else "the exact library the outage was missing")

    # The real guard: walk what the pinned packages themselves require, and
    # fail on any UNBOUNDED dependency that is not pinned here. An unbounded
    # edge is a promise that some future release of a package nobody in this
    # repository chose will keep working — which is the promise that broke.
    import importlib.metadata as md
    from packaging.requirements import Requirement

    unbounded = {}
    for name in sorted(pinned):
        try:
            deps = md.requires(name) or []
        except md.PackageNotFoundError:
            continue
        for raw in deps:
            try:
                r = Requirement(raw)
            except Exception:
                continue
            if r.marker is not None and not r.marker.evaluate():
                continue            # extras / platform-specific, not installed
            dep = r.name.lower().replace("_", "-")
            if not r.specifier and dep not in pinned:
                unbounded.setdefault(dep, []).append(name)

    # Native packages are the ones that can actually take the app down, because
    # they carry shared objects that must find system libraries at import time.
    risky = {d: v for d, v in unbounded.items()
             if any(k in d for k in ("opencv", "cv2", "torch", "tensorflow",
                                     "pyqt", "pyside", "vtk", "wx"))}
    check("no unpinned native transitive dependency (the outage class)",
          not risky,
          "clean" if not risky
          else "; ".join(f"{d} (pulled in by {', '.join(v)})"
                         for d, v in risky.items()))

    check("opencv is pinned to the 4.x line the rasteriser was fitted against",
          all(v.startswith("4.") for k, v in pinned.items() if "opencv" in k)
          and any("opencv-contrib" in k for k in pinned),
          ", ".join(f"{k}=={v}" for k, v in pinned.items() if "opencv" in k))

    if unbounded:
        print(f"       (note: {len(unbounded)} pure-python transitive deps are "
              "unbounded — lower risk, listed for awareness)")

    # ...and prove the pins still form a solvable set, rather than trusting it.
    try:
        dry = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--dry-run", "--quiet",
             "--report", os.path.join(ROOT, "reports", "_pipreport.json"),
             "-r", req_path],
            capture_output=True, text=True, timeout=240)
        rep = os.path.join(ROOT, "reports", "_pipreport.json")
        got = {}
        if os.path.exists(rep):
            with open(rep) as fh:
                for item in json.load(fh).get("install", []):
                    got[item["metadata"]["name"].lower()] = item["metadata"]["version"]
            os.remove(rep)
        bad = {n: v for n, v in got.items() if "opencv" in n and not v.startswith("4.")}
        check("requirements.txt still resolves, with no OpenCV 5.x",
              dry.returncode == 0 and not bad,
              "resolved" if dry.returncode == 0 and not bad
              else (f"would install {bad}" if bad
                    else dry.stderr.strip().splitlines()[-1:] or "pip failed"))
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(f"       (dependency resolution not checked: {exc})")

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
