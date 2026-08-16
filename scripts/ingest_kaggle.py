#!/usr/bin/env python3
"""
ingest_kaggle.py — Convert a real image dataset into the FallGuard corpus format.

The procedural corpus and a real dataset are interchangeable: both are just the
five arrays described in ``src/dataset.py``. This script runs MediaPipe Pose
over a folder of labelled images and emits exactly that, so the entire
pipeline — features, rendering, training, evaluation, deployment — is reused
unchanged.

USAGE
-----
1. Download a fall-detection dataset, e.g.

       kaggle datasets download -d uttejkumarkandagatla/fall-detection-dataset
       unzip fall-detection-dataset.zip -d data/raw

2. Arrange it as one folder per class. Folder names are matched
   case-insensitively against the aliases in ``CLASS_ALIASES`` below, so most
   public datasets need no renaming:

       data/raw/
         fall/          or  Fall Detected/ , falling/ , fall_down/
         walking/
         sitting/
         standing/
         normal/        or  Normal Activity/ , bending/ , adl/

3. Convert, then retrain:

       python scripts/ingest_kaggle.py --input data/raw --output data/real.npz
       python scripts/train.py --dataset data/real.npz --reset

MIXING REAL AND SYNTHETIC
-------------------------
       python scripts/ingest_kaggle.py --input data/raw --output data/real.npz \\
              --blend-synthetic 1200

adds 1200 procedurally generated samples per class alongside the real ones,
which is usually the strongest configuration: real images supply authentic
pose-estimator noise and camera geometry, while the synthetic samples fill in
the rare fall configurations that public datasets under-represent.

NOTE ON LABEL QUALITY
---------------------
Many public fall datasets are binary (fall / no-fall). Ingesting one of those
gives you two populated classes and three empty ones; ``--binary`` collapses
the label set accordingly so training does not waste capacity on classes with
no support.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.dataset import TRAIN_FRAC, VAL_FRAC, save          # noqa: E402
from src.skeleton import CLASS_NAMES, generate_sample       # noqa: E402

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

#: folder-name aliases → canonical class index
CLASS_ALIASES = {
    0: {"fall", "falls", "falling", "fall detected", "fall_detected", "fell",
        "fall down", "fall_down", "falldown", "lying", "lying down"},
    1: {"walk", "walking", "walk forward", "ambulation", "gait"},
    2: {"sit", "sitting", "sit down", "sitting down", "seated", "chair"},
    3: {"stand", "standing", "stand up", "standing up", "upright"},
    4: {"normal", "normal activity", "not fall", "no fall", "nofall", "non-fall",
        "adl", "bending", "bend", "picking", "daily activity", "other"},
}


def resolve_label(folder: str) -> int | None:
    key = folder.strip().lower().replace("-", " ").replace("_", " ")
    for idx, names in CLASS_ALIASES.items():
        if key in {n.replace("_", " ") for n in names}:
            return idx
    for idx, names in CLASS_ALIASES.items():          # substring fallback
        if any(n in key for n in names):
            return idx
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of per-class subfolders")
    ap.add_argument("--output", default=os.path.join(ROOT, "data", "real.npz"))
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = no cap")
    ap.add_argument("--blend-synthetic", type=int, default=0,
                    help="synthetic samples per class to append")
    ap.add_argument("--binary", action="store_true",
                    help="collapse to Fall Detected vs Normal Activity")
    args = ap.parse_args()

    import cv2
    from src import pose

    ok, msg = pose.available()
    if not ok:
        raise SystemExit(f"MediaPipe unavailable: {msg}")

    rng = np.random.default_rng(args.seed)
    P_all, V_all, y_all, src_all = [], [], [], []
    skipped = 0

    subdirs = sorted(d for d in os.listdir(args.input)
                     if os.path.isdir(os.path.join(args.input, d)))
    if not subdirs:
        raise SystemExit(f"no class subfolders found under {args.input}")

    for d in subdirs:
        label = resolve_label(d)
        if label is None:
            print(f"  ?  '{d}' — unrecognised class name, skipped "
                  f"(add it to CLASS_ALIASES)")
            continue
        if args.binary:
            label = 0 if label == 0 else 4

        folder = os.path.join(args.input, d)
        files = [f for f in sorted(os.listdir(folder))
                 if os.path.splitext(f)[1].lower() in IMAGE_EXT]
        if args.max_per_class:
            files = files[: args.max_per_class]

        kept = 0
        for f in files:
            img = cv2.imread(os.path.join(folder, f))
            if img is None:
                skipped += 1
                continue
            if max(img.shape[:2]) > 1280:
                sc = 1280 / max(img.shape[:2])
                img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            res = pose.estimate(img)
            if res is None:                    # no detectable person — drop it
                skipped += 1
                continue
            P, V = res
            P_all.append(P)
            V_all.append(V)
            y_all.append(label)
            src_all.append("kaggle")
            kept += 1

        print(f"  ✓  {d:<28} → {CLASS_NAMES[label]:<16} {kept:>5} usable "
              f"/ {len(files)} images")

    if not P_all:
        raise SystemExit("no usable images — check --input layout and folder names")

    if args.blend_synthetic:
        for label in range(len(CLASS_NAMES)):
            for _ in range(args.blend_synthetic):
                P, V = generate_sample(label, rng)
                P_all.append(P)
                V_all.append(V)
                y_all.append(label)
                src_all.append("synthetic")
        print(f"  +  blended {args.blend_synthetic} synthetic samples per class")

    P = np.asarray(P_all, dtype=np.float32)
    V = np.asarray(V_all, dtype=np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    src = np.asarray(src_all, dtype="<U12")

    # stratified 70/15/15 — assigned within each class so val/test stay balanced
    split = np.empty(len(y), dtype="<U5")
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        n_tr = int(round(len(idx) * TRAIN_FRAC))
        n_va = int(round(len(idx) * VAL_FRAC))
        split[idx[:n_tr]] = "train"
        split[idx[n_tr:n_tr + n_va]] = "val"
        split[idx[n_tr + n_va:]] = "test"

    order = rng.permutation(len(y))
    corpus = {"landmarks": P[order], "visibility": V[order], "labels": y[order],
              "split": split[order], "source": src[order]}

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    save(corpus, args.output)

    print(f"\n{len(y)} samples written to {args.output}  "
          f"({skipped} images skipped — no person detected)")
    for i, name in enumerate(CLASS_NAMES):
        n = int((y == i).sum())
        flag = "   ← EMPTY" if n == 0 else ""
        print(f"  {name:<18}{n:>7}{flag}")
    print("\nnext:  python scripts/train.py --dataset "
          f"{os.path.relpath(args.output, ROOT)} --reset")


if __name__ == "__main__":
    main()
