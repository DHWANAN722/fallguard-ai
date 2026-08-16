"""
dataset.py — Corpus construction and the mandated 70 / 15 / 15 split.

The on-disk artefact (``data/fallguard_dataset.npz``) is the single interchange
format for the whole project. Anything that can produce these five arrays can
drive the pipeline — the procedural generator here, or
``scripts/ingest_kaggle.py`` running MediaPipe over a real image dataset.

    landmarks   (N, 33, 2)  float32   normalised frame coordinates
    visibility  (N, 33)     float32   per-landmark confidence
    labels      (N,)        int64     index into skeleton.CLASS_NAMES
    split       (N,)        <U5       "train" | "val" | "test"
    source      (N,)        <U12      provenance tag
"""

from __future__ import annotations

import numpy as np

from .skeleton import CLASS_NAMES, generate_sample

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15


def build_corpus(
    n_per_class: int = 2600,
    seed: int = 20260816,
    source: str = "synthetic",
) -> dict:
    """Generate a balanced corpus with a stratified 70/15/15 split.

    Stratifying the split *within* each class guarantees the val and test sets
    stay balanced, so accuracy is not silently inflated by class imbalance and
    the per-class recall figures are computed over equal support.
    """
    rng = np.random.default_rng(seed)
    n_classes = len(CLASS_NAMES)
    total = n_per_class * n_classes

    P = np.zeros((total, 33, 2), dtype=np.float32)
    V = np.zeros((total, 33), dtype=np.float32)
    y = np.zeros(total, dtype=np.int64)
    split = np.empty(total, dtype="<U5")

    n_tr = int(round(n_per_class * TRAIN_FRAC))
    n_va = int(round(n_per_class * VAL_FRAC))

    k = 0
    for label in range(n_classes):
        tags = np.array(["train"] * n_tr + ["val"] * n_va
                        + ["test"] * (n_per_class - n_tr - n_va))
        rng.shuffle(tags)
        for i in range(n_per_class):
            P[k], V[k] = generate_sample(label, rng)
            y[k] = label
            split[k] = tags[i]
            k += 1

    order = rng.permutation(total)
    return {
        "landmarks": P[order],
        "visibility": V[order],
        "labels": y[order],
        "split": split[order],
        "source": np.array([source] * total, dtype="<U12"),
    }


def save(corpus: dict, path: str) -> None:
    np.savez_compressed(path, **corpus)


def load(path: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def split_arrays(corpus: dict, tag: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(landmarks, visibility, labels)`` for one split tag."""
    m = corpus["split"] == tag
    return corpus["landmarks"][m], corpus["visibility"][m], corpus["labels"][m]


def describe(corpus: dict) -> str:
    """Human-readable composition table, printed by the notebook and trainer."""
    y, sp = corpus["labels"], corpus["split"]
    lines = [f"{'class':<18}{'train':>8}{'val':>8}{'test':>8}{'total':>8}"]
    lines.append("-" * 50)
    for i, name in enumerate(CLASS_NAMES):
        row = [int(((y == i) & (sp == t)).sum()) for t in ("train", "val", "test")]
        lines.append(f"{name:<18}{row[0]:>8}{row[1]:>8}{row[2]:>8}{sum(row):>8}")
    lines.append("-" * 50)
    tot = [int((sp == t).sum()) for t in ("train", "val", "test")]
    lines.append(f"{'TOTAL':<18}{tot[0]:>8}{tot[1]:>8}{tot[2]:>8}{sum(tot):>8}")
    pct = [100 * t / sum(tot) for t in tot]
    lines.append(f"{'share':<18}{pct[0]:>7.1f}%{pct[1]:>7.1f}%{pct[2]:>7.1f}%")
    return "\n".join(lines)
