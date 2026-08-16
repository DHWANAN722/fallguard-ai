"""
cnn_numpy.py — Dependency-free inference runtime for the exported CNN.

This replays the op list written by ``models.export_cnn_npz`` using nothing but
NumPy, so the deployed dashboard needs no TensorFlow. That is not a shortcut:
TensorFlow is a ~600 MB install whose import alone costs several seconds and
which, alongside MediaPipe and OpenCV, does not reliably fit in a free
Streamlit Cloud container. Serving the same trained weights through ~90 lines
of NumPy makes the deployment small, fast to cold-start, and immune to
TF/Keras version drift.

Numerical agreement with Keras is asserted in ``scripts/train.py`` — the export
is rejected if the maximum absolute probability difference exceeds 1e-4.
"""

from __future__ import annotations

import numpy as np


def _same_padding(in_size: int, k: int, stride: int) -> tuple[int, int, int]:
    """TensorFlow's SAME padding, which is *asymmetric* whenever stride > 1.

    Getting this wrong is the classic source of a silent 1-pixel shift between
    a Keras model and a hand-written runtime, so it is computed exactly as
    TensorFlow does rather than assumed to be ``k // 2`` on each side.
    """
    out = -(-in_size // stride)                     # ceil division
    total = max((out - 1) * stride + k - in_size, 0)
    before = total // 2
    return out, before, total - before


def _im2col(x: np.ndarray, kh: int, kw: int, stride: int) -> np.ndarray:
    """(N,H,W,C) → (N, OH, OW, kh*kw*C) patch matrix, TF SAME padding."""
    n, h, w, c = x.shape
    oh, pt, pb = _same_padding(h, kh, stride)
    ow, pl, pr = _same_padding(w, kw, stride)
    xp = np.pad(x, ((0, 0), (pt, pb), (pl, pr), (0, 0)), mode="constant")

    # stride tricks beat an explicit loop by ~40x at 64x64
    s = xp.strides
    view = np.lib.stride_tricks.as_strided(
        xp,
        shape=(n, oh, ow, kh, kw, c),
        strides=(s[0], s[1] * stride, s[2] * stride, s[1], s[2], s[3]),
        writeable=False,
    )
    return view.reshape(n, oh, ow, kh * kw * c)


def conv2d_same(
    x: np.ndarray,
    kernel: np.ndarray,
    bias: np.ndarray | None,
    stride: int = 1,
) -> np.ndarray:
    kh, kw, _cin, cout = kernel.shape
    cols = _im2col(x, kh, kw, stride)               # (N,OH,OW,kh*kw*cin)
    out = cols @ kernel.reshape(-1, cout)
    if bias is not None:
        out += bias
    return out


def batchnorm(x, gamma, beta, mean, var, eps):
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def maxpool(x: np.ndarray, k: int) -> np.ndarray:
    n, h, w, c = x.shape
    h2, w2 = h // k, w // k
    x = x[:, : h2 * k, : w2 * k, :]
    return x.reshape(n, h2, k, w2, k, c).max(axis=(2, 4))


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def run_ops(x: np.ndarray, ops: list[str], w: dict, prefix: str) -> np.ndarray:
    """Replay one exported op sequence over ``x``."""
    for op in ops:
        parts = op.split(":")
        kind, i = parts[0], parts[1]
        key = f"{prefix}{i}"

        if kind == "conv":
            bias = w.get(f"{key}_bias") if parts[2] == "bias" else None
            stride = int(parts[3]) if len(parts) > 3 else 1
            x = conv2d_same(x, w[f"{key}_kernel"], bias, stride)
        elif kind == "bn":
            x = batchnorm(x, w[f"{key}_gamma"], w[f"{key}_beta"],
                          w[f"{key}_mean"], w[f"{key}_var"],
                          float(w[f"{key}_eps"]))
        elif kind == "relu":
            x = np.maximum(x, 0.0)
        elif kind == "maxpool":
            x = maxpool(x, int(parts[2]))
        elif kind == "gap":
            x = x.mean(axis=(1, 2))
        elif kind == "dropout":
            pass                                    # identity at inference
        elif kind == "dense":
            x = x @ w[f"{key}_kernel"] + w[f"{key}_bias"]
            act = parts[2]
            if act == "relu":
                x = np.maximum(x, 0.0)
            elif act == "softmax":
                x = softmax(x)
            elif act != "linear":                   # pragma: no cover
                raise ValueError(f"unsupported dense activation {act!r}")
        else:                                       # pragma: no cover
            raise ValueError(f"unsupported op {op!r}")
    return x


class NumpyCNN:
    """Replay a single-input CNN exported by ``models.export_cnn_npz``."""

    def __init__(self, npz_path: str):
        with np.load(npz_path, allow_pickle=False) as z:
            self.w = {k: z[k] for k in z.files if k != "__ops__"}
            self.ops = [str(o) for o in z["__ops__"]]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """``(N,64,64,3)`` float32 in [0,1] → ``(N,n_classes)`` probabilities."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x[None]
        return run_ops(x, self.ops, self.w, "w").astype(np.float32)


class NumpyHybrid:
    """Replay the two-branch model exported by ``models.export_hybrid_npz``.

    The feature standardiser travels inside the same ``.npz`` as the weights,
    so serving cannot silently drift away from the statistics the network was
    fitted against — a failure that would be invisible in the UI and would
    quietly degrade every prediction.
    """

    def __init__(self, npz_path: str):
        with np.load(npz_path, allow_pickle=False) as z:
            self.w = {k: z[k] for k in z.files if not k.startswith("__ops")}
            self.ops_i = [str(o) for o in z["__ops_i__"]]
            self.ops_f = [str(o) for o in z["__ops_f__"]]
            self.ops_h = [str(o) for o in z["__ops_h__"]]
        self.mean = self.w["feat_mean"]
        self.std = np.maximum(self.w["feat_std"], 1e-6)

    def __call__(self, img, feat):
        return self.predict(img, feat)

    def predict(self, img: np.ndarray, feat: np.ndarray) -> np.ndarray:
        """``(N,64,64,3)`` + ``(N,126)`` → ``(N,n_classes)`` probabilities."""
        img = np.asarray(img, dtype=np.float32)
        feat = np.asarray(feat, dtype=np.float32)
        if img.ndim == 3:
            img = img[None]
        if feat.ndim == 1:
            feat = feat[None]

        feat = (feat - self.mean) / self.std
        a = run_ops(img, self.ops_i, self.w, "i")
        b = run_ops(feat, self.ops_f, self.w, "f")
        return run_ops(np.concatenate([a, b], axis=-1),
                       self.ops_h, self.w, "h").astype(np.float32)
