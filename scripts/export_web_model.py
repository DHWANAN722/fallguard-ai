#!/usr/bin/env python3
"""
export_web_model.py — Export the trained hybrid model for in-browser inference.

    python scripts/export_web_model.py

Writes ``models/fallguard_web.json``: the same weights the Python runtime uses,
rearranged so a few hundred lines of JavaScript can run them at video framerate
in the user's browser, with no server round-trip.

Two transformations happen here.

**BatchNorm is folded into the preceding layer.** For ``y = gamma * (Wx - mean)
/ sqrt(var + eps) + beta`` the whole normalisation collapses into a per-output
scale and offset::

    s = gamma / sqrt(var + eps)
    W' = W * s          (broadcast over the output-channel axis)
    b' = beta - mean * s

so inference becomes plain convolution-plus-bias. That removes an entire op
from the JavaScript and one pass over every feature map.

**Weights are stored as float16.** The payload is embedded in the page, so
774 KB of float32 becomes 387 KB — and inference at this size is nowhere near
precision-limited; the verification below measures the actual cost.

A **golden test case** is embedded alongside the weights: one fixed skeleton and
the probabilities the Python model produces for it. The browser runs that on
load and reports agreement to the console, so a porting mistake announces
itself instead of silently degrading every prediction.
"""

from __future__ import annotations

import base64
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODELS = os.path.join(ROOT, "models")
SRC = os.path.join(MODELS, "fallguard_cnn.npz")
OUT = os.path.join(MODELS, "fallguard_web.json")


def fold_bn(kernel, bias, gamma, beta, mean, var, eps):
    """Fold BatchNorm into the preceding conv/dense weights."""
    s = gamma / np.sqrt(var + eps)
    return kernel * s, (bias if bias is not None else 0.0) * s + (beta - mean * s)


def main() -> None:
    from src.cnn_numpy import NumpyHybrid
    from src.features import N_FEATURES, extract
    from src.render import render_cnn
    from src.skeleton import CLASS_NAMES, generate_sample

    z = dict(np.load(SRC, allow_pickle=False))
    ops = {p: [str(o) for o in z[f"__ops_{p}__"]] for p in ("i", "f", "h")}

    layers: list[dict] = []
    blobs: list[np.ndarray] = []

    def add(kind, W, b, **extra):
        layers.append({"kind": kind, "shape": list(W.shape),
                       "w": len(blobs), "b": len(blobs) + 1, **extra})
        blobs.append(np.asarray(W, np.float32).reshape(-1))
        blobs.append(np.asarray(b, np.float32).reshape(-1))

    # ---- walk each branch, folding BN into whatever preceded it ----------
    for prefix in ("i", "f", "h"):
        pending = None                     # (kind, W, b, extra) awaiting a BN
        for op in ops[prefix]:
            part = op.split(":")
            kind, idx = part[0], part[1]

            if kind == "conv":
                W = z[f"{prefix}{idx}_kernel"]
                b = z.get(f"{prefix}{idx}_bias", np.zeros(W.shape[-1], np.float32))
                pending = ("conv", W, b, {"stride": int(part[3]) if len(part) > 3 else 1})
            elif kind == "dense":
                W = z[f"{prefix}{idx}_kernel"]
                b = z[f"{prefix}{idx}_bias"]
                pending = ("dense", W, b, {"act": part[2]})
            elif kind == "bn":
                assert pending is not None, "BatchNorm with nothing before it"
                k, W, b, extra = pending
                W, b = fold_bn(W, b, z[f"{prefix}{idx}_gamma"], z[f"{prefix}{idx}_beta"],
                               z[f"{prefix}{idx}_mean"], z[f"{prefix}{idx}_var"],
                               float(z[f"{prefix}{idx}_eps"]))
                pending = (k, W, b, extra)
            else:
                if pending is not None:
                    k, W, b, extra = pending
                    add(k, W, b, branch=prefix, **extra)
                    pending = None
                if kind == "relu":
                    layers.append({"kind": "relu", "branch": prefix})
                elif kind == "maxpool":
                    layers.append({"kind": "maxpool", "branch": prefix,
                                   "k": int(part[2])})
                elif kind == "gap":
                    layers.append({"kind": "gap", "branch": prefix})
                elif kind == "dropout":
                    pass                    # identity at inference
                else:                       # pragma: no cover
                    raise ValueError(f"unsupported op {op!r}")
        if pending is not None:
            k, W, b, extra = pending
            add(k, W, b, branch=prefix, **extra)

    # ---- pack every blob into one float16 buffer -------------------------
    flat = np.concatenate(blobs).astype(np.float16)
    offsets, pos = [], 0
    for blob in blobs:
        offsets.append([pos, blob.size])
        pos += blob.size

    for layer in layers:
        if "w" in layer:
            layer["w"] = offsets[layer["w"]]
            layer["b"] = offsets[layer["b"]]

    # ---- golden tests --------------------------------------------------
    # Deliberately chosen to be AMBIGUOUS. A saturated 1.0/0.0 case is a weak
    # test: a port with a transposed kernel or a wrong padding offset would
    # still land on the right argmax and look fine. Cases where the model is
    # genuinely undecided exercise the arithmetic, so a small error moves the
    # numbers visibly.
    # Each case carries the OpenCV-rendered tensor as well as the landmarks,
    # because the browser has to verify two independent things and conflating
    # them would make a failure unreadable:
    #
    #   arithmetic — feed the embedded tensor straight in and the probabilities
    #                must match to float16 rounding. Ambiguity is a virtue
    #                here: a transposed kernel moves a 0.53/0.47 case visibly.
    #
    #   rasterise  — JavaScript has no OpenCV, so the browser reproduces
    #                cv2's antialiased stroke arithmetically. That is compared
    #                against this same tensor as a mean pixel difference, on a
    #                tolerance, because it is an approximation by construction.
    #
    # Judging the raster by these deliberately knife-edge probabilities would
    # be meaningless: they are chosen to be maximally sensitive to any input
    # perturbation, so a 2-in-1000 pixel difference can flip them.
    runtime = NumpyHybrid(SRC)
    rng = np.random.default_rng(20260817)
    golden = []
    for cls in (0, 1, 3, 4):
        best = None
        for _ in range(400):
            P, V = generate_sample(cls, rng)
            T, F = render_cnn(P, V), extract(P, V)
            pr = runtime.predict(T[None], F[None])[0]
            spread = float(np.sort(pr)[-1] - np.sort(pr)[-2])
            if best is None or spread < best[0]:
                best = (spread, P, V, T, F, pr)
        _, P, V, T, F, pr = best
        # render_cnn quantises to 8-bit, so uint8 storage is lossless
        u8 = np.round(T * 255.0).astype(np.uint8)
        assert np.abs(u8.astype(np.float32) / 255.0 - T).max() < 1e-6
        golden.append({
            "landmarks": P.astype(float).round(6).tolist(),
            "visibility": V.astype(float).round(6).tolist(),
            "features": F.astype(float).round(6).tolist(),
            "tensor_u8_b64": base64.b64encode(u8.tobytes()).decode(),
            "expected": pr.astype(float).round(8).tolist(),
        })

    payload = {
        "classes": CLASS_NAMES,
        "n_features": int(N_FEATURES),
        "layers": layers,
        "feat_mean": z["feat_mean"].astype(np.float32).tolist(),
        "feat_std": np.maximum(z["feat_std"], 1e-6).astype(np.float32).tolist(),
        "weights_f16_b64": base64.b64encode(flat.tobytes()).decode(),
        "golden": golden,
    }

    with open(OUT, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))

    # ---- report what the float16 rounding actually costs -----------------
    deg = np.abs(flat.astype(np.float32) - np.concatenate(blobs)).max()
    print(f"layers          {len(layers)}")
    print(f"scalars         {flat.size:,}")
    print(f"float16 buffer  {flat.nbytes / 1024:.0f} KB "
          f"({len(payload['weights_f16_b64']) / 1024:.0f} KB base64)")
    print(f"file            {os.path.getsize(OUT) / 1024:.0f} KB")
    print(f"max weight error from float16 rounding: {deg:.2e}")
    print(f"golden cases    {len(golden)} (chosen for minimum margin)")
    for g in golden:
        pr = np.array(g["expected"])
        top = np.argsort(pr)[::-1][:2]
        print(f"   {CLASS_NAMES[top[0]]:<16} {pr[top[0]]:.4f}   vs   "
              f"{CLASS_NAMES[top[1]]:<16} {pr[top[1]]:.4f}")


if __name__ == "__main__":
    main()
