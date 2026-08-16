#!/usr/bin/env python3
"""
train.py — Train, evaluate, compare and export every FallGuard model.

    python scripts/train.py                       # train to completion
    python scripts/train.py --epochs 60
    python scripts/train.py --time-budget 30      # resumable chunks (see below)

Resumability
------------
The script checkpoints after every chunk of training and can be re-invoked to
continue exactly where it left off — optimiser state, epoch counter, history,
and early-stopping bookkeeping all persist to ``models/train_state.json``.

That exists because this project was developed in an environment that caps any
single process at well under a minute of wall clock. ``--time-budget N`` trains
until N seconds have elapsed, saves, and exits cleanly; running the command in a
loop converges to the same result as one long run. On a normal machine or in
Colab you can ignore the flag entirely and it trains straight through.

Artefacts produced (under ``models/`` and ``reports/``):

    models/fallguard_cnn.keras          Keras model  (archival)
    models/fallguard_cnn.npz            NumPy weights (what the app serves)
    models/labels.json                  class order + serving metadata
    reports/metrics.json                every number quoted in the report
    reports/confusion_matrix.png        CNN, test split, counts + row-normalised
    reports/training_curves.png         accuracy and loss vs epoch
    reports/model_comparison.png        CNN vs Random Forest vs SVM
    reports/per_class_metrics.png       precision / recall / F1 per class
    reports/classification_report.txt   sklearn text report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

T_START = time.time()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from src import dataset as ds                                    # noqa: E402
from src.features import extract_batch                           # noqa: E402
from src.render import render_batch                              # noqa: E402
from src.skeleton import CLASS_NAMES, FALL                       # noqa: E402

MODELS = os.path.join(ROOT, "models")
REPORTS = os.path.join(ROOT, "reports")
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(DATA, "cache")

CKPT = os.path.join(MODELS, "checkpoint.keras")
STATE = os.path.join(MODELS, "train_state.json")

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#0b0f1a", "axes.facecolor": "#11162a",
        "text.color": "#e8ecff", "axes.labelcolor": "#e8ecff",
        "xtick.color": "#9fb0d9", "ytick.color": "#9fb0d9",
        "axes.edgecolor": "#2a3358", "grid.color": "#1e2749",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    })
    return plt


NEON = ["#ff2d78", "#00e5ff", "#8cff2b", "#ffb300", "#b26bff"]


def build_inputs(n_per_class: int, dataset: str | None = None) -> tuple[dict, dict, dict, dict]:
    """Load or build the corpus, then load or compute the model inputs.

    Rendered tensors are cached to disk as **uint8** rather than float32. That
    is a 4x memory saving (447 MB → 112 MB for the training split) and it makes
    re-invocation cheap, which matters when training is resumed many times.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = dataset or os.path.join(DATA, "fallguard_dataset.npz")
    if not os.path.exists(path):
        if dataset:
            raise SystemExit(f"dataset not found: {path}\n"
                             "build one with scripts/ingest_kaggle.py first")
        print("building corpus ...", flush=True)
        ds.save(ds.build_corpus(n_per_class), path)
    corpus = ds.load(path)

    # the render/feature cache is keyed to a specific corpus; a different
    # dataset must not silently reuse the previous one's tensors
    stamp = os.path.join(CACHE, "source.txt")
    prev = open(stamp).read().strip() if os.path.exists(stamp) else ""
    if prev != os.path.abspath(path):
        for f in os.listdir(CACHE):
            os.remove(os.path.join(CACHE, f))
        with open(stamp, "w") as fh:
            fh.write(os.path.abspath(path))

    R: dict[str, np.ndarray] = {}
    F: dict[str, np.ndarray] = {}
    Y: dict[str, np.ndarray] = {}

    need = any(not os.path.exists(os.path.join(CACHE, f"R_{t}.npy")) for t in SPLITS)
    if need:
        print("rendering skeleton tensors and extracting features ...", flush=True)
        t0 = time.time()
        for t in SPLITS:
            P, V, y = ds.split_arrays(corpus, t)
            np.save(os.path.join(CACHE, f"R_{t}.npy"),
                    (render_batch(P, V) * 255).astype(np.uint8))
            np.save(os.path.join(CACHE, f"F_{t}.npy"), extract_batch(P, V))
            np.save(os.path.join(CACHE, f"Y_{t}.npy"), y)
        print(f"  cached in {time.time() - t0:.1f}s", flush=True)

    for t in SPLITS:
        R[t] = np.load(os.path.join(CACHE, f"R_{t}.npy"), mmap_mode="r")
        F[t] = np.load(os.path.join(CACHE, f"F_{t}.npy"))
        Y[t] = np.load(os.path.join(CACHE, f"Y_{t}.npy"))
    return corpus, R, F, Y


def as_float(a: np.ndarray) -> np.ndarray:
    """uint8 cache → float32 in [0,1] for Keras."""
    return np.asarray(a, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------
def finalize(model, corpus, R, F, Y, state, args) -> None:
    """Evaluate, export, plot and write every artefact. Runs once, at the end."""
    import tensorflow as tf                                       # noqa: F401
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score, precision_score,
                                 recall_score)

    from src.cnn_numpy import NumpyHybrid
    from src.models import export_hybrid_npz, train_random_forest, train_svm

    os.makedirs(REPORTS, exist_ok=True)

    keras_path = os.path.join(MODELS, "fallguard_cnn.keras")
    model.save(keras_path)
    npz_path = os.path.join(MODELS, "fallguard_cnn.npz")
    export_hybrid_npz(model, state["feat_mean"], state["feat_std"], npz_path)

    # --- verify the NumPy runtime reproduces Keras exactly ----------------
    runtime = NumpyHybrid(npz_path)
    pi, pf = as_float(R["test"][:256]), F["test"][:256]
    ref = model.predict([pi, (pf - np.asarray(state["feat_mean"], np.float32))
                         / np.maximum(np.asarray(state["feat_std"], np.float32), 1e-6)],
                        verbose=0)
    delta = float(np.abs(runtime.predict(pi, pf) - ref).max())
    print(f"\nNumPy-vs-Keras max |Δprob| = {delta:.2e}", flush=True)
    if delta > 1e-4:
        raise SystemExit(f"export verification FAILED (Δ={delta:.2e})")
    print("export verified ✓", flush=True)

    results: dict[str, dict] = {}

    def score(name, ytrue, ypred, secs):
        results[name] = {
            "accuracy": float(accuracy_score(ytrue, ypred)),
            "precision_macro": float(precision_score(ytrue, ypred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(ytrue, ypred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(ytrue, ypred, average="macro", zero_division=0)),
            "fall_precision": float(precision_score(ytrue, ypred, labels=[FALL], average="micro", zero_division=0)),
            "fall_recall": float(recall_score(ytrue, ypred, labels=[FALL], average="micro", zero_division=0)),
            "train_seconds": round(secs, 1),
        }
        r = results[name]
        print(f"{name:<16} acc {r['accuracy']:.4f}  F1 {r['f1_macro']:.4f}  "
              f"fall-recall {r['fall_recall']:.4f}", flush=True)

    print("\n--- test-set performance ---", flush=True)
    cnn_pred = runtime.predict(as_float(R["test"]), F["test"]).argmax(1)
    score("Hybrid CNN", Y["test"], cnn_pred, state["seconds"])

    t0 = time.time()
    rf = train_random_forest(F["train"], Y["train"], args.seed)
    score("RandomForest", Y["test"], rf.predict(F["test"]), time.time() - t0)

    if not args.skip_svm:
        t0 = time.time()
        sub = slice(None, 4000)          # RBF SVM is O(n^2); subsample to stay sane
        svm = train_svm(F["train"][sub], Y["train"][sub], args.seed)
        score("SVM (RBF)", Y["test"], svm.predict(F["test"]), time.time() - t0)

    cm = confusion_matrix(Y["test"], cnn_pred)
    txt = classification_report(Y["test"], cnn_pred, target_names=CLASS_NAMES, digits=4)
    print("\n" + txt, flush=True)
    with open(os.path.join(REPORTS, "classification_report.txt"), "w") as fh:
        fh.write(txt)

    plt = _plt()

    # confusion matrix — counts and row-normalised side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    cmn = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    for ax, M, title, fmt in (
        (axes[0], cm, "Confusion Matrix — counts", "d"),
        (axes[1], cmn, "Confusion Matrix — row-normalised (recall)", ".2f"),
    ):
        im = ax.imshow(M, cmap="magma", vmin=0)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=32, ha="right")
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.set_title(title)
        thr = M.max() * 0.55
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, format(M[i, j], fmt), ha="center", va="center",
                        color="white" if M[i, j] < thr else "black",
                        fontsize=10, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("FallGuard AI · CNN · test split", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "confusion_matrix.png"), dpi=140)
    plt.close(fig)

    # training curves
    h = state["history"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, keys, title, ylab in (
        (axes[0], ("accuracy", "val_accuracy"), "Model Accuracy", "accuracy"),
        (axes[1], ("loss", "val_loss"), "Model Loss", "loss"),
    ):
        ax.plot(h[keys[0]], color=NEON[1], lw=2.2, label="train")
        ax.plot(h[keys[1]], color=NEON[0], lw=2.2, label="validation")
        ax.set_title(title); ax.set_xlabel("epoch"); ax.set_ylabel(ylab)
        ax.grid(alpha=0.3); ax.legend(facecolor="#11162a", edgecolor="#2a3358")
    fig.suptitle("FallGuard AI · CNN training history", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "training_curves.png"), dpi=140)
    plt.close(fig)

    # model comparison
    names = list(results)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    w = 0.8 / len(metrics)
    xs = np.arange(len(names))
    for i, m in enumerate(metrics):
        vals = [results[n][m] for n in names]
        b = ax.bar(xs + i * w - 0.4 + w / 2, vals, w, label=m.replace("_", " "),
                   color=NEON[i], edgecolor="#0b0f1a")
        ax.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylim(0, 1.14); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Model comparison — held-out test split")
    ax.legend(ncol=4, facecolor="#11162a", edgecolor="#2a3358", loc="upper center")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "model_comparison.png"), dpi=140)
    plt.close(fig)

    # per-class precision / recall / F1 for the deployed CNN
    p = precision_score(Y["test"], cnn_pred, average=None, zero_division=0)
    r = recall_score(Y["test"], cnn_pred, average=None, zero_division=0)
    f = f1_score(Y["test"], cnn_pred, average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    xs = np.arange(len(CLASS_NAMES))
    for i, (vals, lab) in enumerate(((p, "precision"), (r, "recall"), (f, "F1"))):
        b = ax.bar(xs + i * 0.26 - 0.26, vals, 0.26, label=lab,
                   color=NEON[i], edgecolor="#0b0f1a")
        ax.bar_label(b, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(xs); ax.set_xticklabels(CLASS_NAMES, rotation=12)
    ax.set_ylim(0, 1.14); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Per-class performance — deployed CNN")
    ax.legend(facecolor="#11162a", edgecolor="#2a3358")
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS, "per_class_metrics.png"), dpi=140)
    plt.close(fig)

    meta = {
        "classes": CLASS_NAMES,
        "fall_index": FALL,
        "input_shape": list(R["train"].shape[1:]),
        "n_parameters": int(model.count_params()),
        "epochs_run": state["epoch"],
        "best_val_accuracy": state["best"],
        "numpy_keras_max_delta": delta,
        "train_seconds": round(state["seconds"], 1),
        "dataset": {
            "total": int(len(corpus["labels"])),
            "split": {t: int((corpus["split"] == t).sum()) for t in SPLITS},
            "source": str(corpus["source"][0]),
        },
    }
    with open(os.path.join(MODELS, "labels.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    with open(os.path.join(REPORTS, "metrics.json"), "w") as fh:
        json.dump({
            "models": results,
            "confusion_matrix": cm.tolist(),
            "per_class": {n: {"precision": float(p[i]), "recall": float(r[i]),
                              "f1": float(f[i])}
                          for i, n in enumerate(CLASS_NAMES)},
            "history": h,
            "meta": meta,
        }, fh, indent=2)

    print(f"\nartefacts written to {MODELS} and {REPORTS}", flush=True)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--n-per-class", type=int, default=2600)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--time-budget", type=float, default=0.0,
                    help="seconds of wall clock before checkpointing and exiting; "
                         "0 = train straight through")
    ap.add_argument("--skip-svm", action="store_true")
    ap.add_argument("--reset", action="store_true", help="discard any checkpoint")
    ap.add_argument("--dataset", default=None,
                    help="path to a corpus .npz (e.g. one built by "
                         "scripts/ingest_kaggle.py); defaults to the synthetic corpus")
    args = ap.parse_args()

    for d in (MODELS, REPORTS, DATA):
        os.makedirs(d, exist_ok=True)
    if args.reset:
        for p in (CKPT, STATE):
            if os.path.exists(p):
                os.remove(p)

    corpus, R, F, Y = build_inputs(args.n_per_class, args.dataset)
    print(ds.describe(corpus), flush=True)

    import tensorflow as tf
    from src.models import build_hybrid

    # ------------------------------------------------------------- resume
    if os.path.exists(CKPT) and os.path.exists(STATE):
        model = tf.keras.models.load_model(CKPT)
        with open(STATE) as fh:
            state = json.load(fh)
        print(f"\nresumed from epoch {state['epoch']} "
              f"(best val_accuracy {state['best']:.4f})", flush=True)
    else:
        model = build_hybrid(img_shape=R["train"].shape[1:],
                             n_features=F["train"].shape[1],
                             n_classes=len(CLASS_NAMES), seed=args.seed)
        model.summary()
        # standardiser fitted on the TRAINING split only, then frozen — fitting
        # it on all data would leak validation statistics into training
        mu = F["train"].mean(axis=0)
        sd = F["train"].std(axis=0)
        state = {"epoch": 0, "best": -1.0, "wait": 0, "seconds": 0.0, "done": False,
                 "feat_mean": mu.tolist(), "feat_std": sd.tolist(),
                 "history": {k: [] for k in
                             ("accuracy", "loss", "val_accuracy", "val_loss", "lr")}}

    if state["done"]:
        print("\ntraining already complete — finalising only", flush=True)
        finalize(model, corpus, R, F, Y, state, args)
        return

    mu = np.asarray(state["feat_mean"], dtype=np.float32)
    sd = np.maximum(np.asarray(state["feat_std"], dtype=np.float32), 1e-6)
    Rtr, Rva = as_float(R["train"]), as_float(R["val"])
    Ftr, Fva = (F["train"] - mu) / sd, (F["val"] - mu) / sd
    ytr, yva = Y["train"], Y["val"]

    # ------------------------------------------------------------- train
    print(f"\ntraining from epoch {state['epoch']} → {args.epochs}", flush=True)
    while state["epoch"] < args.epochs:
        if args.time_budget and (time.time() - T_START) > args.time_budget:
            print(f"\ntime budget reached at epoch {state['epoch']} — "
                  f"checkpointing; re-run to continue", flush=True)
            break

        t0 = time.time()
        h = model.fit([Rtr, Ftr], ytr, validation_data=([Rva, Fva], yva),
                      epochs=1, batch_size=args.batch_size, verbose=0)
        state["seconds"] += time.time() - t0
        state["epoch"] += 1

        acc = float(h.history["accuracy"][0])
        loss = float(h.history["loss"][0])
        vacc = float(h.history["val_accuracy"][0])
        vloss = float(h.history["val_loss"][0])
        # Keras 3 exposes the LR as a plain Variable; the legacy
        # backend.get_value/set_value helpers break on a reloaded optimizer.
        lr = float(np.asarray(model.optimizer.learning_rate))
        for k, v in (("accuracy", acc), ("loss", loss), ("val_accuracy", vacc),
                     ("val_loss", vloss), ("lr", lr)):
            state["history"][k].append(v)

        # --- manual early stopping + LR schedule, persisted across runs ----
        improved = vacc > state["best"] + 1e-4
        if improved:
            state["best"] = vacc
            state["wait"] = 0
            model.save(os.path.join(MODELS, "best.keras"))
        else:
            state["wait"] += 1
            if state["wait"] in (3, 5, 7) and lr > 1.1e-5:
                model.optimizer.learning_rate.assign(lr * 0.5)
                print(f"  lr → {lr * 0.5:.2e}", flush=True)

        print(f"epoch {state['epoch']:>3}/{args.epochs}  "
              f"acc {acc:.4f}  loss {loss:.4f}  "
              f"val_acc {vacc:.4f}  val_loss {vloss:.4f}"
              f"{'  ★' if improved else ''}", flush=True)

        if state["wait"] >= args.patience:
            print(f"\nearly stopping — no improvement for {args.patience} epochs",
                  flush=True)
            state["epoch"] = args.epochs
            break

    model.save(CKPT)
    finished = state["epoch"] >= args.epochs
    state["done"] = finished
    with open(STATE, "w") as fh:
        json.dump(state, fh)

    if finished:
        best = os.path.join(MODELS, "best.keras")
        if os.path.exists(best):
            model = tf.keras.models.load_model(best)     # restore best weights
            print(f"\nrestored best weights (val_accuracy {state['best']:.4f})",
                  flush=True)
        finalize(model, corpus, R, F, Y, state, args)
    else:
        print(f"\ncheckpointed at epoch {state['epoch']}/{args.epochs} "
              f"({time.time() - T_START:.0f}s elapsed)", flush=True)


if __name__ == "__main__":
    main()
