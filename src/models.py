"""
models.py — Model definitions. **Training-time only** (imports TensorFlow).

The deployed Streamlit app never imports this module; it loads the exported
NumPy weights via ``src.cnn_numpy`` instead. Keeping the two apart is what lets
the dashboard boot in ~2 s inside a 1 GB Streamlit Cloud container that could
not hold TensorFlow at all.

Three models are trained so that the *choice* of CNN is evidenced rather than
asserted (Step 4 of the brief):

    CNN            on 64x64x3 rendered skeleton tensors  ← deployed
    Random Forest  on the 126-dim geometric feature vector
    SVM (RBF)      on the same feature vector, standardised
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------
# CNN
# --------------------------------------------------------------------------
def build_cnn(input_shape=(64, 64, 3), n_classes: int = 5, seed: int = 42):
    """A compact VGG-style CNN sized for the skeleton-tensor domain.

    Design notes
    ------------
    * **Stride-2 stem.** The first convolution downsamples 64→32 immediately.
      Skeleton tensors are sparse line drawings whose information is in the
      *arrangement* of strokes, not in single-pixel detail, so paying full
      resolution through the widest early layers buys nothing. This one change
      cuts training cost ~6x with no measurable accuracy loss, which is what
      makes the model trainable on a CPU-only runtime. (The renderer draws 2 px
      strokes so nothing is lost to the downsample.)
    * Three conv stages at 32/64/128 filters. A deeper, wider backbone would
      overfit this domain long before it helped.
    * BatchNorm after every convolution keeps the sparse activations
      well-conditioned and lets us train at a higher learning rate.
    * GlobalAveragePooling instead of Flatten — it cuts the classifier head
      from ~260 k parameters to 16 k and makes the network far more robust to
      the subject appearing at different scales in frame.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.keras.utils.set_random_seed(seed)

    def conv_bn(x, filters, strides=1):
        x = layers.Conv2D(filters, 3, strides=strides, padding="same",
                          use_bias=False)(x)
        # momentum=0.9, NOT the Keras default of 0.99. With ~72 steps per epoch
        # the default decays its initialisation by only 0.99^72 ≈ 0.49 per
        # epoch, so the moving mean/variance stay dominated by their priors
        # (0 and 1) for hundreds of epochs. Because BatchNorm uses *batch*
        # statistics while training but *moving* statistics at inference, that
        # produces the pathological signature of ~0.93 training accuracy
        # against ~0.25 validation accuracy — the network is fine, its
        # inference-time normalisation is not. At 0.9 the statistics converge
        # within a single epoch.
        x = layers.BatchNormalization(momentum=0.9)(x)
        return layers.ReLU()(x)

    inp = layers.Input(shape=input_shape, name="skeleton_tensor")
    x = conv_bn(inp, 32, strides=2)          # 64 → 32   (cheap stem)
    x = layers.MaxPooling2D(2)(x)            # 32 → 16
    x = conv_bn(x, 64)
    x = conv_bn(x, 64)
    x = layers.MaxPooling2D(2)(x)            # 16 →  8
    x = conv_bn(x, 128)
    x = layers.MaxPooling2D(2)(x)            #  8 →  4
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias=True)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.40, seed=seed)(x)
    out = layers.Dense(n_classes, activation="softmax", name="activity")(x)

    model = models.Model(inp, out, name="FallGuard_CNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_hybrid(
    img_shape=(64, 64, 3),
    n_features: int = 126,
    n_classes: int = 5,
    seed: int = 42,
):
    """Two-branch network: the CNN above, fused with the geometric features.

    Why this exists
    ---------------
    The pure CNN reaches ~91% but concentrates essentially *all* of its error in
    Walking-vs-Standing. That is not a capacity problem, it is an information
    problem: at 64x64 the horizontal gap between the ankles — the cue that
    actually separates a stride from a stance — is a handful of pixels, and the
    stride-2 stem blurs it further. The Random Forest, which receives
    ``ankle_horizontal_split`` as an explicit number, scores ~95% on the very
    same split.

    So rather than choose between them, this model keeps both views:

        image branch    64x64x3 skeleton tensor → CNN → 128-d
        feature branch  126 geometric descriptors → MLP → 64-d
        head            concat(192) → Dense → softmax

    The CNN contributes holistic posture shape; the feature branch contributes
    precise scalar geometry the raster cannot preserve. Each branch is its own
    Sequential sub-model, which keeps the NumPy export walker simple.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    tf.keras.utils.set_random_seed(seed)

    img_branch = models.Sequential(name="img_branch")
    img_branch.add(layers.Input(shape=img_shape))
    for filters, stride, pool in ((32, 2, True), (64, 1, False),
                                  (64, 1, True), (128, 1, True)):
        img_branch.add(layers.Conv2D(filters, 3, strides=stride,
                                     padding="same", use_bias=False))
        img_branch.add(layers.BatchNormalization(momentum=0.9))
        img_branch.add(layers.ReLU())
        if pool:
            img_branch.add(layers.MaxPooling2D(2))
    img_branch.add(layers.GlobalAveragePooling2D())
    img_branch.add(layers.Dense(128))
    img_branch.add(layers.ReLU())

    feat_branch = models.Sequential(name="feat_branch")
    feat_branch.add(layers.Input(shape=(n_features,)))
    feat_branch.add(layers.Dense(128))
    feat_branch.add(layers.BatchNormalization(momentum=0.9))
    feat_branch.add(layers.ReLU())
    feat_branch.add(layers.Dropout(0.25, seed=seed))
    feat_branch.add(layers.Dense(64))
    feat_branch.add(layers.ReLU())

    head = models.Sequential(name="head")
    head.add(layers.Input(shape=(192,)))
    head.add(layers.Dense(128))
    head.add(layers.ReLU())
    head.add(layers.Dropout(0.35, seed=seed))
    head.add(layers.Dense(n_classes, activation="softmax"))

    img_in = layers.Input(shape=img_shape, name="skeleton_tensor")
    feat_in = layers.Input(shape=(n_features,), name="geometry")
    fused = layers.Concatenate()([img_branch(img_in), feat_branch(feat_in)])
    out = head(fused)

    model = models.Model([img_in, feat_in], out, name="FallGuard_Hybrid")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------
# classical baselines
# --------------------------------------------------------------------------
def train_random_forest(Xtr, ytr, seed: int = 42):
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(
        n_estimators=300, max_depth=18, min_samples_leaf=2,
        n_jobs=-1, random_state=seed, class_weight="balanced",
    )
    clf.fit(Xtr, ytr)
    return clf


def train_svm(Xtr, ytr, seed: int = 42):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    clf = make_pipeline(
        StandardScaler(),
        SVC(C=6.0, gamma="scale", kernel="rbf", probability=True,
            random_state=seed, class_weight="balanced"),
    )
    clf.fit(Xtr, ytr)
    return clf


# --------------------------------------------------------------------------
# export the trained CNN to plain NumPy arrays for dependency-free serving
# --------------------------------------------------------------------------
def _export_layers(model_layers, arrays: dict, prefix: str) -> list[str]:
    """Serialise a flat layer sequence into ``arrays`` and return its op list.

    The exported graph is described by these small op strings, which
    ``src.cnn_numpy`` replays. Only the ops these architectures actually use are
    supported — deliberately, since a general ONNX-style runtime would be far
    more code than the problem needs.
    """
    from tensorflow.keras import layers as L

    ops: list[str] = []
    idx = 0

    for layer in model_layers:
        if isinstance(layer, L.InputLayer):
            continue
        if isinstance(layer, L.Conv2D):
            if layer.padding != "same":                        # pragma: no cover
                raise ValueError("only padding='same' convolutions are exportable")
            w = layer.get_weights()
            arrays[f"{prefix}{idx}_kernel"] = w[0].astype(np.float32)
            if len(w) > 1:
                arrays[f"{prefix}{idx}_bias"] = w[1].astype(np.float32)
            stride = int(layer.strides[0])
            ops.append(f"conv:{idx}:{'bias' if len(w) > 1 else 'nobias'}:{stride}")
        elif isinstance(layer, L.BatchNormalization):
            g, b, m, v = layer.get_weights()
            arrays[f"{prefix}{idx}_gamma"] = g.astype(np.float32)
            arrays[f"{prefix}{idx}_beta"] = b.astype(np.float32)
            arrays[f"{prefix}{idx}_mean"] = m.astype(np.float32)
            arrays[f"{prefix}{idx}_var"] = v.astype(np.float32)
            arrays[f"{prefix}{idx}_eps"] = np.float32(layer.epsilon)
            ops.append(f"bn:{idx}")
        elif isinstance(layer, L.ReLU):
            ops.append(f"relu:{idx}")
        elif isinstance(layer, L.MaxPooling2D):
            ops.append(f"maxpool:{idx}:{layer.pool_size[0]}")
        elif isinstance(layer, L.GlobalAveragePooling2D):
            ops.append(f"gap:{idx}")
        elif isinstance(layer, L.Dropout):
            ops.append(f"dropout:{idx}")          # inference no-op, kept for clarity
        elif isinstance(layer, L.Dense):
            w = layer.get_weights()
            arrays[f"{prefix}{idx}_kernel"] = w[0].astype(np.float32)
            arrays[f"{prefix}{idx}_bias"] = w[1].astype(np.float32)
            act = getattr(layer.activation, "__name__", "linear")
            ops.append(f"dense:{idx}:{act}")
        else:                                      # pragma: no cover
            raise TypeError(f"cannot serialise {type(layer).__name__}")
        idx += 1

    return ops


def export_cnn_npz(model, path: str) -> dict:
    """Export a single-input Keras CNN to ``.npz`` for ``NumpyCNN``."""
    arrays: dict[str, np.ndarray] = {}
    ops = _export_layers(model.layers, arrays, "w")
    arrays["__ops__"] = np.array(ops, dtype="<U40")
    np.savez_compressed(path, **arrays)
    return arrays


def export_hybrid_npz(model, feat_mean, feat_std, path: str) -> dict:
    """Export the two-branch model to ``.npz`` for ``NumpyHybrid``.

    Each of the three Sequential sub-models is serialised under its own key
    prefix, and the feature standardiser is baked in alongside the weights so
    the serving runtime needs no scikit-learn and cannot drift out of sync with
    the statistics the network was trained against.
    """
    arrays: dict[str, np.ndarray] = {}
    for name, prefix in (("img_branch", "i"), ("feat_branch", "f"), ("head", "h")):
        sub = model.get_layer(name)
        ops = _export_layers(sub.layers, arrays, prefix)
        arrays[f"__ops_{prefix}__"] = np.array(ops, dtype="<U40")

    arrays["feat_mean"] = np.asarray(feat_mean, dtype=np.float32)
    arrays["feat_std"] = np.asarray(feat_std, dtype=np.float32)
    np.savez_compressed(path, **arrays)
    return arrays
