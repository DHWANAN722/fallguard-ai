# Developing an AI-Powered Elderly Fall Detection System

**Formative Assessment 2 — Building and Deploying the Model**
CRS Artificial Intelligence · Y2C1 Machine Learning and Deep Learning

---

## Executive summary

FallGuard AI detects elderly falls from images and video by reducing each frame
to 33 body landmarks with MediaPipe BlazePose and classifying the resulting
skeleton with a hybrid convolutional network. On a held-out test split of 3 000
samples it reaches **98.37 % accuracy** with **1.000 precision and 1.000 recall
on the fall class**, and raises **zero false alarms** on bending — the posture
that defeats most deployed fall detectors. The system is live as a Streamlit
dashboard offering image analysis, video monitoring, scenario simulation and
in-app evaluation evidence.

The central design argument of this report is that **a fall detector is not an
accuracy problem, it is a false-alarm problem**. That claim shapes every
decision below.

---

## Step 4 — Model Selection

### 4.1 Pose estimation

| Candidate | Assessment | Decision |
|---|---|---|
| **MediaPipe BlazePose** | 33 landmarks including feet and heels, per-landmark visibility scores, real-time on CPU, weights ship inside the pip wheel, permissive licence. | **Selected** |
| YOLOv8-Pose | Excellent accuracy, native multi-person. But only 17 COCO keypoints (no foot/heel detail), heavier runtime, and an AGPL licence that complicates deployment. | Rejected |
| OpenPose | Strong multi-person accuracy. Large model, effectively GPU-bound, awkward build. | Rejected |

Two properties decided it. First, **per-landmark visibility scores**: they let
the system express uncertainty under occlusion rather than silently guessing,
and the dashboard surfaces that uncertainty to the caregiver. Second, **no
cold-start download** — the 0.10.x wheels bundle the model, which matters on a
free cloud host where a first-request network fetch is a real failure mode.

### 4.2 Activity classification

The brief recommends a CNN. Rather than accept that on faith, three models were
trained on identical splits:

| Model | Input | Accuracy | Macro F1 | Fall recall |
|---|---|---|---|---|
| **Hybrid CNN** | skeleton tensor **+** geometric features | **0.9837** | **0.9837** | 1.000 |
| Random Forest | 126 geometric features | 0.9827 | 0.9827 | 1.000 |
| SVM (RBF) | 126 geometric features | 0.9533 | 0.9535 | 1.000 |

The deployed model is a **two-branch hybrid**: a convolutional branch over a
64×64×3 rendered skeleton tensor, fused with a dense branch over 126 explicit
geometric descriptors.

This architecture was not the first attempt, and the reason it exists is
instructive. A pure CNN reached 91.1 % and concentrated essentially *all* of its
error in Walking↔Standing (155 of 174 misclassifications). The Random Forest —
which receives `ankle_horizontal_split` as an explicit number — scored 94.7 % on
the identical split. The CNN was not short of capacity; it was short of
**precision in one specific measurement**, because at 64×64 the horizontal gap
between the ankles is a handful of pixels and the stride-2 stem blurs it
further. Rather than choose between the two views, the hybrid keeps both, and
now beats both parents.

### 4.3 Why classify a skeleton rather than the photograph

1. **Privacy.** A care-home monitor that ships resident imagery to a model is a
   serious data-protection problem. Reducing every frame to 33 coordinates
   before classification means no identifiable image reaches the model or needs
   to be stored. This is a structural guarantee, not a policy promise.
2. **Generalisation.** A pixel CNN would latch onto room decor, lighting and
   clothing. Geometry transfers across rooms.
3. **Data efficiency.** The skeleton domain is learnable from thousands rather
   than millions of examples.

### 4.4 Fall detection logic

Classification alone is insufficient for a clinical decision. The deployed logic
has three stages:

**Two-tier corroboration.** An independent, fully transparent geometric rule
scores each skeleton on trunk inclination, bounding-box aspect ratio, pelvis
height and leg verticality — the four descriptors used throughout the
fall-detection literature. The CNN and the rule must **agree** before any alert
is raised. Because they fail differently, agreement is much stronger evidence
than either alone.

**Temporal persistence.** On video, escalation to EMERGENCY requires the
corroborated state to persist across consecutive frames.

**Impact velocity.** Pelvis descent is tracked in frame-heights per second. A
rapid drop is the signature of a genuine fall and escalates immediately —
distinguishing *falling* from *already sitting on the floor*.

Alert levels: `NORMAL` → `WATCH` (one detector fired; logged, nobody paged) →
`ALERT` (both agree) → `EMERGENCY` (sustained, or agreement plus impact).

---

## Step 5 — Model Design and Training

### 5.1 Dataset and provenance

**Disclosure.** Kaggle was not reachable from the environment this project was
built in. The training corpus is therefore **procedurally generated**: a 2D
forward-kinematic model of the human body, driven by joint-angle distributions
taken from the gait and fall-biomechanics literature, then degraded by the
nuisance factors that break real deployments:

* camera roll (imperfect mounting) — σ ≈ 7.5°
* camera yaw / horizontal foreshortening — ±62°
* pose-estimator landmark jitter — σ ∈ [0.003, 0.016] of frame
* occlusion — 28 % of samples lose a contiguous body group to furniture
* anthropometric variation — per-subject segment lengths
* scale and position variation in frame

This is a disclosed engineering choice, not a claim of real data. It is
defensible because the classifier consumes **normalised landmark geometry** —
precisely the representation MediaPipe emits from a real photograph — so the
domain gap is far narrower than it would be for a pixel model. The deployed app
runs on genuine MediaPipe output from whatever the user uploads.
`scripts/ingest_kaggle.py` converts any real labelled image dataset into the
identical schema, so the corpus can be swapped or blended and everything
downstream retrains unchanged.

### 5.2 The five classes

| Class | Definition |
|---|---|
| Fall Detected | Prone, collapsed or sideways on the floor. Trunk near-horizontal, pelvis low in frame. |
| Walking | Mid-gait with observable stride. |
| Sitting | Pelvis at seat height, thighs ~horizontal, shins vertical. |
| Standing | Upright, legs beneath the body. |
| Normal Activity | **Bending / reaching / stooping** — deep trunk flexion over extended, vertical legs, pelvis still at standing height. |

The fifth class is the crux. It shares a near-horizontal trunk with a fall while
being entirely benign, so a detector thresholding on trunk angle alone fires on
it constantly. Including it as a first-class category forces the model to learn
pelvis height and leg configuration as well.

### 5.3 Data split

Stratified **within each class**, so validation and test remain perfectly
balanced and per-class recall is measured over equal support:

| Split | Samples | Share |
|---|---|---|
| Training | 14 000 | 70 % |
| Validation | 3 000 | 15 % |
| Test | 3 000 | 15 % |
| **Total** | **20 000** | 4 000 per class |

### 5.4 Architecture

```
skeleton tensor (64,64,3) ──> Conv32 stride2 ─ BN ─ ReLU ─ MaxPool
                              Conv64 ─ BN ─ ReLU
                              Conv64 ─ BN ─ ReLU ─ MaxPool
                              Conv128 ─ BN ─ ReLU ─ MaxPool
                              GlobalAvgPool ─ Dense128 ─ ReLU        ──┐
                                                                       ├─> concat(192)
geometry (126) ────────────>  Dense128 ─ BN ─ ReLU ─ Dropout(0.25)   ──┘      │
                              Dense64 ─ ReLU                                   │
                                                    Dense128 ─ Dropout(0.35) ─ Dense5 softmax
```

197 925 parameters · 0.8 MB · ~1.3 ms per frame on CPU.

Three decisions worth defending:

* **Stride-2 stem.** Skeleton tensors are sparse line drawings; their
  information is in the arrangement of strokes, not single-pixel detail.
  Downsampling immediately cuts training cost ~6× with no accuracy loss. The
  renderer compensates by drawing 2 px strokes so no bone is lost.
* **GlobalAveragePooling, not Flatten.** Cuts the head from ~260 k parameters to
  16 k and makes the network robust to scale changes.
* **Frame coordinates in the tensor.** The skeleton is *not* re-centred before
  rasterisation, so the subject's height in the room — the strongest single fall
  cue — survives into the CNN input.

### 5.5 Training configuration

Adam (lr 1e-3), batch 128, sparse categorical cross-entropy, early stopping on
validation accuracy (patience 10, best weights restored), LR halved on plateau.
Converged at epoch 22 with best validation accuracy 0.9890.

### 5.6 A debugging finding

The first training run produced 93 % training accuracy against **25 %**
validation accuracy — barely above the 20 % chance rate. The obvious reading is
catastrophic overfitting, and acting on it (more dropout, more regularisation)
would have wasted hours.

The diagnostic that settled it was evaluating the model on **its own training
data** in inference mode:

| Evaluation | Accuracy |
|---|---|
| Training data, training mode (batch statistics) | 0.9050 |
| Training data, **inference mode** (moving statistics) | 0.2403 |

Same weights, same data, 65 points apart. Generalisation cannot explain that —
only the fact that BatchNorm normalises with batch statistics while training and
moving statistics at inference. With 72 steps per epoch, Keras' default
momentum of 0.99 decays the initialisation by only `0.99^72 ≈ 0.49` per epoch,
so the moving statistics stay dominated by their priors. Inspection confirmed
it: `moving_variance ≈ 0.38` uniformly across channels, exactly the
`0.99^96 ≈ 0.38` predicted from initialisation alone. Setting `momentum=0.9`
converges them within one epoch and the gap vanished.

**Lesson: a train/validation gap that appears within the first few epochs and is
this extreme is a bug, not overfitting.**

A second finding was a data bug. Sampling gait phase uniformly over `[0, 2π)`
means walking frames at phase ≈ 0 or π have the feet passing each other — a
silhouette *identical* to standing. Those frames carry no signal distinguishing
the classes, so labelling them "Walking" injected irreducible label noise and
capped achievable accuracy. Restricting the sampled phase to the swing portion
of the gait cycle removed the contradiction and lifted accuracy from 93.3 % to
98.4 %. When two classes refuse to separate, check that the labels are
self-consistent before blaming the model.

---

## Step 6 — Model Evaluation and Testing

### 6.1 Overall metrics — validation and test

Both held-out splits are reported. Their close agreement is the evidence that
the test score is not itself the product of repeated model selection.

| Metric | Validation (n = 3 000) | Test (n = 3 000) |
|---|---|---|
| Accuracy | 0.9890 | 0.9837 |
| Precision (macro) | 0.9890 | 0.9838 |
| Recall (macro) | 0.9890 | 0.9837 |
| F1 (macro) | 0.9890 | 0.9837 |
| Fall precision | 0.9967 | 1.0000 |
| Fall recall | 1.0000 | 1.0000 |

Weights are chosen by **validation** accuracy and the test split is scored once,
at the end. The 0.5-point validation-to-test drop is the expected cost of having
used validation for model selection — reporting the higher number as the
headline would be selection bias.

Full per-class reports for **both** splits are in
`reports/classification_report.txt`; both confusion matrices are in
`reports/metrics.json`.

### 6.2 Per-class performance

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| **Fall Detected** | **1.0000** | **1.0000** | **1.0000** | 600 |
| Walking | 0.9605 | 0.9733 | 0.9669 | 600 |
| Sitting | 1.0000 | 1.0000 | 1.0000 | 600 |
| Standing | 0.9604 | 0.9700 | 0.9652 | 600 |
| Normal Activity | 0.9983 | 0.9750 | 0.9865 | 600 |

### 6.3 Confusion matrix

| actual ↓ / predicted → | Fall | Walking | Sitting | Standing | Normal |
|---|---|---|---|---|---|
| **Fall Detected** | **600** | 0 | 0 | 0 | 0 |
| **Walking** | 0 | **584** | 0 | 16 | 0 |
| **Sitting** | 0 | 0 | **600** | 0 | 0 |
| **Standing** | 0 | 17 | 0 | **582** | 1 |
| **Normal Activity** | 0 | 7 | 0 | 8 | **585** |

Reading it:

* **The fall row and the fall column are both clean.** No fall was missed, and
  nothing else was ever called a fall. In a safety system these two facts matter
  far more than overall accuracy, because their costs are wildly asymmetric: a
  missed fall can be fatal, while a false alarm erodes the trust that keeps the
  system switched on.
* **Normal Activity is never confused with Fall** — the bending false-alarm case
  is fully solved.
* The residual error is Walking↔Standing (33 of 49 errors), which is
  irreducibly ambiguous from a single frame and is resolved temporally in the
  video path.

### 6.4 Prediction outputs

Annotated prediction panels for every class are committed under
`reports/predictions/`, each showing the pose overlay, the predicted class and
confidence, the full probability distribution, the biomechanical evidence and
the resulting alert level:

| File | Contents |
|---|---|
| `01_fall_detected.png` … `05_normal_activity.png` | one panel per activity class |
| `06_false_alarm_test.png` | a real fall beside a deep bend — the decisive comparison |
| `07_grid_all_classes.png` | contact sheet across all five classes |

These are rendered by `scripts/make_predictions.py` rather than screen-captured,
so re-running it after any retraining regenerates them against the current
model instead of leaving stale screenshots in the repository.

### 6.5 Robustness under real-world conditions

Rather than assert robustness, it was measured by re-generating the test set at
progressively harsher landmark-noise levels. Note that *lighting* is not
simulated directly: lighting does not act on the classifier at all, it acts on
MediaPipe upstream, and arrives as landmark jitter and lost visibility — exactly
what is swept here.

| Landmark jitter σ | Overall accuracy | Fall recall |
|---|---|---|
| 0.004 | 0.9485 | **1.0000** |
| 0.010 | 0.9438 | **1.0000** |
| 0.020 | 0.9285 | **1.0000** |
| 0.035 | 0.8862 | **1.0000** |
| 0.055 | 0.7715 | **1.0000** |

*(These runs add extra occlusion on top of the baseline augmentation, which is
why overall accuracy starts below the 98.37 % headline.)*

**Fall recall holds at 1.000 across a 14× sweep** while overall accuracy falls
from 95 % to 77 %. That is the correct failure mode: as conditions degrade the
system loses the ability to distinguish walking from standing — which nobody is
paged about — long before it loses the ability to detect that someone is on the
floor.

### 6.6 Threshold calibration

The corroboration threshold was measured, not guessed:

| Class | median rule score | 95th percentile |
|---|---|---|
| Fall Detected | 0.74 | 0.97 |
| Sitting | 0.11 | 0.28 |
| Normal Activity | 0.11 | 0.41 |
| Walking | 0.00 | 0.06 |
| Standing | 0.00 | 0.00 |

| Threshold | Falls corroborated | False vote on non-falls |
|---|---|---|
| 0.35 | 0.983 | 0.0280 |
| **0.42** | **0.959** | **0.0113** |
| 0.50 | 0.906 | 0.0022 |

0.42 was selected. Because an alert *also* requires the CNN to agree, and the
CNN has 1.000 fall precision, the **joint** false-alarm rate is effectively
zero. Raising the threshold to 0.50 would silently demote ~6 % of real falls to
a non-paging state for no practical gain in specificity — the wrong trade in a
safety system.

### 6.7 End-to-end alert behaviour

Single-frame alert levels, 300 samples per class:

| Class | NORMAL | WATCH | ALERT |
|---|---|---|---|
| Fall Detected | 0 | 16 | **284** |
| Walking | 300 | 0 | 0 |
| Sitting | 299 | 1 | 0 |
| Standing | 300 | 0 | 0 |
| Normal Activity | 291 | 9 | 0 |

**Zero false alarms across every non-fall class**, including bending. On a
scripted video sequence (standing → walking → collapse) the engine escalates
`NORMAL` → `ALERT` → `EMERGENCY` once persistence is satisfied.

### 6.8 Deployment challenges considered

| Challenge | Mitigation |
|---|---|
| **Lighting variation** | Acts on MediaPipe, not the classifier; arrives as jitter and lost visibility. Swept in §6.4. Visibility below 0.45 is flagged as provisional in the UI. |
| **Camera angle** | ±62° yaw foreshortening and σ 7.5° roll modelled during training. |
| **Occlusion** | 28 % of training samples lose a contiguous body group; occluded landmarks are omitted from the tensor rather than drawn wrongly. |
| **Similar postures** | The bending case is an explicit class; the two-tier rule adds pelvis height as an independent check. |
| **False fall detections** | Corroboration + persistence + impact velocity. Measured: zero on 1 200 non-fall frames. |
| **Multi-person scenes** | Not handled — a known limitation, see Step 8. |

---

## Step 7 — Deployment

Deployed to Streamlit Community Cloud. Four surfaces:

* **Image Analysis** — pose overlay, activity with confidence, class-probability
  bars, biomechanical evidence panel, alert banner, plus a view of the exact
  64×64×3 tensor the CNN receives.
* **Video Monitoring** — per-frame analysis, annotated feed, fall-evidence
  timeline, activity distribution, emergency event log, downloadable incident CSV.
* **Live Simulation** — five scripted scenarios including a bending false-alarm test.
* **Model & Metrics** — model comparison, per-class metrics, confusion matrix,
  training curves.

**Engineering note.** The dashboard does not run TensorFlow. TensorFlow is a
~600 MB install whose import alone costs seconds and which, alongside MediaPipe
and OpenCV, does not reliably fit a free Streamlit Cloud container. The CNN is
trained with Keras, then exported to plain NumPy arrays and replayed by
`src/cnn_numpy.py` in ~90 lines. The export is verified numerically and rejected
if it disagrees with Keras by more than 1e-4 — measured agreement is **1.7e-06**.

Accessibility: every alert state carries a text label and an icon as well as a
colour, and the pulsing emergency animation respects
`prefers-reduced-motion`.

---

## Step 8 — Monitoring and Maintenance

### Known limitations

* **Single-person.** MediaPipe Pose tracks one subject; a second person is
  ignored. Multi-resident rooms need YOLOv8-Pose or per-track cropping.
* **No true 3D.** A fall directly toward or away from the camera foreshortens
  into a posture resembling standing.
* **Walking vs standing** is irreducibly ambiguous from a single frame.
* **Procedurally generated training geometry.** Real footage should be ingested
  before clinical use. This is a coursework prototype, not a medical device.

### Retraining loop

1. Log the landmarks — **not the imagery** — of every EMERGENCY frame, keeping
   the loop privacy-preserving by construction.
2. Have care staff confirm or reject each alert in the dashboard.
3. Ingest confirmed events monthly via `scripts/ingest_kaggle.py`; retrain.
4. **Gate promotion on fall recall never regressing** against a frozen benchmark
   set. Overall accuracy is explicitly the secondary metric.

### Roadmap

| Priority | Improvement | Rationale |
|---|---|---|
| High | Temporal model (CNN+LSTM) over landmark sequences | Turns Walking from a guess into a measurement; also sharpens impact detection. |
| High | Ingest real annotated footage | Closes the remaining domain gap. |
| Medium | Multi-person tracking | Required for shared living spaces. |
| Medium | Infrared for low light | The constraint is MediaPipe's detection rate, not the classifier. |
| Medium | Direct RTSP/CCTV ingest | Moves from upload-based to continuous monitoring. |
| Low | Per-resident calibration | Gait and posture baselines differ substantially between individuals. |

---

## Conclusion

FallGuard AI achieves **98.37 % test accuracy with perfect precision and recall
on the fall class and zero false alarms on bending**. The result rests less on
the network than on three design choices: treating bending as a first-class
category rather than an afterthought, requiring two independent detectors to
agree before raising an alarm, and treating a fall as a temporal event rather
than a frame.

The project also produced two transferable engineering lessons — that an extreme
early train/validation gap indicates a BatchNorm statistics bug rather than
overfitting, and that when two classes stubbornly refuse to separate it is worth
auditing the labels for self-consistency before adding model capacity.
