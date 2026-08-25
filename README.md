# FallGuard AI — Elderly Fall Detection & Healthcare Monitoring

Real-time fall detection from images and video using **MediaPipe BlazePose** for
pose estimation and a **hybrid CNN** for activity classification, deployed as a
Streamlit dashboard.

> **CRS Artificial Intelligence · Y2C1 Machine Learning and Deep Learning · Formative Assessment 2**
> Developing an AI-Powered Elderly Fall Detection System — Building and Deploying the Model

**Live app:** https://fallguard-ai-dhwanan.streamlit.app/
**Source:** https://github.com/DHWANAN722/fallguard-ai

---

## Results

| Metric | Value |
|---|---|
| Validation accuracy | **99.03 %** |
| Test accuracy | **99.03 %** |
| Macro F1 | **0.9903** |
| **Fall recall** (subject fully in frame) | **1.000** |
| **Fall recall** (overall) | **0.997** |
| **Fall precision** | **1.000** — nothing else is ever mistaken for a fall |
| False fall alerts (900 non-fall poses × 3 framings) | **0** |
| Inference latency | ~1.2 ms/frame (CPU, excluding pose estimation) |
| Model size | 197 925 parameters · 0.8 MB |

Evaluated on a held-out test split of 3 000 samples (600 per class) that no
model saw during training or model selection. Weights are selected on the
**validation** split; the test split is scored exactly once, at the end.

| Model | Accuracy | Macro F1 | Fall recall |
|---|---|---|---|
| **Hybrid CNN** (deployed) | **0.9903** | **0.9903** | 0.997 |
| Random Forest | 0.9773 | 0.9773 | 0.992 |
| SVM (RBF) | 0.9480 | 0.9479 | 0.980 |

Recall is quoted twice on purpose. The corpus contains skeletons framed as a
laptop webcam frames them — subject cropped at the waist, hips extrapolated
off-screen — and a fall filmed that tightly is materially harder to identify
(recall 0.92 on that subgroup, n=25). Reporting one blended figure would let
that disappear behind the easy majority. See
[known limitations](#known-limitations).

---

## What makes this more than a classifier

Falls are the leading cause of injury-related death in adults over 65. The hard
part of an automated monitor is not recognising a fall — it is recognising one
**without crying wolf**. A monitor that pages a caregiver every time a resident
bends down gets muted within a week, and a muted monitor detects nothing.

Three mechanisms address that directly:

**1 · `Bending` is a class in its own right, deliberately.** The fifth class is
modelled as bending/reaching/stooping — deep trunk flexion over *extended,
vertical legs* with the pelvis still at standing height. It is the posture that
most resembles a fall while being completely benign, and it is the single
largest source of false positives in deployed systems. Training against it
explicitly forces the model to learn pelvis height and leg configuration, not
just trunk angle.

> **Naming.** The assignment brief calls this category *Normal Activity*. It is
> displayed throughout as **Bending**, because that is exactly what it models
> and because a label should match what a viewer can see the subject doing.
> Same class, same index, clearer name.

**2 · Two-tier corroboration.** Alongside the network, a fully transparent
geometric rule scores each skeleton on trunk inclination, bounding-box aspect
ratio, pelvis height and leg verticality. **Both must agree** before any alert.
They fail in different ways — the CNN can be fooled by unusual limb
configurations, the rule by a deep bend — so agreement is far stronger evidence
than either alone. The threshold is calibrated on held-out data, not guessed.

**3 · Temporal persistence and impact velocity.** A fall is an event, not a
frame. On video an EMERGENCY requires the corroborated state to persist across
consecutive frames, or to coincide with rapid pelvis descent — the impact
signature that separates *falling* from *already sitting on the floor*.

**Privacy by construction.** The classifier never sees your photograph. Every
frame is reduced to 33 coordinates before anything else happens, so no
identifiable imagery reaches the model or would need to be retained by a
deployed monitoring system.

---

## Quick start

```bash
git clone https://github.com/DHWANAN722/fallguard-ai.git
cd fallguard-ai

pip install -r requirements.txt
streamlit run app.py
```

The repository ships with trained weights, so the dashboard runs immediately.
To retrain from scratch:

```bash
pip install -r requirements.txt -r requirements-dev.txt
python scripts/train.py --epochs 60
```

---

## The dashboard

| Tab | What it does |
|---|---|
| **Image Analysis** | Upload a photo → neon pose overlay, predicted activity with confidence, class-probability bars, biomechanical evidence panel, alert banner. Includes a panel showing the exact 64×64×3 tensor the CNN receives. |
| **Video Monitoring** | Upload a clip → per-frame analysis, annotated feed, fall-evidence timeline, activity distribution, emergency event log, downloadable incident CSV. |
| **Real-Time Monitoring** | Opens the device camera and classifies **every frame as it arrives** — live skeleton overlay, class, confidence, biomechanical evidence and escalating alert level, at video framerate. Both the pose estimator and the network run inside the browser, so no video is uploaded or stored. |
| **Live Simulation** | Five scripted scenarios — including a *bending-over false-alarm test* — for demonstrating the alert logic without footage. |
| **Model & Metrics** | Model comparison, per-class metrics, confusion matrix, training curves, and a plain-English explanation of how a decision is made. |

Alert levels: `NORMAL` → `WATCH` (one detector fired, logged only) → `ALERT`
(both agree) → `EMERGENCY` (sustained, or agreement plus impact velocity).
Every level carries a text label and an icon as well as a colour — a caregiver
may be colour-blind, and a red glow is easy to miss peripherally.

---

## Architecture

```
frame ──> MediaPipe BlazePose ──> 33 landmarks + visibility
                                        │
              ┌─────────────────────────┴──────────────────────────┐
              │                                                     │
   64×64×3 skeleton tensor                             126 geometric features
   (channels: torso+head / arms / legs,                (pelvis-normalised coords,
    drawn in FRAME coordinates so                       visibility, 27 clinical
    height in the room is preserved)                    descriptors)
              │                                                     │
      CNN branch → 128-d                                MLP branch → 64-d
              └──────────────────► concat(192) ◄────────────────────┘
                                        │
                            Dense128 → Dropout → softmax(5)
                                        │
                    ┌───────────────────┴───────────────────┐
              CNN fall vote                    biomechanical rule vote
                    └───────────────────┬───────────────────┘
                              corroboration + persistence
                                        │
                        NORMAL / WATCH / ALERT / EMERGENCY
```

**Classes:** `Fall Detected` · `Walking` · `Sitting` · `Standing` · `Bending` (the brief's "Normal Activity")

---

## Repository layout

```
fallguard-ai/
├── app.py                       Streamlit dashboard (Step 7)
├── requirements.txt             serving deps — no TensorFlow, deliberately
├── requirements-dev.txt         training deps
├── REPORT.md                    written report, Steps 4–8
├── VIDEO_SCRIPT.md              scene-by-scene screen-recording script
├── src/
│   ├── skeleton.py              biomechanical pose synthesis (33-landmark BlazePose format)
│   ├── features.py              126-d feature extraction + rule-based fall score
│   ├── render.py                skeleton → CNN tensor; neon overlay for the UI
│   ├── dataset.py               corpus construction + stratified 70/15/15 split
│   ├── models.py                CNN, hybrid, RF, SVM + NumPy exporters (training only)
│   ├── cnn_numpy.py             dependency-free inference runtime
│   ├── pose.py                  MediaPipe wrapper
│   ├── infer.py                 alert state machine
│   ├── video.py                 frame sampling + annotated encoding
│   └── theme.py                 cyberpunk visual system
├── scripts/
│   ├── train.py                 train, evaluate, compare, export
│   ├── selftest.py              40+ end-to-end checks; gates a commit
│   ├── make_predictions.py      renders the prediction panels
│   ├── ingest_kaggle.py         convert a real image dataset into the corpus format
│   └── make_notebook.py         generates the training notebook
├── notebooks/
│   └── FallGuard_Training.ipynb Steps 4–6, Colab-ready
├── models/                      trained weights + serving metadata
└── reports/                     confusion matrix, curves, metrics.json
    └── predictions/             annotated prediction panels, one per class
```

---

## Data provenance — please read

Kaggle was not reachable from the environment this project was built in, so the
training corpus is **procedurally generated**: a 2D forward-kinematic model of
the human body driven by joint-angle distributions from the gait and
fall-biomechanics literature, then degraded by the nuisance factors that break
real deployments — camera roll and yaw, pose-estimator landmark jitter,
occlusion, anthropometric variation, and scale/position change.

This is a disclosed engineering choice, not a claim of real data. What makes it
defensible is that the classifier consumes **normalised landmark geometry** —
the same representation MediaPipe emits from a real photograph — so the domain
gap is far narrower than it would be for a pixel model. The deployed app runs on
genuine MediaPipe output from whatever you upload.

**To train on real data instead**, `scripts/ingest_kaggle.py` converts any
labelled image dataset into the identical schema:

```bash
kaggle datasets download -d uttejkumarkandagatla/fall-detection-dataset
unzip fall-detection-dataset.zip -d data/raw

# folder names are matched case-insensitively against a list of common aliases
python scripts/ingest_kaggle.py --input data/raw --output data/real.npz \
       --blend-synthetic 1200

python scripts/train.py --dataset data/real.npz --reset
```

Blending is usually strongest: real images supply authentic pose-estimator noise
and camera geometry, while synthetic samples fill in the rare fall
configurations that public datasets under-represent.

---

## Deploying to Streamlit Community Cloud

1. Push this folder to a **public** GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repository, branch `main`, main file `app.py`.
4. Under **Advanced settings**, set **Python version 3.11**.
5. Deploy. First build takes 3–5 minutes.

Commit `models/` — the app needs the weights and they are only ~0.8 MB.

### If the build fails

Every pin in `requirements.txt` is load-bearing; the combination is the one that
actually resolves.

| Symptom | Cause |
|---|---|
| `module 'mediapipe' has no attribute 'solutions'` | MediaPipe 1.x installed. The 1.x line removed that API in favour of Tasks, which also downloads its model bundle at runtime. Pin `mediapipe==0.10.18`. |
| `FieldDescriptor object has no attribute 'label'` | protobuf ≥5 with MediaPipe 0.10.x. Pin `protobuf==4.25.8`. |
| Streamlit/protobuf resolver conflict | Streamlit ≥1.45 requires protobuf ≥5.26, which breaks MediaPipe. `streamlit==1.44.1` is the newest release still allowing protobuf<6. |
| `libGL.so.1: cannot open shared object file` | Use `opencv-python-headless`, not `opencv-python`. |
| Container runs out of memory | Something pulled in TensorFlow. It is intentionally absent — the CNN is served from exported NumPy weights via `src/cnn_numpy.py`. |
| Video player shows nothing | The host has no H.264 encoder. The app detects this and falls back to annotated stills plus a download button. |

---

## Technical notes

**Why no TensorFlow at serving time.** TensorFlow is a ~600 MB install whose
import alone costs seconds and which, alongside MediaPipe and OpenCV, does not
reliably fit in a free Streamlit Cloud container. The CNN is trained with Keras,
then exported to plain NumPy arrays and replayed by `src/cnn_numpy.py` in ~90
lines. The export is **verified numerically** — `scripts/train.py` rejects it if
it disagrees with Keras by more than 1e-4 (measured: **1.7e-06**).

**Why the hybrid architecture.** An earlier pure-CNN version reached 91% and put
essentially all of its error in Walking↔Standing, while the Random Forest —
which receives `ankle_horizontal_split` as an explicit number — scored 95% on
the identical split. The CNN was not short of capacity; it was short of
precision in one specific measurement, because at 64×64 the gap between the
ankles is a handful of pixels and the stride-2 stem blurs it further. Fusing
both views resolved it, and the hybrid beats both parents.

**A BatchNorm gotcha worth knowing.** Early runs showed 93% training accuracy
against 25% validation accuracy — apparent catastrophic overfitting. It was not.
Evaluating on the *training data* in inference mode also gave 24%, which
generalisation cannot explain. With 72 steps per epoch, Keras' default
BatchNorm momentum of 0.99 decays its initialisation by only `0.99^72 ≈ 0.49`
per epoch, so the moving statistics used at inference stay dominated by their
priors. `momentum=0.9` fixed it outright. A train/validation gap that appears in
the first few epochs and is that extreme is a bug, not overfitting.

**How the camera tab runs in real time.** Streamlit executes Python on a
server, so classifying frames in Python means uploading every one and waiting —
a request/response cycle, not a video feed. The usual escape is WebRTC, but
that needs Streamlit ≥1.45, which needs protobuf ≥5, and MediaPipe 0.10 needs
protobuf 4; adding `streamlit-webrtc` upgrades numpy, protobuf and Streamlit
together and breaks pose estimation outright (verified, not assumed).

So the model was moved to where the frames already are.
`scripts/export_web_model.py` folds every BatchNorm into the convolution before
it and packs the weights into 384 KB of float16; `assets/live_monitor.js`
re-implements the inference runtime, all 126 feature descriptors and the
biomechanical rule in JavaScript; MediaPipe's WebAssembly build supplies the
same BlazePose landmarks. Inference costs ~9 ms/frame, and no video ever leaves
the machine — a better privacy property than uploading it would have been, and
the architecture a real deployment would use anyway.

**Verifying a hand-written port.** A JavaScript re-implementation of a network
is the kind of code that looks like it works while being subtly wrong, so it is
checked rather than trusted, in two parts:

* *Arithmetic.* Four cases ship with the OpenCV-rendered tensor, Python's
  features and Python's output probabilities. The browser feeds the tensor
  straight in, isolating the convolutions, the asymmetric SAME padding and the
  float16 unpacking. The cases are deliberately **ambiguous** — two sit at
  0.53 against 0.47 — because a saturated case would still land on the right
  argmax with a transposed kernel and prove nothing. Agreement: **1.8 × 10⁻³**,
  which is float16 rounding.
* *Rasterisation.* JavaScript has no OpenCV, and `cv2.line(..., 2, LINE_AA)` is
  not a 2 px stroke: measured directly, it covers a capsule of radius ≈1.65 px
  and extends past both endpoints. A `<canvas>` stroke is roughly half that,
  and canvas "lighter" compositing would *add* the joint dots that OpenCV
  *overwrites*. So the raster is computed arithmetically by supersampled area
  coverage instead — deterministic, and identical in every browser. Measured
  against OpenCV over 3,000 held-out skeletons: mean pixel difference 0.0029,
  label agreement **99.74%**, accuracy 98.41% against 99.03%. Every
  disagreement was Standing/Walking; **fall recall and precision both stayed at
  1.000 and no disagreement involved the fall class at all.**

Both checks run in the browser on load and print to the console, and
`scripts/selftest.py` executes them under Node so a regression fails CI rather
than waiting to be noticed in a demo.

**A seated man classified as a fall, at 99.3% confidence.** Testing the live
tab on a laptop at a desk produced the worst possible output: *Fall Detected*,
99.3%, on somebody sitting upright and perfectly still. The evidence panel
showed the two detectors flatly contradicting each other — the network at 99%,
the biomechanical rule at 0% — and one number explained both: **pelvis height
−0.09**.

MediaPipe does not decline to locate a hip it cannot see. It extrapolates one
and reports a position outside the frame. A camera at desk height sees a torso
and puts the hips somewhere below the bottom edge, so the single strongest fall
cue in the model — *how low is the pelvis in the room* — reads as lower than
the floor. Every part of the system had been built and tested on skeletons
standing wholly inside the frame, so nothing had ever seen this and nothing
rejected it. Reproduced synthetically: seated desk framing gave *Fall Detected*
at 100%, and standing gave *Walking* at 98%.

The fix has two halves, because there were genuinely two bugs:

* **The corpus never contained the case.** Training now includes camera-framing
  augmentation — a zoom about the chest that pushes the hips off-screen, with
  off-frame landmarks losing visibility on a decay curve rather than being
  deleted, since MediaPipe reports them with reduced confidence rather than not
  at all. About 15% of the corpus is now partially framed.
* **The rule treated missing evidence as evidence.** `biomechanical_fall_score`
  withdraws the pelvis-height term entirely when the hips are off-frame or
  poorly localised, instead of reading "below the floor" as maximal descent.
  And because absence of evidence is not evidence of absence, the multiplicative
  gate uses a *higher* floor when the measurement is missing than when it is
  present and negative — otherwise a genuine fall filmed too close scores
  exactly on the threshold and resolves on noise. Swept over 2 500 skeletons:
  worth ~1.5 points of fall sensitivity at zero cost in false alarms.

There were in fact *three* instances of the same mistake, and finding the third
took measurement rather than reasoning. Leg verticality was still being trusted
when the legs were off-frame, and every remaining false alarm on bending had a
leg-verticality between 0.01 and 0.55 on a posture whose legs are vertical by
construction — noise from never-observed limbs, walking in behind the pelvis
term because the two are combined with `max()`.

The fix there is judged on **visibility, not position**, and that distinction
was worth eight points of fall sensitivity. Screening out knees below the frame
edge as well seemed obviously right and was wrong: someone lying on the floor
legitimately has knees at the edge, and the estimator has genuinely seen them —
visibility 0.90 at y = 1.10. Position at the edge means the leg is *low*, which
is evidence **for** a fall. Only invisibility means the measurement is absent.

Result on the reproduction: seated desk framing goes from *Fall Detected* 100%
to **Sitting 100%**, and across 900 non-fall poses at three framings there are
now **zero false fall alerts** — while fall recall on a properly framed subject
returned to **1.000**. The residual cost is stated plainly in the limitations
above rather than averaged away.

**Robustness.** Under a 14× sweep of landmark jitter, overall accuracy falls
from 95% to 78% while **fall recall stays essentially flat**. That is the failure mode
you want: as conditions degrade the system loses the ability to tell walking
from standing — which nobody is paged about — long before it loses the ability
to detect that someone is on the floor.

---

## The app broke without anyone touching it

Weeks after deployment the live app died on `import cv2`. Every file in this
repository was byte-identical to the version that had been working. The cause
was upstream:

```
mediapipe 0.10.18  requires  opencv-contrib-python      ← no upper bound
                             opencv-contrib-python 5.0.0.93 published
                             container rebuilt → pip takes the new major
                             → cv2/qt/plugins/platforms/libqxcb.so
                             → needs libGL.so.1, absent on Streamlit Cloud
                             → ImportError, app down
```

`opencv-python-headless` was already pinned, and that turned out to be no
defence. **Every OpenCV wheel unpacks into the same `cv2/` directory**, so
whichever is installed last wins — and pip still had to satisfy MediaPipe's
`opencv-contrib-python`, which is the GUI build. The pin was real; it was just
pinning the wrong end of the problem.

The lesson is that pinning direct requirements says nothing about what those
requirements *drag in*, and an unbounded transitive edge is a standing promise
that some future release of a package nobody here chose will keep working.

Fixed on both sides, because either alone leaves a gap:

* `requirements.txt` pins `opencv-contrib-python` to the 4.x line, so the
  unbounded edge cannot jump a major version again. There is a second reason to
  stay on 4.x: the browser rasteriser's capsule radius was fitted by measuring
  *this* OpenCV's `LINE_AA`, and a silent change there would not crash anything
  — it would just quietly degrade every tensor the CNN sees, which is far
  harder to notice.
* Crucially, **both** OpenCV distributions are pinned to the *same* version.
  They share one `cv2/` directory, so whichever installs last wins. At equal
  versions the file layouts match and `opencv-python-headless` — a direct
  requirement, installed after transitive dependencies — overwrites the contrib
  build completely. At different versions the extension module has a different
  filename (`cv2.cpython-3XX-<plat>.so` in 4.x versus `cv2.abi3.so` in 5.x), so
  neither overwrites the other, a mixture is left on disk, and the GUI build can
  win the import. That mixture was the outage.

**What did not work, twice.** The obvious hardening was a `packages.txt` telling
Streamlit Cloud to `apt-get install libgl1 libglib2.0-0`, so that any OpenCV
variant would import. It broke the deployment twice:

1. The file carried an explanatory comment header. **Streamlit Cloud passes
   every non-empty line straight to `apt-get` and does not strip `#`**, so apt
   tried to install packages named `Every`, `OpenCV` and `headless`, failed, and
   aborted the whole install.
2. Stripped to bare names, `libglib2.0-0` was then unsatisfiable: the base image
   had moved to Debian trixie, where it is renamed `libglib2.0-0t64` and its
   `libffi7`/`libpcre3` dependencies no longer exist. `held broken packages`.

Both failures came from pinning against a host image this project does not
control and cannot test locally. The version pin needs no apt at all, so
`packages.txt` was removed — the belt-and-braces was less reliable than the belt.

And because a fix nobody checks is a fix that rots, `scripts/selftest.py [11]`
walks the dependency graph of every pinned package and **fails** on any
unbounded native transitive dependency, then runs `pip install --dry-run` to
confirm the set still resolves with no OpenCV 5.x. Removing the pin makes it
fail with `opencv-contrib-python (pulled in by mediapipe)` — verified by
actually reintroducing the bug, not by assuming.

---

## Known limitations

* **Falls are harder when the subject is cropped at the waist.** If the camera
  sits at desk height and sees only a torso, recall drops to 0.92 (n=25) against
  1.000 when the whole body is in frame, and the *alert* rate drops further
  (14% against 83%), because an extrapolated hip corrupts the trunk-angle
  measurement as well as pelvis height. This is intrinsic rather than a tuning
  failure: the strongest fall cue is where the pelvis sits in the room, and if
  the pelvis is outside the frame that evidence does not exist. The system is
  built to *withhold* the cue rather than invent it — see below — which costs
  sensitivity in exchange for never crying wolf. A ceiling or wall-mounted
  camera, which is what a real deployment uses, avoids the situation entirely.
* **Single-person.** MediaPipe Pose tracks one subject; a second person in frame
  is ignored. Multi-resident rooms need YOLOv8-Pose or per-track cropping.
* **No true 3D.** A fall directly toward or away from the camera foreshortens
  into a posture resembling standing. Depth or a second camera resolves this.
* **Walking vs standing from a single frame** is irreducibly ambiguous. The
  video path resolves it temporally; the image tab cannot.
* **Trained on procedurally generated geometry.** Real footage should be
  ingested before any clinical use. This system is a coursework prototype and is
  not a medical device.

---

## Licence

MIT — coursework submission.
