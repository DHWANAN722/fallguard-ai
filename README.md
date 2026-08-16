# FallGuard AI — Elderly Fall Detection & Healthcare Monitoring

Real-time fall detection from images and video using **MediaPipe BlazePose** for
pose estimation and a **hybrid CNN** for activity classification, deployed as a
Streamlit dashboard.

> **CRS Artificial Intelligence · Y2C1 Machine Learning and Deep Learning · Formative Assessment 2**
> Developing an AI-Powered Elderly Fall Detection System — Building and Deploying the Model

---

## Results

| Metric | Value |
|---|---|
| Validation accuracy | **98.90 %** |
| Test accuracy | **98.37 %** |
| Macro F1 | **0.9837** |
| **Fall recall** | **1.000** — no fall missed |
| **Fall precision** | **1.000** — nothing else mistaken for a fall |
| False alarms on bending | **0** |
| Inference latency | ~1.3 ms/frame (CPU, excluding pose estimation) |
| Model size | 197 925 parameters · 0.8 MB |

Evaluated on a held-out test split of 3 000 samples (600 per class) that no
model saw during training or model selection. Weights are selected on the
**validation** split; the test split is scored exactly once, at the end.

| Model | Accuracy | Macro F1 | Fall recall |
|---|---|---|---|
| **Hybrid CNN** (deployed) | **0.9837** | **0.9837** | 1.000 |
| Random Forest | 0.9827 | 0.9827 | 1.000 |
| SVM (RBF) | 0.9533 | 0.9535 | 1.000 |

---

## What makes this more than a classifier

Falls are the leading cause of injury-related death in adults over 65. The hard
part of an automated monitor is not recognising a fall — it is recognising one
**without crying wolf**. A monitor that pages a caregiver every time a resident
bends down gets muted within a week, and a muted monitor detects nothing.

Three mechanisms address that directly:

**1 · `Normal Activity` is bending, deliberately.** The fifth class is modelled
as bending/reaching/stooping — deep trunk flexion over *extended, vertical legs*
with the pelvis still at standing height. It is the posture that most resembles
a fall while being completely benign, and it is the single largest source of
false positives in deployed systems. Training against it explicitly forces the
model to learn pelvis height and leg configuration, not just trunk angle.

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
git clone https://github.com/<your-username>/fallguard-ai.git
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

**Classes:** `Fall Detected` · `Walking` · `Sitting` · `Standing` · `Normal Activity`

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

**Robustness.** Under a 14× sweep of landmark jitter, overall accuracy falls
from 95% to 78% while **fall recall stays at 1.000**. That is the failure mode
you want: as conditions degrade the system loses the ability to tell walking
from standing — which nobody is paged about — long before it loses the ability
to detect that someone is on the floor.

---

## Known limitations

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
