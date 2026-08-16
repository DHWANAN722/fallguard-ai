# Screen-recording script — FallGuard AI

**Target length: 8–10 minutes.** The brief requires: project overview, dataset
explanation, pose estimation outputs, model prediction screenshots, Streamlit
dashboard, evaluation metrics, confusion matrix, alert system demonstration.
All eight are covered below, in order.

**Before you record**

- Open the deployed Streamlit URL (not localhost) so the live link is visible.
- Have ready: one photo of a person standing, one of someone bending over, one
  of someone lying on the floor, and one short clip (10–20 s) of a fall.
- Open `notebooks/FallGuard_Training.ipynb` in a second tab, already executed.
- Close notifications. Record at 1080p.

Read the **bold** lines aloud; the italics are stage directions.

---

## Scene 1 · Overview (0:00 – 1:15)

*Screen: the deployed dashboard, top of the page.*

> **Hi, I'm [name], and this is FallGuard AI — an elderly fall detection and
> healthcare monitoring system built for Formative Assessment 2.**
>
> **Falls are the leading cause of injury-related death in adults over 65. But
> the hard part of building a monitor for this isn't detecting a fall. It's
> detecting one without crying wolf. A system that alerts every time a resident
> bends down to pick something up gets muted within a week — and a muted monitor
> detects nothing at all.**
>
> **So the question I designed around wasn't "can I detect falls", it was "can I
> detect falls without false alarms". That shapes everything you're about to
> see.**

*Scroll slowly down the page so the four tabs are visible.*

> **The pipeline is: a camera frame goes to MediaPipe BlazePose, which returns
> 33 body landmarks. Those landmarks — not the photograph — go to a hybrid
> convolutional network that classifies the activity into one of five classes.
> Then a separate, transparent geometric rule has to agree before any alert is
> raised.**
>
> **Headline result: 98.4% accuracy on held-out test data, with perfect
> precision and perfect recall on the fall class, and zero false alarms on
> bending.**

---

## Scene 2 · Dataset (1:15 – 2:45)

*Switch to the notebook, section 2. Show the 5×6 grid of skeleton samples.*

> **First, the data — and I want to be upfront about this.**
>
> **Kaggle wasn't reachable from my build environment, so the training corpus is
> procedurally generated: a forward-kinematic model of the human body driven by
> joint-angle distributions from the gait and fall-biomechanics literature.**
>
> **What makes that defensible is that the classifier never sees pixels. It sees
> normalised landmark geometry — exactly the representation MediaPipe produces
> from a real photograph. So the gap between synthetic and real is much narrower
> than it would be for an image model. And I ship a script,
> `ingest_kaggle.py`, that converts any real labelled dataset into the same
> format so the whole pipeline retrains unchanged.**

*Point at the sample grid.*

> **Every sample is degraded the way real deployments degrade it: camera roll,
> camera yaw and foreshortening, landmark jitter, and occlusion — you can see
> here where the legs are missing because furniture is in the way.**

*Point specifically at row 1 (Fall) and row 5 (Normal Activity).*

> **These two rows are the whole design. Row one is a fall. Row five is
> "Normal Activity" — which I defined specifically as bending and reaching.**
>
> **Both have a near-horizontal trunk. The difference is where the pelvis is and
> what the legs are doing. Bending is the posture that breaks naive fall
> detectors, so I made it a first-class category the model has to learn against
> rather than an afterthought.**

*Show the split table.*

> **20,000 samples, split 70/15/15 — stratified within each class so validation
> and test stay perfectly balanced.**

---

## Scene 3 · Pose estimation output (2:45 – 3:45)

*Dashboard → Image Analysis tab. Upload the standing photo.*

> **Let's run a real photograph through it.**

*Wait for the result.*

> **MediaPipe has localised 33 landmarks — you can see the neon skeleton overlaid
> on the original image. Cyan is the torso, magenta the arms, green the legs.**
>
> **The system classifies this as Standing, with the confidence shown here.**

*Expand the "What the CNN actually sees" panel.*

> **And this is important — this is the actual input to the network. A 64×64
> three-channel tensor with the bones split by body part. The photograph goes no
> further than the pose estimator.**
>
> **That's a privacy property by construction: no identifiable imagery ever
> reaches the model, or would need to be stored by a deployed system in a care
> home.**

---

## Scene 4 · Prediction and evidence (3:45 – 5:00)

*Upload the fallen-person photo.*

> **Now a fall.**

*Let the red EMERGENCY-style banner appear.*

> **Fall Detected, and the alert banner has gone red. But look at the panel on
> the right — this is what I think makes the system trustworthy rather than just
> accurate.**

*Point at the biomechanical evidence rows.*

> **Trunk inclination, bounding-box aspect ratio, pelvis height in frame, leg
> verticality. These are quantities a clinician can actually reason about. The
> system isn't just saying "fall" — it's showing its working.**
>
> **And that bottom number, the biomechanical score, is computed by a rule that
> is completely independent of the neural network. Both have to agree before an
> alert fires.**

*Now upload the bending photo. This is the money shot.*

> **Here's the test that matters. This person is bending over to pick something
> up. Trunk angle is around 60 degrees — a naive detector fires here.**

*Point at the ALL CLEAR banner.*

> **All clear. Classified as Normal Activity. The pelvis is still at standing
> height and the legs are still vertical, so the biomechanical rule refuses to
> corroborate — and no alert is raised.**
>
> **That single case is the difference between a system people keep switched on
> and one they mute.**

---

## Scene 5 · Video monitoring and the alert system (5:00 – 6:30)

*Video Monitoring tab. Upload the fall clip.*

> **Single frames only get you so far. A fall is an event, not a frame.**

*Let it process.*

> **The system sampled the clip at 6 frames per second and ran pose estimation
> and classification on every sampled frame.**

*Point at the timeline chart.*

> **This timeline shows both detectors over time — the red line is the network's
> fall probability, the blue dotted line is the independent biomechanical
> score. You can see them both spike at the moment of impact.**

*Point at the alert banner and the event log.*

> **Notice the escalation. A single corroborated frame is an ALERT. It only
> becomes an EMERGENCY once the state persists across consecutive frames, or if
> it coincides with rapid pelvis descent — the impact signature.**
>
> **That's what separates someone falling from someone who was already sitting
> on the floor.**

*Scroll to the analytics tiles and the event log.*

> **And these are the monitoring metrics a caregiver would see: total detections,
> falls detected, normal activity, mean confidence, activity distribution, and a
> full emergency event log — which downloads as a CSV for the incident record.**

*Optional: Live Simulation tab → run "Bending to pick something up".*

> **There's also a simulation tab for demonstrating the logic without footage —
> including a dedicated bending false-alarm test.**

---

## Scene 6 · Evaluation metrics and confusion matrix (6:30 – 8:15)

*Model & Metrics tab.*

> **Now the evaluation. Everything here is on a held-out test split of 3,000
> samples that no model saw during training or model selection.**

*Point at the model comparison table.*

> **The brief recommended a CNN, but I didn't want to just assert that, so I
> trained three models on identical splits. The hybrid CNN at 98.4%, a Random
> Forest at 98.3%, and an SVM at 95.3%.**
>
> **And there's a story here. My first version was a pure CNN — it got 91%, and
> almost all of its errors were confusing walking with standing. Meanwhile the
> Random Forest, which gets the distance between the ankles as an explicit
> number, beat it.**
>
> **The CNN wasn't short of capacity. It was short of precision in one specific
> measurement — at 64×64 the gap between the ankles is only a few pixels. So
> instead of choosing, I fused both: a convolutional branch for overall posture
> shape, and a dense branch for the exact geometry. The hybrid beats both
> parents.**

*Scroll to the confusion matrix.*

> **This is the confusion matrix, and I want to draw attention to one thing.**

*Point at the top row and the first column.*

> **The fall row is clean and the fall column is clean. Six hundred falls, six
> hundred detected — nothing missed. And nothing else was ever mistaken for a
> fall.**
>
> **In a safety system those two numbers matter far more than overall accuracy,
> because the costs are wildly asymmetric. A missed fall can be fatal. A false
> alarm destroys the trust that keeps the system running.**
>
> **The remaining errors are walking versus standing — which is genuinely
> ambiguous from a single still frame, and which the video path resolves
> temporally.**

*Point at the training curves.*

> **Training converged at epoch 22 with validation accuracy of 98.9%.**

*Optional but strong — switch to notebook §4.1.*

> **One debugging note I'd flag. My first runs showed 93% training accuracy
> against 25% validation — which looks exactly like catastrophic overfitting.
> It wasn't. Evaluating on the training data itself in inference mode also gave
> 24%, and generalisation can't explain that.**
>
> **It was BatchNorm. Keras defaults to a momentum of 0.99, and with only 72
> steps per epoch the moving statistics used at inference were still dominated
> by their initial values. Setting momentum to 0.9 fixed it outright.**

---

## Scene 7 · Deployment and close (8:15 – 9:30)

*Show the browser URL bar with the live Streamlit link.*

> **The whole thing is deployed on Streamlit Community Cloud at this URL, and
> the source is on GitHub.**

*Briefly show the repo.*

> **One deployment decision worth mentioning: the dashboard doesn't run
> TensorFlow at all. TensorFlow is a 600-megabyte dependency that doesn't
> reliably fit alongside MediaPipe and OpenCV in a free container. So I train
> with Keras, then export the weights to plain NumPy arrays and run inference in
> about 90 lines of NumPy.**
>
> **And the export is verified — the training script rejects it if the NumPy
> runtime disagrees with Keras by more than one part in ten thousand. Measured
> agreement is about two parts in a million.**

*Back to the dashboard.*

> **To summarise: 98.4% accuracy, perfect precision and recall on falls, zero
> false alarms on bending, and about 1.3 milliseconds per frame.**
>
> **But the number I'd actually defend is the false-alarm rate — because that's
> the one that decides whether a system like this is still switched on six
> months after it's installed.**
>
> **Known limitations: it tracks one person at a time, it can't resolve a fall
> directly toward the camera without depth, and it should be retrained on real
> footage before any clinical use. The retraining loop and roadmap are in the
> report.**
>
> **Thanks for watching.**

---

## Checklist — confirm each is visible on camera

- [ ] Project overview explanation — Scene 1
- [ ] Dataset explanation — Scene 2
- [ ] Pose estimation outputs — Scene 3
- [ ] Model prediction screenshots — Scenes 3, 4
- [ ] Streamlit dashboard — throughout
- [ ] Evaluation metrics — Scene 6
- [ ] Confusion matrix — Scene 6
- [ ] Alert system demonstration — Scenes 4, 5
- [ ] Live Streamlit URL visible in the address bar — Scenes 1, 7
