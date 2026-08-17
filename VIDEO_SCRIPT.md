# Screen-recording script — FallGuard AI

**Target: 8–10 minutes.** Everything in **bold** is spoken aloud.
Everything in `[SCREEN: ...]` is what you do with the mouse at that moment.

The brief requires eight things, and all eight are covered in order:
project overview · dataset explanation · pose estimation outputs · model
prediction screenshots · Streamlit dashboard · evaluation metrics · confusion
matrix · alert system demonstration.

---

## Before you press record — 5 minutes of setup

1. **Open two browser tabs, in this order:**
   - Tab 1: https://fallguard-ai-dhwanan.streamlit.app/
   - Tab 2: `notebooks/FallGuard_Training.ipynb` on GitHub —
     https://github.com/DHWANAN722/fallguard-ai/blob/main/notebooks/FallGuard_Training.ipynb
     (GitHub renders the notebook with all the charts already visible; you do
     not need Colab.)
2. **Wake the app up.** Click through all four tabs once, and click one sample,
   so nothing is cold when you record. Then reload Tab 1 so it is back to a
   clean Image Analysis view.
3. **Silence everything.** Do Not Disturb on. Close Slack, Mail, Messages.
4. **Hide clutter.** Close every other tab. Hide your bookmarks bar
   (`Cmd + Shift + B`) so the URL is prominent.
5. **Start recording:** `Cmd + Shift + 5` → **Record Entire Screen** →
   **Options → Microphone → MacBook Air Microphone** → **Record**.
   Stop from the ■ in the menu bar. The file lands on your Desktop.

**Delivery notes.** Speak slower than feels natural — about 20% slower. Pause
for a full second after each click so the viewer's eye catches up. If you
fumble a sentence, stop, pause three seconds, and say it again from the start
of that sentence; it is trivial to trim later, and re-recording the whole thing
is not.

---

# SCENE 1 · Overview — 0:00 to 1:20

`[SCREEN: Tab 1, the live app, scrolled to the very top. The URL bar is visible.]`

> **Hi, I'm Dhwanan Bhatt, and this is FallGuard AI — an elderly fall detection
> and healthcare monitoring system, built for Formative Assessment 2 of the
> Machine Learning and Deep Learning module.**

> **This is the live application, deployed on Streamlit Cloud — you can see the
> URL at the top of the screen.**

`[SCREEN: Move the mouse up and circle the address bar slowly once.]`

> **Falls are the leading cause of injury-related death in adults over 65. But
> when I started building this, I realised the hard problem isn't detecting a
> fall. It's detecting a fall without crying wolf.**

> **A monitor that alerts every time a resident bends down to pick something up
> gets muted within a week. And a muted monitor detects nothing at all. So the
> question I designed around wasn't "can I detect falls" — it was "can I detect
> falls without false alarms". That single decision shapes everything you're
> about to see.**

`[SCREEN: Slowly scroll down a little so all four tabs are clearly visible, then
scroll back up. Hover over each tab name in turn as you say the next lines.]`

> **The dashboard has four sections. Image Analysis runs a single photograph.
> Video Monitoring runs a clip frame by frame. Live Simulation demonstrates the
> alert logic without any footage. And Model and Metrics shows the training
> evidence.**

`[SCREEN: Point at the sidebar on the left.]`

> **And on the left are the live detection thresholds and the video sampling
> controls, so everything I'm about to describe can be adjusted and tested.**

---

# SCENE 2 · Dataset — 1:20 to 3:00

`[SCREEN: Switch to Tab 2 — the notebook on GitHub. Scroll to section 2,
"Step 5a — The dataset".]`

> **First, the data — and I want to be completely upfront about this.**

> **Kaggle was not reachable from the environment I built this in. So rather
> than fake it, the training corpus is procedurally generated: a two-dimensional
> forward-kinematic model of the human body, driven by joint-angle distributions
> taken from the gait and fall-biomechanics literature.**

> **What makes that defensible is that my classifier never sees pixels. It sees
> normalised landmark geometry — which is exactly what MediaPipe produces from a
> real photograph. So the gap between synthetic and real is far narrower than it
> would be for an image-based model. And I ship a script, ingest underscore
> kaggle dot py, that converts any real labelled dataset into the identical
> format, so the whole pipeline retrains unchanged.**

`[SCREEN: Scroll down to the 5×6 grid of coloured skeletons.]`

> **Here are samples from the corpus. Every one is degraded the way real
> deployments degrade: camera roll, camera yaw and foreshortening, landmark
> jitter, and occlusion.**

`[SCREEN: Point at any skeleton in the grid that is missing its legs.]`

> **You can see occlusion here — the legs are missing, because in a real room
> furniture would be in the way.**

`[SCREEN: Move the mouse to the first row, then down to the fifth row.]`

> **Now, these two rows are the entire design of this project. Row one is
> "Fall Detected". Row five is "Normal Activity" — and I defined Normal Activity
> very specifically as bending and reaching.**

> **Look at them. Both have a near-horizontal trunk. The difference is where the
> pelvis is, and what the legs are doing. Bending over is the posture that
> breaks naive fall detectors, so instead of ignoring it, I made it a
> first-class category that the model is forced to learn against.**

`[SCREEN: Scroll to the split table showing train / val / test counts.]`

> **Twenty thousand samples in total, split seventy, fifteen, fifteen — and the
> split is stratified within each class, so validation and test stay perfectly
> balanced at six hundred samples per class.**

---

# SCENE 3 · Pose estimation output — 3:00 to 4:10

`[SCREEN: Switch back to Tab 1. You should be on the Image Analysis tab.
Click the "Standing" button.]`

> **Now let's run a real photograph through it. I've bundled three of my own
> photos into the app, so anyone opening this link can try it in one click.**

`[SCREEN: Wait for the result to appear. Do not talk over the loading spinner.]`

> **That's me standing. MediaPipe BlazePose has localised thirty-three body
> landmarks, and you can see the skeleton drawn on top of the actual
> photograph. Cyan is the torso, magenta is the arms, green is the legs.**

`[SCREEN: Point at the "Predicted activity" card on the right.]`

> **The system classifies this as Standing, at a hundred percent confidence,
> and the alert banner says ALL CLEAR.**

`[SCREEN: Scroll down and click to expand "What the CNN actually sees".]`

> **And this is the part I think is most important. This little image is the
> actual input to the neural network — a sixty-four by sixty-four, three-channel
> tensor, with the bones split by body part.**

> **The photograph goes no further than the pose estimator. That's a privacy
> property built into the architecture: in a real care home, no identifiable
> image of a resident ever reaches the model, and none would need to be stored.**

---

# SCENE 4 · The false-alarm test — 4:10 to 5:40

**This is the most important 90 seconds of the video. Do not rush it.**

`[SCREEN: Scroll back up. Click the "Fallen on floor" button.]`

> **Now a fall. This is me lying on the floor.**

`[SCREEN: Wait for the orange FALL ALERT banner. Pause a beat.]`

> **Fall Detected, ninety-six point six percent, and the banner has gone to FALL
> ALERT. But look at the panel on the right — this is what makes the system
> trustworthy rather than just accurate.**

`[SCREEN: Scroll so the "Biomechanical evidence" panel is visible. Move the
mouse slowly down the four rows as you name them.]`

> **Trunk inclination, eighty-nine degrees — almost horizontal. Bounding-box
> aspect ratio, zero point six seven — wider than it is tall. Pelvis height in
> frame. And leg verticality, zero point zero four — the legs are nowhere near
> underneath the body.**

> **These are quantities a clinician can actually reason about. The system isn't
> just saying "fall" — it's showing its working.**

`[SCREEN: Point at the "Biomechanical score" row at the bottom of the panel.]`

> **And this bottom number, the biomechanical score, is one hundred percent. It
> is computed by a completely separate geometric rule that does not use the
> neural network at all. Both of them have to agree before any alert is raised.**

`[SCREEN: Scroll up and click the "Bending over" button.]`

> **So here's the test that actually matters. This is me bending over to pick a
> pen up off the floor.**

`[SCREEN: Wait for the result. Let the green ALL CLEAR banner sit on screen for
a couple of seconds before speaking.]`

> **ALL CLEAR. Classified as Normal Activity, one hundred percent.**

`[SCREEN: Point at the trunk inclination row.]`

> **And look at the trunk angle — eighty degrees. That is nearly as horizontal
> as the fall I just showed you. A detector that thresholds on trunk angle alone
> fires here, every single time.**

`[SCREEN: Point at the pelvis height and leg verticality rows.]`

> **But the pelvis is still at standing height, and leg verticality is zero
> point nine six — the legs are still directly underneath the body. So the
> biomechanical rule refuses to corroborate, the score stays at twenty percent,
> and no alarm is raised.**

> **That one case is the difference between a system people keep switched on,
> and one they mute in the first week.**

---

# SCENE 5 · Video monitoring and the alert system — 5:40 to 7:20

`[SCREEN: Click the "VIDEO MONITORING" tab, then click the "Fall clip" button.]`

> **Single frames only get you so far, though. A fall is an event, not a frame.
> So here's a real clip of me walking and then falling.**

`[SCREEN: Wait through the progress bar. Stay quiet while it processes — it
takes a few seconds.]`

> **The system sampled the clip at six frames per second and ran pose estimation
> and classification on every sampled frame.**

`[SCREEN: Point at the annotated feed, which is looping on its own.]`

> **This is the annotated monitoring feed, playing on a loop. You can watch the
> skeleton track me through the movement, and the label change as the
> classification changes.**

`[SCREEN: Point at the small warning line under the alert banner that reads
"...sampled frames contained no detectable person and were skipped".]`

> **And notice this line. Several frames were skipped. During the fastest part
> of the fall there's motion blur, and the pose estimator's confidence collapses
> — so the system rejects those frames instead of guessing. I added that
> deliberately, because a fall alert built on unreliable landmarks is worse than
> no alert at all.**

`[SCREEN: Scroll down to the "Fall evidence over time" line chart.]`

> **This timeline shows both detectors over time. The solid red line is the
> neural network's fall probability. The dotted blue line is the independent
> biomechanical score. You can see the red line jump to one hundred percent at
> the moment of impact.**

`[SCREEN: Scroll up briefly to the alert banner, then back down to the event log
table.]`

> **The clip reaches FALL ALERT. And here's the escalation logic: one
> corroborated frame is an ALERT. It only becomes a full EMERGENCY if the state
> persists across four consecutive frames, or if it coincides with a rapid
> pelvis-descent signature — an impact.**

> **That's the difference between someone falling, and someone who was already
> sitting on the floor.**

`[SCREEN: Scroll to the six metric tiles, then to the activity distribution
chart, then to the event log and the CSV download button.]`

> **And these are the monitoring metrics a caregiver would actually see. Total
> detections, falls detected, normal activity, mean confidence, alert frames.
> The activity distribution chart. And a full emergency event log with
> timestamps, which downloads as a CSV for the incident record.**

---

# SCENE 6 · Evaluation metrics and confusion matrix — 7:20 to 9:00

`[SCREEN: Click the "MODEL & METRICS" tab.]`

> **Now the evaluation. Everything here is on a held-out test split of three
> thousand samples that no model saw during training or model selection.**

`[SCREEN: Point at the four metric tiles across the top.]`

> **A hundred and ninety-seven thousand parameters, fourteen thousand training
> samples, ninety-eight point four percent test accuracy, and a hundred percent
> fall recall.**

`[SCREEN: Point at the "Validation vs test" table.]`

> **I'm showing validation and test side by side deliberately. Ninety-eight
> point nine on validation, ninety-eight point four on test. They agree closely,
> which is the evidence that the test score isn't itself overfitted — I selected
> the weights on validation and scored the test split once, at the end.**

`[SCREEN: Scroll to the "Model comparison" table.]`

> **The brief recommended a CNN, but I didn't want to just assert that, so I
> trained three models on identical splits. My hybrid CNN at ninety-eight point
> four, a Random Forest at ninety-eight point three, and an SVM at ninety-five
> point three.**

> **And there's a real story here. My first version was a pure CNN — it scored
> ninety-one percent, and almost every error was confusing walking with
> standing. Meanwhile the Random Forest, which receives the distance between the
> ankles as an explicit number, beat it.**

> **The CNN wasn't short of capacity. It was short of precision in one specific
> measurement — at sixty-four by sixty-four, the gap between the ankles is only
> a few pixels. So instead of choosing between them, I fused both: a
> convolutional branch for overall posture shape, and a dense branch for the
> exact geometry. The hybrid beats both of its parents.**

`[SCREEN: Scroll down to the confusion matrix image.]`

> **This is the confusion matrix on the test split, and I want to draw attention
> to one thing specifically.**

`[SCREEN: Move the mouse along the top row of the matrix, then down the first
column.]`

> **The fall row is clean, and the fall column is clean. Six hundred falls, six
> hundred detected — nothing missed. And nothing else was ever mistaken for a
> fall.**

> **In a safety system those two numbers matter far more than overall accuracy,
> because the costs are wildly asymmetric. A missed fall can be fatal. A false
> alarm destroys the trust that keeps the system switched on.**

> **The remaining errors are walking versus standing, which is genuinely
> ambiguous from a single still frame — and which the video path resolves
> temporally.**

`[SCREEN: Scroll to the training curves image.]`

> **Training converged at epoch twenty-two, with validation accuracy just under
> ninety-nine percent.**

`[SCREEN: Scroll further to the "Prediction gallery" contact sheet.]`

> **And this is the prediction gallery — the model's output across all five
> activity classes.**

---

# SCENE 7 · Deployment, limitations, close — 9:00 to 10:00

`[SCREEN: Scroll up so the URL bar is visible again.]`

> **To summarise. The whole system is deployed and live at this URL, and the
> source is on GitHub with the training notebook, the written report, and a
> fifty-check test suite.**

> **Ninety-eight point four percent test accuracy. Perfect precision and perfect
> recall on the fall class. Zero false alarms on bending. And about one point
> three milliseconds per frame.**

> **One deployment decision worth mentioning: the dashboard doesn't run
> TensorFlow at all. TensorFlow is a six-hundred-megabyte dependency that
> doesn't reliably fit alongside MediaPipe in a free container. So I train with
> Keras, then export the weights to plain NumPy and run inference in about
> ninety lines. And the export is verified — the training script rejects it if
> the NumPy runtime disagrees with Keras by more than one part in ten thousand.**

> **Finally, the honest limitations. It tracks one person at a time. It can't
> resolve a fall directly toward the camera without depth information. And it
> should be retrained on real annotated footage before any clinical use — the
> retraining loop and the roadmap for that are both in my report.**

> **But the number I'd actually defend is the false-alarm rate. Because that's
> the one that decides whether a system like this is still switched on six
> months after it's installed.**

> **Thank you for watching.**

`[SCREEN: Stop recording — click the ■ in the menu bar.]`

---

## Checklist — tick each off after you watch it back

- [ ] Project overview explained — Scene 1
- [ ] Dataset explained, including the split — Scene 2
- [ ] Pose estimation output visible on a real photo — Scene 3
- [ ] Model predictions with confidence shown — Scenes 3, 4
- [ ] Streamlit dashboard shown throughout — all scenes
- [ ] Evaluation metrics shown — Scene 6
- [ ] Confusion matrix shown — Scene 6
- [ ] Alert system demonstrated — Scenes 4, 5
- [ ] The live URL is legible in the address bar — Scenes 1 and 7
- [ ] Audio is clear and there is no background noise
- [ ] Total length is between 8 and 10 minutes

## If something goes wrong on camera

**A sample button does nothing.** Reload the page and click it again — the app
sleeps after inactivity and the first click can wake it instead of running.

**The video clip takes a long time.** That is normal; it runs pose estimation on
every sampled frame. Stay quiet and let it finish rather than filling the
silence.

**You get a different number than the script says.** Read out whatever is on
screen. The numbers in this script are what the app produced when it was last
verified, but confidences shift by a fraction of a percent between runs. Never
read a number that contradicts what the marker can see.
