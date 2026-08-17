# Screen-recording script — FallGuard AI

**Target: 9–10 minutes.** Everything in **bold** is spoken aloud.
Everything in `[SCREEN: ...]` is what you do with the mouse at that moment.

The script is about 1,330 spoken words. Read at a normal pace it runs roughly
9 minutes, plus a minute or so of silence while the app processes — so it lands
comfortably in range. If you want it shorter, the two paragraphs marked
*(optional)* in Scene 6 can be dropped without losing anything the brief asks
for.

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

# SCENE 1 · Overview — 0:00 to 1:00

`[SCREEN: Tab 1, the live app, scrolled to the very top. The URL bar is visible.]`

> **Hi, I'm Dhwanan Bhatt, and this is FallGuard AI — an elderly fall detection
> and healthcare monitoring system, built for Formative Assessment 2. This is the
> live app, deployed on Streamlit Cloud — the URL is at the top of the screen.**

`[SCREEN: Move the mouse up and circle the address bar slowly once.]`

> **Falls are the leading cause of injury-related death in adults over 65. But
> the hard problem isn't detecting a fall — it's detecting one without crying
> wolf.**

> **A monitor that alerts every time a resident bends down gets muted within a
> week — and a muted monitor detects nothing at all. So the question I designed
> around wasn't "can I detect falls", it was "can I detect falls without false
> alarms". That shapes everything you're about to see.**

`[SCREEN: Slowly scroll down a little so all four tabs are clearly visible, then
scroll back up. Hover over each tab name in turn as you say the next lines.]`

> **The dashboard has four sections: Image Analysis for a single photo, Video
> Monitoring for a clip, Live Simulation, and Model and Metrics for the training
> evidence.**

---

# SCENE 2 · Dataset — 1:00 to 2:30

`[SCREEN: Switch to Tab 2 — the notebook on GitHub. Scroll to section 2,
"Step 5a — The dataset".]`

> **First, the data — and I want to be completely upfront about this.**

> **Kaggle wasn't reachable from the environment I built this in. So rather
> than fake it, the training corpus is procedurally generated — a
> forward-kinematic model of the human body driven by joint-angle distributions
> from the gait and fall-biomechanics literature.**

> **What makes that defensible is that my classifier never sees pixels — it sees
> normalised landmark geometry, exactly what MediaPipe produces from a real
> photograph. And I ship a script that converts any real dataset into the same
> format, so the pipeline retrains unchanged.**

`[SCREEN: Scroll down to the 5×6 grid of coloured skeletons.]`

> **Here are samples from the corpus. Every one is degraded the way real
> deployments degrade: camera roll, yaw and foreshortening, landmark jitter, and
> occlusion — you can see skeletons here missing their legs, because in a real
> room furniture would be in the way.**

`[SCREEN: Move the mouse to the first row, then down to the fifth row.]`

> **These two rows are the entire design of this project. Row one is "Fall
> Detected". Row five is "Normal Activity", which I defined specifically as
> bending and reaching.**

> **Both have a near-horizontal trunk. The difference is where the pelvis is and
> what the legs are doing. Bending is the posture that breaks naive fall
> detectors, so instead of ignoring it I made it a category the model is forced
> to learn against.**

`[SCREEN: Scroll to the split table showing train / val / test counts.]`

> **Twenty thousand samples, split seventy, fifteen, fifteen — stratified
> within each class, so validation and test stay perfectly balanced.**

---

# SCENE 3 · Pose estimation output — 2:30 to 3:40

`[SCREEN: Switch back to Tab 1. You should be on the Image Analysis tab.
Click the "Standing" button.]`

> **Let's run a real photograph through it. I've bundled three of my own photos
> in, so anyone opening this link can try it in one click.**

`[SCREEN: Wait for the result to appear. Do not talk over the loading spinner.]`

> **That's me standing. MediaPipe BlazePose has localised thirty-three body
> landmarks — cyan is the torso, magenta the arms, green the legs.**

`[SCREEN: Point at the "Predicted activity" card on the right.]`

> **The system classifies this as Standing, at a hundred percent confidence,
> and the alert banner says ALL CLEAR.**

`[SCREEN: Scroll down and click to expand "What the CNN actually sees".]`

> **This is the actual input to the network — a sixty-four by sixty-four
> three-channel tensor, bones split by body part. The photograph goes no further
> than the pose estimator. That's a privacy property built into the
> architecture: no identifiable image of a resident ever reaches the model.**

---

# SCENE 4 · The false-alarm test — 3:40 to 5:20

**This is the most important 90 seconds of the video. Do not rush it.**

`[SCREEN: Scroll back up. Click the "Fallen on floor" button.]`

> **Now a fall. This is me lying on the floor.**

`[SCREEN: Wait for the orange FALL ALERT banner. Pause a beat.]`

> **Fall Detected, ninety-six percent, and the banner has gone to FALL ALERT.
> But look at the panel on the right — this is what makes it trustworthy rather
> than just accurate.**

`[SCREEN: Scroll so the "Biomechanical evidence" panel is visible. Move the
mouse slowly down the four rows as you name them.]`

> **Trunk inclination, eighty-nine degrees — almost horizontal. Aspect ratio
> zero point six seven, wider than it is tall. And leg verticality zero point
> zero four — the legs are nowhere near underneath the body.**

> **These are quantities a clinician can reason about — it's showing its
> working.**

`[SCREEN: Point at the "Biomechanical score" row at the bottom of the panel.]`

> **And this bottom number, the biomechanical score, is a hundred percent — from
> a separate geometric rule that doesn't use the neural network at all. Both have
> to agree before any alert is raised.**

`[SCREEN: Scroll up and click the "Bending over" button.]`

> **So here's the test that actually matters. This is me bending over to pick a
> pen up off the floor.**

`[SCREEN: Wait for the result. Let the green ALL CLEAR banner sit on screen for
a couple of seconds before speaking.]`

> **ALL CLEAR. Classified as Normal Activity, one hundred percent.**

`[SCREEN: Point at the trunk inclination row.]`

> **Look at the trunk angle — eighty degrees. Nearly as horizontal as the fall I
> just showed you. A detector thresholding on trunk angle fires here every time.**

`[SCREEN: Point at the pelvis height and leg verticality rows.]`

> **But the pelvis is still at standing height and leg verticality is zero point
> nine six — the legs are still under the body. So the rule refuses to
> corroborate and no alarm is raised. That one case is the difference between a
> system people keep switched on and one they mute in the first week.**

---

# SCENE 5 · Video monitoring and the alert system — 5:20 to 7:00

`[SCREEN: Click the "VIDEO MONITORING" tab, then click the "Fall clip" button.]`

> **Single frames only get you so far. A fall is an event, not a frame. Here's a
> real clip of me walking, then falling.**

`[SCREEN: Wait through the progress bar. Stay quiet while it processes — it
takes a few seconds.]`

> **It sampled at six frames per second and ran pose estimation on every frame.
> This is the annotated feed, looping — you can watch the skeleton track me and
> the label change as the classification changes.**

`[SCREEN: Point at the small warning line under the alert banner that reads
"...sampled frames contained no detectable person and were skipped".]`

> **And notice this line — several frames were skipped. During the fastest part
> of the fall there's motion blur, the pose estimator's confidence collapses, and
> the system rejects those frames instead of guessing. An alert built on
> unreliable landmarks is worse than no alert.**

`[SCREEN: Scroll down to the "Fall evidence over time" line chart.]`

> **This timeline shows both detectors over time — red is the network's fall
> probability, dotted blue is the independent biomechanical score. Red jumps to
> a hundred percent at the moment of impact.**

`[SCREEN: Scroll up briefly to the alert banner, then back down to the event log
table.]`

> **The clip reaches FALL ALERT. One corroborated frame is an ALERT; it only
> becomes an EMERGENCY if that persists across four frames, or coincides with a
> rapid pelvis-descent impact. That's the difference between someone falling and
> someone already sitting on the floor.**

`[SCREEN: Scroll to the six metric tiles, then to the activity distribution
chart, then to the event log and the CSV download button.]`

> **And these are the metrics a caregiver would see — total detections, falls
> detected, mean confidence, the activity distribution, and a full event log that
> downloads as a CSV for the incident record.**

---

# SCENE 6 · Evaluation metrics and confusion matrix — 7:00 to 8:40

`[SCREEN: Click the "MODEL & METRICS" tab.]`

> **Now the evaluation. Everything here is on a held-out test split of three
> thousand samples that no model saw during training or model selection.**

`[SCREEN: Point at the four metric tiles across the top.]`

> **A hundred and ninety-seven thousand parameters, fourteen thousand training
> samples, ninety-eight point four percent accuracy, and a hundred percent fall
> recall.**

`[SCREEN: Point at the "Validation vs test" table.]`

> **I show validation and test side by side deliberately. They agree closely,
> which is the evidence the test score isn't itself overfitted — I selected
> weights on validation and scored test once, at the end.**

`[SCREEN: Scroll to the "Model comparison" table.]`

*(optional — cut this and the next paragraph if you need to save 45 seconds)*

> **The brief recommended a CNN, but I didn't want to just assert that, so I
> trained three models on identical splits — my hybrid CNN, a Random Forest, and
> an SVM.**

> **My first version was a pure CNN. It scored ninety-one percent, and almost
> every error confused walking with standing — while the Random Forest, which
> gets the ankle separation as an explicit number, beat it. So I fused both: a
> convolutional branch for posture shape, a dense branch for exact geometry. The
> hybrid beats both parents.**

`[SCREEN: Scroll down to the confusion matrix image.]`

> **This is the confusion matrix on the test split.**

`[SCREEN: Move the mouse along the top row of the matrix, then down the first
column.]`

> **The fall row is clean, and the fall column is clean. Six hundred falls, six
> hundred detected — nothing missed. And nothing else was ever mistaken for a
> fall.**

> **In a safety system those matter far more than overall accuracy, because the
> costs are asymmetric — a missed fall can be fatal, a false alarm destroys the
> trust that keeps the system on. The remaining errors are walking versus
> standing, genuinely ambiguous from one still frame.**

`[SCREEN: Scroll to the training curves, then to the "Prediction gallery".]`

> **Training converged at epoch twenty-two at just under ninety-nine percent
> validation accuracy. And this is the prediction gallery — output across all
> five activity classes.**

---

# SCENE 7 · Deployment, limitations, close — 8:40 to 9:40

`[SCREEN: Scroll up so the URL bar is visible again.]`

> **To summarise: the system is live at this URL, with the source, notebook and
> written report on GitHub.**

> **Ninety-eight point four percent test accuracy, perfect precision and recall
> on the fall class, zero false alarms on bending, and about one and a half
> milliseconds per frame.**

> **The honest limitations: it tracks one person at a time, it can't resolve a
> fall directly toward the camera without depth, and it should be retrained on
> real footage before clinical use. The retraining loop is in my report.**

> **But the number I'd actually defend is the false-alarm rate — because that's
> what decides whether a system like this is still switched on six months after
> it's installed.**

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
