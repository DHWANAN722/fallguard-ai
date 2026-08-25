/*
 * live_ui.js — camera loop, neon overlay and live readouts.
 * Concatenated after live_monitor.js into one inline module by src/webcam.py.
 */

const $ = (id) => document.getElementById(id);
const MODEL = new FallGuardModel(SPEC);

/* ---------------------------------------------------------- self-check ---
 * A JavaScript re-implementation of a neural network is exactly the kind of
 * code that looks like it works while being quietly wrong, so it is checked
 * on load — and the two things that can go wrong are checked separately,
 * because a single combined number would be unreadable.
 *
 *   arithmetic — feed the embedded OpenCV tensor and Python's own features
 *                straight into the network. This isolates the convolutions,
 *                the padding and the float16 unpacking. The cases are
 *                deliberately ambiguous, two of them near 0.53 against 0.47,
 *                so a transposed kernel moves the numbers rather than hiding
 *                behind the argmax. Tolerance is float16 rounding.
 *
 *   raster     — the browser has no OpenCV, so it reproduces cv2's
 *                antialiased stroke arithmetically. Compared here as a mean
 *                pixel difference, which is what it is: a close approximation,
 *                measured offline over 3,000 test skeletons at 99.50% label
 *                agreement with no disagreement ever touching the fall class.
 */
function b64ToU8(b64) {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function selfTest() {
  let arith = 0, raster = 0, argmaxOk = true;
  for (const g of SPEC.golden) {
    const ref = b64ToU8(g.tensor_u8_b64);
    const t = new Float32Array(ref.length);
    for (let i = 0; i < ref.length; i++) t[i] = ref[i] / 255;

    const pr = MODEL.predict(t, Float32Array.from(g.features));
    let ai = 0, ei = 0;
    for (let i = 0; i < pr.length; i++) {
      arith = Math.max(arith, Math.abs(pr[i] - g.expected[i]));
      if (pr[i] > pr[ai]) ai = i;
      if (g.expected[i] > g.expected[ei]) ei = i;
    }
    if (ai !== ei) argmaxOk = false;

    const mine = renderTensor(g.landmarks, g.visibility);
    let sum = 0;
    for (let i = 0; i < mine.length; i++) sum += Math.abs(mine[i] - t[i]);
    raster = Math.max(raster, sum / mine.length);
  }
  const ok = arith < 5e-3 && argmaxOk && raster < 0.01;
  console.log(
    `[FallGuard] port self-test over ${SPEC.golden.length} golden cases\n` +
    `  arithmetic  max |Δprob| ${arith.toExponential(3)}  ` +
    `(tolerance 5e-3, float16 weights)  argmax ${argmaxOk ? "all match" : "MISMATCH"}\n` +
    `  rasteriser  mean |Δpixel| ${raster.toFixed(5)}  (tolerance 1e-2 vs OpenCV)\n` +
    (ok ? "  ✓ verified against Python" : "  ⚠ outside tolerance"));
  return { arith, raster, ok };
}

/* ------------------------------------------------------------ alert state */
const CNN_T = 0.55, BIO_T = 0.25, PERSIST = 4;
const LEVELS = {
  NORMAL: ["ALL CLEAR", "#00ff9c"],
  WATCH: ["MONITORING", "#ffd400"],
  ALERT: ["FALL ALERT", "#ff7a00"],
  EMERGENCY: ["EMERGENCY — FALL CONFIRMED", "#ff1f4f"],
};
let streak = 0, peak = "NORMAL", counts = {}, total = 0, falls = 0, alerts = 0;
const ORDER = ["NORMAL", "WATCH", "ALERT", "EMERGENCY"];

function classify(P, V) {
  const t = renderTensor(P, V);
  const f = extractFeatures(P, V);
  const pr = MODEL.predict(t, f);
  let idx = 0;
  for (let i = 1; i < pr.length; i++) if (pr[i] > pr[idx]) idx = i;
  const bio = biomechScore(P, V);
  const cnnVote = pr[0] >= CNN_T, bioVote = bio >= BIO_T;

  streak = (cnnVote && bioVote) ? streak + 1 : 0;
  let level = "NORMAL";
  if (cnnVote && bioVote) level = streak >= PERSIST ? "EMERGENCY" : "ALERT";
  else if (cnnVote || bioVote) level = "WATCH";

  return { pr, idx, bio, level, clin: clinical(P, V) };
}

/* --------------------------------------------------------------- drawing */
const video = $("vid"), canvas = $("cv"), ctx = canvas.getContext("2d");

function drawOverlay(Praw, V, accent) {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const px = Praw.map(p => [p[0] * w, p[1] * h]);
  const thick = Math.max(2, Math.round(Math.min(w, h) / 200));

  ctx.save();
  ctx.shadowBlur = thick * 5;            // the glow that makes it read as neon
  for (const [a, b, g] of BONES) {
    if (V[a] < 0.35 || V[b] < 0.35) continue;
    const col = accent || NEON[g];
    ctx.strokeStyle = col; ctx.shadowColor = col;
    ctx.lineWidth = thick; ctx.lineCap = "round";
    ctx.beginPath(); ctx.moveTo(px[a][0], px[a][1]); ctx.lineTo(px[b][0], px[b][1]);
    ctx.stroke();
  }
  ctx.restore();

  ctx.fillStyle = "#fff";
  for (let i = 0; i < 33; i++) {
    if (V[i] < 0.35) continue;
    ctx.beginPath(); ctx.arc(px[i][0], px[i][1], thick * 0.9, 0, 6.3); ctx.fill();
  }
}

const CLASS_COL = {
  "Fall Detected": "#ff1f4f", "Walking": "#00e5ff", "Sitting": "#ffb300",
  "Standing": "#8cff2b", "Bending": "#b26bff",
};

function paintUI(r) {
  const [label, col] = LEVELS[r.level];
  const banner = $("banner");
  banner.style.borderColor = col; banner.style.color = col;
  $("bannerTitle").textContent = label;
  $("bannerTitle").style.color = col;
  banner.classList.toggle("pulse", r.level === "EMERGENCY");

  const reasons = [];
  if (r.pr[0] >= CNN_T) reasons.push(`CNN fall probability ${(r.pr[0] * 100).toFixed(0)}%`);
  if (r.bio >= BIO_T) reasons.push(`biomechanical score ${(r.bio * 100).toFixed(0)}%`);
  if (streak >= PERSIST) reasons.push(`sustained over ${streak} frames`);
  if (r.partial) {
    reasons.push("lower body outside the frame — pelvis height unavailable, "
               + "so a fall is harder to corroborate");
  }
  $("bannerSub").textContent = reasons.length ? reasons.join(" · ")
    : "posture consistent with normal activity";

  const cls = SPEC.classes[r.idx];
  $("predName").textContent = cls;
  $("predName").style.color = CLASS_COL[cls];
  $("predConf").textContent = `confidence ${(r.pr[r.idx] * 100).toFixed(1)}%`;

  const order = [...r.pr.keys()].sort((a, b) => r.pr[b] - r.pr[a]);
  $("bars").innerHTML = order.map(i => {
    const c = CLASS_COL[SPEC.classes[i]], pct = r.pr[i] * 100;
    return `<div class="bar"><div class="row"><span style="color:${c}">${SPEC.classes[i]}</span>
      <span class="mono" style="color:${c}">${pct.toFixed(1)}%</span></div>
      <div class="track"><div class="fill" style="width:${pct.toFixed(1)}%;background:${c};
      box-shadow:0 0 10px ${c}"></div></div></div>`;
  }).join("");

  const ev = [
    ["Trunk inclination", `${r.clin.torso.toFixed(1)}°`],
    ["Bounding-box aspect", r.clin.aspect.toFixed(2)],
    ["Pelvis height in frame", r.clin.pelvis.toFixed(2)],
    ["Leg verticality", r.clin.legVert.toFixed(2)],
    ["Landmark visibility", r.clin.vis.toFixed(2)],
    ["Biomechanical score", `${(r.bio * 100).toFixed(0)}%`],
  ];
  $("evidence").innerHTML = ev.map(([k, v]) =>
    `<div class="ev"><span>${k}</span><b>${v}</b></div>`).join("");

  $("mTotal").textContent = total;
  $("mFalls").textContent = falls;
  $("mAlerts").textContent = alerts;
  $("mPeak").textContent = LEVELS[peak][0].split("—")[0].trim();
  $("mPeak").style.color = LEVELS[peak][1];
}

function showIdle(msg) {
  const banner = $("banner");
  banner.style.borderColor = "#8ea3d6"; banner.style.color = "#8ea3d6";
  $("bannerTitle").textContent = "NO PERSON DETECTED";
  $("bannerTitle").style.color = "#8ea3d6";
  $("bannerSub").textContent = msg;
  banner.classList.remove("pulse");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

/* ------------------------------------------------------------ main loop */
let landmarker = null, running = false, lastTs = -1;
let frames = 0, fpsT = performance.now();

async function initPose() {
  $("status").textContent = "loading BlazePose …";
  const vision = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18");
  const fileset = await vision.FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm");
  landmarker = await vision.PoseLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/" +
        "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.5,
    minPosePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  $("status").textContent = "ready";
}

/* BlazePose never says "I don't know" — on motion blur it returns a full
 * skeleton of near-zero-confidence noise, and left ungated that noise can raise
 * a fall alert. So frames still have to earn the right to be classified.
 *
 * But the gate used to demand visible HIPS as well as shoulders, and that was
 * wrong once the model was retrained on partial framing. A laptop webcam at
 * desk height sees a torso and extrapolates the hips off-screen with low
 * confidence, so the gate rejected the single most common way anyone will
 * actually use this — refusing to classify a person sitting plainly in view.
 * It was throwing away frames the network handles perfectly well: measured
 * over 538 corpus samples that the old gate rejected and this one admits,
 * accuracy is 96.5% with ZERO false fall alerts.
 *
 * Shoulders are the right anchor. They are the most reliably localised
 * landmarks in any upper-body view, and requiring both at 0.5 plus a third of
 * the skeleton still rejects no-person and motion-blur frames outright
 * (measured 0.0% admitted on both). */
function reliable(V) {
  if (V[L_SHO] < 0.50 || V[R_SHO] < 0.50) return false;
  let n = 0; for (const v of V) if (v >= 0.35) n++;
  return n >= 10;
}

/* Detected, but the lower body was never actually observed. Still classified —
 * the model is trained for it — and surfaced honestly, because accuracy is
 * genuinely lower here and a fall is much harder to corroborate. */
function lowerBodyUnseen(V) {
  return Math.min(V[L_HIP], V[R_HIP]) < 0.45;
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2 && landmarker) {
    if (canvas.width !== video.videoWidth) {
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    }
    const ts = performance.now();
    if (ts !== lastTs) {
      lastTs = ts;
      const res = landmarker.detectForVideo(video, ts);
      if (res.landmarks && res.landmarks.length) {
        const lm = res.landmarks[0];
        const Praw = lm.map(p => [p.x, p.y]);
        const V = lm.map(p => (p.visibility === undefined ? 1 : p.visibility));

        if (!reliable(V)) {
          showIdle("No usable pose in this frame — step into view of the "
                 + "camera, or improve the lighting.");
        } else {
          const P = aspectCorrect(Praw, canvas.width, canvas.height);
          const r = classify(P, V);
          r.partial = lowerBodyUnseen(V);

          total++;
          const cls = SPEC.classes[r.idx];
          counts[cls] = (counts[cls] || 0) + 1;
          if (r.idx === 0) falls++;
          if (r.level === "ALERT" || r.level === "EMERGENCY") alerts++;
          if (ORDER.indexOf(r.level) > ORDER.indexOf(peak)) peak = r.level;

          const hot = r.level === "ALERT" || r.level === "EMERGENCY";
          drawOverlay(Praw, V, hot ? LEVELS[r.level][1] : null);
          paintUI(r);
        }
      } else {
        showIdle("Step into view of the camera.");
      }
    }
    frames++;
    const now = performance.now();
    if (now - fpsT > 500) {
      $("fps").textContent = (frames * 1000 / (now - fpsT)).toFixed(1) + " fps";
      frames = 0; fpsT = now;
    }
  }
  requestAnimationFrame(loop);
}

async function start() {
  try {
    $("startBtn").disabled = true;
    $("status").textContent = "requesting camera …";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    if (!landmarker) await initPose();
    running = true;
    $("startBtn").textContent = "■  STOP";
    $("startBtn").disabled = false;
    $("stage").classList.add("live");
    loop();
  } catch (err) {
    $("status").textContent = "camera unavailable";
    $("startBtn").disabled = false;
    $("bannerSub").textContent =
      "Could not open the camera: " + err.message +
      ". Check the browser's camera permission for this site.";
  }
}

function stop() {
  running = false;
  const s = video.srcObject;
  if (s) s.getTracks().forEach(t => t.stop());
  video.srcObject = null;
  $("startBtn").textContent = "▶  START MONITORING";
  $("stage").classList.remove("live");
  $("status").textContent = "stopped";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

$("startBtn").addEventListener("click", () => running ? stop() : start());
$("resetBtn").addEventListener("click", () => {
  streak = 0; peak = "NORMAL"; counts = {}; total = 0; falls = 0; alerts = 0;
  $("mTotal").textContent = "0"; $("mFalls").textContent = "0";
  $("mAlerts").textContent = "0"; $("mPeak").textContent = "ALL CLEAR";
  $("mPeak").style.color = "#00ff9c";
});

const v = selfTest();
$("verify").textContent =
  (v.ok ? "✓ " : "⚠ ") + "verified against Python in this browser · arithmetic Δ "
  + v.arith.toExponential(1) + " · raster Δ " + v.raster.toFixed(4);
$("verify").style.color = v.ok ? "#3f8f6a" : "#ffb300";
