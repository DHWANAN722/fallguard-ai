"""
webcam.py — live in-browser monitoring component.

Streamlit runs Python on a server. A webcam frame lives in the browser. Any
design that classifies frames in Python must therefore ship every frame across
the network and wait for a reply, which is why ``st.camera_input`` is a shutter
button rather than a monitor: it is a request/response cycle, not a video feed.

So the model is moved to where the frames already are. ``scripts/export_web_model
.py`` folds BatchNorm away and packs the weights into 384 KB of float16;
``assets/live_monitor.js`` re-implements the NumPy inference runtime, the 126
feature descriptors and the biomechanical rule in JavaScript;
``assets/live_ui.js`` drives the camera loop. MediaPipe's WebAssembly build
supplies the same BlazePose landmarks the Python path uses.

Nothing leaves the machine: no frame is uploaded, and no server sees the video.
That is a privacy property worth having in an elder-care monitor, and it is
what makes framerate inference possible at all.

The port is verified rather than assumed. Four deliberately ambiguous golden
cases — two sitting at roughly 0.53 vs 0.47, where any transposed kernel or
off-by-one padding offset would move the numbers visibly — are embedded with
the probabilities Python produced, replayed on load, and the maximum deviation
is printed to the console and shown under the video.
"""

from __future__ import annotations

import functools
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_JSON = os.path.join(ROOT, "models", "fallguard_web.json")
ASSETS = os.path.join(ROOT, "assets")

HEIGHT = 900


@functools.lru_cache(maxsize=1)
def _payload() -> tuple[str, str, str]:
    with open(MODEL_JSON) as fh:
        spec = fh.read()
    with open(os.path.join(ASSETS, "live_monitor.js")) as fh:
        core = fh.read()
    with open(os.path.join(ASSETS, "live_ui.js")) as fh:
        ui = fh.read()
    return spec, core, ui


def available() -> tuple[bool, str]:
    """Whether the exported browser model is present."""
    if not os.path.exists(MODEL_JSON):
        return False, "models/fallguard_web.json missing — run scripts/export_web_model.py"
    kb = os.path.getsize(MODEL_JSON) / 1024
    return True, f"browser model ready ({kb:.0f} KB)"


CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Rajdhani:wght@500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0; background: transparent;
  font-family: 'Rajdhani', sans-serif; color: #e8f0ff;
}
.wrap { display: grid; grid-template-columns: 1.35fr 1fr; gap: 16px; }
@media (max-width: 900px) { .wrap { grid-template-columns: 1fr; } }
.panel {
  background: linear-gradient(160deg, rgba(15,20,40,.92), rgba(10,12,26,.96));
  border: 1px solid rgba(0,229,255,.28); border-radius: 14px; padding: 14px 16px;
  box-shadow: 0 0 26px rgba(0,229,255,.10), inset 0 0 40px rgba(0,229,255,.03);
}
.panel h3 {
  font-family: 'Orbitron', sans-serif; font-size: 12px; letter-spacing: .22em;
  text-transform: uppercase; color: #00e5ff; margin: 0 0 12px;
  text-shadow: 0 0 12px rgba(0,229,255,.55);
}
#stage {
  position: relative; border-radius: 12px; overflow: hidden; background: #05070f;
  border: 1px solid rgba(0,229,255,.22); aspect-ratio: 4/3;
}
#stage.live { border-color: rgba(0,255,156,.5); box-shadow: 0 0 30px rgba(0,255,156,.18); }
#vid, #cv { position: absolute; inset: 0; width: 100%; height: 100%; }
#vid { object-fit: cover; transform: scaleX(-1); }
#cv  { object-fit: cover; transform: scaleX(-1); pointer-events: none; }
.hud {
  position: absolute; top: 10px; left: 12px; right: 12px; display: flex;
  justify-content: space-between; font-family: 'JetBrains Mono', monospace;
  font-size: 11px; color: #7fe9ff; text-shadow: 0 0 8px rgba(0,0,0,.9);
  pointer-events: none; z-index: 3;
}
#banner {
  margin-top: 12px; border: 1px solid #8ea3d6; border-left-width: 5px;
  border-radius: 10px; padding: 12px 14px; background: rgba(0,0,0,.35);
}
#bannerTitle {
  font-family: 'Orbitron', sans-serif; font-weight: 800; font-size: 17px;
  letter-spacing: .12em;
}
#bannerSub { font-size: 14px; color: #a8bbe0; margin-top: 3px; }
.pulse { animation: pl 1s ease-in-out infinite; }
@keyframes pl { 0%,100% { opacity: 1 } 50% { opacity: .45 } }
@media (prefers-reduced-motion: reduce) { .pulse { animation: none } }
.btns { display: flex; gap: 10px; margin-top: 12px; }
button {
  font-family: 'Orbitron', sans-serif; font-size: 12px; letter-spacing: .14em;
  padding: 11px 18px; border-radius: 9px; cursor: pointer;
  background: linear-gradient(135deg, #00e5ff, #b26bff); color: #05070f;
  border: 0; font-weight: 800; transition: transform .12s, box-shadow .12s;
}
button:hover { transform: translateY(-1px); box-shadow: 0 0 20px rgba(0,229,255,.5); }
button:disabled { opacity: .5; cursor: wait; transform: none; }
#resetBtn { background: transparent; color: #7fe9ff; border: 1px solid rgba(0,229,255,.4); }
#predName {
  font-family: 'Orbitron', sans-serif; font-size: 22px; font-weight: 800;
  letter-spacing: .06em;
}
#predConf { font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #8ea3d6; }
.bar { margin-bottom: 9px; }
.bar .row { display: flex; justify-content: space-between; font-size: 13px;
            font-weight: 600; margin-bottom: 3px; }
.track { height: 7px; border-radius: 4px; background: rgba(255,255,255,.07); overflow: hidden; }
.fill { height: 100%; border-radius: 4px; transition: width .12s linear; }
.ev {
  display: flex; justify-content: space-between; font-size: 13px;
  padding: 5px 0; border-bottom: 1px dashed rgba(255,255,255,.08);
}
.ev b { font-family: 'JetBrains Mono', monospace; color: #00e5ff; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
.tile {
  background: rgba(0,229,255,.05); border: 1px solid rgba(0,229,255,.18);
  border-radius: 9px; padding: 9px 6px; text-align: center;
}
.tile b { display: block; font-family: 'Orbitron', sans-serif; font-size: 17px; color: #00ff9c; }
.tile span { font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: #8ea3d6; }
#verify {
  margin-top: 10px; font-family: 'JetBrains Mono', monospace; font-size: 10px;
  color: #5d7196; text-align: right;
}
.note { font-size: 12.5px; color: #8ea3d6; line-height: 1.5; margin-top: 10px; }
</style>
"""

BODY = """
<div class="wrap">
  <div>
    <div class="panel">
      <h3>Live Camera — On-Device Inference</h3>
      <div id="stage">
        <video id="vid" playsinline muted></video>
        <canvas id="cv"></canvas>
        <div class="hud"><span id="status">idle</span><span id="fps">— fps</span></div>
      </div>
      <div id="banner">
        <div id="bannerTitle">STANDBY</div>
        <div id="bannerSub">Press start — the camera opens and every frame is
          classified in this browser. No video is uploaded.</div>
      </div>
      <div class="btns">
        <button id="startBtn">&#9654;&nbsp; START MONITORING</button>
        <button id="resetBtn">RESET COUNTERS</button>
      </div>
      <div id="verify"></div>
    </div>
  </div>

  <div>
    <div class="panel">
      <h3>Current Classification</h3>
      <div id="predName" style="color:#8ea3d6">—</div>
      <div id="predConf">awaiting frames</div>
      <div style="margin-top:14px" id="bars"></div>
    </div>
    <div class="panel" style="margin-top:14px">
      <h3>Biomechanical Evidence</h3>
      <div id="evidence"><div class="note">Measurements appear once a pose is
        detected. The alert requires the network and this independent
        geometric rule to agree.</div></div>
    </div>
    <div class="panel" style="margin-top:14px">
      <h3>Session</h3>
      <div class="tiles">
        <div class="tile"><b id="mTotal">0</b><span>Frames</span></div>
        <div class="tile"><b id="mFalls">0</b><span>Fall</span></div>
        <div class="tile"><b id="mAlerts">0</b><span>Alerts</span></div>
        <div class="tile"><b id="mPeak" style="color:#00ff9c">ALL CLEAR</b><span>Peak</span></div>
      </div>
    </div>
  </div>
</div>
"""


def render(st) -> None:
    """Draw the live monitoring component into the current Streamlit container."""
    import streamlit.components.v1 as components

    ok, msg = available()
    if not ok:
        st.error(msg)
        return

    spec, core, ui = _payload()
    html = (
        "<!doctype html><html><head><meta charset='utf-8'>" + CSS + "</head><body>"
        + BODY
        + '<script type="module">\n'
        + "const SPEC = " + spec + ";\n"
        + core + "\n" + ui + "\n"
        + "</script></body></html>"
    )
    components.html(html, height=HEIGHT, scrolling=False)
