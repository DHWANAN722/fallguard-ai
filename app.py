"""
FallGuard AI — Elderly Fall Detection & Healthcare Monitoring Dashboard
=======================================================================

CRS Artificial Intelligence · Y2C1 Machine Learning and Deep Learning · FA-2
Step 7: Model Deployment using Streamlit.

Run locally:      streamlit run app.py
Deployed:         Streamlit Community Cloud (see README.md)

The dashboard is organised as four surfaces:

    IMAGE ANALYSIS     upload a photo → pose overlay, activity, alert, evidence
    VIDEO MONITORING   upload a clip  → per-frame timeline, event log, analytics
    LIVE SIMULATION    synthesise a scenario when no footage is to hand
    MODEL & METRICS    the training evidence, in-app

Everything the brief asks the dashboard to display — totals, fall count, normal
count, confidence, pose visualisation, emergency messaging, distribution charts
— is surfaced on the analysis tabs.
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src import theme, video as vid                                  # noqa: E402
from src.infer import (ALERT, EMERGENCY, LEVEL_STYLE, NORMAL, WATCH,  # noqa: E402
                       FallDetector)
from src.render import draw_overlay, render_cnn                      # noqa: E402
from src.skeleton import CLASS_NAMES, FALL, generate_sample          # noqa: E402

MODELS = os.path.join(ROOT, "models")
REPORTS = os.path.join(ROOT, "reports")

LEVEL_ICON = {NORMAL: "✔", WATCH: "◎", ALERT: "▲", EMERGENCY: "⛑"}

st.set_page_config(
    page_title="FallGuard AI · Elderly Fall Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(theme.css(), unsafe_allow_html=True)


# ==========================================================================
# resources
# ==========================================================================
@st.cache_resource(show_spinner=False)
def load_detector(cnn_t: float, bio_t: float, persist: int) -> FallDetector:
    return FallDetector(MODELS, cnn_prob_threshold=cnn_t,
                        biomech_threshold=bio_t, persistence_frames=persist)


@st.cache_resource(show_spinner=False)
def pose_status() -> tuple[bool, str]:
    from src import pose
    return pose.available()


@st.cache_data(show_spinner=False)
def load_metrics() -> dict | None:
    p = os.path.join(REPORTS, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def estimate_pose(frame_bgr: np.ndarray):
    from src import pose
    return pose.estimate(frame_bgr)


# ==========================================================================
# presentation helpers
# ==========================================================================
def card(title: str, body: str) -> str:
    return f'<div class="fg-card"><h4>{title}</h4>{body}</div>'


def metric_tile(label: str, value: str, colour: str) -> str:
    return card("", f'<div class="fg-metric"><div class="v" style="color:{colour};'
                    f'text-shadow:0 0 22px {colour}66">{value}</div>'
                    f'<div class="l">{label}</div></div>')


def alert_banner(level: str, detail: str) -> str:
    label, colour = LEVEL_STYLE[level]
    pulse = " fg-emergency" if level == EMERGENCY else ""
    return (
        f'<div class="fg-alert{pulse}" style="border-color:{colour}; color:{colour}">'
        f'<div class="icon">{LEVEL_ICON[level]}</div><div>'
        f'<div class="t" style="color:{colour}">{label}</div>'
        f'<div class="d">{detail}</div></div></div>'
    )


def prob_bars(probs: np.ndarray) -> str:
    rows = []
    for i in np.argsort(probs)[::-1]:
        name = CLASS_NAMES[i]
        col = theme.CLASS_COLOURS[name]
        pct = float(probs[i]) * 100
        rows.append(
            f'<div class="fg-bar"><div class="row">'
            f'<span style="color:{col}">{name}</span>'
            f'<span class="fg-mono" style="color:{col}">{pct:5.1f}%</span></div>'
            f'<div class="track"><div class="fill" style="width:{pct:.1f}%;'
            f'background:linear-gradient(90deg,{col}55,{col});'
            f'box-shadow:0 0 12px {col}99"></div></div></div>'
        )
    return "".join(rows)


def evidence_rows(pred) -> str:
    c = pred.clinical
    items = [
        ("Trunk inclination", f"{c['torso_angle']:.1f}°", "0° upright · 90° horizontal"),
        ("Bounding-box aspect", f"{c['aspect_ratio']:.2f}", "height ÷ width · &lt;1 = lying"),
        ("Pelvis height in frame", f"{c['pelvis_height']:.2f}", "1.0 = top · 0.0 = floor"),
        ("Leg verticality", f"{c['leg_verticality']:.2f}", "1.0 = legs under body"),
        ("Landmark visibility", f"{c['mean_visibility']:.2f}", "pose-estimator confidence"),
        ("Biomechanical score", f"{pred.biomech_score:.0%}", "rule-based, independent of CNN"),
    ]
    return "".join(
        f'<div class="fg-ev"><span>{k}<br><span class="fg-mono" '
        f'style="font-size:.72rem">{h}</span></span><b>{v}</b></div>'
        for k, v, h in items
    )


def bgr_to_png_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode() if ok else ""


def synth_frame(P: np.ndarray, V: np.ndarray, size: int = 460) -> np.ndarray:
    """Dark studio backdrop for skeletons that have no source photograph."""
    f = np.full((size, size, 3), 12, dtype=np.uint8)
    f[:, :, 0] = 26
    for g in range(0, size, 40):
        cv2.line(f, (0, g), (size, g), (34, 20, 12), 1)
        cv2.line(f, (g, 0), (g, size), (34, 20, 12), 1)
    cv2.line(f, (0, int(size * 0.94)), (size, int(size * 0.94)), (90, 60, 30), 2)
    return f


# ==========================================================================
# sidebar
# ==========================================================================
with st.sidebar:
    st.markdown('<div class="fg-title" style="font-size:1.5rem">FALLGUARD AI</div>'
                '<div class="fg-sub" style="font-size:.62rem">Detection Console</div>',
                unsafe_allow_html=True)
    st.markdown('<hr class="fg-rule">', unsafe_allow_html=True)

    st.markdown("#### Detection thresholds")
    cnn_t = st.slider("CNN fall probability", 0.10, 0.95, 0.55, 0.05,
                      help="Minimum network confidence for a fall vote.")
    bio_t = st.slider("Biomechanical score", 0.10, 0.95, 0.42, 0.02,
                      help="Minimum rule-based score for the corroborating vote. "
                           "0.42 is calibrated on held-out data.")
    persist = st.slider("Persistence (frames)", 1, 12, 4, 1,
                        help="Consecutive corroborated frames before EMERGENCY.")

    st.markdown('<hr class="fg-rule">', unsafe_allow_html=True)
    st.markdown("#### Video sampling")
    target_fps = st.select_slider("Analysis rate (fps)", [2, 4, 6, 8, 12], value=6)
    max_frames = st.select_slider("Max frames", [60, 120, 180, 240, 300], value=180)

    st.markdown('<hr class="fg-rule">', unsafe_allow_html=True)
    ok_pose, pose_msg = pose_status()
    model_ok = os.path.exists(os.path.join(MODELS, "fallguard_cnn.npz"))
    st.markdown(
        f'<div class="fg-mono">'
        f'<span class="fg-chip" style="color:{theme.LIME if model_ok else theme.RED}">'
        f'CNN {"ONLINE" if model_ok else "MISSING"}</span><br><br>'
        f'<span class="fg-chip" style="color:{theme.LIME if ok_pose else theme.AMBER}">'
        f'POSE {"ONLINE" if ok_pose else "OFFLINE"}</span><br>'
        f'<span style="font-size:.68rem">MediaPipe {pose_msg}</span>'
        f'</div>', unsafe_allow_html=True)

    if not ok_pose:
        st.warning("Pose estimation unavailable — image/video upload is disabled. "
                   "The Live Simulation tab still works.", icon="⚠️")

# ==========================================================================
# header
# ==========================================================================
st.markdown(
    '<div class="fg-title">FALLGUARD&nbsp;AI</div>'
    '<div class="fg-sub">Elderly Fall Detection &amp; Healthcare Monitoring</div>'
    '<hr class="fg-rule">', unsafe_allow_html=True)

if not model_ok:
    st.error("Trained model not found in `models/`. Run `python scripts/train.py` first.",
             icon="🚫")
    st.stop()

detector = load_detector(cnn_t, bio_t, persist)

tab_img, tab_vid, tab_sim, tab_model = st.tabs(
    ["◈  IMAGE ANALYSIS", "◈  VIDEO MONITORING", "◈  LIVE SIMULATION", "◈  MODEL & METRICS"]
)


# ==========================================================================
# helpers shared by the analysis tabs
# ==========================================================================
def render_result(frame: np.ndarray, pred, caption: str) -> None:
    """Two-column result: annotated frame | prediction, evidence, probabilities."""
    _, colour = LEVEL_STYLE[pred.level]
    detail = " · ".join(pred.reasons) if pred.reasons else "—"
    st.markdown(alert_banner(pred.level, detail), unsafe_allow_html=True)

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        overlay = draw_overlay(frame, pred.landmarks, pred.visibility)
        st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption=caption,
                 use_container_width=True)
    with right:
        cls_col = theme.CLASS_COLOURS[pred.label]
        st.markdown(card(
            "Predicted activity",
            f'<div class="fg-metric"><div class="v" style="color:{cls_col};'
            f'text-shadow:0 0 24px {cls_col}66;font-size:1.65rem">{pred.label}</div>'
            f'<div class="l">confidence {pred.confidence:.1%}</div></div>'),
            unsafe_allow_html=True)
        st.markdown("")
        st.markdown(card("Class probabilities", prob_bars(pred.probabilities)),
                    unsafe_allow_html=True)
        st.markdown("")
        st.markdown(card("Biomechanical evidence", evidence_rows(pred)),
                    unsafe_allow_html=True)


def analytics_block(detector: FallDetector, key: str) -> None:
    """Monitoring analytics: totals, distribution, timeline, event log."""
    import plotly.graph_objects as go

    s = detector.summary()
    tl = detector.timeline()
    if not s["total"]:
        return

    st.markdown("### Monitoring analytics")
    cols = st.columns(6)
    tiles = [
        ("Total detections", f"{s['total']}", theme.CYAN),
        ("Falls detected", f"{s['fall_frames']}", theme.RED),
        ("Normal activity", f"{s['normal_activity_frames']}", theme.VIOLET),
        ("Non-fall frames", f"{s['non_fall_frames']}", theme.LIME),
        ("Mean confidence", f"{s['mean_confidence']:.0%}", theme.AMBER),
        ("Alert frames", f"{s['alert_frames']}", theme.MAGENTA),
    ]
    for c, (l, v, col) in zip(cols, tiles):
        c.markdown(metric_tile(l, v, col), unsafe_allow_html=True)

    st.markdown("")
    c1, c2 = st.columns([1, 1.35], gap="large")

    with c1:
        counts = s["counts"]
        present = {k: v for k, v in counts.items() if v}
        fig = go.Figure(go.Bar(
            x=list(present.values()), y=list(present.keys()), orientation="h",
            marker=dict(color=[theme.CLASS_COLOURS[k] for k in present],
                        line=dict(color="rgba(0,0,0,.4)", width=1)),
            text=list(present.values()), textposition="outside",
        ))
        fig.update_layout(title="Activity distribution", showlegend=False,
                          **theme.plotly_layout(330))
        st.plotly_chart(fig, use_container_width=True, key=f"{key}_dist")

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=tl["t"], y=tl["fall_prob"], name="CNN fall probability",
            line=dict(color=theme.RED, width=2.4), fill="tozeroy",
            fillcolor="rgba(255,31,79,.16)"))
        fig.add_trace(go.Scatter(
            x=tl["t"], y=tl["biomech"], name="Biomechanical score",
            line=dict(color=theme.CYAN, width=2.2, dash="dot")))
        fig.add_hline(y=cnn_t, line=dict(color=theme.AMBER, width=1, dash="dash"),
                      annotation_text="threshold", annotation_font_color=theme.AMBER)
        fig.update_layout(title="Fall evidence over time", yaxis_range=[0, 1.05],
                          xaxis_title="seconds", **theme.plotly_layout(330))
        st.plotly_chart(fig, use_container_width=True, key=f"{key}_time")

    events = [p for p in detector.history if p.level in (ALERT, EMERGENCY)]
    if events:
        st.markdown("### Emergency event log")
        st.markdown(
            f'<div class="fg-mono" style="color:{theme.RED}">'
            f'{len(events)} corroborated fall frame(s) — '
            f'first at t = {events[0].timestamp:.2f}s</div>', unsafe_allow_html=True)
        df = pd.DataFrame([{
            "t (s)": round(p.timestamp, 2), "frame": p.frame_index,
            "level": p.level, "activity": p.label,
            "confidence": f"{p.confidence:.1%}",
            "CNN fall p": f"{p.probabilities[FALL]:.1%}",
            "biomech": f"{p.biomech_score:.0%}",
            "trunk °": f"{p.clinical['torso_angle']:.0f}",
        } for p in events])
        st.dataframe(df, use_container_width=True, hide_index=True, height=240)

        st.download_button(
            "⬇  Download incident report (CSV)",
            df.to_csv(index=False).encode(),
            file_name=f"fallguard_incident_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv", key=f"{key}_dl")


# ==========================================================================
# TAB 1 — image
# ==========================================================================
with tab_img:
    st.markdown("#### Upload a photograph for single-frame assessment")
    up = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"],
                          key="img_up", label_visibility="collapsed",
                          disabled=not ok_pose)

    if up is not None:
        data = np.frombuffer(up.getvalue(), np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Could not decode that image file.", icon="🚫")
        else:
            if max(frame.shape[:2]) > 1280:                 # cap cost + memory
                sc = 1280 / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)

            with st.spinner("Running BlazePose + CNN ..."):
                res = estimate_pose(frame)

            if res is None:
                st.markdown(alert_banner(
                    WATCH, "No person detected. Check framing, lighting and that the "
                           "subject is not heavily occluded."), unsafe_allow_html=True)
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                         caption="uploaded frame", use_container_width=True)
            else:
                P, V = res
                detector.reset()
                pred = detector.predict(P, V, timestamp=0.0, temporal=False)
                render_result(frame, pred, f"{up.name} · pose landmarks overlaid")

                # The brief requires the monitoring counters to be visible on
                # the dashboard; a user who only ever uploads a still should
                # still see them, so they are surfaced here too (n = 1).
                st.markdown('<hr class="fg-rule">', unsafe_allow_html=True)
                st.markdown("### Monitoring analytics")
                s = detector.summary()
                cols = st.columns(5)
                for c, (lab, val, col) in zip(cols, [
                    ("Total detections", f"{s['total']}", theme.CYAN),
                    ("Falls detected", f"{s['fall_frames']}", theme.RED),
                    ("Normal activity", f"{s['normal_activity_frames']}", theme.VIOLET),
                    ("Confidence", f"{pred.confidence:.0%}", theme.AMBER),
                    ("Alert level", LEVEL_STYLE[pred.level][0].split("—")[0].strip(),
                     LEVEL_STYLE[pred.level][1]),
                ]):
                    c.markdown(metric_tile(lab, val, col), unsafe_allow_html=True)

                with st.expander("What the CNN actually sees"):
                    a, b = st.columns([1, 2])
                    with a:
                        tensor = (render_cnn(P, V) * 255).astype(np.uint8)
                        st.image(cv2.resize(tensor, (256, 256),
                                            interpolation=cv2.INTER_NEAREST),
                                 caption="64×64×3 skeleton tensor")
                    with b:
                        st.markdown(
                            "The classifier never sees your photograph. BlazePose "
                            "reduces it to 33 landmarks, which are rasterised into "
                            "this three-channel tensor — **red = torso and head, "
                            "green = arms, blue = legs** — in *frame* coordinates, so "
                            "the subject's height in the room is preserved.\n\n"
                            "This is a deliberate privacy property: no identifiable "
                            "imagery reaches the model or would need to be retained "
                            "by a deployed monitoring system.")
    else:
        st.info("Upload a photo of a person standing, sitting, walking, bending or "
                "fallen. The system localises 33 body landmarks and classifies the "
                "posture.", icon="ℹ️")


# ==========================================================================
# TAB 2 — video
# ==========================================================================
with tab_vid:
    st.markdown("#### Upload footage for continuous monitoring")
    upv = st.file_uploader("Video", type=["mp4", "mov", "avi", "mkv", "webm"],
                           key="vid_up", label_visibility="collapsed",
                           disabled=not ok_pose)

    if upv is not None:
        import tempfile
        tmp = os.path.join(tempfile.gettempdir(), f"fg_in_{os.getpid()}_{upv.name}")
        with open(tmp, "wb") as fh:
            fh.write(upv.getvalue())

        try:
            info = vid.probe(tmp)
        except RuntimeError as exc:
            st.error(f"{exc}", icon="🚫")
            info = None

        if info:
            st.markdown(
                f'<div class="fg-mono">{info["width"]}×{info["height"]} · '
                f'{info["fps"]:.1f} fps source · {info["duration"]:.1f}s · '
                f'sampling at {target_fps} fps</div>', unsafe_allow_html=True)

            detector.reset()
            annotated: list[np.ndarray] = []
            no_person = 0
            bar = st.progress(0.0, text="Analysing frames ...")
            budget = min(max_frames, max(1, int(info["duration"] * target_fps) + 1))

            for n, (idx, ts, frame) in enumerate(
                    vid.iter_frames(tmp, target_fps, max_frames)):
                if max(frame.shape[:2]) > 960:
                    sc = 960 / max(frame.shape[:2])
                    frame = cv2.resize(frame, None, fx=sc, fy=sc,
                                       interpolation=cv2.INTER_AREA)
                res = estimate_pose(frame)
                if res is None:
                    no_person += 1
                else:
                    P, V = res
                    pred = detector.predict(P, V, timestamp=ts, frame_index=idx,
                                            temporal=True)
                    _, colour = LEVEL_STYLE[pred.level]
                    bgr = tuple(int(colour.lstrip("#")[i:i + 2], 16)
                                for i in (4, 2, 0))
                    ann = draw_overlay(frame, P, V,
                                       accent=bgr if pred.level in (ALERT, EMERGENCY)
                                       else None)
                    cv2.putText(ann, f"{pred.label}  {pred.confidence:.0%}",
                                (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                                bgr, 2, cv2.LINE_AA)
                    if pred.level in (ALERT, EMERGENCY):
                        cv2.rectangle(ann, (0, 0),
                                      (ann.shape[1] - 1, ann.shape[0] - 1), bgr, 6)
                    annotated.append(ann)
                bar.progress(min(1.0, (n + 1) / budget),
                             text=f"Analysing frame {n + 1} ...")
            bar.empty()

            s = detector.summary()
            if not s["total"]:
                st.markdown(alert_banner(
                    WATCH, "No person detected in any sampled frame."),
                    unsafe_allow_html=True)
            else:
                peak = s["peak_level"]
                worst = max((p for p in detector.history),
                            key=lambda p: float(p.probabilities[FALL]))
                detail = (f"{s['alert_frames']} corroborated fall frame(s) across "
                          f"{s['total']} analysed" if peak in (ALERT, EMERGENCY)
                          else f"{s['total']} frames analysed · no fall corroborated")
                st.markdown(alert_banner(peak, detail), unsafe_allow_html=True)

                if no_person:
                    st.caption(f"⚠ {no_person} sampled frame(s) contained no "
                               f"detectable person and were skipped.")

                clip = vid.write_annotated(annotated, fps=float(target_fps))
                playable = bool(clip) and clip[1] in vid.BROWSER_SAFE

                c1, c2 = st.columns([1.3, 1], gap="large")
                with c1:
                    if playable:
                        st.video(clip[0])
                        st.caption("Annotated monitoring feed")
                    elif annotated:
                        # no browser-decodable encoder on this host — show the
                        # frames directly rather than an inert video player
                        hi = max(range(len(annotated)),
                                 key=lambda i: detector.history[i].probabilities[FALL]
                                 if i < len(detector.history) else 0)
                        st.image(cv2.cvtColor(annotated[hi], cv2.COLOR_BGR2RGB),
                                 caption="Highest-risk annotated frame",
                                 use_container_width=True)
                        if clip:
                            st.download_button(
                                "⬇  Download annotated clip (MP4)",
                                open(clip[0], "rb").read(),
                                file_name="fallguard_annotated.mp4",
                                mime="video/mp4", key="vid_clip_dl")
                            st.caption("This host has no H.264 encoder, so the clip "
                                       "is MPEG-4 Part 2 — download it to view.")
                with c2:
                    st.markdown(card(
                        "Highest-risk frame",
                        f'<div class="fg-metric"><div class="v" '
                        f'style="color:{theme.RED};font-size:1.5rem">'
                        f'{worst.label}</div><div class="l">'
                        f't = {worst.timestamp:.2f}s · CNN fall '
                        f'{worst.probabilities[FALL]:.0%}</div></div>'),
                        unsafe_allow_html=True)
                    st.markdown("")
                    st.markdown(card("Class probabilities at that frame",
                                     prob_bars(worst.probabilities)),
                                unsafe_allow_html=True)

                st.markdown('<hr class="fg-rule">', unsafe_allow_html=True)
                analytics_block(detector, "vid")
    else:
        st.info("Upload a short clip (≤30 s works best). Frames are sampled at the "
                "analysis rate set in the sidebar; a fall must persist across "
                "consecutive frames, or show an impact-velocity signature, before "
                "the system escalates to EMERGENCY.", icon="ℹ️")


# ==========================================================================
# TAB 3 — simulation
# ==========================================================================
with tab_sim:
    st.markdown("#### Scenario simulator")
    st.caption("Generates skeletons from the same biomechanical model used to build "
               "the training corpus — useful for demonstrating the alert logic when "
               "no camera or footage is available.")

    c1, c2, c3 = st.columns([1.4, 1, 1])
    scenario = c1.selectbox(
        "Scenario",
        ["Fall incident (walking → collapse)", "Normal ambulation",
         "Sitting down", "Bending to pick something up (false-alarm test)",
         "Standing still"])
    n_frames = c2.slider("Frames", 8, 60, 24, 4)
    seed = c3.number_input("Seed", 0, 9999, 7, 1)

    if st.button("▶  RUN SIMULATION", use_container_width=True):
        script = {
            "Fall incident (walking → collapse)": [1] * 6 + [0] * 18,
            "Normal ambulation": [1] * 24,
            "Sitting down": [3] * 6 + [2] * 18,
            "Bending to pick something up (false-alarm test)": [3] * 5 + [4] * 14 + [3] * 5,
            "Standing still": [3] * 24,
        }[scenario]
        seq = [script[int(i * len(script) / n_frames)] for i in range(n_frames)]

        rng = np.random.default_rng(int(seed))
        detector.reset()
        frames = []
        for i, lab in enumerate(seq):
            P, V = generate_sample(int(lab), rng)
            pred = detector.predict(P, V, timestamp=i / 6.0, frame_index=i,
                                    temporal=True)
            base = synth_frame(P, V)
            _, colour = LEVEL_STYLE[pred.level]
            bgr = tuple(int(colour.lstrip("#")[j:j + 2], 16) for j in (4, 2, 0))
            ann = draw_overlay(base, P, V,
                               accent=bgr if pred.level in (ALERT, EMERGENCY) else None)
            cv2.putText(ann, f"{pred.label} {pred.confidence:.0%}", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, bgr, 2, cv2.LINE_AA)
            frames.append(ann)

        s = detector.summary()
        st.markdown(alert_banner(
            s["peak_level"],
            f"{s['alert_frames']} corroborated fall frame(s) across {s['total']} "
            f"simulated frames"), unsafe_allow_html=True)

        st.markdown("##### Frame strip")
        step = max(1, len(frames) // 8)
        picks = frames[::step][:8]
        for row_start in range(0, len(picks), 4):
            cols = st.columns(4)
            for c, f in zip(cols, picks[row_start:row_start + 4]):
                c.image(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), use_container_width=True)

        st.markdown('<hr class="fg-rule">', unsafe_allow_html=True)
        analytics_block(detector, "sim")


# ==========================================================================
# TAB 4 — model & metrics
# ==========================================================================
with tab_model:
    m = load_metrics()
    meta = detector.meta

    c = st.columns(4)
    c[0].markdown(metric_tile("CNN parameters", f"{meta['n_parameters']:,}", theme.CYAN),
                  unsafe_allow_html=True)
    c[1].markdown(metric_tile("Training samples",
                              f"{meta['dataset']['split']['train']:,}", theme.VIOLET),
                  unsafe_allow_html=True)
    if m:
        # the deployed model is whichever entry is the neural one; look it up by
        # name rather than hard-coding a key that changes with the architecture
        deployed = next((k for k in m["models"] if "CNN" in k), next(iter(m["models"])))
        c[2].markdown(metric_tile("Test accuracy",
                                  f"{m['models'][deployed]['accuracy']:.1%}", theme.LIME),
                      unsafe_allow_html=True)
        c[3].markdown(metric_tile("Fall recall",
                                  f"{m['models'][deployed]['fall_recall']:.1%}", theme.RED),
                      unsafe_allow_html=True)

    st.markdown('<hr class="fg-rule">', unsafe_allow_html=True)

    if m:
        if m.get("validation"):
            v = m["validation"]
            t = m["models"][deployed]
            st.markdown("### Validation vs test")
            st.caption("Close agreement between the two held-out splits is the "
                       "evidence that the test score is not itself overfitted.")
            st.dataframe(pd.DataFrame([
                {"split": "validation", "accuracy": f"{v['accuracy']:.4f}",
                 "F1 (macro)": f"{v['f1_macro']:.4f}",
                 "fall recall": f"{v['fall_recall']:.4f}",
                 "fall precision": f"{v['fall_precision']:.4f}"},
                {"split": "test", "accuracy": f"{t['accuracy']:.4f}",
                 "F1 (macro)": f"{t['f1_macro']:.4f}",
                 "fall recall": f"{t['fall_recall']:.4f}",
                 "fall precision": f"{t['fall_precision']:.4f}"},
            ]), use_container_width=True, hide_index=True)

        st.markdown("### Model comparison")
        st.caption("All rows scored on the same held-out test split.")
        rows = []
        for name, r in m["models"].items():
            rows.append({
                "model": name,
                "accuracy": f"{r['accuracy']:.4f}",
                "precision (macro)": f"{r['precision_macro']:.4f}",
                "recall (macro)": f"{r['recall_macro']:.4f}",
                "F1 (macro)": f"{r['f1_macro']:.4f}",
                "fall precision": f"{r['fall_precision']:.4f}",
                "fall recall": f"{r['fall_recall']:.4f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("### Per-class performance")
        st.dataframe(pd.DataFrame([
            {"class": k, "precision": f"{v['precision']:.4f}",
             "recall": f"{v['recall']:.4f}", "F1": f"{v['f1']:.4f}"}
            for k, v in m["per_class"].items()
        ]), use_container_width=True, hide_index=True)

    st.markdown("### Evaluation artefacts")
    figs = [("confusion_matrix.png", "Confusion matrix — test split"),
            ("training_curves.png", "Accuracy and loss vs epoch"),
            ("per_class_metrics.png", "Per-class precision / recall / F1"),
            ("model_comparison.png", "CNN vs Random Forest vs SVM")]
    shown = [(f, cap) for f, cap in figs if os.path.exists(os.path.join(REPORTS, f))]
    for i in range(0, len(shown), 2):
        cols = st.columns(2)
        for col, (f, cap) in zip(cols, shown[i:i + 2]):
            col.image(os.path.join(REPORTS, f), caption=cap, use_container_width=True)

    pred_dir = os.path.join(REPORTS, "predictions")
    if os.path.isdir(pred_dir):
        st.markdown("### Prediction gallery")
        grid = os.path.join(pred_dir, "07_grid_all_classes.png")
        if os.path.exists(grid):
            st.image(grid, caption="Predictions across all five activity classes",
                     use_container_width=True)
        combo = os.path.join(pred_dir, "06_false_alarm_test.png")
        if os.path.exists(combo):
            with st.expander("False-alarm test — a real fall beside a deep bend"):
                st.image(combo, use_container_width=True)
        with st.expander("Per-class prediction panels"):
            for f in sorted(os.listdir(pred_dir)):
                if f.startswith(("06", "07")) or not f.endswith(".png"):
                    continue
                st.image(os.path.join(pred_dir, f), use_container_width=True)

    st.markdown('<hr class="fg-rule">', unsafe_allow_html=True)
    st.markdown("### How a decision is made")
    st.markdown("""
Every frame passes through four stages:

1. **Pose estimation** — MediaPipe BlazePose returns 33 body landmarks with
   per-landmark visibility. No pixels travel further than this step.
2. **Rasterisation** — landmarks become a 64×64×3 tensor, channel-split by
   anatomical group, drawn in *frame* coordinates so height in the room is
   preserved.
3. **Classification** — the CNN outputs a distribution over the five activity
   classes.
4. **Corroboration** — an independent geometric rule scores the same skeleton on
   trunk inclination, aspect ratio, pelvis height and leg verticality. The two
   must agree before any alert is raised, and on video the agreement must
   persist across frames or coincide with a pelvis-descent impact signature
   before the system escalates to EMERGENCY.

Stage 4 is what separates a *fall* from *bending over to pick something up* —
the case that produces most false alarms in deployed systems, and the reason
`Normal Activity` exists as its own class in this model.
""")
