"""
theme.py — The cyberpunk visual system for the dashboard.

Everything visual lives here so ``app.py`` stays about behaviour. The palette
is built around three emissive accents on a near-black substrate, with an
important constraint: **alert state is never encoded by colour alone.** Every
status also carries a text label and an icon glyph, because a caregiver may be
colour-blind and because a red glow on a dark UI is easy to miss peripherally.
"""

CYAN = "#00e5ff"
MAGENTA = "#ff2d78"
LIME = "#8cff2b"
AMBER = "#ffb300"
VIOLET = "#b26bff"
RED = "#ff1f4f"

BG = "#05070f"
PANEL = "#0b1020"
PANEL_2 = "#111838"
GRID = "#1b2450"
TEXT = "#e6ecff"
MUTED = "#8ea3d6"

CLASS_COLOURS = {
    "Fall Detected": RED,
    "Walking": CYAN,
    "Sitting": AMBER,
    "Standing": LIME,
    "Bending": VIOLET,
}


def css() -> str:
    """Global stylesheet, injected once per session."""
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Rajdhani:wght@400;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

:root {{
  --cyan:{CYAN}; --magenta:{MAGENTA}; --lime:{LIME}; --amber:{AMBER};
  --violet:{VIOLET}; --red:{RED};
  --bg:{BG}; --panel:{PANEL}; --panel2:{PANEL_2}; --grid:{GRID};
  --text:{TEXT}; --muted:{MUTED};
}}

/* ---------- substrate: dark base + slow-drifting neon grid ---------- */
.stApp {{
  background:
    radial-gradient(1200px 700px at 12% -8%, rgba(0,229,255,.14), transparent 62%),
    radial-gradient(1000px 620px at 92% 4%, rgba(255,45,120,.13), transparent 60%),
    radial-gradient(900px 700px at 50% 108%, rgba(178,107,255,.10), transparent 62%),
    linear-gradient(180deg, #05070f 0%, #070b18 55%, #05070f 100%);
  color: var(--text);
  font-family: 'Rajdhani', system-ui, sans-serif;
}}
.stApp::before {{
  content:""; position:fixed; inset:-50%; z-index:0; pointer-events:none;
  background-image:
    linear-gradient(rgba(0,229,255,.055) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,229,255,.055) 1px, transparent 1px);
  background-size: 46px 46px;
  transform: perspective(420px) rotateX(58deg);
  animation: drift 22s linear infinite;
  opacity:.5;
}}
@keyframes drift {{ from {{ background-position:0 0; }} to {{ background-position:0 46px; }} }}

/* scanline veil */
.stApp::after {{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background: repeating-linear-gradient(180deg, rgba(255,255,255,.020) 0 1px, transparent 1px 3px);
  mix-blend-mode: overlay;
}}
.block-container {{ position:relative; z-index:1; padding-top:1.6rem; max-width:1500px; }}

h1,h2,h3,h4 {{ font-family:'Orbitron',sans-serif !important; letter-spacing:.045em; color:var(--text); }}

/* ---------- title ---------- */
.fg-title {{
  font-family:'Orbitron',sans-serif; font-weight:800;
  font-size: clamp(1.9rem, 4.6vw, 3.3rem); line-height:1.04; margin:0;
  background: linear-gradient(92deg, var(--cyan) 0%, #7be9ff 22%, var(--magenta) 62%, var(--violet) 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
  filter: drop-shadow(0 0 20px rgba(0,229,255,.42));
}}
.fg-sub {{ color:var(--muted); font-size:1.02rem; letter-spacing:.30em; text-transform:uppercase; margin-top:.35rem; }}
.fg-rule {{ height:2px; margin:1.0rem 0 1.3rem;
  background:linear-gradient(90deg,transparent,var(--cyan),var(--magenta),var(--violet),transparent);
  box-shadow:0 0 16px rgba(0,229,255,.55); border:0; }}

/* ---------- panels ---------- */
.fg-card {{
  background:linear-gradient(160deg, rgba(17,24,56,.90), rgba(11,16,32,.90));
  border:1px solid rgba(0,229,255,.26); border-radius:14px; padding:1.05rem 1.2rem;
  box-shadow:0 0 0 1px rgba(255,255,255,.03) inset, 0 10px 34px rgba(0,0,0,.55);
  position:relative; overflow:hidden; height:100%;
}}
.fg-card::before {{
  content:""; position:absolute; inset:0 0 auto 0; height:2px;
  background:linear-gradient(90deg,transparent,var(--cyan),transparent);
}}
.fg-card h4 {{ margin:.1rem 0 .55rem; font-size:.74rem; letter-spacing:.20em;
  text-transform:uppercase; color:var(--muted); font-family:'Rajdhani',sans-serif !important; font-weight:700; }}

/* ---------- metric tiles ---------- */
.fg-metric {{ text-align:left; }}
.fg-metric .v {{ font-family:'Orbitron',sans-serif; font-weight:800;
  font-size:2.15rem; line-height:1; }}
.fg-metric .l {{ color:var(--muted); font-size:.72rem; letter-spacing:.18em;
  text-transform:uppercase; margin-top:.42rem; }}

/* ---------- alert banner ---------- */
.fg-alert {{
  border-radius:16px; padding:1.25rem 1.5rem; margin:.3rem 0 1.1rem;
  display:flex; align-items:center; gap:1.15rem; border:2px solid;
  background:linear-gradient(100deg, rgba(11,16,32,.94), rgba(17,24,56,.80));
}}
.fg-alert .icon {{ font-size:2.5rem; line-height:1; filter:drop-shadow(0 0 12px currentColor); }}
.fg-alert .t {{ font-family:'Orbitron',sans-serif; font-weight:800;
  font-size:1.45rem; letter-spacing:.05em; }}
.fg-alert .d {{ color:var(--muted); font-size:.96rem; margin-top:.22rem; }}
.fg-emergency {{ animation: siren 1.05s ease-in-out infinite; }}
@keyframes siren {{
  0%,100% {{ box-shadow:0 0 0 0 rgba(255,31,79,.62), 0 0 30px rgba(255,31,79,.42) inset; }}
  50%     {{ box-shadow:0 0 0 20px rgba(255,31,79,0), 0 0 62px rgba(255,31,79,.18) inset; }}
}}
@media (prefers-reduced-motion: reduce) {{
  .fg-emergency {{ animation:none; }}
  .stApp::before {{ animation:none; }}
}}

/* ---------- probability bars ---------- */
.fg-bar {{ margin:.5rem 0; }}
.fg-bar .row {{ display:flex; justify-content:space-between; font-size:.9rem;
  margin-bottom:.22rem; font-weight:600; }}
.fg-bar .track {{ height:9px; border-radius:6px; background:rgba(255,255,255,.07); overflow:hidden; }}
.fg-bar .fill {{ height:100%; border-radius:6px; transition:width .45s cubic-bezier(.2,.8,.2,1); }}

/* ---------- misc ---------- */
.fg-chip {{ display:inline-block; padding:.20rem .68rem; border-radius:999px;
  font-size:.70rem; letter-spacing:.14em; text-transform:uppercase; font-weight:700;
  border:1px solid currentColor; margin-right:.4rem; }}
.fg-mono {{ font-family:'JetBrains Mono',monospace; font-size:.84rem; color:var(--muted); }}
.fg-ev {{ display:flex; justify-content:space-between; padding:.42rem 0;
  border-bottom:1px dashed rgba(255,255,255,.09); font-size:.94rem; }}
.fg-ev:last-child {{ border-bottom:0; }}
.fg-ev b {{ font-family:'JetBrains Mono',monospace; color:var(--text); }}

section[data-testid="stSidebar"] {{
  background:linear-gradient(180deg,#080c18,#0b1020);
  border-right:1px solid rgba(0,229,255,.20);
}}
.stButton>button {{
  background:linear-gradient(92deg, rgba(0,229,255,.16), rgba(255,45,120,.16));
  border:1px solid rgba(0,229,255,.50); color:var(--text);
  font-family:'Orbitron',sans-serif; font-weight:600; letter-spacing:.08em;
  border-radius:10px; transition:all .18s ease;
}}
.stButton>button:hover {{
  border-color:var(--magenta); box-shadow:0 0 22px rgba(255,45,120,.42);
  transform:translateY(-1px);
}}
[data-testid="stFileUploaderDropzone"] {{
  background:rgba(11,16,32,.72); border:1.6px dashed rgba(0,229,255,.42); border-radius:12px;
}}
.stTabs [data-baseweb="tab"] {{
  font-family:'Orbitron',sans-serif; font-size:.80rem; letter-spacing:.09em; color:var(--muted);
}}
.stTabs [aria-selected="true"] {{ color:var(--cyan) !important; }}
[data-testid="stMetricValue"] {{ font-family:'Orbitron',sans-serif; }}
hr {{ border-color:rgba(255,255,255,.10); }}
</style>
"""


def plotly_layout(height: int = 300) -> dict:
    """Shared Plotly styling so every chart matches the shell."""
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,32,0.55)",
        font=dict(color=TEXT, family="Rajdhani, sans-serif", size=13),
        margin=dict(l=10, r=10, t=42, b=10),
        height=height,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        # `title_font`, not `title=dict(font=...)` — callers pass their own
        # `title="..."` to update_layout, and a nested title dict here would
        # collide with it ("got multiple values for keyword argument 'title'").
        title_font=dict(family="Orbitron, sans-serif", size=14),
    )
