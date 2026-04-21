"""
Football Body Intelligence — Streamlit Dashboard
Bundesliga AWI + PQI Analytics Platform
"""
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.benchmark_report import generate_benchmark_summary

st.set_page_config(
    page_title="Football Body Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme detection ───────────────────────────────────────────────────────────
_theme_base = st.get_option("theme.base") or "dark"
IS_DARK = _theme_base != "light"

# ── Palette ───────────────────────────────────────────────────────────────────
C_AWI    = "#38BDF8"
C_PQI    = "#FB923C"
C_GOLD   = "#FBBF24"
C_GREEN  = "#34D399"
C_RED    = "#F87171"
C_PURPLE = "#A78BFA"

if IS_DARK:
    C_BG            = "#0A0E1A"
    C_SURFACE       = "#111827"
    C_BORDER        = "#1F2937"
    C_MUTED         = "#6B7280"
    C_TEXT          = "#F9FAFB"
    THEME           = "plotly_dark"
    LEGEND_BG       = "rgba(17,24,39,0.85)"
    MARKER_OUTLINE  = "white"
else:
    C_BG            = "#F8FAFC"
    C_SURFACE       = "#FFFFFF"
    C_BORDER        = "#CBD5E1"
    C_MUTED         = "#64748B"
    C_TEXT          = "#0F172A"
    THEME           = "plotly_white"
    LEGEND_BG       = "rgba(255,255,255,0.92)"
    MARKER_OUTLINE  = "#374151"

POS_COLORS: dict[str, str] = {
    "GK": "#94A3B8", "CB": "#38BDF8", "FB": "#818CF8",
    "DM": "#FB923C", "CM": "#FBBF24", "WM": "#34D399", "FW": "#F472B6",
}

POS_MAP: dict[str, str] = {
    "TW": "GK",
    "IVL": "CB", "IVR": "CB", "IVZ": "CB",
    "LA": "FB", "RA": "FB", "LV": "FB", "RV": "FB",
    "DML": "DM", "DMR": "DM", "DMZ": "DM", "DRM": "DM", "DLM": "DM",
    "ZO": "CM", "RM": "CM",
    "OHL": "WM", "OHR": "WM", "OLM": "WM", "ORM": "WM", "HL": "WM", "HR": "WM",
    "STL": "FW", "STR": "FW", "STZ": "FW",
}

TEAM_NAMES: dict[tuple, str] = {
    ("BVB-VFB", 1): "BVB", ("BVB-VFB", 0): "VfB",
    ("FCB-HSV", 1): "FCB", ("FCB-HSV", 0): "HSV",
    ("FCU-FCB", 1): "FCB", ("FCU-FCB", 0): "FCU",
    ("SGE-FCB", 1): "FCB", ("SGE-FCB", 0): "SGE",
    ("SGE-FCU", 1): "SGE", ("SGE-FCU", 0): "FCU",
}

POS_FULL_MAP: dict[str, str] = {
    "TW": "Goalkeeper", "IVL": "Centre-back (L)", "IVR": "Centre-back (R)",
    "IVZ": "Centre-back (C)", "LA": "Left back", "RA": "Right back",
    "LV": "Left wing-back", "RV": "Right wing-back",
    "DML": "Def. Mid (L)", "DMR": "Def. Mid (R)", "DMZ": "Def. Mid (C)",
    "DLM": "Def. Mid (L)", "DRM": "Def. Mid (R)",
    "ZO": "Central Mid", "RM": "Attacking Mid",
    "OHL": "Wide Att. (L)", "OHR": "Wide Att. (R)",
    "OLM": "Wide Mid (L)", "ORM": "Wide Mid (R)", "HL": "Wide (L)", "HR": "Wide (R)",
    "STL": "Striker (L)", "STR": "Striker (R)", "STZ": "Striker (C)",
}

PQI_COMPONENTS: dict[str, tuple[str, str]] = {
    "orientation_mean": (
        "Body Orientation",
        "How directly the player faces the ball carrier. 100 = square-on, 0 = facing 90° away. Weight: 40% of PQI.",
    ),
    "stance_mean": (
        "Stance Quality",
        "Knee flexion during pressing. Peaks at 130° — the biomechanically optimal pressing position. Weight: 30% of PQI.",
    ),
    "proximity_mean": (
        "Proximity",
        "Distance to ball carrier. 100 = 0 m, 0 = 5 m or beyond. Weight: 30% of PQI.",
    ),
}

POS_REF: dict[str, list[tuple[str, str]]] = {
    "GK": [("TW", "Goalkeeper")],
    "CB": [("IVL", "Centre-back L"), ("IVR", "Centre-back R"), ("IVZ", "Centre-back C")],
    "FB": [("LA", "Left back"), ("RA", "Right back"), ("LV", "Left wing-back"), ("RV", "Right wing-back")],
    "DM": [("DML", "Def. mid L"), ("DMR", "Def. mid R"), ("DMZ", "Def. mid C"), ("DLM", "Def. mid L"), ("DRM", "Def. mid R")],
    "CM": [("ZO", "Central mid"), ("RM", "Attacking mid")],
    "WM": [("OHL", "Wide att. L"), ("OHR", "Wide att. R"), ("OLM", "Wide mid L"), ("ORM", "Wide mid R"), ("HL", "Wide L"), ("HR", "Wide R")],
    "FW": [("STL", "Striker L"), ("STR", "Striker R"), ("STZ", "Striker C")],
}

ROLE_ORDER: list[str] = ["GK", "CB", "FB", "DM", "CM", "WM", "FW"]


# ── CSS ───────────────────────────────────────────────────────────────────────
def _inject_css() -> None:
    st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

  .main .block-container {{
      max-width: 1280px;
      padding: 0 2rem 3rem;
      margin: 0 auto;
  }}

  section[data-testid="stSidebar"] {{
      border-right: 1px solid {C_BORDER};
  }}

  div[data-testid="stTabs"] > div:first-child {{
      position: sticky;
      top: 0;
      z-index: 100;
      background-color: var(--background-color);
      border-bottom: 1px solid {C_BORDER};
      padding: 0.5rem 0 0;
  }}
  div[data-testid="stTabContent"] {{ padding-top: 1.5rem; }}

  div[data-testid="stPlotlyChart"] .modebar {{ display: none !important; }}

  .kpi {{
      background: {C_SURFACE};
      border: 1px solid {C_BORDER};
      border-radius: 10px;
      padding: 1rem 0.75rem 0.85rem;
      text-align: center;
      min-height: 96px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      gap: 2px;
  }}
  .kpi-label {{
      font-size: 0.6rem;
      font-weight: 600;
      color: {C_MUTED};
      text-transform: uppercase;
      letter-spacing: 0.1em;
  }}
  .kpi-value {{
      font-size: 1.8rem;
      font-weight: 700;
      line-height: 1.1;
      color: {C_TEXT};
  }}
  .kpi-sub {{
      font-size: 0.62rem;
      color: {C_MUTED};
  }}

  .sec {{
      font-size: 0.6rem;
      font-weight: 700;
      color: {C_MUTED};
      text-transform: uppercase;
      letter-spacing: 0.14em;
      border-bottom: 1px solid {C_BORDER};
      padding-bottom: 5px;
      margin: 1.5rem 0 0.9rem;
  }}

  .gcap {{
      text-align: center;
      font-size: 0.72rem;
      color: {C_MUTED};
      margin-top: -0.6rem;
      padding-bottom: 0.4rem;
  }}
  .gcap b {{ color: {C_TEXT}; font-weight: 600; }}

  /* ── KPI tooltip ── */
  .kpi-info {{
      position: relative;
      display: inline-block;
      cursor: help;
      color: {C_MUTED};
      font-size: 0.65rem;
      margin-left: 3px;
      vertical-align: middle;
  }}
  .kpi-info .kpi-tooltip {{
      visibility: hidden;
      opacity: 0;
      width: 200px;
      background: #1e2533;
      color: {C_TEXT};
      font-size: 0.7rem;
      font-weight: 400;
      text-transform: none;
      letter-spacing: 0;
      line-height: 1.5;
      text-align: left;
      border-radius: 6px;
      border: 1px solid {C_BORDER};
      padding: 7px 10px;
      position: absolute;
      z-index: 9999;
      bottom: 130%;
      left: 50%;
      transform: translateX(-50%);
      transition: opacity 0.15s ease;
      pointer-events: none;
      white-space: normal;
  }}
  .kpi-info .kpi-tooltip::after {{
      content: "";
      position: absolute;
      top: 100%;
      left: 50%;
      transform: translateX(-50%);
      border: 5px solid transparent;
      border-top-color: {C_BORDER};
  }}
  .kpi-info:hover .kpi-tooltip {{
      visibility: visible;
      opacity: 1;
  }}

  .ph-name {{ font-size: 1.65rem; font-weight: 700; color: {C_TEXT}; margin: 0; }}
  .ph-meta {{ font-size: 0.85rem; color: {C_MUTED}; margin-top: 2px; }}

  .pill {{
      display: inline-block;
      background: {C_BORDER};
      border-radius: 20px;
      padding: 3px 10px;
      font-size: 0.68rem;
      color: {C_MUTED};
      margin: 2px 3px;
  }}
  .pill-hi {{ background: rgba(56,189,248,0.12); color: {C_AWI}; }}
  .pill-pqi {{ background: rgba(251,146,60,0.12); color: {C_PQI}; }}
  .pill-green {{ background: rgba(52,211,153,0.12); color: {C_GREEN}; }}
  .pill-red {{ background: rgba(248,113,113,0.12); color: {C_RED}; }}

  .narrative-para {{
      font-size: 0.83rem;
      line-height: 1.75;
      color: {C_TEXT};
      margin-bottom: 0.75rem;
  }}
  .narrative-source {{
      font-size: 0.65rem;
      color: {C_MUTED};
      margin-top: 0.5rem;
      border-top: 1px solid {C_BORDER};
      padding-top: 0.5rem;
  }}
</style>
""", unsafe_allow_html=True)


_inject_css()


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_float(val: object, default: float = 0.0) -> float:
    try:
        return float(val or default)
    except (TypeError, ValueError):
        return default


def kpi(label: str, value: str, sub: str = "", color: str | None = None, help: str | None = None) -> None:
    col = color or C_TEXT
    info = (f' <span class="kpi-info">ℹ<span class="kpi-tooltip">{help}</span></span>'
            if help else "")
    st.markdown(
        f'<div class="kpi">'
        f'<div class="kpi-label">{label}{info}</div>'
        f'<div class="kpi-value" style="color:{col}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>', unsafe_allow_html=True)


def sec(title: str) -> None:
    st.markdown(f'<div class="sec">{title}</div>', unsafe_allow_html=True)


def sp(h: float = 0.5) -> None:
    st.markdown(f'<div style="height:{h}rem"></div>', unsafe_allow_html=True)


def tip(r: pd.Series) -> str:
    pqi = f"{r['mean_pqi']:.1f}" if pd.notna(r.get("mean_pqi")) else "n/a"
    pos = r.get("position", "—") or "—"
    return (
        f"<b>{r['name']}</b> &nbsp;#{int(r['jersey'])}<br>"
        f"<span style='color:{C_MUTED}'>{pos} · {r['match_id']} · {r['phase_label']}</span><br>"
        f"AWI <b>{r['awi_per_minute']:.2f}</b> scans/min &nbsp;·&nbsp; PQI <b>{pqi}</b>"
    )


def add_tips(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_tip"] = out.apply(tip, axis=1)
    return out


def guard(df: pd.DataFrame) -> bool:
    if df.empty:
        st.warning("No data matches the current filters.")
        return True
    return False


def pct_rank(series: pd.Series, val: float) -> int:
    return int((series.dropna() < val).mean() * 100)


def gauge_fig(value: float, title: str, color: str, axis_max: float, median: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"size": 11, "color": C_MUTED}},
        number={"font": {"size": 34, "color": C_TEXT}, "valueformat": ".2f"},
        gauge={
            "axis": {"range": [0, axis_max], "tickwidth": 1,
                     "tickcolor": C_BORDER, "tickfont": {"size": 9, "color": C_MUTED}},
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, axis_max * 0.33], "color": "rgba(255,255,255,0.02)"},
                {"range": [axis_max * 0.33, axis_max * 0.66], "color": "rgba(255,255,255,0.04)"},
                {"range": [axis_max * 0.66, axis_max], "color": "rgba(255,255,255,0.07)"},
            ],
            "threshold": {"line": {"color": C_MUTED, "width": 2},
                          "thickness": 0.75, "value": median},
        },
    ))
    fig.update_layout(height=240, margin=dict(t=50, b=10, l=20, r=20),
                      template=THEME, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)")
    return fig


def gauge_cap(value: float, median: float, unit: str) -> None:
    diff = value - median
    sign = "+" if diff >= 0 else ""
    col  = C_GREEN if diff >= 0 else C_RED
    st.markdown(
        f'<div class="gcap"><b>{value:.2f}</b> {unit} &nbsp;·&nbsp; '
        f'median <b>{median:.2f}</b> &nbsp;·&nbsp; '
        f'<span style="color:{col}">{sign}{diff:.2f}</span></div>',
        unsafe_allow_html=True)


def chart_layout(fig: go.Figure, h: int = 360, t: int = 20, b: int = 45, l: int = 45, r: int = 15) -> go.Figure:
    fig.update_layout(
        height=h, margin=dict(t=t, b=b, l=l, r=r),
        template=THEME, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=C_MUTED, size=11),
    )
    return fig


# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    for p in ("results/awi_full.csv", "results/pqi_full.csv"):
        if not os.path.exists(p):
            st.error(f"`{p}` not found — run the pipeline first.")
            st.stop()
    awi = pd.read_csv("results/awi_full.csv")
    pqi = pd.read_csv("results/pqi_full.csv")
    pqi_cols = ["jersey", "team", "match_id", "phase_label",
                "mean_pqi", "median_pqi", "std_pqi",
                "n_press_frames", "press_minutes",
                "orientation_mean", "stance_mean", "proximity_mean"]
    df = awi.merge(pqi[[c for c in pqi_cols if c in pqi.columns]],
                   on=["jersey", "team", "match_id", "phase_label"], how="left")
    df["pos_group"] = df["position"].map(POS_MAP)
    df["team_name"] = df.apply(
        lambda r: TEAM_NAMES.get((r["match_id"], r["team"]), str(r["team"])), axis=1)
    return df


@st.cache_data
def load_narratives() -> pd.DataFrame:
    path = "results/narratives.csv"
    if not os.path.exists(path):
        return pd.DataFrame(columns=["jersey", "team", "match_id", "phase_label", "narrative"])
    return pd.read_csv(path)


df = load_data()
narratives = load_narratives()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"<div style='font-size:1.05rem;font-weight:700;color:{C_TEXT}'>Football Body Intelligence</div>",
                unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.72rem;color:{C_MUTED};margin-bottom:0.5rem'>"
                f"AWI · PQI · Bundesliga Analysis</div>", unsafe_allow_html=True)
    st.divider()

    st.markdown(f"<div style='font-size:0.65rem;font-weight:600;color:{C_MUTED};"
                f"text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem'>"
                f"Filters</div>", unsafe_allow_html=True)

    sel_match = st.selectbox("Match", ["All"] + sorted(df["match_id"].unique()))
    sel_phase = st.selectbox("Phase", ["All", "1st half", "2nd half"])
    all_positions = sorted(df["position"].dropna().unique())
    sel_pos   = st.multiselect("Position", all_positions, placeholder="All positions")
    sel_cov   = st.slider("Min Coverage %", 0, 100, 50,
                          help="Exclude players tracked for less than this % of the phase")
    st.divider()

    with st.expander("Metric Definitions", expanded=False):
        st.markdown(f"""
<div style="font-size:0.72rem;line-height:1.6;color:{C_MUTED}">

<div style="color:{C_AWI};font-weight:700;margin-bottom:2px">AWI — Awareness Index</div>
Discrete head-rotation events per minute. A scan = head turn ≥45° within 0.5 s,
detected from 3D nose/neck/ear keypoints at 50 fps. Measures how actively a player
checks their surroundings before receiving or releasing the ball.

<div style="color:{C_PQI};font-weight:700;margin-top:10px;margin-bottom:2px">PQI — Press Quality Index (0–100)</div>
Composite score of body mechanics during pressing actions (player within 5 m of
ball carrier for ≥10 consecutive frames). Three sub-scores:

<div style="margin-top:6px;padding-left:8px;border-left:2px solid {C_BORDER}">
<span style="color:{C_TEXT};font-weight:600">Orientation (40%)</span><br>
How directly the player faces the ball carrier.
100 = square-on, 0 = facing 90° away.

<div style="margin-top:6px">
<span style="color:{C_TEXT};font-weight:600">Stance (30%)</span><br>
Knee flexion quality. Peaks at 130° — the biomechanically optimal
pressing position (low centre of gravity, ready to change direction).
A player pressing upright or with locked knees scores low.
</div>

<div style="margin-top:6px">
<span style="color:{C_TEXT};font-weight:600">Proximity (30%)</span><br>
Distance to ball carrier. 100 = 0 m, 0 = 5 m or beyond.
Measures whether the player closes down tight enough to be effective.
</div>
</div>

<div style="margin-top:8px;color:{C_MUTED}">
PQI = 0.40 × Orientation + 0.30 × Stance + 0.30 × Proximity
</div>
</div>
""", unsafe_allow_html=True)

    with st.expander("Position Code Reference", expanded=False):
        for group, codes in POS_REF.items():
            color = POS_COLORS.get(group, C_MUTED)
            st.markdown(
                f'<div style="margin-bottom:6px">'
                f'<span style="color:{color};font-weight:700;font-size:0.72rem">{group}</span>'
                f'<span style="color:{C_MUTED};font-size:0.65rem"> — '
                + " · ".join(f'<span style="color:{C_TEXT}">{code}</span> {name}' for code, name in codes)
                + '</span></div>',
                unsafe_allow_html=True,
            )

    st.caption("AWS World Sports Innovation Cup 2026")


# ── Filter ────────────────────────────────────────────────────────────────────
def apply_filters(
    base: pd.DataFrame,
    match: str,
    phase: str,
    positions: list[str],
    min_cov: int,
) -> pd.DataFrame:
    mask = pd.Series(True, index=base.index)
    if match != "All":
        mask &= base["match_id"] == match
    if phase != "All":
        mask &= base["phase_label"] == phase
    if positions:
        mask &= base["position"].isin(positions)
    if "coverage_pct" in base.columns:
        mask &= base["coverage_pct"] >= min_cov / 100
    mask &= base["awi_per_minute"] > 0
    mask &= base["pos_group"].notna()
    return base[mask].copy()


fdf = apply_filters(df, sel_match, sel_phase, sel_pos, sel_cov)
fdf = add_tips(fdf)


# ── Tab renderers ─────────────────────────────────────────────────────────────
def render_player_profile(fdf: pd.DataFrame) -> None:
    if guard(fdf):
        return

    player_labels = sorted(
        fdf.apply(lambda r: f"{r['name']}  (#{int(r['jersey'])})", axis=1).unique())
    sel_str    = st.selectbox("Select Player", player_labels, key="ps",
                              label_visibility="collapsed")
    sel_jersey = int(sel_str.split("#")[1].rstrip(")").strip())
    sel_name   = sel_str.split("  (#")[0]

    mask = (fdf["jersey"] == sel_jersey) & (fdf["name"] == sel_name)
    rows = fdf[mask]
    if rows.empty:
        st.warning("Player not found.")
        return
    p = rows.iloc[0]

    pos      = p["position"] if pd.notna(p.get("position")) else "—"
    pg       = p.get("pos_group", "—")
    pg_c     = POS_COLORS.get(pg, C_MUTED)
    pos_name = POS_FULL_MAP.get(pos, "")
    pos_suffix = f" ({pos_name})" if pos_name else ""
    st.markdown(
        f'<div class="ph-name">{p["name"]}</div>'
        f'<div class="ph-meta">#{int(p["jersey"])} &nbsp;·&nbsp; '
        f'<span style="color:{pg_c};font-weight:600">{pos}</span>'
        f'<span style="color:{C_MUTED};font-weight:400">{pos_suffix}</span>'
        f' &nbsp;·&nbsp; {p["match_id"]} &nbsp;·&nbsp; {p["phase_label"]}</div>',
        unsafe_allow_html=True)
    sp(0.5)

    # ── KPI row ───────────────────────────────────────────────────────────────
    sec("KEY METRICS")
    awi_val = safe_float(p.get("awi_per_minute"))
    pqi_val = safe_float(p.get("mean_pqi"))
    cov_val = safe_float(p.get("coverage_pct"))
    scans   = int(p.get("scan_count", 0) or 0)
    mins    = safe_float(p.get("total_minutes"))

    awi_pct = pct_rank(fdf["awi_per_minute"], awi_val)
    pqi_pct = pct_rank(fdf["mean_pqi"], pqi_val) if pqi_val > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1: kpi("AWI",         f"{awi_val:.2f}", f"Top {100-awi_pct}% · scans/min", C_AWI,
                 help="Awareness Index: head-rotation events per minute. Higher = more active scanning.")
    with c2: kpi("Mean PQI",    f"{pqi_val:.1f}", f"Top {100-pqi_pct}% · 0–100",    C_PQI,
                 help="Press Quality Index: composite of Orientation (40%), Stance (30%), Proximity (30%) during press frames.")
    with c3: kpi("Total Scans", f"{scans:,}",     f"{mins:.0f} min on pitch",
                 help="Total discrete head-rotation events detected in this phase.")
    with c4: kpi("Coverage",    f"{cov_val*100:.0f}%", "time on pitch",
                 help="Fraction of the phase where skeleton tracking data was available for this player.")
    with c5: kpi("Press Mins",
                 f"{safe_float(p.get('press_minutes')):.0f}",
                 "min in press frames",
                 help="Minutes spent within 5 m of the ball carrier for ≥10 consecutive frames — the PQI measurement window.")
    sp(0.75)

    # ── Gauges ────────────────────────────────────────────────────────────────
    sec("PERFORMANCE GAUGES")
    awi_max = max(fdf["awi_per_minute"].max() * 1.15, 1)
    awi_med = fdf["awi_per_minute"].median()
    pqi_med = fdf["mean_pqi"].median()

    g1, g2, _ = st.columns([1, 1, 0.04])
    with g1:
        st.plotly_chart(gauge_fig(round(awi_val, 2),
                                  "AWI — Awareness Index (scans/min)",
                                  C_AWI, awi_max, awi_med), width="stretch")
        gauge_cap(awi_val, awi_med, "scans/min")
    with g2:
        st.plotly_chart(gauge_fig(round(pqi_val, 1),
                                  "PQI — Press Quality Index (0–100)",
                                  C_PQI, 100, pqi_med), width="stretch")
        gauge_cap(pqi_val, pqi_med, "PQI")
    sp(0.5)

    # ── PQI radar + scatter ───────────────────────────────────────────────────
    sec("PLAYER IN CONTEXT")
    left, right = st.columns([3, 2], gap="medium")

    with left:
        fig_sc = px.scatter(
            fdf, x="awi_per_minute", y="mean_pqi",
            color="pos_group", color_discrete_map=POS_COLORS,
            custom_data=["_tip"], opacity=0.55, template=THEME,
            labels={"pos_group": "Role"},
        )
        fig_sc.update_traces(marker_size=7,
                             hovertemplate="%{customdata[0]}<extra></extra>")
        hl = fdf[mask]
        fig_sc.add_trace(go.Scatter(
            x=hl["awi_per_minute"], y=hl["mean_pqi"], mode="markers",
            marker=dict(size=20, color=C_GOLD, symbol="star",
                        line=dict(color=MARKER_OUTLINE, width=1.5)),
            name=sel_name,
            customdata=hl[["_tip"]].values,
            hovertemplate="%{customdata[0]}<extra></extra>",
            showlegend=True,
        ))
        fig_sc.add_hline(y=fdf["mean_pqi"].median(), line_dash="dot",
                         line_color=C_BORDER, line_width=1.5,
                         annotation_text="PQI median",
                         annotation_font=dict(color=C_MUTED, size=10),
                         annotation_position="bottom right")
        fig_sc.add_vline(x=fdf["awi_per_minute"].median(), line_dash="dot",
                         line_color=C_BORDER, line_width=1.5,
                         annotation_text="AWI median",
                         annotation_font=dict(color=C_MUTED, size=10),
                         annotation_position="top right")
        chart_layout(fig_sc, h=360, t=15, b=45, l=50, r=15)
        fig_sc.update_layout(
            xaxis_title="AWI (scans/min)", yaxis_title="Mean PQI",
            legend=dict(title="Role", orientation="v", x=1.01, y=1, font_size=10),
        )
        st.plotly_chart(fig_sc, width="stretch")

    with right:
        comp_keys   = ["orientation_mean", "stance_mean", "proximity_mean"]
        comp_labels = ["Orientation", "Stance", "Proximity"]
        player_vals = [safe_float(p.get(k)) for k in comp_keys]
        role_df     = fdf[fdf["pos_group"] == pg] if pg != "—" else fdf
        group_vals  = [role_df[k].mean() for k in comp_keys]

        if any(v > 0 for v in player_vals):
            fig_rad = go.Figure()
            cats = comp_labels + [comp_labels[0]]
            fig_rad.add_trace(go.Scatterpolar(
                r=player_vals + [player_vals[0]], theta=cats,
                fill="toself", name=sel_name,
                line_color=C_AWI, fillcolor="rgba(56,189,248,0.15)",
            ))
            fig_rad.add_trace(go.Scatterpolar(
                r=group_vals + [group_vals[0]], theta=cats,
                fill="toself", name=f"{pg} avg",
                line_color=C_PQI, fillcolor="rgba(251,146,60,0.1)",
                line_dash="dot",
            ))
            fig_rad.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 100],
                                   tickfont=dict(size=9, color=C_MUTED),
                                   gridcolor=C_BORDER, linecolor=C_BORDER),
                    angularaxis=dict(tickfont=dict(size=11, color=C_TEXT),
                                     gridcolor=C_BORDER, linecolor=C_BORDER),
                ),
                showlegend=True,
                legend=dict(orientation="h", y=-0.12, font_size=10),
                height=360, margin=dict(t=30, b=40, l=40, r=40),
                template=THEME, paper_bgcolor="rgba(0,0,0,0)",
                title=dict(text="PQI Sub-scores vs Role Average",
                           font=dict(size=12, color=C_MUTED), x=0.5),
            )
            st.plotly_chart(fig_rad, width="stretch")
        else:
            st.info("PQI sub-score data not available for this player.")

    # ── PQI component breakdown ───────────────────────────────────────────────
    avail_comp = {k: v for k, v in PQI_COMPONENTS.items()
                  if k in fdf.columns and pd.notna(p.get(k))}
    if avail_comp:
        sp(0.25)
        sec("PQI COMPONENT BREAKDOWN")
        comp_cols = st.columns(len(avail_comp), gap="small")
        for i, (col_key, (label, tooltip)) in enumerate(avail_comp.items()):
            val = float(p[col_key])
            med = fdf[col_key].median()
            pct = int((fdf[col_key].dropna() < val).mean() * 100)
            with comp_cols[i]:
                kpi(label, f"{val:.1f}", f"Median {med:.1f} · Top {100-pct}%",
                    help=tooltip)

    # ── Half-time trend ───────────────────────────────────────────────────────
    all_phases = fdf[fdf["name"] == sel_name].copy()
    if len(all_phases) >= 2:
        sp(0.25)
        sec("HALF-TIME PERFORMANCE TREND")
        trend_cols = st.columns(len(all_phases), gap="small")
        for i, (_, row) in enumerate(all_phases.iterrows()):
            with trend_cols[i]:
                awi_d = row["awi_per_minute"] - all_phases["awi_per_minute"].mean()
                col   = C_GREEN if awi_d >= 0 else C_RED
                kpi(f"{row['match_id']} · {row['phase_label']}",
                    f"{row['awi_per_minute']:.2f}",
                    f"PQI {row['mean_pqi']:.1f}" if pd.notna(row.get('mean_pqi')) else "PQI —",
                    col)

    # ── AI Scouting Narrative ─────────────────────────────────────────────────
    narr_row = narratives[
        (narratives["jersey"] == int(p["jersey"])) &
        (narratives["team"] == int(p["team"])) &
        (narratives["match_id"] == p["match_id"]) &
        (narratives["phase_label"] == p["phase_label"])
    ]
    if not narr_row.empty:
        sp(0.25)
        sec("AI SCOUTING NARRATIVE")
        with st.expander("View AI-generated scouting report", expanded=False):
            text = narr_row.iloc[0]["narrative"]
            for para in str(text).split("\n\n"):
                para = para.strip()
                if para:
                    st.markdown(
                        f'<p class="narrative-para">{para}</p>',
                        unsafe_allow_html=True,
                    )
            st.markdown(
                f'<div class="narrative-source">'
                f'Generated by Amazon Bedrock &nbsp;·&nbsp; {p["match_id"]} &nbsp;·&nbsp; {p["phase_label"]}'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_match_overview(fdf: pd.DataFrame) -> None:
    if guard(fdf):
        return

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    sec("MATCH SUMMARY")
    top_row = fdf.loc[fdf["awi_per_minute"].idxmax()]
    s1, s2, s3, s4, s5 = st.columns(5, gap="small")
    with s1: kpi("Players", str(fdf["name"].nunique()),
                 f"{fdf['match_id'].nunique()} match(es)",
                 help="Number of unique players included after applying the current filters.")
    with s2: kpi("Avg AWI",  f"{fdf['awi_per_minute'].mean():.2f}", "scans/min", C_AWI,
                 help="Average Awareness Index across all filtered players. AWI = head-rotation events per minute.")
    with s3: kpi("Avg PQI",  f"{fdf['mean_pqi'].mean():.1f}",       "0–100",     C_PQI,
                 help="Average Press Quality Index across all filtered players. Composite of Orientation (40%), Stance (30%), Proximity (30%).")
    with s4: kpi("Top AWI",  top_row["name"].split()[-1],
                 f"{top_row['awi_per_minute']:.2f} scans/min", C_GOLD,
                 help="Player with the highest AWI in the current filtered set.")
    with s5:
        n_elite = len(fdf[(fdf["awi_per_minute"] >= fdf["awi_per_minute"].quantile(0.75)) &
                          (fdf["mean_pqi"] >= fdf["mean_pqi"].quantile(0.75))]["name"].unique())
        kpi("Elite Players", str(n_elite), "top 25% AWI & PQI", C_GREEN,
            help="Players ranking in the top 25th percentile on both AWI and PQI simultaneously.")
    sp(0.75)

    # ── Quadrant scatter ──────────────────────────────────────────────────────
    sec("AWI vs PQI — PLAYER QUADRANT ANALYSIS")

    i1, i2, i3, i4 = st.columns(4, gap="small")
    with i1:
        st.markdown(f'<div style="font-size:0.72rem;color:{C_MUTED};padding:0.4rem 0">'
                    f'<span style="color:{C_AWI};font-weight:600">X-axis — AWI</span><br>'
                    f'Scanning rate (scans/min)</div>', unsafe_allow_html=True)
    with i2:
        st.markdown(f'<div style="font-size:0.72rem;color:{C_MUTED};padding:0.4rem 0">'
                    f'<span style="color:{C_PQI};font-weight:600">Y-axis — PQI</span><br>'
                    f'Pressing quality (0–100)</div>', unsafe_allow_html=True)
    with i3:
        st.markdown(f'<div style="font-size:0.72rem;color:{C_MUTED};padding:0.4rem 0">'
                    f'<span style="color:{C_MUTED};font-weight:600">Dashed lines</span><br>'
                    f'75th percentile threshold</div>', unsafe_allow_html=True)
    with i4:
        st.markdown(f'<div style="font-size:0.72rem;color:{C_MUTED};padding:0.4rem 0">'
                    f'<span style="color:{C_GREEN};font-weight:600">Green — Top performers</span><br>'
                    f'High on both AWI & PQI</div>', unsafe_allow_html=True)
    sp(0.25)

    awi_q = fdf["awi_per_minute"].quantile(0.75)
    pqi_q = fdf["mean_pqi"].quantile(0.75)

    fdf_q = fdf.copy()
    fdf_q["quadrant"] = fdf_q.apply(lambda r: (
        "Elite"      if r["awi_per_minute"] >= awi_q and r["mean_pqi"] >= pqi_q else
        "Cognitive"  if r["awi_per_minute"] >= awi_q else
        "Mechanical" if r["mean_pqi"] >= pqi_q else
        "Developing"
    ), axis=1)

    q_cfg = {
        "Elite":      (C_GREEN,  "Top performers (high AWI & PQI)",  10, 0.9),
        "Cognitive":  (C_AWI,    "High scanning, lower pressing",     8,  0.65),
        "Mechanical": (C_PQI,    "Strong pressing, lower scanning",   8,  0.65),
        "Developing": (C_MUTED,  "Below average on both",             7,  0.35),
    }
    fig_q = go.Figure()
    for quad, (color, label, size, opacity) in q_cfg.items():
        pts = fdf_q[fdf_q["quadrant"] == quad]
        if pts.empty:
            continue
        fig_q.add_trace(go.Scatter(
            x=pts["awi_per_minute"], y=pts["mean_pqi"],
            mode="markers", name=label,
            marker=dict(size=size, color=color, opacity=opacity,
                        symbol="circle", line=dict(width=0)),
            customdata=pts[["_tip"]].values,
            hovertemplate="%{customdata[0]}<extra></extra>",
        ))

    elite_pts = (fdf_q[fdf_q["quadrant"] == "Elite"]
                 .sort_values("awi_per_minute", ascending=False)
                 .drop_duplicates(subset="name")
                 .copy())
    elite_pts = elite_pts.sort_values("mean_pqi", ascending=False).reset_index(drop=True)
    pqi_range = fdf["mean_pqi"].max() - fdf["mean_pqi"].min()
    offset = max(pqi_range * 0.04, 0.5)
    y_offsets = [
        elite_pts.loc[i, "mean_pqi"] + (offset if i % 2 == 0 else -offset)
        for i in range(len(elite_pts))
    ]

    fig_q.add_trace(go.Scatter(
        x=elite_pts["awi_per_minute"],
        y=y_offsets,
        mode="text",
        text=elite_pts["name"].apply(lambda n: n.split()[-1]),
        textfont=dict(size=9, color=C_GREEN),
        showlegend=False, hoverinfo="skip",
    ))
    fig_q.add_hline(y=pqi_q, line_dash="dash", line_color=C_BORDER, line_width=1,
                    annotation_text=f"PQI 75th ({pqi_q:.1f})",
                    annotation_font=dict(color=C_MUTED, size=9),
                    annotation_position="bottom right")
    fig_q.add_vline(x=awi_q, line_dash="dash", line_color=C_BORDER, line_width=1,
                    annotation_text=f"AWI 75th ({awi_q:.1f})",
                    annotation_font=dict(color=C_MUTED, size=9),
                    annotation_position="top right")
    chart_layout(fig_q, h=420, t=15, b=55, l=55, r=15)
    fig_q.update_layout(
        xaxis_title="AWI (scans/min)", yaxis_title="Mean PQI",
        legend=dict(
            orientation="v",
            x=0.01, y=0.01,
            xanchor="left", yanchor="bottom",
            bgcolor=LEGEND_BG,
            bordercolor=C_BORDER, borderwidth=1,
            font_size=10, itemsizing="constant",
        ),
        template=THEME,
    )
    st.plotly_chart(fig_q, width="stretch")
    from src.quadrant_analysis import bootstrap_elite_quadrant
    bs = bootstrap_elite_quadrant(fdf)
    st.caption(
        f"Elite quadrant: {bs['observed_count']} players observed, "
        f"95pct bootstrap CI [{bs['ci_lower_95']:.1f}, {bs['ci_upper_95']:.1f}] "
        f"across 1000 resamples."
    )
    sp(0.25)

    # ── Position group analysis ───────────────────────────────────────────────
    sec("POSITION GROUP ANALYSIS")
    pa1, pa2 = st.columns(2, gap="medium")

    pos_stats = (fdf.groupby("pos_group")
                 .agg(awi_mean=("awi_per_minute", "mean"),
                      awi_std=("awi_per_minute", "std"),
                      pqi_mean=("mean_pqi", "mean"),
                      n=("name", "count"))
                 .reset_index()
                 .sort_values("awi_mean", ascending=False))

    with pa1:
        fig_lol = go.Figure()
        for _, row in pos_stats.iterrows():
            c = POS_COLORS.get(row["pos_group"], C_MUTED)
            fig_lol.add_trace(go.Scatter(
                x=[0, row["awi_mean"]], y=[row["pos_group"], row["pos_group"]],
                mode="lines", line=dict(color=C_BORDER, width=2),
                showlegend=False, hoverinfo="skip",
            ))
            fig_lol.add_trace(go.Scatter(
                x=[row["awi_mean"]], y=[row["pos_group"]],
                mode="markers",
                marker=dict(size=14, color=c, line=dict(color=MARKER_OUTLINE, width=1.5)),
                name=row["pos_group"],
                hovertemplate=f"<b>{row['pos_group']}</b><br>"
                              f"AWI: {row['awi_mean']:.2f} scans/min<br>"
                              f"n={int(row['n'])}<extra></extra>",
                showlegend=False,
            ))
        chart_layout(fig_lol, h=300, t=15, b=40, l=10, r=20)
        fig_lol.update_layout(
            title=dict(text="Avg AWI by Role", font_size=12, x=0, font_color=C_MUTED),
            xaxis_title="AWI (scans/min)",
            yaxis=dict(categoryorder="total ascending"),
        )
        st.plotly_chart(fig_lol, width="stretch")

    with pa2:
        sub_data = (fdf.groupby("pos_group")
                    [["orientation_mean", "stance_mean", "proximity_mean"]]
                    .mean().reset_index()
                    .sort_values("orientation_mean", ascending=False))

        fig_stack = go.Figure()
        fig_stack.add_trace(go.Bar(
            name="Orientation (40%)", x=sub_data["pos_group"],
            y=sub_data["orientation_mean"] * 0.4,
            marker_color=C_AWI, opacity=0.9,
            hovertemplate="<b>%{x}</b><br>Orientation: %{customdata:.1f}<extra></extra>",
            customdata=sub_data["orientation_mean"],
        ))
        fig_stack.add_trace(go.Bar(
            name="Stance (30%)", x=sub_data["pos_group"],
            y=sub_data["stance_mean"] * 0.3,
            marker_color=C_PURPLE, opacity=0.9,
            hovertemplate="<b>%{x}</b><br>Stance: %{customdata:.1f}<extra></extra>",
            customdata=sub_data["stance_mean"],
        ))
        fig_stack.add_trace(go.Bar(
            name="Proximity (30%)", x=sub_data["pos_group"],
            y=sub_data["proximity_mean"] * 0.3,
            marker_color=C_PQI, opacity=0.9,
            hovertemplate="<b>%{x}</b><br>Proximity: %{customdata:.1f}<extra></extra>",
            customdata=sub_data["proximity_mean"],
        ))
        chart_layout(fig_stack, h=300, t=15, b=40, l=45, r=15)
        fig_stack.update_layout(
            barmode="stack",
            title=dict(text="PQI Decomposition by Role", font_size=12, x=0, font_color=C_MUTED),
            yaxis_title="Weighted contribution",
            legend=dict(orientation="h", y=-0.3, font_size=10),
        )
        st.plotly_chart(fig_stack, width="stretch")

    st.divider()

    # ── Half-time delta ───────────────────────────────────────────────────────
    sec("HALF-TIME COGNITIVE FATIGUE — AWI DELTA (2nd − 1st Half)")
    h1 = (fdf[fdf["phase_label"] == "1st half"]
          [["name", "pos_group", "awi_per_minute", "mean_pqi"]]
          .rename(columns={"awi_per_minute": "awi_h1", "mean_pqi": "pqi_h1"}))
    h2 = (fdf[fdf["phase_label"] == "2nd half"]
          [["name", "awi_per_minute", "mean_pqi"]]
          .rename(columns={"awi_per_minute": "awi_h2", "mean_pqi": "pqi_h2"}))
    halves = h1.merge(h2, on="name", how="inner")

    if not halves.empty:
        halves["awi_delta"]  = halves["awi_h2"] - halves["awi_h1"]
        halves["pqi_delta"]  = halves["pqi_h2"] - halves["pqi_h1"]
        halves["short"]      = halves["name"].apply(lambda n: n.split()[-1])
        halves["bar_color"]  = halves["awi_delta"].apply(lambda d: C_GREEN if d >= 0 else C_RED)
        top_h = halves.nlargest(16, "awi_h1")

        d1, d2 = st.columns([2, 1], gap="medium")
        with d1:
            fig_d = go.Figure()
            fig_d.add_trace(go.Bar(
                x=top_h["short"], y=top_h["awi_delta"],
                marker_color=top_h["bar_color"].tolist(),
                marker_line_width=0, opacity=0.9,
                hovertemplate="<b>%{x}</b><br>AWI Δ: %{y:.2f} scans/min<extra></extra>",
                name="AWI Δ",
            ))
            fig_d.add_hline(y=0, line_color=C_BORDER, line_width=1.5)
            chart_layout(fig_d, h=320, t=15, b=70, l=50, r=15)
            fig_d.update_layout(
                title=dict(text="AWI Change 2nd vs 1st Half (top players by 1st-half AWI)",
                           font_size=11, x=0, font_color=C_MUTED),
                xaxis_tickangle=-35, yaxis_title="AWI Δ (scans/min)",
                showlegend=False,
            )
            st.plotly_chart(fig_d, width="stretch")

        with d2:
            fig_hh = px.scatter(
                halves, x="awi_h1", y="awi_h2",
                color="pos_group", color_discrete_map=POS_COLORS,
                hover_data={"name": True, "awi_h1": ":.2f", "awi_h2": ":.2f",
                            "pos_group": False},
                template=THEME, opacity=0.75,
                labels={"pos_group": "Role"},
            )
            mx = max(halves["awi_h1"].max(), halves["awi_h2"].max()) * 1.05
            fig_hh.add_trace(go.Scatter(
                x=[0, mx], y=[0, mx], mode="lines",
                line=dict(color=C_BORDER, dash="dot", width=1.5),
                showlegend=False, hoverinfo="skip",
            ))
            fig_hh.add_annotation(
                x=mx * 0.6, y=mx * 0.6,
                text="no change", showarrow=False,
                font=dict(size=9, color=C_MUTED),
                textangle=-45,
            )
            chart_layout(fig_hh, h=320, t=15, b=50, l=50, r=15)
            fig_hh.update_layout(
                title=dict(text="1st vs 2nd Half AWI",
                           font_size=11, x=0, font_color=C_MUTED),
                xaxis_title="AWI 1st half (scans/min)",
                yaxis_title="AWI 2nd half (scans/min)",
                legend=dict(orientation="v", x=1.01, y=1, font_size=9),
            )
            st.plotly_chart(fig_hh, width="stretch")
    else:
        st.info("Select 'All' phases to see the half-time comparison.")

    st.divider()

    # ── Team comparison ───────────────────────────────────────────────────────
    sec("TEAM AWI COMPARISON")
    team_stats = (fdf.groupby(["match_id", "team_name"])
                  .agg(awi_mean=("awi_per_minute", "mean"),
                       awi_med=("awi_per_minute", "median"),
                       n=("name", "count"))
                  .reset_index()
                  .sort_values("awi_mean", ascending=False))

    all_teams    = sorted(team_stats["team_name"].unique())
    team_palette = [C_AWI, C_PQI, C_GREEN, C_GOLD, C_PURPLE, C_RED, "#A8DADC", "#F472B6"]
    team_colors  = {t: team_palette[i % len(team_palette)] for i, t in enumerate(all_teams)}

    fig_team = go.Figure()
    for _, row in team_stats.iterrows():
        color = team_colors[row["team_name"]]
        fig_team.add_trace(go.Bar(
            x=[f"{row['team_name']}<br><span style='font-size:9px'>{row['match_id']}</span>"],
            y=[row["awi_mean"]],
            marker_color=color, opacity=0.85,
            hovertemplate=f"<b>{row['team_name']}</b> ({row['match_id']})<br>"
                          f"Avg AWI: {row['awi_mean']:.2f}<br>"
                          f"Median: {row['awi_med']:.2f}<br>"
                          f"n={int(row['n'])}<extra></extra>",
            showlegend=False,
        ))
    chart_layout(fig_team, h=280, t=15, b=50, l=50, r=15)
    fig_team.update_layout(
        yaxis_title="Avg AWI (scans/min)",
        bargap=0.35,
    )
    st.plotly_chart(fig_team, width="stretch")


def render_leaderboard(fdf: pd.DataFrame) -> None:
    if guard(fdf):
        return

    # ── Summary KPIs ──────────────────────────────────────────────────────────
    sec("OVERVIEW")
    o1, o2, o3, o4 = st.columns(4, gap="small")
    elite = fdf[(fdf["awi_per_minute"] >= fdf["awi_per_minute"].quantile(0.75)) &
                (fdf["mean_pqi"] >= fdf["mean_pqi"].quantile(0.75))]
    with o1: kpi("Total Players", str(fdf["name"].nunique()), "active player-phases",
                 help="Unique players with at least one tracked phase in the current filtered dataset.")
    with o2: kpi("Elite Quadrant", str(elite["name"].nunique()),
                 "top 25% AWI & PQI", C_GREEN,
                 help="Players ranking in the top 25th percentile on both AWI and PQI simultaneously.")
    with o3: kpi("Avg AWI",  f"{fdf['awi_per_minute'].mean():.2f}", "scans/min", C_AWI,
                 help="Average Awareness Index across all filtered players. AWI = head-rotation events per minute.")
    with o4: kpi("Avg PQI",  f"{fdf['mean_pqi'].mean():.1f}",       "0–100",     C_PQI,
                 help="Average Press Quality Index across all filtered players. Composite of Orientation (40%), Stance (30%), Proximity (30%).")
    sp(0.75)

    # ── Leaderboard table ─────────────────────────────────────────────────────
    sec("PLAYER RANKINGS")
    use_adjusted = st.toggle("Position-adjusted PQI", value=False)
    display_cols = ["jersey", "name", "position", "match_id",
                    "phase_label", "awi_per_minute", "mean_pqi", "coverage_pct",
                    "orientation_mean", "stance_mean", "proximity_mean"]
    if "pqi_position_adjusted" in fdf.columns:
        display_cols.append("pqi_position_adjusted")
    avail = [c for c in display_cols if c in fdf.columns]
    if use_adjusted:
        if "pqi_position_adjusted" in fdf.columns:
            sort_col = "pqi_position_adjusted"
        else:
            st.info("Position-adjusted PQI is not available in the loaded data. "
                    "Run normalize_pqi_by_position to generate this column.")
            sort_col = "mean_pqi"
    else:
        sort_col = "mean_pqi"
    tbl = fdf[avail].sort_values(sort_col, ascending=False).reset_index(drop=True)
    tbl.index += 1
    if "coverage_pct" in tbl.columns:
        tbl["coverage_pct"] = (tbl["coverage_pct"] * 100).round(1)

    st.dataframe(
        tbl, width="stretch",
        height=min(45 + len(tbl) * 35, 480),
        column_config={
            "jersey":                   st.column_config.NumberColumn("#",               format="%d"),
            "name":                     st.column_config.TextColumn("Player"),
            "position":                 st.column_config.TextColumn("Position"),
            "match_id":                 st.column_config.TextColumn("Match"),
            "phase_label":              st.column_config.TextColumn("Phase"),
            "awi_per_minute":           st.column_config.NumberColumn("AWI (scans/min)", format="%.2f"),
            "mean_pqi":                 st.column_config.NumberColumn("Mean PQI",        format="%.1f"),
            "pqi_position_adjusted":    st.column_config.NumberColumn("Adj. PQI (z)",   format="%.2f"),
            "coverage_pct":             st.column_config.NumberColumn("Coverage %",      format="%.1f"),
            "orientation_mean":         st.column_config.NumberColumn("Orientation",     format="%.1f"),
            "stance_mean":              st.column_config.NumberColumn("Stance",          format="%.1f"),
            "proximity_mean":           st.column_config.NumberColumn("Proximity",       format="%.1f"),
        },
    )
    sp(0.25)
    st.divider()

    # ── Role analytics ────────────────────────────────────────────────────────
    sec("ROLE ANALYTICS")
    hm1, hm2 = st.columns(2, gap="medium")

    hm_cols  = ["awi_per_minute", "mean_pqi", "orientation_mean", "stance_mean", "proximity_mean"]
    avail_hm = [c for c in hm_cols if c in fdf.columns]
    hm_labels = {"awi_per_minute": "AWI", "mean_pqi": "PQI",
                 "orientation_mean": "Orientation", "stance_mean": "Stance",
                 "proximity_mean": "Proximity"}

    role_avg = (fdf.groupby("pos_group")[avail_hm].mean()
                .dropna(how="all").reset_index())
    role_avg["_order"] = role_avg["pos_group"].map({r: i for i, r in enumerate(ROLE_ORDER)})
    role_avg = role_avg.sort_values("_order").drop(columns="_order")

    with hm1:
        if not role_avg.empty:
            hm_data = role_avg.set_index("pos_group")[avail_hm].copy()
            hm_data.columns = [hm_labels.get(c, c) for c in hm_data.columns]
            fig_hm = px.imshow(
                hm_data, text_auto=".1f",
                color_continuous_scale=[[0, C_SURFACE], [0.5, "#1D4ED8"], [1, C_AWI]],
                aspect="auto", template=THEME,
            )
            fig_hm.update_traces(
                hovertemplate="<b>%{y}</b> · %{x}<br>%{z:.1f}<extra></extra>")
            chart_layout(fig_hm, h=max(260, len(role_avg) * 50 + 70),
                         t=15, b=40, l=10, r=10)
            fig_hm.update_layout(
                title=dict(text="Role Averages Heatmap",
                           font_size=12, x=0, font_color=C_MUTED),
                xaxis_tickangle=-20, coloraxis_showscale=False,
            )
            st.plotly_chart(fig_hm, width="stretch")

    with hm2:
        role_awi = (fdf.groupby("pos_group")["awi_per_minute"]
                    .agg(mean="mean", std="std", count="count")
                    .reset_index())
        role_awi["_order"] = role_awi["pos_group"].map({r: i for i, r in enumerate(ROLE_ORDER)})
        role_awi = role_awi.sort_values(["_order", "mean"]).drop(columns="_order")
        role_awi["color"] = role_awi["pos_group"].map(POS_COLORS).fillna(C_MUTED)

        fig_bar = go.Figure()
        for _, row in role_awi.iterrows():
            fig_bar.add_trace(go.Bar(
                y=[row["pos_group"]],
                x=[row["mean"]],
                orientation="h",
                marker_color=row["color"],
                marker_line_width=0,
                error_x=dict(type="data", array=[row["std"]],
                             color=C_MUTED, thickness=1.5, width=5),
                hovertemplate=(
                    f"<b>{row['pos_group']}</b><br>"
                    f"Mean AWI: {row['mean']:.2f} scans/min<br>"
                    f"Std: ±{row['std']:.2f}<br>"
                    f"n={int(row['count'])}<extra></extra>"
                ),
                showlegend=False,
            ))
        chart_layout(fig_bar, h=max(260, len(role_awi) * 50 + 70),
                     t=15, b=50, l=10, r=30)
        fig_bar.update_layout(
            title=dict(text="Avg AWI by Role (mean ± std)",
                       font_size=12, x=0, font_color=C_MUTED),
            xaxis_title="AWI (scans/min)",
            bargap=0.3,
        )
        st.plotly_chart(fig_bar, width="stretch")

    if "mean_pqi" in fdf.columns:
        st.divider()
        sec("PQI DISTRIBUTION BY ROLE")
        role_order_present = [r for r in ROLE_ORDER if r in fdf["pos_group"].unique()]
        fig_pq = px.box(
            fdf.dropna(subset=["mean_pqi"]),
            x="pos_group", y="mean_pqi",
            color="pos_group", color_discrete_map=POS_COLORS,
            points="outliers",
            template=THEME,
            category_orders={"pos_group": role_order_present},
            labels={"mean_pqi": "Mean PQI", "pos_group": ""},
        )
        fig_pq.update_traces(
            hovertemplate="<b>%{x}</b><br>PQI: %{y:.1f}<extra></extra>")
        chart_layout(fig_pq, h=320, t=15, b=50, l=50, r=10)
        fig_pq.update_layout(showlegend=False)
        st.plotly_chart(fig_pq, width="stretch")


def render_fan_view(fdf: pd.DataFrame) -> None:
    """Fan Engagement View — simple, broadcast-ready Body Intelligence display."""
    if guard(fdf):
        return

    st.markdown(
        f'<div style="font-size:1.1rem;font-weight:700;color:{C_TEXT};margin-bottom:0.25rem">'
        f'⚡ Body Intelligence — Fan View</div>'
        f'<div style="font-size:0.78rem;color:{C_MUTED};margin-bottom:1.2rem">'
        f'Live-style awareness &amp; pressing leaderboard · Bundesliga 2025/26</div>',
        unsafe_allow_html=True,
    )

    # ── Top-3 awareness counter (broadcast-style) ─────────────────────────────
    sec("🧠 TOP SCANNERS — AWARENESS COUNTER")
    st.markdown(
        f'<div style="font-size:0.72rem;color:{C_MUTED};margin-bottom:0.75rem">'
        f'Head-rotation events per minute · higher = more active scanning before decisions</div>',
        unsafe_allow_html=True,
    )

    top_awi = (fdf.sort_values("awi_per_minute", ascending=False)
               .drop_duplicates(subset="name")
               .head(3)
               .reset_index(drop=True))

    medals = ["🥇", "🥈", "🥉"]
    cols_awi = st.columns(3, gap="medium")
    for i, (_, row) in enumerate(top_awi.iterrows()):
        with cols_awi[i]:
            pos_group = row.get("pos_group", "—")
            pos_color = POS_COLORS.get(pos_group, C_MUTED)
            awi_val   = row["awi_per_minute"]
            # Broadcast-style big number card
            st.markdown(f"""
<div style="
    background:{C_SURFACE};
    border:1px solid {C_BORDER};
    border-top: 3px solid {C_AWI};
    border-radius:12px;
    padding:1.2rem 1rem 1rem;
    text-align:center;
">
  <div style="font-size:1.6rem;margin-bottom:0.1rem">{medals[i]}</div>
  <div style="font-size:1.05rem;font-weight:700;color:{C_TEXT};line-height:1.2">{row['name']}</div>
  <div style="font-size:0.7rem;color:{pos_color};font-weight:600;margin:3px 0 8px">{row['position']} · {row['match_id']}</div>
  <div style="font-size:2.8rem;font-weight:800;color:{C_AWI};line-height:1">{awi_val:.1f}</div>
  <div style="font-size:0.65rem;color:{C_MUTED};margin-top:4px">scans / min</div>
</div>
""", unsafe_allow_html=True)
    sp(1.0)

    # ── 4-quadrant fan comparison ─────────────────────────────────────────────
    sec("🎯 PLAYER COMPARISON — BODY INTELLIGENCE QUADRANT")
    st.markdown(
        f'<div style="font-size:0.72rem;color:{C_MUTED};margin-bottom:0.75rem">'
        f'Every player ranked by scanning awareness (AWI) vs pressing quality (PQI). '
        f'Elite players are in the top-right. Tap any dot to see who it is.</div>',
        unsafe_allow_html=True,
    )

    awi_q75 = fdf["awi_per_minute"].quantile(0.75)
    pqi_q75 = fdf["mean_pqi"].quantile(0.75)

    fdf_fan = fdf.copy()
    fdf_fan["quadrant_label"] = fdf_fan.apply(lambda r: (
        "⭐ Elite"       if r["awi_per_minute"] >= awi_q75 and r["mean_pqi"] >= pqi_q75 else
        "🧠 Smart"       if r["awi_per_minute"] >= awi_q75 else
        "💪 Physical"    if r["mean_pqi"] >= pqi_q75 else
        "📈 Developing"
    ), axis=1)

    fan_colors = {
        "⭐ Elite":      C_GREEN,
        "🧠 Smart":      C_AWI,
        "💪 Physical":   C_PQI,
        "📈 Developing": C_MUTED,
    }

    fig_fan = px.scatter(
        fdf_fan,
        x="awi_per_minute", y="mean_pqi",
        color="quadrant_label",
        color_discrete_map=fan_colors,
        hover_data={"name": True, "position": True,
                    "awi_per_minute": ":.1f", "mean_pqi": ":.1f",
                    "quadrant_label": False},
        template=THEME,
        labels={
            "awi_per_minute": "Scanning Awareness (AWI)",
            "mean_pqi": "Pressing Quality (PQI)",
            "quadrant_label": "Type",
        },
        opacity=0.8,
    )
    fig_fan.update_traces(marker_size=10)

    # Label elite players
    elite_fan = (fdf_fan[fdf_fan["quadrant_label"] == "⭐ Elite"]
                 .drop_duplicates(subset="name")
                 .sort_values("awi_per_minute", ascending=False)
                 .head(8))
    fig_fan.add_trace(go.Scatter(
        x=elite_fan["awi_per_minute"],
        y=elite_fan["mean_pqi"] + 0.8,
        mode="text",
        text=elite_fan["name"].apply(lambda n: n.split()[-1]),
        textfont=dict(size=9, color=C_GREEN),
        showlegend=False, hoverinfo="skip",
    ))

    fig_fan.add_hline(y=pqi_q75, line_dash="dash", line_color=C_BORDER, line_width=1.5,
                      annotation_text="Top 25% pressing",
                      annotation_font=dict(color=C_MUTED, size=10),
                      annotation_position="bottom right")
    fig_fan.add_vline(x=awi_q75, line_dash="dash", line_color=C_BORDER, line_width=1.5,
                      annotation_text="Top 25% scanning",
                      annotation_font=dict(color=C_MUTED, size=10),
                      annotation_position="top right")

    # Quadrant labels
    x_max = fdf_fan["awi_per_minute"].max()
    y_max = fdf_fan["mean_pqi"].max()
    x_min = fdf_fan["awi_per_minute"].min()
    y_min = fdf_fan["mean_pqi"].min()

    for txt, x, y, col in [
        ("⭐ ELITE", x_max * 0.97, y_max * 0.99, C_GREEN),
        ("🧠 SMART", x_max * 0.97, y_min + (pqi_q75 - y_min) * 0.15, C_AWI),
        ("💪 PHYSICAL", x_min + (awi_q75 - x_min) * 0.15, y_max * 0.99, C_PQI),
    ]:
        fig_fan.add_annotation(
            x=x, y=y, text=txt, showarrow=False,
            font=dict(size=9, color=col), opacity=0.5,
            xanchor="right" if x > awi_q75 else "left",
            yanchor="top" if y > pqi_q75 else "bottom",
        )

    chart_layout(fig_fan, h=460, t=20, b=55, l=60, r=15)
    fig_fan.update_layout(
        legend=dict(
            title="Player Type", orientation="v",
            x=0.01, y=0.01, xanchor="left", yanchor="bottom",
            bgcolor=LEGEND_BG, bordercolor=C_BORDER, borderwidth=1,
            font_size=11,
        ),
    )
    st.plotly_chart(fig_fan, width="stretch")
    sp(0.5)

    # ── Awareness leaderboard (fan-friendly) ──────────────────────────────────
    sec("🏆 BODY INTELLIGENCE LEADERBOARD")
    st.markdown(
        f'<div style="font-size:0.72rem;color:{C_MUTED};margin-bottom:0.75rem">'
        f'Top 15 players ranked by combined Body Intelligence score '
        f'(AWI percentile + PQI percentile) / 2</div>',
        unsafe_allow_html=True,
    )

    lb = fdf.copy()
    lb["awi_pct"] = lb["awi_per_minute"].rank(pct=True) * 100
    lb["pqi_pct"] = lb["mean_pqi"].rank(pct=True) * 100
    lb["bi_score"] = (lb["awi_pct"] + lb["pqi_pct"]) / 2
    lb = (lb.sort_values("bi_score", ascending=False)
          .drop_duplicates(subset="name")
          .head(15)
          .reset_index(drop=True))
    lb.index += 1

    lb_display = lb[["name", "position", "match_id", "awi_per_minute", "mean_pqi", "bi_score"]].copy()
    lb_display.columns = ["Player", "Position", "Match", "AWI (scans/min)", "PQI", "Body Intelligence Score"]
    lb_display["AWI (scans/min)"] = lb_display["AWI (scans/min)"].round(2)
    lb_display["PQI"] = lb_display["PQI"].round(1)
    lb_display["Body Intelligence Score"] = lb_display["Body Intelligence Score"].round(1)

    st.dataframe(
        lb_display, width="stretch",
        height=min(45 + len(lb_display) * 35, 560),
        column_config={
            "Player":                  st.column_config.TextColumn("Player"),
            "Position":                st.column_config.TextColumn("Pos"),
            "Match":                   st.column_config.TextColumn("Match"),
            "AWI (scans/min)":         st.column_config.NumberColumn("🧠 AWI", format="%.2f",
                                           help="Scanning awareness: head-rotation events per minute"),
            "PQI":                     st.column_config.NumberColumn("💪 PQI", format="%.1f",
                                           help="Pressing quality: body mechanics score 0–100"),
            "Body Intelligence Score": st.column_config.ProgressColumn(
                                           "⭐ Body Intelligence",
                                           format="%.1f",
                                           min_value=0, max_value=100,
                                           help="Combined percentile rank on AWI and PQI"),
        },
    )
    sp(0.5)

    # ── Fun fact callout ──────────────────────────────────────────────────────
    if not fdf.empty:
        top_player = fdf.loc[fdf["awi_per_minute"].idxmax()]
        st.markdown(f"""
<div style="
    background: rgba(56,189,248,0.07);
    border: 1px solid rgba(56,189,248,0.25);
    border-left: 4px solid {C_AWI};
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    font-size: 0.82rem;
    color: {C_TEXT};
    line-height: 1.6;
">
  <span style="font-weight:700;color:{C_AWI}">Did you know?</span>
  &nbsp;{top_player['name'].split()[-1]} scanned <strong>{top_player['awi_per_minute']:.0f} times per minute</strong>
  during {top_player['match_id']} ({top_player['phase_label']}) — that's once every
  <strong>{60/top_player['awi_per_minute']:.1f} seconds</strong>.
  AWI spikes <strong>+57%</strong> in the 5 seconds before a player passes the ball.
  That's what "playing with your head up" looks like in data.
</div>
""", unsafe_allow_html=True)


# ── Cross-Domain Benchmark Tab ───────────────────────────────────────────────
def render_benchmark(fdf: pd.DataFrame) -> None:
    """Track 3: Cross-domain benchmarking against NFL, NBA, Tennis, Rugby, Aviation."""
    from src.benchmark_reference import (
        build_comparison_table,
        get_all_references,
        sample_reference_distribution,
    )

    st.markdown(
        f'<div style="font-size:1.1rem;font-weight:700;color:{C_TEXT};margin-bottom:0.25rem">'
        f'🌍 Cross-Domain Benchmark</div>'
        f'<div style="font-size:0.78rem;color:{C_MUTED};margin-bottom:1.2rem">'
        f'AWI and PQI placed in context of NFL, NBA, Tennis, Rugby and Aviation systems</div>',
        unsafe_allow_html=True,
    )

    # ── Concept lineage table ─────────────────────────────────────────────────
    sec("CONCEPT LINEAGE — WHERE AWI AND PQI COME FROM")
    st.markdown(
        f'<div style="font-size:0.72rem;color:{C_MUTED};margin-bottom:0.75rem">'
        f'Every sub-score in this platform maps directly to a concept already validated '
        f'at scale in another sport or domain. The table below shows the lineage.</div>',
        unsafe_allow_html=True,
    )

    refs = get_all_references()
    lineage_rows = []
    for r in refs:
        lineage_rows.append({
            "External System": r.system,
            "Sport / Domain": r.sport,
            "Concept": r.metric_name,
            "Maps to": r.metric_type.replace("_", " "),
            "Source": r.source,
        })
    lineage_df = pd.DataFrame(lineage_rows)
    st.dataframe(
        lineage_df,
        width="stretch",
        height=min(45 + len(lineage_df) * 38, 320),
        column_config={
            "External System": st.column_config.TextColumn("External System", width="medium"),
            "Sport / Domain":  st.column_config.TextColumn("Sport / Domain", width="small"),
            "Concept":         st.column_config.TextColumn("Concept", width="medium"),
            "Maps to":         st.column_config.TextColumn("Maps to AWI/PQI", width="small"),
            "Source":          st.column_config.TextColumn("Source", width="large"),
        },
        hide_index=True,
    )
    sp(0.5)

    # ── Player selector for percentile comparison ─────────────────────────────
    sec("PLAYER PERCENTILE RANKING VS CROSS-DOMAIN REFERENCES")
    st.markdown(
        f'<div style="font-size:0.72rem;color:{C_MUTED};margin-bottom:0.75rem">'
        f'Select a player to see where their AWI and PQI sub-scores rank within '
        f'each reference system\'s population and elite cohort.</div>',
        unsafe_allow_html=True,
    )

    if guard(fdf):
        return

    player_options = (
        fdf.dropna(subset=["awi_per_minute"])
        .sort_values("awi_per_minute", ascending=False)
        .drop_duplicates(subset="name")["name"]
        .tolist()
    )
    selected_name = st.selectbox(
        "Select player", player_options, key="bench_player_select",
        label_visibility="collapsed",
    )

    player_rows = fdf[fdf["name"] == selected_name]
    if player_rows.empty:
        st.warning("No data for selected player.")
        return

    # Use best phase (highest AWI) for the comparison
    best = player_rows.sort_values("awi_per_minute", ascending=False).iloc[0]

    awi_val  = float(best["awi_per_minute"]) if pd.notna(best.get("awi_per_minute")) else None
    pqi_val  = float(best["mean_pqi"])       if pd.notna(best.get("mean_pqi"))       else None
    ori_val  = float(best["orientation_mean"]) if pd.notna(best.get("orientation_mean")) else None
    sta_val  = float(best["stance_mean"])    if pd.notna(best.get("stance_mean"))    else None
    prox_val = float(best["proximity_mean"]) if pd.notna(best.get("proximity_mean")) else None

    # KPI row
    cols_kpi = st.columns(5, gap="small")
    kpi_vals = [
        ("AWI", f"{awi_val:.1f}" if awi_val else "n/a", "scans/min", C_AWI),
        ("PQI", f"{pqi_val:.1f}" if pqi_val else "n/a", "composite", C_PQI),
        ("Orientation", f"{ori_val:.1f}" if ori_val else "n/a", "sub-score", C_PURPLE),
        ("Stance", f"{sta_val:.1f}" if sta_val else "n/a", "sub-score", C_GREEN),
        ("Proximity", f"{prox_val:.1f}" if prox_val else "n/a", "sub-score", C_GOLD),
    ]
    for col, (label, val, sub, color) in zip(cols_kpi, kpi_vals):
        with col:
            kpi(label, val, sub, color)
    sp(0.5)

    # Build comparison table
    rows = build_comparison_table(
        awi_value=awi_val,
        pqi_value=pqi_val,
        orientation_value=ori_val,
        stance_value=sta_val,
        proximity_value=prox_val,
    )

    if not rows:
        st.info("Not enough data to build cross-domain comparison for this player.")
        return

    cmp_df = pd.DataFrame(rows)

    # ── Percentile bar chart ──────────────────────────────────────────────────
    fig_pct = go.Figure()
    fig_pct.add_trace(go.Bar(
        name="vs Population",
        x=cmp_df["system"],
        y=cmp_df["pct_vs_population"],
        marker_color=C_AWI,
        opacity=0.85,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Player value: %{customdata[0]:.1f} %{customdata[1]}<br>"
            "Ref mean: %{customdata[2]:.1f}<br>"
            "Percentile vs population: <b>%{y:.0f}th</b><extra></extra>"
        ),
        customdata=cmp_df[["player_value", "unit", "ref_mean"]].values,
    ))
    fig_pct.add_trace(go.Bar(
        name="vs Elite cohort",
        x=cmp_df["system"],
        y=cmp_df["pct_vs_elite"],
        marker_color=C_PQI,
        opacity=0.85,
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Player value: %{customdata[0]:.1f} %{customdata[1]}<br>"
            "Elite mean: %{customdata[2]:.1f}<br>"
            "Percentile vs elite: <b>%{y:.0f}th</b><extra></extra>"
        ),
        customdata=cmp_df[["player_value", "unit", "ref_elite_mean"]].values,
    ))
    fig_pct.add_hline(
        y=50, line_dash="dash", line_color=C_BORDER, line_width=1.5,
        annotation_text="50th percentile",
        annotation_font=dict(color=C_MUTED, size=10),
        annotation_position="bottom right",
    )
    chart_layout(fig_pct, h=360, t=30, b=120, l=55, r=15)
    fig_pct.update_layout(
        barmode="group",
        title=dict(
            text=f"{selected_name} — percentile rank within each reference system",
            font_size=12, x=0, font_color=C_MUTED,
        ),
        yaxis=dict(title="Percentile", range=[0, 105]),
        xaxis=dict(tickangle=-30),
        legend=dict(
            orientation="h", x=0.01, y=1.08, xanchor="left",
            bgcolor=LEGEND_BG, bordercolor=C_BORDER, borderwidth=1, font_size=11,
        ),
    )
    st.plotly_chart(fig_pct, width="stretch")
    sp(0.5)

    # ── Distribution overlay charts ───────────────────────────────────────────
    sec("DISTRIBUTION OVERLAYS — PLAYER VALUE VS REFERENCE POPULATION")
    st.markdown(
        f'<div style="font-size:0.72rem;color:{C_MUTED};margin-bottom:0.75rem">'
        f'Each chart shows the reference population distribution (shaded) with the '
        f'selected player\'s value marked. The dashed line is the elite cohort mean.</div>',
        unsafe_allow_html=True,
    )

    ref_lookup = {r.system: r for r in get_all_references()}
    n_charts = len(rows)
    cols_per_row = 3
    for row_start in range(0, n_charts, cols_per_row):
        chunk = rows[row_start: row_start + cols_per_row]
        cols = st.columns(len(chunk), gap="medium")
        for col, row_data in zip(cols, chunk):
            ref = ref_lookup[row_data["system"]]
            samples = sample_reference_distribution(ref, n=800, cohort="population")
            player_val = row_data["player_value"]

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=samples,
                nbinsx=30,
                marker_color=C_BORDER,
                opacity=0.6,
                name="Reference population",
                hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
            ))
            fig_dist.add_vline(
                x=player_val,
                line_color=C_AWI, line_width=2.5,
                annotation_text=f"{selected_name.split()[-1]}: {player_val:.1f}",
                annotation_font=dict(color=C_AWI, size=9),
                annotation_position="top right",
            )
            fig_dist.add_vline(
                x=ref.elite_mean,
                line_dash="dash", line_color=C_GOLD, line_width=1.5,
                annotation_text=f"Elite avg: {ref.elite_mean:.0f}",
                annotation_font=dict(color=C_GOLD, size=9),
                annotation_position="top left",
            )
            chart_layout(fig_dist, h=220, t=30, b=40, l=35, r=10)
            fig_dist.update_layout(
                showlegend=False,
                title=dict(
                    text=f"{ref.system}<br><span style='font-size:9px'>{row_data['metric_name']}</span>",
                    font_size=10, x=0, font_color=C_MUTED,
                ),
                xaxis_title=row_data["unit"],
                yaxis_title="",
            )
            with col:
                st.plotly_chart(fig_dist, width="stretch")
    sp(0.5)

    # ── Comparison summary table ──────────────────────────────────────────────
    sec("FULL COMPARISON TABLE")
    disp = cmp_df[[
        "system", "sport", "metric_name", "player_value",
        "ref_mean", "ref_elite_mean", "pct_vs_population", "pct_vs_elite", "unit",
    ]].copy()
    disp.columns = [
        "System", "Sport", "Metric", "Player Value",
        "Ref Mean", "Elite Mean", "Pct vs Population", "Pct vs Elite", "Unit",
    ]
    st.dataframe(
        disp,
        width="stretch",
        height=min(45 + len(disp) * 38, 400),
        column_config={
            "System":            st.column_config.TextColumn("System"),
            "Sport":             st.column_config.TextColumn("Sport"),
            "Metric":            st.column_config.TextColumn("Metric"),
            "Player Value":      st.column_config.NumberColumn("Player", format="%.2f"),
            "Ref Mean":          st.column_config.NumberColumn("Pop. Mean", format="%.1f"),
            "Elite Mean":        st.column_config.NumberColumn("Elite Mean", format="%.1f"),
            "Pct vs Population": st.column_config.ProgressColumn(
                "vs Population", format="%.0f%%", min_value=0, max_value=100),
            "Pct vs Elite":      st.column_config.ProgressColumn(
                "vs Elite", format="%.0f%%", min_value=0, max_value=100),
            "Unit":              st.column_config.TextColumn("Unit"),
        },
        hide_index=True,
    )
    sp(0.5)

    # ── Source callout ────────────────────────────────────────────────────────
    with st.expander("Sources and methodology", expanded=False):
        st.markdown(
            f'<div style="font-size:0.75rem;color:{C_MUTED};line-height:1.7">'
            f'Reference distributions are parameterised from published summary statistics. '
            f'No external data is fetched at runtime. Percentiles use a normal CDF '
            f'approximation (mean, std from literature). Elite cohort = top-quartile '
            f'performers in each reference system.</div>',
            unsafe_allow_html=True,
        )
        for r in refs:
            st.markdown(
                f'<div style="font-size:0.72rem;color:{C_TEXT};margin-top:0.5rem">'
                f'<b>{r.system}</b> ({r.sport}): {r.concept}<br>'
                f'<span style="color:{C_MUTED}">Source: {r.source}</span></div>',
                unsafe_allow_html=True,
            )


def render_broadcast_demo_tab(fdf: pd.DataFrame) -> None:
    """Broadcast Demo tab - embeds broadcast overlay inline."""
    from dashboard.broadcast_demo import render_broadcast_overlay, load_broadcast_data
    # Load full unfiltered dataset for the overlay
    broadcast_df = load_broadcast_data()
    render_broadcast_overlay(broadcast_df, standalone=False)
    st.caption(
        "Simulated broadcast overlay. AWI and PQI are real-time capable "
        "with under 15 seconds latency at 50 fps."
    )


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Player Profile", "Match Overview", "Leaderboard", "⚡ Fan View", "🌍 Cross-Domain Benchmark",
    "📊 Cross-Sport Context", "📡 Broadcast Demo",
])

with tab1:
    render_player_profile(fdf)

with tab2:
    render_match_overview(fdf)

with tab3:
    render_leaderboard(fdf)

with tab4:
    render_fan_view(fdf)

with tab5:
    render_benchmark(fdf)

with tab6:
    # ── Cross-Sport Context ───────────────────────────────────────────────────
    st.markdown(
        "AWI and PQI are direct anatomical measures computed from actual joint angles "
        "and skeletal keypoints captured by the TRACAB TF15 system at 50 Hz. "
        "The benchmarked alternatives (NBA Second Spectrum, NFL Next Gen Stats, "
        "Cricket Hawk-Eye, and Industrial Motion Capture) use either proxy-based "
        "approaches (inferring cognitive load from movement patterns) or apply the "
        "same joint-angle signal to different domains (injury risk, ergonomic "
        "assessment). This positions AWI and PQI as methodologically grounded in "
        "established cross-sport and cross-industry practice."
    )
    _benchmark_df = generate_benchmark_summary()
    st.dataframe(_benchmark_df, width="stretch", hide_index=True)

with tab7:
    render_broadcast_demo_tab(fdf)
