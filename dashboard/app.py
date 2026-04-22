"""
Football Body Intelligence -- Streamlit Dashboard
Bundesliga AWI + PQI Analytics Platform
"""
import os
import sys

# Ensure the project root is on sys.path so `src` is importable
# regardless of the working directory Streamlit is launched from.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
st.set_page_config(
    page_title="Football Body Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -- DFL Bundesliga Palette ------------------------------------------------
DFL_RED         = "#D10214"   # Official Bundesliga brand red

C_AWI    = "#38BDF8"   # sky-400 -- pops on dark
C_PQI    = "#FB923C"   # orange-400 -- warm on dark
C_GOLD   = "#FBBF24"   # amber-400
C_GREEN  = "#4ADE80"   # green-400
C_RED    = "#F87171"   # red-400
C_PURPLE = "#A78BFA"   # violet-400

C_BG            = "#0B0F1A"   # deep navy-black
C_SURFACE       = "#111827"   # slate-900
C_SURFACE2      = "#1E293B"   # slate-800 -- elevated cards
C_BORDER        = "#1E293B"   # subtle border
C_BORDER_LT     = "#334155"   # lighter border for emphasis
C_MUTED         = "#94A3B8"   # slate-400
C_TEXT          = "#F1F5F9"   # slate-100
THEME           = "plotly_dark"
LEGEND_BG       = "rgba(17,24,39,0.92)"
MARKER_OUTLINE  = "#334155"

POS_COLORS: dict[str, str] = {
    "GK": "#94A3B8", "CB": "#38BDF8", "FB": "#818CF8",
    "DM": "#FB923C", "CM": "#FBBF24", "WM": "#4ADE80", "FW": "#F472B6",
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
        "How directly the player faces the ball carrier. 100 = square-on, 0 = facing 90 degrees away. Weight: 40% of PQI.",
    ),
    "stance_mean": (
        "Stance Quality",
        "Knee flexion during pressing. Peaks at 130 degrees - the biomechanically optimal pressing position. Weight: 30% of PQI.",
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


# -- CSS -------------------------------------------------------------------
def _inject_css() -> None:
    st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

  /* -- Base -- */
  html, body {{
    font-family: 'Inter', sans-serif !important;
    background-color: {C_BG} !important;
    color: {C_TEXT};
    -webkit-font-smoothing: antialiased;
  }}

  [data-testid="stApp"],
  [data-testid="stAppViewContainer"],
  .appview-container, .stApp {{
    background-color: {C_BG} !important;
  }}

  /* -- Streamlit top bar -- */
  [data-testid="stHeader"] {{
    background-color: {C_BG} !important;
    border-bottom: none !important;
    box-shadow: none !important;
    pointer-events: none;
  }}
  [data-testid="stHeader"] [data-testid="stToolbar"],
  [data-testid="stHeader"] [data-testid="stStatusWidget"],
  [data-testid="stHeader"] button,
  [data-testid="stHeader"] a {{
    pointer-events: auto;
  }}

  /* -- Main content -- */
  [data-testid="stMainBlockContainer"],
  .main .block-container {{
    background-color: {C_BG} !important;
    max-width: 1400px;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 2.5rem !important;
    padding-bottom: 3rem !important;
    margin: 0 auto;
  }}

  /* -- Sidebar -- */
  section[data-testid="stSidebar"] {{
    background-color: {C_SURFACE} !important;
    border-right: 1px solid {C_BORDER_LT};
    width: 280px !important;
  }}
  section[data-testid="stSidebar"] * {{ color: {C_TEXT}; }}

  /* ============================
     TAB NAV - Bundesliga style
     ============================ */
  div[data-testid="stTabs"] {{
    margin-top: 0.25rem;
  }}

  div[data-testid="stTabs"] [role="tablist"] {{
    background: {C_SURFACE};
    border: 1px solid {C_BORDER_LT};
    border-top: 3px solid {DFL_RED};
    border-bottom: none;
    padding: 0 0.75rem;
    gap: 2px;
    border-radius: 0;
  }}

  div[data-testid="stTabs"] [role="tab"] {{
    color: {C_MUTED} !important;
    background: transparent !important;
    padding: 0.9rem 1.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    white-space: nowrap !important;
    border-radius: 0 !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    margin-bottom: 0;
    transition: all 0.2s ease;
  }}

  div[data-testid="stTabs"] [role="tab"] * {{
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    letter-spacing: inherit !important;
    text-transform: inherit !important;
  }}

  div[data-testid="stTabs"] [role="tab"]:hover {{
    color: {C_TEXT} !important;
    background: rgba(209,2,20,0.06) !important;
    border-bottom-color: rgba(209,2,20,0.35) !important;
  }}
  div[data-testid="stTabs"] [role="tab"]:hover * {{
    color: {C_TEXT} !important;
  }}

  div[data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
    color: #FFFFFF !important;
    font-weight: 700 !important;
    border-bottom: 3px solid {DFL_RED} !important;
    background: rgba(209,2,20,0.12) !important;
  }}
  div[data-testid="stTabs"] [role="tab"][aria-selected="true"] * {{
    color: #FFFFFF !important;
    font-weight: 700 !important;
  }}

  div[data-testid="stTabContent"] {{ padding-top: 1.5rem; }}

  /* -- Charts -- */
  div[data-testid="stPlotlyChart"] .modebar {{ display: none !important; }}
  div[data-testid="stPlotlyChart"],
  div[data-testid="stPlotlyChart"] > div {{ background-color: transparent !important; }}

  /* -- KPI cards (stadium scoreboard style) -- */
  .kpi {{
    background: {C_SURFACE2};
    border: 1px solid {C_BORDER};
    border-top: 2px solid {DFL_RED};
    border-radius: 10px;
    padding: 1.1rem 0.9rem 0.9rem;
    text-align: center;
    min-height: 108px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transition: box-shadow 0.25s ease, transform 0.2s ease, border-color 0.2s ease;
    position: relative;
  }}
  .kpi:hover {{
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    transform: translateY(-2px);
    border-color: {C_BORDER_LT};
    border-top-color: {DFL_RED};
  }}
  .kpi-label {{
    font-size: 0.6rem;
    font-weight: 700;
    color: {C_MUTED};
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }}
  .kpi-value {{
    font-size: 1.85rem;
    font-weight: 800;
    line-height: 1.1;
    color: {C_TEXT};
    letter-spacing: -0.02em;
  }}
  .kpi-sub {{
    font-size: 0.62rem;
    color: {C_MUTED};
    margin-top: 3px;
    letter-spacing: 0.01em;
  }}

  /* -- Section headers (red accent bar) -- */
  .sec {{
    font-size: 0.65rem;
    font-weight: 700;
    color: {C_TEXT};
    text-transform: uppercase;
    letter-spacing: 0.16em;
    border-left: 3px solid {DFL_RED};
    padding: 3px 0 3px 10px;
    margin: 2rem 0 1rem;
  }}

  /* -- Player identity card -- */
  .ph-card {{
    background: {C_SURFACE2};
    border: 1px solid {C_BORDER};
    border-left: 5px solid {DFL_RED};
    border-radius: 0 12px 12px 0;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    transition: box-shadow 0.2s ease;
  }}
  .ph-card:hover {{
    box-shadow: 0 6px 24px rgba(0,0,0,0.35);
  }}
  .ph-name {{
    font-size: 1.75rem;
    font-weight: 800;
    color: {C_TEXT};
    letter-spacing: -0.03em;
    line-height: 1.1;
    margin: 0;
  }}
  .ph-meta {{
    font-size: 0.78rem;
    color: {C_MUTED};
    margin-top: 0.3rem;
    font-weight: 400;
    letter-spacing: 0.01em;
  }}

  /* -- Chart caption -- */
  .gcap {{
    text-align: center;
    font-size: 0.72rem;
    color: {C_MUTED};
    margin-top: -0.3rem;
    padding-bottom: 0.5rem;
    letter-spacing: 0.01em;
  }}
  .gcap b {{ color: {C_TEXT}; font-weight: 600; }}

  /* -- KPI tooltip -- */
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
    width: 220px;
    background: {C_SURFACE};
    color: {C_TEXT};
    font-size: 0.7rem;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
    line-height: 1.55;
    text-align: left;
    border-radius: 8px;
    border: 1px solid {C_BORDER_LT};
    padding: 10px 14px;
    position: absolute;
    z-index: 9999;
    bottom: 130%;
    left: 50%;
    transform: translateX(-50%);
    transition: opacity 0.2s ease;
    pointer-events: none;
    white-space: normal;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4);
  }}
  .kpi-info .kpi-tooltip::after {{
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: {C_BORDER_LT};
  }}
  .kpi-info:hover .kpi-tooltip {{ visibility: visible; opacity: 1; }}

  /* -- Pills -- */
  .pill {{
    display: inline-block;
    background: {C_SURFACE2};
    border: 1px solid {C_BORDER};
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 0.67rem;
    font-weight: 500;
    color: {C_MUTED};
    margin: 2px 3px;
  }}
  .pill-hi    {{ background: rgba(56,189,248,0.12);   border-color: rgba(56,189,248,0.3);  color: {C_AWI}; }}
  .pill-pqi   {{ background: rgba(251,146,60,0.12);   border-color: rgba(251,146,60,0.3);  color: {C_PQI}; }}
  .pill-green {{ background: rgba(74,222,128,0.12);   border-color: rgba(74,222,128,0.3);  color: {C_GREEN}; }}
  .pill-red   {{ background: rgba(209,2,20,0.1);      border-color: rgba(209,2,20,0.25);   color: {DFL_RED}; }}

  /* -- Narrative -- */
  .narrative-para {{ font-size: 0.82rem; line-height: 1.75; color: {C_TEXT}; margin-bottom: 0.7rem; letter-spacing: 0.01em; }}
  .narrative-source {{
    font-size: 0.68rem; color: {C_MUTED}; margin-top: 0.5rem;
    border-top: 1px solid {C_BORDER}; padding-top: 0.5rem;
  }}

  /* -- Widget label overrides -- */
  div[data-testid="stSelectbox"] label,
  div[data-testid="stMultiSelect"] label,
  div[data-testid="stSlider"] label {{
    color: {C_MUTED} !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
  }}
  div[data-baseweb="select"] > div,
  div[data-baseweb="input"] > div {{
    background-color: {C_SURFACE2} !important;
    border-color: {C_BORDER} !important;
    color: {C_TEXT} !important;
    border-radius: 8px !important;
  }}
  div[data-testid="stMetric"] {{
    background: {C_SURFACE2};
    border: 1px solid {C_BORDER};
    border-top: 2px solid {DFL_RED};
    border-radius: 10px;
    padding: 0.85rem 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
  }}
  div[data-testid="stExpander"] {{
    background: {C_SURFACE2} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important;
  }}
  div[data-testid="stDataFrame"] {{
    background: {C_SURFACE2} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 10px !important;
  }}
  .stAlert {{
    background: {C_SURFACE2} !important;
    border: 1px solid {C_BORDER} !important;
    color: {C_TEXT} !important;
    border-radius: 10px !important;
  }}

  hr, .stDivider {{ border-color: {C_BORDER} !important; opacity: 1 !important; }}

  ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
  ::-webkit-scrollbar-track {{ background: {C_BG}; }}
  ::-webkit-scrollbar-thumb {{ background: {C_BORDER_LT}; border-radius: 3px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: {C_MUTED}; }}

  /* -- Stat card -- */
  .stat-card {{
    background: {C_SURFACE2};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transition: box-shadow 0.25s ease, transform 0.2s ease, border-color 0.2s ease;
  }}
  .stat-card:hover {{
    box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    transform: translateY(-2px);
    border-color: {C_BORDER_LT};
  }}

  .rank-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 50%;
    font-size: 0.72rem;
    font-weight: 800;
    color: #fff;
    flex-shrink: 0;
  }}

  /* -- Context bar -- */
  .context-bar {{
    background: {C_SURFACE2};
    border: 1px solid {C_BORDER};
    border-radius: 10px;
    padding: 0.6rem 1rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
    font-size: 0.72rem;
    color: {C_MUTED};
    box-shadow: 0 1px 4px rgba(0,0,0,0.15);
  }}
  .context-bar .ctx-label {{
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-size: 0.6rem;
    color: {C_MUTED};
  }}
  .context-bar .ctx-value {{
    font-weight: 600;
    color: {C_TEXT};
  }}
  .context-bar .ctx-sep {{
    color: {C_BORDER_LT};
  }}

  div[data-testid="stToggle"] label span {{
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    color: {C_MUTED} !important;
  }}

  .stCaption, [data-testid="stCaptionContainer"] {{
    color: {C_MUTED} !important;
  }}
</style>
""", unsafe_allow_html=True)


_inject_css()


# -- Helpers ---------------------------------------------------------------
def safe_float(val: object, default: float = 0.0) -> float:
    try:
        return float(val or default)
    except (TypeError, ValueError):
        return default


def kpi(label: str, value: str, sub: str = "", color: str | None = None, help: str | None = None, border: str | None = None) -> None:
    col = color or C_TEXT
    bt  = f"border-top-color:{border};" if border else ""
    info = (f' <span class="kpi-info">\u2139<span class="kpi-tooltip">{help}</span></span>'
            if help else "")
    st.markdown(
        f'<div class="kpi" style="{bt}">'
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
    pos = r.get("position", "-") or "-"
    return (
        f"<b>{r['name']}</b> &nbsp;#{int(r['jersey'])}<br>"
        f"<span style='color:{C_MUTED}'>{pos} / {r['match_id']} / {r['phase_label']}</span><br>"
        f"AWI <b>{r['awi_per_minute']:.2f}</b> scans/min &nbsp;|&nbsp; PQI <b>{pqi}</b>"
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
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Inter", color=C_MUTED))
    return fig


def gauge_cap(value: float, median: float, unit: str) -> None:
    diff = value - median
    sign = "+" if diff >= 0 else ""
    col  = C_GREEN if diff >= 0 else C_RED
    st.markdown(
        f'<div class="gcap"><b>{value:.2f}</b> {unit} &nbsp;|&nbsp; '
        f'median <b>{median:.2f}</b> &nbsp;|&nbsp; '
        f'<span style="color:{col}">{sign}{diff:.2f}</span></div>',
        unsafe_allow_html=True)


def chart_layout(fig: go.Figure, h: int = 360, t: int = 20, b: int = 45, l: int = 45, r: int = 15) -> go.Figure:
    fig.update_layout(
        height=h, margin=dict(t=t, b=b, l=l, r=r),
        template=THEME, paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=C_MUTED, size=11),
        xaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER),
        yaxis=dict(gridcolor=C_BORDER, zerolinecolor=C_BORDER),
    )
    return fig


def context_bar(items: list[tuple[str, str]]) -> None:
    """Render a horizontal context bar showing active filter state."""
    parts = []
    for label, value in items:
        parts.append(
            f'<span class="ctx-label">{label}</span> '
            f'<span class="ctx-value">{value}</span>'
        )
    html = ' <span class="ctx-sep">|</span> '.join(parts)
    st.markdown(f'<div class="context-bar">{html}</div>', unsafe_allow_html=True)


# -- Data ------------------------------------------------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    for p in ("results/awi_full.csv", "results/pqi_full.csv"):
        if not os.path.exists(p):
            st.error(f"`{p}` not found - run the pipeline first.")
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
    # Position-adjusted PQI: z-score within coarse position group (GK/DEF/MID/FWD)
    _POS_ADJ_MAP = {
        "TW": "GK",
        "IVL": "DEF", "IVR": "DEF", "IVZ": "DEF",
        "LA": "DEF", "RA": "DEF", "LV": "DEF", "RV": "DEF",
        "DML": "MID", "DMR": "MID", "DMZ": "MID", "DLM": "MID", "DRM": "MID",
        "ZO": "MID", "RM": "MID",
        "OHL": "MID", "OHR": "MID", "OLM": "MID", "ORM": "MID",
        "HL": "MID", "HR": "MID",
        "STL": "FWD", "STR": "FWD", "STZ": "FWD",
    }
    adj_group = df["position"].map(_POS_ADJ_MAP)
    grp_mean = df["mean_pqi"].groupby(adj_group).transform("mean")
    grp_std = df["mean_pqi"].groupby(adj_group).transform("std")
    df["pqi_position_adjusted"] = (df["mean_pqi"] - grp_mean) / grp_std
    return df


@st.cache_data
def load_narratives() -> pd.DataFrame:
    path = "results/narratives.csv"
    if not os.path.exists(path):
        return pd.DataFrame(columns=["jersey", "team", "match_id", "phase_label", "narrative"])
    return pd.read_csv(path)


df = load_data()
narratives = load_narratives()

# -- Sidebar ---------------------------------------------------------------
with st.sidebar:
    st.markdown(f"""
<div style="border-left:3px solid {DFL_RED};padding:0.5rem 0 0.5rem 0.8rem;margin-bottom:0.5rem">
  <div style="font-size:1.1rem;font-weight:900;color:{C_TEXT};letter-spacing:-0.02em;line-height:1.15">Football Body<br>Intelligence</div>
  <div style="font-size:0.62rem;color:{C_MUTED};margin-top:4px;letter-spacing:0.08em;text-transform:uppercase">Bundesliga 2025/26</div>
</div>
""", unsafe_allow_html=True)

    # Active dataset summary
    n_matches = df["match_id"].nunique()
    n_players = df["name"].nunique()
    st.markdown(
        f'<div style="background:{C_SURFACE2};border:1px solid {C_BORDER_LT};border-radius:8px;'
        f'padding:0.5rem 0.7rem;margin:0.5rem 0 0.75rem;font-size:0.65rem;color:{C_MUTED}">'
        f'<span style="color:{C_TEXT};font-weight:700">{n_matches}</span> matches '
        f'<span style="color:{C_BORDER_LT}">|</span> '
        f'<span style="color:{C_TEXT};font-weight:700">{n_players}</span> players '
        f'<span style="color:{C_BORDER_LT}">|</span> '
        f'<span style="color:{C_TEXT};font-weight:700">400</span> observations'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"<div style='font-size:0.6rem;font-weight:700;color:{C_MUTED};"
                f"text-transform:uppercase;letter-spacing:0.12em;margin-bottom:0.3rem'>"
                f"Filters</div>", unsafe_allow_html=True)

    sel_match = st.selectbox("Match", ["All"] + sorted(df["match_id"].unique()))
    sel_phase = st.selectbox("Phase", ["All", "1st half", "2nd half"])
    all_positions = sorted(df["position"].dropna().unique())
    sel_pos   = st.multiselect("Position", all_positions, placeholder="All positions")
    sel_cov   = st.slider("Min Coverage %", 0, 100, 50,
                          help="Exclude players tracked for less than this % of the phase")

    # Show active filter count
    n_filters = sum([sel_match != "All", sel_phase != "All", len(sel_pos) > 0, sel_cov > 0])
    if n_filters > 0:
        st.markdown(
            f'<div style="font-size:0.62rem;color:{DFL_RED};font-weight:600;margin-top:0.25rem">'
            f'{n_filters} filter{"s" if n_filters > 1 else ""} active</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    with st.expander("Metric Definitions", expanded=False):
        st.markdown(f"""
<div style="font-size:0.72rem;line-height:1.6;color:{C_MUTED}">

<div style="color:{C_AWI};font-weight:700;margin-bottom:2px">AWI - Awareness Index</div>
Discrete head-rotation events per minute. A scan = head turn of 45 degrees or more within 0.5 s,
detected from 3D nose/neck/ear keypoints at 50 fps. Measures how actively a player
checks their surroundings before receiving or releasing the ball.

<div style="color:{C_PQI};font-weight:700;margin-top:10px;margin-bottom:2px">PQI - Press Quality Index (0-100)</div>
Composite score of body mechanics during pressing actions (player within 5 m of
ball carrier for 10 or more consecutive frames). Three sub-scores:

<div style="margin-top:6px;padding-left:8px;border-left:2px solid {C_BORDER}">
<span style="color:{C_TEXT};font-weight:600">Orientation (40%)</span><br>
How directly the player faces the ball carrier.
100 = square-on, 0 = facing 90 degrees away.

<div style="margin-top:6px">
<span style="color:{C_TEXT};font-weight:600">Stance (30%)</span><br>
Knee flexion quality. Peaks at 130 degrees - the biomechanically optimal
pressing position (low centre of gravity, ready to change direction).
</div>

<div style="margin-top:6px">
<span style="color:{C_TEXT};font-weight:600">Proximity (30%)</span><br>
Distance to ball carrier. 100 = 0 m, 0 = 5 m or beyond.
</div>
</div>

<div style="margin-top:8px;color:{C_MUTED};font-size:0.65rem">
PQI = 0.40 x Orientation + 0.30 x Stance + 0.30 x Proximity
</div>
</div>
""", unsafe_allow_html=True)

    with st.expander("Position Code Reference", expanded=False):
        for group, codes in POS_REF.items():
            color = POS_COLORS.get(group, C_MUTED)
            st.markdown(
                f'<div style="margin-bottom:6px">'
                f'<span style="color:{color};font-weight:700;font-size:0.72rem">{group}</span>'
                f'<span style="color:{C_MUTED};font-size:0.65rem"> - '
                + " / ".join(f'<span style="color:{C_TEXT}">{code}</span> {name}' for code, name in codes)
                + '</span></div>',
                unsafe_allow_html=True,
            )

    st.caption("AWS World Sports Innovation Cup 2026")


# -- Filter ----------------------------------------------------------------
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


# -- Tab renderers ---------------------------------------------------------
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

    pos      = p["position"] if pd.notna(p.get("position")) else "-"
    pg       = p.get("pos_group", "-")
    pg_c     = POS_COLORS.get(pg, C_MUTED)
    pos_name = POS_FULL_MAP.get(pos, "")

    # Player identity card
    st.markdown(f"""
<div class="ph-card">
  <div>
    <div class="ph-name">{p['name']}</div>
    <div class="ph-meta">
      #{int(p['jersey'])} &nbsp;&middot;&nbsp; {p['match_id']} &nbsp;&middot;&nbsp; {p['phase_label']}
    </div>
  </div>
  <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px">
    <span style="display:inline-block;background:{pg_c}18;color:{pg_c};border:1px solid {pg_c}40;
      font-size:0.68rem;font-weight:700;padding:4px 14px;border-radius:20px;letter-spacing:0.05em;
      text-transform:uppercase">{pg}</span>
    <span style="font-size:0.72rem;color:{C_MUTED};font-weight:500">{pos_name or pos}</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # -- KPI row -----------------------------------------------------------
    sec("KEY METRICS")
    awi_val = safe_float(p.get("awi_per_minute"))
    pqi_val = safe_float(p.get("mean_pqi"))
    cov_val = safe_float(p.get("coverage_pct"))
    scans   = int(p.get("scan_count", 0) or 0)
    mins    = safe_float(p.get("total_minutes"))

    awi_pct = pct_rank(fdf["awi_per_minute"], awi_val)
    pqi_pct = pct_rank(fdf["mean_pqi"], pqi_val) if pqi_val > 0 else 0

    c1, c2, c3, c4, c5 = st.columns(5, gap="small")
    with c1: kpi("AWI",         f"{awi_val:.2f}", f"Top {100-awi_pct}% | scans/min", C_AWI,
                 border=C_AWI,
                 help="Awareness Index: head-rotation events per minute. Higher = more active scanning.")
    with c2: kpi("Mean PQI",    f"{pqi_val:.1f}", f"Top {100-pqi_pct}% | 0-100",    C_PQI,
                 border=C_PQI,
                 help="Press Quality Index: composite of Orientation (40%), Stance (30%), Proximity (30%) during press frames.")
    with c3: kpi("Total Scans", f"{scans:,}",     f"{mins:.0f} min on pitch",
                 help="Total discrete head-rotation events detected in this phase.")
    with c4: kpi("Coverage",    f"{cov_val*100:.0f}%", "time on pitch",
                 help="Fraction of the phase where skeleton tracking data was available for this player.")
    with c5: kpi("Press Mins",
                 f"{safe_float(p.get('press_minutes')):.0f}",
                 "min in press frames",
                 help="Minutes spent within 5 m of the ball carrier for 10+ consecutive frames.")
    sp(0.75)

    # -- Gauges ------------------------------------------------------------
    sec("PERFORMANCE GAUGES")
    awi_max = max(fdf["awi_per_minute"].max() * 1.15, 1)
    awi_med = fdf["awi_per_minute"].median()
    pqi_med = fdf["mean_pqi"].median()

    g1, g2 = st.columns(2, gap="medium")
    with g1:
        st.plotly_chart(gauge_fig(round(awi_val, 2),
                                  "AWI - Awareness Index (scans/min)",
                                  C_AWI, awi_max, awi_med), width="stretch")
        gauge_cap(awi_val, awi_med, "scans/min")
    with g2:
        st.plotly_chart(gauge_fig(round(pqi_val, 1),
                                  "PQI - Press Quality Index (0-100)",
                                  C_PQI, 100, pqi_med), width="stretch")
        gauge_cap(pqi_val, pqi_med, "PQI")
    sp(0.5)

    # -- PQI radar + scatter -----------------------------------------------
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
        role_df     = fdf[fdf["pos_group"] == pg] if pg != "-" else fdf
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
                line_color=C_PQI, line_dash="dot",
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

    # -- PQI component breakdown -------------------------------------------
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
                kpi(label, f"{val:.1f}", f"Median {med:.1f} | Top {100-pct}%",
                    help=tooltip)

    # -- Half-time trend ---------------------------------------------------
    all_phases = fdf[fdf["name"] == sel_name].copy()
    if len(all_phases) >= 2:
        sp(0.25)
        sec("HALF-TIME PERFORMANCE TREND")
        trend_cols = st.columns(len(all_phases), gap="small")
        for i, (_, row) in enumerate(all_phases.iterrows()):
            with trend_cols[i]:
                awi_d = row["awi_per_minute"] - all_phases["awi_per_minute"].mean()
                col   = C_GREEN if awi_d >= 0 else C_RED
                kpi(f"{row['match_id']} | {row['phase_label']}",
                    f"{row['awi_per_minute']:.2f}",
                    f"PQI {row['mean_pqi']:.1f}" if pd.notna(row.get('mean_pqi')) else "PQI -",
                    col)

    # -- AI Scouting Narrative ---------------------------------------------
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
                f'Generated by Amazon Bedrock &nbsp;|&nbsp; {p["match_id"]} &nbsp;|&nbsp; {p["phase_label"]}'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_match_overview(fdf: pd.DataFrame) -> None:
    if guard(fdf):
        return

    # -- Context bar showing active filters --------------------------------
    ctx_items = []
    n_matches = fdf["match_id"].nunique()
    n_players = fdf["name"].nunique()
    ctx_items.append(("Matches", str(n_matches)))
    ctx_items.append(("Players", str(n_players)))
    ctx_items.append(("Observations", str(len(fdf))))
    context_bar(ctx_items)

    # -- Summary KPIs ------------------------------------------------------
    sec("MATCH SUMMARY")
    top_row = fdf.loc[fdf["awi_per_minute"].idxmax()]
    s1, s2, s3, s4, s5 = st.columns(5, gap="small")
    with s1: kpi("Players", str(fdf["name"].nunique()),
                 f"{fdf['match_id'].nunique()} match(es)",
                 help="Number of unique players included after applying the current filters.")
    with s2: kpi("Avg AWI",  f"{fdf['awi_per_minute'].mean():.2f}", "scans/min", C_AWI,
                 border=C_AWI,
                 help="Average Awareness Index across all filtered players.")
    with s3: kpi("Avg PQI",  f"{fdf['mean_pqi'].mean():.1f}",       "0-100",     C_PQI,
                 border=C_PQI,
                 help="Average Press Quality Index across all filtered players.")
    with s4: kpi("Top AWI",  top_row["name"].split()[-1],
                 f"{top_row['awi_per_minute']:.2f} scans/min", C_GOLD,
                 border=C_GOLD,
                 help="Player with the highest AWI in the current filtered set.")
    with s5:
        n_elite = len(fdf[(fdf["awi_per_minute"] >= fdf["awi_per_minute"].quantile(0.75)) &
                          (fdf["mean_pqi"] >= fdf["mean_pqi"].quantile(0.75))]["name"].unique())
        kpi("Elite Players", str(n_elite), "top 25% AWI & PQI", C_GREEN,
            border=C_GREEN,
            help="Players ranking in the top 25th percentile on both AWI and PQI simultaneously.")
    sp(0.75)

    # -- Quadrant scatter --------------------------------------------------
    sec("AWI vs PQI - PLAYER QUADRANT ANALYSIS")

    # Quadrant legend as styled cards
    i1, i2, i3, i4 = st.columns(4, gap="small")
    with i1:
        st.markdown(f'<div style="font-size:0.72rem;color:{C_MUTED};padding:0.4rem 0">'
                    f'<span style="color:{C_AWI};font-weight:700">X-axis</span> '
                    f'AWI scanning rate</div>', unsafe_allow_html=True)
    with i2:
        st.markdown(f'<div style="font-size:0.72rem;color:{C_MUTED};padding:0.4rem 0">'
                    f'<span style="color:{C_PQI};font-weight:700">Y-axis</span> '
                    f'PQI pressing quality</div>', unsafe_allow_html=True)
    with i3:
        st.markdown(f'<div style="font-size:0.72rem;color:{C_MUTED};padding:0.4rem 0">'
                    f'<span style="font-weight:700">Dashed lines</span> '
                    f'75th percentile</div>', unsafe_allow_html=True)
    with i4:
        st.markdown(f'<div style="font-size:0.72rem;color:{C_MUTED};padding:0.4rem 0">'
                    f'<span style="color:{C_GREEN};font-weight:700">Green</span> '
                    f'Top performers</div>', unsafe_allow_html=True)
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
    chart_layout(fig_q, h=440, t=15, b=55, l=55, r=15)
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

    # -- Position group analysis -------------------------------------------
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

    # -- Half-time delta ---------------------------------------------------
    sec("HALF-TIME COGNITIVE FATIGUE - AWI DELTA (2nd - 1st Half)")
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
                hovertemplate="<b>%{x}</b><br>AWI change: %{y:.2f} scans/min<extra></extra>",
                name="AWI change",
            ))
            fig_d.add_hline(y=0, line_color=C_BORDER, line_width=1.5)
            chart_layout(fig_d, h=320, t=15, b=70, l=50, r=15)
            fig_d.update_layout(
                title=dict(text="AWI Change 2nd vs 1st Half (top players by 1st-half AWI)",
                           font_size=11, x=0, font_color=C_MUTED),
                xaxis_tickangle=-35, yaxis_title="AWI change (scans/min)",
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

    # -- Team comparison ---------------------------------------------------
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

    # -- Summary KPIs ------------------------------------------------------
    sec("OVERVIEW")
    o1, o2, o3, o4 = st.columns(4, gap="small")
    elite = fdf[(fdf["awi_per_minute"] >= fdf["awi_per_minute"].quantile(0.75)) &
                (fdf["mean_pqi"] >= fdf["mean_pqi"].quantile(0.75))]
    with o1: kpi("Total Players", str(fdf["name"].nunique()), "active player-phases",
                 help="Unique players with at least one tracked phase in the current filtered dataset.")
    with o2: kpi("Elite Quadrant", str(elite["name"].nunique()),
                 "top 25% AWI & PQI", C_GREEN,
                 border=C_GREEN,
                 help="Players ranking in the top 25th percentile on both AWI and PQI simultaneously.")
    with o3: kpi("Avg AWI",  f"{fdf['awi_per_minute'].mean():.2f}", "scans/min", C_AWI,
                 border=C_AWI,
                 help="Average Awareness Index across all filtered players.")
    with o4: kpi("Avg PQI",  f"{fdf['mean_pqi'].mean():.1f}",       "0-100",     C_PQI,
                 border=C_PQI,
                 help="Average Press Quality Index across all filtered players.")
    sp(0.75)

    # -- Top 3 podium ------------------------------------------------------
    sec("TOP PERFORMERS")
    top3 = (fdf.sort_values("awi_per_minute", ascending=False)
            .drop_duplicates(subset="name")
            .head(3)
            .reset_index(drop=True))

    if len(top3) >= 3:
        podium_colors = [C_GOLD, "#94A3B8", "#CD7F32"]  # gold, silver, bronze
        podium_labels = ["\U0001F947", "\U0001F948", "\U0001F949"]
        p1, p2, p3 = st.columns(3, gap="small")
        for i, (col, (_, row)) in enumerate(zip([p1, p2, p3], top3.iterrows())):
            with col:
                pg = row.get("pos_group", "-")
                pg_c = POS_COLORS.get(pg, C_MUTED)
                pqi_str = f"{row['mean_pqi']:.1f}" if pd.notna(row.get("mean_pqi")) else "n/a"
                st.markdown(f"""
<div class="stat-card" style="text-align:center;border-top:3px solid {podium_colors[i]}">
  <div style="font-size:1.5rem;margin-bottom:0.3rem">{podium_labels[i]}</div>
  <div style="font-size:1rem;font-weight:800;color:{C_TEXT};line-height:1.2">{row['name']}</div>
  <div style="margin:0.4rem 0;display:flex;justify-content:center;gap:6px">
    <span style="display:inline-block;background:{pg_c}18;color:{pg_c};font-size:0.6rem;font-weight:600;padding:2px 8px;border-radius:10px">{pg}</span>
    <span style="display:inline-block;background:{C_BG};color:{C_MUTED};font-size:0.6rem;padding:2px 8px;border-radius:10px">{row['match_id']}</span>
  </div>
  <div style="display:flex;justify-content:center;gap:1.2rem;margin-top:0.5rem">
    <div>
      <div style="font-size:1.4rem;font-weight:900;color:{C_AWI}">{row['awi_per_minute']:.2f}</div>
      <div style="font-size:0.55rem;color:{C_MUTED};text-transform:uppercase;letter-spacing:0.1em">AWI</div>
    </div>
    <div style="width:1px;background:{C_BORDER}"></div>
    <div>
      <div style="font-size:1.4rem;font-weight:900;color:{C_PQI}">{pqi_str}</div>
      <div style="font-size:0.55rem;color:{C_MUTED};text-transform:uppercase;letter-spacing:0.1em">PQI</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
    sp(0.75)

    # -- Leaderboard table -------------------------------------------------
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
        height="auto",
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

    # -- Role analytics ----------------------------------------------------
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
                hovertemplate="<b>%{y}</b> | %{x}<br>%{z:.1f}<extra></extra>")
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
                    f"Std: +/-{row['std']:.2f}<br>"
                    f"n={int(row['count'])}<extra></extra>"
                ),
                showlegend=False,
            ))
        chart_layout(fig_bar, h=max(260, len(role_awi) * 50 + 70),
                     t=15, b=50, l=10, r=30)
        fig_bar.update_layout(
            title=dict(text="Avg AWI by Role (mean +/- std)",
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
    """Fan Engagement View - Bundesliga broadcast-ready Body Intelligence display."""
    _DFL_RED = "#D10214"

    if guard(fdf):
        return

    # -- Page banner -------------------------------------------------------
    st.markdown(f"""
<div style="
    border-left:4px solid {_DFL_RED};
    padding:0.75rem 1.2rem;
    margin-bottom:1.5rem;
    background:{C_SURFACE};
    border-radius:0 10px 10px 0;
    display:flex;
    align-items:center;
    gap:1.2rem;
    box-shadow:0 2px 8px rgba(0,0,0,0.04);
">
  <div style="flex:1">
    <div style="font-size:1.5rem;font-weight:900;color:{C_TEXT};letter-spacing:-0.02em;line-height:1.1">BODY INTELLIGENCE</div>
    <div style="font-size:0.75rem;color:{C_MUTED};margin-top:4px;letter-spacing:0.04em">Bundesliga &nbsp;|&nbsp; Awareness &amp; Pressing Analytics &nbsp;|&nbsp; 5 Matches &nbsp;|&nbsp; 2025/26</div>
  </div>
  <div style="background:{_DFL_RED};color:#fff;font-size:0.58rem;font-weight:700;letter-spacing:0.15em;padding:5px 12px;border-radius:4px;text-transform:uppercase;flex-shrink:0">LIVE VIEW</div>
</div>
""", unsafe_allow_html=True)

    # -- League-wide KPI row -----------------------------------------------
    n_players   = int(fdf["name"].nunique())
    league_awi  = float(fdf[fdf["awi_per_minute"] > 0]["awi_per_minute"].mean())
    awi_q75     = fdf["awi_per_minute"].quantile(0.75)
    pqi_q75     = fdf["mean_pqi"].quantile(0.75)
    n_elite     = int(fdf[(fdf["awi_per_minute"] >= awi_q75) & (fdf["mean_pqi"] >= pqi_q75)].shape[0])
    top_pqi_val = float(fdf["mean_pqi"].max())

    cols_kpi = st.columns(4, gap="small")
    with cols_kpi[0]:
        kpi("Players Tracked", str(n_players), "across 5 matches", C_TEXT)
    with cols_kpi[1]:
        kpi("League Avg AWI", f"{league_awi:.1f}", "scans / min", C_AWI, border=C_AWI)
    with cols_kpi[2]:
        kpi("Elite Players", str(n_elite), "top 25% AWI + PQI", _DFL_RED, border=_DFL_RED)
    with cols_kpi[3]:
        kpi("Peak PQI", f"{top_pqi_val:.1f}", "max pressing quality", C_PQI, border=C_PQI)
    sp(0.75)

    # -- Top-3 awareness counter -------------------------------------------
    sec("TOP SCANNERS - AWARENESS COUNTER")
    st.markdown(
        f'<div style="font-size:0.72rem;color:{C_MUTED};margin-bottom:1rem">'
        f'Head-rotation events per minute. Higher = more active pre-decision scanning.</div>',
        unsafe_allow_html=True,
    )

    top_awi = (fdf.sort_values("awi_per_minute", ascending=False)
               .drop_duplicates(subset="name")
               .head(3)
               .reset_index(drop=True))

    rank_colors = [_DFL_RED, "#94A3B8", "#CD7F32"]
    rank_labels = ["#1", "#2", "#3"]
    awi_max = float(fdf["awi_per_minute"].max())

    cols_awi = st.columns(3, gap="medium")
    for i, (_, row) in enumerate(top_awi.iterrows()):
        with cols_awi[i]:
            pos_group  = str(row.get("pos_group", "")) or "-"
            pos_color  = POS_COLORS.get(pos_group, C_MUTED)
            awi_val    = float(row["awi_per_minute"])
            diff       = awi_val - league_awi
            sign       = "+" if diff >= 0 else ""
            bar_pct    = int((awi_val / awi_max) * 100)
            lg_bar_pct = int((league_awi / awi_max) * 100)
            diff_color = C_GREEN if diff >= 0 else C_MUTED
            st.markdown(f"""
<div class="stat-card" style="text-align:center;border-top:3px solid {rank_colors[i]}">
  <div style="
    display:inline-block;
    background:{rank_colors[i]};
    color:#fff;
    font-size:0.6rem;
    font-weight:700;
    letter-spacing:0.12em;
    padding:3px 10px;
    border-radius:20px;
    margin-bottom:0.7rem;
  ">{rank_labels[i]}</div>
  <div style="font-size:1.05rem;font-weight:700;color:{C_TEXT};line-height:1.2;margin-bottom:6px">{row['name']}</div>
  <div style="display:flex;justify-content:center;gap:6px;margin-bottom:0.9rem;flex-wrap:wrap">
    <span style="display:inline-block;background:{pos_color}22;color:{pos_color};font-size:0.62rem;font-weight:600;padding:2px 9px;border-radius:12px">{pos_group}</span>
    <span style="display:inline-block;background:{C_BORDER};color:{C_MUTED};font-size:0.62rem;padding:2px 9px;border-radius:12px">{row['match_id']}</span>
  </div>
  <div style="font-size:3rem;font-weight:800;color:{rank_colors[i]};line-height:1">{awi_val:.1f}</div>
  <div style="font-size:0.65rem;color:{C_MUTED};margin:4px 0 1rem;text-transform:uppercase;letter-spacing:0.1em">scans / min</div>
  <div style="background:{C_BORDER};border-radius:4px;height:4px;margin:0 0 0.45rem;position:relative;overflow:visible">
    <div style="background:{rank_colors[i]};border-radius:4px;height:100%;width:{bar_pct}%"></div>
    <div style="position:absolute;top:-3px;left:{lg_bar_pct}%;width:2px;height:10px;background:{C_MUTED};border-radius:1px"></div>
  </div>
  <div style="font-size:0.65rem;color:{diff_color}">{sign}{diff:.1f} vs league avg</div>
</div>
""", unsafe_allow_html=True)
    sp(1.2)

    # -- 4-quadrant fan comparison -----------------------------------------
    sec("PLAYER COMPARISON - BODY INTELLIGENCE QUADRANT")
    st.markdown(
        f'<div style="font-size:0.72rem;color:{C_MUTED};margin-bottom:0.75rem">'
        f'Every player ranked by scanning awareness (AWI) vs pressing quality (PQI). '
        f'Elite players are in the top-right.</div>',
        unsafe_allow_html=True,
    )

    fdf_fan = fdf.copy()
    fdf_fan["quadrant_label"] = fdf_fan.apply(lambda r: (
        "ELITE"      if r["awi_per_minute"] >= awi_q75 and r["mean_pqi"] >= pqi_q75 else
        "SMART"      if r["awi_per_minute"] >= awi_q75 else
        "PHYSICAL"   if r["mean_pqi"] >= pqi_q75 else
        "DEVELOPING"
    ), axis=1)

    fan_colors = {
        "ELITE":      _DFL_RED,
        "SMART":      C_AWI,
        "PHYSICAL":   C_PQI,
        "DEVELOPING": C_MUTED,
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
        opacity=0.85,
    )
    fig_fan.update_traces(marker_size=10, marker_line_width=1, marker_line_color=MARKER_OUTLINE)

    elite_fan = (fdf_fan[fdf_fan["quadrant_label"] == "ELITE"]
                 .drop_duplicates(subset="name")
                 .sort_values("awi_per_minute", ascending=False)
                 .head(8))
    fig_fan.add_trace(go.Scatter(
        x=elite_fan["awi_per_minute"],
        y=elite_fan["mean_pqi"] + 0.8,
        mode="text",
        text=elite_fan["name"].apply(lambda n: n.split()[-1]),
        textfont=dict(size=9, color=_DFL_RED),
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

    x_max = fdf_fan["awi_per_minute"].max()
    y_max = fdf_fan["mean_pqi"].max()
    x_min = fdf_fan["awi_per_minute"].min()
    y_min = fdf_fan["mean_pqi"].min()

    for txt, x, y, col in [
        ("ELITE",    x_max * 0.97, y_max * 0.99,                    _DFL_RED),
        ("SMART",    x_max * 0.97, y_min + (pqi_q75 - y_min) * 0.15, C_AWI),
        ("PHYSICAL", x_min + (awi_q75 - x_min) * 0.15, y_max * 0.99, C_PQI),
    ]:
        fig_fan.add_annotation(
            x=x, y=y, text=txt, showarrow=False,
            font=dict(size=9, color=col), opacity=0.6,
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

    # -- Body Intelligence leaderboard -------------------------------------
    sec("BODY INTELLIGENCE LEADERBOARD")
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
    lb_display.columns = ["Player", "Pos", "Match", "AWI", "PQI", "Body Intelligence"]
    lb_display["AWI"] = lb_display["AWI"].round(2)
    lb_display["PQI"] = lb_display["PQI"].round(1)
    lb_display["Body Intelligence"] = lb_display["Body Intelligence"].round(1)

    st.dataframe(
        lb_display, width="stretch",
        height="auto",
        column_config={
            "Player":           st.column_config.TextColumn("Player", width="medium"),
            "Pos":              st.column_config.TextColumn("Pos", width="small"),
            "Match":            st.column_config.TextColumn("Match", width="small"),
            "AWI":              st.column_config.NumberColumn("AWI (scans/min)", format="%.2f",
                                    help="Scanning awareness: head-rotation events per minute"),
            "PQI":              st.column_config.NumberColumn("PQI", format="%.1f",
                                    help="Pressing quality: body mechanics score 0-100"),
            "Body Intelligence": st.column_config.ProgressColumn(
                                    "Body Intelligence",
                                    format="%.1f",
                                    min_value=0, max_value=100,
                                    help="Combined percentile rank on AWI and PQI"),
        },
    )
    sp(0.75)

    # -- DFL Insight ticker ------------------------------------------------
    if not fdf.empty:
        top_player = fdf.loc[fdf["awi_per_minute"].idxmax()]
        cadence = 60 / top_player["awi_per_minute"]
        st.markdown(f"""
<div style="
    background:{C_SURFACE2};
    border:1px solid {C_BORDER_LT};
    border-left:3px solid {_DFL_RED};
    color:{C_TEXT};
    border-radius:8px;
    padding:0.85rem 1.3rem;
    display:flex;
    align-items:baseline;
    gap:0.75rem;
    flex-wrap:wrap;
">
  <span style="font-size:0.6rem;font-weight:700;letter-spacing:0.14em;text-transform:uppercase;color:{_DFL_RED};flex-shrink:0">DFL INSIGHT</span>
  <span style="font-size:0.8rem;font-weight:400;line-height:1.7;color:{C_MUTED}">
    {top_player['name'].split()[-1]} scanned <strong style="color:{C_TEXT}">{top_player['awi_per_minute']:.0f} times per minute</strong>
    in {top_player['match_id']} ({top_player['phase_label']}), once every <strong style="color:{C_TEXT}">{cadence:.1f} seconds</strong>.
    AWI spikes <strong style="color:{C_AWI}">+57%</strong> in the 5 seconds before a pass.
  </span>
</div>
""", unsafe_allow_html=True)


def render_benchmark_tab(fdf: pd.DataFrame) -> None:
    """Render the cross-domain benchmark comparison tab."""
    try:
        from src.benchmark_reference import (
            get_all_references,
            get_references_for_metric,
            percentile_in_reference,
            sample_reference_distribution,
        )
    except ImportError:
        st.error("benchmark_reference module not found.")
        return

    if guard(fdf):
        return

    st.markdown("## Cross-Domain Benchmark")
    st.markdown(
        "AWI and PQI are not isolated inventions. Each metric maps directly to a concept "
        "validated at scale in another sport or domain. The Bundesliga data sits within the "
        "expected ranges of all six reference systems."
    )

    # ── Block 1: Reference Catalogue ─────────────────────────────────────────
    st.markdown("### Reference Systems")
    refs = get_all_references()
    cat_rows = [
        {
            "Sport":            r.sport,
            "System":           r.system,
            "Maps to":          r.metric_type.replace("_", " "),
            "Pop. mean ± std":  f"{r.mean:.0f} ± {r.std:.0f}",
            "Elite mean":       f"{r.elite_mean:.0f}",
            "Unit":             r.unit,
        }
        for r in refs
    ]
    st.dataframe(pd.DataFrame(cat_rows), hide_index=True, use_container_width=True)

    # ── Block 2: AWI vs Aviation ──────────────────────────────────────────────
    st.markdown("### AWI vs Aviation Cognitive Load")
    awi_ref = get_references_for_metric("AWI")[0]
    awi_samples = sample_reference_distribution(awi_ref, n=1000)
    bundesliga_awi = fdf["awi_per_minute"].dropna()
    bundesliga_awi = bundesliga_awi[bundesliga_awi > 0]

    bl_mean = float(bundesliga_awi.mean()) if len(bundesliga_awi) > 0 else awi_ref.mean
    bl_pct = percentile_in_reference(bl_mean, awi_ref)

    fig_awi = go.Figure()
    fig_awi.add_trace(go.Histogram(
        x=awi_samples, nbinsx=40,
        name=f"Aviation reference (μ={awi_ref.mean:.0f} scans/min)",
        marker_color=C_MUTED, opacity=0.5,
    ))
    fig_awi.add_trace(go.Histogram(
        x=bundesliga_awi, nbinsx=30,
        name=f"Bundesliga AWI (μ={bl_mean:.1f} scans/min)",
        marker_color=C_AWI, opacity=0.75,
    ))
    fig_awi.add_vline(
        x=awi_ref.elite_mean, line_dash="dash", line_color=C_GOLD,
        annotation_text=f"Aviation elite: {awi_ref.elite_mean:.0f}",
        annotation_font_color=C_GOLD,
    )
    fig_awi.add_vline(
        x=bl_mean, line_dash="solid", line_color=C_AWI,
        annotation_text=f"Bundesliga mean: {bl_mean:.1f}",
        annotation_font_color=C_AWI,
    )
    fig_awi.update_layout(
        template=THEME, barmode="overlay",
        paper_bgcolor=C_BG, plot_bgcolor=C_SURFACE,
        title=f"Bundesliga mean AWI: {bl_pct:.0f}th percentile of aviation reference",
        xaxis_title="Scans per minute", yaxis_title="Count",
        legend=dict(bgcolor=LEGEND_BG),
        height=350,
    )
    st.plotly_chart(fig_awi, use_container_width=True)

    # ── Block 3: PQI Sub-Scores vs References ─────────────────────────────────
    st.markdown("### PQI Sub-Scores vs Cross-Domain References")
    sub_configs = [
        ("orientation_mean", "PQI_orientation", "Orientation", C_PURPLE, "vs NBA Second Spectrum"),
        ("stance_mean",      "PQI_stance",      "Stance",      C_GREEN,  "vs Tennis Hawk-Eye"),
        ("proximity_mean",   "PQI_proximity",   "Proximity",   C_GOLD,   "vs NFL Next Gen Stats"),
    ]
    cols3 = st.columns(3)
    for col, (data_col, metric_type, label, color, subtitle) in zip(cols3, sub_configs):
        with col:
            if data_col not in fdf.columns or fdf[data_col].dropna().empty:
                st.info(f"Sub-score `{data_col}` not available")
                continue
            ref = get_references_for_metric(metric_type)[0]
            ref_samples = sample_reference_distribution(ref, n=800)
            bl_vals = fdf[data_col].dropna()
            fig_sub = go.Figure()
            fig_sub.add_trace(go.Histogram(
                x=ref_samples, nbinsx=30,
                name=ref.system, marker_color=C_MUTED, opacity=0.5,
            ))
            fig_sub.add_trace(go.Histogram(
                x=bl_vals, nbinsx=25,
                name=f"Bundesliga {label}", marker_color=color, opacity=0.75,
            ))
            fig_sub.add_vline(
                x=ref.elite_mean, line_dash="dash", line_color=C_GOLD,
                annotation_text=f"Elite: {ref.elite_mean:.0f}",
                annotation_font_color=C_GOLD,
            )
            fig_sub.update_layout(
                template=THEME, barmode="overlay",
                paper_bgcolor=C_BG, plot_bgcolor=C_SURFACE,
                title=f"PQI {label} {subtitle}",
                xaxis_title="Score (0-100)", yaxis_title="Count",
                showlegend=False, height=300,
            )
            st.plotly_chart(fig_sub, use_container_width=True)

    # ── Block 4: Pre-Decision Scan Burst ──────────────────────────────────────
    st.markdown("### Pre-Decision Scan Burst")
    st.markdown(
        "The +57% AWI spike in the 5 seconds before a pass mirrors the pre-decision scan "
        "burst documented in fighter-pilot studies (aviation: +40-65%). "
        "The Bundesliga finding sits at the midpoint of this aviation range."
    )
    bl_pre_pass = bl_mean * 1.57
    categories = [
        "Aviation<br>(low workload)",
        "Aviation<br>(high workload)",
        "Aviation<br>(pre-decision)",
        "Football<br>(baseline AWI)",
        "Football<br>(pre-pass AWI)",
    ]
    values = [14.0, 24.0, 34.0, bl_mean, bl_pre_pass]
    bar_colors = [C_MUTED, C_MUTED, C_GOLD, C_AWI, C_GREEN]

    fig_burst = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=bar_colors, opacity=0.85,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(color=C_TEXT),
    ))
    fig_burst.add_annotation(
        x="Football<br>(pre-pass AWI)", y=bl_pre_pass,
        ax="Football<br>(baseline AWI)", ay=bl_mean,
        xref="x", yref="y", axref="x", ayref="y",
        text="+57%", showarrow=True, arrowhead=2,
        arrowcolor=C_GREEN, font=dict(color=C_GREEN, size=13),
    )
    fig_burst.add_annotation(
        x="Aviation<br>(pre-decision)", y=34.0,
        ax="Aviation<br>(high workload)", ay=24.0,
        xref="x", yref="y", axref="x", ayref="y",
        text="+42%", showarrow=True, arrowhead=2,
        arrowcolor=C_GOLD, font=dict(color=C_GOLD, size=13),
    )
    fig_burst.update_layout(
        template=THEME, paper_bgcolor=C_BG, plot_bgcolor=C_SURFACE,
        title="Pre-Decision Scan Burst: Aviation vs Football",
        yaxis_title="Scans per minute",
        yaxis=dict(range=[0, max(values) * 1.25]),
        height=420,
    )
    st.plotly_chart(fig_burst, use_container_width=True)


def render_broadcast_demo_tab(fdf: pd.DataFrame) -> None:
    """Broadcast Demo tab - renders broadcast overlay inline within the main dashboard.

    Delegates to broadcast_demo.render_broadcast_overlay with standalone=False
    so the ticker does not block the Streamlit thread.
    """
    from dashboard.broadcast_demo import load_broadcast_data, render_broadcast_overlay

    st.markdown(
        f'<div style="font-size:0.75rem;color:{C_MUTED};margin-bottom:1rem">'
        f'Simulated broadcast lower-third overlay. AWI and PQI are real-time '
        f'capable with under 15 seconds latency at 50 fps.</div>',
        unsafe_allow_html=True,
    )

    broadcast_df = load_broadcast_data()
    if broadcast_df.empty:
        st.warning("No broadcast data available.")
        return

    render_broadcast_overlay(broadcast_df, standalone=False)


# -- Tabs ------------------------------------------------------------------
tab_profile, tab_match, tab_board, tab_fan, tab_broadcast, tab_benchmark = st.tabs([
    "Player Profile",
    "Match Overview",
    "Leaderboard",
    "Fan View",
    "Broadcast Demo",
    "Benchmark",
])

with tab_profile:
    render_player_profile(fdf)

with tab_match:
    render_match_overview(fdf)

with tab_board:
    render_leaderboard(fdf)

with tab_fan:
    render_fan_view(fdf)

with tab_broadcast:
    render_broadcast_demo_tab(fdf)

with tab_benchmark:
    render_benchmark_tab(fdf)
