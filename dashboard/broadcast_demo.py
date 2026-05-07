"""
Broadcast Demo - Fan Broadcast Overlay for Football Body Intelligence Platform

Standalone broadcast-style visualization mockup showing AWI and PQI as live
match statistics in a Bundesliga broadcast lower-third format.

Usage:
    streamlit run dashboard/broadcast_demo.py
"""
import math
import os
import sys
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Ensure src/ is importable when running standalone
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.s3_data_loader import load_csv  # noqa: E402

# ── DFL Color System ──────────────────────────────────────────────────────────
DFL_RED   = "#D10214"
DFL_BLACK = "#000000"
DFL_WHITE = "#FFFFFF"
DFL_GREY  = "#8A8A8A"

# Light-theme palette (mirrors app.py) used when embedded in the main dashboard
LIGHT_BG      = "#0B0F1A"
LIGHT_SURFACE = "#111827"
LIGHT_BORDER  = "#1E293B"
LIGHT_MUTED   = "#94A3B8"
LIGHT_TEXT    = "#F1F5F9"

# ── Ticker Content ────────────────────────────────────────────────────────────
TICKER_MESSAGES: list[str] = [
    "+59% pre-pass scan spike",
    "r=-0.11 AWI/PQI independence",
    "r=0.660 cross-half stability",
]

TICKER_INTERVAL_SECONDS: int = 3

# ── Position Group Mapping (mirrors app.py, no import to avoid circular deps) ─
POS_MAP: dict[str, str] = {
    "TW": "GK",
    "IVL": "CB", "IVR": "CB", "IVZ": "CB",
    "LA": "FB", "RA": "FB", "LV": "FB", "RV": "FB",
    "DML": "DM", "DMR": "DM", "DMZ": "DM", "DRM": "DM", "DLM": "DM",
    "ZO": "CM", "RM": "CM",
    "OHL": "WM", "OHR": "WM", "OLM": "WM", "ORM": "WM", "HL": "WM", "HR": "WM",
    "STL": "FW", "STR": "FW", "STZ": "FW",
}


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_broadcast_data() -> pd.DataFrame:
    """Load the merged AWI + PQI dataset for broadcast overlay.

    Uses the S3 data loader (tries S3 first, falls back to local files).
    Merges ``awi_full.csv`` and ``pqi_full.csv`` on the fly so the broadcast
    demo always reflects the latest pipeline output.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added ``pos_group`` column.
    """
    awi = load_csv("awi_full.csv", "results/awi_full.csv")
    pqi = load_csv("pqi_full.csv", "results/pqi_full.csv")

    if awi is None or pqi is None:
        st.error(
            "Result CSVs not found. Configure S3 credentials (via .env or "
            "Streamlit secrets) or run the pipeline locally first."
        )
        st.stop()

    pqi_cols = ["jersey", "team", "match_id", "phase_label",
                "mean_pqi", "median_pqi", "std_pqi",
                "n_press_frames", "press_minutes",
                "orientation_mean", "stance_mean", "proximity_mean"]
    df = awi.merge(pqi[[c for c in pqi_cols if c in pqi.columns]],
                   on=["jersey", "team", "match_id", "phase_label"],
                   how="left")

    df["pos_group"] = df["position"].map(POS_MAP)
    return df


# ── Pure Computation Functions ────────────────────────────────────────────────

def compute_league_mean_awi(df: pd.DataFrame) -> float:
    """Return mean AWI across all players with positive awi_per_minute.

    Parameters
    ----------
    df:
        DataFrame containing an awi_per_minute column.

    Returns
    -------
    float
        Mean of awi_per_minute for rows where awi_per_minute > 0.
    """
    positive = df[df["awi_per_minute"] > 0]["awi_per_minute"]
    return float(positive.mean())


def compute_role_mean_pqi(df: pd.DataFrame) -> dict[str, float]:
    """Return mapping of pos_group to mean mean_pqi.

    Parameters
    ----------
    df:
        DataFrame containing pos_group and mean_pqi columns.

    Returns
    -------
    dict[str, float]
        Mapping from position group label to mean PQI value (NaN rows skipped).
    """
    result: dict[str, float] = {}
    for group, group_df in df.groupby("pos_group"):
        result[str(group)] = float(group_df["mean_pqi"].mean(skipna=True))
    return result


def compute_quadrant_thresholds(df: pd.DataFrame) -> tuple[float, float]:
    """Return the 75th percentile thresholds for AWI and PQI.

    Parameters
    ----------
    df:
        DataFrame containing awi_per_minute and mean_pqi columns.

    Returns
    -------
    tuple[float, float]
        (awi_q75, pqi_q75) where each is the 75th percentile of its column.
    """
    awi_q75 = float(df["awi_per_minute"].quantile(0.75))
    pqi_q75 = float(df["mean_pqi"].quantile(0.75))
    return awi_q75, pqi_q75


def classify_quadrant(
    awi: float,
    pqi: float | None,
    awi_q75: float,
    pqi_q75: float,
) -> str:
    """Classify a player into one of four performance quadrants.

    Uses AWI and PQI values relative to their 75th percentile thresholds.
    When pqi is None or NaN, falls back to AWI-only classification.

    Parameters
    ----------
    awi:
        Player AWI value (scans per minute).
    pqi:
        Player PQI value (0-100), or None/NaN if not available.
    awi_q75:
        75th percentile threshold for AWI.
    pqi_q75:
        75th percentile threshold for PQI.

    Returns
    -------
    str
        One of "ELITE", "AWARE", "PRESSER", or "DEVELOPING".
    """
    pqi_missing = pqi is None or (isinstance(pqi, float) and math.isnan(pqi))

    if pqi_missing:
        # AWI-only fallback when PQI is not available
        if awi >= awi_q75:
            return "AWARE"
        return "DEVELOPING"

    high_awi = awi >= awi_q75
    high_pqi = float(pqi) >= pqi_q75

    if high_awi and high_pqi:
        return "ELITE"
    if high_awi and not high_pqi:
        return "AWARE"
    if not high_awi and high_pqi:
        return "PRESSER"
    return "DEVELOPING"


def format_pqi_display(pqi: float | None) -> str:
    """Format a PQI value for display.

    Returns "n/a" for NaN, None, or non-finite values. Returns a formatted
    numeric string for valid finite floats.

    Parameters
    ----------
    pqi:
        PQI value to format.

    Returns
    -------
    str
        Formatted string like "72.4" or "n/a".
    """
    if pqi is None:
        return "n/a"
    try:
        val = float(pqi)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(val):
        return "n/a"
    return f"{val:.1f}"


# ── Theme helpers ─────────────────────────────────────────────────────────────

def _theme(dark: bool) -> dict[str, str]:
    """Return a dict of color tokens for the requested theme.

    Parameters
    ----------
    dark:
        True for the standalone broadcast dark theme, False for the light
        dashboard theme.

    Returns
    -------
    dict[str, str]
        Keys: bg, surface, text, muted, border, accent, plotly_template.
    """
    if dark:
        return {
            "bg": DFL_BLACK,
            "surface": DFL_BLACK,
            "text": DFL_WHITE,
            "muted": DFL_GREY,
            "border": "#333333",
            "accent": DFL_RED,
            "plotly_template": "plotly_dark",
            "gauge_step_lo": "rgba(255,255,255,0.02)",
            "gauge_step_mid": "rgba(255,255,255,0.04)",
            "gauge_step_hi": "rgba(255,255,255,0.07)",
        }
    return {
        "bg": LIGHT_BG,
        "surface": LIGHT_SURFACE,
        "text": LIGHT_TEXT,
        "muted": LIGHT_MUTED,
        "border": LIGHT_BORDER,
        "accent": DFL_RED,
        "plotly_template": "plotly_dark",
        "gauge_step_lo": "rgba(255,255,255,0.02)",
        "gauge_step_mid": "rgba(255,255,255,0.04)",
        "gauge_step_hi": "rgba(255,255,255,0.07)",
    }


# ── Rendering Functions ───────────────────────────────────────────────────────

def inject_broadcast_css() -> str:
    """Return CSS string for the standalone broadcast overlay layout.

    Enforces a 16:9 aspect ratio container, black background, DFL font sizing,
    and white text labels. Designed for a 1280x720 presentation viewport.
    Only injected when running standalone (not embedded in the main dashboard).

    Returns
    -------
    str
        A <style> block as a string, ready for st.markdown(..., unsafe_allow_html=True).
    """
    return f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    background-color: {DFL_BLACK};
    color: {DFL_WHITE};
  }}

  .main .block-container {{
    background-color: {DFL_BLACK};
    padding: 1rem 2rem 2rem;
    max-width: 1280px;
  }}

  section[data-testid="stSidebar"] {{
    background-color: #0d0d0d;
  }}

  div[data-testid="stPlotlyChart"] .modebar {{ display: none !important; }}
</style>
"""


def build_gauge_fig(
    value: float,
    reference: float,
    label: str,
    color: str,
    axis_max: float,
    dark: bool = True,
) -> go.Figure:
    """Build a Plotly circular gauge figure.

    Adapts colors to the active theme so the gauge is legible on both the
    dark standalone page and the light main dashboard.

    Parameters
    ----------
    value:
        The numeric value to display on the gauge.
    reference:
        The reference value shown as a threshold line (e.g. league mean).
    label:
        Title text displayed above the gauge.
    color:
        Hex color string for the gauge bar fill (e.g. DFL_RED).
    axis_max:
        Maximum value for the gauge axis range.
    dark:
        True for dark broadcast theme, False for light dashboard theme.

    Returns
    -------
    go.Figure
        A Plotly Figure containing a styled go.Indicator gauge.
    """
    t = _theme(dark)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": label, "font": {"size": 11, "color": t["muted"], "family": "Inter"}},
        number={"font": {"size": 34, "color": t["text"], "family": "Inter"}, "valueformat": ".2f"},
        gauge={
            "axis": {
                "range": [0, axis_max],
                "tickwidth": 1,
                "tickcolor": t["muted"],
                "tickfont": {"size": 9, "color": t["muted"], "family": "Inter"},
            },
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, axis_max * 0.33], "color": t["gauge_step_lo"]},
                {"range": [axis_max * 0.33, axis_max * 0.66], "color": t["gauge_step_mid"]},
                {"range": [axis_max * 0.66, axis_max], "color": t["gauge_step_hi"]},
            ],
            "threshold": {
                "line": {"color": t["muted"], "width": 2},
                "thickness": 0.75,
                "value": reference,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=50, b=10, l=20, r=20),
        template=t["plotly_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=t["muted"]),
        modebar_remove=[
            "zoom", "pan", "select", "lasso",
            "zoomIn", "zoomOut", "autoScale", "resetScale", "toImage",
        ],
    )
    return fig


def ticker_loop(
    messages: list[str] = TICKER_MESSAGES,
    interval: float = TICKER_INTERVAL_SECONDS,
) -> None:
    """Animate a ticker by cycling through messages indefinitely.

    Uses st.empty() to update a single Streamlit slot and time.sleep() to
    pace the rotation. This blocks the Streamlit thread and is intended for
    standalone mode only.

    Parameters
    ----------
    messages:
        List of strings to cycle through.
    interval:
        Seconds to display each message before advancing.
    """
    placeholder = st.empty()
    idx = 0
    while True:
        msg = messages[idx % len(messages)]
        placeholder.markdown(
            f'<div style="background-color:#1E293B;color:#94A3B8;'
            f'border:1px solid #334155;border-left:3px solid {DFL_RED};'
            f'font-size:0.78rem;font-weight:500;padding:0.5rem 1rem;'
            f'letter-spacing:0.04em;margin-top:0.5rem;'
            f'border-radius:6px"><span style="color:{DFL_RED};font-weight:700;'
            f'font-size:0.6rem;letter-spacing:0.14em;text-transform:uppercase;'
            f'margin-right:0.6rem">DFL INSIGHT</span>{msg}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(interval)
        idx += 1


def render_broadcast_overlay(df: pd.DataFrame, standalone: bool = True) -> None:
    """Top-level Streamlit render function for the broadcast overlay.

    Renders a broadcast-style lower-third visualization with player selector,
    AWI and PQI circular gauges, a quadrant badge, and a scrolling ticker.

    Parameters
    ----------
    df:
        DataFrame from load_broadcast_data(). Must contain awi_per_minute,
        mean_pqi, pos_group, name, and position columns.
    standalone:
        When True, injects dark-theme CSS and runs the animated ticker_loop()
        (blocks the thread). When False (integrated tab mode), uses the light
        theme and renders only the first ticker message statically to avoid
        blocking the main dashboard.
    """
    if df.empty:
        st.warning("No broadcast data available.")
        return

    dark = standalone
    t = _theme(dark)

    # Only inject the dark broadcast CSS when running standalone.
    if standalone:
        st.markdown(inject_broadcast_css(), unsafe_allow_html=True)

    # ── Player selector ───────────────────────────────────────────────────────
    player_names = sorted(df["name"].dropna().unique())
    selected_name = st.selectbox(
        "Select Player",
        player_names,
        key="broadcast_player_select",
        label_visibility="visible",
    )

    player_rows = df[df["name"] == selected_name]
    if player_rows.empty:
        st.warning(f"No data found for {selected_name}.")
        return

    player = player_rows.iloc[0]
    position_code = str(player.get("position", "")) if pd.notna(player.get("position")) else "N/A"
    pos_group = str(player.get("pos_group", "")) if pd.notna(player.get("pos_group")) else ""

    # ── Player name bar ───────────────────────────────────────────────────────
    surface = t["surface"]
    border = t["border"]
    text_color = t["text"]
    muted = t["muted"]

    st.markdown(
        f'<div style="background:{surface};border:1px solid {border};'
        f'border-left:5px solid {DFL_RED};'
        f'padding:0.8rem 1.2rem;margin-bottom:1.2rem;border-radius:0 10px 10px 0;'
        f'box-shadow:0 1px 4px rgba(0,0,0,0.05)">'
        f'<div style="font-size:1.5rem;font-weight:800;color:{text_color};'
        f'line-height:1.2;letter-spacing:-0.02em">{selected_name}</div>'
        f'<div style="font-size:0.75rem;font-weight:500;color:{muted};'
        f'text-transform:uppercase;letter-spacing:0.12em;margin-top:0.2rem">'
        f'{position_code}'
        f'{" / " + pos_group if pos_group else ""}'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # ── Compute thresholds and reference values ───────────────────────────────
    awi_q75, pqi_q75 = compute_quadrant_thresholds(df)
    league_mean_awi = compute_league_mean_awi(df)
    role_mean_pqi_map = compute_role_mean_pqi(df)

    player_awi = float(player.get("awi_per_minute", 0.0) or 0.0)
    raw_pqi = player.get("mean_pqi")
    player_pqi: float | None = None
    if raw_pqi is not None and pd.notna(raw_pqi):
        try:
            player_pqi = float(raw_pqi)
        except (TypeError, ValueError):
            player_pqi = None

    role_ref_pqi = role_mean_pqi_map.get(pos_group, 50.0)
    pqi_display = format_pqi_display(player_pqi)

    # ── Section label helper ──────────────────────────────────────────────────
    def _label(text: str) -> None:
        st.markdown(
            f'<div style="font-size:0.6rem;font-weight:700;color:{muted};'
            f'text-transform:uppercase;letter-spacing:0.16em;margin-bottom:0.3rem">'
            f'{text}</div>',
            unsafe_allow_html=True,
        )

    # ── Gauge columns ─────────────────────────────────────────────────────────
    g1, g2, g3 = st.columns([1, 1, 1], gap="medium")

    awi_axis_max = max(float(df["awi_per_minute"].max()) * 1.1, 1.0)

    with g1:
        _label("AWI - Awareness Index")
        awi_fig = build_gauge_fig(
            value=round(player_awi, 2),
            reference=league_mean_awi,
            label="scans / min",
            color=DFL_RED,
            axis_max=awi_axis_max,
            dark=dark,
        )
        st.plotly_chart(awi_fig, width="stretch")
        # Caption below gauge
        diff_awi = player_awi - league_mean_awi
        sign_awi = "+" if diff_awi >= 0 else ""
        diff_color = "#16A34A" if diff_awi >= 0 else "#DC2626"
        st.markdown(
            f'<div style="text-align:center;font-size:0.68rem;color:{muted};margin-top:-0.3rem">'
            f'League avg <b style="color:{text_color}">{league_mean_awi:.2f}</b>'
            f' &nbsp;/&nbsp; <span style="color:{diff_color}">{sign_awi}{diff_awi:.2f}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with g2:
        _label("PQI - Press Quality Index")
        if pqi_display == "n/a" or player_pqi is None:
            pqi_fig = build_gauge_fig(
                value=0.0,
                reference=role_ref_pqi,
                label="n/a (no press frames)",
                color=muted,
                axis_max=100.0,
                dark=dark,
            )
        else:
            pqi_fig = build_gauge_fig(
                value=round(player_pqi, 1),
                reference=role_ref_pqi,
                label="0 - 100",
                color=DFL_RED,
                axis_max=100.0,
                dark=dark,
            )
        st.plotly_chart(pqi_fig, width="stretch")
        # Caption below gauge
        if player_pqi is not None:
            diff_pqi = player_pqi - role_ref_pqi
            sign_pqi = "+" if diff_pqi >= 0 else ""
            diff_pqi_color = "#16A34A" if diff_pqi >= 0 else "#DC2626"
            st.markdown(
                f'<div style="text-align:center;font-size:0.68rem;color:{muted};margin-top:-0.3rem">'
                f'Role avg <b style="color:{text_color}">{role_ref_pqi:.1f}</b>'
                f' &nbsp;/&nbsp; <span style="color:{diff_pqi_color}">{sign_pqi}{diff_pqi:.1f}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div style="text-align:center;font-size:0.68rem;color:{muted};margin-top:-0.3rem">'
                f'No press frames detected for this player</div>',
                unsafe_allow_html=True,
            )

    with g3:
        # ── Quadrant badge ────────────────────────────────────────────────────
        _label("Performance Quadrant")
        quadrant = classify_quadrant(player_awi, player_pqi, awi_q75, pqi_q75)

        if quadrant == "ELITE":
            badge_bg = DFL_RED
            badge_fg = DFL_WHITE
        elif quadrant == "DEVELOPING":
            badge_bg = muted
            badge_fg = DFL_WHITE
        elif quadrant == "AWARE":
            badge_bg = "#0284C7"
            badge_fg = DFL_WHITE
        else:  # PRESSER
            badge_bg = "#EA580C"
            badge_fg = DFL_WHITE

        st.markdown(
            f'<div style="margin-top:2rem;text-align:center">'
            f'<div style="display:inline-block;padding:0.5rem 1.6rem;font-size:1.1rem;'
            f'font-weight:700;letter-spacing:0.1em;border-radius:6px;'
            f'background:{badge_bg};color:{badge_fg};text-align:center">'
            f'{quadrant}</div>'
            f'<div style="font-size:0.68rem;color:{muted};margin-top:0.6rem;line-height:1.6">'
            f'AWI {player_awi:.2f} vs Q75 {awi_q75:.2f}<br>'
            f'PQI {pqi_display} vs Q75 {pqi_q75:.1f}'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # ── Ticker ────────────────────────────────────────────────────────────────
    st.markdown(
        f"<hr style='border-color:{border};margin:0.8rem 0 0.5rem'>",
        unsafe_allow_html=True,
    )

    if standalone:
        ticker_loop(TICKER_MESSAGES, TICKER_INTERVAL_SECONDS)
    else:
        # Show all ticker messages as a static strip
        msgs_html = " &nbsp;&middot;&nbsp; ".join(TICKER_MESSAGES)
        st.markdown(
            f'<div style="background:{LIGHT_SURFACE};color:{LIGHT_MUTED};'
            f'border:1px solid #334155;border-left:3px solid {DFL_RED};'
            f'font-size:0.75rem;font-weight:500;padding:0.55rem 1.2rem;'
            f'letter-spacing:0.03em;border-radius:6px">'
            f'<span style="color:{DFL_RED};font-weight:700;font-size:0.6rem;'
            f'letter-spacing:0.14em;text-transform:uppercase;margin-right:0.75rem">'
            f'DFL INSIGHT</span>{msgs_html}</div>',
            unsafe_allow_html=True,
        )


# ── Standalone Entry Point ────────────────────────────────────────────────────
# Only render when this file is the Streamlit entry point (i.e. the user ran
# `streamlit run dashboard/broadcast_demo.py`).  When app.py imports us, the
# __name__ is "dashboard.broadcast_demo", not "__main__", so this block is
# skipped and no dark-theme CSS leaks into the main dashboard.
def _is_standalone_entry() -> bool:
    """Return True only when this module is the Streamlit entry point."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        ctx = get_script_run_ctx()
        if ctx is None:
            return False
        # Check if this file is the one Streamlit was told to run
        import __main__
        main_file = getattr(__main__, "__file__", "") or ""
        return os.path.basename(main_file) == "broadcast_demo.py"
    except Exception:
        return False


if _is_standalone_entry():
    _broadcast_df = load_broadcast_data()
    render_broadcast_overlay(_broadcast_df, standalone=True)
