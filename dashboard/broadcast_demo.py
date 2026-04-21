"""
Broadcast Demo - Fan Broadcast Overlay for Football Body Intelligence Platform

Standalone broadcast-style visualization mockup showing AWI and PQI as live
match statistics in a Bundesliga broadcast lower-third format.

Usage:
    streamlit run dashboard/broadcast_demo.py
"""
import math
import os
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── DFL Color System ──────────────────────────────────────────────────────────
DFL_RED   = "#D20515"
DFL_BLACK = "#000000"
DFL_WHITE = "#FFFFFF"
DFL_GREY  = "#8A8A8A"

# ── Ticker Content ────────────────────────────────────────────────────────────
TICKER_MESSAGES: list[str] = [
    "+57% pre-pass scan spike",
    "r=-0.11 AWI/PQI independence",
    "R=0.854 cross-half stability",
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

def load_broadcast_data(path: str = "results/combined_full.csv") -> pd.DataFrame:
    """Load and enrich the merged dataset.

    Reads the combined CSV, maps position codes to position groups via POS_MAP,
    and returns the enriched DataFrame. Calls st.error() and st.stop() if the
    file is not found.

    Parameters
    ----------
    path:
        Path to the combined CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added pos_group column.
    """
    if not os.path.exists(path):
        st.error(f"`{path}` not found. Run the pipeline first to generate the data file.")
        st.stop()
    df = pd.read_csv(path)
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


# ── Rendering Functions ───────────────────────────────────────────────────────

def inject_broadcast_css() -> str:
    """Return CSS string for the broadcast overlay layout.

    Enforces a 16:9 aspect ratio container, black background, DFL font sizing,
    and white text labels. Designed for a 1280x720 presentation viewport.

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

  .broadcast-container {{
    background-color: {DFL_BLACK};
    aspect-ratio: 16 / 9;
    max-width: 1280px;
    margin: 0 auto;
    padding: 1rem;
    color: {DFL_WHITE};
  }}

  .broadcast-player-bar {{
    background-color: {DFL_BLACK};
    border-left: 4px solid {DFL_RED};
    padding: 0.5rem 1rem;
    margin-bottom: 1rem;
  }}

  .broadcast-player-name {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {DFL_WHITE};
    line-height: 1.2;
  }}

  .broadcast-player-pos {{
    font-size: 0.85rem;
    font-weight: 500;
    color: {DFL_GREY};
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }}

  .broadcast-label {{
    font-size: 0.65rem;
    font-weight: 600;
    color: {DFL_GREY};
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin-bottom: 0.2rem;
  }}

  .broadcast-badge {{
    display: inline-block;
    padding: 0.4rem 1.2rem;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    border-radius: 4px;
    text-align: center;
    color: {DFL_WHITE};
  }}

  .broadcast-ticker {{
    background-color: {DFL_RED};
    color: {DFL_WHITE};
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0.35rem 1rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 0.5rem;
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
) -> go.Figure:
    """Build a Plotly circular gauge figure with DFL broadcast styling.

    Matches the layout style of gauge_fig() in app.py: transparent backgrounds,
    Inter font, and a threshold reference line. Uses the passed color for the
    gauge bar fill.

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

    Returns
    -------
    go.Figure
        A Plotly Figure containing a styled go.Indicator gauge.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": label, "font": {"size": 11, "color": DFL_GREY, "family": "Inter"}},
        number={"font": {"size": 34, "color": DFL_WHITE, "family": "Inter"}, "valueformat": ".2f"},
        gauge={
            "axis": {
                "range": [0, axis_max],
                "tickwidth": 1,
                "tickcolor": DFL_GREY,
                "tickfont": {"size": 9, "color": DFL_GREY, "family": "Inter"},
            },
            "bar": {"color": color, "thickness": 0.32},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, axis_max * 0.33], "color": "rgba(255,255,255,0.02)"},
                {"range": [axis_max * 0.33, axis_max * 0.66], "color": "rgba(255,255,255,0.04)"},
                {"range": [axis_max * 0.66, axis_max], "color": "rgba(255,255,255,0.07)"},
            ],
            "threshold": {
                "line": {"color": DFL_GREY, "width": 2},
                "thickness": 0.75,
                "value": reference,
            },
        },
    ))
    fig.update_layout(
        height=240,
        margin=dict(t=50, b=10, l=20, r=20),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color=DFL_WHITE),
        modebar_remove=["zoom", "pan", "select", "lasso", "zoomIn", "zoomOut", "autoScale", "resetScale", "toImage"],
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
            f'<div class="broadcast-ticker">DFL INSIGHT: {msg}</div>',
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
        When True, runs the animated ticker_loop() (blocks the thread).
        When False (integrated tab mode), renders only the first ticker
        message statically to avoid blocking the main dashboard.
    """
    if df.empty:
        st.warning("No broadcast data available.")
        return

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

    # ── Player name bar ───────────────────────────────────────────────────────
    st.markdown(
        f'<div class="broadcast-player-bar">'
        f'<div class="broadcast-player-name">{selected_name}</div>'
        f'<div class="broadcast-player-pos">{position_code}</div>'
        f'</div>',
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

    pos_group = str(player.get("pos_group", "")) if pd.notna(player.get("pos_group")) else ""
    role_ref_pqi = role_mean_pqi_map.get(pos_group, 50.0)

    # ── Gauge columns ─────────────────────────────────────────────────────────
    g1, g2, g3 = st.columns([1, 1, 1], gap="medium")

    with g1:
        st.markdown('<div class="broadcast-label">AWI - Awareness Index</div>', unsafe_allow_html=True)
        awi_axis_max = float(df["awi_per_minute"].max()) * 1.1
        awi_fig = build_gauge_fig(
            value=round(player_awi, 2),
            reference=league_mean_awi,
            label="scans / min",
            color=DFL_RED,
            axis_max=awi_axis_max,
        )
        st.plotly_chart(awi_fig, width='stretch')

    with g2:
        st.markdown('<div class="broadcast-label">PQI - Press Quality Index</div>', unsafe_allow_html=True)
        pqi_display = format_pqi_display(player_pqi)
        if pqi_display == "n/a" or player_pqi is None:
            # Show a placeholder gauge at 0 with "n/a" label when PQI is missing
            pqi_fig = build_gauge_fig(
                value=0.0,
                reference=role_ref_pqi,
                label="n/a (no press frames)",
                color=DFL_GREY,
                axis_max=100.0,
            )
        else:
            pqi_fig = build_gauge_fig(
                value=round(player_pqi, 1),
                reference=role_ref_pqi,
                label="0 - 100",
                color=DFL_RED,
                axis_max=100.0,
            )
        st.plotly_chart(pqi_fig, width='stretch')

    with g3:
        # ── Quadrant badge ────────────────────────────────────────────────────
        st.markdown('<div class="broadcast-label">Performance Quadrant</div>', unsafe_allow_html=True)
        quadrant = classify_quadrant(player_awi, player_pqi, awi_q75, pqi_q75)

        if quadrant == "ELITE":
            badge_color = DFL_RED
        elif quadrant == "DEVELOPING":
            badge_color = DFL_GREY
        else:
            badge_color = "#333333"

        st.markdown(
            f'<div style="margin-top:2.5rem; text-align:center;">'
            f'<div class="broadcast-badge" style="background-color:{badge_color};">'
            f'{quadrant}'
            f'</div>'
            f'<div style="font-size:0.7rem;color:{DFL_GREY};margin-top:0.5rem;">'
            f'AWI {player_awi:.2f} vs threshold {awi_q75:.2f}<br>'
            f'PQI {pqi_display} vs threshold {pqi_q75:.1f}'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── Ticker ────────────────────────────────────────────────────────────────
    st.markdown("<hr style='border-color:#222;margin:0.5rem 0'>", unsafe_allow_html=True)

    if standalone:
        ticker_loop(TICKER_MESSAGES, TICKER_INTERVAL_SECONDS)
    else:
        # Static first message for integrated tab mode (avoids blocking the thread)
        first_msg = TICKER_MESSAGES[0] if TICKER_MESSAGES else ""
        st.markdown(
            f'<div class="broadcast-ticker">DFL INSIGHT: {first_msg}</div>',
            unsafe_allow_html=True,
        )


# ── Standalone Entry Point ────────────────────────────────────────────────────
# Streamlit executes the entire module on each run. We detect a live Streamlit
# runtime by checking for the script run context, which is only present when
# the module is executed via `streamlit run`, not during pytest import.
def _is_streamlit_runtime() -> bool:
    """Return True only when running inside a live Streamlit session."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if _is_streamlit_runtime():
    _broadcast_df = load_broadcast_data()
    render_broadcast_overlay(_broadcast_df, standalone=True)
