import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Football Body Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global style tweaks ───────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        /* tighten sidebar padding */
        section[data-testid="stSidebar"] { padding-top: 1rem; }
        /* remove top whitespace on main area */
        .block-container { padding-top: 1.5rem; }
        /* push tab content below the sticky tab bar */
        div[data-testid="stTabContent"] { padding-top: 1.2rem; }
        /* subtle card-like chart containers */
        div[data-testid="stPlotlyChart"] {
            border-radius: 8px;
            overflow: hidden;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

CHART_THEME = "plotly_dark"
PRIMARY   = "#4C9BE8"   # blue  – AWI
SECONDARY = "#F4845F"   # coral – PQI
GOLD      = "#FFD700"


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    awi_path = "results/awi_full.csv"
    pqi_path = "results/pqi_full.csv"

    for path in (awi_path, pqi_path):
        if not os.path.exists(path):
            st.error(f"`{path}` not found. Run the corresponding pipeline notebook first.")
            st.stop()

    awi = pd.read_csv(awi_path)
    pqi = pd.read_csv(pqi_path)

    pqi_cols = [
        "jersey", "team", "match_id", "phase_label",
        "mean_pqi", "median_pqi", "std_pqi",
        "n_press_frames", "press_minutes",
        "orientation_mean", "stance_mean", "proximity_mean", "coverage_pct",
    ]
    merged = awi.merge(
        pqi[[c for c in pqi_cols if c in pqi.columns]],
        on=["jersey", "team", "match_id", "phase_label"],
        how="left",
    )
    return merged


df = load_data()


# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")

    match_options = ["All"] + sorted(df["match_id"].unique())
    selected_match = st.selectbox("Match", match_options)

    phase_options = ["All", "1st half", "2nd half"]
    selected_phase = st.selectbox("Phase", phase_options)

    position_options = sorted(df["position"].dropna().unique())
    selected_positions = st.multiselect(
        "Position",
        position_options,
        placeholder="All positions",
    )

    min_coverage = st.slider("Min Coverage %", 0, 100, 0)

    st.divider()
    st.caption("Football Body Intelligence · Hackathon 2026")


# ── Apply filters ─────────────────────────────────────────────────────────────
filtered_df = df.copy()

if selected_match != "All":
    filtered_df = filtered_df[filtered_df["match_id"] == selected_match]

if selected_phase != "All":
    filtered_df = filtered_df[filtered_df["phase_label"] == selected_phase]

if selected_positions:
    filtered_df = filtered_df[filtered_df["position"].isin(selected_positions)]

if "coverage_pct" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["coverage_pct"] >= min_coverage / 100]

# Drop rows with no meaningful data for display purposes
filtered_df_valid = filtered_df[filtered_df["awi_per_minute"] > 0].copy()


# ── Helper: empty-state guard ─────────────────────────────────────────────────
def _empty(frame: pd.DataFrame) -> bool:
    if frame.empty:
        st.warning("No data matches the current filters.")
        return True
    return False


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["👤  Player Profile", "📊  Match Overview", "🏆  League Leaderboard"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Player Profile
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Player Profile")

    if _empty(filtered_df_valid):
        st.stop()

    player_options = sorted(
        filtered_df_valid
        .apply(lambda r: f"{r['name']} (#{int(r['jersey'])})", axis=1)
        .unique()
    )
    selected_player_str = st.selectbox("Select Player", player_options, key="player_select")

    selected_jersey = int(selected_player_str.split("#")[1].rstrip(")"))
    selected_name   = selected_player_str.split(" (#")[0]

    mask = (
        (filtered_df_valid["jersey"] == selected_jersey) &
        (filtered_df_valid["name"]   == selected_name)
    )
    player_rows = filtered_df_valid[mask]

    if player_rows.empty:
        st.warning("Player not found in filtered data.")
        st.stop()

    player_row = player_rows.iloc[0]

    # ── KPI gauges ────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        awi_val  = float(player_row.get("awi_per_minute", 0) or 0)
        awi_max  = max(filtered_df_valid["awi_per_minute"].max() * 1.15, 1)
        fig_awi  = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(awi_val, 2),
            title={"text": "AWI (scans / min)", "font": {"size": 16}},
            number={"font": {"size": 40}},
            gauge={
                "axis": {"range": [0, awi_max], "tickwidth": 1},
                "bar":  {"color": PRIMARY},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, awi_max * 0.33], "color": "rgba(76,155,232,0.1)"},
                    {"range": [awi_max * 0.33, awi_max * 0.66], "color": "rgba(76,155,232,0.2)"},
                    {"range": [awi_max * 0.66, awi_max], "color": "rgba(76,155,232,0.3)"},
                ],
            },
        ))
        fig_awi.update_layout(
            height=280, margin=dict(t=40, b=10, l=20, r=20),
            template=CHART_THEME, paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_awi, width="stretch")

    with col2:
        pqi_val = float(player_row.get("mean_pqi", 0) or 0)
        fig_pqi = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(pqi_val, 1),
            title={"text": "Mean PQI (0 – 100)", "font": {"size": 16}},
            number={"font": {"size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar":  {"color": SECONDARY},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0,  33], "color": "rgba(244,132,95,0.1)"},
                    {"range": [33, 66], "color": "rgba(244,132,95,0.2)"},
                    {"range": [66, 100], "color": "rgba(244,132,95,0.3)"},
                ],
            },
        ))
        fig_pqi.update_layout(
            height=280, margin=dict(t=40, b=10, l=20, r=20),
            template=CHART_THEME, paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_pqi, width="stretch")

    st.divider()

    # ── Scatter + Scan direction ───────────────────────────────────────────────
    fwd   = player_row.get("scan_forward_pct",   None)
    lat   = player_row.get("scan_lateral_pct",   None)
    blind = player_row.get("scan_blindside_pct", None)
    has_scan = fwd is not None and not pd.isna(fwd)

    if has_scan:
        scatter_col, pie_col = st.columns([3, 2])
    else:
        scatter_col = st.container()

    with scatter_col:
        fig_scatter = px.scatter(
            filtered_df_valid,
            x="awi_per_minute", y="mean_pqi",
            color="position",
            hover_data=["name", "jersey", "match_id", "phase_label"],
            title="AWI vs PQI — selected player highlighted",
            opacity=0.55,
            template=CHART_THEME,
        )
        highlight = filtered_df_valid[mask]
        fig_scatter.add_trace(go.Scatter(
            x=highlight["awi_per_minute"],
            y=highlight["mean_pqi"],
            mode="markers",
            marker=dict(size=18, color=GOLD, symbol="star",
                        line=dict(color="white", width=1)),
            name=selected_name,
            showlegend=True,
        ))
        fig_scatter.update_layout(
            height=380, margin=dict(t=50, b=40, l=40, r=20),
            legend=dict(orientation="v", x=1.01, y=1),
        )
        st.plotly_chart(fig_scatter, width="stretch")

    if has_scan:
        with pie_col:  # type: ignore[possibly-undefined]
            fig_pie = go.Figure(go.Pie(
                labels=["Forward", "Lateral", "Blind-side"],
                values=[fwd, lat, blind],
                hole=0.35,
                marker_colors=[PRIMARY, SECONDARY, "#A8DADC"],
            ))
            fig_pie.update_layout(
                title="Scan Direction Breakdown",
                height=380, margin=dict(t=50, b=20, l=20, r=20),
                template=CHART_THEME, paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.1),
            )
            st.plotly_chart(fig_pie, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Match Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Match Overview")

    if _empty(filtered_df_valid):
        st.stop()

    col1, col2 = st.columns(2)

    with col1:
        top_awi = (
            filtered_df_valid
            .nlargest(10, "awi_per_minute")[["name", "awi_per_minute", "position"]]
            .copy()
        )
        fig_awi_bar = px.bar(
            top_awi, x="awi_per_minute", y="name", orientation="h",
            color="position",
            title="Top 10 Players by AWI",
            labels={"awi_per_minute": "AWI (scans/min)", "name": ""},
            template=CHART_THEME,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_awi_bar.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=380, margin=dict(t=50, b=40, l=10, r=20),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_awi_bar, width="stretch")

    with col2:
        top_pqi = (
            filtered_df_valid
            .dropna(subset=["mean_pqi"])
            .nlargest(10, "mean_pqi")[["name", "mean_pqi", "position"]]
            .copy()
        )
        fig_pqi_bar = px.bar(
            top_pqi, x="mean_pqi", y="name", orientation="h",
            color="position",
            title="Top 10 Players by PQI",
            labels={"mean_pqi": "Mean PQI", "name": ""},
            template=CHART_THEME,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_pqi_bar.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=380, margin=dict(t=50, b=40, l=10, r=20),
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_pqi_bar, width="stretch")

    st.divider()

    # AWI vs PQI scatter – all players
    fig_match_scatter = px.scatter(
        filtered_df_valid,
        x="awi_per_minute", y="mean_pqi",
        color="position",
        hover_data=["name", "jersey", "match_id", "phase_label"],
        title="AWI vs PQI — All Players",
        opacity=0.7,
        template=CHART_THEME,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig_match_scatter.update_layout(
        height=420, margin=dict(t=50, b=40, l=40, r=20),
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_match_scatter, width="stretch")

    st.divider()

    # Half-time delta
    half1 = (
        filtered_df_valid[filtered_df_valid["phase_label"] == "1st half"]
        [["name", "awi_per_minute", "mean_pqi"]]
        .rename(columns={"awi_per_minute": "awi_h1", "mean_pqi": "pqi_h1"})
    )
    half2 = (
        filtered_df_valid[filtered_df_valid["phase_label"] == "2nd half"]
        [["name", "awi_per_minute", "mean_pqi"]]
        .rename(columns={"awi_per_minute": "awi_h2", "mean_pqi": "pqi_h2"})
    )
    halves = half1.merge(half2, on="name", how="inner")

    if not halves.empty:
        halves["awi_delta"] = halves["awi_h2"] - halves["awi_h1"]
        halves["pqi_delta"] = halves["pqi_h2"] - halves["pqi_h1"]
        top_halves = halves.nlargest(15, "awi_h1")

        fig_delta = go.Figure()
        fig_delta.add_trace(go.Bar(
            name="AWI Δ", x=top_halves["name"], y=top_halves["awi_delta"],
            marker_color=PRIMARY, opacity=0.85,
        ))
        fig_delta.add_trace(go.Bar(
            name="PQI Δ", x=top_halves["name"], y=top_halves["pqi_delta"],
            marker_color=SECONDARY, opacity=0.85,
        ))
        fig_delta.update_layout(
            title="Half-Time Delta (2nd − 1st Half)",
            barmode="group",
            xaxis_tickangle=-40,
            height=420, margin=dict(t=50, b=100, l=40, r=20),
            template=CHART_THEME,
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig_delta, width="stretch")
    else:
        st.info("Both halves needed for delta comparison — adjust filters.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – League Leaderboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("League Leaderboard")

    if _empty(filtered_df_valid):
        st.stop()

    # Sortable table – only rows with real data
    display_cols = ["jersey", "name", "position", "match_id", "phase_label",
                    "awi_per_minute", "mean_pqi", "coverage_pct"]
    available_cols = [c for c in display_cols if c in filtered_df_valid.columns]

    table_df = (
        filtered_df_valid[available_cols]
        .sort_values("awi_per_minute", ascending=False)
        .reset_index(drop=True)
    )
    # Friendly column formatting
    if "coverage_pct" in table_df.columns:
        table_df["coverage_pct"] = (table_df["coverage_pct"] * 100).round(1)

    st.dataframe(
        table_df,
        width="stretch",
        height=min(40 + len(table_df) * 35, 500),
        column_config={
            "awi_per_minute": st.column_config.NumberColumn("AWI (scans/min)", format="%.2f"),
            "mean_pqi":       st.column_config.NumberColumn("Mean PQI",        format="%.1f"),
            "coverage_pct":   st.column_config.NumberColumn("Coverage %",      format="%.1f"),
        },
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        heatmap_cols = ["awi_per_minute", "mean_pqi", "orientation_mean", "stance_mean", "proximity_mean"]
        avail_hm = [c for c in heatmap_cols if c in filtered_df_valid.columns]
        pos_avg  = (
            filtered_df_valid
            .groupby("position")[avail_hm]
            .mean()
            .dropna(how="all")
            .reset_index()
        )

        if not pos_avg.empty and avail_hm:
            fig_hm = px.imshow(
                pos_avg.set_index("position")[avail_hm],
                text_auto=".1f",
                color_continuous_scale="YlOrRd",
                title="Position Average Heatmap",
                aspect="auto",
                template=CHART_THEME,
            )
            fig_hm.update_layout(
                height=max(250, len(pos_avg) * 55 + 80),
                margin=dict(t=50, b=60, l=20, r=20),
                xaxis_tickangle=-30,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig_hm, width="stretch")

    with col2:
        fig_dist = px.box(
            filtered_df_valid, x="position", y="awi_per_minute",
            title="AWI Distribution by Position",
            color="position",
            template=CHART_THEME,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_dist.update_layout(
            showlegend=False,
            xaxis_tickangle=-40,
            height=max(250, len(pos_avg) * 55 + 80) if not pos_avg.empty else 350,
            margin=dict(t=50, b=80, l=40, r=20),
        )
        st.plotly_chart(fig_dist, width="stretch")

    if "mean_pqi" in filtered_df_valid.columns:
        fig_pqi_dist = px.box(
            filtered_df_valid.dropna(subset=["mean_pqi"]),
            x="position", y="mean_pqi",
            title="PQI Distribution by Position",
            color="position",
            template=CHART_THEME,
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        fig_pqi_dist.update_layout(
            showlegend=False,
            xaxis_tickangle=-40,
            height=380, margin=dict(t=50, b=80, l=40, r=20),
        )
        st.plotly_chart(fig_pqi_dist, width="stretch")
