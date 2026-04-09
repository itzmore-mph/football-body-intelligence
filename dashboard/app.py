import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Football Body Intelligence", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    awi_path = "results/awi_full.csv"
    pqi_path = "results/pqi_full.csv"

    if not os.path.exists(pqi_path):
        st.error(
            "results/pqi_full.csv not found. "
            "Please run notebooks/run_pqi_pipeline.ipynb to generate it."
        )
        st.stop()

    awi = pd.read_csv(awi_path)
    pqi = pd.read_csv(pqi_path)

    merged = awi.merge(
        pqi[
            [
                "jersey",
                "team",
                "match_id",
                "phase_label",
                "mean_pqi",
                "median_pqi",
                "std_pqi",
                "n_press_frames",
                "press_minutes",
                "orientation_mean",
                "stance_mean",
                "proximity_mean",
                "coverage_pct",
            ]
        ],
        on=["jersey", "team", "match_id", "phase_label"],
        how="left",
    )
    return merged


df = load_data()

# Sidebar filters
st.sidebar.header("Filters")

match_options = ["All"] + sorted(df["match_id"].unique())
selected_match = st.sidebar.selectbox("Match", match_options)

phase_options = ["All", "1st half", "2nd half"]
selected_phase = st.sidebar.selectbox("Phase", phase_options)

position_options = sorted(df["position"].dropna().unique())
selected_positions = st.sidebar.multiselect("Position", position_options)

min_coverage = st.sidebar.slider("Min Coverage %", 0, 100, 0)

# Apply filters
filtered_df = df.copy()

if selected_match != "All":
    filtered_df = filtered_df[filtered_df["match_id"] == selected_match]

if selected_phase != "All":
    filtered_df = filtered_df[filtered_df["phase_label"] == selected_phase]

if selected_positions:
    filtered_df = filtered_df[filtered_df["position"].isin(selected_positions)]

if "coverage_pct" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["coverage_pct"] >= min_coverage / 100]

# Tabs
tab1, tab2, tab3 = st.tabs(["Player Profile", "Match Overview", "League Leaderboard"])

with tab1:
    st.subheader("Player Profile")

    if filtered_df.empty:
        st.warning("No data matches the current filters.")
    else:
        # Player selector
        player_options = sorted(
            filtered_df.apply(lambda r: f"{r['name']} (#{r['jersey']})", axis=1).unique()
        )
        selected_player_str = st.selectbox("Select Player", player_options)

        # Extract jersey from selection
        selected_jersey = int(selected_player_str.split("#")[1].rstrip(")"))
        selected_name = selected_player_str.split(" (#")[0]
        player_row = filtered_df[
            (filtered_df["jersey"] == selected_jersey) &
            (filtered_df["name"] == selected_name)
        ].iloc[0] if not filtered_df[
            (filtered_df["jersey"] == selected_jersey) &
            (filtered_df["name"] == selected_name)
        ].empty else None

        if player_row is not None:
            col1, col2 = st.columns(2)

            with col1:
                # AWI gauge
                awi_val = player_row.get("awi_per_minute", 0) or 0
                fig_awi = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=awi_val,
                    title={"text": "AWI (scans/min)"},
                    gauge={"axis": {"range": [0, filtered_df["awi_per_minute"].max() * 1.1]},
                           "bar": {"color": "steelblue"}},
                ))
                fig_awi.update_layout(height=300)
                st.plotly_chart(fig_awi, use_container_width=True)

            with col2:
                # PQI gauge
                pqi_val = player_row.get("mean_pqi", 0) or 0
                fig_pqi = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=pqi_val,
                    title={"text": "Mean PQI (0–100)"},
                    gauge={"axis": {"range": [0, 100]},
                           "bar": {"color": "coral"}},
                ))
                fig_pqi.update_layout(height=300)
                st.plotly_chart(fig_pqi, use_container_width=True)

            col3, col4 = st.columns(2)

            with col3:
                # AWI vs PQI scatter with player highlighted
                fig_scatter = px.scatter(
                    filtered_df,
                    x="awi_per_minute",
                    y="mean_pqi",
                    color="position",
                    hover_data=["name", "jersey", "match_id", "phase_label"],
                    title="AWI vs PQI (selected player highlighted)",
                    opacity=0.6,
                )
                # Highlight selected player
                highlight = filtered_df[
                    (filtered_df["jersey"] == selected_jersey) &
                    (filtered_df["name"] == selected_name)
                ]
                fig_scatter.add_trace(go.Scatter(
                    x=highlight["awi_per_minute"],
                    y=highlight["mean_pqi"],
                    mode="markers",
                    marker=dict(size=16, color="gold", symbol="star", line=dict(color="black", width=1)),
                    name=selected_name,
                    showlegend=True,
                ))
                st.plotly_chart(fig_scatter, use_container_width=True)

            with col4:
                # Scan direction breakdown pie
                fwd = player_row.get("scan_forward_pct", None)
                lat = player_row.get("scan_lateral_pct", None)
                blind = player_row.get("scan_blindside_pct", None)

                if fwd is not None and not pd.isna(fwd):
                    fig_pie = go.Figure(go.Pie(
                        labels=["Forward", "Lateral", "Blind-side"],
                        values=[fwd, lat, blind],
                        hole=0.3,
                    ))
                    fig_pie.update_layout(title="Scan Direction Breakdown")
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("Scan direction data not available for this player.")

with tab2:
    st.subheader("Match Overview")

    if filtered_df.empty:
        st.warning("No data matches the current filters.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            # Top 10 AWI bar chart
            top_awi = filtered_df.nlargest(10, "awi_per_minute")[["name", "awi_per_minute", "position"]].copy()
            fig_awi_bar = px.bar(
                top_awi, x="awi_per_minute", y="name", orientation="h",
                color="position", title="Top 10 Players by AWI",
                labels={"awi_per_minute": "AWI (scans/min)", "name": ""},
            )
            fig_awi_bar.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_awi_bar, use_container_width=True)

        with col2:
            # Top 10 PQI bar chart
            top_pqi = filtered_df.dropna(subset=["mean_pqi"]).nlargest(10, "mean_pqi")[["name", "mean_pqi", "position"]].copy()
            fig_pqi_bar = px.bar(
                top_pqi, x="mean_pqi", y="name", orientation="h",
                color="position", title="Top 10 Players by PQI",
                labels={"mean_pqi": "Mean PQI", "name": ""},
            )
            fig_pqi_bar.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_pqi_bar, use_container_width=True)

        # AWI vs PQI scatter for all players in match
        fig_match_scatter = px.scatter(
            filtered_df, x="awi_per_minute", y="mean_pqi",
            color="position", hover_data=["name", "jersey", "match_id", "phase_label"],
            title="AWI vs PQI — All Players",
            opacity=0.7,
        )
        st.plotly_chart(fig_match_scatter, use_container_width=True)

        # 1st/2nd half delta comparison
        half1 = filtered_df[filtered_df["phase_label"] == "1st half"][["name", "awi_per_minute", "mean_pqi"]].rename(
            columns={"awi_per_minute": "awi_h1", "mean_pqi": "pqi_h1"})
        half2 = filtered_df[filtered_df["phase_label"] == "2nd half"][["name", "awi_per_minute", "mean_pqi"]].rename(
            columns={"awi_per_minute": "awi_h2", "mean_pqi": "pqi_h2"})
        halves = half1.merge(half2, on="name", how="inner")

        if not halves.empty:
            halves["awi_delta"] = halves["awi_h2"] - halves["awi_h1"]
            halves["pqi_delta"] = halves["pqi_h2"] - halves["pqi_h1"]
            top_halves = halves.nlargest(15, "awi_h1")

            fig_delta = go.Figure()
            fig_delta.add_trace(go.Bar(
                name="AWI delta", x=top_halves["name"], y=top_halves["awi_delta"],
                marker_color="steelblue", opacity=0.8,
            ))
            fig_delta.add_trace(go.Bar(
                name="PQI delta", x=top_halves["name"], y=top_halves["pqi_delta"],
                marker_color="coral", opacity=0.8,
            ))
            fig_delta.update_layout(
                title="Half-Time Delta (2nd − 1st Half)",
                barmode="group",
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_delta, use_container_width=True)
        else:
            st.info("Both halves needed for delta comparison. Adjust filters.")

with tab3:
    st.subheader("League Leaderboard")

    if filtered_df.empty:
        st.warning("No data matches the current filters.")
    else:
        # Sortable table
        display_cols = ["jersey", "name", "position", "match_id", "phase_label",
                        "awi_per_minute", "mean_pqi", "coverage_pct"]
        available_cols = [c for c in display_cols if c in filtered_df.columns]
        st.dataframe(
            filtered_df[available_cols].sort_values("awi_per_minute", ascending=False).reset_index(drop=True),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            # Position average heatmap
            heatmap_cols = ["awi_per_minute", "mean_pqi", "orientation_mean", "stance_mean", "proximity_mean"]
            avail_hm = [c for c in heatmap_cols if c in filtered_df.columns]
            pos_avg = filtered_df.groupby("position")[avail_hm].mean().dropna(how="all").reset_index()

            if not pos_avg.empty and len(avail_hm) > 0:
                fig_hm = px.imshow(
                    pos_avg.set_index("position")[avail_hm],
                    text_auto=".1f",
                    color_continuous_scale="YlOrRd",
                    title="Position Average Heatmap",
                    aspect="auto",
                )
                st.plotly_chart(fig_hm, use_container_width=True)

        with col2:
            # AWI distribution by position
            if "awi_per_minute" in filtered_df.columns:
                fig_dist = px.box(
                    filtered_df, x="position", y="awi_per_minute",
                    title="AWI Distribution by Position",
                    color="position",
                )
                fig_dist.update_layout(showlegend=False, xaxis_tickangle=-45)
                st.plotly_chart(fig_dist, use_container_width=True)

        if "mean_pqi" in filtered_df.columns:
            fig_pqi_dist = px.box(
                filtered_df.dropna(subset=["mean_pqi"]),
                x="position", y="mean_pqi",
                title="PQI Distribution by Position",
                color="position",
            )
            fig_pqi_dist.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_pqi_dist, use_container_width=True)
