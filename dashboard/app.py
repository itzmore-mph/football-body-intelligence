"""
Football Body Intelligence — Streamlit Dashboard
"""
import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Football Body Intelligence",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
PRIMARY    = "#4C9BE8"   # blue  – AWI
SECONDARY  = "#F4845F"   # coral – PQI
GOLD       = "#FFD700"   # highlight
BG         = "#0e1117"
CARD_BG    = "#161b22"
BORDER     = "#30363d"
TEXT_MUTED = "#8b949e"
CHART_THEME = "plotly_dark"
PASTEL = px.colors.qualitative.Pastel

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  /* ── Layout ── */
  .block-container {{ padding-top: 0.75rem; padding-bottom: 2rem; }}
  section[data-testid="stSidebar"] {{ padding-top: 1.25rem; background-color: {CARD_BG}; }}

  /* ── Sticky tab bar ── */
  div[data-testid="stTabs"] > div:first-child {{
      position: sticky; top: 0; z-index: 100;
      background-color: {BG};
      border-bottom: 1px solid {BORDER};
      padding-bottom: 2px;
  }}
  div[data-testid="stTabContent"] {{ padding-top: 1.5rem; }}

  /* ── KPI metric cards ── */
  .kpi-card {{
      background: {CARD_BG};
      border: 1px solid {BORDER};
      border-radius: 10px;
      padding: 1rem 1.25rem 0.75rem;
      text-align: center;
  }}
  .kpi-label {{ font-size: 0.75rem; color: {TEXT_MUTED}; text-transform: uppercase;
                letter-spacing: 0.08em; margin-bottom: 0.2rem; }}
  .kpi-value {{ font-size: 2rem; font-weight: 700; line-height: 1.1; }}
  .kpi-sub   {{ font-size: 0.75rem; color: {TEXT_MUTED}; margin-top: 0.2rem; }}

  /* ── Section headers ── */
  .section-header {{
      font-size: 0.7rem; font-weight: 600; color: {TEXT_MUTED};
      text-transform: uppercase; letter-spacing: 0.1em;
      border-bottom: 1px solid {BORDER}; padding-bottom: 4px;
      margin-bottom: 0.75rem; margin-top: 0.5rem;
  }}

  /* ── Charts ── */
  div[data-testid="stPlotlyChart"] {{ border-radius: 8px; overflow: hidden; }}

  /* ── Sidebar filter labels ── */
  .sidebar-label {{ font-size: 0.7rem; color: {TEXT_MUTED}; text-transform: uppercase;
                    letter-spacing: 0.08em; margin-bottom: 2px; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi(label: str, value: str, sub: str = "", color: str = "white") -> None:
    st.markdown(
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color}">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def _tooltip(r: pd.Series) -> str:
    pqi = f"{r['mean_pqi']:.1f}" if pd.notna(r.get("mean_pqi")) else "n/a"
    return (
        f"<b>{r['name']}</b> &nbsp;#{int(r['jersey'])}<br>"
        f"<span style='color:#8b949e'>{r['position']} · {r['match_id']} · {r['phase_label']}</span><br>"
        f"AWI: <b>{r['awi_per_minute']:.2f}</b> scans/min<br>"
        f"PQI: <b>{pqi}</b>"
    )


def _empty(frame: pd.DataFrame) -> bool:
    if frame.empty:
        st.warning("No data matches the current filters.")
        return True
    return False


def _scatter_with_tooltips(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_tip"] = out.apply(_tooltip, axis=1)
    return out


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    for path in ("results/awi_full.csv", "results/pqi_full.csv"):
        if not os.path.exists(path):
            st.error(f"`{path}` not found — run the pipeline notebook first.")
            st.stop()

    awi = pd.read_csv("results/awi_full.csv")
    pqi = pd.read_csv("results/pqi_full.csv")

    pqi_cols = [
        "jersey", "team", "match_id", "phase_label",
        "mean_pqi", "median_pqi", "std_pqi",
        "n_press_frames", "press_minutes",
        "orientation_mean", "stance_mean", "proximity_mean",
    ]
    merged = awi.merge(
        pqi[[c for c in pqi_cols if c in pqi.columns]],
        on=["jersey", "team", "match_id", "phase_label"],
        how="left",
    )
    # single coverage_pct from AWI (already 0-1)
    return merged


df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚽ FBI Dashboard")
    st.markdown(f"<span style='color:{TEXT_MUTED};font-size:0.8rem'>Football Body Intelligence</span>",
                unsafe_allow_html=True)
    st.divider()

    match_options = ["All"] + sorted(df["match_id"].unique())
    selected_match = st.selectbox("Match", match_options)

    phase_options = ["All", "1st half", "2nd half"]
    selected_phase = st.selectbox("Phase", phase_options)

    position_options = sorted(df["position"].dropna().unique())
    selected_positions = st.multiselect("Position", position_options, placeholder="All positions")

    min_coverage = st.slider("Min Coverage %", 0, 100, 0,
                             help="Filter out players with low time-on-pitch coverage")

    st.divider()
    st.caption("Hackathon 2026 · Slalom")

# ── Filter ────────────────────────────────────────────────────────────────────
fdf = df.copy()
if selected_match != "All":
    fdf = fdf[fdf["match_id"] == selected_match]
if selected_phase != "All":
    fdf = fdf[fdf["phase_label"] == selected_phase]
if selected_positions:
    fdf = fdf[fdf["position"].isin(selected_positions)]
if "coverage_pct" in fdf.columns:
    fdf = fdf[fdf["coverage_pct"] >= min_coverage / 100]

# Only rows with real AWI data
fdf = fdf[fdf["awi_per_minute"] > 0].copy()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["👤  Player Profile", "📊  Match Overview", "🏆  Leaderboard"])


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Player Profile
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if _empty(fdf):
        st.stop()

    # Player selector
    player_labels = sorted(
        fdf.apply(lambda r: f"{r['name']}  (#{int(r['jersey'])})", axis=1).unique()
    )
    sel_str = st.selectbox("Select Player", player_labels, key="player_select")
    sel_jersey = int(sel_str.split("#")[1].rstrip(")").strip())
    sel_name   = sel_str.split("  (#")[0]

    mask = (fdf["jersey"] == sel_jersey) & (fdf["name"] == sel_name)
    player_rows = fdf[mask]
    if player_rows.empty:
        st.warning("Player not found in filtered data.")
        st.stop()
    p = player_rows.iloc[0]

    # ── Identity bar ─────────────────────────────────────────────────────────
    st.markdown(
        f"<h2 style='margin:0.2rem 0 0.1rem'>{p['name']}"
        f"<span style='color:{TEXT_MUTED};font-size:1rem;font-weight:400'>"
        f"  &nbsp;#{int(p['jersey'])} · {p['position']} · {p['match_id']} · {p['phase_label']}"
        f"</span></h2>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # ── KPI row ───────────────────────────────────────────────────────────────
    awi_val  = float(p.get("awi_per_minute", 0) or 0)
    pqi_val  = float(p.get("mean_pqi", 0) or 0)
    cov_val  = float(p.get("coverage_pct", 0) or 0)
    scans    = int(p.get("scan_count", 0) or 0)
    minutes  = float(p.get("total_minutes", 0) or 0)

    # Percentile ranks vs filtered dataset
    awi_pct = int((fdf["awi_per_minute"] < awi_val).mean() * 100)
    pqi_pct = int((fdf["mean_pqi"].dropna() < pqi_val).mean() * 100) if pqi_val > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        kpi("AWI", f"{awi_val:.2f}", f"Top {100-awi_pct}% · scans/min", PRIMARY)
    with k2:
        kpi("Mean PQI", f"{pqi_val:.1f}", f"Top {100-pqi_pct}% · quality score", SECONDARY)
    with k3:
        kpi("Total Scans", f"{scans:,}", f"{minutes:.0f} min on pitch")
    with k4:
        kpi("Coverage", f"{cov_val*100:.0f}%", "time on pitch")
    with k5:
        orient = p.get("orientation_mean")
        kpi("Orientation", f"{orient:.1f}°" if pd.notna(orient) else "—",
            "avg body angle during press")

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # ── Gauges ────────────────────────────────────────────────────────────────
    section("PERFORMANCE GAUGES")
    g1, g2 = st.columns(2)

    awi_max = max(fdf["awi_per_minute"].max() * 1.15, 1)

    def _gauge(value, title, color, axis_max, suffix=""):
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            delta={
                "reference": fdf["awi_per_minute"].median() if "AWI" in title else fdf["mean_pqi"].median(),
                "valueformat": ".2f",
                "increasing": {"color": "#2ecc71"},
                "decreasing": {"color": "#e74c3c"},
            },
            title={"text": title, "font": {"size": 14, "color": TEXT_MUTED}},
            number={"font": {"size": 44}, "suffix": suffix},
            gauge={
                "axis": {"range": [0, axis_max], "tickwidth": 1, "tickcolor": BORDER},
                "bar":  {"color": color, "thickness": 0.25},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, axis_max * 0.33], "color": "rgba(255,255,255,0.03)"},
                    {"range": [axis_max * 0.33, axis_max * 0.66], "color": "rgba(255,255,255,0.06)"},
                    {"range": [axis_max * 0.66, axis_max], "color": "rgba(255,255,255,0.09)"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 2},
                    "thickness": 0.75,
                    "value": fdf["awi_per_minute"].median() if "AWI" in title else fdf["mean_pqi"].median(),
                },
            },
        ))
        fig.update_layout(
            height=260, margin=dict(t=30, b=0, l=30, r=30),
            template=CHART_THEME, paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    with g1:
        st.plotly_chart(_gauge(round(awi_val, 2), "AWI — Awareness Index (scans/min)",
                               PRIMARY, awi_max), width="stretch")
        st.caption(f"Median in selection: {fdf['awi_per_minute'].median():.2f} scans/min")

    with g2:
        st.plotly_chart(_gauge(round(pqi_val, 1), "PQI — Press Quality Index (0–100)",
                               SECONDARY, 100), width="stretch")
        st.caption(f"Median in selection: {fdf['mean_pqi'].median():.1f}")

    st.divider()

    # ── Scatter: player in context ────────────────────────────────────────────
    section("PLAYER IN CONTEXT — AWI vs PQI")

    fwd   = p.get("scan_forward_pct",   None)
    lat   = p.get("scan_lateral_pct",   None)
    blind = p.get("scan_blindside_pct", None)
    has_scan = fwd is not None and not pd.isna(fwd)

    sc_col, pie_col = (st.columns([3, 2]) if has_scan else (st.columns([1])[0], None))

    with sc_col:
        sd = _scatter_with_tooltips(fdf)
        fig_sc = px.scatter(
            sd, x="awi_per_minute", y="mean_pqi", color="position",
            custom_data=["_tip"],
            opacity=0.5, template=CHART_THEME,
            color_discrete_sequence=PASTEL,
        )
        fig_sc.update_traces(
            marker_size=8,
            hovertemplate="%{customdata[0]}<extra></extra>",
        )
        hl = _scatter_with_tooltips(fdf[mask])
        fig_sc.add_trace(go.Scatter(
            x=hl["awi_per_minute"], y=hl["mean_pqi"],
            mode="markers",
            marker=dict(size=20, color=GOLD, symbol="star",
                        line=dict(color="white", width=1.5)),
            name=sel_name,
            customdata=hl[["_tip"]].values,
            hovertemplate="%{customdata[0]}<extra></extra>",
            showlegend=True,
        ))
        # Median crosshairs
        fig_sc.add_hline(y=fdf["mean_pqi"].median(), line_dash="dot",
                         line_color=TEXT_MUTED, line_width=1,
                         annotation_text="PQI median", annotation_font_color=TEXT_MUTED,
                         annotation_position="bottom right")
        fig_sc.add_vline(x=fdf["awi_per_minute"].median(), line_dash="dot",
                         line_color=TEXT_MUTED, line_width=1,
                         annotation_text="AWI median", annotation_font_color=TEXT_MUTED,
                         annotation_position="top right")
        fig_sc.update_layout(
            height=400, margin=dict(t=20, b=50, l=50, r=20),
            xaxis_title="AWI (scans/min)", yaxis_title="Mean PQI",
            legend=dict(orientation="v", x=1.01, y=1, font_size=11),
        )
        st.plotly_chart(fig_sc, width="stretch")

    if has_scan and pie_col is not None:
        with pie_col:
            fig_pie = go.Figure(go.Pie(
                labels=["Forward", "Lateral", "Blind-side"],
                values=[fwd, lat, blind],
                hole=0.4,
                marker_colors=[PRIMARY, SECONDARY, "#A8DADC"],
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>",
            ))
            fig_pie.update_layout(
                title=dict(text="Scan Direction", font_size=13, x=0.5),
                height=400, margin=dict(t=50, b=20, l=10, r=10),
                template=CHART_THEME, paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.05, font_size=11),
            )
            st.plotly_chart(fig_pie, width="stretch")

    # ── PQI component breakdown ───────────────────────────────────────────────
    comp_cols = {
        "orientation_mean": "Body Orientation",
        "stance_mean":      "Stance Quality",
        "proximity_mean":   "Proximity Score",
    }
    avail_comp = {k: v for k, v in comp_cols.items() if k in fdf.columns and pd.notna(p.get(k))}

    if avail_comp:
        st.divider()
        section("PQI COMPONENT BREAKDOWN")
        comp_cols_ui = st.columns(len(avail_comp))
        for i, (col_key, label) in enumerate(avail_comp.items()):
            val = float(p[col_key])
            med = fdf[col_key].median()
            pct = int((fdf[col_key].dropna() < val).mean() * 100)
            with comp_cols_ui[i]:
                kpi(label, f"{val:.1f}", f"Median {med:.1f} · Top {100-pct}%")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Match Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    if _empty(fdf):
        st.stop()

    # ── Summary KPIs ─────────────────────────────────────────────────────────
    section("MATCH SUMMARY")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        kpi("Players Tracked", str(fdf["name"].nunique()), f"{fdf['match_id'].nunique()} match(es)")
    with s2:
        kpi("Avg AWI", f"{fdf['awi_per_minute'].mean():.2f}", "scans/min across selection")
    with s3:
        kpi("Avg PQI", f"{fdf['mean_pqi'].mean():.1f}", "mean quality score")
    with s4:
        top_player = fdf.loc[fdf["awi_per_minute"].idxmax(), "name"]
        kpi("Top AWI Player", top_player.split()[-1],
            f"{fdf['awi_per_minute'].max():.2f} scans/min")

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── Top 10 bars ───────────────────────────────────────────────────────────
    section("TOP PERFORMERS")
    b1, b2 = st.columns(2)

    with b1:
        top_awi = fdf.nlargest(10, "awi_per_minute")[["name", "awi_per_minute", "position"]].copy()
        top_awi["label"] = top_awi["name"].apply(lambda n: n.split()[-1])
        fig_ab = px.bar(
            top_awi, x="awi_per_minute", y="label", orientation="h",
            color="position", template=CHART_THEME,
            color_discrete_sequence=PASTEL,
            labels={"awi_per_minute": "AWI (scans/min)", "label": ""},
            title="Top 10 by AWI",
        )
        fig_ab.update_traces(
            hovertemplate="<b>%{y}</b><br>AWI: %{x:.2f} scans/min<extra></extra>",
        )
        fig_ab.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=360, margin=dict(t=40, b=40, l=10, r=10),
            legend=dict(orientation="h", y=-0.25, font_size=10),
        )
        st.plotly_chart(fig_ab, width="stretch")

    with b2:
        top_pqi = (fdf.dropna(subset=["mean_pqi"])
                   .nlargest(10, "mean_pqi")[["name", "mean_pqi", "position"]].copy())
        top_pqi["label"] = top_pqi["name"].apply(lambda n: n.split()[-1])
        fig_pb = px.bar(
            top_pqi, x="mean_pqi", y="label", orientation="h",
            color="position", template=CHART_THEME,
            color_discrete_sequence=PASTEL,
            labels={"mean_pqi": "Mean PQI", "label": ""},
            title="Top 10 by PQI",
        )
        fig_pb.update_traces(
            hovertemplate="<b>%{y}</b><br>PQI: %{x:.1f}<extra></extra>",
        )
        fig_pb.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=360, margin=dict(t=40, b=40, l=10, r=10),
            legend=dict(orientation="h", y=-0.25, font_size=10),
        )
        st.plotly_chart(fig_pb, width="stretch")

    st.divider()

    # ── AWI vs PQI scatter ────────────────────────────────────────────────────
    section("AWI vs PQI — ALL PLAYERS")
    sd2 = _scatter_with_tooltips(fdf)
    fig_ms = px.scatter(
        sd2, x="awi_per_minute", y="mean_pqi", color="position",
        custom_data=["_tip"], opacity=0.7, template=CHART_THEME,
        color_discrete_sequence=PASTEL,
        size_max=12,
    )
    fig_ms.update_traces(
        marker_size=9,
        hovertemplate="%{customdata[0]}<extra></extra>",
    )
    fig_ms.add_hline(y=fdf["mean_pqi"].median(), line_dash="dot",
                     line_color=TEXT_MUTED, line_width=1,
                     annotation_text="PQI median", annotation_font_color=TEXT_MUTED,
                     annotation_position="bottom right")
    fig_ms.add_vline(x=fdf["awi_per_minute"].median(), line_dash="dot",
                     line_color=TEXT_MUTED, line_width=1,
                     annotation_text="AWI median", annotation_font_color=TEXT_MUTED,
                     annotation_position="top right")
    fig_ms.update_layout(
        height=440, margin=dict(t=20, b=50, l=50, r=20),
        xaxis_title="AWI (scans/min)", yaxis_title="Mean PQI",
        legend=dict(orientation="h", y=-0.15, font_size=11),
    )
    st.plotly_chart(fig_ms, width="stretch")

    st.divider()

    # ── Half-time delta ───────────────────────────────────────────────────────
    section("HALF-TIME DELTA  (2nd − 1st Half)")
    h1 = (fdf[fdf["phase_label"] == "1st half"][["name", "awi_per_minute", "mean_pqi"]]
          .rename(columns={"awi_per_minute": "awi_h1", "mean_pqi": "pqi_h1"}))
    h2 = (fdf[fdf["phase_label"] == "2nd half"][["name", "awi_per_minute", "mean_pqi"]]
          .rename(columns={"awi_per_minute": "awi_h2", "mean_pqi": "pqi_h2"}))
    halves = h1.merge(h2, on="name", how="inner")

    if not halves.empty:
        halves["awi_delta"] = halves["awi_h2"] - halves["awi_h1"]
        halves["pqi_delta"] = halves["pqi_h2"] - halves["pqi_h1"]
        halves["short_name"] = halves["name"].apply(lambda n: n.split()[-1])
        top_h = halves.nlargest(15, "awi_h1")

        fig_d = go.Figure()
        fig_d.add_trace(go.Bar(
            name="AWI Δ (scans/min)", x=top_h["short_name"], y=top_h["awi_delta"],
            marker_color=PRIMARY, opacity=0.85,
            hovertemplate="<b>%{x}</b><br>AWI Δ: %{y:.2f} scans/min<extra></extra>",
        ))
        fig_d.add_trace(go.Bar(
            name="PQI Δ", x=top_h["short_name"], y=top_h["pqi_delta"],
            marker_color=SECONDARY, opacity=0.85,
            hovertemplate="<b>%{x}</b><br>PQI Δ: %{y:.1f}<extra></extra>",
        ))
        fig_d.add_hline(y=0, line_color=BORDER, line_width=1)
        fig_d.update_layout(
            barmode="group", template=CHART_THEME,
            xaxis_tickangle=-35,
            height=400, margin=dict(t=20, b=90, l=50, r=20),
            legend=dict(orientation="h", y=1.08, font_size=11),
            yaxis_title="Delta (2nd − 1st half)",
        )
        st.plotly_chart(fig_d, width="stretch")
    else:
        st.info("Select 'All' phases to see the half-time delta comparison.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – Leaderboard
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    if _empty(fdf):
        st.stop()

    # ── Sortable table ────────────────────────────────────────────────────────
    section("FULL LEADERBOARD")

    display_cols = ["jersey", "name", "position", "match_id", "phase_label",
                    "awi_per_minute", "mean_pqi", "coverage_pct",
                    "orientation_mean", "stance_mean", "proximity_mean"]
    avail_cols = [c for c in display_cols if c in fdf.columns]

    table_df = fdf[avail_cols].sort_values("awi_per_minute", ascending=False).reset_index(drop=True)
    table_df.index += 1  # 1-based rank

    if "coverage_pct" in table_df.columns:
        table_df["coverage_pct"] = (table_df["coverage_pct"] * 100).round(1)

    st.dataframe(
        table_df,
        width="stretch",
        height=min(45 + len(table_df) * 35, 520),
        column_config={
            "jersey":           st.column_config.NumberColumn("#",              format="%d"),
            "name":             st.column_config.TextColumn("Player"),
            "position":         st.column_config.TextColumn("Pos"),
            "match_id":         st.column_config.TextColumn("Match"),
            "phase_label":      st.column_config.TextColumn("Phase"),
            "awi_per_minute":   st.column_config.NumberColumn("AWI (scans/min)", format="%.2f"),
            "mean_pqi":         st.column_config.NumberColumn("Mean PQI",        format="%.1f"),
            "coverage_pct":     st.column_config.NumberColumn("Coverage %",      format="%.1f"),
            "orientation_mean": st.column_config.NumberColumn("Orientation",     format="%.1f"),
            "stance_mean":      st.column_config.NumberColumn("Stance",          format="%.1f"),
            "proximity_mean":   st.column_config.NumberColumn("Proximity",       format="%.1f"),
        },
    )

    st.divider()

    # ── Position analytics ────────────────────────────────────────────────────
    section("POSITION ANALYTICS")
    pa1, pa2 = st.columns(2)

    hm_cols = ["awi_per_minute", "mean_pqi", "orientation_mean", "stance_mean", "proximity_mean"]
    avail_hm = [c for c in hm_cols if c in fdf.columns]
    pos_avg = (fdf.groupby("position")[avail_hm].mean()
               .dropna(how="all").reset_index()
               .sort_values("awi_per_minute", ascending=False))

    hm_labels = {
        "awi_per_minute":   "AWI",
        "mean_pqi":         "PQI",
        "orientation_mean": "Orientation",
        "stance_mean":      "Stance",
        "proximity_mean":   "Proximity",
    }

    with pa1:
        if not pos_avg.empty and avail_hm:
            hm_data = pos_avg.set_index("position")[[c for c in avail_hm if c in pos_avg.columns]]
            hm_data.columns = [hm_labels.get(c, c) for c in hm_data.columns]
            fig_hm = px.imshow(
                hm_data,
                text_auto=".1f",
                color_continuous_scale="Blues",
                title="Position Averages",
                aspect="auto",
                template=CHART_THEME,
            )
            fig_hm.update_layout(
                height=max(280, len(pos_avg) * 40 + 80),
                margin=dict(t=40, b=50, l=10, r=10),
                xaxis_tickangle=-20,
                coloraxis_showscale=False,
            )
            fig_hm.update_traces(
                hovertemplate="<b>%{y}</b> · %{x}<br>Value: %{z:.1f}<extra></extra>",
            )
            st.plotly_chart(fig_hm, width="stretch")

    with pa2:
        fig_box = px.box(
            fdf, x="position", y="awi_per_minute",
            color="position", template=CHART_THEME,
            color_discrete_sequence=PASTEL,
            title="AWI Distribution by Position",
            labels={"awi_per_minute": "AWI (scans/min)", "position": ""},
        )
        fig_box.update_traces(
            hovertemplate="<b>%{x}</b><br>AWI: %{y:.2f}<extra></extra>",
        )
        fig_box.update_layout(
            showlegend=False, xaxis_tickangle=-35,
            height=max(280, len(pos_avg) * 40 + 80),
            margin=dict(t=40, b=80, l=50, r=10),
        )
        st.plotly_chart(fig_box, width="stretch")

    # ── PQI distribution ──────────────────────────────────────────────────────
    if "mean_pqi" in fdf.columns:
        st.divider()
        section("PQI DISTRIBUTION BY POSITION")
        fig_pq = px.box(
            fdf.dropna(subset=["mean_pqi"]),
            x="position", y="mean_pqi",
            color="position", template=CHART_THEME,
            color_discrete_sequence=PASTEL,
            labels={"mean_pqi": "Mean PQI", "position": ""},
        )
        fig_pq.update_traces(
            hovertemplate="<b>%{x}</b><br>PQI: %{y:.1f}<extra></extra>",
        )
        fig_pq.update_layout(
            showlegend=False, xaxis_tickangle=-35,
            height=380, margin=dict(t=20, b=80, l=50, r=10),
        )
        st.plotly_chart(fig_pq, width="stretch")
