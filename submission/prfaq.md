# PRFAQ: Football Body Intelligence Platform

**AWS World Sports Innovation Cup 2026 · Challenge 2**
Team: itzmore

---

## Overview

### Football Body Intelligence Platform: AWI + PQI

The Football Body Intelligence Platform transforms TRACAB TF15 3D skeleton data from five Bundesliga matches into two complementary player intelligence metrics — AWI (Awareness Index) and PQI (Pressure Quality Index) — surfaced through an interactive Streamlit dashboard and AI-generated scouting narratives via Amazon Bedrock.

**AWI (Awareness Index)** measures how actively a player scans their environment during a match. It counts discrete head-scanning events per minute: a rapid head-direction change of ≥45° within a 0.5-second window (25 frames at 50 fps), computed from the 3D positions of the nose, neck, and ear keypoints. Unlike 2D tracking, which records where players move, TF15 skeleton data records where players *look* — making AWI possible at matchday scale for the first time. AWI spikes +57% in the 5 seconds before a player releases the ball, confirming that the metric captures pre-decision cognitive load rather than incidental head movement.

**PQI (Pressure Quality Index)** measures the quality of a player's body mechanics during pressing actions. It is a composite score [0, 100] combining three sub-scores: body orientation toward the ball carrier (40% weight), knee-flexion stance quality relative to the biomechanical optimum of 130° (30% weight), and proximity to the ball carrier (30% weight). PQI is computed only during genuine press frames — windows where a player's pelvis is within 5 metres of the ball carrier for at least 10 consecutive frames (0.2 s at 50 fps).

**The central analytical finding: AWI and PQI are statistically independent** (Pearson r = −0.11, p = 0.12). Knowing a player's scan rate tells you almost nothing about their pressing mechanics. Body Intelligence has two genuinely orthogonal dimensions — and a player must be measured on both to be fully understood.

Computed across 5 Bundesliga matches and 400 player-phase observations, the platform identifies 10 unique players who score above the 75th percentile on both dimensions simultaneously — the elite quadrant — including Oscar Winther Höjlund (DMZ, 26.90 scans/min, PQI 63.9), Joshua Kimmich (DMR, 21.77 scans/min, PQI 64.7), and Luka Vušković (IVZ, 18.33 scans/min, PQI 63.7) — a centre-back whose scanning profile matches a defensive midfielder.

---

## Dashboard

The interactive Streamlit dashboard (`dashboard/app.py`) loads the merged AWI + PQI dataset and provides three tabs for exploring body intelligence data at different levels of granularity.

![AWI vs PQI Scatter — all players, coloured by position](figures/fig1_awi_vs_pqi_scatter.png)

*Figure 1: AWI vs PQI scatter plot for all player-phases across 5 Bundesliga matches, coloured by position. The near-zero correlation (r = −0.11) confirms that scanning awareness and pressing quality are independent skills — both metrics are needed to characterise a player fully. The elite quadrant (top-right) contains 10 unique players across 15 player-phases.*

The dashboard provides three views:

- **Player Profile** — AWI/PQI gauges vs selection median, 5-metric KPI row with percentile ranks and inline tooltips, scatter in context (colored by role group), PQI radar vs role average, PQI component breakdown (Orientation/Stance/Proximity with explanations), per-phase trend cards
- **Match Overview** — summary KPIs, quadrant scatter with 4-quadrant classification and elite player labels, role lollipop chart, PQI decomposition stacked bar by role, half-time fatigue bar + 1st vs 2nd half AWI scatter, team AWI comparison (color per club)
- **Leaderboard** — sortable table with DFL position codes, role averages heatmap, AWI bar chart (mean ± std per role), PQI box distribution by role

Sidebar: Match, Phase, Position (DFL codes), Min Coverage % (default 50%). Collapsible **Metric Definitions** expander explains AWI, PQI, and all three sub-scores. **Position Code Reference** expander maps all DFL codes to English names.

To launch: `bash dashboard/run_dashboard.sh` (runs on port 8501).

---

## Frequently Asked Questions

### What exactly is a "scan event"?

A scan event is a discrete head rotation of ≥45° completed within a 0.5-second window (25 frames at 50 fps), detected from the smoothed head yaw time series. "Discrete" means one sustained rotation counts as one event, not one per frame: a player turning their head 90° over 0.3 seconds registers as one scan, not 15. This matches how coaches and sports scientists define the term — a deliberate shoulder-check, not a continuous slow turn.

### Why 45°? The literature uses 30°.

The 30° threshold in sports-science literature is measured from direct video, where head angles are observed at full 3D resolution. The TF15 XY-plane projection compresses apparent rotation: a 3D head turn of 45° projects to approximately 30° in the XY plane depending on pitch-facing direction. The 45° threshold was empirically tuned on Kimmich (FCB-HSV, 1st half), where the resulting AWI of 21.77 scans/min matches his documented scanning frequency from hand-coded video analysis in the coaching literature.

### Why is AWI a "stable trait"? Could it just reflect match context?

Cross-half Pearson correlation R = 0.854 (p < 0.001, n = 69 active player-phases). Players who rank high in AWI in the first half rank high in the second half, independently of match context. Kimmich across two full matches: 21.77 / 21.15 (FCB-HSV) and 23.38 / 11.29 (FCU-FCB). The second-match second-half drop to 11.29 is itself informative — a 52% decline that no GPS or positional metric would detect, potentially indicating late-match fatigue or a tactical instruction to hold position.

### What does the pre-pass AWI finding mean?

In the 5 seconds before a player releases the ball, their AWI is +57% above their full-phase baseline. This confirms the metric captures the pre-decision cognitive window that coaches call "playing with your head up." It also validates that AWI measures intentional scanning — players actively increase their head-rotation rate precisely when they need to assess the field before acting.

### What is PQI and how is it computed?

PQI (Pressure Quality Index) is a composite score [0, 100] measuring the quality of a player's body mechanics during pressing actions. It is computed only during press frames — windows where a player's pelvis is within 5 metres of the ball carrier for at least 10 consecutive frames (0.2 s at 50 fps).

PQI combines three sub-scores:

- **Orientation score** (40% weight): how well the player's body faces the ball carrier. Score = `max(0, 100 − (angle_to_target / 90) × 100)`. A player facing the ball carrier directly scores 100; facing 90° away scores 0.
- **Stance score** (30% weight): how close the player's knee flexion is to the biomechanical optimum of 130°. Score = `100 × exp(−0.5 × ((knee_flexion − 130) / 25)²)`. A player at exactly 130° knee flexion scores 100.
- **Proximity score** (30% weight): how close the player is to the ball carrier. Score = `max(0, 100 × (1 − distance_m / 5.0))`. A player at 0 m scores 100; at 5 m or beyond scores 0.

The composite: `PQI = 0.40 × orientation + 0.30 × stance + 0.30 × proximity`.

### Why do goalkeepers lead PQI?

Goalkeepers (TW) post the highest mean PQI at 66.6, driven primarily by their **orientation sub-score** (70.4 vs 54.7 outfield average). When closing down in their penalty area, goalkeepers approach the ball carrier square-on by necessity — they are defending their goal and must stay central — which maximises the orientation component. Their proximity sub-scores (91.5) are actually slightly below the outfield average (95.2), so proximity is not the driver. This is a structural role effect: the PQI formula rewards direct body-facing, and goalkeepers are structurally constrained to face the ball carrier directly. Position-specific benchmarking is recommended for outfield comparisons. Among outfield players, DMZ (62.9) and DMR (62.1) lead, confirming that defensive midfielders dominate both cognitive and physical pressing dimensions.

### What does the AWI vs PQI independence finding mean in practice?

The near-zero correlation (r = −0.11, p = 0.12) is the most analytically significant result in the dataset. It means:

1. **They measure genuinely different things.** A player who scans frequently does not automatically press well, and vice versa. The two skills are independently distributed across the population.
2. **Both metrics are necessary.** Using only AWI or only PQI gives an incomplete picture. A player in the elite quadrant (top 25% on both) is rare — only 10 unique players across 400 observations qualify.
3. **Development pathways are separable.** AWI can be targeted through scanning drills and cognitive training independently of PQI, which responds to pressing mechanics coaching. The two dimensions don't trade off against each other.

### What about goalkeepers and substitutes?

Goalkeepers are included in the data. Their AWI is low (mean 3.5 scans/min) because they primarily track the ball rather than scanning for opponents — a valid positional difference, not a data quality issue. Their PQI is high for structural reasons explained above. Players with coverage < 50% (less than ~25 minutes of skeleton data in a phase) are flagged in the output CSV; late substitutes naturally fall below this threshold.

### Can AWI and PQI be used in real time?

Yes. AWI is computed as a rolling count over a sliding window and can be produced with approximately 12 seconds of lag (25-frame detection window plus smoothing buffer). PQI can similarly be computed in near-real-time once press frame detection is applied to the live stream. Both metrics are suitable for broadcast overlays with sub-15-second latency.

### How does AWI compare to scanning rates in football research?

The positional hierarchy in our data matches the academic literature. Jordet et al. (2020) measured pre-reception scanning frequency in the English Premier League, finding elite central midfielders substantially outscanning forwards — the same gradient AWI produces from continuous 3D skeleton data.

| Measure | Rate | Method |
|---|---|---|
| AWI: DMZ (this study) | 15.6 scans/min | Full-phase mean, TF15 3D skeleton, 50 fps |
| AWI: DMR (this study) | 10.4 scans/min | Full-phase mean, TF15 3D skeleton, 50 fps |
| AWI: STZ (this study) | 6.2 scans/min | Full-phase mean, TF15 3D skeleton, 50 fps |
| Pre-reception scanning, EPL (Jordet et al., 2020) | ~26 scans/min | Video-coded, last 10 s before receiving |

Note: AWI measures continuous full-phase rate across ~50 minutes; video-based studies capture short pre-reception bursts. These are complementary, not interchangeable. The +57% pre-pass spike in our data bridges the two approaches.

### How does this differ from existing cognitive metrics in football?

Existing approaches (SportVU off-ball tracking, GPS-based positioning metrics) are proxies: they measure movement patterns correlated with awareness. AWI is a direct measure — it reads where the head is pointing at 50 fps from anatomical joint data. PQI similarly reads actual joint angles and positions rather than inferring pressing quality from GPS proximity alone. No coaching staff survey, no subjective coding, no GPS proxy. These are the first matchday-grade body intelligence metrics derived from the actual anatomical signal.

### What do the AI-generated scouting reports look like?

The platform uses Amazon Bedrock (Amazon Nova Lite, `eu.amazon.nova-lite-v1:0`, deployed in `eu-central-1`) to convert raw AWI and PQI numbers into natural-language scouting narratives. Below is the actual generated narrative for the top-ranked player:

> **Oscar Winther Höjlund | DMZ | SGE-FCB | 1st Half**
>
> Oscar Winther Höjlund's AWI of 26.9 scans per minute ranks #1 of 400 players — 72% above the league mean of 15.6. This exceptional cognitive awareness allows him to quickly assess and react to dynamic game situations, making him a formidable presence in defensive midfield. His superior scanning ability indicates a proactive approach in identifying and intercepting threats.
>
> His PQI of 63.9 reflects a generally effective pressing strategy, particularly strong in proximity (96.0), indicating he is adept at closing down opponents efficiently. However, his orientation (59.0) and stance (38.1) scores suggest room for improvement in positioning and body posture during pressures. Focusing on these aspects can help Höjlund become more disruptive in the opponent's build-up play.

Full narratives for all top-10 players are saved to `results/narratives.csv` and generated via `notebooks/bedrock_reports.ipynb`.

### How are the AI narratives generated?

The `BedrockClient` (`src/bedrock_client.py`) constructs a structured prompt containing the player's name, position, match label, phase label, AWI score, league rank, position average, PQI mean, and all three PQI sub-scores. The prompt is sent to `eu.amazon.nova-lite-v1:0` in `eu-central-1`. The client retries up to 3 times with exponential backoff on throttling errors. To use Claude Sonnet instead, set `BEDROCK_MODEL_ID=eu.anthropic.claude-sonnet-4-6` in `.env` — requires the inference profile to route within your account's permitted regions.

### What are the limitations?

1. **XY-plane projection**: head turns are quantified by angular magnitude, not gaze direction — a scan toward the goalkeeper and a scan toward a nearby defender look identical in the signal.
2. **Occlusion artifacts**: if the skeleton tracker loses nose/neck joints, the pipeline falls back to the ear vector; if that fails, the frame is excluded. High-movement phases (headers, sprints) have higher NA rates.
3. **Single-player threshold calibration**: Kimmich is the primary validation anchor; additional hand-coded video comparisons across positions would further strengthen threshold confidence.
4. **No ball-possession context for AWI**: AWI counts all head turns equally whether or not the player is in possession. Pre-pass AWI partially addresses this — the +57% spike demonstrates the signal is decision-related.
5. **PQI press frame minimum**: the 10-consecutive-frame filter (0.2 s) may under-count players who press in very short bursts.
6. **PQI stance sub-score**: the 130° knee-flexion optimum is from biomechanics literature for athletic pressing stance; individual variation in optimal knee angle is not accounted for.

### What comes next?

- **Pre-pass AWI → pass quality correlation**: correlate with progressive pass rate and turnover rate — the cognitive→decision quality link; data is already computed
- **Full-season AWI/PQI**: track development arcs, fatigue profiles, and load management signals across a complete Bundesliga season
- **Live broadcast overlay**: integrate with DFL's matchday data product for real-time AWI and PQI visualization (<15 s latency, already validated)
- **Body Intelligence scouting filter**: expose the quadrant chart as an interactive filter in the dashboard — search for players above any AWI/PQI threshold across all available data
- **Extended Bedrock narratives**: generate reports for all players (not just top-10) and surface them directly in the dashboard's Player Profile tab
