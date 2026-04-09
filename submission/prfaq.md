# PRFAQ: Football Body Intelligence Platform

**AWS World Sports Innovation Cup 2026 · Challenge 2**
Team: itzmore

---

## Overview

### Football Body Intelligence Platform: AWI + PQI

The Football Body Intelligence Platform transforms TRACAB TF15 3D skeleton data from five Bundesliga matches into two complementary player intelligence metrics — AWI (Awareness Index) and PQI (Pressure Quality Index) — and surfaces them through an interactive Streamlit dashboard and AI-generated scouting narratives via Amazon Bedrock.

**AWI (Awareness Index)** measures how actively a player scans their environment during a match. It is defined as the rate of discrete head-scanning events per minute: a rapid head direction change of ≥45° within a 0.5-second window (25 frames at 50 fps), computed from the 3D position of the nose, neck, and ear keypoints. Unlike 2D tracking, which records where players move, TF15 skeleton data records where players *look* — enabling AWI for the first time at matchday scale.

**PQI (Pressure Quality Index)** measures the quality of a player's body mechanics during pressing actions. It is a composite score [0, 100] combining three sub-scores: body orientation toward the ball carrier (40% weight), knee-flexion stance quality relative to the biomechanical optimum of 130° (30% weight), and proximity to the ball carrier (30% weight). PQI is computed only during genuine press frames — windows where a player's pelvis is within 5 metres of the ball carrier for at least 10 consecutive frames (0.2 s at 50 fps).

The most important finding from the combined analysis: **AWI and PQI are statistically independent** (Pearson r = −0.11, p = 0.12). Knowing a player's scan rate tells you almost nothing about their pressing mechanics. This means Body Intelligence has two genuinely orthogonal dimensions — and a player must be measured on both to be fully understood.

Computed across 5 Bundesliga matches and 400 player-phase observations, the platform identifies 10 unique players who score above the 75th percentile on both dimensions simultaneously — the elite quadrant. Among them: Oscar Winther Höjlund (DMZ, 26.90 scans/min, PQI 63.9), Joshua Kimmich (DMR, 21.77 scans/min, PQI 64.7), and Luka Vušković (IVZ, 18.33 scans/min, PQI 63.7) — a centre-back whose scanning profile matches a defensive midfielder.

---

## Dashboard

The interactive Streamlit dashboard (`dashboard/app.py`) loads the merged AWI + PQI dataset and provides three tabs for exploring body intelligence data at different levels of granularity.

![AWI vs PQI Scatter — all players, coloured by position](figures/fig1_awi_vs_pqi_scatter.png)

*Figure 1: AWI vs PQI scatter plot for all player-phases across 5 Bundesliga matches, coloured by position. The near-zero correlation (r = −0.11) confirms that scanning awareness and pressing quality are independent skills — both metrics are needed to characterise a player fully. The elite quadrant (top-right) contains 10 unique players across 15 player-phases.*

The dashboard provides three views:

- **Player Profile** — AWI gauge, PQI gauge, scatter with the selected player highlighted, and sub-score breakdown
- **Match Overview** — top-10 AWI and PQI bar charts, full-match scatter, and 1st-half vs 2nd-half delta comparison
- **League Leaderboard** — sortable table across all five matches, position-average heatmap, and distribution charts by position

To launch: `bash dashboard/run_dashboard.sh` (runs on port 8501).

---

## Frequently Asked Questions

### What exactly is a "scan event"?

A scan event is a discrete head rotation of ≥45° completed within a 0.5-second window (25 frames at 50 fps), detected from the smoothed head yaw time series. "Discrete" means one sustained rotation counts as one event, not one per frame. A player turning their head 90° over 0.3 seconds registers as one scan, not 15. This matches how coaches and sports scientists define the term: a deliberate shoulder/head check, not a continuous slow turn.

### Why 45°? The literature uses 30°.

The 30° threshold in sports-science literature is measured from direct video, where head angles are observed at full 3D resolution. The TF15 XY-plane projection compresses apparent rotation: a 3D head turn of 45° projects to approximately 30° in the XY plane depending on pitch-facing direction. The 45° threshold was empirically tuned on Kimmich (FCB-HSV, 1st half), where the resulting AWI of 21.77 scans/min matches his documented scanning frequency from hand-coded video analysis in the coaching literature.

### Why is AWI a "stable trait"? Could it just reflect match context?

Cross-half Pearson correlation R = 0.854 (p < 0.001, n = 69 active player-phases). Players who rank high in AWI in the first half rank high in the second half, independently of match context. Kimmich's AWI across two matches where he played full games: 21.77 / 21.15 (FCB-HSV) and 23.38 / 11.29 (FCU-FCB). The second-match second-half drop to 11.29 is itself informative — a 52% decline that no GPS or positional metric would detect, potentially indicating fatigue or tactical instruction.

### What is PQI and how is it computed?

PQI (Pressure Quality Index) is a composite score [0, 100] that measures the quality of a player's body mechanics during pressing actions. It is computed only during press frames — windows where a player's pelvis is within 5 metres of the ball carrier for at least 10 consecutive frames (0.2 s at 50 fps).

PQI combines three sub-scores:

- **Orientation score** (40% weight): how well the player's body faces the ball carrier. Score = `max(0, 100 − (angle_to_target / 90) × 100)`. A player facing the ball carrier directly scores 100; facing 90° away scores 0.
- **Stance score** (30% weight): how close the player's knee flexion is to the biomechanical optimum of 130°. Score = `100 × exp(−0.5 × ((knee_flexion − 130) / 25)²)`. A player at exactly 130° knee flexion scores 100.
- **Proximity score** (30% weight): how close the player is to the ball carrier. Score = `max(0, 100 × (1 − distance_m / 5.0))`. A player at 0 m scores 100; at 5 m or beyond scores 0.

The composite PQI = `0.40 × orientation + 0.30 × stance + 0.30 × proximity`.

### Why do goalkeepers lead PQI?

Goalkeepers (TW) have the highest mean PQI at 66.6, driven by exceptional proximity scores (92+). This is structurally expected: goalkeepers frequently close down ball carriers in their penalty area at very short range, which maximises the proximity sub-score. Their orientation scores are also high (70–75) because they face the ball carrier directly when closing down. This is a valid finding — goalkeepers are genuinely excellent pressers in their zone — but for outfield pressing comparisons, position-specific benchmarking is recommended. Among outfield players, DMZ (62.9) and DMR (62.1) lead PQI, confirming that defensive midfielders dominate both cognitive and physical pressing dimensions.

### What does the AWI vs PQI independence finding mean?

The near-zero correlation (r = −0.11, p = 0.12) between AWI and PQI is the most analytically significant result in the dataset. It means:

1. **They measure genuinely different things.** A player who scans frequently does not automatically press well, and vice versa. The two skills are independently distributed across the population.
2. **Both metrics are necessary.** Using only AWI or only PQI gives an incomplete picture. A player in the elite quadrant (top 25% on both) is rare — only 10 unique players across 400 observations qualify.
3. **Development pathways are separable.** A coach can target AWI improvement (cognitive training, scanning drills) independently of PQI improvement (pressing mechanics, stance coaching) because the two dimensions don't trade off against each other.

### What about goalkeepers and substitutes?

Goalkeepers are included in the data. Their AWI is low (mean 3.5 scans/min) because they primarily track the ball rather than scanning for opponents — a valid positional difference, not a data quality issue. Their PQI is high for structural reasons explained above. For cross-position leaderboards, filtering by position is recommended. Players with coverage < 50% (less than ~25 minutes of skeleton data in a phase) are flagged in the output CSV; late substitutes naturally fall below this threshold.

### Can AWI and PQI be used in real time?

Yes. AWI is computed as a rolling count over a sliding window; the pipeline already produces per-minute bins. Real-time AWI requires the same 50 fps skeleton feed and can be computed with approximately 12 seconds of lag (25-frame detection window plus smoothing). PQI can similarly be computed in near-real-time once press frame detection is applied to the live stream. Both metrics are suitable for broadcast overlays with sub-15-second latency.

### How does AWI compare to scanning rates in football research?

**Contextual benchmark from video-based research:** Jordet et al. (2020) measured scanning frequency in the 10 seconds before a player receives the ball in the English Premier League, finding a mean of approximately 0.44 scans/s (26 scans/min) across all positions during that pre-reception window. Elite central midfielders scored substantially higher. Forwards consistently scanned the least. This positional gradient matches the AWI ranking in our data.

| Measure | Rate | Method |
|---|---|---|
| AWI: DMZ (this study) | 15.6 scans/min | Full-phase mean, TF15 3D skeleton, 50 fps |
| AWI: DMR (this study) | 10.4 scans/min | Full-phase mean, TF15 3D skeleton, 50 fps |
| AWI: STZ (this study) | 6.2 scans/min | Full-phase mean, TF15 3D skeleton, 50 fps |
| Pre-reception scanning, EPL (Jordet et al., 2020) | ~26 scans/min | Video-coded, last 10 s before receiving |

Note: AWI measures full-phase rate continuously across ~50 minutes of play; video-based studies typically measure short pre-reception bursts. These are complementary but not directly interchangeable metrics.

### How does this differ from existing cognitive metrics in football?

Existing approaches (SportVU off-ball tracking, GPS-based positioning metrics) are proxies: they measure movement patterns correlated with awareness. AWI is a direct measure: it reads *where the head is pointing* at 50 fps. PQI similarly reads actual joint angles and positions rather than inferring pressing quality from GPS proximity alone. No coaching staff survey, no subjective coding, no GPS proxy. These are the first matchday-grade body intelligence metrics derived from the actual anatomical signal.

### What do the AI-generated scouting reports look like?

The platform uses Amazon Bedrock (Amazon Nova Lite, `eu.amazon.nova-lite-v1:0`) to convert raw AWI and PQI numbers into natural-language scouting narratives. Below is the actual generated narrative for the top-ranked player:

> **Oscar Winther Höjlund | DMZ | SGE-FCB | 1st Half**
>
> Oscar Winther Höjlund's AWI of 26.9 scans per minute ranks #1 of 400 players — 72% above the league mean of 15.6. This exceptional cognitive awareness allows him to quickly assess and react to dynamic game situations, making him a formidable presence in defensive midfield. His superior scanning ability indicates a proactive approach in identifying and intercepting threats.
>
> His PQI of 63.9 reflects a generally effective pressing strategy, particularly strong in proximity (96.0), indicating he is adept at closing down opponents efficiently. However, his orientation (59.0) and stance (38.1) scores suggest room for improvement in positioning and body posture during pressures. Focusing on these aspects can help Höjlund become more disruptive in the opponent's build-up play.

Full narratives for all top-10 players are saved to `results/narratives.csv` and generated via `notebooks/bedrock_reports.ipynb`.

### How are the AI narratives generated?

The `BedrockClient` (`src/bedrock_client.py`) constructs a structured prompt containing the player's name, position, match label, phase label, AWI score, league rank, position average, PQI mean, and all three PQI sub-scores. The prompt is sent to `eu.amazon.nova-lite-v1:0` in region `eu-central-1`. The client retries up to 3 times with exponential backoff on throttling errors. The model selection uses Amazon Nova rather than Claude Sonnet because the hackathon account's Service Control Policy restricts cross-region routing to `eu-south-1`, which Claude Sonnet 4.x inference profiles require. Nova Lite routes within `eu-central-1` without restriction.

### What are the limitations?

1. **XY-plane projection**: we can't distinguish a head turn aimed at the goalkeeper from one aimed at a nearby defender; only the angular magnitude is captured, not gaze direction.
2. **Occlusion artifacts**: if the skeleton tracker loses joints (nose/neck), we fall back to the ear vector; if that fails, the frame is excluded. High-movement phases (headers, sprints) have higher NA rates.
3. **Single-player threshold validation**: Kimmich is our strongest anchor; systematic validation against hand-coded video for additional players would strengthen confidence in the 45° threshold.
4. **No ball context for AWI**: AWI counts all head turns equally, whether or not the player is in possession. Pre-pass AWI partially addresses this.
5. **PQI press frame threshold**: the 10-consecutive-frame minimum (0.2 s) filters out very brief proximity events. Players who press in short bursts may have fewer press frames than their actual pressing volume suggests.
6. **PQI stance sub-score**: the 130° knee-flexion optimum is derived from biomechanics literature for athletic pressing stance. Individual variation in optimal knee angle is not accounted for.

### What comes next?

- **Pre-pass AWI**: complete the 5-match enrichment and correlate with pass outcome (progressive pass rate, turnover rate) — the cognitive → decision quality link
- **Seasonal trend**: compute AWI and PQI across a full Bundesliga season to track development arcs and fatigue effects
- **Live overlay**: integrate with DFL's matchday data product for broadcast visualization of both AWI and PQI in real time
- **Quadrant-based scouting tool**: expose the Body Intelligence quadrant chart as an interactive scouting filter in the dashboard
- **Extended Bedrock narratives**: generate reports for all players (not just top-10) and expose them through the dashboard's Player Profile tab
