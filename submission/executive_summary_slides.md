# Executive Summary: Football Body Intelligence Platform
**AWS World Sports Innovation Cup 2026 · Challenge 2**
Team: itzmore

---

## Slide 1: The Problem - What 3D Skeleton Data Unlocks

### Standard tracking tells you *where* players move. TF15 tells you *how* they move and *where they look*.

The Bundesliga's TRACAB TF15 system captures 141 million data points per match - not just player positions, but the full 3D skeleton: every joint, every frame, at 50 fps. Yet today this data powers only automated event detection. The body intelligence it contains - *is this player scanning before receiving? are they pressing with correct biomechanics?* - goes unmeasured.

We built two metrics that change that.

**AWI (Awareness Index)** - cognitive scanning rate: discrete head-rotation events per minute, derived from 3D nose/neck/ear keypoints. Impossible with 2D tracking. Directly measures where a player is *looking*, not just where they are standing.

**PQI (Pressure Quality Index)** - pressing mechanics score [0–100]: body orientation toward the ball carrier (40%), knee-flexion stance quality at the biomechanical optimum of 130° (30%), and proximity (30%). Computed only during genuine press frames - ≥10 consecutive frames within 5 m of the ball carrier.

> *Together, AWI and PQI form a complete Body Intelligence picture. A player can scan brilliantly but press with poor mechanics (high AWI, low PQI), or press perfectly without pre-scanning (low AWI, high PQI). Elite players score high on both - and the data shows exactly who they are.*

---

## Slide 2: Technical Approach

### Pipeline: S3 → Skeleton → Angles → Metrics → Dashboard → AI Narratives

```
TF15 Parquet (S3, ~4 GB/match)
  → pyarrow row-group pushdown        # stream, never full download
  → _extract_angles_vectorized()      # head yaw: nose/neck primary, ear fallback
                                      # body yaw: shoulder primary, hip fallback
  → detect_scans()                    # 11-frame circular smooth → 25-frame delta → ≥45° leading edge
  → compute_awi()                     # scan_count / phase_minutes
  → pqi_calculator.py                 # orientation + stance + proximity, vectorized
  → Streamlit dashboard               # 6 tabs: Player Profile, Match Overview, Leaderboard, Fan View, Broadcast Demo, Benchmark
  → Amazon Bedrock (Nova Lite)        # natural-language scouting narratives
```

**Key engineering decisions:**

| Decision | Why it matters |
|----------|---------------|
| Circular smoothing (sin/cos decomposition) | Standard rolling mean breaks at ±180°; this doesn't |
| Leading-edge counting `(is_scan & ~is_scan.shift(1)).sum()` | 1 sustained rotation = 1 event, not N frames |
| 45° threshold (not 30°) | XY-plane projection compresses 3D angles; fixed constant validated against Kimmich anchor (not fit by a sweep) |
| Press frame filter: ≥10 consecutive frames within 5 m | Excludes incidental proximity; captures genuine pressing intent |
| SageMaker Processing (10 parallel jobs) | 5 matches × 2 metrics in ~15 min vs ~2 hrs local; zero cold-start overhead |

**Scale:** 5 Bundesliga matches · ~40 players × 2 halves = **400 player-phase rows** · 334 unit + property-based tests · production-containerised pipeline (Docker → ECR)

---

## Slide 3: Live Dashboard Demo

### [Live Streamlit Dashboard - all results explored interactively]

This slide is replaced by a live walkthrough of the hosted dashboard (2-3 min), showing:

**Key findings surfaced in the demo:**

- **Höjlund leads AWI at 26.9 scans/min** - a defensive midfielder scanning nearly once every 2 seconds
- **Kimmich's consistency:** 21.77 → 21.15 across a full half - AWI is a stable trait (cross-half r = 0.854)
- **Position hierarchy:** DMZ (15.6) > CB (10.6) > FW (6.2) > GK (3.5) - matches Jordet et al. (2020) EPL gradient
- **Pre-pass spike: +59%** - players ramp scanning in the 5 seconds before releasing the ball
- **Fan View + Broadcast Demo:** broadcast-ready overlays with real-time AWI/PQI gauges

> *The dashboard runs from S3 via our data loader - no local files needed. Six tabs cover coaching analytics (Track 1), fan engagement (Track 2), and cross-domain validation (Track 3).*

---

## Slide 4: The Key Insight - Two Independent Dimensions of Body Intelligence

### AWI and PQI are statistically independent (r = -0.11). Both are needed.

This is the central analytical finding: a player who scans brilliantly does not automatically press well. The two skills are orthogonal - Body Intelligence has two genuinely separate dimensions.

**The four quadrants (400 player-phase observations):**

| | High PQI | Low PQI |
|---|---|---|
| **High AWI** | **Elite** (10 players) - scans AND presses well | Cognitive strength, mechanical gap |
| **Low AWI** | Physical strength, awareness gap | Development priority |

**Elite quadrant highlights:** Höjlund (26.9 / 63.9), Kimmich (21.8 / 64.7), Vuskovic (18.3 / 63.7 - a centre-back scanning like a defensive midfielder)

**Fatigue detection:** Kimmich's AWI drops 52% from 1st to 2nd half (23.4 → 11.3) in one match - a cognitive disengagement signal invisible to GPS or positional tracking.

**Cross-domain validation:** AWI maps to cockpit scanning research in aviation (Lounis et al., 2021). PQI sub-scores map to NBA defensive alignment, tennis biomechanics, and NFL tracking metrics. All peer-reviewed citations - the constructs we measure are well-established across sports science.

---

## Slide 5: Business Value & What Comes Next

### From 50 fps skeleton data to coaching decisions in plain English

**Kimmich's fatigue signal:** His AWI drops from 23.38 (FCU-FCB 1st half) to 11.29 (FCU-FCB 2nd half) - a **52% decline** in cognitive engagement that no GPS metric, heat-map, or positional model would detect.

**For clubs - three immediate applications:**
- **Scouting:** rank candidates by cognitive awareness (AWI) and pressing mechanics (PQI) independently
- **Fatigue monitoring:** AWI half-time comparisons surface cognitive disengagement before it becomes a tactical problem
- **Positional versatility:** players like Vuskovic whose scanning profiles cross position boundaries are flagged automatically

**For broadcasters & fans:** Fan View tab with broadcast-style top-3 counter, quadrant classification, and Body Intelligence leaderboard. Live overlay: *"Hojlund scanned 27 times before that interception."*

**For the DFL:** First matchday-grade metrics derived solely from TF15 skeleton data - differentiates the 3D product from any 2D competitor.

> *The +59% AWI spike before a pass confirms the metric captures the pre-decision cognitive window that coaches call "playing with your head up." TF15 makes this measurement possible at matchday scale for the first time in team sports.*
