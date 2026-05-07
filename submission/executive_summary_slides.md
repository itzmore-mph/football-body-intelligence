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
| 45° threshold (not 30°) | XY-plane projection compresses 3D angles; empirically tuned on Kimmich as validation anchor |
| Press frame filter: ≥10 consecutive frames within 5 m | Excludes incidental proximity; captures genuine pressing intent |
| SageMaker Processing (10 parallel jobs) | 5 matches × 2 metrics in ~15 min vs ~2 hrs local; zero cold-start overhead |

**Scale:** 5 Bundesliga matches · ~40 players × 2 halves = **400 player-phase rows** · 334 unit + property-based tests · production-containerised pipeline (Docker → ECR)

---

## Slide 3: Results - AWI Leaderboard, Position Patterns & Pre-Pass Signal

### Defensive midfielders are the most cognitively active players on the pitch - and AWI spikes 59% before a pass

**Top 10 by AWI (scans/min):**

| # | Player | Position | Match | Half | AWI |
|---|--------|----------|-------|------|-----|
| 1 | Oscar Winther Höjlund | DMZ | SGE-FCB | 1st | **26.90** |
| 2 | Hugo Emanuel Larsson | DMZ | SGE-FCU | 1st | 26.26 |
| 3 | Joshua Kimmich | DMR | FCU-FCB | 1st | 23.38 |
| 4 | Rani Khedira | DMR | SGE-FCU | 1st | 22.95 |
| 5 | Oscar Winther Höjlund | DMZ | SGE-FCB | 2nd | 22.74 |
| 6 | Rani Khedira | DMR | SGE-FCU | 2nd | 22.27 |
| 7 | Aljoscha Kemlein | DMZ | FCU-FCB | 2nd | 21.89 |
| 8 | Joshua Kimmich | DMR | FCB-HSV | 1st | 21.77 |
| 9 | Fábio Vieira | RA | FCB-HSV | 1st | 21.34 |
| 10 | Joshua Kimmich | DMR | FCB-HSV | 2nd | 21.15 |

**Position averages (AWI scans/min):**

| Position | Avg AWI | Role |
|----------|---------|------|
| DMZ - Defensive Mid Centre | 15.6 | Highest - face the most opponents |
| IVL/IVR - Centre-back | 10.6 / 5.2 | Wide role-dependent range |
| STZ - Striker Centre | 6.2 | Lowest outfield |
| TW - Goalkeeper | 3.5 | Ball-tracking dominant |

**Stability:** Cross-half Pearson r = **0.660** (n = 79 active player-phases): AWI is a stable player trait across the match. Kimmich: 21.77 (FCB-HSV 1st) → 21.15 (FCB-HSV 2nd) - a 0.6-scan variance across 47 minutes of football.

**Pre-pass AWI is +59% above the full-phase baseline.** Players ramp their scanning rate in the 5 seconds before releasing the ball. AWI directly measures the pre-decision cognitive window - the moment coaches describe as "playing with your head up."

---

## Slide 4: The Surprising Finding - AWI and PQI Are Independent

### Scanning awareness and pressing mechanics are distinct, orthogonal skills

AWI vs PQI Pearson r = **−0.11** (p = 0.12, n = 198 matched rows). The two metrics are statistically independent - knowing a player's scan rate tells you almost nothing about their pressing mechanics, and vice versa.

This is the most important analytical finding: **Body Intelligence has two orthogonal dimensions.** A single-metric model - tracking position, GPS load, or proximity alone - misses half the picture.

**The four quadrants:**

| | High PQI (≥62.6) | Low PQI (<62.6) |
|---|---|---|
| **High AWI (≥14.7 scans/min)** | 🟢 **Elite** - scans AND presses well | 🟡 Cognitive strength, mechanical gap |
| **Low AWI (<14.7 scans/min)** | 🟡 Physical strength, awareness gap | 🔴 Development priority |

**Elite quadrant (both above 75th percentile): 10 unique players, 15 player-phases**

Top performers: Oscar Höjlund (DMZ, 26.90 / 63.9), Joshua Kimmich (DMR, 21.77 / 64.7), Aljoscha Kemlein (DMZ), Nicolai Remberg (DMR), Luka Vušković (IVZ)

**Outlier worth noting:** Luka Vušković is a centre-back (IVZ) who scans at 18.33 scans/min - a rate that sits firmly in the elite defensive-midfielder range. The data flags his tactical versatility before any scout would.

**PQI leaders among outfield players:** DMZ (62.9) and DMR (62.1) lead, confirming that defensive midfielders dominate both cognitive and physical pressing dimensions. Kimmich's proximity sub-score of 96.0 is the highest among all outfield players in the dataset.

**Cross-domain validation:** AWI and PQI each map to a validated construct in another sport or industry. AWI aligns with cockpit visual-scanning research (Lounis et al., 2021, PLOS ONE), where expert pilots exhibit higher scan-transition rates and pre-decision scan bursts - the same pattern as our +59% pre-pass AWI spike. PQI sub-scores map to NBA defensive alignment analytics (Cervone et al., 2016, JASA; orientation), tennis ready-position biomechanics (Elliott, 2006, BJSM; stance), and NFL player-tracking separation metrics (Eager et al., 2023, MIT Sloan; proximity). The composite structure mirrors rugby 3D tackle biomechanics (Hendricks et al., 2021). All citations are peer-reviewed; cross-domain parallels are qualitative analogies, not direct statistical comparisons.

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
