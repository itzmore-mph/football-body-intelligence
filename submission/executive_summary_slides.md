# Executive Summary: Football Body Intelligence Platform
**AWS World Sports Innovation Cup 2026 · Challenge 2**
Team: itzmore

---

## Slide 1: The Problem — What 3D Skeleton Data Unlocks

### Standard tracking tells you *where* players move. TF15 tells you *how* they move and *where they look*.

The Bundesliga's TRACAB TF15 system captures 141 million data points per match — not just player positions, but the full 3D skeleton: every joint, every frame, at 50 fps. Yet today this data powers only automated event detection. The body intelligence it contains — *is this player scanning before receiving? are they pressing with correct biomechanics?* — goes unmeasured.

We built two metrics that change that.

**AWI (Awareness Index)** — cognitive scanning rate: discrete head-rotation events per minute, derived from 3D nose/neck/ear keypoints. Impossible with 2D tracking. Directly measures where a player is *looking*, not just where they are standing.

**PQI (Pressure Quality Index)** — pressing mechanics score [0–100]: body orientation toward the ball carrier (40%), knee-flexion stance quality at the biomechanical optimum of 130° (30%), and proximity (30%). Computed only during genuine press frames — ≥10 consecutive frames within 5 m of the ball carrier.

> *Together, AWI and PQI form a complete Body Intelligence picture. A player can scan brilliantly but press with poor mechanics (high AWI, low PQI), or press perfectly without pre-scanning (low AWI, high PQI). Elite players score high on both — and the data shows exactly who they are.*

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
  → Streamlit dashboard               # 3 tabs: Player Profile, Match Overview, Leaderboard
  → Amazon Bedrock (Nova Lite)        # natural-language scouting narratives
```

**Key engineering decisions:**

| Decision | Why it matters |
|----------|---------------|
| Circular smoothing (sin/cos decomposition) | Standard rolling mean breaks at ±180°; this doesn't |
| Leading-edge counting `(is_scan & ~is_scan.shift(1)).sum()` | 1 sustained rotation = 1 event, not N frames |
| 45° threshold (not 30°) | XY-plane projection compresses 3D angles; tuned on Kimmich as validation anchor |
| Press frame filter: ≥10 consecutive frames within 5 m | Excludes incidental proximity; captures genuine pressing intent |

**Scale:** 5 matches × ~40 players × 2 halves = **400 player-phase rows** · 177 unit tests · SageMaker Processing for parallel compute (~15 min vs ~2 hrs local)

---

## Slide 3: Results — AWI Leaderboard & Position Patterns

### Defensive midfielders are the most cognitively active players on the pitch

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
| DMZ (Defensive Mid, Centre) | 15.6 | Highest — face most opponents |
| IVL/IVR (Centre-back) | 10.6 / 5.2 | Wide range by role |
| STZ (Striker, Centre) | 6.2 | Lowest outfield |
| TW (Goalkeeper) | 3.5 | Primarily ball-tracking |

**Kimmich as validation anchor:** 21.77 (FCB-HSV 1st) → 21.15 (FCB-HSV 2nd) → 23.38 (FCU-FCB 1st). Consistent with his documented scanning reputation. Cross-half Pearson R = **0.854** (p < 0.001, n = 69 active player-phases): AWI is a stable player trait, not match noise.

---

## Slide 4: The Surprising Finding — AWI and PQI Are Independent

### Scanning awareness and pressing mechanics are distinct, complementary skills

AWI vs PQI Pearson r = **−0.11** (p = 0.12, n = 198 matched rows). The two metrics are statistically independent — knowing a player's scan rate tells you almost nothing about their pressing mechanics, and vice versa.

This is the most important analytical finding: **Body Intelligence has two orthogonal dimensions.**

**The four quadrants:**

| | High PQI (≥62.6) | Low PQI (<62.6) |
|---|---|---|
| **High AWI (≥14.7 scans/min)** | 🟢 **Elite** — scans AND presses well | 🟡 Cognitive strength, mechanical gap |
| **Low AWI (<14.7 scans/min)** | 🟡 Physical strength, awareness gap | 🔴 Development priority |

**Elite quadrant (both above 75th percentile): 10 unique players, 15 player-phases**

Top performers: Oscar Höjlund (DMZ), Joshua Kimmich (DMR), Aljoscha Kemlein (DMZ), Nicolai Remberg (DMR), Luka Vušković (IVZ)

**PQI leaders by position:** Goalkeepers (TW) lead PQI at 66.6 — driven by exceptional proximity scores (92+) as they constantly close down ball carriers in their area. Among outfield players, DMZ (62.9) and DMR (62.1) lead, confirming that defensive midfielders dominate both cognitive and physical pressing dimensions.

**Kimmich's Body Intelligence profile:** AWI 21.77 / PQI 64.7 (FCB-HSV 1st half) — top 5% on both dimensions simultaneously. His proximity score of 96.0 is the highest among outfield players in the dataset.

---

## Slide 5: Business Value, AI Narratives & What Comes Next

### From 50 fps skeleton data to coaching decisions in plain English

**For clubs — three immediate applications:**
- **Scouting:** rank candidates by cognitive awareness (AWI) and pressing mechanics (PQI) independently — find players who excel at both, or identify which dimension needs development
- **Fatigue tracking:** Kimmich's AWI drops from 23.38 (FCU-FCB 1st) to 11.29 (FCU-FCB 2nd) — a 52% decline that no GPS metric would catch
- **Positional outliers:** Luka Vušković (CB, Hamburg) scores in the elite quadrant — his scanning profile matches a defensive midfielder, flagging tactical versatility

**For broadcasters:** AWI as a live overlay — *"Höjlund scanned 27 times in the last minute before that interception"* — a number every fan understands.

**For the DFL:** AWI and PQI are the first matchday-grade metrics derived solely from TF15 skeleton data. They differentiate the 3D product from any 2D competitor.

**AI-generated scouting narratives (Amazon Bedrock):**

> *"Oscar Winther Höjlund's AWI of 26.9 scans/min ranks #1 of 400 players — 72% above the league mean. His PQI of 63.9 reflects strong proximity management (96.0) with clear development headroom in orientation (59.0) and stance (38.1). The combination of elite scanning frequency and disciplined pressing makes him a rare dual-threat midfielder. Recommendation: deploy in high-press systems where pre-scanning directly enables ball recovery."*

**What comes next:**
- Pre-pass AWI → correlate with progressive pass rate and turnover rate
- Full-season AWI/PQI trend → fatigue arcs and development tracking
- Live broadcast overlay via DFL's matchday data stream

> *AWI and PQI turn 141 million data points per match into two numbers every coach already understands — and one quadrant chart that tells you who your elite players really are.*
