# Football Body Intelligence Platform

**AWS World Sports Innovation Cup 2026 · Challenge 2 – Unlock the Power of 3D Football Data**

**Team:** _itzmore_ · [GitHub](https://github.com/itzmore-mph/football-body-intelligence)

---

## What We Built

Two matchday-grade player intelligence metrics derived entirely from TRACAB TF15 3D skeleton data — impossible with 2D tracking:

| Metric | What it measures | Range |
|--------|-----------------|-------|
| **AWI** — Awareness Index | Cognitive scanning: discrete head-rotation events per minute, from 3D nose/neck/ear keypoints | 3.3 – 26.9 scans/min |
| **PQI** — Pressure Quality Index | Pressing mechanics: body orientation, knee-flexion stance, and proximity during genuine press actions | 0 – 100 |

A player can scan brilliantly but press with poor mechanics (high AWI, low PQI), or press perfectly without pre-scanning (low AWI, high PQI). **Elite players score high on both — and the data shows exactly who they are.**

---

## Key Results

### AWI — Awareness Index

**5 matches · 400 player-phase rows · Median: 12.59 scans/min**

| Position | Avg AWI | Interpretation |
|----------|---------|----------------|
| DMZ — Defensive Mid Centre | 15.6 | Highest: face the most opponents, must scan constantly |
| IVL — Centre-back (left) | 10.6 | Elevated: wide role demands spatial awareness |
| STZ — Striker Centre | 6.2 | Lowest outfield: forward-facing role, fewer opponents to track |
| TW — Goalkeeper | 3.5 | Ball-tracking dominant, not opponent-scanning |

**Stability:** Cross-half Pearson R = **0.854** (p < 0.001, n = 69 active phases) — AWI is a stable player trait, not match noise.

**Validation:** Kimmich (FCB-HSV): 21.77 → 21.15 across halves — consistent with his documented scanning reputation. His FCU-FCB 2nd-half drop to 11.29 (−52%) is a fatigue signal no GPS or positional metric captures.

**Pre-pass signal:** AWI is **+57% above full-phase baseline** in the 5 seconds before a pass — confirming the metric measures pre-decision cognitive load, not incidental movement.

### PQI — Pressure Quality Index

```
PQI = 0.40 × orientation_score + 0.30 × stance_score + 0.30 × proximity_score
```

**Outfield leaders:** DMZ (62.9), DMR (62.1), IVL (62.0)

Goalkeepers (TW) post the highest mean PQI (66.6), driven primarily by their **orientation sub-score** (70.4 vs 54.7 outfield average) — when closing down in their penalty area, goalkeepers face the ball carrier directly and square-on by necessity, which maximises the orientation component. This is a structural role effect; position-specific benchmarking is recommended for outfield comparisons.

### The Central Finding: AWI and PQI Are Independent

**Pearson r = −0.11** (p = 0.12, n = 198 matched observations)

Scanning awareness and pressing mechanics are statistically orthogonal. Both dimensions are required to characterise a player fully.

**Elite quadrant** (top 25% on both metrics): **10 unique players** across 400 observations

| Player | Position | AWI | PQI |
|--------|----------|-----|-----|
| Oscar Winther Höjlund | DMZ | 26.90 | 63.9 |
| Joshua Kimmich | DMR | 21.77 | 64.7 |
| Luka Vušković | IVZ | 18.33 | 63.7 |

Vušković is a centre-back whose scanning profile matches a defensive midfielder — the data flags his tactical versatility before any scout would.

---

## Project Structure

```
src/
  awi_calculator.py          Scan detection and AWI aggregation
  batch_pipeline.py          Multi-player, multi-phase, multi-match AWI orchestration
  body_orientation.py        Body yaw from shoulder/hip vectors
  bedrock_client.py          AWS Bedrock narrative generation (Nova / Claude)
  eda_helpers.py             AWS session factory, S3 utilities
  event_parser.py            MatchInformation XML parser
  pre_pass_awi.py            Pre-pass AWI enrichment (5s window before each pass)
  pressure_pipeline.py       PQI orchestration across all matches and phases
  pqi_calculator.py          PQI sub-scores: orientation, stance, proximity (vectorized)
  skeleton_parser.py         TF15 parser — head yaw extraction

scripts/
  run_awi_job.py             SageMaker Processing entry point — AWI for one match
  run_pqi_job.py             SageMaker Processing entry point — PQI for one match
  aggregate_results.py       Concatenates per-match CSVs into awi_full / pqi_full

pipelines/
  sagemaker_pipeline.py      Submits all 10 match jobs in parallel via boto3, then aggregates
  build_and_push.sh          Builds Docker image and pushes to ECR

tests/                       176 unit tests — no S3 access required

notebooks/
  eda_exploration.ipynb            EDA + AWI smoke test (requires S3)
  run_awi_pipeline.ipynb           Batch AWI for all 5 matches (requires S3)
  run_pqi_pipeline.ipynb           Batch PQI for all 5 matches (requires S3)
  analysis_awi_results.ipynb       AWI results analysis and leaderboard (CSV only)
  analysis_awi_pqi_combined.ipynb  Combined AWI+PQI analysis, 4 figures (CSV only)
  bedrock_reports.ipynb            AI narrative generation via Bedrock

dashboard/
  app.py                     Streamlit dashboard (Player Profile, Match Overview, Leaderboard)
  run_dashboard.sh           Launch script → http://localhost:8501

results/                     Generated by pipeline (gitignored)
submission/                  HTML slides, PRFAQ, build script
```

---

## AWI Pipeline

```
S3 Parquet (TF15, ~4 GB/match)
  └─ pyarrow row-group pushdown       # stream, never full download
       │
       ▼
_extract_angles_vectorized()
  • head yaw: nose/neck primary, ear fallback
  • body yaw: shoulder primary, hip fallback
       │
       ▼
detect_scans()
  • 11-frame circular rolling mean    # handles ±180° wrap via sin/cos decomposition
  • 25-frame delta (0.5 s window)
  • ≥45° threshold                   # XY-projection corrected; tuned on Kimmich
  • leading-edge count               # 1 sustained rotation = 1 event, not N frames
       │
       ▼
compute_awi() → scan_count / phase_minutes
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Frame rate | 50 fps | TF15 spec |
| Scan window | 25 frames (0.5 s) | Matches sports-science literature |
| Threshold | 45° | XY-projection compresses 3D angles; tuned on Kimmich |
| Smooth window | 11 frames (0.22 s) | Suppresses single-frame tracking artefacts |

---

## PQI Pipeline

Press frames: player's pelvis within 5 m of ball carrier for ≥10 consecutive frames (0.2 s at 50 fps).

| Sub-score | Formula | Peak |
|-----------|---------|------|
| Orientation (40%) | `max(0, 100 − (angle_to_carrier / 90) × 100)` | 100 when facing carrier directly |
| Stance (30%) | `100 × exp(−0.5 × ((knee_flex − 130) / 25)²)` | 100 at 130° knee flexion |
| Proximity (30%) | `max(0, 100 × (1 − distance_m / 5.0))` | 100 at 0 m, 0 at ≥5 m |

---

## Setup

### Prerequisites

- Python 3.11+
- AWS CLI with SSO configured (`aws configure sso`)
- Docker Desktop (only needed to rebuild the SageMaker container)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
# Set AWS_PROFILE, HACKATHON_BUCKET, BEDROCK_MODEL_ID

cp project_start.sh.template project_start.sh
# Set SM_ROLE_ARN, SM_IMAGE_URI, and other account-specific values
```

```bash
source project_start.sh   # activates venv, SSO login, exports env vars
```

Use `source` (not `bash`) — the script exports env vars that must persist in your shell.

### Run tests

```bash
pytest tests/ -v
```

176 tests, all passing. No S3 access required.

---

## Running the Platform

### Option A — SageMaker Pipeline (recommended, ~15–20 min)

```bash
# One-time container build (only after changes to src/ or scripts/)
./pipelines/build_and_push.sh

# Run all 10 jobs in parallel
source project_start.sh
python pipelines/sagemaker_pipeline.py --action run
```

Outputs: `results/awi_full.csv` and `results/pqi_full.csv`

### Option B — Local notebooks (~90–120 min)

1. `notebooks/run_awi_pipeline.ipynb` → `results/awi_full.csv`
2. `notebooks/run_pqi_pipeline.ipynb` → `results/pqi_full.csv`
3. `notebooks/analysis_awi_pqi_combined.ipynb` → `figures/`
4. `bash dashboard/run_dashboard.sh` → http://localhost:8501
5. `notebooks/bedrock_reports.ipynb` → `results/narratives.csv`

---

## AWS Services

| Service | Purpose |
|---------|---------|
| S3 | TF15 Parquet match data (read-only) + pipeline output |
| SageMaker Processing | Parallel AWI + PQI compute (10 jobs, `ml.m5.xlarge`, ~15–20 min) |
| ECR | Docker image registry for the Processing container |
| Bedrock | Player narrative generation (`eu.amazon.nova-lite-v1:0`) |

---

## Data

Five Bundesliga matches provided by DFL via the hackathon S3 bucket. No match data is stored in this repository.

| Match | File | Size |
|-------|------|------|
| FC Bayern München vs Hamburger SV | FCB-HSV.parquet | ~4.4 GB |
| Borussia Dortmund vs VfB Stuttgart | BVB-VFB.parquet | ~4.1 GB |
| Eintracht Frankfurt vs FC Bayern | SGE-FCB.parquet | ~3.7 GB |
| Eintracht Frankfurt vs Union Berlin | SGE-FCU.parquet | ~4.2 GB |
| Union Berlin vs FC Bayern | FCU-FCB.parquet | ~3.6 GB |
