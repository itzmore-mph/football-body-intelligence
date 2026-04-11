# Football Body Intelligence Platform

**AWS World Sports Innovation Cup 2026 · Challenge 2 – Unlock the Power of 3D Football Data**

**Team:** _itzmore_ · [GitHub](https://github.com/itzmore-mph/football-body-intelligence)

## Overview

Transforms raw TRACAB TF15 3D skeleton data from 5 Bundesliga matches into two complementary player intelligence metrics:

| Metric | What it measures | Output |
|--------|-----------------|--------|
| **AWI** — Awareness Index | Cognitive scanning: discrete head-rotation events per minute | `results/awi_full.csv` |
| **PQI** — Pressure Quality Index | Physical pressing mechanics: body orientation, stance, and proximity during press actions | `results/pqi_full.csv` |

A player can scan well but press poorly (high AWI, low PQI), or press with perfect mechanics but without pre-scanning (low AWI, high PQI). Elite players score high on both.

Results are surfaced through a Streamlit dashboard and AI-generated scouting narratives via AWS Bedrock.

---

## Key Results (5 matches, 400 player-phase rows)

### AWI — Awareness Index

**Range:** 3.28 – 26.90 scans/min · **Median:** 12.59

| Position | Avg AWI | Role |
|----------|---------|------|
| DMZ — Defensive Mid Centre | 15.6 | Highest: face the most opponents |
| IVL/IVR — Centre-back | 10.6 / 5.2 | Wide role-dependent range |
| STZ — Striker Centre | 6.2 | Lowest outfield |
| TW — Goalkeeper | 3.5 | Primarily ball-tracking |

**Stability:** Cross-half Pearson R = 0.854 (p < 0.001, n = 69 active phases) — AWI is a stable player trait, not match noise.

**Kimmich validation anchor:** 21.77 → 21.15 (FCB-HSV, halves 1 & 2) — consistent with his documented scanning reputation. Second-match second-half drop to 11.29 (−52%) flags fatigue or tactical instruction — a signal no GPS or positional metric captures.

**Pre-pass context:** AWI is +57% above full-phase baseline in the 5 s before a pass, confirming the metric measures pre-decision cognitive load rather than general movement.

### PQI — Pressure Quality Index

**Range:** 0 – 100 composite · **Outfield leaders:** DMZ (62.9), DMR (62.1)

Goalkeepers (TW) post the highest mean PQI (66.6) driven by exceptional proximity sub-scores (92+) — structurally expected given their role; position-specific benchmarking recommended for outfield comparisons.

### The Central Finding: AWI and PQI Are Independent

Pearson r = **−0.11** (p = 0.12, n = 198 matched observations). Scanning awareness and pressing mechanics are statistically orthogonal — both dimensions are required to characterise a player fully.

**Elite quadrant** (top 25% on both): **10 unique players** across 400 observations, including Oscar Winther Höjlund (DMZ, 26.90 AWI / PQI 63.9), Joshua Kimmich (DMR, 21.77 AWI / PQI 64.7), and Luka Vušković (IVZ, 18.33 AWI / PQI 63.7) — a centre-back whose scanning profile matches a defensive midfielder.

---

## Project Structure

```
Dockerfile                     Container image for SageMaker Processing jobs (Python 3.11-slim)
requirements.txt               Full local dependencies (dev, notebooks, dashboard, tests)
requirements-processing.txt   Stripped-down dependencies for the Processing container
pyproject.toml                 Pytest + Ruff configuration
.env.example                   Environment variable template — copy to .env and fill in values
project_start.sh.template      Startup script template — copy to project_start.sh and fill in values

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

tests/                         176 unit tests, no S3 access required

notebooks/
  eda_exploration.ipynb            EDA + AWI smoke test (requires S3)
  run_awi_pipeline.ipynb           Batch AWI for all 5 matches (requires S3)
  run_pqi_pipeline.ipynb           Batch PQI for all 5 matches (requires S3)
  analysis_awi_results.ipynb       AWI results analysis and leaderboard (CSV only)
  analysis_awi_pqi_combined.ipynb  Combined AWI+PQI analysis, 4 figures (CSV only)
  bedrock_reports.ipynb            AI narrative generation via Bedrock

dashboard/
  app.py                     Streamlit dashboard (Player Profile, Match Overview, Leaderboard)
  run_dashboard.sh           Launch script (port 8501)

results/                       Generated by pipeline (gitignored); committed: sample_awi.csv
figures/                       Generated by analysis notebooks (gitignored)
submission/                    HTML slides, PRFAQ, build script
```

---

## AWI Pipeline

```
S3 Parquet (TF15)
  └─ pyarrow row-group pushdown (no full download)
       │
       ▼
batch_pipeline.py → _extract_angles_vectorized()
  • head yaw: nose/neck primary, ear fallback
  • body yaw: shoulder primary, hip fallback
       │
       ▼
awi_calculator.py → detect_scans()
  • 11-frame circular rolling mean   (handles ±180° wrap)
  • 25-frame delta                   0.5s lookback window
  • 45° threshold                    flag large head turns
  • leading-edge count               1 rotation = 1 event
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

PQI measures pressing quality during frames where a player's pelvis is within 5 m of the ball carrier for ≥10 consecutive frames (0.2 s at 50 fps).

```
PQI = 0.40 × orientation_score + 0.30 × stance_score + 0.30 × proximity_score
```

| Sub-score | Formula | Peak |
|-----------|---------|------|
| Orientation | `max(0, 100 − (angle_to_carrier / 90) × 100)` | 100 when facing carrier directly |
| Stance | `100 × exp(−0.5 × ((knee_flex − 130) / 25)²)` | 100 at 130° knee flexion |
| Proximity | `max(0, 100 × (1 − distance_m / 5.0))` | 100 at 0 m, 0 at ≥5 m |

---

## Setup

### Prerequisites

- Python 3.11+
- AWS CLI with SSO configured for your sandbox account
- Docker Desktop (only needed to rebuild the SageMaker Processing container)

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure environment

```bash
cp .env.example .env
# Edit .env — set AWS_PROFILE, HACKATHON_BUCKET, and optionally BEDROCK_MODEL_ID
```

```bash
cp project_start.sh.template project_start.sh
# Edit project_start.sh — set your account-specific values (SM_ROLE_ARN, SM_IMAGE_URI, etc.)
```

Then source the startup script once per terminal session:

```bash
source project_start.sh
```

Use `source` (not `bash`) so the exported variables persist in your shell. The script activates the venv, logs in via SSO, verifies S3 access, and exports all SageMaker environment variables.

### Run tests

```bash
pytest tests/ -v
```

176 tests, all passing. No S3 access required.

---

## Running the Platform

### Option A — SageMaker Pipeline (recommended, ~15-20 min)

**One-time container build** (only needed after changes to `src/` or `scripts/`):

```bash
./pipelines/build_and_push.sh
```

**Run all 10 jobs in parallel:**

```bash
source project_start.sh
python pipelines/sagemaker_pipeline.py --action run
```

When complete, `results/awi_full.csv` and `results/pqi_full.csv` are written locally automatically.

**Check status of a previous run:**

```bash
python pipelines/sagemaker_pipeline.py --action status --run-id <run-id>
```

---

### Option B — Local notebooks (fallback, ~90-120 min)

The SSO token expires after 60 minutes. If it does, re-run `source project_start.sh` and re-execute from the last checkpoint — completed phases are skipped automatically.

1. **AWI** (requires S3): `notebooks/run_awi_pipeline.ipynb` → `results/awi_full.csv`
2. **PQI** (requires S3): `notebooks/run_pqi_pipeline.ipynb` → `results/pqi_full.csv`
3. **Analysis** (CSV only): `notebooks/analysis_awi_pqi_combined.ipynb` → `figures/`
4. **Dashboard**: `bash dashboard/run_dashboard.sh` → http://localhost:8501
5. **AI narratives** (requires Bedrock): `notebooks/bedrock_reports.ipynb` → `results/narratives.csv`

---

## Bedrock Narrative Generation

The `bedrock_reports.ipynb` notebook generates scouting narratives via AWS Bedrock. The model is configured via `BEDROCK_MODEL_ID` in `.env`.

**Default:** `eu.amazon.nova-lite-v1:0` (works in `eu-central-1` without SCP restrictions)

**To use Claude instead:** set `BEDROCK_MODEL_ID=eu.anthropic.claude-sonnet-4-6` in `.env` — requires your account's SCP to allow `bedrock:InvokeModel` across all EU regions that the inference profile may route to.

> Note: bare model IDs like `anthropic.claude-sonnet-4-6` are not supported for on-demand invocation. Always use an inference profile ID (prefixed with `eu.`, `us.`, or `global.`).

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

---

## AWS Services Used

| Service | Purpose |
|---------|---------|
| S3 | TF15 Parquet match data (read-only) + pipeline output storage |
| SageMaker Processing | Parallel AWI + PQI compute (10 jobs, `ml.m5.xlarge`, ~15-20 min) |
| ECR | Docker image registry for the Processing container |
| Bedrock | Player narrative generation (`eu.amazon.nova-lite-v1:0` default) |
