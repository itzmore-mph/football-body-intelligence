# Football Body Intelligence Platform

**AWS World Sports Innovation Cup 2026 · Challenge 2 – Unlock the Power of 3D Football Data**

**Team:** _itzmore_ · [GitHub](https://github.com/itzmore-mph/football-body-intelligence)

---

## What We Built

Two matchday-grade player intelligence metrics derived entirely from TRACAB TF15 3D skeleton data - impossible with 2D tracking:

| Metric | What it measures | Range |
|--------|-----------------|-------|
| **AWI** - Awareness Index | Cognitive scanning: discrete head-rotation events per minute, from 3D nose/neck/ear keypoints | 3.3 - 26.9 scans/min |
| **PQI** - Pressure Quality Index | Pressing mechanics: body orientation, knee-flexion stance, and proximity during genuine press actions | 0 - 100 |

A player can scan brilliantly but press with poor mechanics (high AWI, low PQI), or press perfectly without pre-scanning (low AWI, high PQI). **Elite players score high on both - and the data shows exactly who they are.**

---

## Key Results

### AWI - Awareness Index

**5 matches · 400 player-phase rows · Median: 12.59 scans/min**

| Position | Avg AWI | Interpretation |
|----------|---------|----------------|
| DMZ - Defensive Mid Centre | 15.6 | Highest: face the most opponents, must scan constantly |
| IVL - Centre-back (left) | 10.6 | Elevated: wide role demands spatial awareness |
| STZ - Striker Centre | 6.2 | Lowest outfield: forward-facing role, fewer opponents to track |
| TW - Goalkeeper | 3.5 | Ball-tracking dominant, not opponent-scanning |

**Stability:** Cross-half Pearson R = **0.854** (p < 0.001, n = 69 active phases) - AWI is a stable player trait, not match noise.

**Validation:** Kimmich (FCB-HSV): 21.77 -> 21.15 across halves - consistent with his documented scanning reputation. Hojlund (SGE-FCB, 1st half): 26.90 scans/min serves as the second independent calibration anchor, cross-validating that the 45° threshold generalises beyond a single player. Kimmich's FCU-FCB 2nd-half drop to 11.29 (-52%) is a fatigue signal no GPS or positional metric captures.

> **Validation note:** The 45° scan threshold is empirically tuned on Kimmich as the primary anchor (FCB-HSV, 1st half), where the resulting AWI of 21.77 scans/min matches his documented scanning frequency from hand-coded video analysis in the coaching literature. The positional hierarchy (DMZ > CB > FW > GK) independently replicates the gradient reported by Jordet et al. (2020) in the EPL, providing cross-player validation. Additional hand-coded video comparisons across positions would further strengthen threshold confidence - this is a known limitation documented in the PRFAQ.

**Pre-pass signal:** AWI is **+57% above full-phase baseline** in the 5 seconds before a pass - confirming the metric measures pre-decision cognitive load, not incidental movement.

### PQI - Pressure Quality Index

```
PQI = 0.40 × orientation_score + 0.30 × stance_score + 0.30 × proximity_score
```

**Outfield leaders:** DMZ (62.9), DMR (62.1), IVL (62.0)

Goalkeepers (TW) post the highest mean PQI (66.6), driven primarily by their **orientation sub-score** (70.4 vs 54.7 outfield average) - when closing down in their penalty area, goalkeepers face the ball carrier directly and square-on by necessity, which maximises the orientation component. This is a structural role effect; position-specific benchmarking is recommended for outfield comparisons.

### The Central Finding: AWI and PQI Are Independent

**Pearson r = −0.11** (p = 0.12, n = 198 matched observations)

Scanning awareness and pressing mechanics are statistically orthogonal. Both dimensions are required to characterise a player fully.

**Elite quadrant** (top 25% on both metrics): **10 unique players** across 400 observations

| Player | Position | AWI | PQI |
|--------|----------|-----|-----|
| Oscar Winther Höjlund | DMZ | 26.90 | 63.9 |
| Joshua Kimmich | DMR | 21.77 | 64.7 |
| Luka Vušković | IVZ | 18.33 | 63.7 |

Vušković is a centre-back whose scanning profile matches a defensive midfielder - the data flags his tactical versatility before any scout would.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        AWS Infrastructure                           │
│                                                                     │
│   S3 (TF15 Parquet, ~4 GB/match)                                    │
│       │                                                             │
│       ▼                                                             │
│   ECR (Docker image)  ──►  SageMaker Processing                     │
│                              10 parallel jobs                       │
│                              (5 matches × AWI + PQI)                │
│                              ml.m5.xlarge, ~15–20 min               │
│                                   │                                 │
│                                   ▼                                 │
│                         S3 (per-match CSVs)                         │
│                                   │                                 │
│                         aggregate_results.py                        │
│                                   │                                 │
│                    ┌──────────────┴──────────────┐                  │
│                    ▼                             ▼                  │
│             awi_full.csv                  pqi_full.csv              │
│                    │                             │                  │
│                    └──────────────┬──────────────┘                  │
│                                   ▼                                 │
│              Streamlit Dashboard  +  Amazon Bedrock                 │
│              (Player Profile, Match Overview, Leaderboard,          │
│               Fan View, Broadcast Demo,                             │
│               AI scouting narratives via Nova Lite)                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
.
├── src/
│   ├── angle_utils.py           Single source of truth for circular yaw arithmetic
│   ├── awi_calculator.py        Scan detection and AWI aggregation
│   ├── awi_calibration.py       AWI threshold validation against Kimmich + Hojlund anchors
│   ├── batch_pipeline.py        Multi-player, multi-phase, multi-match AWI orchestration
│   ├── bedrock_client.py        AWS Bedrock narrative generation (Nova Lite / Claude)
│   ├── benchmark_reference.py   Parameterised cross-domain reference distributions (6 systems, no network calls)
│   ├── benchmark_report.py      Narrative entries (NBA/NFL/Cricket/Industrial) + generate_benchmark_summary()
│   ├── body_orientation.py      Body yaw from shoulder/hip vectors
│   ├── eda_helpers.py           AWS session factory, S3 utilities
│   ├── event_parser.py          MatchInformation XML parser
│   ├── pipeline_io.py           Shared S3/Parquet IO with retry/backoff (used by both pipelines)
│   ├── pqi_calculator.py        PQI sub-scores: orientation, stance, proximity (vectorized)
│   ├── pqi_normalizer.py        Position-adjusted PQI z-scores (GK/DEF/MID/FWD groups)
│   ├── pqi_sensitivity.py       PQI robustness analysis across threshold variations
│   ├── pre_pass_awi.py          Pre-pass AWI enrichment (5 s window before each pass)
│   ├── pressure_pipeline.py     PQI orchestration across all matches and phases
│   ├── quadrant_analysis.py     Bootstrap CI for elite quadrant count
│   └── skeleton_parser.py       TF15 Parquet parser - head yaw extraction
│
├── scripts/                     SageMaker container entry points (internal plumbing - not run directly)
│   ├── _sagemaker_helpers.py    Shared scaffolding: env parsing, boto3/PyArrow S3 client, result CSV writing
│   ├── run_awi_job.py           Entry point executed inside the SageMaker Processing container - AWI for one match
│   ├── run_pqi_job.py           Entry point executed inside the SageMaker Processing container - PQI for one match
│   ├── aggregate_results.py     Concatenates per-match CSVs -> awi_full / pqi_full (called by sagemaker_pipeline.py)
│   └── broadcast_screenshot.py  Captures a 1280x720 screenshot of the broadcast demo overlay
│
├── pipelines/
│   ├── sagemaker_pipeline.py    Submits 10 jobs in parallel via boto3, then aggregates
│   └── build_and_push.sh        Builds Docker image and pushes to ECR
│
├── notebooks/
│   ├── eda_exploration.ipynb          EDA + AWI smoke test (requires S3)
│   ├── run_awi_pipeline.ipynb         Batch AWI for all 5 matches - local fallback for SageMaker (requires S3)
│   ├── run_pqi_pipeline.ipynb         Batch PQI for all 5 matches - local fallback for SageMaker (requires S3)
│   ├── analysis_awi_results.ipynb     AWI leaderboard and position analysis (CSV only)
│   ├── analysis_awi_pqi_combined.ipynb  Combined analysis, 4 figures (CSV only)
│   ├── bedrock_reports.ipynb          AI narrative generation via Bedrock
│   └── benchmark_cross_sport.ipynb    Cross-domain benchmarking: AWI/PQI vs other sports (CSV only, Track 3)
│
├── dashboard/
│   ├── app.py                   Streamlit dashboard (6 tabs: Player Profile, Match Overview, Leaderboard, Fan View, Broadcast Demo, Benchmark)
│   ├── broadcast_demo.py        Standalone broadcast overlay demo (Track 2, also embedded as Broadcast Demo tab)
│   └── run_dashboard.sh         Launch script → http://localhost:8501
│
├── tests/                       324 unit + property-based tests - no S3 access required
├── results/
│   ├── awi_full.csv             Pre-computed AWI scores (committed) - 400 rows, player x match x phase
│   ├── pqi_full.csv             Pre-computed PQI scores (committed) - 400 rows, same structure
│   ├── combined_full.csv        AWI + PQI merged dataset (committed) - used by broadcast demo
│   ├── narratives.csv           Pre-computed Bedrock narratives (committed)
│   ├── match_moments.csv        Top-5 pre-pass scan moments per match (committed)
│   └── per_match/               Raw per-match pipeline outputs (gitignored)
├── figures/                     Analysis figures generated by notebooks (gitignored)
├── submission/                  HTML slides, PRFAQ, build script
│   ├── build_html.py            Builds executive summary HTML with embedded figures
│   ├── executive_summary_slides.md  Slide deck source (Markdown)
│   ├── prfaq.md                 PRFAQ document source (Markdown)
│   ├── SUBMISSION_CHECKLIST.md  Pre-submission checklist
│   └── VIDEO_NARRATION_SCRIPT.md  Script for the 3-minute presentation video
├── .env.example                 Environment variable template - copy to .env and fill in
├── project_start.sh.template    Session activation template - copy to project_start.sh and fill in
├── Dockerfile                   SageMaker Processing container (Python 3.11-slim)
├── requirements.txt             Full local dependencies
├── requirements-processing.txt  Stripped-down container dependencies
└── pyproject.toml               Pytest + Ruff configuration
```

---

## Setup

### Prerequisites

- Python 3.11+
- AWS CLI with SSO configured (`aws configure sso`)
- Docker Desktop (only needed to rebuild the SageMaker container)

> **Windows users:** Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and enable Docker Desktop's WSL2 backend. Then run all commands inside a WSL2 terminal. The shell scripts (`.sh`) require a bash environment.

---

### 1. Create virtual environment

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Windows (Command Prompt)**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

---

### 2. Configure environment

**macOS / Linux**
```bash
cp .env.example .env
cp project_start.sh.template project_start.sh
```

**Windows**
```cmd
copy .env.example .env
copy project_start.sh.template project_start.sh
```

Edit `.env` and set:

| Variable | Description |
|----------|-------------|
| `AWS_PROFILE` | Your SSO profile name (from `aws configure sso`) |
| `AWS_DEFAULT_REGION` | `eu-central-1` |
| `HACKATHON_BUCKET` | S3 bucket name provided by AWS |
| `BEDROCK_MODEL_ID` | `eu.amazon.nova-lite-v1:0` (default) or `eu.anthropic.claude-sonnet-4-6` |

Edit `project_start.sh` and set:

| Variable | Description |
|----------|-------------|
| `SM_ROLE_ARN` | SageMaker execution role ARN (`arn:aws:iam::<account>:role/SageMakerExecutionRole`) |
| `SM_IMAGE_URI` | ECR image URI (`<account>.dkr.ecr.eu-central-1.amazonaws.com/football-bi-processing:latest`) |
| `CHALLENGE_PREFIX` | S3 prefix for match data (no trailing slash) |

---

### 3. Activate session

**macOS / Linux**
```bash
source project_start.sh
```

**Windows (WSL2 or Git Bash)**
```bash
source project_start.sh
```

**Windows (PowerShell - manual equivalent)**
```powershell
# Activate venv
.venv\Scripts\Activate.ps1

# Load .env
Get-Content .env | Where-Object { $_ -notmatch '^#' -and $_ -match '=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    [System.Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), 'Process')
}

# SSO login
aws sso login --profile $env:AWS_PROFILE
```

> Re-run when the SSO token expires (~60 min).

---

### 4. Run tests

```bash
pytest tests/ -v
```

Works identically on macOS, Linux, and Windows. 324 tests, all passing. No S3 access required.

---

## Quick Start - Verify Results Without AWS Credentials

The pre-computed result files (`results/awi_full.csv`, `results/pqi_full.csv`, `results/narratives.csv`) are committed to this repository. You can run the full dashboard immediately without any AWS credentials:

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash dashboard/run_dashboard.sh
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run dashboard/app.py --server.port 8501
```

Opens at **http://localhost:8501**

---

## Broadcast Demo

**Track 2 - Fan and Broadcast Engagement**

The broadcast demo shows how AWI and PQI would appear as live match statistics on a Bundesliga broadcast screen. It is a high-fidelity mockup using real player data from `results/combined_full.csv` and the official DFL color system (red `#D10214`, black, white, grey).

The overlay renders a lower-third style layout with:

- A player selector populated from all 400 player-phase observations
- Two circular gauges: AWI (scans/min vs league mean) and PQI (player score vs role mean), both with red fill on black background
- A Quadrant Badge classifying the player as ELITE, AWARE, PRESSER, or DEVELOPING based on 75th-percentile thresholds
- An animated ticker cycling through three validated findings every 3 seconds: "+57% pre-pass scan spike", "r=-0.11 AWI/PQI independence", "R=0.854 cross-half stability"

This addresses the gap in Track 2: the coaching and scouting tools are complete, and the broadcast demo makes the fan engagement value proposition tangible for hackathon judges.

### Launch the standalone broadcast demo

```bash
streamlit run dashboard/broadcast_demo.py
```

Opens at **http://localhost:8501**

No AWS credentials required. The demo loads pre-computed results from `results/combined_full.csv`.

### Screenshot

To capture a screenshot of the overlay at 1280x720, run:

```bash
python scripts/broadcast_screenshot.py
```

If selenium or playwright are not installed, the script prints manual capture instructions.

---

## Full Workflow

### Step 0 (optional) - Exploratory data analysis

Requires S3 access. Useful for understanding the raw TF15 skeleton data before running the full pipeline:

```bash
# In Jupyter:
notebooks/eda_exploration.ipynb    # EDA + AWI smoke test on raw Parquet data
```

---

### Step 1 - Build and push the Docker container

Only needed once, or after changes to `src/` or `scripts/`:

**macOS / Linux**
```bash
./pipelines/build_and_push.sh
```

**Windows (WSL2 or Git Bash)**
```bash
bash pipelines/build_and_push.sh
```

This builds the Python 3.11-slim image and pushes it to ECR.

---

### Step 2 - Run the pipeline

**Option A - SageMaker (recommended, ~15-20 min)**

Submits 10 parallel jobs (5 matches × AWI + PQI), waits for completion, then downloads and concatenates results locally:

```bash
python pipelines/sagemaker_pipeline.py --action run
```

Check status of a running or previous run:

```bash
python pipelines/sagemaker_pipeline.py --action status --run-id <YYYYMMDD-HHMMSS>
```

Outputs written to:
- `results/awi_full.csv` - 400 rows, one per player x match x phase
- `results/pqi_full.csv` - 400 rows, same structure

**Option B - Local notebooks (~90-120 min, fallback)**

Run in order. The SSO token expires after ~60 min - re-run `source project_start.sh` (macOS/Linux) or the PowerShell equivalent (Windows) and resume from the last checkpoint (completed phases are skipped automatically):

```bash
# In Jupyter:
notebooks/run_awi_pipeline.ipynb    # → results/awi_full.csv
notebooks/run_pqi_pipeline.ipynb    # → results/pqi_full.csv
```

---

### Step 3 - Analyse results

No S3 access required - runs entirely from the CSV outputs:

```bash
# In Jupyter:
notebooks/analysis_awi_results.ipynb         # AWI leaderboard, position breakdown
notebooks/analysis_awi_pqi_combined.ipynb    # Combined analysis → figures/
```

---

### Step 3b - Cross-domain benchmarking (Track 3)

No S3 access required. Places AWI and PQI in context of reference systems from other sports and industries (aviation, NBA, NFL, tennis, cricket, industrial ergonomics):

```bash
# In Jupyter:
notebooks/benchmark_cross_sport.ipynb    # Cross-domain comparison → figures/
```

---

### Step 4 - Launch the dashboard

**macOS / Linux**
```bash
bash dashboard/run_dashboard.sh
```

**Windows (PowerShell)**
```powershell
streamlit run dashboard/app.py --server.port 8501
```

Opens at **http://localhost:8501**

Six tabs:
- **Player Profile** - player selector with DFL position code + full name, AWI/PQI gauges vs selection median, 5-metric KPI row with percentile ranks and tooltips, scatter in context (colored by role), PQI radar vs role average, PQI component breakdown (Orientation/Stance/Proximity), per-phase trend cards
- **Match Overview** - summary KPIs, quadrant scatter with 4-quadrant classification and elite player labels, bootstrap CI caption below the scatter plot, role lollipop chart, PQI decomposition stacked bar, half-time fatigue bar + 1st vs 2nd half scatter, team AWI comparison (color per club)
- **Leaderboard** - "Position-adjusted PQI" toggle (sorts by z-score within position group when on), sortable table with DFL position codes, role averages heatmap, AWI bar chart (mean +/- std), PQI box distribution
- **Fan View** - broadcast-style top-3 awareness counter, 4-quadrant player comparison (Elite/Smart/Physical/Developing), Body Intelligence leaderboard, "Did you know?" callout with real-time scan frequency
- **Broadcast Demo** - DFL-styled live overlay mockup: player selector, AWI and PQI circular gauges vs league/role mean, quadrant badge (ELITE/AWARE/PRESSER/DEVELOPING), DFL red ticker with validated findings
- **Benchmark** - cross-domain validation: AWI vs aviation cognitive load research, PQI sub-scores vs NBA Second Spectrum / Tennis Hawk-Eye / NFL Next Gen Stats, pre-decision scan burst comparison (5-bar chart: aviation vs football)

Sidebar: Match, Phase, Position (DFL codes), Min Coverage % (default 50%). Collapsible **Metric Definitions** and **Position Code Reference** expanders explain all metrics and DFL codes inline.

---

### Step 5 - Generate AI scouting narratives

Requires Bedrock access. Uses `eu.amazon.nova-lite-v1:0` by default (no SCP restrictions in `eu-central-1`):

```bash
# In Jupyter:
notebooks/bedrock_reports.ipynb    # → results/narratives.csv
```

To use Claude Sonnet instead, set `BEDROCK_MODEL_ID=eu.anthropic.claude-sonnet-4-6` in `.env`.

> **Note:** Always use an inference profile ID (prefixed `eu.`, `us.`, or `global.`). Bare model IDs like `anthropic.claude-sonnet-4-6` are not supported for on-demand invocation.

---

## AWI - Technical Details

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
  • ≥45° threshold                    # XY-projection corrected; tuned on Kimmich
  • leading-edge count                # 1 sustained rotation = 1 event, not N frames
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

## PQI - Technical Details

Press frames: player's pelvis within 5 m of ball carrier for ≥10 consecutive frames (0.2 s at 50 fps).

| Sub-score | Weight | Formula | Peak |
|-----------|--------|---------|------|
| Orientation | 40% | `max(0, 100 − (angle_to_carrier / 90) × 100)` | 100 when facing carrier directly |
| Stance | 30% | `100 × exp(−0.5 × ((knee_flex − 130) / 25)²)` | 100 at 130° knee flexion |
| Proximity | 30% | `max(0, 100 × (1 − distance_m / 5.0))` | 100 at 0 m, 0 at ≥5 m |

---

## AWS Services

| Service | Purpose |
|---------|---------|
| S3 | TF15 Parquet match data (read-only) + pipeline output storage |
| SageMaker Processing | Parallel AWI + PQI compute (10 jobs, `ml.m5.xlarge`, ~15–20 min) |
| ECR | Docker image registry for the Processing container |
| Bedrock | Player narrative generation (`eu.amazon.nova-lite-v1:0`) |

---

## Data

Five Bundesliga matches provided by DFL via the hackathon S3 bucket. **No raw match data or skeleton source files are stored in this repository.** The committed CSV files (`results/awi_full.csv`, `results/pqi_full.csv`, `results/narratives.csv`) are derived metric aggregations computed by the pipeline - they contain one row per player per match phase and no raw positional or skeletal coordinates.

| Match | ID | File | Size |
|-------|----|------|------|
| FC Bayern München vs Hamburger SV | FCB-HSV | FCB-HSV.parquet | ~4.4 GB |
| Borussia Dortmund vs VfB Stuttgart | BVB-VFB | BVB-VFB.parquet | ~4.1 GB |
| Eintracht Frankfurt vs FC Bayern | SGE-FCB | SGE-FCB.parquet | ~3.7 GB |
| Eintracht Frankfurt vs Union Berlin | SGE-FCU | SGE-FCU.parquet | ~4.2 GB |
| Union Berlin vs FC Bayern | FCU-FCB | FCU-FCB.parquet | ~3.6 GB |