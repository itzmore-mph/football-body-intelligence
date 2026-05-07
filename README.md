# Football Body Intelligence Platform

**AWS World Sports Innovation Cup 2026 - Challenge 2: Unlock the Power of 3D Football Data**

**Team:** _itzmore_ - [GitHub](https://github.com/itzmore-mph/football-body-intelligence)

---

## What We Built

Two matchday-grade player intelligence metrics derived entirely from TRACAB TF15 3D skeleton data (50 fps, ~4 GB/match, 5 Bundesliga matches). Both are impossible with 2D tracking alone.

| Metric | What it measures | Range |
|--------|-----------------|-------|
| **AWI** (Awareness Index) | Cognitive scanning: discrete head-rotation events per minute, from 3D nose/neck/ear keypoints | 3.3 - 26.9 scans/min |
| **PQI** (Pressure Quality Index) | Pressing mechanics: body orientation, knee-flexion stance, and proximity during genuine press actions | 0 - 100 |

AWI and PQI are statistically independent (Pearson r = -0.11, p = 0.12). A player can scan brilliantly but press with poor mechanics, or vice versa. Elite players score high on both, and the data identifies exactly who they are.

---

## Challenge Tracks

| Track | Requirement | Where to find it |
|-------|-------------|------------------|
| **Track 1: Innovative KPIs** | Novel metrics from 3D skeleton data | `src/awi_calculator.py`, `src/pqi_calculator.py`, dashboard tabs Player Profile + Match Overview + Leaderboard |
| **Track 2: Fan Engagement** | Solutions for clubs, partners, fans | Dashboard tabs Fan View + Broadcast Demo, `dashboard/broadcast_demo.py` |
| **Track 3: Cross-Domain Benchmark** | Best practices from other sports/industries | Dashboard tab Benchmark, `src/benchmark_reference.py`, `notebooks/benchmark_cross_sport.ipynb` |

---

## Key Results

| Finding | Value | Detail |
|---------|-------|--------|
| Player-phase observations | 400 | ~40 players x 2 halves x 5 matches |
| AWI median | 12.59 scans/min | DMZ highest (15.6), TW lowest (3.5) |
| PQI outfield leaders | DMZ 62.9, DMR 62.1 | Position-specific benchmarking recommended |
| AWI-PQI independence | r = -0.11 (p = 0.12) | Both dimensions needed to characterise a player |
| Cross-half AWI stability | r = 0.660 (n = 79) | AWI is a stable trait, not match noise |
| Pre-pass AWI spike | +59% above baseline | Confirms cognitive load measurement |
| Elite quadrant | 10 unique players | Top 25% on both AWI and PQI |
| Test coverage | 334 tests | Unit + property-based, all passing, no AWS required |

**Validation anchors:** Kimmich (FCB-HSV, 21.77 scans/min) matches coaching literature. Hojlund (SGE-FCB, 26.90 scans/min) cross-validates the 45-degree threshold. Positional hierarchy (DMZ > CB > FW > GK) replicates Jordet et al. (2020) EPL findings.

**Known limitations:** Single-player threshold calibration (Kimmich as primary anchor), XY-plane projection compresses 3D angles, occlusion artifacts in high-movement phases, no ball-possession context for AWI. Full details in the [PRFAQ](submission/prfaq.md).

---

## Quick Start: View the Dashboard

The dashboard loads data from S3 via `src/s3_data_loader.py`. Configure credentials, then launch:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                             # fill in HACKATHON_BUCKET + AWS credentials
bash dashboard/run_dashboard.sh                  # http://localhost:8501
```

The S3 data loader reads `awi_full.csv` and `pqi_full.csv` from the configured bucket. It supports:
- **Streamlit secrets** (`st.secrets["aws"]`) for Streamlit Community Cloud deployment
- **Environment variables** (`HACKATHON_BUCKET`, `AWS_ACCESS_KEY_ID`, etc.) for local use
- **Local file fallback** if `results/` CSVs exist on disk (e.g. after running the pipeline)

> **Note:** Result CSVs are not committed to this repository (hackathon data rules). They are either loaded from S3 at runtime or generated locally via the full pipeline below.

---

## Full Workflow: Reproduce from Scratch

Requires AWS credentials with access to the hackathon S3 bucket.

### Step 1. Environment setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env                                # fill in values (see table below)
cp project_start.sh.template project_start.sh       # fill in SM_ROLE_ARN, SM_IMAGE_URI, CHALLENGE_PREFIX
source project_start.sh                             # activates venv + SSO login + exports env vars
```

**.env variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| `AWS_PROFILE` | SSO profile name (from `aws configure sso`) | - |
| `AWS_DEFAULT_REGION` | AWS region | `eu-central-1` |
| `HACKATHON_BUCKET` | S3 bucket name provided by AWS | - |
| `BEDROCK_MODEL_ID` | Bedrock model for narratives | `eu.amazon.nova-lite-v1:0` |

**project_start.sh variables:**

| Variable | Description |
|----------|-------------|
| `SM_ROLE_ARN` | SageMaker execution role ARN |
| `SM_IMAGE_URI` | ECR image URI for the Processing container |
| `CHALLENGE_PREFIX` | S3 prefix for match data (no trailing slash) |

> Re-run `source project_start.sh` when the SSO token expires (~60 min).

### Step 2. Build the Docker container (once)

```bash
./pipelines/build_and_push.sh
```

Builds a Python 3.11-slim image and pushes it to ECR. Only needed once, or after changes to `src/` or `scripts/`.

### Step 3. Run the compute pipeline

**Option A: SageMaker (recommended, ~15-20 min)**

```bash
python pipelines/sagemaker_pipeline.py --action run
```

Submits 10 parallel jobs (5 matches x AWI + PQI), waits for completion, downloads and concatenates results.

**Option B: Local notebooks (fallback, ~90-120 min)**

```bash
# In Jupyter, run in order:
notebooks/run_awi_pipeline.ipynb       # -> results/awi_full.csv + results/match_moments.csv
notebooks/run_pqi_pipeline.ipynb       # -> results/pqi_full.csv
```

Checkpoint-based: completed phases are skipped on re-run after token expiry.

**Outputs after this step:**

| File | Content |
|------|---------|
| `results/awi_full.csv` | 400 rows: AWI + enrichment (HBD, pre-pass AWI, scan direction) |
| `results/pqi_full.csv` | 400 rows: PQI + sub-scores (orientation, stance, proximity) |
| `results/match_moments.csv` | Top-5 pre-pass scan moments per match |

### Step 4. Analyse results and generate figures

No S3 access required from here on. Runs from the CSVs produced in Step 3.

```bash
# In Jupyter:
notebooks/analysis_awi_results.ipynb           # AWI leaderboard, position breakdown
notebooks/analysis_awi_pqi_combined.ipynb       # Combined analysis -> figures/
notebooks/benchmark_cross_sport.ipynb           # Cross-domain benchmarking (Track 3) -> figures/
```

### Step 5. Generate AI scouting narratives

Requires Bedrock access (`eu.amazon.nova-lite-v1:0` by default).

```bash
# In Jupyter:
notebooks/bedrock_reports.ipynb                # -> results/narratives.csv
```

To use Claude Sonnet instead, set `BEDROCK_MODEL_ID=eu.anthropic.claude-sonnet-4-6` in `.env`.

> Always use an inference profile ID (prefixed `eu.`, `us.`, or `global.`). Bare model IDs are not supported for on-demand invocation.

### Step 6. Launch the dashboard

```bash
bash dashboard/run_dashboard.sh                # http://localhost:8501
```

The dashboard reads `results/awi_full.csv` and `results/pqi_full.csv` directly (merges on the fly). Narratives from `results/narratives.csv` are loaded if available but not required.

**Six tabs:**

| Tab | Purpose | Track |
|-----|---------|-------|
| Player Profile | Per-player AWI/PQI gauges, radar, PQI decomposition, phase trends | 1 |
| Match Overview | Quadrant scatter, role comparison, fatigue analysis, team AWI | 1 |
| Leaderboard | Sortable table, position-adjusted PQI toggle, role heatmap | 1 |
| Fan View | Top-3 awareness counter, 4-quadrant classification, Body Intelligence leaderboard | 2 |
| Broadcast Demo | DFL-styled live overlay mockup with AWI/PQI gauges and ticker | 2 |
| Benchmark | Cross-domain validation: qualitative parallels with aviation, NBA, NFL, tennis, rugby | 3 |

Sidebar filters: Match, Phase, Position (DFL codes), Min Coverage %. Collapsible expanders explain all metrics and DFL position codes.

The Broadcast Demo tab is also available as a standalone app:

```bash
streamlit run dashboard/broadcast_demo.py
```

### Step 7. Run tests

```bash
pytest tests/ -v
```

334 tests (unit + property-based), all passing. No AWS access required.

---

## Architecture

```
S3 (TF15 Parquet, ~4 GB/match)
  |
  v
ECR (Docker image) --> SageMaker Processing
                         10 parallel jobs (5 matches x AWI + PQI)
                         ml.m5.xlarge, ~15-20 min
                           |
                           v
                       S3 (per-match CSVs)
                           |
                       aggregate locally
                           |
                +----------+----------+
                v                     v
          awi_full.csv          pqi_full.csv
                |                     |
                +----------+----------+
                           |
            +--------------+--------------+
            v              v              v
      Analysis        Dashboard       Bedrock
      Notebooks       (Streamlit)     Narratives
      + Figures        6 tabs         (Nova Lite)
```

---

## Project Structure

```
src/                              Core metric computation
  awi_calculator.py                 Scan detection and AWI aggregation
  pqi_calculator.py                 PQI sub-scores: orientation, stance, proximity
  skeleton_parser.py                TF15 Parquet parser, head yaw extraction
  batch_pipeline.py                 Multi-match AWI orchestration + enrichment
  pressure_pipeline.py              Multi-match PQI orchestration
  pre_pass_awi.py                   Pre-pass AWI enrichment (5s window before passes)
  body_orientation.py               Body yaw from shoulder/hip vectors
  angle_utils.py                    Circular yaw arithmetic
  event_parser.py                   MatchInformation XML parser
  pipeline_io.py                    Shared S3/Parquet IO with retry/backoff
  eda_helpers.py                    AWS session factory, S3 utilities
  bedrock_client.py                 Bedrock narrative generation
  s3_data_loader.py                S3/local CSV loader (Streamlit Cloud + local fallback)
  benchmark_reference.py            Cross-domain reference catalogue (6 systems)
  benchmark_report.py               Benchmark narrative entries + summary
  awi_calibration.py                AWI threshold validation (Kimmich + Hojlund)
  pqi_normalizer.py                 Position-adjusted PQI z-scores
  pqi_sensitivity.py                PQI weight sensitivity analysis
  quadrant_analysis.py              Bootstrap CI for elite quadrant

scripts/                          SageMaker container entry points
  run_awi_job.py                    AWI for one match (inside container)
  run_pqi_job.py                    PQI for one match (inside container)
  aggregate_results.py              Concatenates per-match CSVs
  _sagemaker_helpers.py             Shared scaffolding for container jobs
  broadcast_screenshot.py           Captures broadcast demo screenshot

pipelines/
  sagemaker_pipeline.py             Submits 10 parallel jobs, aggregates results
  build_and_push.sh                 Builds Docker image, pushes to ECR

notebooks/
  run_awi_pipeline.ipynb            Batch AWI, local fallback (requires S3)
  run_pqi_pipeline.ipynb            Batch PQI, local fallback (requires S3)
  analysis_awi_results.ipynb        AWI leaderboard + position analysis (CSV only)
  analysis_awi_pqi_combined.ipynb   Combined analysis, figures (CSV only)
  benchmark_cross_sport.ipynb       Cross-domain benchmarking (CSV only, Track 3)
  bedrock_reports.ipynb             AI narrative generation (requires Bedrock)

dashboard/
  app.py                            Streamlit dashboard (6 tabs)
  broadcast_demo.py                 Standalone broadcast overlay demo
  run_dashboard.sh                  Launch script

tests/                            334 unit + property-based tests (no AWS required)
results/                          Pipeline outputs (gitignored, loaded from S3 at runtime)
figures/                          Analysis figures (4 tracked, rest gitignored)
submission/                       PRFAQ, slides, HTML exports, video script

Dockerfile                        SageMaker Processing container (Python 3.11-slim)
requirements.txt                  Full local dependencies
requirements-processing.txt       Container dependencies (stripped down)
pyproject.toml                    Pytest + Ruff configuration
.env.example                      Environment variable template
project_start.sh.template         Session activation template
```

---

## Technical Details

### AWI (Awareness Index)

```
TF15 Parquet -> pyarrow row-group pushdown (stream, never full download)
  -> _extract_angles_vectorized()
       head yaw: nose/neck primary, ear fallback
       body yaw: shoulder primary, hip fallback
  -> detect_scans()
       11-frame circular rolling mean (handles +/-180 wrap via sin/cos)
       25-frame delta (0.5s window)
       >= 45 degree threshold (XY-projection corrected, tuned on Kimmich)
       leading-edge count (1 sustained rotation = 1 event)
  -> compute_awi() = scan_count / phase_minutes
```

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Frame rate | 50 fps | TF15 spec |
| Scan window | 25 frames (0.5 s) | Sports-science literature |
| Threshold | 45 degrees | XY-projection compresses 3D angles; tuned on Kimmich |
| Smooth window | 11 frames (0.22 s) | Suppresses single-frame tracking artefacts |

### PQI (Pressure Quality Index)

Press frame definition: player's pelvis within 5 m of ball carrier for >= 10 consecutive frames (0.2 s at 50 fps).

```
PQI = 0.40 x orientation + 0.30 x stance + 0.30 x proximity
```

| Sub-score | Weight | Formula | Peak |
|-----------|--------|---------|------|
| Orientation | 40% | `max(0, 100 - (angle_to_carrier / 90) x 100)` | 100 when facing carrier directly |
| Stance | 30% | `100 x exp(-0.5 x ((knee_flex - 130) / 25)^2)` | 100 at 130 degree knee flexion |
| Proximity | 30% | `max(0, 100 x (1 - distance_m / 5.0))` | 100 at 0 m, 0 at >= 5 m |

---

## AWS Services

| Service | Purpose |
|---------|---------|
| **S3** | TF15 Parquet match data (read-only) + pipeline output storage |
| **SageMaker Processing** | Parallel AWI + PQI compute (10 jobs, `ml.m5.xlarge`, ~15-20 min) |
| **ECR** | Docker image registry for the Processing container |
| **Bedrock** | AI narrative generation (`eu.amazon.nova-lite-v1:0`) |

---

## Data

Five Bundesliga matches provided by DFL via the hackathon S3 bucket. No match data or derived result files are stored in this repository. All CSVs under `results/` are pipeline outputs and are gitignored.

| Match | ID | Size |
|-------|----|------|
| FC Bayern Munchen vs Hamburger SV | FCB-HSV | ~4.4 GB |
| Borussia Dortmund vs VfB Stuttgart | BVB-VFB | ~4.1 GB |
| Eintracht Frankfurt vs FC Bayern | SGE-FCB | ~3.7 GB |
| Eintracht Frankfurt vs Union Berlin | SGE-FCU | ~4.2 GB |
| Union Berlin vs FC Bayern | FCU-FCB | ~3.6 GB |

---

## Deployment: Streamlit Community Cloud

The dashboard can be hosted on [Streamlit Community Cloud](https://share.streamlit.io) without any local setup:

1. Deploy from the GitHub repo, main file: `dashboard/app.py`
2. In app settings, add secrets (`.streamlit/secrets.toml` format):

```toml
[aws]
bucket = "hackathon-data-603974305500"
results_prefix = "results"
aws_access_key_id = "AKIA..."
aws_secret_access_key = "..."
region_name = "eu-central-1"
```

The S3 data loader (`src/s3_data_loader.py`) reads these secrets automatically. See `.streamlit/secrets.toml.example` for the template.

---

## Windows Users

Install [WSL2](https://learn.microsoft.com/en-us/windows/wsl/install) and enable Docker Desktop's WSL2 backend. Run all commands inside a WSL2 terminal. The shell scripts (`.sh`) require bash.

Alternative without WSL2:

```powershell
# Create venv
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Load .env manually
Get-Content .env | Where-Object { $_ -notmatch '^#' -and $_ -match '=' } | ForEach-Object {
    $k, $v = $_ -split '=', 2
    [System.Environment]::SetEnvironmentVariable($k.Trim(), $v.Trim(), 'Process')
}

# SSO login
aws sso login --profile $env:AWS_PROFILE

# Launch dashboard
streamlit run dashboard/app.py --server.port 8501
```
