# Cross-Domain Benchmark Integration — Design Spec

**Project:** Football Body Intelligence Platform (AWS World Sports Innovation Cup 2026)
**Date:** 2026-04-22
**Scope:** Option B — two new `src/` modules + Benchmark tab in dashboard

---

## Context

The competition requirement asks for benchmarking and adaptation of best practices from other sports or industries that leverage 3D data. The notebook `notebooks/benchmark_cross_sport.ipynb` is already fully written and maps each AWI/PQI sub-score to a validated external system. It imports from two modules that do not yet exist:

- `src/benchmark_reference.py`
- `src/benchmark_report.py`

This spec covers implementing those two modules, wiring them into a new Benchmark tab in `dashboard/app.py`, and adding offline tests.

---

## 1. `src/benchmark_reference.py`

### Purpose

Parameterised reference distributions for six external tracking systems. No runtime data fetches — all parameters are encoded from published summary statistics with full citations in docstrings.

### `ReferenceDistribution` dataclass

```python
@dataclass
class ReferenceDistribution:
    system: str           # e.g. "Aviation HUD research"
    sport: str            # e.g. "Aviation"
    metric_name: str      # e.g. "Head-scan rate"
    metric_type: str      # e.g. "AWI" — used for lookup
    mean: float           # population mean
    std: float            # population std
    elite_mean: float     # top-quartile mean
    unit: str             # e.g. "scans/min" or "0-100"
    citation: str         # APA short-form
```

### `REFERENCE_CATALOGUE` — 6 entries

| `metric_type` | System | Sport | Mean | Std | Elite mean | Unit |
|---|---|---|---|---|---|---|
| `AWI` | Aviation HUD research | Aviation | 24 | 5 | 34 | scans/min |
| `PQI_orientation` | NBA Second Spectrum | Basketball | 62 | 18 | 82 | 0–100 |
| `PQI_stance` | Tennis Hawk-Eye | Tennis | 58 | 16 | 78 | 0–100 |
| `PQI_proximity` | NFL Next Gen Stats | American Football | 55 | 20 | 78 | 0–100 |
| `PQI_composite` | Rugby Catapult/Pulsar | Rugby | 52 | 17 | 74 | 0–100 |
| `PQI_stance` (2nd) | Medical gait analysis | Biomechanics | 60 | 14 | 80 | 0–100 |

### Public API

```python
def get_all_references() -> list[ReferenceDistribution]: ...
def get_references_for_metric(metric_type: str) -> list[ReferenceDistribution]: ...
def percentile_in_reference(value: float, ref: ReferenceDistribution) -> float: ...
    # Normal CDF lookup; returns 0-100
def sample_reference_distribution(
    ref: ReferenceDistribution, n: int, cohort: str = "population"
) -> np.ndarray: ...
    # cohort="population" -> N(mean, std)
    # cohort="elite"      -> N(elite_mean, std * 0.6)
def build_comparison_table(
    awi_value: float | None = None,
    pqi_value: float | None = None,
    orientation_value: float | None = None,
    stance_value: float | None = None,
    proximity_value: float | None = None,
) -> list[dict]: ...
    # Returns list of dicts with keys:
    # system, sport, metric_type, pct_vs_population, pct_vs_elite, unit
    # Only includes entries where the corresponding value is not None
```

### Error handling

- `get_references_for_metric` returns an empty list (not an error) for unknown metric types
- `percentile_in_reference` raises `ValueError` for NaN or infinite input
- `sample_reference_distribution` raises `ValueError` for unknown `cohort` values or `n < 1`

---

## 2. `src/benchmark_report.py`

### Purpose

Narrative reference entries for the four detailed cross-sport comparisons (NBA, NFL, Cricket, Industrial), plus a summary table generator that produces the DataFrame shown in the notebook's Section 8.

### `BENCHMARK_REFERENCES` dict

Four keys: `nba_second_spectrum`, `nfl_next_gen_stats`, `cricket_hawk_eye`, `industrial_motion_capture`.

Each value is a flat dict with these fields:

```python
{
    "system": str,                # full system name
    "sport": str,
    "metric_analog": str,         # which AWI/PQI component this maps to
    "measurement_approach": str,  # how the external system measures its signal
    "temporal_resolution": str,   # e.g. "25 Hz optical"
    "application_domain": str,    # injury risk / pressing quality / etc.
    "citation": str,              # APA short-form with DOI
}
```

### `generate_benchmark_summary() -> pd.DataFrame`

Returns a 6-row DataFrame (one per reference system from the notebook's opening table) with columns:

`System`, `Sport`, `Maps to`, `Shared technique`, `Key distinction`, `Citation`

No external data required. All content is encoded in the function.

---

## 3. Dashboard — Benchmark tab

### Integration point

Fifth tab added to `dashboard/app.py` after "Leaderboard". Tab label: `"Benchmark"`.

### Data loading

```python
# Primary: combined_full.csv (has awi_per_minute + mean_pqi + sub-score columns)
# Fallback: merge awi_full.csv + pqi_full.csv on join keys:
#   ["jersey", "team", "name", "position", "match_id", "phase_label"]
```

Fail silently with a `st.warning` if neither source is available — don't crash other tabs.

Sub-score columns (`orientation_mean`, `stance_mean`, `proximity_mean`) may be absent from the fallback merge. Block 3 (PQI sub-scores panel) should check for column presence and show `st.info("Sub-score columns not available")` per missing panel rather than raising.

### Layout (four blocks, top to bottom)

**Block 1 — Reference Catalogue table**
Styled `st.dataframe` of all 6 references: Sport, System, Maps to, Population mean ± std, Elite mean, Unit. No interactivity needed.

**Block 2 — AWI vs Aviation**
Plotly histogram overlay:
- Grey: aviation reference distribution (sampled from `benchmark_reference`)
- Blue (`#38BDF8`): Bundesliga AWI values (filtered `> 0`)
- Vertical lines: aviation elite mean (gold dashed), Bundesliga mean (blue solid)
- Annotation: Bundesliga mean percentile in aviation reference

**Block 3 — PQI Sub-Scores vs Cross-Domain References**
Three Plotly histograms in `st.columns(3)`:
- Orientation (`orientation_mean`) vs NBA Second Spectrum
- Stance (`stance_mean`) vs Tennis Hawk-Eye
- Proximity (`proximity_mean`) vs NFL Next Gen Stats

Each panel: grey reference distribution + coloured Bundesliga distribution + gold elite mean line.

**Block 4 — Pre-Decision Scan Burst**
Plotly bar chart, 5 bars:
1. Aviation low workload — 14 scans/min
2. Aviation high workload — 24 scans/min
3. Aviation pre-decision — 34 scans/min (gold)
4. Football baseline AWI — computed from data mean
5. Football pre-pass AWI — baseline × 1.57 (green)

Annotated with +57% (football) and +42% (aviation) arrows.

### Styling

All charts use the existing dashboard dark palette: background `#0B0F1A`, surface `#111827`, border `#1E293B`. Colour constants already defined in `app.py` — reuse them.

### No new dependencies

`benchmark_reference.py` supplies all reference data. Only `plotly`, `pandas`, `numpy`, and `streamlit` are used — all already in `requirements.txt`.

---

## 4. Tests — `tests/test_benchmark.py`

All offline. No S3, no results CSV.

### `benchmark_reference.py` coverage

| Test | Assertion |
|---|---|
| `get_all_references()` returns 6 entries | `len == 6` |
| Each entry has positive mean, std, elite_mean | `> 0` for all |
| `get_references_for_metric('AWI')` | returns 1 entry, `unit == 'scans/min'` |
| `get_references_for_metric('unknown')` | returns `[]` |
| `percentile_in_reference(mean_value, ref)` | returns value in [49, 51] |
| `percentile_in_reference(elite_mean, ref)` | returns > 75 |
| `sample_reference_distribution(ref, n=100)` | array length 100, values in plausible range |
| `sample_reference_distribution(ref, n=50, cohort='elite')` | mean > population mean |
| `sample_reference_distribution` invalid cohort | raises `ValueError` |
| `build_comparison_table(awi_value=24.0, pqi_value=60.0, ...)` | list of dicts, each with `pct_vs_population` in [0, 100] and non-empty `system` |
| `build_comparison_table()` (all None) | returns `[]` |

### `benchmark_report.py` coverage

| Test | Assertion |
|---|---|
| `BENCHMARK_REFERENCES` has exactly 4 keys | `len == 4` |
| Each entry contains required fields | `system`, `sport`, `metric_analog`, `citation` all present and non-empty |
| `generate_benchmark_summary()` shape | 6 rows, columns include `System`, `Sport`, `Maps to`, `Key distinction`, `Citation` |
| `generate_benchmark_summary()` no NaN cells | `df.isnull().any().any() == False` |

---

## File changes

| File | Action |
|---|---|
| `src/benchmark_reference.py` | Create |
| `src/benchmark_report.py` | Create |
| `tests/test_benchmark.py` | Create |
| `dashboard/app.py` | Add Benchmark tab |
| `CLAUDE.md` | Add benchmark modules to Architecture section |

---

## Out of scope (Option C — later)

- Updating `submission/executive_summary_slides.md`
- Updating `submission/prfaq.md`
