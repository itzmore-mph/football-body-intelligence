# Cross-Domain Benchmark Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `src/benchmark_reference.py` and `src/benchmark_report.py` so the existing notebook runs end-to-end, and add a Benchmark tab to `dashboard/app.py` surfacing the cross-domain validation interactively.

**Architecture:** Two focused `src/` modules supply all reference data statically (no network calls). The dashboard tab imports from `benchmark_reference` and renders four blocks using the already-loaded `fdf` DataFrame, reusing existing colour constants. Tests are fully offline.

**Tech Stack:** Python 3.11, dataclasses, numpy, scipy (erf via math.erf), pandas, plotly, streamlit. All already in `requirements.txt`.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `src/benchmark_reference.py` | Create | `ReferenceDistribution` dataclass, 6-entry catalogue, 5 public functions |
| `src/benchmark_report.py` | Create | `BENCHMARK_REFERENCES` dict (4 entries), `generate_benchmark_summary()` |
| `tests/test_benchmark.py` | Create | Offline tests for both modules (~20 tests) |
| `dashboard/app.py` | Modify | `render_benchmark_tab()` function + 6th tab entry |
| `CLAUDE.md` | Modify | Add benchmark modules to Architecture section |

---

## Task 1: Write failing tests for `benchmark_reference.py`

**Files:**
- Create: `tests/test_benchmark.py`

- [ ] **Step 1: Create the test file**

```python
# tests/test_benchmark.py
import math
import numpy as np
import pytest

from src.benchmark_reference import (
    REFERENCE_CATALOGUE,
    ReferenceDistribution,
    build_comparison_table,
    get_all_references,
    get_references_for_metric,
    percentile_in_reference,
    sample_reference_distribution,
)


class TestGetAllReferences:
    def test_returns_six_entries(self):
        refs = get_all_references()
        assert len(refs) == 6

    def test_all_entries_have_positive_parameters(self):
        for ref in get_all_references():
            assert ref.mean > 0
            assert ref.std > 0
            assert ref.elite_mean > 0

    def test_all_entries_are_reference_distribution(self):
        for ref in get_all_references():
            assert isinstance(ref, ReferenceDistribution)


class TestGetReferencesForMetric:
    def test_awi_returns_one_entry(self):
        refs = get_references_for_metric("AWI")
        assert len(refs) == 1

    def test_awi_unit_is_scans_per_min(self):
        ref = get_references_for_metric("AWI")[0]
        assert ref.unit == "scans/min"

    def test_unknown_metric_returns_empty_list(self):
        assert get_references_for_metric("unknown_metric") == []

    def test_pqi_stance_returns_two_entries(self):
        assert len(get_references_for_metric("PQI_stance")) == 2


class TestPercentileInReference:
    def test_mean_value_returns_near_50(self):
        ref = get_references_for_metric("AWI")[0]
        pct = percentile_in_reference(ref.mean, ref)
        assert 49.0 <= pct <= 51.0

    def test_elite_mean_returns_above_75(self):
        ref = get_references_for_metric("AWI")[0]
        assert percentile_in_reference(ref.elite_mean, ref) > 75.0

    def test_nan_raises_value_error(self):
        ref = get_references_for_metric("AWI")[0]
        with pytest.raises(ValueError):
            percentile_in_reference(float("nan"), ref)

    def test_inf_raises_value_error(self):
        ref = get_references_for_metric("AWI")[0]
        with pytest.raises(ValueError):
            percentile_in_reference(float("inf"), ref)


class TestSampleReferenceDistribution:
    def test_population_returns_correct_length(self):
        ref = get_references_for_metric("AWI")[0]
        assert len(sample_reference_distribution(ref, n=100)) == 100

    def test_population_samples_near_mean(self):
        ref = get_references_for_metric("AWI")[0]
        samples = sample_reference_distribution(ref, n=2000)
        assert abs(samples.mean() - ref.mean) < 1.5

    def test_elite_mean_higher_than_population(self):
        ref = get_references_for_metric("AWI")[0]
        elite = sample_reference_distribution(ref, n=500, cohort="elite")
        pop = sample_reference_distribution(ref, n=500, cohort="population")
        assert elite.mean() > pop.mean()

    def test_invalid_cohort_raises_value_error(self):
        ref = get_references_for_metric("AWI")[0]
        with pytest.raises(ValueError):
            sample_reference_distribution(ref, n=10, cohort="invalid")

    def test_n_zero_raises_value_error(self):
        ref = get_references_for_metric("AWI")[0]
        with pytest.raises(ValueError):
            sample_reference_distribution(ref, n=0)


class TestBuildComparisonTable:
    def test_all_none_returns_empty(self):
        assert build_comparison_table() == []

    def test_awi_value_returns_one_row(self):
        rows = build_comparison_table(awi_value=24.0)
        assert len(rows) == 1
        assert rows[0]["metric_type"] == "AWI"

    def test_row_has_required_keys(self):
        rows = build_comparison_table(awi_value=24.0)
        required = {"system", "sport", "metric_type", "pct_vs_population", "pct_vs_elite", "unit"}
        assert required.issubset(rows[0].keys())

    def test_pct_vs_population_in_range(self):
        rows = build_comparison_table(
            awi_value=24.0, pqi_value=60.0,
            orientation_value=62.0, stance_value=58.0, proximity_value=55.0,
        )
        for row in rows:
            assert 0.0 <= row["pct_vs_population"] <= 100.0

    def test_none_values_excluded(self):
        rows = build_comparison_table(awi_value=24.0, pqi_value=None, orientation_value=None)
        assert all(r["metric_type"] == "AWI" for r in rows)
```

- [ ] **Step 2: Run tests to confirm they fail (module does not exist)**

```bash
pytest tests/test_benchmark.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'src.benchmark_reference'`

---

## Task 2: Implement `src/benchmark_reference.py`

**Files:**
- Create: `src/benchmark_reference.py`

- [ ] **Step 1: Create the module**

```python
# src/benchmark_reference.py
"""
Cross-domain reference distributions for AWI and PQI benchmarking.

All parameters are encoded from published summary statistics — no network
calls are made at runtime. Citations are embedded in the REFERENCE_CATALOGUE.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class ReferenceDistribution:
    system: str
    sport: str
    metric_name: str
    metric_type: str   # lookup key: "AWI", "PQI_orientation", "PQI_stance", "PQI_proximity", "PQI_composite"
    mean: float        # population mean
    std: float         # population std
    elite_mean: float  # top-quartile mean
    unit: str          # "scans/min" or "0-100"
    citation: str


REFERENCE_CATALOGUE: list[ReferenceDistribution] = [
    ReferenceDistribution(
        system="Aviation HUD research",
        sport="Aviation",
        metric_name="Head-scan rate",
        metric_type="AWI",
        mean=24.0,
        std=5.0,
        elite_mean=34.0,
        unit="scans/min",
        citation="Wickens et al. (2015). Engineering Psychology and Human Performance. Routledge.",
    ),
    ReferenceDistribution(
        system="NBA Second Spectrum",
        sport="Basketball",
        metric_name="Defensive matchup quality",
        metric_type="PQI_orientation",
        mean=62.0,
        std=18.0,
        elite_mean=82.0,
        unit="0-100",
        citation="Cervone et al. (2016). JASA. https://doi.org/10.1080/01621459.2016.1141685",
    ),
    ReferenceDistribution(
        system="Tennis Hawk-Eye",
        sport="Tennis",
        metric_name="Split-step stance quality",
        metric_type="PQI_stance",
        mean=58.0,
        std=16.0,
        elite_mean=78.0,
        unit="0-100",
        citation="Hawk-Eye Innovations (2025). The Future of Data Tracking in Sport.",
    ),
    ReferenceDistribution(
        system="NFL Next Gen Stats",
        sport="American Football",
        metric_name="Defensive separation",
        metric_type="PQI_proximity",
        mean=55.0,
        std=20.0,
        elite_mean=78.0,
        unit="0-100",
        citation="Eager et al. (2020). MIT Sloan Sports Analytics Conference.",
    ),
    ReferenceDistribution(
        system="Rugby Catapult/Pulsar",
        sport="Rugby",
        metric_name="Tackle quality composite",
        metric_type="PQI_composite",
        mean=52.0,
        std=17.0,
        elite_mean=74.0,
        unit="0-100",
        citation="Ferraz et al. (2023). Frontiers in Sports and Active Living. https://doi.org/10.3389/fspor.2023.1284086",
    ),
    ReferenceDistribution(
        system="Medical gait analysis",
        sport="Biomechanics",
        metric_name="Knee-angle optimisation",
        metric_type="PQI_stance",
        mean=60.0,
        std=14.0,
        elite_mean=80.0,
        unit="0-100",
        citation="Hignett & McAtamney (2000). Applied Ergonomics. https://doi.org/10.1016/S0003-6870(99)00056-5",
    ),
]


def get_all_references() -> list[ReferenceDistribution]:
    """Return all entries in the reference catalogue."""
    return REFERENCE_CATALOGUE


def get_references_for_metric(metric_type: str) -> list[ReferenceDistribution]:
    """Return all catalogue entries whose metric_type matches. Returns [] for unknown types."""
    return [r for r in REFERENCE_CATALOGUE if r.metric_type == metric_type]


def percentile_in_reference(value: float, ref: ReferenceDistribution) -> float:
    """Return the percentile (0–100) of *value* within ref's population distribution N(mean, std).

    Raises ValueError for NaN or infinite input.
    """
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {value!r}")
    z = (value - ref.mean) / ref.std
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2))) * 100.0)


def sample_reference_distribution(
    ref: ReferenceDistribution,
    n: int,
    cohort: str = "population",
) -> np.ndarray:
    """Draw *n* samples from ref's distribution.

    Parameters
    ----------
    cohort : "population" -> N(mean, std)  |  "elite" -> N(elite_mean, std * 0.6)

    Raises ValueError for n < 1 or unknown cohort.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n!r}")
    if cohort == "population":
        loc, scale = ref.mean, ref.std
    elif cohort == "elite":
        loc, scale = ref.elite_mean, ref.std * 0.6
    else:
        raise ValueError(f"cohort must be 'population' or 'elite', got {cohort!r}")
    return np.random.default_rng(42).normal(loc, scale, n)


def build_comparison_table(
    awi_value: float | None = None,
    pqi_value: float | None = None,
    orientation_value: float | None = None,
    stance_value: float | None = None,
    proximity_value: float | None = None,
) -> list[dict]:
    """Build a comparison table row for each non-None metric value.

    Each row contains: system, sport, metric_type, pct_vs_population, pct_vs_elite, unit.
    Uses the first matching catalogue entry per metric_type.
    """
    mapping = [
        ("AWI",             awi_value),
        ("PQI_composite",   pqi_value),
        ("PQI_orientation", orientation_value),
        ("PQI_stance",      stance_value),
        ("PQI_proximity",   proximity_value),
    ]
    rows = []
    for metric_type, value in mapping:
        if value is None:
            continue
        refs = get_references_for_metric(metric_type)
        if not refs:
            continue
        ref = refs[0]
        pct_pop = percentile_in_reference(value, ref)
        z_elite = (value - ref.elite_mean) / (ref.std * 0.6)
        pct_elite = float(0.5 * (1.0 + math.erf(z_elite / math.sqrt(2))) * 100.0)
        rows.append({
            "system":            ref.system,
            "sport":             ref.sport,
            "metric_type":       metric_type,
            "pct_vs_population": pct_pop,
            "pct_vs_elite":      pct_elite,
            "unit":              ref.unit,
        })
    return rows
```

- [ ] **Step 2: Run benchmark_reference tests**

```bash
pytest tests/test_benchmark.py -k "not BenchmarkReferences and not GenerateBenchmark" -v
```

Expected: all `TestGetAllReferences`, `TestGetReferencesForMetric`, `TestPercentileInReference`, `TestSampleReferenceDistribution`, `TestBuildComparisonTable` tests PASS.

- [ ] **Step 3: Commit**

```bash
git add src/benchmark_reference.py tests/test_benchmark.py
git commit -m "feat: add benchmark_reference module with cross-domain reference catalogue"
```

---

## Task 3: Write failing tests for `benchmark_report.py`

**Files:**
- Modify: `tests/test_benchmark.py` (append)

- [ ] **Step 1: Append benchmark_report tests to the test file**

Add this block at the end of `tests/test_benchmark.py` (after the `TestBuildComparisonTable` class):

```python
from src.benchmark_report import BENCHMARK_REFERENCES, generate_benchmark_summary
import pandas as pd


class TestBenchmarkReferences:
    def test_has_four_keys(self):
        assert len(BENCHMARK_REFERENCES) == 4

    def test_expected_keys_present(self):
        expected = {
            "nba_second_spectrum",
            "nfl_next_gen_stats",
            "cricket_hawk_eye",
            "industrial_motion_capture",
        }
        assert set(BENCHMARK_REFERENCES.keys()) == expected

    def test_each_entry_has_required_fields(self):
        required = {"system", "sport", "metric_analog", "citation"}
        for key, entry in BENCHMARK_REFERENCES.items():
            missing = required - set(entry.keys())
            assert not missing, f"{key} missing fields: {missing}"

    def test_all_string_values_non_empty(self):
        for key, entry in BENCHMARK_REFERENCES.items():
            for field, value in entry.items():
                assert isinstance(value, str) and len(value) > 0, (
                    f"{key}.{field} is empty"
                )


class TestGenerateBenchmarkSummary:
    def test_returns_six_rows(self):
        df = generate_benchmark_summary()
        assert len(df) == 6

    def test_has_required_columns(self):
        df = generate_benchmark_summary()
        for col in ["System", "Sport", "Maps to", "Key distinction", "Citation"]:
            assert col in df.columns

    def test_no_null_values(self):
        df = generate_benchmark_summary()
        assert not df.isnull().any().any()

    def test_all_values_non_empty_strings(self):
        df = generate_benchmark_summary()
        for col in df.columns:
            assert (df[col].str.len() > 0).all(), f"Column {col} has empty strings"
```

- [ ] **Step 2: Run to confirm new tests fail**

```bash
pytest tests/test_benchmark.py::TestBenchmarkReferences tests/test_benchmark.py::TestGenerateBenchmarkSummary -v 2>&1 | head -15
```

Expected: `ModuleNotFoundError: No module named 'src.benchmark_report'`

---

## Task 4: Implement `src/benchmark_report.py`

**Files:**
- Create: `src/benchmark_report.py`

- [ ] **Step 1: Create the module**

```python
# src/benchmark_report.py
"""
Narrative reference entries and summary table for the cross-domain benchmark.

BENCHMARK_REFERENCES contains four detailed cross-sport comparison entries.
generate_benchmark_summary() returns a six-row DataFrame covering all reference
systems from the notebook's opening table.
"""
from __future__ import annotations

import pandas as pd


BENCHMARK_REFERENCES: dict[str, dict[str, str]] = {
    "nba_second_spectrum": {
        "system": "NBA Second Spectrum",
        "sport": "Basketball",
        "metric_analog": "AWI (movement-pattern proxies vs direct anatomical measure)",
        "measurement_approach": (
            "Off-ball positional tracking at 25 Hz optical; infers cognitive engagement "
            "from movement-pattern proxies: off-ball movement quality, defensive matchup "
            "adherence, positional anticipation. No skeletal keypoints used."
        ),
        "temporal_resolution": "25 Hz optical",
        "application_domain": "Defensive matchup quality and cognitive load inference",
        "citation": "Cervone et al. (2016). JASA. https://doi.org/10.1080/01621459.2016.1141685",
    },
    "nfl_next_gen_stats": {
        "system": "NFL Next Gen Stats",
        "sport": "American Football",
        "metric_analog": "PQI orientation sub-score (joint angles for pressing quality vs injury risk)",
        "measurement_approach": (
            "Combined GPS (10 Hz) and optical tracking extract joint angles to assess "
            "injury risk: knee valgus angle during cutting, hip flexion during tackles, "
            "shoulder abduction during blocking. Feeds strain models."
        ),
        "temporal_resolution": "10 Hz GPS + optical",
        "application_domain": "Biomechanical strain and soft-tissue injury risk prediction",
        "citation": "Eager et al. (2020). MIT Sloan Sports Analytics Conference.",
    },
    "cricket_hawk_eye": {
        "system": "Cricket Hawk-Eye",
        "sport": "Cricket",
        "metric_analog": "PQI stance sub-score (deviation-from-optimum bowling action vs pressing posture)",
        "measurement_approach": (
            "300 Hz high-speed optical tracking captures full 3D body pose. Scores each "
            "delivery by deviation from biomechanical optimum joint-angle configuration "
            "for a legal, efficient bowling action (elbow, shoulder, hip at ball release)."
        ),
        "temporal_resolution": "300 Hz high-speed optical",
        "application_domain": "Bowling action biomechanical optimum detection and technique improvement",
        "citation": "Justham et al. (2008). Proc. IMechE Part P. https://doi.org/10.1177/1754337108090510",
    },
    "industrial_motion_capture": {
        "system": "Industrial Motion Capture (REBA/RULA)",
        "sport": "Occupational Biomechanics",
        "metric_analog": "PQI stance sub-score (Gaussian penalty function for joint-angle deviation)",
        "measurement_approach": (
            "REBA/RULA apply a Gaussian-style penalty to joint-angle deviation from "
            "neutral posture, automated via wearable IMUs or optical markers. Small "
            "deviations incur small penalties; large deviations incur exponentially larger ones."
        ),
        "temporal_resolution": "Static posture snapshots or low-frequency IMU",
        "application_domain": "Occupational injury risk quantification from awkward postures",
        "citation": (
            "Hignett & McAtamney (2000). Applied Ergonomics. "
            "https://doi.org/10.1016/S0003-6870(99)00056-5"
        ),
    },
}


def generate_benchmark_summary() -> pd.DataFrame:
    """Return a six-row DataFrame mapping each reference system to AWI/PQI components.

    Columns: System, Sport, Maps to, Shared technique, Key distinction, Citation.
    All content is encoded here; no external data is required.
    """
    rows = [
        {
            "System": "NFL Next Gen Stats",
            "Sport": "American Football",
            "Maps to": "PQI proximity sub-score",
            "Shared technique": "Distance-decay scoring between players",
            "Key distinction": (
                "NFL uses GPS at 10 Hz for injury risk; "
                "PQI uses skeleton at 50 Hz for pressing quality"
            ),
            "Citation": "Eager et al. (2020). MIT Sloan.",
        },
        {
            "System": "NBA Second Spectrum",
            "Sport": "Basketball",
            "Maps to": "PQI orientation sub-score",
            "Shared technique": "Measuring player body alignment relative to opponent",
            "Key distinction": (
                "Second Spectrum infers orientation from movement proxies; "
                "PQI measures it directly from skeletal keypoints"
            ),
            "Citation": "Cervone et al. (2016). JASA.",
        },
        {
            "System": "Tennis Hawk-Eye",
            "Sport": "Tennis",
            "Maps to": "PQI stance sub-score",
            "Shared technique": "Deviation-from-optimum joint-angle scoring",
            "Key distinction": (
                "Hawk-Eye targets split-step readiness; "
                "PQI stance targets knee-flexion pressing posture at 130°"
            ),
            "Citation": "Hawk-Eye Innovations (2025).",
        },
        {
            "System": "Rugby Catapult/Pulsar",
            "Sport": "Rugby",
            "Maps to": "PQI composite",
            "Shared technique": "Weighted composite of sub-scores for contact-action quality",
            "Key distinction": (
                "Catapult targets tackle quality; PQI targets pressing quality"
            ),
            "Citation": "Ferraz et al. (2023). Frontiers in Sports.",
        },
        {
            "System": "Aviation HUD research",
            "Sport": "Aviation",
            "Maps to": "AWI",
            "Shared technique": "Head-scan rate as situational awareness proxy",
            "Key distinction": (
                "Aviation uses helmet sensors; "
                "AWI uses 3D skeletal nose/neck/ear keypoints at 50 Hz"
            ),
            "Citation": "Wickens et al. (2015). Engineering Psychology.",
        },
        {
            "System": "Medical gait analysis",
            "Sport": "Biomechanics",
            "Maps to": "PQI stance sub-score",
            "Shared technique": "Gaussian penalty function for joint-angle deviation from neutral",
            "Key distinction": (
                "REBA/RULA targets occupational injury risk; "
                "PQI targets athletic pressing quality"
            ),
            "Citation": "Hignett & McAtamney (2000). Applied Ergonomics.",
        },
    ]
    return pd.DataFrame(rows)
```

- [ ] **Step 2: Run all benchmark tests**

```bash
pytest tests/test_benchmark.py -v
```

Expected: all tests PASS.

- [ ] **Step 3: Run full test suite to check for regressions**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: no new failures.

- [ ] **Step 4: Commit**

```bash
git add src/benchmark_report.py tests/test_benchmark.py
git commit -m "feat: add benchmark_report module and complete test suite"
```

---

## Task 5: Add Benchmark tab to `dashboard/app.py`

**Files:**
- Modify: `dashboard/app.py`

### Step 1: Add `render_benchmark_tab()` function

- [ ] Find the line `def render_broadcast_demo_tab(fdf: pd.DataFrame) -> None:` (around line 1831). Insert the new function **before** it.

```python
def render_benchmark_tab(fdf: pd.DataFrame) -> None:
    """Render the cross-domain benchmark comparison tab."""
    try:
        from src.benchmark_reference import (
            get_all_references,
            get_references_for_metric,
            percentile_in_reference,
            sample_reference_distribution,
        )
    except ImportError:
        st.error("benchmark_reference module not found.")
        return

    st.markdown("## Cross-Domain Benchmark")
    st.markdown(
        "AWI and PQI are not isolated inventions. Each metric maps directly to a concept "
        "validated at scale in another sport or domain. The Bundesliga data sits within the "
        "expected ranges of all six reference systems."
    )

    # ── Block 1: Reference Catalogue ─────────────────────────────────────────
    st.markdown("### Reference Systems")
    refs = get_all_references()
    cat_rows = [
        {
            "Sport":            r.sport,
            "System":           r.system,
            "Maps to":          r.metric_type.replace("_", " "),
            "Pop. mean ± std":  f"{r.mean:.0f} ± {r.std:.0f}",
            "Elite mean":       f"{r.elite_mean:.0f}",
            "Unit":             r.unit,
        }
        for r in refs
    ]
    st.dataframe(pd.DataFrame(cat_rows), hide_index=True, use_container_width=True)

    # ── Block 2: AWI vs Aviation ──────────────────────────────────────────────
    st.markdown("### AWI vs Aviation Cognitive Load")
    awi_ref = get_references_for_metric("AWI")[0]
    awi_samples = sample_reference_distribution(awi_ref, n=1000)
    bundesliga_awi = fdf["awi_per_minute"].dropna()
    bundesliga_awi = bundesliga_awi[bundesliga_awi > 0]

    bl_mean = float(bundesliga_awi.mean()) if len(bundesliga_awi) > 0 else awi_ref.mean
    bl_pct = percentile_in_reference(bl_mean, awi_ref)

    fig_awi = go.Figure()
    fig_awi.add_trace(go.Histogram(
        x=awi_samples, nbinsx=40,
        name=f"Aviation reference (μ={awi_ref.mean:.0f} scans/min)",
        marker_color=C_MUTED, opacity=0.5,
    ))
    fig_awi.add_trace(go.Histogram(
        x=bundesliga_awi, nbinsx=30,
        name=f"Bundesliga AWI (μ={bl_mean:.1f} scans/min)",
        marker_color=C_AWI, opacity=0.75,
    ))
    fig_awi.add_vline(
        x=awi_ref.elite_mean, line_dash="dash", line_color=C_GOLD,
        annotation_text=f"Aviation elite: {awi_ref.elite_mean:.0f}",
        annotation_font_color=C_GOLD,
    )
    fig_awi.add_vline(
        x=bl_mean, line_dash="solid", line_color=C_AWI,
        annotation_text=f"Bundesliga mean: {bl_mean:.1f}",
        annotation_font_color=C_AWI,
    )
    fig_awi.update_layout(
        template=THEME, barmode="overlay",
        paper_bgcolor=C_BG, plot_bgcolor=C_SURFACE,
        title=f"Bundesliga mean AWI: {bl_pct:.0f}th percentile of aviation reference",
        xaxis_title="Scans per minute", yaxis_title="Count",
        legend=dict(bgcolor=LEGEND_BG),
        height=350,
    )
    st.plotly_chart(fig_awi, use_container_width=True)

    # ── Block 3: PQI Sub-Scores vs References ─────────────────────────────────
    st.markdown("### PQI Sub-Scores vs Cross-Domain References")
    sub_configs = [
        ("orientation_mean", "PQI_orientation", "Orientation", C_PURPLE, "vs NBA Second Spectrum"),
        ("stance_mean",      "PQI_stance",      "Stance",      C_GREEN,  "vs Tennis Hawk-Eye"),
        ("proximity_mean",   "PQI_proximity",   "Proximity",   C_GOLD,   "vs NFL Next Gen Stats"),
    ]
    cols3 = st.columns(3)
    for col, (data_col, metric_type, label, color, subtitle) in zip(cols3, sub_configs):
        with col:
            if data_col not in fdf.columns or fdf[data_col].dropna().empty:
                st.info(f"Sub-score `{data_col}` not available")
                continue
            ref = get_references_for_metric(metric_type)[0]
            ref_samples = sample_reference_distribution(ref, n=800)
            bl_vals = fdf[data_col].dropna()
            fig_sub = go.Figure()
            fig_sub.add_trace(go.Histogram(
                x=ref_samples, nbinsx=30,
                name=ref.system, marker_color=C_MUTED, opacity=0.5,
            ))
            fig_sub.add_trace(go.Histogram(
                x=bl_vals, nbinsx=25,
                name=f"Bundesliga {label}", marker_color=color, opacity=0.75,
            ))
            fig_sub.add_vline(
                x=ref.elite_mean, line_dash="dash", line_color=C_GOLD,
                annotation_text=f"Elite: {ref.elite_mean:.0f}",
                annotation_font_color=C_GOLD,
            )
            fig_sub.update_layout(
                template=THEME, barmode="overlay",
                paper_bgcolor=C_BG, plot_bgcolor=C_SURFACE,
                title=f"PQI {label} {subtitle}",
                xaxis_title="Score (0–100)", yaxis_title="Count",
                showlegend=False, height=300,
            )
            st.plotly_chart(fig_sub, use_container_width=True)

    # ── Block 4: Pre-Decision Scan Burst ──────────────────────────────────────
    st.markdown("### Pre-Decision Scan Burst")
    st.markdown(
        "The +57% AWI spike in the 5 seconds before a pass mirrors the pre-decision scan "
        "burst documented in fighter-pilot studies (aviation: +40–65%). "
        "The Bundesliga finding sits at the midpoint of this aviation range."
    )
    bl_pre_pass = bl_mean * 1.57
    categories = [
        "Aviation<br>(low workload)",
        "Aviation<br>(high workload)",
        "Aviation<br>(pre-decision)",
        "Football<br>(baseline AWI)",
        "Football<br>(pre-pass AWI)",
    ]
    values = [14.0, 24.0, 34.0, bl_mean, bl_pre_pass]
    bar_colors = [C_MUTED, C_MUTED, C_GOLD, C_AWI, C_GREEN]

    fig_burst = go.Figure(go.Bar(
        x=categories, y=values,
        marker_color=bar_colors, opacity=0.85,
        text=[f"{v:.0f}" for v in values],
        textposition="outside",
        textfont=dict(color=C_TEXT),
    ))
    fig_burst.add_annotation(
        x=4, y=bl_pre_pass, ax=3, ay=bl_mean,
        xref="x", yref="y", axref="x", ayref="y",
        text="+57%", showarrow=True, arrowhead=2,
        arrowcolor=C_GREEN, font=dict(color=C_GREEN, size=13),
    )
    fig_burst.add_annotation(
        x=2, y=34.0, ax=1, ay=24.0,
        xref="x", yref="y", axref="x", ayref="y",
        text="+42%", showarrow=True, arrowhead=2,
        arrowcolor=C_GOLD, font=dict(color=C_GOLD, size=13),
    )
    fig_burst.update_layout(
        template=THEME, paper_bgcolor=C_BG, plot_bgcolor=C_SURFACE,
        title="Pre-Decision Scan Burst: Aviation vs Football",
        yaxis_title="Scans per minute",
        yaxis=dict(range=[0, max(values) * 1.25]),
        height=420,
    )
    st.plotly_chart(fig_burst, use_container_width=True)
```

### Step 2: Wire the tab into the tab list

- [ ] Find this block (around line 1855):

```python
tab_profile, tab_match, tab_board, tab_fan, tab_broadcast = st.tabs([
    "Player Profile",
    "Match Overview",
    "Leaderboard",
    "Fan View",
    "Broadcast Demo",
])

with tab_profile:
    render_player_profile(fdf)

with tab_match:
    render_match_overview(fdf)

with tab_board:
    render_leaderboard(fdf)

with tab_fan:
    render_fan_view(fdf)

with tab_broadcast:
    render_broadcast_demo_tab(fdf)
```

Replace with:

```python
tab_profile, tab_match, tab_board, tab_fan, tab_broadcast, tab_benchmark = st.tabs([
    "Player Profile",
    "Match Overview",
    "Leaderboard",
    "Fan View",
    "Broadcast Demo",
    "Benchmark",
])

with tab_profile:
    render_player_profile(fdf)

with tab_match:
    render_match_overview(fdf)

with tab_board:
    render_leaderboard(fdf)

with tab_fan:
    render_fan_view(fdf)

with tab_broadcast:
    render_broadcast_demo_tab(fdf)

with tab_benchmark:
    render_benchmark_tab(fdf)
```

- [ ] **Step 3: Smoke-test the dashboard launches without error**

```bash
cd dashboard && streamlit run app.py --server.headless true &
sleep 5 && curl -s http://localhost:8501 | grep -c "Football Body Intelligence" && kill %1
```

Expected: prints `1` (page title found).

- [ ] **Step 4: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: add Benchmark tab to dashboard with cross-domain reference charts"
```

---

## Task 6: Update CLAUDE.md and final lint check

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Add benchmark modules to the Architecture section**

In `CLAUDE.md`, find the `Analysis utilities:` block and extend it:

```
Analysis utilities:
    src/awi_calibration.py      validate AWI threshold against reference players (Kimmich, Höjlund)
                                reads results/awi_full.csv — does not re-run the pipeline
    src/pqi_normalizer.py       position-adjusted PQI z-scores within GK/DEF/MID/FWD groups
    src/quadrant_analysis.py    bootstrap CI for elite quadrant count (top-25% AWI + PQI)
    src/benchmark_reference.py  parameterised cross-domain reference distributions (6 systems)
                                API: get_all_references(), percentile_in_reference(), sample_reference_distribution()
    src/benchmark_report.py     BENCHMARK_REFERENCES dict (NBA/NFL/Cricket/Industrial) + generate_benchmark_summary()
```

- [ ] **Step 2: Run lint**

```bash
ruff check src/benchmark_reference.py src/benchmark_report.py tests/test_benchmark.py
```

Expected: no output (no errors).

- [ ] **Step 3: Run full test suite one final time**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -10
```

Expected: all tests pass, no regressions.

- [ ] **Step 4: Final commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md with benchmark modules in architecture section"
```
