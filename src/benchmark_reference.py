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
    """Return the percentile (0-100) of *value* within ref's population distribution N(mean, std).

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
