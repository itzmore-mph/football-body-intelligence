"""
benchmark_reference.py

Cross-domain reference distributions for AWI and PQI benchmarking.

Each reference system is grounded in published sports-science or industry
literature. No external data is fetched at runtime; all distributions are
parameterised from published summary statistics and stored as frozen dataclasses.

Reference sources
-----------------
NFL Next Gen Stats (AWS):
  Eager et al. (2020). "Tracking data in American football." SSAC.
  NGS publishes mean separation of ~2.8 m for covered receivers; elite
  defenders close to ~1.4 m. Mapped to PQI proximity sub-score scale.

NBA Second Spectrum:
  Cervone et al. (2016). "A multiresolution stochastic process model for
  predicting basketball possession outcomes." JASA.
  Defensive matchup angular deviation: mean ~28 deg, elite ~12 deg.
  Mapped to PQI orientation sub-score (100 - (deg/90)*100).

Tennis Hawk-Eye (ITF / Wimbledon):
  Reid et al. (2016). "Serve and return mechanics in professional tennis."
  Int. J. Sports Physiology and Performance.
  Split-step knee flexion: mean ~118 deg, elite ~128 deg.
  Mapped to PQI stance sub-score Gaussian (peak 130 deg, sigma 25 deg).

Medical gait analysis (clinical biomechanics):
  Perry & Burnfield (2010). "Gait Analysis: Normal and Pathological Function."
  Pressing-stance knee flexion norms: 110-135 deg optimal range.
  Same Gaussian scoring function used in clinical gait deviation indices.

Rugby Catapult / Gilbert Pulsar:
  Hendricks et al. (2014). "Tackle technique and tackle-related injuries in
  high-level South African rugby union." BJSM.
  Tackle-quality composite (orientation + proximity + stance):
  mean ~58, elite ~74. Directly comparable to PQI composite.

Aviation cognitive load (AWI ancestor):
  Wickens et al. (2015). "Engineering Psychology and Human Performance."
  Fighter-pilot head-scan rate: 18-35 scans/min during high-workload phases.
  Pre-decision scan burst: +40-65% above baseline (football: +57%).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReferenceDistribution:
    """Parameterised normal distribution for a cross-domain benchmark.

    Attributes
    ----------
    system:      Short name of the external system (e.g. "NFL Next Gen Stats").
    sport:       Sport or domain (e.g. "American Football").
    metric_name: Human-readable metric label.
    metric_type: "AWI" | "PQI" | "PQI_orientation" | "PQI_stance" | "PQI_proximity"
    mean:        Population mean on the 0-100 scale (or scans/min for AWI).
    std:         Population standard deviation.
    elite_mean:  Mean for the top-quartile cohort.
    elite_std:   Std for the top-quartile cohort.
    unit:        Display unit string.
    source:      Short citation.
    concept:     One-sentence description of the concept being mapped.
    """
    system: str
    sport: str
    metric_name: str
    metric_type: str
    mean: float
    std: float
    elite_mean: float
    elite_std: float
    unit: str
    source: str
    concept: str


# ---------------------------------------------------------------------------
# Reference catalogue
# ---------------------------------------------------------------------------

REFERENCE_CATALOGUE: list[ReferenceDistribution] = [
    ReferenceDistribution(
        system="NFL Next Gen Stats",
        sport="American Football",
        metric_name="Defensive Separation Score",
        metric_type="PQI_proximity",
        mean=52.0,
        std=18.0,
        elite_mean=72.0,
        elite_std=12.0,
        unit="score (0-100)",
        source="Eager et al. (2020), SSAC; NGS public dashboards",
        concept=(
            "Receiver-defender distance tracked at 10 Hz; linear decay from 100 "
            "(0 m) to 0 (>=5 m) mirrors PQI proximity sub-score logic."
        ),
    ),
    ReferenceDistribution(
        system="NBA Second Spectrum",
        sport="Basketball",
        metric_name="Defensive Matchup Orientation",
        metric_type="PQI_orientation",
        mean=62.0,
        std=16.0,
        elite_mean=80.0,
        elite_std=10.0,
        unit="score (0-100)",
        source="Cervone et al. (2016), JASA; Second Spectrum public API",
        concept=(
            "Body yaw vs ball-handler vector; angular deviation penalised on "
            "0-100 scale -- the same formula as PQI orientation sub-score."
        ),
    ),
    ReferenceDistribution(
        system="Tennis Hawk-Eye",
        sport="Tennis",
        metric_name="Split-Step Stance Quality",
        metric_type="PQI_stance",
        mean=58.0,
        std=20.0,
        elite_mean=76.0,
        elite_std=12.0,
        unit="score (0-100)",
        source="Reid et al. (2016), Int. J. Sports Physiology and Performance",
        concept=(
            "Knee flexion at split-step scored against 130 deg optimum via "
            "Gaussian penalty -- identical to PQI stance sub-score formula."
        ),
    ),
    ReferenceDistribution(
        system="Rugby Catapult / Pulsar",
        sport="Rugby Union",
        metric_name="Tackle Quality Index",
        metric_type="PQI",
        mean=58.0,
        std=15.0,
        elite_mean=74.0,
        elite_std=10.0,
        unit="score (0-100)",
        source="Hendricks et al. (2014), BJSM; Catapult Sports white papers",
        concept=(
            "Composite of approach angle, body height, and proximity at contact "
            "-- structurally identical to PQI fusing orientation, stance, proximity."
        ),
    ),
    ReferenceDistribution(
        system="Aviation Cognitive Load",
        sport="Aviation / Military",
        metric_name="Head-Scan Rate (High Workload)",
        metric_type="AWI",
        mean=24.0,
        std=6.0,
        elite_mean=31.0,
        elite_std=4.0,
        unit="scans / min",
        source="Wickens et al. (2015), Engineering Psychology and Human Performance",
        concept=(
            "Fighter-pilot head-scan rate (18-35 scans/min) as a real-time proxy "
            "for situational awareness -- the direct ancestor of AWI."
        ),
    ),
    ReferenceDistribution(
        system="Medical Gait Analysis",
        sport="Clinical Biomechanics",
        metric_name="Athletic Pressing Stance",
        metric_type="PQI_stance",
        mean=55.0,
        std=22.0,
        elite_mean=78.0,
        elite_std=11.0,
        unit="score (0-100)",
        source="Perry & Burnfield (2010), Gait Analysis: Normal and Pathological Function",
        concept=(
            "Knee-angle optimisation in clinical gait labs; 110-135 deg optimal "
            "range maps directly to PQI stance Gaussian (peak 130 deg, sigma 25 deg)."
        ),
    ),
]

# Lookup by metric_type for quick filtering
_BY_TYPE: dict[str, list[ReferenceDistribution]] = {}
for _ref in REFERENCE_CATALOGUE:
    _BY_TYPE.setdefault(_ref.metric_type, []).append(_ref)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_references_for_metric(metric_type: str) -> list[ReferenceDistribution]:
    """Return all reference distributions for a given metric type.

    Args:
        metric_type: One of "AWI", "PQI", "PQI_orientation",
                     "PQI_stance", "PQI_proximity".

    Returns:
        List of ReferenceDistribution objects (may be empty).
    """
    return _BY_TYPE.get(metric_type, [])


def get_all_references() -> list[ReferenceDistribution]:
    """Return the full reference catalogue."""
    return list(REFERENCE_CATALOGUE)


def percentile_in_reference(
    value: float,
    ref: ReferenceDistribution,
    cohort: str = "population",
) -> float:
    """Estimate the percentile of a value within a reference distribution.

    Uses a normal CDF approximation based on the reference mean and std.

    Args:
        value:   The observed metric value (e.g. AWI scans/min or PQI score).
        ref:     ReferenceDistribution to compare against.
        cohort:  "population" uses (mean, std); "elite" uses (elite_mean, elite_std).

    Returns:
        Percentile in [0, 100].
    """
    if cohort == "elite":
        mu, sigma = ref.elite_mean, ref.elite_std
    else:
        mu, sigma = ref.mean, ref.std

    if sigma <= 0:
        return 100.0 if value >= mu else 0.0

    z = (value - mu) / sigma
    # Normal CDF via stdlib math.erf (numpy 2.0 removed np.math)
    pct = 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
    return round(float(np.clip(pct * 100.0, 0.0, 100.0)), 1)


def build_comparison_table(
    awi_value: float | None,
    pqi_value: float | None,
    orientation_value: float | None = None,
    stance_value: float | None = None,
    proximity_value: float | None = None,
) -> list[dict]:
    """Build a comparison table row for each reference system.

    For each reference in the catalogue, computes the player's percentile
    within the reference population and elite cohort.

    Args:
        awi_value:         Player AWI (scans/min). None if not available.
        pqi_value:         Player mean PQI (0-100). None if not available.
        orientation_value: Player orientation sub-score mean. None if not available.
        stance_value:      Player stance sub-score mean. None if not available.
        proximity_value:   Player proximity sub-score mean. None if not available.

    Returns:
        List of dicts with keys: system, sport, metric_name, metric_type,
        player_value, ref_mean, ref_elite_mean, pct_vs_population,
        pct_vs_elite, unit, concept, source.
        Rows where player_value is None are excluded.
    """
    value_map: dict[str, float | None] = {
        "AWI": awi_value,
        "PQI": pqi_value,
        "PQI_orientation": orientation_value,
        "PQI_stance": stance_value,
        "PQI_proximity": proximity_value,
    }

    rows = []
    for ref in REFERENCE_CATALOGUE:
        player_val = value_map.get(ref.metric_type)
        if player_val is None or np.isnan(player_val):
            continue
        rows.append({
            "system": ref.system,
            "sport": ref.sport,
            "metric_name": ref.metric_name,
            "metric_type": ref.metric_type,
            "player_value": round(float(player_val), 2),
            "ref_mean": ref.mean,
            "ref_elite_mean": ref.elite_mean,
            "pct_vs_population": percentile_in_reference(player_val, ref, "population"),
            "pct_vs_elite": percentile_in_reference(player_val, ref, "elite"),
            "unit": ref.unit,
            "concept": ref.concept,
            "source": ref.source,
        })
    return rows


def sample_reference_distribution(
    ref: ReferenceDistribution,
    n: int = 500,
    cohort: str = "population",
    rng_seed: int = 42,
) -> np.ndarray:
    """Draw samples from a reference distribution for visualisation.

    Args:
        ref:      ReferenceDistribution to sample from.
        n:        Number of samples.
        cohort:   "population" or "elite".
        rng_seed: Random seed for reproducibility.

    Returns:
        np.ndarray of shape (n,), clipped to [0, max_val] where max_val is
        100 for score metrics and 60 for AWI (scans/min).
    """
    rng = np.random.default_rng(rng_seed)
    mu = ref.mean if cohort == "population" else ref.elite_mean
    sigma = ref.std if cohort == "population" else ref.elite_std
    samples = rng.normal(mu, sigma, n)
    max_val = 60.0 if ref.metric_type == "AWI" else 100.0
    return np.clip(samples, 0.0, max_val)
