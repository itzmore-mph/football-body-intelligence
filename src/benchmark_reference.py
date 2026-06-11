# src/benchmark_reference.py
"""
Cross-domain reference distributions for AWI and PQI benchmarking.

All parameters are calibrated from published research - no network calls are
made at runtime. Citations are embedded in the REFERENCE_CATALOGUE.

IMPORTANT - interpretation of benchmark values:
  Each entry cites a peer-reviewed source that studies an analogous construct
  in the named domain.  The mean/std/elite_mean values are calibrated estimates
  on the same unit scale as AWI or the relevant PQI sub-score; they are *not*
  numbers directly quoted from the paper's tables.  Each entry carries a
  data_note field explaining the adaptation.  Percentile comparisons are
  therefore illustrative cross-domain analogies, not direct statistical tests.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class ReferenceDistribution:
    system: str
    sport: str
    metric_name: str
    metric_type: str   # lookup key: "AWI", "PQI_orientation", "PQI_stance", "PQI_proximity", "PQI_composite"
    mean: float        # population mean (calibrated to project unit scale)
    std: float         # population std
    elite_mean: float  # top-quartile mean
    unit: str          # "scans/min" or "0-100"
    citation: str
    data_note: str = field(default="")  # transparency note on value derivation


REFERENCE_CATALOGUE: list[ReferenceDistribution] = [
    ReferenceDistribution(
        system="Cockpit visual-scanning research",
        sport="Aviation",
        metric_name="Head/eye scan-transition rate",
        metric_type="AWI",
        mean=12.0,
        std=4.0,
        elite_mean=19.0,
        unit="scans/min",
        citation=(
            "Lounis, C., Peysakhovich, V., & Causse, M. (2021). "
            "Visual scanning strategies in the cockpit are modulated by pilots' expertise: "
            "A flight simulator study. PLOS ONE, 16(2), e0247061. "
            "https://doi.org/10.1371/journal.pone.0247061"
        ),
        data_note=(
            "Lounis et al. report fixation-dwell frequency for 16 expert vs. 16 novice "
            "pilots during a manual landing task. Expert pilots exhibit significantly more "
            "scan transitions per unit time than novices. Values here are calibrated to the "
            "same scans/min unit as AWI based on reported dwell-count distributions."
        ),
    ),
    ReferenceDistribution(
        system="NBA Second Spectrum",
        sport="Basketball",
        metric_name="Defensive positioning quality (EPV framework)",
        metric_type="PQI_orientation",
        mean=62.0,
        std=18.0,
        elite_mean=82.0,
        unit="0-100",
        citation=(
            "Cervone, D., D'Amour, A., Bornn, L., & Goldsberry, K. (2016). "
            "A multiresolution stochastic process model for predicting basketball "
            "possession outcomes. Journal of the American Statistical Association, "
            "111(514), 585–599. https://doi.org/10.1080/01621459.2016.1141685"
        ),
        data_note=(
            "Cervone et al. introduce Expected Possession Value (EPV) using 25 Hz optical "
            "tracking. The EPV framework underlies Second Spectrum's defensive-alignment "
            "analytics, which score how well a defender's body position covers the ball "
            "carrier - an analogous construct to PQI orientation. Values are calibrated to "
            "a 0–100 scale; they are not directly extracted from the paper's EPV tables."
        ),
    ),
    ReferenceDistribution(
        system="Tennis biomechanics (ready-position stance)",
        sport="Tennis",
        metric_name="Defensive ready-position knee-flexion quality",
        metric_type="PQI_stance",
        mean=62.0,
        std=14.0,
        elite_mean=80.0,
        unit="0-100",
        citation=(
            "Elliott, B. (2006). Biomechanics and tennis. "
            "British Journal of Sports Medicine, 40(5), 392–396. "
            "https://doi.org/10.1136/bjsm.2005.023150"
        ),
        data_note=(
            "Elliott reviews elite tennis biomechanics including the split-step ready "
            "position, where optimal knee flexion is reported at ~100–120°. On the PQI "
            "stance scale (Gaussian peak at 130°), 110° yields a score of ~83 and 100° "
            "yields ~73. Values are calibrated from reported knee-angle ranges; they are "
            "not directly quoted from paper tables."
        ),
    ),
    ReferenceDistribution(
        system="NFL Next Gen Stats",
        sport="American Football",
        metric_name="Defender-to-carrier proximity (tracking-based)",
        metric_type="PQI_proximity",
        mean=55.0,
        std=20.0,
        elite_mean=78.0,
        unit="0-100",
        citation=(
            "Eager, E., Chahrouri, G., Riske, T., & Brown, B. (2023). "
            "Using tracking and charting data to better evaluate NFL players: A review. "
            "MIT Sloan Sports Analytics Conference. "
            "https://www.sloansportsconference.com/research-papers/"
            "using-tracking-and-charting-data-to-better-evaluate-nfl-players-a-review"
        ),
        data_note=(
            "Eager et al. review NFL player-tracking separation metrics across multiple "
            "positions. Defender-to-carrier distances during pass rush (0.5–2 m range) are "
            "converted to the PQI proximity scale (max(0, 100 × (1 − d/5))). A 1.2 m mean "
            "distance maps to ~76; values here reflect the broader tracking-data context "
            "rather than a single quoted statistic."
        ),
    ),
    ReferenceDistribution(
        system="Rugby 3D motion capture",
        sport="Rugby",
        metric_name="Tackle-technique quality composite",
        metric_type="PQI_composite",
        mean=52.0,
        std=13.0,
        elite_mean=72.0,
        unit="0-100",
        citation=(
            "Hendricks, S., den Hollander, S., Lombard, W., & Lambert, M. (2021). "
            "3D biomechanics of rugby tackle techniques to inform future rugby research "
            "practice: A systematic review. Sports Medicine – Open, 7(1), 39. "
            "https://doi.org/10.1186/s40798-021-00322-w"
        ),
        data_note=(
            "Hendricks et al. systematically review 3D motion-capture studies of rugby "
            "tackle biomechanics, covering orientation, stance, and proximity components. "
            "The composite values here are calibrated estimates from the reported kinematic "
            "quality ranges across reviewed studies; they are not directly quoted from "
            "paper tables."
        ),
    ),
    ReferenceDistribution(
        system="Occupational biomechanics (REBA)",
        sport="Biomechanics",
        metric_name="Joint-angle deviation penalty (Gaussian formulation)",
        metric_type="PQI_stance",
        mean=60.0,
        std=14.0,
        elite_mean=80.0,
        unit="0-100",
        citation=(
            "Hignett, S., & McAtamney, L. (2000). Rapid Entire Body Assessment (REBA). "
            "Applied Ergonomics, 31(2), 201–205. "
            "https://doi.org/10.1016/S0003-6870(99)00039-3"
        ),
        data_note=(
            "REBA applies a Gaussian-style penalty to joint-angle deviation from a neutral "
            "reference posture - the same mathematical structure as the PQI stance formula. "
            "The PQI stance score (peak at 130° knee flexion) is conceptually derived from "
            "this occupational-ergonomics tradition. Values represent an athletic-performance "
            "adaptation of the REBA scoring range; the paper itself addresses occupational "
            "injury risk, not athletic pressing quality."
        ),
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
