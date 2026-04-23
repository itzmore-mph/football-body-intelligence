"""
pqi_sensitivity.py
Grid scan over PQI weight combinations — rank-stability analysis.
No I/O, no S3 access, no side effects.
"""

from dataclasses import dataclass

import pandas as pd
from scipy.stats import spearmanr  # noqa: F401 (used in run_sensitivity)


@dataclass
class SensitivityResult:
    correlations: pd.DataFrame
    rank_deltas: pd.DataFrame
    baseline_ranking: pd.Series


BASELINE_WEIGHTS: dict[str, float] = {
    "orientation": 0.40,
    "stance": 0.30,
    "proximity": 0.30,
}


def _weight_combo_key(w: dict[str, float]) -> str:
    return f"o{w['orientation']:.2f}_s{w['stance']:.2f}_p{w['proximity']:.2f}"


def generate_weight_grid(step: float = 0.05) -> list[dict[str, float]]:
    """
    Return all weight triplets (w_orientation, w_stance, w_proximity) where each
    value is a multiple of ``step`` and the three values sum to 1.0.

    At step=0.05 this yields C(22, 2) = 231 combinations, including degenerate
    cases where one or two weights are zero.

    Args:
        step: Grid resolution. Must divide 1.0 exactly (e.g. 0.05, 0.10, 0.25).

    Returns:
        List of dicts with keys "orientation", "stance", "proximity".
    """
    n = round(1.0 / step)
    combos: list[dict[str, float]] = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            combos.append({
                "orientation": round(i * step, 10),
                "stance": round(j * step, 10),
                "proximity": round(k * step, 10),
            })
    return combos
