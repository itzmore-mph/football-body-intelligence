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
