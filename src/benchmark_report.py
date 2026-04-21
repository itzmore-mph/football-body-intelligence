"""
benchmark_report.py

Human-readable Comparison_Table and structured citation dict for the
cross-sport benchmark feature (Track 3, AWS World Sports Innovation Cup 2026).

This module is intentionally thin: it owns the Comparison_Table schema and
the BENCHMARK_REFERENCES citation dict. The notebook owns narrative prose and
visualisations; the dashboard owns interactive display.

No network requests are made at import time or at function call time.
No em dash characters (U+2014) appear anywhere in this file.
"""

from __future__ import annotations

import pandas as pd


# ---------------------------------------------------------------------------
# Literature citations
# ---------------------------------------------------------------------------

BENCHMARK_REFERENCES: dict[str, dict] = {
    "nba_second_spectrum": {
        "sport": "Basketball",
        "author": "Cervone et al.",
        "year": 2016,
        "url": "https://doi.org/10.1080/01621459.2016.1141685",
    },
    "nfl_next_gen_stats": {
        "sport": "American Football",
        "author": "Eager et al.",
        "year": 2020,
        "url": "https://www.sloansportsconference.com/research-papers/tracking-data-in-american-football",
    },
    "cricket_hawk_eye": {
        "sport": "Cricket",
        "author": "Justham et al.",
        "year": 2008,
        "url": "https://doi.org/10.1177/1754337108090510",
    },
    "industrial_motion_capture": {
        "sport": "Occupational Biomechanics",
        "author": "Hignett and McAtamney",
        "year": 2000,
        "url": "https://doi.org/10.1016/S0003-6870(99)00056-5",
    },
}


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

_COMPARISON_ROWS: list[dict] = [
    {
        "Sport/Industry": "Basketball",
        "Data Source": "NBA Second Spectrum",
        "Signal Type": "Off-ball movement tracking",
        "Temporal Resolution": "25 Hz optical",
        "Closest AWI/PQI Analog": "AWI (decision-readiness)",
        "Key Difference": (
            "Proxy via movement patterns, not direct skeletal keypoints - "
            "AWI uses anatomical head-rotation detection"
        ),
    },
    {
        "Sport/Industry": "American Football",
        "Data Source": "NFL Next Gen Stats",
        "Signal Type": "Joint angles for injury risk and strain",
        "Temporal Resolution": "10 Hz GPS + optical",
        "Closest AWI/PQI Analog": "PQI (joint-angle composite)",
        "Key Difference": (
            "Same joint-angle signal as PQI but applied to injury risk "
            "assessment rather than pressing quality"
        ),
    },
    {
        "Sport/Industry": "Cricket",
        "Data Source": "Hawk-Eye bowling action analysis",
        "Signal Type": "3D body pose deviation from biomechanical optimum",
        "Temporal Resolution": "300 Hz high-speed optical",
        "Closest AWI/PQI Analog": "PQI stance sub-score",
        "Key Difference": (
            "Same deviation-from-optimum scoring technique as PQI stance "
            "sub-score, applied to bowling action rather than pressing posture"
        ),
    },
    {
        "Sport/Industry": "Occupational Biomechanics",
        "Data Source": "Industrial Motion Capture (REBA/RULA)",
        "Signal Type": "Ergonomic risk via joint-angle deviation",
        "Temporal Resolution": "Static posture snapshots",
        "Closest AWI/PQI Analog": "PQI stance sub-score (Gaussian penalty)",
        "Key Difference": (
            "Same Gaussian penalty function for joint-angle deviation as PQI "
            "stance sub-score, applied to occupational injury risk rather than "
            "athletic pressing quality"
        ),
    },
]


def generate_benchmark_summary() -> pd.DataFrame:
    """Return the cross-sport Comparison_Table as a pandas DataFrame.

    Pure function: no required arguments, no I/O, no network calls,
    no randomness. Always returns the same static DataFrame.

    Returns
    -------
    pd.DataFrame
        Exactly six columns in this order:
        - Sport/Industry
        - Data Source
        - Signal Type
        - Temporal Resolution
        - Closest AWI/PQI Analog
        - Key Difference

        At least 4 rows covering NBA Second Spectrum, NFL Next Gen Stats,
        Cricket Hawk-Eye, and Industrial Motion Capture (REBA/RULA).
    """
    columns = [
        "Sport/Industry",
        "Data Source",
        "Signal Type",
        "Temporal Resolution",
        "Closest AWI/PQI Analog",
        "Key Difference",
    ]
    return pd.DataFrame(_COMPARISON_ROWS, columns=columns)
