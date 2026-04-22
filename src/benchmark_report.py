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
                "PQI stance targets knee-flexion pressing posture at 130 degrees"
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
