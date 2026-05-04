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
        "metric_analog": "AWI (movement-pattern proxies vs. direct skeletal head-rotation measure)",
        "measurement_approach": (
            "Off-ball positional tracking at 25 Hz optical; models Expected Possession "
            "Value (EPV) via a multiresolution stochastic process. Defensive-alignment "
            "quality is inferred from movement proxies (positional anticipation, matchup "
            "adherence). No skeletal keypoints used; AWI measures head-rotation directly "
            "from 3D nose/neck/ear keypoints at 50 Hz."
        ),
        "temporal_resolution": "25 Hz optical",
        "application_domain": "Possession outcome prediction and defensive positioning quality",
        "citation": (
            "Cervone, D., D'Amour, A., Bornn, L., & Goldsberry, K. (2016). "
            "A multiresolution stochastic process model for predicting basketball "
            "possession outcomes. Journal of the American Statistical Association, "
            "111(514), 585–599. https://doi.org/10.1080/01621459.2016.1141685"
        ),
    },
    "nfl_next_gen_stats": {
        "system": "NFL Next Gen Stats",
        "sport": "American Football",
        "metric_analog": "PQI proximity sub-score (distance-decay scoring, defender-to-carrier)",
        "measurement_approach": (
            "Player-tracking and charting data combined to derive separation metrics "
            "between ball carriers and defenders. Proximity is the primary factor in "
            "pass-rush and run-defense evaluation, using distance-decay functions "
            "comparable to the PQI proximity formula (max(0, 100 × (1 − d/5)))."
        ),
        "temporal_resolution": "10 Hz GPS + optical",
        "application_domain": "Multi-position player evaluation using tracking and charting data",
        "citation": (
            "Eager, E., Chahrouri, G., Riske, T., & Brown, B. (2023). "
            "Using tracking and charting data to better evaluate NFL players: A review. "
            "MIT Sloan Sports Analytics Conference. "
            "https://www.sloansportsconference.com/research-papers/"
            "using-tracking-and-charting-data-to-better-evaluate-nfl-players-a-review"
        ),
    },
    "cricket_hawk_eye": {
        "system": "Cricket Hawk-Eye",
        "sport": "Cricket",
        "metric_analog": "PQI stance sub-score (deviation-from-optimum joint-angle scoring)",
        "measurement_approach": (
            "High-speed optical tracking captures full 3D body pose of the bowler. "
            "Each delivery is scored by the deviation of key joint angles (elbow, shoulder, "
            "hip at ball release) from a biomechanically optimal configuration, penalising "
            "deviations that reduce efficiency or legality - the same penalty structure as "
            "the PQI stance Gaussian formula."
        ),
        "temporal_resolution": "High-speed optical (multi-camera)",
        "application_domain": "Bowling-action optimum detection and coaching feedback",
        "citation": (
            "Justham, L., West, A., & Cork, A. (2008). "
            "Quantification and characterisation of cricket bowling technique for the "
            "development of a novel training system. "
            "Proceedings of the Institution of Mechanical Engineers, Part P: "
            "Journal of Sports Engineering and Technology, 222(P1), 15–26. "
            "https://doi.org/10.1177/1754337108090510"
        ),
    },
    "industrial_motion_capture": {
        "system": "Occupational Biomechanics (REBA)",
        "sport": "Occupational Biomechanics",
        "metric_analog": "PQI stance sub-score (Gaussian penalty function for joint-angle deviation)",
        "measurement_approach": (
            "REBA (Rapid Entire Body Assessment) applies a structured penalty to joint-angle "
            "deviation from a neutral reference posture, via wearable IMUs or optical markers. "
            "Small deviations incur small penalties; large deviations incur exponentially "
            "larger penalties - the direct conceptual predecessor of the PQI stance formula."
        ),
        "temporal_resolution": "Static posture snapshots or low-frequency IMU",
        "application_domain": "Occupational injury risk quantification from awkward postures",
        "citation": (
            "Hignett, S., & McAtamney, L. (2000). Rapid Entire Body Assessment (REBA). "
            "Applied Ergonomics, 31(2), 201–205. "
            "https://doi.org/10.1016/S0003-6870(99)00039-3"
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
            "System": "Tennis biomechanics (ready-position stance)",
            "Sport": "Tennis",
            "Maps to": "PQI stance sub-score",
            "Shared technique": "Deviation-from-optimum joint-angle scoring",
            "Key distinction": (
                "Elliott (2006) targets split-step knee flexion (~100–120°) for readiness; "
                "PQI stance targets pressing posture, Gaussian-peaked at 130° knee flexion"
            ),
            "Citation": "Elliott (2006). British Journal of Sports Medicine. https://doi.org/10.1136/bjsm.2005.023150",
        },
        {
            "System": "Rugby 3D motion capture",
            "Sport": "Rugby",
            "Maps to": "PQI composite",
            "Shared technique": "Multi-component kinematic quality scoring for contact actions",
            "Key distinction": (
                "Hendricks et al. (2021) review tackle biomechanics (orientation, stance, "
                "proximity); PQI applies the same three components to football pressing"
            ),
            "Citation": "Hendricks et al. (2021). Sports Medicine – Open. https://doi.org/10.1186/s40798-021-00322-w",
        },
        {
            "System": "Cockpit visual-scanning research",
            "Sport": "Aviation",
            "Maps to": "AWI",
            "Shared technique": "Discrete scan-transition rate as situational awareness proxy",
            "Key distinction": (
                "Lounis et al. (2021) measure fixation-dwell frequency via eye-tracker; "
                "AWI counts head-rotation onsets from 3D skeletal nose/neck/ear keypoints at 50 Hz"
            ),
            "Citation": "Lounis et al. (2021). PLOS ONE. https://doi.org/10.1371/journal.pone.0247061",
        },
        {
            "System": "Occupational Biomechanics (REBA)",
            "Sport": "Biomechanics",
            "Maps to": "PQI stance sub-score",
            "Shared technique": "Gaussian penalty function for joint-angle deviation from neutral",
            "Key distinction": (
                "REBA targets occupational injury risk from awkward postures; "
                "PQI stance adapts the same Gaussian penalty structure for athletic pressing quality"
            ),
            "Citation": "Hignett & McAtamney (2000). Applied Ergonomics. https://doi.org/10.1016/S0003-6870(99)00039-3",
        },
    ]
    return pd.DataFrame(rows)
