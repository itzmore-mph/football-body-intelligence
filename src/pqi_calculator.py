"""
pqi_calculator.py
Pure, stateless, vectorized computation of the Pressure Quality Index (PQI).
No I/O, no S3 access, no side effects.
"""

import numpy as np
import pandas as pd

# --- Constants ---
PQI_WEIGHTS = {"orientation": 0.40, "stance": 0.30, "proximity": 0.30}

STANCE_PEAK_DEG = 130.0
STANCE_SIGMA_DEG = 25.0
PROXIMITY_MAX_M = 5.0
CM_TO_M = 0.01  # TF15 Parquet position_x/y/z are in centimetres; divide by 100 to get metres


def compute_orientation_score(
    body_yaw_deg: np.ndarray,
    ball_direction_deg: np.ndarray,
) -> np.ndarray:
    """
    Compute orientation sub-score from body yaw and ball direction.

    orientation_score = max(0, 100 - (angle_to_target / 90) * 100)
    angle_to_target  = |((body_yaw - ball_direction) + 180) % 360 - 180|

    Uses the same circular delta formula as awi_calculator._angular_delta.

    Args:
        body_yaw_deg:      np.ndarray of body yaw angles in degrees.
        ball_direction_deg: np.ndarray of ball direction angles in degrees.

    Returns:
        np.ndarray of float64, shape (n,), values in [0, 100].
    """
    body_yaw_deg = np.asarray(body_yaw_deg, dtype=np.float64)
    ball_direction_deg = np.asarray(ball_direction_deg, dtype=np.float64)

    raw = body_yaw_deg - ball_direction_deg
    angle_to_target = np.abs(((raw + 180.0) % 360.0) - 180.0)
    score = 100.0 - (angle_to_target / 90.0) * 100.0
    return np.maximum(0.0, score)


def compute_stance_score(knee_flexion_deg: np.ndarray) -> np.ndarray:
    """
    Compute stance sub-score from knee flexion angle.

    Gaussian peak at 130°, sigma=25°:
    stance_score = 100 * exp(-0.5 * ((knee_flexion - 130) / 25)^2)

    Args:
        knee_flexion_deg: np.ndarray of knee flexion angles in degrees, range [0, 180].

    Returns:
        np.ndarray of float64, shape (n,), values in [0, 100].
    """
    knee_flexion_deg = np.asarray(knee_flexion_deg, dtype=np.float64)
    exponent = -0.5 * ((knee_flexion_deg - STANCE_PEAK_DEG) / STANCE_SIGMA_DEG) ** 2
    return 100.0 * np.exp(exponent)


def compute_proximity_score(distance_m: np.ndarray) -> np.ndarray:
    """
    Compute proximity sub-score from presser-to-ball-carrier distance.

    Linear decay from 100 at 0 m to 0 at 5 m:
    proximity_score = max(0, 100 * (1 - distance_m / 5.0))

    The caller is responsible for converting cm to metres before calling this
    function: distance_m = sqrt(dx^2 + dy^2) / 100.0

    Args:
        distance_m: np.ndarray of distances in metres (>= 0).

    Returns:
        np.ndarray of float64, shape (n,), values in [0, 100].
    """
    distance_m = np.asarray(distance_m, dtype=np.float64)
    return np.maximum(0.0, 100.0 * (1.0 - distance_m / PROXIMITY_MAX_M))


def compute_pqi(
    orientation_score: np.ndarray,
    stance_score: np.ndarray,
    proximity_score: np.ndarray,
) -> np.ndarray:
    """
    Compute composite Pressure Quality Index from three sub-scores.

    PQI = 0.40 * orientation_score + 0.30 * stance_score + 0.30 * proximity_score

    Weights sum to 1.0 (verified by PQI_WEIGHTS dict).

    Args:
        orientation_score: np.ndarray in [0, 100].
        stance_score:      np.ndarray in [0, 100].
        proximity_score:   np.ndarray in [0, 100].

    Returns:
        np.ndarray of float64, shape (n,), values in [0, 100].
    """
    orientation_score = np.asarray(orientation_score, dtype=np.float64)
    stance_score = np.asarray(stance_score, dtype=np.float64)
    proximity_score = np.asarray(proximity_score, dtype=np.float64)

    return (
        PQI_WEIGHTS["orientation"] * orientation_score
        + PQI_WEIGHTS["stance"] * stance_score
        + PQI_WEIGHTS["proximity"] * proximity_score
    )


def compute_knee_flexion(
    knee_x: np.ndarray,
    knee_y: np.ndarray,
    hip_x: np.ndarray,
    hip_y: np.ndarray,
    ankle_x: np.ndarray,
    ankle_y: np.ndarray,
) -> np.ndarray:
    """
    Compute knee flexion angle using the dot product of (knee→hip) and (knee→ankle) vectors.

    Uses law of cosines on 2D projected positions. Unit cancels in the dot/magnitude
    ratio, so no cm→m conversion is needed here.

    Args:
        knee_x, knee_y:   Knee joint 2D positions.
        hip_x, hip_y:     Hip joint 2D positions.
        ankle_x, ankle_y: Ankle joint 2D positions.

    Returns:
        np.ndarray of float64, shape (n,), degrees in [0, 180].
        NaN where either vector has zero length (degenerate frame).
    """
    knee_x = np.asarray(knee_x, dtype=np.float64)
    knee_y = np.asarray(knee_y, dtype=np.float64)
    hip_x = np.asarray(hip_x, dtype=np.float64)
    hip_y = np.asarray(hip_y, dtype=np.float64)
    ankle_x = np.asarray(ankle_x, dtype=np.float64)
    ankle_y = np.asarray(ankle_y, dtype=np.float64)

    # Vectors from knee to hip and knee to ankle
    v1_x = hip_x - knee_x
    v1_y = hip_y - knee_y
    v2_x = ankle_x - knee_x
    v2_y = ankle_y - knee_y

    dot = v1_x * v2_x + v1_y * v2_y
    mag1 = np.sqrt(v1_x ** 2 + v1_y ** 2)
    mag2 = np.sqrt(v2_x ** 2 + v2_y ** 2)

    denom = mag1 * mag2
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_angle = np.where(denom > 0, dot / denom, np.nan)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


def aggregate_pqi_for_player(
    pqi_series: pd.Series,
    orientation_series: pd.Series,
    stance_series: pd.Series,
    proximity_series: pd.Series,
    phase_start: int,
    phase_end: int,
) -> dict:
    """
    Aggregate frame-level PQI and sub-scores into a phase-level summary.

    Args:
        pqi_series:         Frame-level PQI values (press frames only).
        orientation_series: Frame-level orientation scores.
        stance_series:      Frame-level stance scores.
        proximity_series:   Frame-level proximity scores.
        phase_start:        Phase start frame number.
        phase_end:          Phase end frame number.

    Returns:
        Dict with keys: mean_pqi, median_pqi, std_pqi, n_press_frames,
                        orientation_mean, stance_mean, proximity_mean.
        All score fields are NaN when n_press_frames == 0.
    """
    n = int(pqi_series.notna().sum())

    if n == 0:
        return {
            "mean_pqi": float("nan"),
            "median_pqi": float("nan"),
            "std_pqi": float("nan"),
            "n_press_frames": 0,
            "orientation_mean": float("nan"),
            "stance_mean": float("nan"),
            "proximity_mean": float("nan"),
        }

    return {
        "mean_pqi": float(pqi_series.mean(skipna=True)),
        "median_pqi": float(pqi_series.median(skipna=True)),
        "std_pqi": float(pqi_series.std(skipna=True)),
        "n_press_frames": n,
        "orientation_mean": float(orientation_series.mean(skipna=True)),
        "stance_mean": float(stance_series.mean(skipna=True)),
        "proximity_mean": float(proximity_series.mean(skipna=True)),
    }
