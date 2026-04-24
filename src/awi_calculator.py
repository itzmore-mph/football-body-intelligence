"""
awi_calculator.py
Computes the Awareness Index (AWI) from TRACAB 3D skeleton data.
AWI = scanning events per minute, based on rapid head direction changes.
"""

from math import degrees, atan2
import numpy as np
import pandas as pd
from src.angle_utils import circular_diff
from src.skeleton_parser import extract_head_angles

# Constants
FRAMERATE = 50
SCAN_WINDOW_FRAMES = 25       # 0.5 second window
SCAN_THRESHOLD_DEG = 45.0     # XY-projected head angles are compressed vs. true 3D rotation;
                              # 45° empirically separates purposeful scans from postural sway


def _angular_delta(a: "pd.Series", b: "pd.Series") -> "pd.Series":
    """Compute the minimum angular difference between two yaw series (degrees).

    Thin wrapper preserved for backward compatibility. Prefer
    :func:`src.angle_utils.circular_diff` for new code.
    """
    return circular_diff(a, b)


def _smooth_yaw_circular(series: pd.Series, window: int = 11) -> pd.Series:
    """Circular mean smoothing for yaw angles.

    Standard rolling mean breaks at the ±180° boundary (e.g., mean of +179°
    and -179° should be ±180°, not 0°). This method decomposes into sin/cos,
    smooths each component, then reconstructs with atan2.

    Args:
        series: Raw head_yaw_deg series.
        window: Rolling window size in frames (default 11 = 0.22s at 50 Hz).

    Returns:
        Smoothed yaw series with same index.
    """
    rad = np.deg2rad(series)
    smooth_sin = (
        np.sin(rad).rolling(window=window, center=True, min_periods=1).mean()
    )
    smooth_cos = (
        np.cos(rad).rolling(window=window, center=True, min_periods=1).mean()
    )
    return pd.Series(
        np.rad2deg(np.arctan2(smooth_sin, smooth_cos)), index=series.index
    )


def detect_scans(
    head_angles_df: pd.DataFrame,
    window_frames: int = SCAN_WINDOW_FRAMES,
    threshold_deg: float = SCAN_THRESHOLD_DEG,
    smooth_window: int = 11,
) -> pd.DataFrame:
    """
    Detect scanning events from head yaw angle time series.

    Pipeline:
      1. Apply circular smoothing (11-frame rolling = 0.22 s at 50 Hz) to suppress
         tracking outliers while preserving genuine scans (>= 0.3 s duration).
      2. Compute angular delta over `window_frames` (0.5 s at 50 Hz = 25 frames).
      3. Flag frames where delta >= threshold_deg as scan events.

    Uses minimum angular difference (handles ±180° wraparound) so a head
    turn from +170° to -170° is correctly measured as 20°, not 340°.

    Args:
        head_angles_df: DataFrame with [frame_number, head_yaw_deg]
        window_frames:  Number of frames to look back for delta computation (default 25)
        threshold_deg:  Minimum yaw change in degrees to count as a scan (default 45°)
        smooth_window:  Circular rolling mean window in frames (default 11 = 0.22 s at 50 Hz)

    Returns:
        DataFrame with [frame_number, head_yaw_deg, delta_yaw, is_scan]
    """
    df = head_angles_df.copy().reset_index(drop=True)

    # Step 1: circular smoothing eliminates single-frame noise spikes
    df["head_yaw_smooth"] = _smooth_yaw_circular(df["head_yaw_deg"], window=smooth_window)

    # Step 2: window-based delta (yaw[t] vs yaw[t - 0.5s])
    df["delta_yaw"] = _angular_delta(
        df["head_yaw_smooth"],
        df["head_yaw_smooth"].shift(window_frames),
    )
    df["is_scan"] = df["delta_yaw"].fillna(0) >= threshold_deg
    return df[["frame_number", "head_yaw_deg", "delta_yaw", "is_scan"]]


def compute_awi(
    scan_df: pd.DataFrame,
    phase_start: int,
    phase_end: int,
    framerate: int = FRAMERATE,
) -> dict:
    """
    Compute AWI for a given match phase.

    Counts discrete scan *events* (leading edges of is_scan=True runs),
    not individual flagged frames. A continuous head rotation lasting N frames
    counts as 1 scan event, not N.

    Args:
        scan_df:     DataFrame with [frame_number, is_scan]
        phase_start: Start frame of the phase (e.g. kickoff frame)
        phase_end:   End frame of the phase
        framerate:   Frames per second

    Returns:
        Dict with phase_start, phase_end, total_minutes, scan_count, awi_per_minute
    """
    phase_df = scan_df[
        (scan_df["frame_number"] >= phase_start) &
        (scan_df["frame_number"] <= phase_end)
    ].reset_index(drop=True)

    total_minutes = (phase_end - phase_start) / framerate / 60

    # Count leading edges: frames where is_scan flips from False to True.
    # This gives discrete scan events rather than frame counts.
    is_scan = phase_df["is_scan"]
    prev_scan = is_scan.shift(1, fill_value=False)
    scan_count = int((is_scan & ~prev_scan).sum())

    awi_per_minute = scan_count / total_minutes if total_minutes > 0 else 0.0

    return {
        "phase_start": phase_start,
        "phase_end": phase_end,
        "total_minutes": round(total_minutes, 2),
        "scan_count": scan_count,
        "awi_per_minute": round(awi_per_minute, 2),
    }


def compute_player_awi(
    df: pd.DataFrame,
    jersey_number: int,
    team: int,
    phase_start: int,
    phase_end: int,
) -> dict:
    """
    Orchestrates AWI computation for a single player in a match phase.

    Args:
        df:            Raw parquet DataFrame with skeleton data
        jersey_number: Player jersey number
        team:          Team encoding (1=Home, 0=Away, 3=Referee)
        phase_start:   Phase start frame
        phase_end:     Phase end frame

    Returns:
        AWI result dict including jersey_number and team
    """
    head_angles = extract_head_angles(df, jersey_number, team)
    scan_df = detect_scans(head_angles)
    result = compute_awi(scan_df, phase_start, phase_end)
    result["jersey_number"] = jersey_number
    result["team"] = team
    return result