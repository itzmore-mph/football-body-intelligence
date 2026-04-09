"""
tests/test_awi.py
Unit tests for awi_calculator.py

Key behaviors tested:
  detect_scans:
    - Uses minimum angular difference (circular, handles ±180° wraparound)
    - Returns correct columns
    - Correctly flags / does not flag based on threshold

  compute_awi:
    - Counts discrete scan *events* (leading edges of is_scan runs), not frames
    - Filters correctly to the given phase
    - Returns all required keys
"""

import pandas as pd
import pytest
from src.awi_calculator import detect_scans, compute_awi


def make_head_angles(yaw_values: list, start_frame: int = 0) -> pd.DataFrame:
    """Helper to build a head_angles DataFrame from a list of yaw values."""
    return pd.DataFrame({
        "frame_number": range(start_frame, start_frame + len(yaw_values)),
        "head_yaw_deg": yaw_values,
    })


class TestDetectScans:

    def test_returns_correct_columns(self):
        df = make_head_angles([0.0] * 50)
        result = detect_scans(df)
        assert list(result.columns) == ["frame_number", "head_yaw_deg", "delta_yaw", "is_scan"]

    def test_flags_scan_above_threshold(self):
        # Flat for 25 frames, then jump of 45 degrees (no wraparound).
        # The 11-frame centered smoothing window blunts the transition at frame 25
        # (smoothed ≈ 24.5°, below threshold). By frame 30 the smoothed signal
        # has fully settled at 45° and the delta vs the 0° plateau is flagged.
        yaw = [0.0] * 25 + [45.0] * 25
        df = make_head_angles(yaw)
        result = detect_scans(df, window_frames=25, threshold_deg=30.0)
        assert result.loc[result["frame_number"] >= 30, "is_scan"].any(), (
            "A 45° yaw jump must be flagged as a scan once the smoothing window settles"
        )

    def test_does_not_flag_below_threshold(self):
        # Jump of only 10 degrees, below threshold
        yaw = [0.0] * 25 + [10.0] * 25
        df = make_head_angles(yaw)
        result = detect_scans(df, window_frames=25, threshold_deg=30.0)
        assert result["is_scan"].sum() == 0

    def test_no_scan_in_flat_series(self):
        df = make_head_angles([90.0] * 100)
        result = detect_scans(df)
        assert result["is_scan"].sum() == 0

    def test_circular_wraparound_small_turn(self):
        # +170° to -170° is a 20° turn (crossing ±180° boundary).
        # Without circular fix: abs(170 - (-170)) = 340° -> wrongly flagged.
        # With circular fix:    min_angular_diff = 20°  -> correctly not flagged.
        yaw = [170.0] * 25 + [-170.0] * 25
        df = make_head_angles(yaw)
        result = detect_scans(df, window_frames=25, threshold_deg=30.0)
        assert result["is_scan"].sum() == 0, (
            "A 20° turn crossing ±180° must NOT be flagged as a scan"
        )

    def test_circular_wraparound_large_turn(self):
        # +170° to +100° is a 70° turn – should be flagged.
        yaw = [170.0] * 25 + [100.0] * 25
        df = make_head_angles(yaw)
        result = detect_scans(df, window_frames=25, threshold_deg=30.0)
        assert result.loc[result["frame_number"] == 25, "is_scan"].values[0] == True

    def test_delta_yaw_correct_at_boundary(self):
        # +175° to -175° = 10° turn. delta_yaw should be ~10, not ~350.
        # smooth_window=1 bypasses smoothing so we test the raw circular math directly.
        yaw = [175.0] * 25 + [-175.0] * 25
        df = make_head_angles(yaw)
        result = detect_scans(df, window_frames=25, threshold_deg=30.0, smooth_window=1)
        delta_at_25 = result.loc[result["frame_number"] == 25, "delta_yaw"].values[0]
        assert delta_at_25 == pytest.approx(10.0, abs=0.01), (
            f"Expected ~10°, got {delta_at_25:.2f}°"
        )


class TestComputeAwi:

    def test_awi_counts_discrete_events_not_frames(self):
        # One continuous run of 10 True frames = 1 scan event (leading edge),
        # not 10. AWI = 1 scan / 1 minute = 1.0
        frames = list(range(0, 3000))
        is_scan = [True] * 10 + [False] * 2990
        scan_df = pd.DataFrame({
            "frame_number": frames,
            "head_yaw_deg": [0.0] * 3000,
            "delta_yaw": [0.0] * 3000,
            "is_scan": is_scan,
        })
        result = compute_awi(scan_df, phase_start=0, phase_end=2999, framerate=50)
        assert result["scan_count"] == 1, (
            "One run of consecutive True frames must count as 1 scan event"
        )
        assert result["awi_per_minute"] == pytest.approx(1.0, rel=0.01)

    def test_multiple_distinct_events_counted_correctly(self):
        # Three separate True runs = 3 scan events
        is_scan = (
            [True] * 5 + [False] * 20 +   # event 1
            [True] * 8 + [False] * 20 +   # event 2
            [True] * 3 + [False] * 2944   # event 3
        )
        frames = list(range(0, 3000))
        scan_df = pd.DataFrame({
            "frame_number": frames,
            "head_yaw_deg": [0.0] * 3000,
            "delta_yaw": [0.0] * 3000,
            "is_scan": is_scan,
        })
        result = compute_awi(scan_df, phase_start=0, phase_end=2999, framerate=50)
        assert result["scan_count"] == 3

    def test_filters_by_phase(self):
        # All frames are scan-flagged, but only frames 3000-5999 are in phase.
        # The first frame in the phase (3000) is a leading edge -> scan_count = 1.
        frames = list(range(0, 6000))
        is_scan = [True] * 6000
        scan_df = pd.DataFrame({
            "frame_number": frames,
            "head_yaw_deg": [0.0] * 6000,
            "delta_yaw": [35.0] * 6000,
            "is_scan": is_scan,
        })
        result = compute_awi(scan_df, phase_start=3000, phase_end=5999, framerate=50)
        # After phase filter, phase_df starts at frame 3000 (True) with no prior
        # frame in phase_df, so shift gives False -> frame 3000 is a leading edge.
        assert result["scan_count"] == 1

    def test_no_scans_returns_zero(self):
        scan_df = pd.DataFrame({
            "frame_number": list(range(0, 3000)),
            "head_yaw_deg": [0.0] * 3000,
            "delta_yaw": [0.0] * 3000,
            "is_scan": [False] * 3000,
        })
        result = compute_awi(scan_df, phase_start=0, phase_end=2999, framerate=50)
        assert result["scan_count"] == 0
        assert result["awi_per_minute"] == 0.0

    def test_returns_required_keys(self):
        scan_df = pd.DataFrame({
            "frame_number": [0, 1],
            "head_yaw_deg": [0.0, 0.0],
            "delta_yaw": [0.0, 0.0],
            "is_scan": [False, False],
        })
        result = compute_awi(scan_df, phase_start=0, phase_end=1)
        assert all(k in result for k in [
            "phase_start", "phase_end", "total_minutes", "scan_count", "awi_per_minute"
        ])
