"""
tests/test_pqi.py
Unit tests for PQI calculator (task 1.5).
All tests use synthetic numpy arrays / DataFrames — no S3 access.
"""

import math
import numpy as np
import pandas as pd
import pytest

from src.pqi_calculator import (
    PQI_WEIGHTS,
    compute_knee_flexion,
    compute_orientation_score,
    compute_pqi,
    compute_proximity_score,
    compute_stance_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scalar(fn, *args):
    """Call a vectorized function with scalar inputs and return a Python float."""
    arrays = [np.array([a]) for a in args]
    return float(fn(*arrays)[0])


# ---------------------------------------------------------------------------
# Orientation score tests
# ---------------------------------------------------------------------------


def test_orientation_score_perfect_alignment():
    """body_yaw == ball_dir → score == 100.0"""
    score = _scalar(compute_orientation_score, 45.0, 45.0)
    assert score == pytest.approx(100.0)


def test_orientation_score_90_degree_misalignment():
    """90° difference → score == 0.0"""
    score = _scalar(compute_orientation_score, 0.0, 90.0)
    assert score == pytest.approx(0.0)


def test_orientation_score_180_degree_misalignment():
    """180° difference → score == 0.0 (clamped, not negative)"""
    score = _scalar(compute_orientation_score, 0.0, 180.0)
    assert score == pytest.approx(0.0)


def test_orientation_score_circular_wraparound():
    """+170° vs -170° → 20° circular diff → ≈ 77.78"""
    score = _scalar(compute_orientation_score, 170.0, -170.0)
    expected = max(0.0, 100.0 - (20.0 / 90.0) * 100.0)
    assert score == pytest.approx(expected, rel=1e-4)


def test_orientation_score_bounds():
    """Arbitrary inputs stay in [0, 100]."""
    rng = np.random.default_rng(42)
    yaw = rng.uniform(-360, 360, 500)
    ball = rng.uniform(-360, 360, 500)
    scores = compute_orientation_score(yaw, ball)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 100.0)


def test_orientation_score_symmetry():
    """compute_orientation_score(a, b) == compute_orientation_score(b, a)"""
    a = np.array([30.0, -45.0, 170.0])
    b = np.array([-10.0, 120.0, -170.0])
    np.testing.assert_allclose(
        compute_orientation_score(a, b),
        compute_orientation_score(b, a),
    )


# ---------------------------------------------------------------------------
# Stance score tests
# ---------------------------------------------------------------------------


def test_stance_score_peak_at_130():
    """Knee flexion == 130° → score == 100.0"""
    score = _scalar(compute_stance_score, 130.0)
    assert score == pytest.approx(100.0)


def test_stance_score_one_sigma():
    """130 ± 25° → score ≈ 60.65 (exp(-0.5))"""
    expected = 100.0 * math.exp(-0.5)
    for angle in (130.0 + 25.0, 130.0 - 25.0):
        score = _scalar(compute_stance_score, angle)
        assert score == pytest.approx(expected, rel=1e-4)


def test_stance_score_bounds():
    """Inputs in [0, 180] stay in [0, 100]."""
    angles = np.linspace(0, 180, 200)
    scores = compute_stance_score(angles)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 100.0)


def test_stance_score_zero_degrees():
    """0° knee flexion → very low score (far from peak)."""
    score = _scalar(compute_stance_score, 0.0)
    assert 0.0 <= score < 10.0


def test_stance_score_180_degrees():
    """180° knee flexion → low score (far from peak at 130°)."""
    score = _scalar(compute_stance_score, 180.0)
    # At 180°, delta = 50°, score = 100 * exp(-0.5 * (50/25)^2) ≈ 13.5
    expected = 100.0 * math.exp(-0.5 * ((180.0 - 130.0) / 25.0) ** 2)
    assert score == pytest.approx(expected, rel=1e-4)
    assert score < 20.0  # well below peak


# ---------------------------------------------------------------------------
# Proximity score tests
# ---------------------------------------------------------------------------


def test_proximity_score_at_zero_distance():
    """0 m → score == 100.0"""
    score = _scalar(compute_proximity_score, 0.0)
    assert score == pytest.approx(100.0)


def test_proximity_score_at_max_distance():
    """5.0 m → score == 0.0"""
    score = _scalar(compute_proximity_score, 5.0)
    assert score == pytest.approx(0.0)


def test_proximity_score_beyond_max():
    """7.0 m → score == 0.0 (clamped)"""
    score = _scalar(compute_proximity_score, 7.0)
    assert score == pytest.approx(0.0)


def test_proximity_score_midpoint():
    """2.5 m → score == 50.0"""
    score = _scalar(compute_proximity_score, 2.5)
    assert score == pytest.approx(50.0)


def test_proximity_score_bounds():
    """Non-negative distances stay in [0, 100]."""
    distances = np.linspace(0, 10, 300)
    scores = compute_proximity_score(distances)
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 100.0)


# ---------------------------------------------------------------------------
# PQI composite tests
# ---------------------------------------------------------------------------


def test_pqi_weights_sum_to_one():
    """PQI_WEIGHTS values must sum to exactly 1.0."""
    assert sum(PQI_WEIGHTS.values()) == pytest.approx(1.0)


def test_pqi_perfect_scores():
    """All sub-scores == 100 → PQI == 100.0"""
    pqi = _scalar(compute_pqi, 100.0, 100.0, 100.0)
    assert pqi == pytest.approx(100.0)


def test_pqi_zero_scores():
    """All sub-scores == 0 → PQI == 0.0"""
    pqi = _scalar(compute_pqi, 0.0, 0.0, 0.0)
    assert pqi == pytest.approx(0.0)


def test_pqi_bounds():
    """Outputs stay in [0, 100] for sub-scores in [0, 100]."""
    rng = np.random.default_rng(7)
    o = rng.uniform(0, 100, 300)
    s = rng.uniform(0, 100, 300)
    p = rng.uniform(0, 100, 300)
    pqi = compute_pqi(o, s, p)
    assert np.all(pqi >= 0.0)
    assert np.all(pqi <= 100.0)


def test_pqi_weighted_correctly():
    """Known inputs produce expected weighted sum."""
    orientation = np.array([80.0])
    stance = np.array([60.0])
    proximity = np.array([40.0])
    expected = 0.40 * 80.0 + 0.30 * 60.0 + 0.30 * 40.0
    pqi = compute_pqi(orientation, stance, proximity)
    assert float(pqi[0]) == pytest.approx(expected)


def test_pqi_orientation_dominates():
    """Orientation weight (0.40) is the largest single weight."""
    assert PQI_WEIGHTS["orientation"] > PQI_WEIGHTS["stance"]
    assert PQI_WEIGHTS["orientation"] > PQI_WEIGHTS["proximity"]


# ---------------------------------------------------------------------------
# Knee flexion tests
# ---------------------------------------------------------------------------


def test_knee_flexion_straight_leg():
    """Collinear hip-knee-ankle → 180°"""
    # hip above knee, ankle below knee (all on y-axis)
    angle = compute_knee_flexion(
        knee_x=np.array([0.0]),
        knee_y=np.array([0.0]),
        hip_x=np.array([0.0]),
        hip_y=np.array([1.0]),
        ankle_x=np.array([0.0]),
        ankle_y=np.array([-1.0]),
    )
    assert float(angle[0]) == pytest.approx(180.0, abs=1e-6)


def test_knee_flexion_right_angle():
    """Hip directly above knee, ankle directly to the right → 90°"""
    angle = compute_knee_flexion(
        knee_x=np.array([0.0]),
        knee_y=np.array([0.0]),
        hip_x=np.array([0.0]),
        hip_y=np.array([1.0]),
        ankle_x=np.array([1.0]),
        ankle_y=np.array([0.0]),
    )
    assert float(angle[0]) == pytest.approx(90.0, abs=1e-6)


def test_knee_flexion_degenerate():
    """Zero-length vector (knee == hip) → NaN, no exception raised."""
    angle = compute_knee_flexion(
        knee_x=np.array([0.0]),
        knee_y=np.array([0.0]),
        hip_x=np.array([0.0]),   # same as knee → zero-length vector
        hip_y=np.array([0.0]),
        ankle_x=np.array([1.0]),
        ankle_y=np.array([0.0]),
    )
    assert np.isnan(float(angle[0]))


def test_knee_flexion_45_degrees():
    """45° angle between vectors."""
    angle = compute_knee_flexion(
        knee_x=np.array([0.0]),
        knee_y=np.array([0.0]),
        hip_x=np.array([0.0]),
        hip_y=np.array([1.0]),
        ankle_x=np.array([1.0]),
        ankle_y=np.array([1.0]),
    )
    assert float(angle[0]) == pytest.approx(45.0, abs=1e-5)


# ---------------------------------------------------------------------------
# Press frame identification tests (require pressure_pipeline.py)
# ---------------------------------------------------------------------------

try:
    from src.pressure_pipeline import identify_press_frames as _identify_press_frames
    # Check if the function is actually implemented (not just a stub)
    import inspect as _inspect
    _src = _inspect.getsource(_identify_press_frames)
    _PIPELINE_AVAILABLE = "NotImplementedError" not in _src
except ImportError:
    _PIPELINE_AVAILABLE = False


def _make_player_df(frame_numbers, pelvis_x, pelvis_y):
    return pd.DataFrame(
        {"frame_number": frame_numbers, "pelvis_x": pelvis_x, "pelvis_y": pelvis_y}
    )


@pytest.mark.skipif(
    not _PIPELINE_AVAILABLE,
    reason="src/pressure_pipeline.py not available",
)
def test_identify_press_frames_within_range():
    """15 consecutive frames with distance < 5 m → all 15 flagged."""
    n = 15
    frames = np.arange(n)
    # presser at origin, carrier at (200, 0) cm → 2 m apart
    presser = _make_player_df(frames, np.zeros(n), np.zeros(n))
    carrier = _make_player_df(frames, np.full(n, 200.0), np.zeros(n))
    result = _identify_press_frames(presser, carrier, min_run_length=10)
    assert result.sum() == n


@pytest.mark.skipif(
    not _PIPELINE_AVAILABLE,
    reason="src/pressure_pipeline.py not available",
)
def test_identify_press_frames_short_run_excluded():
    """Run of only 5 close frames (< min_run_length=10) → none flagged."""
    n = 5
    frames = np.arange(n)
    presser = _make_player_df(frames, np.zeros(n), np.zeros(n))
    carrier = _make_player_df(frames, np.full(n, 200.0), np.zeros(n))
    result = _identify_press_frames(presser, carrier, min_run_length=10)
    assert result.sum() == 0
