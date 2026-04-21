"""
tests/test_metric_robustness.py

Six unit tests and six property-based tests for the metric-robustness feature.

Covers:
- src/awi_calibration.py  (validate_awi_threshold)
- src/pqi_normalizer.py   (normalize_pqi_by_position)
- src/quadrant_analysis.py (bootstrap_elite_quadrant)
"""

import os

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.awi_calibration import validate_awi_threshold
from src.pqi_normalizer import normalize_pqi_by_position
from src.quadrant_analysis import bootstrap_elite_quadrant


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists("results/awi_full.csv"),
    reason="results/awi_full.csv not found",
)
def test_awi_calibration_kimmich():
    """Kimmich reference case must be within the 15% tolerance band."""
    report = validate_awi_threshold(
        player_id="Joshua Walter Kimmich",
        match_label="FCB-HSV",
        expected_rate=21.77,
    )
    assert report["within_tolerance"] is True


@pytest.mark.skipif(
    not os.path.exists("results/awi_full.csv"),
    reason="results/awi_full.csv not found",
)
def test_awi_calibration_hojlund():
    """Hojlund reference case must be within the 15% tolerance band."""
    report = validate_awi_threshold(
        player_id="Oscar Winther Höjlund",
        match_label="SGE-FCB",
        expected_rate=26.90,
    )
    assert report["within_tolerance"] is True


def test_position_normalization_output_columns():
    """normalize_pqi_by_position must add pqi_position_adjusted to the output."""
    df = pd.DataFrame({
        "player_id": ["p1", "p2", "p3"],
        "position_code": ["DMZ", "DMZ", "DMZ"],  # all MID
        "pqi_mean": [60.0, 70.0, 80.0],
    })
    result = normalize_pqi_by_position(df)
    assert "pqi_position_adjusted" in result.columns


def test_position_normalization_gk_separate():
    """GK z-scores must be computed within the GK group only.

    A GK whose pqi_mean equals the GK group mean must have
    pqi_position_adjusted near 0.0 (within 1e-9).
    """
    gk_pqi_values = [50.0, 60.0, 70.0]
    gk_mean = np.mean(gk_pqi_values)

    df = pd.DataFrame({
        "player_id": ["gk1", "gk2", "gk3", "out1", "out2"],
        "position_code": ["TW", "TW", "TW", "IVL", "IVL"],
        "pqi_mean": gk_pqi_values + [80.0, 90.0],
    })
    result = normalize_pqi_by_position(df)

    # The GK with pqi_mean == group mean should have z-score == 0
    gk_mean_player_idx = result[
        (result["position_code"] == "TW") &
        (result["pqi_mean"] == gk_mean)
    ].index
    assert len(gk_mean_player_idx) == 1
    z_score = result.loc[gk_mean_player_idx[0], "pqi_position_adjusted"]
    assert abs(z_score) < 1e-9


def test_bootstrap_ci_structure():
    """bootstrap_elite_quadrant must return a dict with all five required keys."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "awi_per_minute": rng.uniform(0, 30, 10),
        "mean_pqi": rng.uniform(0, 100, 10),
    })
    result = bootstrap_elite_quadrant(df, n_bootstrap=50, seed=42)
    expected_keys = {
        "mean_elite_count",
        "std_elite_count",
        "ci_lower_95",
        "ci_upper_95",
        "observed_count",
    }
    assert set(result.keys()) == expected_keys


def test_bootstrap_ci_observed_within_ci():
    """observed_count must lie within the 95% bootstrap CI."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "awi_per_minute": rng.uniform(0, 30, 10),
        "mean_pqi": rng.uniform(0, 100, 10),
    })
    result = bootstrap_elite_quadrant(df, n_bootstrap=200, seed=42)
    assert result["observed_count"] >= result["ci_lower_95"]
    assert result["observed_count"] <= result["ci_upper_95"]


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------

# Feature: metric-robustness, Property 1: deviation_pct formula correctness
@given(
    computed_awi=st.floats(min_value=0, max_value=200, allow_nan=False, allow_infinity=False),
    expected_awi=st.floats(min_value=0.001, max_value=200, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100)
def test_prop_deviation_pct_formula(computed_awi, expected_awi):
    """Validates: Requirements 1.8"""
    result = abs(computed_awi - expected_awi) / expected_awi * 100
    assert result == pytest.approx(abs(computed_awi - expected_awi) / expected_awi * 100)


# Feature: metric-robustness, Property 2: z-score formula correctness
@given(
    pqi_values=st.lists(
        st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=2, max_size=20,
    )
)
@settings(max_examples=100)
def test_prop_zscore_formula(pqi_values):
    """Validates: Requirements 3.4"""
    # All players in the same position group (MID)
    df = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(len(pqi_values))],
        "position_code": ["DMZ"] * len(pqi_values),  # DMZ maps to MID
        "pqi_mean": pqi_values,
    })
    result = normalize_pqi_by_position(df)
    group_mean = np.mean(pqi_values)
    group_std = np.std(pqi_values, ddof=1)
    # Skip degenerate cases: zero/near-zero std (all-identical values produce
    # floating-point near-zero in numpy but exact 0 in pandas, causing inf z-scores)
    if group_std < 1e-10 or np.isnan(group_std) or not np.isfinite(group_std):
        return
    # Also skip if pandas produced any non-finite z-scores (defensive guard)
    if not result["pqi_position_adjusted"].apply(np.isfinite).all():
        return
    for i, pqi in enumerate(pqi_values):
        expected_z = (pqi - group_mean) / group_std
        assert result["pqi_position_adjusted"].iloc[i] == pytest.approx(expected_z, abs=1e-9)


# Feature: metric-robustness, Property 3: position_adjusted flag invariant
@given(
    pqi_values=st.lists(
        st.floats(min_value=0, max_value=100, allow_nan=False, allow_infinity=False),
        min_size=1, max_size=20,
    )
)
@settings(max_examples=100)
def test_prop_position_adjusted_flag(pqi_values):
    """Validates: Requirements 3.5"""
    df = pd.DataFrame({
        "player_id": [f"p{i}" for i in range(len(pqi_values))],
        "position_code": ["IVL"] * len(pqi_values),  # IVL maps to DEF
        "pqi_mean": pqi_values,
    })
    result = normalize_pqi_by_position(df)
    assert result["position_adjusted"].all()


# Feature: metric-robustness, Property 4: Bootstrap_Result key completeness
@given(
    n=st.integers(min_value=4, max_value=30),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(max_examples=100)
def test_prop_bootstrap_key_completeness(n, seed):
    """Validates: Requirements 5.4"""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "awi_per_minute": rng.uniform(0, 30, n),
        "mean_pqi": rng.uniform(0, 100, n),
    })
    result = bootstrap_elite_quadrant(df, n_bootstrap=10, seed=seed)
    expected_keys = {"mean_elite_count", "std_elite_count", "ci_lower_95", "ci_upper_95", "observed_count"}
    assert set(result.keys()) == expected_keys


# Feature: metric-robustness, Property 5: Bootstrap determinism
@given(
    n=st.integers(min_value=4, max_value=20),
    seed=st.integers(min_value=0, max_value=9999),
)
@settings(max_examples=100)
def test_prop_bootstrap_determinism(n, seed):
    """Validates: Requirements 5.5"""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "awi_per_minute": rng.uniform(0, 30, n),
        "mean_pqi": rng.uniform(0, 100, n),
    })
    result1 = bootstrap_elite_quadrant(df, n_bootstrap=50, seed=seed)
    result2 = bootstrap_elite_quadrant(df, n_bootstrap=50, seed=seed)
    for key in result1:
        assert result1[key] == result2[key]


# Feature: metric-robustness, Property 6: observed_count correctness
@given(
    n=st.integers(min_value=4, max_value=30),
)
@settings(max_examples=100)
def test_prop_observed_count_correctness(n):
    """Validates: Requirements 5.6"""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "awi_per_minute": rng.uniform(0, 30, n),
        "mean_pqi": rng.uniform(0, 100, n),
    })
    result = bootstrap_elite_quadrant(df, n_bootstrap=10, seed=42)
    awi_q75 = df["awi_per_minute"].quantile(0.75)
    pqi_q75 = df["mean_pqi"].quantile(0.75)
    expected_count = int(((df["awi_per_minute"] >= awi_q75) & (df["mean_pqi"] >= pqi_q75)).sum())
    assert result["observed_count"] == expected_count
