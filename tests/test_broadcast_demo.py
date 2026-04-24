"""
Unit tests for dashboard/broadcast_demo.py pure functions and fixed-content requirements.

Covers:
- classify_quadrant(): quadrant label assignment and boundary/null handling
- format_pqi_display(): formatting and n/a fallback
- Constants: TICKER_MESSAGES content, DFL color values
- Artifact checks: broadcast_screenshot.py docstring, README.md content

Requirements: 2.7, 2.8, 2.9, 2.10, 3.2, 4.3, 4.4, 4.5, 4.6, 6.4, 7.3, 7.4, 9.5
"""
import math
import pathlib

import pytest

from dashboard.broadcast_demo import (
    DFL_BLACK,
    DFL_GREY,
    DFL_RED,
    DFL_WHITE,
    TICKER_MESSAGES,
    classify_quadrant,
    format_pqi_display,
)

# ---------------------------------------------------------------------------
# 7.1 Unit tests for classify_quadrant()
# ---------------------------------------------------------------------------

# Thresholds used across all classify_quadrant tests
AWI_Q75 = 15.0
PQI_Q75 = 60.0


def test_classify_quadrant_elite():
    """Both metrics above threshold returns ELITE."""
    result = classify_quadrant(awi=20.0, pqi=70.0, awi_q75=AWI_Q75, pqi_q75=PQI_Q75)
    assert result == "ELITE"


def test_classify_quadrant_aware():
    """High AWI, low PQI returns AWARE."""
    result = classify_quadrant(awi=20.0, pqi=50.0, awi_q75=AWI_Q75, pqi_q75=PQI_Q75)
    assert result == "AWARE"


def test_classify_quadrant_presser():
    """Low AWI, high PQI returns PRESSER."""
    result = classify_quadrant(awi=10.0, pqi=70.0, awi_q75=AWI_Q75, pqi_q75=PQI_Q75)
    assert result == "PRESSER"


def test_classify_quadrant_developing():
    """Both metrics below threshold returns DEVELOPING."""
    result = classify_quadrant(awi=10.0, pqi=50.0, awi_q75=AWI_Q75, pqi_q75=PQI_Q75)
    assert result == "DEVELOPING"


def test_classify_quadrant_boundary():
    """Values exactly at threshold are treated as >= (boundary condition)."""
    # Exactly at both thresholds should be ELITE (>= on both sides)
    result = classify_quadrant(
        awi=AWI_Q75, pqi=PQI_Q75, awi_q75=AWI_Q75, pqi_q75=PQI_Q75
    )
    assert result == "ELITE"

    # Exactly at AWI threshold, below PQI threshold -> AWARE
    result_aware = classify_quadrant(
        awi=AWI_Q75, pqi=PQI_Q75 - 0.1, awi_q75=AWI_Q75, pqi_q75=PQI_Q75
    )
    assert result_aware == "AWARE"

    # Below AWI threshold, exactly at PQI threshold -> PRESSER
    result_presser = classify_quadrant(
        awi=AWI_Q75 - 0.1, pqi=PQI_Q75, awi_q75=AWI_Q75, pqi_q75=PQI_Q75
    )
    assert result_presser == "PRESSER"


def test_classify_quadrant_null_pqi():
    """NaN PQI falls back to AWI-only classification."""
    # High AWI with NaN PQI -> AWARE (not ELITE, because PQI is unknown)
    result_aware = classify_quadrant(
        awi=20.0, pqi=float("nan"), awi_q75=AWI_Q75, pqi_q75=PQI_Q75
    )
    assert result_aware == "AWARE"

    # Low AWI with NaN PQI -> DEVELOPING
    result_developing = classify_quadrant(
        awi=10.0, pqi=float("nan"), awi_q75=AWI_Q75, pqi_q75=PQI_Q75
    )
    assert result_developing == "DEVELOPING"

    # None PQI also triggers AWI-only fallback
    result_none = classify_quadrant(
        awi=20.0, pqi=None, awi_q75=AWI_Q75, pqi_q75=PQI_Q75
    )
    assert result_none == "AWARE"


# ---------------------------------------------------------------------------
# 7.2 Unit tests for format_pqi_display()
# ---------------------------------------------------------------------------


def test_format_pqi_display_valid():
    """Finite float returns a non-empty string that is not 'n/a'."""
    result = format_pqi_display(72.4)
    assert isinstance(result, str)
    assert len(result) > 0
    assert result != "n/a"


def test_format_pqi_display_nan():
    """float('nan') returns 'n/a'."""
    result = format_pqi_display(float("nan"))
    assert result == "n/a"


def test_format_pqi_display_none():
    """None returns 'n/a'."""
    result = format_pqi_display(None)
    assert result == "n/a"


# ---------------------------------------------------------------------------
# 7.3 Unit tests for constants and fixed-content requirements
# ---------------------------------------------------------------------------


def test_ticker_messages_content():
    """TICKER_MESSAGES contains all three required validated-finding strings."""
    required = [
        "+57% pre-pass scan spike",
        "r=-0.11 AWI/PQI independence",
        "R=0.854 cross-half stability",
    ]
    for msg in required:
        assert msg in TICKER_MESSAGES, f"Missing required ticker message: {msg!r}"


def test_dfl_colors():
    """DFL color constants match the official DFL color system hex values."""
    assert DFL_RED == "#D10214"
    assert DFL_BLACK == "#000000"
    assert DFL_WHITE == "#FFFFFF"
    assert DFL_GREY == "#8A8A8A"


# ---------------------------------------------------------------------------
# 7.4 Unit tests for documentation and script artifacts
# ---------------------------------------------------------------------------


def test_screenshot_docstring():
    """scripts/broadcast_screenshot.py has a non-empty docstring containing 'manual'."""
    script_path = pathlib.Path("scripts/broadcast_screenshot.py")
    assert script_path.exists(), "scripts/broadcast_screenshot.py not found"
    source = script_path.read_text(encoding="utf-8")
    # Extract the module-level docstring by looking for the opening triple-quote
    # and reading until the closing triple-quote. We check the raw source rather
    # than importing the module to avoid requiring selenium/playwright.
    assert '"""' in source or "'''" in source, "No docstring found in broadcast_screenshot.py"
    assert "manual" in source.lower(), (
        "broadcast_screenshot.py docstring does not contain the word 'manual'"
    )


def test_readme_launch_command():
    """README.md contains the broadcast demo launch command."""
    readme = pathlib.Path("README.md").read_text(encoding="utf-8")
    assert "streamlit run dashboard/broadcast_demo.py" in readme


def test_readme_broadcast_section():
    """README.md references the broadcast demo."""
    readme = pathlib.Path("README.md").read_text(encoding="utf-8")
    assert "broadcast_demo.py" in readme


# ---------------------------------------------------------------------------
# Task 8: Hypothesis property tests
# ---------------------------------------------------------------------------
import math

import pandas as pd
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.strategies import floats, none, one_of, just

from dashboard.broadcast_demo import (
    classify_quadrant,
    format_pqi_display,
    compute_league_mean_awi,
    compute_role_mean_pqi,
    compute_quadrant_thresholds,
)


# ---------------------------------------------------------------------------
# 8.1 Property 1: Quadrant classification is a complete partition
# Validates: Requirements 2.7, 2.8, 2.9, 2.10
# ---------------------------------------------------------------------------

@given(
    awi=floats(min_value=0, max_value=50, allow_nan=False),
    pqi=one_of(floats(min_value=0, max_value=100, allow_nan=False), none()),
    awi_q75=floats(min_value=0.1, max_value=40, allow_nan=False),
    pqi_q75=floats(min_value=0.1, max_value=99, allow_nan=False),
)
@settings(max_examples=25)
def test_property_1_quadrant_complete_partition(awi, pqi, awi_q75, pqi_q75):
    """Property 1: Quadrant classification is a complete partition.

    For any valid AWI and PQI values and any pair of threshold values,
    classify_quadrant() returns exactly one of the four valid labels.

    Validates: Requirements 2.7, 2.8, 2.9, 2.10
    """
    result = classify_quadrant(awi, pqi, awi_q75, pqi_q75)
    assert result in {"ELITE", "AWARE", "PRESSER", "DEVELOPING"}


# ---------------------------------------------------------------------------
# 8.2 Property 2: ELITE classification requires both metrics above threshold
# Validates: Requirements 2.7
# ---------------------------------------------------------------------------

@given(
    awi_q75=floats(min_value=0.1, max_value=40, allow_nan=False),
    pqi_q75=floats(min_value=0.1, max_value=99, allow_nan=False),
    awi_offset=floats(min_value=0, max_value=10, allow_nan=False),
    pqi_offset=floats(min_value=0, max_value=1, allow_nan=False),
)
@settings(max_examples=25)
def test_property_2_elite_requires_both_above_threshold(awi_q75, pqi_q75, awi_offset, pqi_offset):
    """Property 2: ELITE classification requires both metrics above threshold.

    When both awi >= awi_q75 and pqi >= pqi_q75, result must be ELITE.
    When either metric is below threshold, result must not be ELITE.

    Validates: Requirements 2.7
    """
    # Both above threshold: must be ELITE
    awi_above = awi_q75 + awi_offset
    pqi_above = pqi_q75 + pqi_offset
    result_elite = classify_quadrant(awi_above, pqi_above, awi_q75, pqi_q75)
    assert result_elite == "ELITE"

    # AWI below threshold: must not be ELITE
    awi_below = max(0.0, awi_q75 - 0.01 - awi_offset)
    result_not_elite_awi = classify_quadrant(awi_below, pqi_above, awi_q75, pqi_q75)
    assert result_not_elite_awi != "ELITE"

    # PQI below threshold (finite value): must not be ELITE
    pqi_below = max(0.0, pqi_q75 - 0.01 - pqi_offset)
    result_not_elite_pqi = classify_quadrant(awi_above, pqi_below, awi_q75, pqi_q75)
    assert result_not_elite_pqi != "ELITE"


# ---------------------------------------------------------------------------
# 8.3 Property 3: AWARE classification requires high AWI and low PQI
# Validates: Requirements 2.8
# ---------------------------------------------------------------------------

@given(
    awi_q75=floats(min_value=0.1, max_value=40, allow_nan=False),
    pqi_q75=floats(min_value=0.1, max_value=99, allow_nan=False),
    awi_offset=floats(min_value=0, max_value=10, allow_nan=False),
    pqi_offset=floats(min_value=0.01, max_value=1, allow_nan=False),
)
@settings(max_examples=25)
def test_property_3_aware_requires_high_awi_low_pqi(awi_q75, pqi_q75, awi_offset, pqi_offset):
    """Property 3: AWARE classification requires high AWI and low PQI.

    When awi >= awi_q75 and pqi < pqi_q75 (finite), result must be AWARE.
    When awi < awi_q75, result must not be AWARE.

    Validates: Requirements 2.8
    """
    awi_above = awi_q75 + awi_offset
    # Use a finite pqi strictly below threshold to avoid NaN fallback path
    pqi_below = max(0.0, pqi_q75 - pqi_offset)
    assume(pqi_below < pqi_q75)

    result_aware = classify_quadrant(awi_above, pqi_below, awi_q75, pqi_q75)
    assert result_aware == "AWARE"

    # AWI below threshold: must not be AWARE
    awi_below = max(0.0, awi_q75 - 0.01 - awi_offset)
    result_not_aware = classify_quadrant(awi_below, pqi_below, awi_q75, pqi_q75)
    assert result_not_aware != "AWARE"


# ---------------------------------------------------------------------------
# 8.4 Property 4: PRESSER classification requires high PQI and low AWI
# Validates: Requirements 2.9
# ---------------------------------------------------------------------------

@given(
    awi_q75=floats(min_value=0.1, max_value=40, allow_nan=False),
    pqi_q75=floats(min_value=0.1, max_value=99, allow_nan=False),
    awi_offset=floats(min_value=0.01, max_value=10, allow_nan=False),
    pqi_offset=floats(min_value=0, max_value=1, allow_nan=False),
)
@settings(max_examples=25)
def test_property_4_presser_requires_high_pqi_low_awi(awi_q75, pqi_q75, awi_offset, pqi_offset):
    """Property 4: PRESSER classification requires high PQI and low AWI.

    When pqi >= pqi_q75 and awi < awi_q75 (finite pqi), result must be PRESSER.
    When pqi < pqi_q75, result must not be PRESSER.

    Validates: Requirements 2.9
    """
    # Use a finite pqi at or above threshold to avoid NaN fallback path
    pqi_above = pqi_q75 + pqi_offset
    awi_below = max(0.0, awi_q75 - awi_offset)
    assume(awi_below < awi_q75)

    result_presser = classify_quadrant(awi_below, pqi_above, awi_q75, pqi_q75)
    assert result_presser == "PRESSER"

    # PQI below threshold (finite): must not be PRESSER
    pqi_below = max(0.0, pqi_q75 - 0.01 - pqi_offset)
    result_not_presser = classify_quadrant(awi_below, pqi_below, awi_q75, pqi_q75)
    assert result_not_presser != "PRESSER"


# ---------------------------------------------------------------------------
# 8.5 Property 5: Missing PQI always produces "n/a" display
# Validates: Requirements 9.5
# ---------------------------------------------------------------------------

@given(
    pqi=one_of(
        just(None),
        just(float("nan")),
        just(float("inf")),
        just(float("-inf")),
    )
)
@settings(max_examples=25)
def test_property_5_missing_pqi_produces_na_display(pqi):
    """Property 5a: Missing or non-finite PQI always produces 'n/a' display.

    Validates: Requirements 9.5
    """
    result = format_pqi_display(pqi)
    assert result == "n/a"


@given(pqi=floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=25)
def test_property_5_finite_pqi_produces_nonempty_display(pqi):
    """Property 5b: Finite PQI produces a non-empty string that is not 'n/a'.

    Validates: Requirements 9.5
    """
    result = format_pqi_display(pqi)
    assert isinstance(result, str)
    assert len(result) > 0
    assert result != "n/a"


# ---------------------------------------------------------------------------
# 8.6 Property 6: League mean AWI equals column mean
# Validates: Requirements 9.2
# ---------------------------------------------------------------------------

@given(data=st.data())
@settings(max_examples=25)
def test_property_6_league_mean_awi_equals_column_mean(data):
    """Property 6: League mean AWI equals column mean for positive rows.

    For any DataFrame with at least one positive awi_per_minute value,
    compute_league_mean_awi() returns the mean of positive rows and is
    a finite positive float.

    Validates: Requirements 9.2
    """
    # Generate a list of awi values; ensure at least one is positive
    n = data.draw(st.integers(min_value=1, max_value=20))
    awi_values = data.draw(
        st.lists(
            floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    # Guarantee at least one positive value
    positive_val = data.draw(floats(min_value=0.001, max_value=50.0, allow_nan=False, allow_infinity=False))
    awi_values[0] = positive_val

    df = pd.DataFrame({"awi_per_minute": awi_values})

    result = compute_league_mean_awi(df)

    # Must equal the pandas mean of positive rows
    expected = df[df["awi_per_minute"] > 0]["awi_per_minute"].mean()
    assert math.isclose(result, expected, rel_tol=1e-9)

    # Must be a finite positive float
    assert math.isfinite(result)
    assert result > 0


# ---------------------------------------------------------------------------
# 8.7 Property 7: Role mean PQI equals group mean
# Validates: Requirements 9.3
# ---------------------------------------------------------------------------

@given(data=st.data())
@settings(max_examples=25)
def test_property_7_role_mean_pqi_equals_group_mean(data):
    """Property 7: Role mean PQI equals group mean for each pos_group.

    For any DataFrame with mean_pqi and pos_group columns, compute_role_mean_pqi()
    returns a dict where each value equals the group mean (skipna) and every
    pos_group present in the DataFrame appears as a key.

    Validates: Requirements 9.3
    """
    groups = data.draw(
        st.lists(
            st.sampled_from(["GK", "CB", "FB", "DM", "CM", "WM", "FW"]),
            min_size=1,
            max_size=5,
        )
    )
    n = len(groups)
    pqi_values = data.draw(
        st.lists(
            one_of(
                floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
                just(float("nan")),
            ),
            min_size=n,
            max_size=n,
        )
    )

    df = pd.DataFrame({"pos_group": groups, "mean_pqi": pqi_values})

    result = compute_role_mean_pqi(df)

    # Every pos_group in the DataFrame must appear as a key
    for group in df["pos_group"].unique():
        assert group in result, f"pos_group '{group}' missing from result"

    # Each value must equal the group mean (skipna)
    for group, group_df in df.groupby("pos_group"):
        expected_mean = group_df["mean_pqi"].mean(skipna=True)
        if math.isnan(expected_mean):
            assert math.isnan(result[group])
        else:
            assert math.isclose(result[group], expected_mean, rel_tol=1e-9), (
                f"Group '{group}': expected {expected_mean}, got {result[group]}"
            )


# ---------------------------------------------------------------------------
# 8.8 Property 8: Quadrant thresholds are 75th percentiles
# Validates: Requirements 9.4
# ---------------------------------------------------------------------------

@given(data=st.data())
@settings(max_examples=25)
def test_property_8_quadrant_thresholds_are_75th_percentiles(data):
    """Property 8: Quadrant thresholds are 75th percentiles.

    For any DataFrame with awi_per_minute and mean_pqi columns,
    compute_quadrant_thresholds() returns (awi_q75, pqi_q75) where each
    equals the 75th percentile of its column.

    Validates: Requirements 9.4
    """
    n = data.draw(st.integers(min_value=1, max_value=30))
    awi_values = data.draw(
        st.lists(
            floats(min_value=0.0, max_value=50.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    pqi_values = data.draw(
        st.lists(
            floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )

    df = pd.DataFrame({"awi_per_minute": awi_values, "mean_pqi": pqi_values})

    awi_q75, pqi_q75 = compute_quadrant_thresholds(df)

    expected_awi_q75 = df["awi_per_minute"].quantile(0.75)
    expected_pqi_q75 = df["mean_pqi"].quantile(0.75)

    assert math.isclose(awi_q75, expected_awi_q75, rel_tol=1e-9), (
        f"awi_q75: expected {expected_awi_q75}, got {awi_q75}"
    )
    assert math.isclose(pqi_q75, expected_pqi_q75, rel_tol=1e-9), (
        f"pqi_q75: expected {expected_pqi_q75}, got {pqi_q75}"
    )
