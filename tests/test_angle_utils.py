"""
tests/test_angle_utils.py

Unit tests for src/angle_utils.py

Covers:
  circular_diff:
    - Scalar inputs: cardinal directions, wraparound at +/-180, identical angles
    - numpy array inputs: element-wise correctness
    - pandas Series inputs: element-wise correctness
    - Symmetry: circular_diff(a, b) == circular_diff(b, a)
    - Output range: always in [0, 180]
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from src.angle_utils import circular_diff


class TestCircularDiffScalar:

    def test_same_angle_returns_zero(self):
        assert circular_diff(45.0, 45.0) == 0.0

    def test_opposite_directions(self):
        assert circular_diff(0.0, 180.0) == 180.0

    def test_right_angle(self):
        assert circular_diff(0.0, 90.0) == 90.0

    def test_wraparound_positive(self):
        # 170 to -170 should be 20, not 340
        assert circular_diff(170.0, -170.0) == pytest.approx(20.0)

    def test_wraparound_negative(self):
        assert circular_diff(-170.0, 170.0) == pytest.approx(20.0)

    def test_350_to_10(self):
        # 350 and 10 are 20 degrees apart across the 0/360 boundary
        assert circular_diff(350.0, 10.0) == pytest.approx(20.0)

    def test_10_to_350(self):
        assert circular_diff(10.0, 350.0) == pytest.approx(20.0)

    def test_zero_to_zero(self):
        assert circular_diff(0.0, 0.0) == 0.0

    def test_small_difference(self):
        assert circular_diff(1.0, 2.0) == pytest.approx(1.0)

    def test_symmetry(self):
        assert circular_diff(30.0, 100.0) == circular_diff(100.0, 30.0)


class TestCircularDiffNumpy:

    def test_array_element_wise(self):
        a = np.array([170.0, 0.0, 90.0])
        b = np.array([-170.0, 180.0, 90.0])
        result = circular_diff(a, b)
        expected = np.array([20.0, 180.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_empty_array(self):
        result = circular_diff(np.array([]), np.array([]))
        assert len(result) == 0

    def test_single_element(self):
        result = circular_diff(np.array([350.0]), np.array([10.0]))
        np.testing.assert_allclose(result, np.array([20.0]))


class TestCircularDiffPandas:

    def test_series_element_wise(self):
        a = pd.Series([170.0, 0.0, 90.0])
        b = pd.Series([-170.0, 180.0, 90.0])
        result = circular_diff(a, b)
        expected = pd.Series([20.0, 180.0, 0.0])
        pd.testing.assert_series_equal(result, expected, check_names=False)

    def test_empty_series(self):
        result = circular_diff(pd.Series(dtype=float), pd.Series(dtype=float))
        assert len(result) == 0


class TestCircularDiffProperties:

    @given(
        a=st.floats(min_value=-360, max_value=360, allow_nan=False, allow_infinity=False),
        b=st.floats(min_value=-360, max_value=360, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_result_in_range(self, a, b):
        result = circular_diff(a, b)
        assert 0.0 <= result <= 180.0 + 1e-9

    @given(
        a=st.floats(min_value=-360, max_value=360, allow_nan=False, allow_infinity=False),
        b=st.floats(min_value=-360, max_value=360, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200)
    def test_symmetry_property(self, a, b):
        assert circular_diff(a, b) == pytest.approx(circular_diff(b, a), abs=1e-9)

    @given(
        a=st.floats(min_value=-360, max_value=360, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_self_diff_is_zero(self, a):
        assert circular_diff(a, a) == pytest.approx(0.0, abs=1e-9)
