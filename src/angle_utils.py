"""
angle_utils.py

Single source of truth for circular (yaw) arithmetic used across the AWI/PQI
pipelines. The formula handles the wraparound at +/- 180 deg correctly, e.g.
the angular distance between 170 deg and -170 deg is 20 deg, not 340 deg.

Formula:
    abs(((a - b) + 180) % 360 - 180)

Works element-wise on Python floats, numpy arrays, and pandas Series, because
Python's built-in abs() dispatches to __abs__ (implemented by both numpy and
pandas). Always import circular_diff from here, do not re-implement inline.
"""

from typing import TypeVar

T = TypeVar("T")


def circular_diff(a: T, b: T) -> T:
    """Minimum angular distance between two yaw values, in degrees.

    Args:
        a, b: Yaw values in degrees. Can be scalars (float, int), numpy
              arrays, or pandas Series. Both arguments must be the same
              shape (or broadcastable per numpy rules).

    Returns:
        Same type as input(s). Result is always non-negative and lies in
        [0, 180].

    Examples:
        >>> circular_diff(170.0, -170.0)
        20.0
        >>> circular_diff(10.0, 350.0)
        20.0
        >>> circular_diff(0.0, 90.0)
        90.0
    """
    return abs(((a - b) + 180) % 360 - 180)
