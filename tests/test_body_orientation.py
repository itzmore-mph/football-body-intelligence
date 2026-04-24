"""
tests/test_body_orientation.py

Unit tests for src/body_orientation.py

Covers:
  body_yaw_from_skeleton:
    - Shoulder-based yaw: facing right (+X), left (-X), up (+Y), down (-Y)
    - Hip fallback when shoulders are missing
    - Returns None when both pairs are missing
    - Returns None when joints coincide (degenerate)

  pre_orientation_angle:
    - Facing ball directly = 0 degrees
    - Back to ball = 180 degrees
    - Perpendicular = 90 degrees
    - Player on ball = 0 degrees (edge case)
    - Wraparound correctness
"""

import math

import pytest

from src.body_orientation import body_yaw_from_skeleton, pre_orientation_angle
from src.skeleton_parser import PART


def _make_parts(**kwargs):
    """Build a minimal parts list for body_yaw_from_skeleton.

    Accepts keyword args like left_shoulder=(x, y), right_shoulder=(x, y), etc.
    Each value is a (position_x, position_y) tuple.
    """
    parts = []
    for name, (x, y) in kwargs.items():
        part_id = PART[name]
        parts.append({"name": part_id, "position_x": x, "position_y": y})
    return parts


class TestBodyYawFromSkeleton:

    def test_facing_positive_x_from_shoulders(self):
        # Left shoulder at y=+1, right shoulder at y=-1 relative to body center.
        # Lateral vec = (0, 2). CW 90 rotation = (2, 0). atan2(0, 2) = 0 degrees.
        parts = _make_parts(left_shoulder=(0.0, 1.0), right_shoulder=(0.0, -1.0))
        yaw = body_yaw_from_skeleton(parts)
        assert yaw == pytest.approx(0.0, abs=0.1)

    def test_facing_positive_y_from_shoulders(self):
        # Left shoulder at x=-1, right at x=+1.
        # Lateral vec = (-2, 0). CW 90 = (0, 2). atan2(2, 0) = 90 degrees.
        parts = _make_parts(left_shoulder=(-1.0, 0.0), right_shoulder=(1.0, 0.0))
        yaw = body_yaw_from_skeleton(parts)
        assert yaw == pytest.approx(90.0, abs=0.1)

    def test_facing_negative_x_from_shoulders(self):
        # Left shoulder at y=-1, right at y=+1.
        # Lateral vec = (0, -2). CW 90 = (-2, 0). atan2(0, -2) = 180 degrees.
        parts = _make_parts(left_shoulder=(0.0, -1.0), right_shoulder=(0.0, 1.0))
        yaw = body_yaw_from_skeleton(parts)
        assert abs(yaw) == pytest.approx(180.0, abs=0.1)

    def test_facing_negative_y_from_shoulders(self):
        # Left shoulder at x=+1, right at x=-1.
        # Lateral vec = (2, 0). CW 90 = (0, -2). atan2(-2, 0) = -90 degrees.
        parts = _make_parts(left_shoulder=(1.0, 0.0), right_shoulder=(-1.0, 0.0))
        yaw = body_yaw_from_skeleton(parts)
        assert yaw == pytest.approx(-90.0, abs=0.1)

    def test_hip_fallback_when_shoulders_missing(self):
        # Only hips provided, same geometry as shoulder test for +X facing.
        parts = _make_parts(left_hip=(0.0, 1.0), right_hip=(0.0, -1.0))
        yaw = body_yaw_from_skeleton(parts)
        assert yaw == pytest.approx(0.0, abs=0.1)

    def test_returns_none_when_no_joints(self):
        assert body_yaw_from_skeleton([]) is None

    def test_returns_none_when_only_one_shoulder(self):
        parts = _make_parts(left_shoulder=(0.0, 1.0))
        assert body_yaw_from_skeleton(parts) is None

    def test_returns_none_when_joints_coincide(self):
        parts = _make_parts(left_shoulder=(1.0, 1.0), right_shoulder=(1.0, 1.0))
        # Hips also coincide
        parts += _make_parts(left_hip=(2.0, 2.0), right_hip=(2.0, 2.0))
        assert body_yaw_from_skeleton(parts) is None

    def test_shoulders_preferred_over_hips(self):
        # Shoulders say facing +X (yaw=0), hips say facing +Y (yaw=90).
        # Shoulders should win.
        parts = _make_parts(
            left_shoulder=(0.0, 1.0), right_shoulder=(0.0, -1.0),
            left_hip=(-1.0, 0.0), right_hip=(1.0, 0.0),
        )
        yaw = body_yaw_from_skeleton(parts)
        assert yaw == pytest.approx(0.0, abs=0.1)


class TestPreOrientationAngle:

    def test_facing_ball_directly(self):
        # Body yaw = 0 (facing +X), ball is at +X from player.
        angle = pre_orientation_angle(0.0, 10.0, 0.0, 0.0, 0.0)
        assert angle == pytest.approx(0.0, abs=0.1)

    def test_back_to_ball(self):
        # Body yaw = 0 (facing +X), ball is at -X from player.
        angle = pre_orientation_angle(0.0, -10.0, 0.0, 0.0, 0.0)
        assert angle == pytest.approx(180.0, abs=0.1)

    def test_perpendicular_to_ball(self):
        # Body yaw = 0 (facing +X), ball is at +Y from player.
        angle = pre_orientation_angle(0.0, 0.0, 10.0, 0.0, 0.0)
        assert angle == pytest.approx(90.0, abs=0.1)

    def test_player_on_ball(self):
        # Ball and player at same position: should return 0.
        angle = pre_orientation_angle(45.0, 5.0, 5.0, 5.0, 5.0)
        assert angle == 0.0

    def test_wraparound(self):
        # Body yaw = 170, ball direction = -170 (20 degrees apart across boundary).
        # Ball at angle -170 from player: atan2(sin(-170), cos(-170))
        ball_x = math.cos(math.radians(-170)) * 10
        ball_y = math.sin(math.radians(-170)) * 10
        angle = pre_orientation_angle(170.0, ball_x, ball_y, 0.0, 0.0)
        assert angle == pytest.approx(20.0, abs=0.5)

    def test_result_range(self):
        # Result should always be in [0, 180].
        for body_yaw in range(-180, 181, 30):
            for ball_angle in range(-180, 181, 30):
                bx = math.cos(math.radians(ball_angle)) * 5
                by = math.sin(math.radians(ball_angle)) * 5
                result = pre_orientation_angle(float(body_yaw), bx, by, 0.0, 0.0)
                assert 0.0 <= result <= 180.0 + 1e-9
