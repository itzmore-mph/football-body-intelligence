"""
body_orientation.py

Derives body facing direction and pre-orientation metrics from skeleton keypoints.

Body yaw uses the same clockwise-90° rotation technique as the ear fallback in
skeleton_parser.py, applied to the shoulder or hip lateral vector.

Pre-orientation angle measures how well a player's body was already aligned
toward the ball at the moment of ball receipt — 0° = facing ball, 90° = sideways,
180° = back to ball.

TRACAB coordinate system:
  X = along pitch length (positive toward right goal)
  Y = along pitch width  (positive toward top touchline)
  Units = meters, origin = center of pitch
"""

from math import atan2, degrees

from src.angle_utils import circular_diff
from src.skeleton_parser import PART

# Keypoint pairs for body yaw, tried in priority order.
# Each pair: (left_id, right_id). CW-90° rotation of (left - right) gives forward.
_BODY_JOINT_PAIRS = [
    (PART["left_shoulder"], PART["right_shoulder"]),
    (PART["left_hip"],      PART["right_hip"]),
]


def body_yaw_from_skeleton(parts_list: list) -> float | None:
    """Compute body facing direction (yaw, degrees) from shoulder or hip keypoints.

    Uses the same clockwise-90° rotation as the ear fallback in skeleton_parser:
      lateral_vec = left_joint - right_joint  (anatomical left direction)
      forward     = cw_rotate_90(lateral_vec) = (lateral_y, -lateral_x)
      yaw         = atan2(forward_y, forward_x)

    Tries shoulder pair first, falls back to hip pair if shoulders are missing.
    Returns None if neither pair is available or both joints coincide.

    Args:
        parts_list: Raw parts list from a skeleton struct (same format as
                    skeleton_parser input).

    Returns:
        Body yaw in degrees, range (-180°, 180°], or None.
    """
    parts = {
        p["name"]: p for p in parts_list
        if p["name"] in {PART["left_shoulder"], PART["right_shoulder"],
                         PART["left_hip"],      PART["right_hip"]}
    }

    for left_id, right_id in _BODY_JOINT_PAIRS:
        left  = parts.get(left_id)
        right = parts.get(right_id)
        if left is None or right is None:
            continue
        dx = left["position_x"] - right["position_x"]
        dy = left["position_y"] - right["position_y"]
        if dx == 0.0 and dy == 0.0:
            continue  # degenerate / tracking artefact
        # CW 90° rotation: (x, y) -> (y, -x)
        forward_x = dy
        forward_y = -dx
        return degrees(atan2(forward_y, forward_x))

    return None


def pre_orientation_angle(
    body_yaw_deg: float,
    ball_x: float,
    ball_y: float,
    player_x: float,
    player_y: float,
) -> float:
    """Compute angle between body direction and direction toward ball.

    Uses :func:`src.angle_utils.circular_diff` for the wraparound-safe angular
    difference between the body yaw and the ball-direction azimuth.

    Interpretation:
      0°   → body fully facing the incoming ball (ideal pre-orientation)
      90°  → body perpendicular to ball direction
      180° → back to the ball (worst case)

    Args:
        body_yaw_deg:   Player's body yaw in degrees (from body_yaw_from_skeleton).
        ball_x, ball_y: Ball position in TRACAB meters coordinates.
        player_x, player_y: Player position (e.g. pelvis keypoint) in meters.

    Returns:
        Angle in degrees, range [0°, 180°].
    """
    dx = ball_x - player_x
    dy = ball_y - player_y
    if dx == 0.0 and dy == 0.0:
        return 0.0  # player on the ball -- treat as perfectly pre-oriented
    ball_direction = degrees(atan2(dy, dx))
    return circular_diff(body_yaw_deg, ball_direction)
