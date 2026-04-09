"""
skeleton_parser.py

Parses TRACAB TF15 Parquet skeleton frames and extracts head orientation angles
for use in the Awareness Index (AWI) pipeline.

Coordinate system (TRACAB):
  X  = along pitch length, positive toward right goal
  Y  = along pitch width, positive toward top touchline
  Z  = vertical, positive upward
  Origin = center of pitch
  Units = meters

Part ID mapping (TF15 v1.1):
  1  = Left ear      2  = Nose          3  = Right ear
  4  = Left shoulder 5  = Neck          6  = Right shoulder
  7  = Left elbow    8  = Right elbow   9  = Left wrist
  10 = Right wrist   11 = Left hip      12 = Pelvis
  13 = Right hip     14 = Left knee     15 = Right knee
  16 = Left ankle    17 = Right ankle   18 = Left heel
  19 = Left toe      20 = Right heel    21 = Right toe

Team encoding:
  0 = Away team   1 = Home team   3 = Referee

Jersey number:
  -1 = unassigned   0 = Head referee   1/2 = Assistants
"""

from math import atan2, degrees
import pandas as pd

# ── Part IDs ────────────────────────────────────────────────────────────────
PART = {
    "left_ear":       1,
    "nose":           2,
    "right_ear":      3,
    "left_shoulder":  4,
    "neck":           5,
    "right_shoulder": 6,
    "left_elbow":     7,
    "right_elbow":    8,
    "left_wrist":     9,
    "right_wrist":    10,
    "left_hip":       11,
    "pelvis":         12,
    "right_hip":      13,
    "left_knee":      14,
    "right_knee":     15,
    "left_ankle":     16,
    "right_ankle":    17,
    "left_heel":      18,
    "left_toe":       19,
    "right_heel":     20,
    "right_toe":      21,
}

# IDs needed for head yaw computation (only extract these for speed)
_HEAD_PART_IDS = {PART["nose"], PART["neck"], PART["left_ear"], PART["right_ear"]}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _parts_to_dict(parts: list) -> dict:
    """Convert a list of Skeleton Target Part dicts to {name_id: part_dict}.

    Filters to head-relevant parts only for efficiency.
    """
    return {p["name"]: p for p in parts if p["name"] in _HEAD_PART_IDS}


def _head_yaw_from_nose_neck(parts: dict) -> float | None:
    """Compute head yaw (degrees) from the nose -> neck forward vector.

    Primary method: projects the nose-neck vector onto the XY plane.
    Returns None if either joint is missing or XY projection is too small.

    XY distance can be near-zero when a player is looking sharply upward/downward
    (Z-component dominates). A threshold of 3 cm filters unstable atan2 results
    without discarding genuinely forward-looking frames.
    """
    nose = parts.get(PART["nose"])
    neck = parts.get(PART["neck"])
    if nose is None or neck is None:
        return None

    dx = nose["position_x"] - neck["position_x"]
    dy = nose["position_y"] - neck["position_y"]

    # Guard against degenerate frame (both joints at exact same XY position)
    if dx == 0.0 and dy == 0.0:
        return None

    return degrees(atan2(dy, dx))


def _head_yaw_from_ears(parts: dict) -> float | None:
    """Compute head yaw (degrees) from left/right ear positions.

    Fallback method: the lateral ear vector rotated 90° clockwise gives
    the forward direction.

    Derivation (TRACAB XY plane):
      - ear_vec = left_ear - right_ear  (points to player's anatomical left)
      - forward = clockwise_rotate_90(ear_vec) = (ear_vec_y, -ear_vec_x)
      - yaw = atan2(forward_y, forward_x)

    Returns None if either ear joint is missing.
    """
    left_ear  = parts.get(PART["left_ear"])
    right_ear = parts.get(PART["right_ear"])
    if left_ear is None or right_ear is None:
        return None

    ear_dx = left_ear["position_x"] - right_ear["position_x"]
    ear_dy = left_ear["position_y"] - right_ear["position_y"]

    # Guard against ears overlapping (tracking artifact)
    if ear_dx == 0.0 and ear_dy == 0.0:
        return None

    # Clockwise 90° rotation: (x, y) -> (y, -x)
    forward_x = ear_dy
    forward_y = -ear_dx

    return degrees(atan2(forward_y, forward_x))


def _compute_yaw(parts_list: list) -> float | None:
    """Compute head yaw from a raw parts list.

    Tries nose/neck first, falls back to ear-based method.
    Returns None if neither method can produce a result.
    """
    parts = _parts_to_dict(parts_list)
    yaw = _head_yaw_from_nose_neck(parts)
    if yaw is None:
        yaw = _head_yaw_from_ears(parts)
    return yaw


# ── Public API ───────────────────────────────────────────────────────────────

def extract_head_angles(
    df: pd.DataFrame,
    jersey_number: int,
    team: int,
) -> pd.DataFrame:
    """Extract per-frame head yaw angles for a single player.

    Uses apply() over the skeletons column for speed instead of iterrows().

    Args:
        df:             Raw parquet DataFrame (columns: frame_number, skeletons).
        jersey_number:  Player jersey number.
        team:           Team encoding: 1=Home, 0=Away, 3=Referee.

    Returns:
        DataFrame with columns [frame_number, head_yaw_deg], sorted by
        frame_number. head_yaw_deg is in degrees, range (-180, 180].
    """
    if df.empty or "skeletons" not in df.columns:
        return pd.DataFrame(columns=["frame_number", "head_yaw_deg"])

    def _yaw_for_row(skeletons):
        if skeletons is None or len(skeletons) == 0:
            return None
        for skeleton in skeletons:
            if skeleton["jersey_number"] == jersey_number and skeleton["team"] == team:
                return _compute_yaw(skeleton["parts"])
        return None

    yaws = df["skeletons"].apply(_yaw_for_row)
    mask = yaws.notna()
    if not mask.any():
        return pd.DataFrame(columns=["frame_number", "head_yaw_deg"])

    return (
        pd.DataFrame({
            "frame_number": df.loc[mask, "frame_number"].values,
            "head_yaw_deg": yaws[mask].values,
        })
        .sort_values("frame_number")
        .reset_index(drop=True)
    )


def extract_head_angles_batch(
    df: pd.DataFrame,
    players: list[tuple[int, int]],
) -> dict[tuple[int, int], pd.DataFrame]:
    """Extract head angles for multiple players in a single pass over the data.

    Args:
        df:      Raw parquet DataFrame.
        players: List of (jersey_number, team) tuples.

    Returns:
        Dict mapping (jersey_number, team) -> DataFrame[frame_number, head_yaw_deg].
    """
    player_set = set(players)
    records: dict[tuple, list] = {p: [] for p in player_set}

    for row in df.itertuples(index=False):
        skeletons = row.skeletons
        if skeletons is None or len(skeletons) == 0:
            continue
        for skeleton in skeletons:
            key = (skeleton["jersey_number"], skeleton["team"])
            if key in player_set:
                yaw = _compute_yaw(skeleton["parts"])
                if yaw is not None:
                    records[key].append((row.frame_number, yaw))

    result = {}
    for key, recs in records.items():
        if recs:
            result[key] = (
                pd.DataFrame(recs, columns=["frame_number", "head_yaw_deg"])
                .sort_values("frame_number")
                .reset_index(drop=True)
            )
        else:
            result[key] = pd.DataFrame(columns=["frame_number", "head_yaw_deg"])

    return result
