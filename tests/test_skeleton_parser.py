"""
tests/test_skeleton_parser.py

Unit tests for skeleton_parser.py

Covers:
  - _head_yaw_from_nose_neck: cardinal directions + missing joints
  - _head_yaw_from_ears: cardinal directions + missing joints
  - extract_head_angles: player filtering, frame ordering, empty cases
  - extract_head_angles_batch: multi-player single-pass
"""

import math
import pandas as pd
import pytest

from src.skeleton_parser import (
    PART,
    _head_yaw_from_nose_neck,
    _head_yaw_from_ears,
    _parts_to_dict,
    extract_head_angles,
    extract_head_angles_batch,
)


# ── Fixtures & helpers ──────────────────────────────────────────────────────

def make_part(name_id: int, x: float, y: float, z: float = 0.0) -> dict:
    return {"name": name_id, "position_x": x, "position_y": y, "position_z": z}


def make_skeleton(jersey: int, team: int, parts: list[dict]) -> dict:
    return {
        "jersey_number": jersey,
        "team": team,
        "parts_count": len(parts),
        "parts": parts,
    }


def make_df(frames: list[dict]) -> pd.DataFrame:
    """Build a minimal parquet-style DataFrame from a list of frame dicts."""
    rows = []
    for f in frames:
        rows.append({
            "frame_number": f["frame_number"],
            "skeletons": f.get("skeletons", []),
            "ball_exists": False,
            "ball": {},
            "skeleton_count": len(f.get("skeletons", [])),
            "type": 1,
            "version": 1,
        })
    return pd.DataFrame(rows)


# ── Tests: _head_yaw_from_nose_neck ─────────────────────────────────────────

class TestHeadYawFromNoseNeck:

    def _parts(self, nose_x, nose_y, neck_x, neck_y) -> dict:
        return _parts_to_dict([
            make_part(PART["nose"], nose_x, nose_y),
            make_part(PART["neck"], neck_x, neck_y),
        ])

    def test_facing_east(self):
        # nose at (10, 0), neck at (0, 0) -> forward = +X -> yaw = 0
        parts = self._parts(10, 0, 0, 0)
        yaw = _head_yaw_from_nose_neck(parts)
        assert yaw == pytest.approx(0.0, abs=1e-6)

    def test_facing_north(self):
        # nose at (0, 10), neck at (0, 0) -> forward = +Y -> yaw = 90
        parts = self._parts(0, 10, 0, 0)
        yaw = _head_yaw_from_nose_neck(parts)
        assert yaw == pytest.approx(90.0, abs=1e-6)

    def test_facing_west(self):
        # nose at (-10, 0), neck at (0, 0) -> forward = -X -> yaw = 180
        parts = self._parts(-10, 0, 0, 0)
        yaw = _head_yaw_from_nose_neck(parts)
        assert abs(yaw) == pytest.approx(180.0, abs=1e-6)

    def test_facing_south(self):
        # nose at (0, -10), neck at (0, 0) -> forward = -Y -> yaw = -90
        parts = self._parts(0, -10, 0, 0)
        yaw = _head_yaw_from_nose_neck(parts)
        assert yaw == pytest.approx(-90.0, abs=1e-6)

    def test_missing_nose_returns_none(self):
        parts = _parts_to_dict([make_part(PART["neck"], 0, 0)])
        assert _head_yaw_from_nose_neck(parts) is None

    def test_missing_neck_returns_none(self):
        parts = _parts_to_dict([make_part(PART["nose"], 10, 0)])
        assert _head_yaw_from_nose_neck(parts) is None

    def test_degenerate_same_position_returns_none(self):
        parts = self._parts(5, 5, 5, 5)
        assert _head_yaw_from_nose_neck(parts) is None


# ── Tests: _head_yaw_from_ears ───────────────────────────────────────────────

class TestHeadYawFromEars:
    """
    Derivation recap:
      ear_vec = left_ear - right_ear  (anatomical left direction)
      forward = clockwise_90(ear_vec) = (ear_vec_y, -ear_vec_x)
      yaw = atan2(forward_y, forward_x)
    """

    def _parts(self, lx, ly, rx, ry) -> dict:
        return _parts_to_dict([
            make_part(PART["left_ear"],  lx, ly),
            make_part(PART["right_ear"], rx, ry),
        ])

    def test_facing_north(self):
        # facing +Y: left ear is at -X, right ear at +X
        # left_ear=(-5,0), right_ear=(5,0)
        # ear_vec = (-10, 0) -> forward_cw90 = (0, 10) -> yaw = 90
        parts = self._parts(-5, 0, 5, 0)
        yaw = _head_yaw_from_ears(parts)
        assert yaw == pytest.approx(90.0, abs=1e-6)

    def test_facing_east(self):
        # facing +X: left ear at +Y, right ear at -Y
        # left_ear=(0,5), right_ear=(0,-5)
        # ear_vec = (0, 10) -> forward_cw90 = (10, 0) -> yaw = 0
        parts = self._parts(0, 5, 0, -5)
        yaw = _head_yaw_from_ears(parts)
        assert yaw == pytest.approx(0.0, abs=1e-6)

    def test_facing_south(self):
        # facing -Y: left ear at +X, right ear at -X
        # left_ear=(5,0), right_ear=(-5,0)
        # ear_vec = (10, 0) -> forward_cw90 = (0, -10) -> yaw = -90
        parts = self._parts(5, 0, -5, 0)
        yaw = _head_yaw_from_ears(parts)
        assert yaw == pytest.approx(-90.0, abs=1e-6)

    def test_facing_west(self):
        # facing -X: left ear at -Y, right ear at +Y
        # left_ear=(0,-5), right_ear=(0,5)
        # ear_vec = (0,-10) -> forward_cw90 = (-10,0) -> yaw = 180
        parts = self._parts(0, -5, 0, 5)
        yaw = _head_yaw_from_ears(parts)
        assert abs(yaw) == pytest.approx(180.0, abs=1e-6)

    def test_missing_left_ear_returns_none(self):
        parts = _parts_to_dict([make_part(PART["right_ear"], 5, 0)])
        assert _head_yaw_from_ears(parts) is None

    def test_missing_right_ear_returns_none(self):
        parts = _parts_to_dict([make_part(PART["left_ear"], -5, 0)])
        assert _head_yaw_from_ears(parts) is None

    def test_ears_at_same_position_returns_none(self):
        parts = self._parts(0, 0, 0, 0)
        assert _head_yaw_from_ears(parts) is None


# ── Tests: extract_head_angles ───────────────────────────────────────────────

class TestExtractHeadAngles:

    def _frame(self, frame_num: int, skeletons: list) -> dict:
        return {"frame_number": frame_num, "skeletons": skeletons}

    def test_returns_correct_columns(self):
        parts = [make_part(PART["nose"], 10, 0), make_part(PART["neck"], 0, 0)]
        sk = make_skeleton(jersey=9, team=1, parts=parts)
        df = make_df([self._frame(100, [sk])])
        result = extract_head_angles(df, jersey_number=9, team=1)
        assert list(result.columns) == ["frame_number", "head_yaw_deg"]

    def test_filters_correct_player(self):
        """Only jersey=9/team=1 should appear, not jersey=7/team=1."""
        parts_9  = [make_part(PART["nose"], 10, 0), make_part(PART["neck"], 0, 0)]
        parts_7  = [make_part(PART["nose"], 0, 10), make_part(PART["neck"], 0, 0)]
        sk9 = make_skeleton(9, 1, parts_9)
        sk7 = make_skeleton(7, 1, parts_7)
        df = make_df([self._frame(200, [sk9, sk7])])

        result_9 = extract_head_angles(df, jersey_number=9, team=1)
        result_7 = extract_head_angles(df, jersey_number=7, team=1)

        assert result_9.iloc[0]["head_yaw_deg"] == pytest.approx(0.0,  abs=1e-5)
        assert result_7.iloc[0]["head_yaw_deg"] == pytest.approx(90.0, abs=1e-5)

    def test_team_filtering(self):
        """Same jersey but different team should not match."""
        parts = [make_part(PART["nose"], 10, 0), make_part(PART["neck"], 0, 0)]
        sk_home = make_skeleton(9, team=1, parts=parts)
        df = make_df([self._frame(300, [sk_home])])

        result_away = extract_head_angles(df, jersey_number=9, team=0)
        assert len(result_away) == 0

    def test_sorted_by_frame_number(self):
        parts = [make_part(PART["nose"], 10, 0), make_part(PART["neck"], 0, 0)]
        sk = make_skeleton(9, 1, parts)
        # Intentionally out of order
        df = make_df([
            self._frame(300, [sk]),
            self._frame(100, [sk]),
            self._frame(200, [sk]),
        ])
        result = extract_head_angles(df, jersey_number=9, team=1)
        assert list(result["frame_number"]) == [100, 200, 300]

    def test_fallback_to_ears_when_nose_neck_missing(self):
        """Falls back to ear method when nose/neck not in parts."""
        # facing north via ear method: left_ear=(-5,0), right_ear=(5,0) -> yaw=90
        parts = [
            make_part(PART["left_ear"],  -5, 0),
            make_part(PART["right_ear"],  5, 0),
        ]
        sk = make_skeleton(9, 1, parts)
        df = make_df([self._frame(100, [sk])])
        result = extract_head_angles(df, jersey_number=9, team=1)
        assert len(result) == 1
        assert result.iloc[0]["head_yaw_deg"] == pytest.approx(90.0, abs=1e-5)

    def test_empty_skeletons_list(self):
        df = make_df([{"frame_number": 100, "skeletons": []}])
        result = extract_head_angles(df, jersey_number=9, team=1)
        assert len(result) == 0

    def test_player_absent_in_some_frames(self):
        """Frames where the player is absent produce no row."""
        parts = [make_part(PART["nose"], 10, 0), make_part(PART["neck"], 0, 0)]
        sk9 = make_skeleton(9, 1, parts)
        sk7 = make_skeleton(7, 1, parts)  # different jersey
        df = make_df([
            self._frame(100, [sk9]),
            self._frame(200, [sk7]),  # player 9 absent
            self._frame(300, [sk9]),
        ])
        result = extract_head_angles(df, jersey_number=9, team=1)
        assert list(result["frame_number"]) == [100, 300]

    def test_empty_dataframe_returns_empty(self):
        df = make_df([])
        result = extract_head_angles(df, jersey_number=9, team=1)
        assert len(result) == 0
        assert list(result.columns) == ["frame_number", "head_yaw_deg"]

    def test_no_valid_joints_produces_no_row(self):
        """If head joints are completely missing, frame is skipped."""
        # Only hip joints - no head data
        parts = [make_part(PART["left_hip"], 0, 0), make_part(PART["right_hip"], 10, 0)]
        sk = make_skeleton(9, 1, parts)
        df = make_df([self._frame(100, [sk])])
        result = extract_head_angles(df, jersey_number=9, team=1)
        assert len(result) == 0


# ── Tests: extract_head_angles_batch ────────────────────────────────────────

class TestExtractHeadAnglesBatch:

    def test_returns_all_requested_players(self):
        parts_9 = [make_part(PART["nose"], 10, 0), make_part(PART["neck"], 0, 0)]
        parts_7 = [make_part(PART["nose"], 0, 10), make_part(PART["neck"], 0, 0)]
        sk9 = make_skeleton(9, 1, parts_9)
        sk7 = make_skeleton(7, 1, parts_7)
        df = make_df([{"frame_number": 100, "skeletons": [sk9, sk7]}])

        result = extract_head_angles_batch(df, players=[(9, 1), (7, 1)])
        assert (9, 1) in result
        assert (7, 1) in result

    def test_batch_matches_single_player_output(self):
        parts = [make_part(PART["nose"], 10, 0), make_part(PART["neck"], 0, 0)]
        sk = make_skeleton(9, 1, parts)
        df = make_df([
            {"frame_number": 100, "skeletons": [sk]},
            {"frame_number": 200, "skeletons": [sk]},
        ])

        single = extract_head_angles(df, jersey_number=9, team=1)
        batch  = extract_head_angles_batch(df, players=[(9, 1)])[(9, 1)]

        pd.testing.assert_frame_equal(single.reset_index(drop=True),
                                      batch.reset_index(drop=True))

    def test_missing_player_returns_empty_df(self):
        parts = [make_part(PART["nose"], 10, 0), make_part(PART["neck"], 0, 0)]
        sk = make_skeleton(9, 1, parts)
        df = make_df([{"frame_number": 100, "skeletons": [sk]}])

        result = extract_head_angles_batch(df, players=[(99, 1)])
        assert len(result[(99, 1)]) == 0
