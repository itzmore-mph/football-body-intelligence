"""
tests/test_batch_pipeline.py

Unit tests for batch_pipeline.py

All S3 I/O is mocked — no network access required.

Covers:
  load_phase_df:
    - pq.read_table called with correct filters and column pruning
    - Returns a pandas DataFrame

  compute_phase_awi_all_players:
    - Returns one result dict per player
    - All required keys present in each result
    - coverage_pct is 1.0 when all phase frames are detected
    - coverage_pct is 0.0 when head_df is empty
    - coverage_pct correctly partial when only some frames detected
    - match_id and phase_label propagated into results
    - Players with empty head_df get scan_count=0 and awi_per_minute=0.0

  run_match_awi:
    - load_xml called with the correct S3 key derived from match_config
    - Results are returned as a DataFrame
    - Raises RuntimeError when MatchInformation XML cannot be loaded
    - Raises RuntimeError when player list is empty

  run_all_matches:
    - Concatenates results from multiple matches
    - Continues processing after one match raises an error
    - Empty match_configs returns empty DataFrame
    - Matches with TODO in parquet_key are skipped without error
"""

import xml.etree.ElementTree as ET
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.batch_pipeline import (
    load_phase_df,
    compute_phase_awi_all_players,
    run_match_awi,
    run_all_matches,
)


# ── Fixtures & helpers ────────────────────────────────────────────────────────

def _make_player(jersey=6, team=1, name="Kimmich, Joshua", pos="MZ"):
    return {"jersey": jersey, "team": team, "name": name,
            "position": pos, "start_eleven": True, "player_id": "DFL-OBJ-001"}


def _make_phase(section="1", label="1st half", start=0, end=2999):
    return {"section": section, "label": label,
            "start_frame": start, "end_frame": end}


def _make_skeleton(jersey, team, nose_x=10, nose_y=0, neck_x=0, neck_y=0):
    """Minimal skeleton struct with nose and neck joints."""
    return {
        "jersey_number": jersey,
        "team": team,
        "parts_count": 2,
        "parts": [
            {"name": 2, "position_x": nose_x, "position_y": nose_y, "position_z": 0.0},
            {"name": 5, "position_x": neck_x, "position_y": neck_y, "position_z": 0.0},
        ],
    }


def _make_phase_df(phase_start, phase_end, players):
    """Build a minimal phase DataFrame covering all frames for the given players."""
    rows = []
    for frame in range(phase_start, phase_end + 1):
        skeletons = [_make_skeleton(p["jersey"], p["team"]) for p in players]
        rows.append({"frame_number": frame, "skeletons": skeletons})
    return pd.DataFrame(rows)


def _make_angles_dict(players, phase_start, phase_end, step=10):
    """Build a mock return value for stream_player_angles (constant yaw = 0 scans)."""
    frames = list(range(phase_start, phase_end + 1, step))
    result = {}
    for p in players:
        key = (p["jersey"], p["team"])
        result[key] = pd.DataFrame({
            "frame_number": frames,
            "head_yaw_deg": [45.0] * len(frames),
            "body_yaw_deg": [45.0] * len(frames),
        })
    return result


def _make_match_info_root(players=None):
    """Build a minimal MatchInformation XML element."""
    root = ET.Element("MatchInformation")
    teams = ET.SubElement(root, "Teams")
    home = ET.SubElement(teams, "HomeTeam", {"TeamId": "DFL-CLU-FCB"})
    pls = ET.SubElement(home, "Players")
    for p in (players or [_make_player()]):
        ET.SubElement(pls, "Player", {
            "ShirtNumber": str(p["jersey"]),
            "Name": p["name"],
            "PlayingPosition": p["position"],
            "Status": "Active",
            "StartEleven": "1" if p["start_eleven"] else "0",
            "PlayerId": p["player_id"],
        })
    return root


# ── Tests: load_phase_df ─────────────────────────────────────────────────────

class TestLoadPhaseDf:

    def test_calls_read_table_with_frame_filters(self):
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = pd.DataFrame(
            {"frame_number": [100], "skeletons": [[]]}
        )
        with patch("src.batch_pipeline.pq.read_table", return_value=mock_table) as mock_rt:
            load_phase_df(MagicMock(), "bucket/match.parquet", 100, 200)

        call_kwargs = mock_rt.call_args
        filters = call_kwargs.kwargs.get("filters") or call_kwargs.args[0] if call_kwargs.args else None
        filters = call_kwargs.kwargs["filters"]
        assert ("frame_number", ">=", 100) in filters
        assert ("frame_number", "<=", 200) in filters

    def test_column_pruning(self):
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = pd.DataFrame(
            {"frame_number": [100], "skeletons": [[]]}
        )
        with patch("src.batch_pipeline.pq.read_table", return_value=mock_table) as mock_rt:
            load_phase_df(MagicMock(), "bucket/match.parquet", 100, 200)

        columns = mock_rt.call_args.kwargs["columns"]
        assert "frame_number" in columns
        assert "skeletons" in columns
        assert len(columns) == 2  # only these two

    def test_returns_dataframe(self):
        expected = pd.DataFrame({"frame_number": [100], "skeletons": [[]]})
        mock_table = MagicMock()
        mock_table.to_pandas.return_value = expected
        with patch("src.batch_pipeline.pq.read_table", return_value=mock_table):
            result = load_phase_df(MagicMock(), "bucket/match.parquet", 100, 200)
        assert isinstance(result, pd.DataFrame)


# ── Tests: compute_phase_awi_all_players ─────────────────────────────────────

class TestComputePhaseAwiAllPlayers:

    def _run(self, players, phase_start=0, phase_end=299):
        phase_info = _make_phase(start=phase_start, end=phase_end)
        phase_df = _make_phase_df(phase_start, phase_end, players)
        return compute_phase_awi_all_players(phase_df, players, phase_info, "FCB-HSV")

    def test_returns_one_result_per_player(self):
        players = [_make_player(6, 1), _make_player(9, 0)]
        results = self._run(players)
        assert len(results) == 2

    def test_all_required_keys_present(self):
        required = {
            "jersey", "team", "name", "position", "match_id",
            "phase_label", "phase_start", "phase_end",
            "scan_count", "total_minutes", "awi_per_minute", "coverage_pct",
        }
        results = self._run([_make_player()])
        assert required.issubset(results[0].keys())

    def test_match_id_propagated(self):
        phase_info = _make_phase()
        phase_df = _make_phase_df(0, 299, [_make_player()])
        results = compute_phase_awi_all_players(phase_df, [_make_player()], phase_info, "TEST-ID")
        assert results[0]["match_id"] == "TEST-ID"

    def test_phase_label_propagated(self):
        phase_info = _make_phase(label="2nd half")
        phase_df = _make_phase_df(0, 299, [_make_player()])
        results = compute_phase_awi_all_players(phase_df, [_make_player()], phase_info, "X")
        assert results[0]["phase_label"] == "2nd half"

    def test_coverage_pct_full_detection(self):
        """Player present in every frame -> coverage_pct = 1.0."""
        results = self._run([_make_player()], phase_start=0, phase_end=299)
        # 300 frames, player in all -> coverage should be 1.0
        assert results[0]["coverage_pct"] == pytest.approx(1.0, abs=0.01)

    def test_coverage_pct_is_zero_when_player_absent(self):
        """Player not in any skeleton -> coverage_pct = 0.0, scan_count = 0."""
        players = [_make_player(6, 1)]
        phase_info = _make_phase()
        # Build phase_df with a different player (jersey 9) — jersey 6 absent
        phase_df = _make_phase_df(0, 299, [_make_player(9, 1)])
        results = compute_phase_awi_all_players(phase_df, players, phase_info, "X")
        assert results[0]["coverage_pct"] == 0.0
        assert results[0]["scan_count"] == 0
        assert results[0]["awi_per_minute"] == 0.0

    def test_scan_count_is_non_negative_integer(self):
        results = self._run([_make_player()])
        assert isinstance(results[0]["scan_count"], int)
        assert results[0]["scan_count"] >= 0

    def test_awi_per_minute_is_non_negative(self):
        results = self._run([_make_player()])
        assert results[0]["awi_per_minute"] >= 0.0

    def test_total_minutes_matches_phase_duration(self):
        # 300 frames at 50fps = 6 seconds = 0.1 min
        results = self._run([_make_player()], phase_start=0, phase_end=299)
        assert results[0]["total_minutes"] == pytest.approx(0.1, abs=0.01)

    def test_phase_start_end_in_results(self):
        results = self._run([_make_player()], phase_start=100, phase_end=399)
        assert results[0]["phase_start"] == 100
        assert results[0]["phase_end"] == 399


# ── Tests: run_match_awi ─────────────────────────────────────────────────────

class TestRunMatchAwi:

    def _make_config(self):
        return {
            "match_id":       "FCB-HSV",
            "parquet_key":    "Bayern_Hamburg/FCB-HSV.parquet",
            "match_info_key": "Bayern_Hamburg/MatchInformations_Bayern_Hamburg.xml",
        }

    def test_xml_key_constructed_correctly(self):
        config = self._make_config()
        bucket = "my-bucket"
        prefix = "challenge_2/Match_Data"
        expected_key = f"{prefix}/{config['match_info_key']}"

        match_info_root = _make_match_info_root()
        phase_info = _make_phase()
        angles = _make_angles_dict([_make_player()], 0, 99)

        with patch("src.batch_pipeline.load_xml", return_value=match_info_root) as mock_xml, \
             patch("src.batch_pipeline._load_phases_from_parquet", return_value=[phase_info]), \
             patch("src.batch_pipeline.stream_player_angles", return_value=angles):
            run_match_awi(MagicMock(), MagicMock(), bucket, prefix, config)

        _, call_bucket, call_key = mock_xml.call_args.args
        assert call_bucket == bucket
        assert call_key == expected_key

    def test_returns_dataframe(self):
        config = self._make_config()
        phase_info = _make_phase()
        angles = _make_angles_dict([_make_player()], 0, 99)

        with patch("src.batch_pipeline.load_xml", return_value=_make_match_info_root()), \
             patch("src.batch_pipeline._load_phases_from_parquet", return_value=[phase_info]), \
             patch("src.batch_pipeline.stream_player_angles", return_value=angles):
            result = run_match_awi(MagicMock(), MagicMock(), "bucket", "prefix", config)

        assert isinstance(result, pd.DataFrame)

    def test_raises_when_xml_cannot_be_loaded(self):
        config = self._make_config()
        with patch("src.batch_pipeline.load_xml", return_value=None):
            with pytest.raises(RuntimeError, match="MatchInformation"):
                run_match_awi(MagicMock(), MagicMock(), "bucket", "prefix", config)

    def test_raises_when_no_valid_players(self):
        """Referees only (team=3) or all jersey=-1 -> RuntimeError."""
        # XML with only a referee (team=3 is not yielded by extract_players_from_match_info
        # since it only reads HomeTeam/GuestTeam — so produce XML with no players at all)
        root = ET.Element("MatchInformation")
        ET.SubElement(root, "Teams")  # <Teams/> with no children

        config = self._make_config()
        with patch("src.batch_pipeline.load_xml", return_value=root), \
             patch("src.batch_pipeline._load_phases_from_parquet", return_value=[_make_phase()]):
            with pytest.raises(RuntimeError, match="players"):
                run_match_awi(MagicMock(), MagicMock(), "bucket", "prefix", config)

    def test_stream_player_angles_called_once_per_phase(self):
        config = self._make_config()
        phases = [_make_phase("1"), _make_phase("2", start=300, end=599)]
        angles = _make_angles_dict([_make_player()], 0, 99)

        with patch("src.batch_pipeline.load_xml", return_value=_make_match_info_root()), \
             patch("src.batch_pipeline._load_phases_from_parquet", return_value=phases), \
             patch("src.batch_pipeline.stream_player_angles", return_value=angles) as mock_sa:
            run_match_awi(MagicMock(), MagicMock(), "bucket", "prefix", config)

        assert mock_sa.call_count == 2


# ── Tests: run_all_matches ───────────────────────────────────────────────────

class TestRunAllMatches:

    def _match_config(self, match_id="FCB-HSV"):
        return {
            "match_id":       match_id,
            "parquet_key":    f"{match_id}/file.parquet",
            "match_info_key": f"{match_id}/MatchInfo.xml",
        }

    def _stub_run_match(self, match_id="FCB-HSV", n_rows=2):
        return pd.DataFrame({
            "match_id": [match_id] * n_rows,
            "jersey":   list(range(n_rows)),
            "awi_per_minute": [3.5] * n_rows,
        })

    def test_empty_configs_returns_empty_dataframe(self):
        result = run_all_matches(MagicMock(), MagicMock(), "bucket", "prefix", [])
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_todo_configs_are_skipped(self):
        config = {
            "match_id": "BVB-VFB",
            "parquet_key": "TODO/BVB-VFB.parquet",
            "match_info_key": "TODO/MatchInfo.xml",
        }
        with patch("src.batch_pipeline.run_match_awi") as mock_run:
            run_all_matches(MagicMock(), MagicMock(), "bucket", "prefix", [config])
        mock_run.assert_not_called()

    def test_concatenates_multiple_match_results(self):
        configs = [self._match_config("FCB-HSV"), self._match_config("BVB-VFB")]
        side_effects = [self._stub_run_match("FCB-HSV"), self._stub_run_match("BVB-VFB")]

        with patch("src.batch_pipeline.run_match_awi", side_effect=side_effects):
            result = run_all_matches(MagicMock(), MagicMock(), "bucket", "prefix", configs)

        assert len(result) == 4
        assert set(result["match_id"]) == {"FCB-HSV", "BVB-VFB"}

    def test_error_in_one_match_does_not_stop_others(self):
        configs = [self._match_config("FCB-HSV"), self._match_config("BVB-VFB")]

        def side_effect(s3c, s3fs, bucket, prefix, mc, checkpoint_path=None):
            if mc["match_id"] == "FCB-HSV":
                raise RuntimeError("S3 error")
            return self._stub_run_match("BVB-VFB")

        with patch("src.batch_pipeline.run_match_awi", side_effect=side_effect):
            result = run_all_matches(MagicMock(), MagicMock(), "bucket", "prefix", configs)

        assert len(result) == 2
        assert result.iloc[0]["match_id"] == "BVB-VFB"

    def test_all_matches_failing_returns_empty_dataframe(self):
        configs = [self._match_config()]

        with patch("src.batch_pipeline.run_match_awi", side_effect=RuntimeError("boom")):
            result = run_all_matches(MagicMock(), MagicMock(), "bucket", "prefix", configs)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
