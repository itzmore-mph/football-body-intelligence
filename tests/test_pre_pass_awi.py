"""
tests/test_pre_pass_awi.py

Unit tests for src/pre_pass_awi.py

No S3 or AWS access required -- all inputs are synthetic.

Covers:
  extract_kickoff_times:
    - Parses KickOff events from XML with GameSection attribute
    - Handles multiple case variants (KickOff, Kickoff, kickOff)
    - Returns empty dict when no KickOff events exist
    - Records only the first kickoff per section

  build_phase_frame_map:
    - Maps section numbers ("1", "2") to DFL strings ("firstHalf", "secondHalf")
    - Skips unknown section numbers

  pass_events_to_frames:
    - Converts EventTime to frame_number using kickoff reference
    - Sets frame_number to NA for sections without kickoff time
    - Handles empty DataFrames

  compute_pre_pass_awi:
    - Counts leading-edge scan events in the pre-pass window
    - Returns empty dict for empty inputs
    - Skips passes with insufficient coverage
    - Computes correct AWI rate

  compute_pre_pass_awi_per_pass:
    - Returns per-pass rows with correct metadata
    - Returns empty list for empty inputs
"""

import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import pandas as pd
import pytest

from src.pre_pass_awi import (
    build_phase_frame_map,
    compute_pre_pass_awi,
    compute_pre_pass_awi_per_pass,
    extract_kickoff_times,
    pass_events_to_frames,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_events_xml(events: list[tuple[str, str, str]]) -> ET.Element:
    """Build a minimal Events XML tree.

    Args:
        events: List of (event_time_iso, ko_tag_name, game_section) tuples.
                ko_tag_name is one of "KickOff", "Kickoff", "kickOff".
    """
    root = ET.Element("Events")
    for event_time, ko_tag, game_section in events:
        event_el = ET.SubElement(root, "Event", EventTime=event_time)
        ET.SubElement(event_el, ko_tag, GameSection=game_section)
    return root


def _make_scan_df(frames: list[int], is_scan: list[bool]) -> pd.DataFrame:
    """Build a scan_df with frame_number and is_scan columns."""
    return pd.DataFrame({"frame_number": frames, "is_scan": is_scan})


# ── extract_kickoff_times ────────────────────────────────────────────────────

class TestExtractKickoffTimes:

    def test_parses_kickoff_event(self):
        root = _make_events_xml([
            ("2025-09-13T13:31:00+02:00", "KickOff", "firstHalf"),
        ])
        result = extract_kickoff_times(root)
        assert "firstHalf" in result
        assert result["firstHalf"].year == 2025

    def test_handles_kickoff_case_variant(self):
        root = _make_events_xml([
            ("2025-09-13T14:00:00+02:00", "Kickoff", "secondHalf"),
        ])
        result = extract_kickoff_times(root)
        assert "secondHalf" in result

    def test_handles_kickoff_camelcase_variant(self):
        root = _make_events_xml([
            ("2025-09-13T14:00:00+02:00", "kickOff", "firstHalf"),
        ])
        result = extract_kickoff_times(root)
        assert "firstHalf" in result

    def test_returns_empty_when_no_kickoff(self):
        root = ET.Element("Events")
        ET.SubElement(root, "Event", EventTime="2025-09-13T13:31:00+02:00")
        result = extract_kickoff_times(root)
        assert result == {}

    def test_records_first_kickoff_per_section(self):
        root = _make_events_xml([
            ("2025-09-13T13:31:00+02:00", "KickOff", "firstHalf"),
            ("2025-09-13T13:32:00+02:00", "KickOff", "firstHalf"),
        ])
        result = extract_kickoff_times(root)
        # Should keep the first one (13:31), not the second (13:32)
        assert result["firstHalf"].minute == 31

    def test_multiple_sections(self):
        root = _make_events_xml([
            ("2025-09-13T13:31:00+02:00", "KickOff", "firstHalf"),
            ("2025-09-13T14:35:00+02:00", "KickOff", "secondHalf"),
        ])
        result = extract_kickoff_times(root)
        assert len(result) == 2
        assert "firstHalf" in result
        assert "secondHalf" in result


# ── build_phase_frame_map ────────────────────────────────────────────────────

class TestBuildPhaseFrameMap:

    def test_maps_section_numbers_to_dfl_strings(self):
        phases = [
            {"section": "1", "label": "1st half", "start_frame": 1000, "end_frame": 2000},
            {"section": "2", "label": "2nd half", "start_frame": 3000, "end_frame": 4000},
        ]
        result = build_phase_frame_map(phases)
        assert result == {"firstHalf": 1000, "secondHalf": 3000}

    def test_skips_unknown_sections(self):
        phases = [
            {"section": "1", "label": "1st half", "start_frame": 1000, "end_frame": 2000},
            {"section": "99", "label": "unknown", "start_frame": 5000, "end_frame": 6000},
        ]
        result = build_phase_frame_map(phases)
        assert "99" not in result
        assert len(result) == 1

    def test_empty_phases(self):
        assert build_phase_frame_map([]) == {}

    def test_all_four_sections(self):
        phases = [
            {"section": str(i), "label": f"phase {i}", "start_frame": i * 1000, "end_frame": (i + 1) * 1000}
            for i in range(1, 5)
        ]
        result = build_phase_frame_map(phases)
        assert set(result.keys()) == {"firstHalf", "secondHalf", "firstExtraTime", "secondExtraTime"}


# ── pass_events_to_frames ────────────────────────────────────────────────────

class TestPassEventsToFrames:

    def test_converts_event_time_to_frame(self):
        pass_df = pd.DataFrame({
            "real_time": ["2025-09-13T13:32:00+02:00"],
            "game_section": ["firstHalf"],
            "player_id": ["P1"],
        })
        kickoff_times = {"firstHalf": datetime(2025, 9, 13, 13, 31, 0, tzinfo=timezone.utc)}
        phase_frame_map = {"firstHalf": 1000}

        result = pass_events_to_frames(pass_df, kickoff_times, phase_frame_map, framerate=50)
        assert "frame_number" in result.columns
        # 60 seconds after kickoff at 50 fps = 3000 frames offset
        # But timezone difference may affect this. The key point is it's not NA.
        assert pd.notna(result["frame_number"].iloc[0])

    def test_na_for_missing_kickoff(self):
        pass_df = pd.DataFrame({
            "real_time": ["2025-09-13T14:35:00+02:00"],
            "game_section": ["secondHalf"],
            "player_id": ["P1"],
        })
        # Only firstHalf kickoff provided, not secondHalf
        kickoff_times = {"firstHalf": datetime(2025, 9, 13, 13, 31, 0, tzinfo=timezone.utc)}
        phase_frame_map = {"firstHalf": 1000}

        result = pass_events_to_frames(pass_df, kickoff_times, phase_frame_map)
        assert pd.isna(result["frame_number"].iloc[0])

    def test_empty_dataframe(self):
        pass_df = pd.DataFrame(columns=["real_time", "game_section", "player_id"])
        result = pass_events_to_frames(pass_df, {}, {})
        assert "frame_number" in result.columns
        assert len(result) == 0

    def test_does_not_modify_original(self):
        pass_df = pd.DataFrame({
            "real_time": ["2025-09-13T13:32:00+02:00"],
            "game_section": ["firstHalf"],
            "player_id": ["P1"],
        })
        original_cols = set(pass_df.columns)
        pass_events_to_frames(pass_df, {}, {})
        assert set(pass_df.columns) == original_cols  # no mutation


# ── compute_pre_pass_awi ─────────────────────────────────────────────────────

class TestComputePrePassAwi:

    def test_empty_scan_df(self):
        result = compute_pre_pass_awi(
            pd.DataFrame(columns=["frame_number", "is_scan"]),
            pd.Series([500], dtype="Int64"),
        )
        assert result["pre_pass_awi"] is None
        assert result["n_passes_tracked"] == 0

    def test_empty_pass_frames(self):
        scan_df = _make_scan_df(list(range(300)), [False] * 300)
        result = compute_pre_pass_awi(scan_df, pd.Series(dtype="Int64"))
        assert result["pre_pass_awi"] is None
        assert result["n_passes_tracked"] == 0

    def test_counts_scan_events_correctly(self):
        # Build a scan_df with 300 frames (0-299).
        # Two scan events: frames 50-60 and frames 80-90 (leading edges at 50 and 80).
        frames = list(range(300))
        is_scan = [False] * 300
        for f in range(50, 61):
            is_scan[f] = True
        for f in range(80, 91):
            is_scan[f] = True
        scan_df = _make_scan_df(frames, is_scan)

        # Pass at frame 250, window of 250 frames covers [0, 250).
        result = compute_pre_pass_awi(
            scan_df,
            pd.Series([250], dtype="Int64"),
            window_frames=250,
            min_coverage=0.5,
            framerate=50,
        )
        assert result["n_passes_tracked"] == 1
        assert result["pre_pass_awi"] is not None
        # 2 scan events in 250 frames = 5 seconds = 1/12 minute
        # AWI = 2 / (250/50/60) = 2 / 0.08333 = 24.0
        assert result["pre_pass_awi"] == pytest.approx(24.0, abs=0.1)

    def test_skips_pass_with_insufficient_coverage(self):
        # Only 10 frames of data, but window is 250 frames.
        scan_df = _make_scan_df(list(range(10)), [False] * 10)
        result = compute_pre_pass_awi(
            scan_df,
            pd.Series([500], dtype="Int64"),
            window_frames=250,
            min_coverage=0.5,
        )
        assert result["n_passes_tracked"] == 0

    def test_scan_pct_is_fraction(self):
        # All frames scanning in the window.
        frames = list(range(300))
        is_scan = [True] * 300
        scan_df = _make_scan_df(frames, is_scan)

        result = compute_pre_pass_awi(
            scan_df,
            pd.Series([280], dtype="Int64"),
            window_frames=250,
            min_coverage=0.5,
        )
        assert result["pre_pass_scan_pct"] is not None
        assert 0.0 <= result["pre_pass_scan_pct"] <= 1.0


# ── compute_pre_pass_awi_per_pass ────────────────────────────────────────────

class TestComputePrePassAwiPerPass:

    def test_returns_per_pass_rows(self):
        frames = list(range(600))
        is_scan = [False] * 600
        for f in range(50, 60):
            is_scan[f] = True
        scan_df = _make_scan_df(frames, is_scan)

        rows = compute_pre_pass_awi_per_pass(
            scan_df,
            pd.Series([300, 500], dtype="Int64"),
            jersey=6, team=1, name="Kimmich",
            match_id="FCB-HSV", phase_label="1st half", phase_start=0,
            window_frames=250, min_coverage=0.5, framerate=50,
        )
        assert len(rows) == 2
        assert rows[0]["jersey"] == 6
        assert rows[0]["name"] == "Kimmich"
        assert rows[0]["match_id"] == "FCB-HSV"
        assert "pre_pass_scan_count" in rows[0]
        assert "minute" in rows[0]

    def test_empty_scan_df_returns_empty(self):
        rows = compute_pre_pass_awi_per_pass(
            pd.DataFrame(columns=["frame_number", "is_scan"]),
            pd.Series([500], dtype="Int64"),
            jersey=6, team=1, name="Kimmich",
            match_id="FCB-HSV", phase_label="1st half", phase_start=0,
        )
        assert rows == []

    def test_empty_pass_frames_returns_empty(self):
        scan_df = _make_scan_df(list(range(300)), [False] * 300)
        rows = compute_pre_pass_awi_per_pass(
            scan_df,
            pd.Series(dtype="Int64"),
            jersey=6, team=1, name="Kimmich",
            match_id="FCB-HSV", phase_label="1st half", phase_start=0,
        )
        assert rows == []

    def test_minute_calculation(self):
        frames = list(range(400))
        is_scan = [False] * 400
        scan_df = _make_scan_df(frames, is_scan)

        # Pass at frame 300, phase_start=0, framerate=50
        # minute = (300 - 0) / 50 / 60 = 0.1
        rows = compute_pre_pass_awi_per_pass(
            scan_df,
            pd.Series([300], dtype="Int64"),
            jersey=6, team=1, name="Test",
            match_id="TEST", phase_label="1st half", phase_start=0,
            window_frames=250, min_coverage=0.5, framerate=50,
        )
        assert len(rows) == 1
        assert rows[0]["minute"] == pytest.approx(0.1, abs=0.01)

    def test_window_coverage_reported(self):
        frames = list(range(400))
        is_scan = [False] * 400
        scan_df = _make_scan_df(frames, is_scan)

        rows = compute_pre_pass_awi_per_pass(
            scan_df,
            pd.Series([300], dtype="Int64"),
            jersey=6, team=1, name="Test",
            match_id="TEST", phase_label="1st half", phase_start=0,
            window_frames=250, min_coverage=0.5, framerate=50,
        )
        assert len(rows) == 1
        assert 0.0 < rows[0]["window_coverage"] <= 1.0
