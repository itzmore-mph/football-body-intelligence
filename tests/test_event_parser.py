"""
tests/test_event_parser.py

Unit tests for event_parser.py

Covers:
  extract_phases_from_metadata:
    - Strategy A: TF15 bytes keys from Parquet KV header (primary path)
    - Strategy A: str-normalised keys (same path, normalised)
    - Zero-value phases are skipped
    - Up to 5 phases returned
    - Strategy B: camelCase JSON flat keys
    - Strategy C: nested list under "sections"/"phases" etc.
    - Strategy D: verbose top-level keys (firstHalfStart etc.)
    - Unrecognised format raises ValueError with key list

  extract_players_from_match_info:
    - HomeTeam parsed as team=1, GuestTeam as team=0
    - AwayTeam fallback (alternative DFL feed variant)
    - GuestTeam takes precedence over AwayTeam (no double-counting)
    - ShirtNumber parsed as int
    - Invalid ShirtNumber -> jersey=-1, no crash
    - StartEleven "1" -> True, "0" -> False
    - Missing <Players> element -> empty list, no crash
    - Missing <Teams> element -> empty list, no crash

  extract_pass_events:
    - Correct output columns
    - Pass (Play+Pass child) -> 1 row with type="pass"
    - Play without Pass child (e.g. Cross) -> 0 rows
    - game_section tracks FinalWhistle events in document order
    - game_section tracks KickOff GameSection attribute
    - Kickoff (lowercase variant) also works
    - Missing Recipient -> recipient_id=""
    - pos_x/pos_y always None
    - Empty <EventData/> -> empty DataFrame with correct columns
    - Non-standard root tag (PutDataRequest) -> still iterates Event children
"""

import json
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from src.event_parser import (
    extract_phases_from_metadata,
    extract_players_from_match_info,
    extract_pass_events,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_match_info_xml(home_players=None, guest_players=None, away_players=None):
    """Build a minimal MatchInformation XML element."""
    root = ET.Element("MatchInformation")
    teams = ET.SubElement(root, "Teams")

    def _add_team(tag, players_data):
        team_el = ET.SubElement(teams, tag, {"TeamId": "DFL-CLU-TEST"})
        players_el = ET.SubElement(team_el, "Players")
        for p in (players_data or []):
            ET.SubElement(players_el, "Player", p)

    _add_team("HomeTeam",  home_players)
    if guest_players is not None:
        _add_team("GuestTeam", guest_players)
    if away_players is not None:
        _add_team("AwayTeam", away_players)

    return root


def _make_events_xml(events: list[dict], root_tag="EventData") -> ET.Element:
    """Build an events XML element from a list of event spec dicts.

    Each dict has:
      type: "pass" | "kickoff" | "final_whistle"
      event_id, event_time (optional)
      game_section (for kickoff/final_whistle)
      play_team, play_player, play_recipient, play_evaluation (for pass)
    """
    root = ET.Element(root_tag)
    for i, spec in enumerate(events):
        ev = ET.SubElement(root, "Event", {
            "EventId": spec.get("event_id", str(i + 1)),
            "EventTime": spec.get("event_time", "2025-09-13T15:30:00.000+02:00"),
        })
        etype = spec.get("type", "pass")

        if etype == "pass":
            play_attrs = {
                "Team":       spec.get("play_team", "DFL-CLU-HOME"),
                "Player":     spec.get("play_player", "DFL-OBJ-001"),
                "Evaluation": spec.get("play_evaluation", "successfulDribbling"),
            }
            if "play_recipient" in spec:
                play_attrs["Recipient"] = spec["play_recipient"]
            play = ET.SubElement(ev, "Play", play_attrs)
            ET.SubElement(play, "Pass")

        elif etype == "cross":
            play = ET.SubElement(ev, "Play", {
                "Team": "DFL-CLU-HOME", "Player": "DFL-OBJ-001",
            })
            ET.SubElement(play, "Cross")

        elif etype == "final_whistle":
            ET.SubElement(ev, "FinalWhistle", {
                "GameSection": spec.get("game_section", "firstHalf"),
            })

        elif etype in ("kickoff", "Kickoff", "KickOff"):
            tag = spec.get("tag", "KickOff")
            ET.SubElement(ev, tag, {
                "GameSection": spec.get("game_section", "firstHalf"),
            })

    return root


# ── Tests: extract_phases_from_metadata ─────────────────────────────────────

class TestExtractPhasesFromMetadata:

    def test_strategy_a_bytes_keys(self):
        """Primary TF15 path: bytes keys from pf.metadata.metadata."""
        meta = {
            b"phase_1_start": b"3330943",
            b"phase_1_end":   b"3484329",
            b"phase_2_start": b"3510000",
            b"phase_2_end":   b"3670000",
        }
        phases = extract_phases_from_metadata(meta)
        assert len(phases) == 2
        assert phases[0]["section"] == "1"
        assert phases[0]["start_frame"] == 3330943
        assert phases[0]["end_frame"]   == 3484329
        assert phases[1]["section"] == "2"
        assert phases[1]["start_frame"] == 3510000

    def test_strategy_a_str_keys(self):
        """str keys (already normalised) also work via Strategy A."""
        meta = {
            "phase_1_start": "100",
            "phase_1_end":   "200",
        }
        phases = extract_phases_from_metadata(meta)
        assert len(phases) == 1
        assert phases[0]["start_frame"] == 100
        assert phases[0]["end_frame"]   == 200

    def test_zero_value_phases_skipped(self):
        """Phases with start=0 or end=0 are placeholder slots and omitted."""
        meta = {
            b"phase_1_start": b"100",
            b"phase_1_end":   b"200",
            b"phase_2_start": b"0",
            b"phase_2_end":   b"0",
        }
        phases = extract_phases_from_metadata(meta)
        assert len(phases) == 1
        assert phases[0]["section"] == "1"

    def test_all_five_phases(self):
        meta = {}
        for n in range(1, 6):
            meta[f"phase_{n}_start"] = str(n * 1000)
            meta[f"phase_{n}_end"]   = str(n * 1000 + 500)
        phases = extract_phases_from_metadata(meta)
        assert len(phases) == 5
        assert [p["section"] for p in phases] == ["1", "2", "3", "4", "5"]

    def test_labels_are_correct(self):
        meta = {"phase_1_start": "100", "phase_1_end": "200",
                "phase_2_start": "300", "phase_2_end": "400"}
        phases = extract_phases_from_metadata(meta)
        assert phases[0]["label"] == "1st half"
        assert phases[1]["label"] == "2nd half"

    def test_strategy_b_camelcase_json(self):
        meta = {
            "phase1Start": "100", "phase1End": "200",
            "phase2Start": "300", "phase2End": "400",
        }
        phases = extract_phases_from_metadata(meta)
        assert len(phases) == 2
        assert phases[0]["start_frame"] == 100

    def test_strategy_c_nested_sections_list(self):
        sections = [
            {"section": "1", "start_frame": 100, "end_frame": 200},
            {"section": "2", "start_frame": 300, "end_frame": 400},
        ]
        meta = {"sections": json.dumps(sections)}
        phases = extract_phases_from_metadata(meta)
        assert len(phases) == 2
        assert phases[0]["start_frame"] == 100

    def test_strategy_c_nested_phases_list(self):
        items = [
            {"half": "1", "startFrame": 100, "endFrame": 200},
        ]
        meta = {"phases": json.dumps(items)}
        phases = extract_phases_from_metadata(meta)
        assert len(phases) == 1
        assert phases[0]["start_frame"] == 100

    def test_strategy_d_verbose_keys(self):
        meta = {
            "firstHalfStart":  "100", "firstHalfEnd":  "200",
            "secondHalfStart": "300", "secondHalfEnd": "400",
        }
        phases = extract_phases_from_metadata(meta)
        assert len(phases) == 2
        assert phases[0]["section"] == "1"
        assert phases[1]["section"] == "2"

    def test_unrecognized_raises_value_error(self):
        meta = {"foo": "bar", "baz": "qux"}
        with pytest.raises(ValueError, match="Unrecognized metadata format"):
            extract_phases_from_metadata(meta)

    def test_error_message_contains_key_list(self):
        meta = {"unknown_key": "123"}
        with pytest.raises(ValueError) as exc_info:
            extract_phases_from_metadata(meta)
        assert "unknown_key" in str(exc_info.value)


# ── Tests: extract_players_from_match_info ───────────────────────────────────

class TestExtractPlayersFromMatchInfo:

    def _player(self, shirt="6", name="Kimmich, Joshua", pos="MZ",
                 status="Active", start="1", pid="DFL-OBJ-001"):
        return {
            "ShirtNumber": shirt, "Name": name, "PlayingPosition": pos,
            "Status": status, "StartEleven": start, "PlayerId": pid,
        }

    def test_home_team_parsed_as_team_1(self):
        root = _make_match_info_xml(home_players=[self._player()])
        players = extract_players_from_match_info(root)
        home = [p for p in players if p["team"] == 1]
        assert len(home) == 1
        assert home[0]["jersey"] == 6

    def test_guest_team_parsed_as_team_0(self):
        root = _make_match_info_xml(guest_players=[self._player(shirt="9")])
        players = extract_players_from_match_info(root)
        away = [p for p in players if p["team"] == 0]
        assert len(away) == 1
        assert away[0]["jersey"] == 9

    def test_away_team_fallback(self):
        """AwayTeam tag (variant feed) is treated as team=0."""
        root = _make_match_info_xml(away_players=[self._player(shirt="9")])
        players = extract_players_from_match_info(root)
        away = [p for p in players if p["team"] == 0]
        assert len(away) == 1

    def test_guest_takes_precedence_over_away(self):
        """Both GuestTeam and AwayTeam present: only GuestTeam counted."""
        root = _make_match_info_xml(
            guest_players=[self._player(shirt="9")],
            away_players=[self._player(shirt="11")],
        )
        players = extract_players_from_match_info(root)
        away = [p for p in players if p["team"] == 0]
        assert len(away) == 1
        assert away[0]["jersey"] == 9

    def test_shirt_number_parsed_as_int(self):
        root = _make_match_info_xml(home_players=[self._player(shirt="17")])
        players = extract_players_from_match_info(root)
        assert players[0]["jersey"] == 17

    def test_invalid_shirt_number_defaults_to_minus_one(self):
        root = _make_match_info_xml(home_players=[self._player(shirt="N/A")])
        players = extract_players_from_match_info(root)
        assert players[0]["jersey"] == -1

    def test_start_eleven_true(self):
        root = _make_match_info_xml(home_players=[self._player(start="1")])
        assert extract_players_from_match_info(root)[0]["start_eleven"] is True

    def test_start_eleven_false(self):
        root = _make_match_info_xml(home_players=[self._player(start="0")])
        assert extract_players_from_match_info(root)[0]["start_eleven"] is False

    def test_name_and_position_extracted(self):
        root = _make_match_info_xml(home_players=[self._player(name="Mueller, Thomas", pos="STZ")])
        p = extract_players_from_match_info(root)[0]
        assert p["name"] == "Mueller, Thomas"
        assert p["position"] == "STZ"

    def test_multiple_players(self):
        root = _make_match_info_xml(
            home_players=[self._player(shirt="6"), self._player(shirt="25")],
            guest_players=[self._player(shirt="9")],
        )
        players = extract_players_from_match_info(root)
        assert len(players) == 3

    def test_missing_players_element_returns_empty(self):
        root = ET.Element("MatchInformation")
        teams = ET.SubElement(root, "Teams")
        ET.SubElement(teams, "HomeTeam", {"TeamId": "X"})  # no <Players> child
        players = extract_players_from_match_info(root)
        assert players == []

    def test_missing_teams_element_returns_empty(self):
        root = ET.Element("MatchInformation")
        # No <Teams> child at all
        players = extract_players_from_match_info(root)
        assert players == []


# ── Tests: extract_pass_events ───────────────────────────────────────────────

class TestExtractPassEvents:

    def test_returns_correct_columns(self):
        root = _make_events_xml([])
        df = extract_pass_events(root)
        assert list(df.columns) == [
            "event_id", "type", "game_section", "team",
            "player_id", "recipient_id", "evaluation",
            "pos_x", "pos_y", "real_time",
        ]

    def test_pass_event_extracted(self):
        root = _make_events_xml([{"type": "pass", "event_id": "42"}])
        df = extract_pass_events(root)
        assert len(df) == 1
        assert df.iloc[0]["type"] == "pass"
        assert df.iloc[0]["event_id"] == "42"

    def test_cross_not_extracted(self):
        """Play without a Pass child (e.g. Cross) must not appear."""
        root = _make_events_xml([{"type": "cross"}])
        df = extract_pass_events(root)
        assert len(df) == 0

    def test_game_section_default_is_first_half(self):
        root = _make_events_xml([{"type": "pass"}])
        df = extract_pass_events(root)
        assert df.iloc[0]["game_section"] == "firstHalf"

    def test_game_section_tracked_by_final_whistle(self):
        events = [
            {"type": "pass"},                                    # firstHalf
            {"type": "final_whistle", "game_section": "firstHalf"},
            {"type": "pass"},                                    # still firstHalf after whistle
        ]
        root = _make_events_xml(events)
        df = extract_pass_events(root)
        assert len(df) == 2
        # Both passes are in firstHalf context; second pass comes after FinalWhistle
        # which sets section to firstHalf — unchanged.
        assert all(df["game_section"] == "firstHalf")

    def test_game_section_changes_at_kickoff(self):
        events = [
            {"type": "pass"},                                          # firstHalf
            {"type": "kickoff", "tag": "KickOff", "game_section": "secondHalf"},
            {"type": "pass"},                                          # secondHalf
        ]
        root = _make_events_xml(events)
        df = extract_pass_events(root)
        assert len(df) == 2
        assert df.iloc[0]["game_section"] == "firstHalf"
        assert df.iloc[1]["game_section"] == "secondHalf"

    def test_kickoff_lowercase_variant(self):
        """Kickoff (lowercase k) tag is also handled."""
        events = [
            {"type": "kickoff", "tag": "Kickoff", "game_section": "secondHalf"},
            {"type": "pass"},
        ]
        root = _make_events_xml(events)
        df = extract_pass_events(root)
        assert df.iloc[0]["game_section"] == "secondHalf"

    def test_recipient_extracted(self):
        root = _make_events_xml([{
            "type": "pass", "play_recipient": "DFL-OBJ-999",
        }])
        df = extract_pass_events(root)
        assert df.iloc[0]["recipient_id"] == "DFL-OBJ-999"

    def test_missing_recipient_defaults_empty_string(self):
        root = _make_events_xml([{"type": "pass"}])  # no play_recipient key
        df = extract_pass_events(root)
        assert df.iloc[0]["recipient_id"] == ""

    def test_pos_x_pos_y_always_none(self):
        """Coordinates are not in the DFL EventData feed."""
        root = _make_events_xml([{"type": "pass"}])
        df = extract_pass_events(root)
        assert df.iloc[0]["pos_x"] is None
        assert df.iloc[0]["pos_y"] is None

    def test_empty_xml_returns_empty_dataframe(self):
        root = ET.Element("EventData")
        df = extract_pass_events(root)
        assert len(df) == 0
        assert "event_id" in df.columns

    def test_non_standard_root_tag_still_works(self):
        """PutDataRequest or any other root tag: Event descendants still found."""
        root = _make_events_xml([{"type": "pass"}], root_tag="PutDataRequest")
        df = extract_pass_events(root)
        assert len(df) == 1

    def test_multiple_passes_all_extracted(self):
        events = [{"type": "pass"} for _ in range(5)]
        root = _make_events_xml(events)
        df = extract_pass_events(root)
        assert len(df) == 5

    def test_team_and_player_extracted(self):
        root = _make_events_xml([{
            "type": "pass",
            "play_team": "DFL-CLU-FCB",
            "play_player": "DFL-OBJ-Kimmich",
            "play_evaluation": "successful",
        }])
        df = extract_pass_events(root)
        assert df.iloc[0]["team"] == "DFL-CLU-FCB"
        assert df.iloc[0]["player_id"] == "DFL-OBJ-Kimmich"
        assert df.iloc[0]["evaluation"] == "successful"
