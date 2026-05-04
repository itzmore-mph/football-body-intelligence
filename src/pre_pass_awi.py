"""
pre_pass_awi.py

Computes per-player AWI in a fixed window before each pass event.

The core question: do players who scan more in the seconds before a pass
make better decisions? Pre-pass AWI bridges the AWI metric and game events.

Data flow:
    Events XML  →  extract_pass_events()   →  pass_df (with EventTime)
    Events XML  →  extract_kickoff_times() →  kickoff_times (section → datetime)
    phases      →  (phase section → start_frame map)
    scan_df     →  detect_scans() output   →  frame-level is_scan per player

    pass_events_to_frames() converts EventTime → frame_number using kickoff reference.
    compute_pre_pass_awi()  computes scan event rate in the window before each pass.

Output columns added to the AWI results DataFrame:
    pre_pass_scan_pct   float   Mean fraction of frames scanning in the 5s pre-pass window
    pre_pass_awi        float   Scan events/min in the 5s pre-pass window
    n_passes_tracked    int     Number of passes with sufficient skeleton coverage
"""

from datetime import datetime, timezone
import xml.etree.ElementTree as ET

import pandas as pd


# ── Phase timing ─────────────────────────────────────────────────────────────

_SECTION_LABEL_MAP = {
    "firstHalf":       "1st half",
    "secondHalf":      "2nd half",
    "firstExtraTime":  "1st extra time",
    "secondExtraTime": "2nd extra time",
}


def extract_kickoff_times(events_root: ET.Element) -> dict[str, datetime]:
    """Extract real-world kickoff timestamps for each game section.

    Scans Events XML in document order for KickOff events and records the
    first occurrence per game section. Used as the time reference to convert
    EventTime strings to frame numbers.

    Args:
        events_root: Parsed ET.Element root of Events XML (any wrapper tag).

    Returns:
        Dict mapping DFL game_section string to timezone-aware datetime, e.g.:
        {"firstHalf": datetime(2025, 9, 13, 13, 31, 0, tzinfo=...), ...}
        Empty dict if no KickOff events are found.
    """
    kickoff_times: dict[str, datetime] = {}
    for event_el in events_root.iter("Event"):
        event_time_str = event_el.get("EventTime", "")
        for ko_tag in ("KickOff", "Kickoff", "kickOff"):
            ko = event_el.find(ko_tag)
            if ko is not None:
                gs = ko.get("GameSection")
                if gs and gs not in kickoff_times and event_time_str:
                    try:
                        kickoff_times[gs] = datetime.fromisoformat(event_time_str)
                    except ValueError:
                        pass
                break
    return kickoff_times


def build_phase_frame_map(phases: list[dict]) -> dict[str, int]:
    """Build a mapping from game_section string to phase_start_frame.

    Converts the phase list from extract_phases_from_metadata() (which uses
    section numbers like "1", "2") to DFL game_section strings ("firstHalf" etc.).

    Args:
        phases: List of phase dicts with keys: section, label, start_frame, end_frame.

    Returns:
        Dict mapping DFL section string to start_frame, e.g.:
        {"firstHalf": 3_330_943, "secondHalf": 3_536_417}
    """
    _SECTION_NUM_TO_DFL = {
        "1": "firstHalf",
        "2": "secondHalf",
        "3": "firstExtraTime",
        "4": "secondExtraTime",
    }
    return {
        _SECTION_NUM_TO_DFL[p["section"]]: p["start_frame"]
        for p in phases
        if p["section"] in _SECTION_NUM_TO_DFL
    }


# ── Pass event frame conversion ───────────────────────────────────────────────

def pass_events_to_frames(
    pass_df: pd.DataFrame,
    kickoff_times: dict[str, datetime],
    phase_frame_map: dict[str, int],
    framerate: int = 50,
) -> pd.DataFrame:
    """Append approximate frame_number to each pass event row.

    Converts EventTime to a frame number using the kickoff event as the
    time-zero reference for each game section:

        frame = phase_start_frame + round((event_time - kickoff_time) * fps)

    Rows where conversion fails (missing kickoff time, bad timestamp, section
    not in phase_frame_map) get frame_number = pd.NA.

    Args:
        pass_df:        DataFrame from extract_pass_events() with real_time and
                        game_section columns.
        kickoff_times:  Dict from extract_kickoff_times().
        phase_frame_map: Dict from build_phase_frame_map().
        framerate:      Frames per second (default 50).

    Returns:
        Copy of pass_df with new int column frame_number (pd.NA where unknown).
    """
    df = pass_df.copy()
    df["frame_number"] = pd.array([pd.NA] * len(df), dtype="Int64")

    for section, start_frame in phase_frame_map.items():
        ko_time = kickoff_times.get(section)
        if ko_time is None:
            continue
        mask = df["game_section"] == section
        if not mask.any():
            continue

        def _convert(ts: str, ko=ko_time, sf=start_frame) -> "int | pd.NA":
            try:
                et = datetime.fromisoformat(ts)
                # Ensure both are timezone-aware or both naive
                if et.tzinfo is None and ko.tzinfo is not None:
                    et = et.replace(tzinfo=timezone.utc)
                elif et.tzinfo is not None and ko.tzinfo is None:
                    ko = ko.replace(tzinfo=timezone.utc)
                delta_s = (et - ko).total_seconds()
                return sf + round(delta_s * framerate)
            except (ValueError, TypeError, AttributeError):
                return pd.NA

        df.loc[mask, "frame_number"] = df.loc[mask, "real_time"].map(_convert)
    return df


# ── Pre-pass AWI computation ──────────────────────────────────────────────────

def compute_pre_pass_awi(
    scan_df: pd.DataFrame,
    pass_frames: pd.Series,
    window_frames: int = 250,
    min_coverage: float = 0.5,
    framerate: int = 50,
) -> dict:
    """Compute AWI rate in a fixed window before each pass.

    For each pass frame, counts leading-edge scan events in the window
    [frame - window_frames, frame) and converts to a scans/min rate.
    Averages across all passes with sufficient skeleton coverage.

    Args:
        scan_df:       DataFrame with columns [frame_number, is_scan].
        pass_frames:   Series of frame numbers where this player passed.
        window_frames: Lookback window in frames (default 250 = 5s at 50 fps).
        min_coverage:  Minimum fraction of window frames present in scan_df to
                       include a pass in the average (default 0.5).
        framerate:     Frames per second (default 50).

    Returns:
        Dict with:
            pre_pass_awi        float | None  Scan events/min in pre-pass window
            pre_pass_scan_pct   float | None  Mean fraction of window where scanning
            n_passes_tracked    int           Passes included in the average
    """
    empty = {"pre_pass_awi": None, "pre_pass_scan_pct": None, "n_passes_tracked": 0}
    if scan_df.empty or pass_frames.dropna().empty:
        return empty

    scan_idx = scan_df.set_index("frame_number")["is_scan"]
    window_minutes = window_frames / framerate / 60

    awi_rates = []
    scan_pcts = []

    for frame in pass_frames.dropna():
        frame = int(frame)
        w_start = frame - window_frames
        window = scan_idx.loc[(scan_idx.index >= w_start) & (scan_idx.index < frame)]

        if len(window) < window_frames * min_coverage:
            continue  # insufficient skeleton coverage in this window

        # Leading-edge count = discrete scan events in the window
        prev = window.shift(1, fill_value=False)
        event_count = int((window & ~prev).sum())

        awi_rates.append(event_count / window_minutes)
        scan_pcts.append(float(window.mean()))

    if not awi_rates:
        return empty

    s = pd.Series(awi_rates)
    return {
        "pre_pass_awi":      round(float(s.mean()), 2),
        "pre_pass_scan_pct": round(float(pd.Series(scan_pcts).mean()), 4),
        "n_passes_tracked":  len(awi_rates),
    }


# ── Match moments (per-pass scan data) ───────────────────────────────────────

def compute_pre_pass_awi_per_pass(
    scan_df: pd.DataFrame,
    pass_frames: pd.Series,
    jersey: int,
    team: int,
    name: str,
    match_id: str,
    phase_label: str,
    phase_start: int,
    window_frames: int = 250,
    min_coverage: float = 0.5,
    framerate: int = 50,
) -> list[dict]:
    """Return per-pass pre-scan data for match moment detection.

    Like compute_pre_pass_awi() but returns individual rows per pass rather than
    an average, enabling the caller to rank moments and extract the top-N.

    Each returned dict represents one pass and includes the scan count in the
    5-second window before it - the raw material for broadcast highlights:
    "Kimmich performed 4 scans in the 5s before his assist at minute 67."

    Args:
        scan_df:       DataFrame with [frame_number, is_scan] from detect_scans().
        pass_frames:   Series of frame numbers where this player passed.
        jersey:        Player jersey number.
        team:          Team encoding (1=home, 0=away).
        name:          Player display name.
        match_id:      Match identifier string.
        phase_label:   Phase label string (e.g. "1st half").
        phase_start:   Start frame of the phase (for minute calculation).
        window_frames: Lookback window in frames (default 250 = 5s at 50 fps).
        min_coverage:  Minimum fraction of window frames present in scan_df.
        framerate:     Frames per second (default 50).

    Returns:
        List of dicts, one per qualifying pass:
        {match_id, phase_label, jersey, team, name, pass_frame, minute,
         pre_pass_scan_count, window_coverage}.
        Empty list if no qualifying passes.
    """
    if scan_df.empty or pass_frames.dropna().empty:
        return []

    scan_idx = scan_df.set_index("frame_number")["is_scan"]
    rows = []

    for frame in pass_frames.dropna():
        frame = int(frame)
        w_start = frame - window_frames
        window = scan_idx.loc[(scan_idx.index >= w_start) & (scan_idx.index < frame)]

        if len(window) < window_frames * min_coverage:
            continue

        prev = window.shift(1, fill_value=False)
        event_count = int((window & ~prev).sum())
        minute = round((frame - phase_start) / framerate / 60, 1)

        rows.append({
            "match_id":            match_id,
            "phase_label":         phase_label,
            "jersey":              jersey,
            "team":                team,
            "name":                name,
            "pass_frame":          frame,
            "minute":              minute,
            "pre_pass_scan_count": event_count,
            "window_coverage":     round(len(window) / window_frames, 3),
        })

    return rows


# ── Match-level enrichment ────────────────────────────────────────────────────

def enrich_match_with_pre_pass_awi(
    phase_df: pd.DataFrame,
    players: list[dict],
    phases: list[dict],
    events_root: ET.Element,
    match_id: str,
    framerate: int = 50,
) -> pd.DataFrame:
    """Compute pre-pass AWI for all players in a match and return an enrichment DataFrame.

    Designed to run alongside the main AWI pipeline (in run_match_awi) using
    the same phase_df that was already loaded from S3.

    Args:
        phase_df:     DataFrame[frame_number, skeletons] for one phase.
        players:      Player list from extract_players_from_match_info().
        phases:       Phase list from extract_phases_from_metadata().
        events_root:  Parsed Events XML root.
        match_id:     Match identifier string.
        framerate:    Frames per second.

    Returns:
        DataFrame with columns: jersey, team, match_id, phase_label,
        pre_pass_awi, pre_pass_scan_pct, n_passes_tracked.
        One row per player × phase. Merge with results/awi_full.csv on
        (jersey, team, match_id, phase_label).
    """
    from src.skeleton_parser import extract_head_angles_batch
    from src.awi_calculator import detect_scans
    from src.event_parser import extract_pass_events

    kickoff_times = extract_kickoff_times(events_root)
    phase_frame_map = build_phase_frame_map(phases)
    pass_df_raw = extract_pass_events(events_root)

    if pass_df_raw.empty or not kickoff_times:
        return pd.DataFrame()

    pass_df = pass_events_to_frames(pass_df_raw, kickoff_times, phase_frame_map, framerate)

    player_keys = [(p["jersey"], p["team"]) for p in players]
    player_meta = {(p["jersey"], p["team"]): p for p in players}
    angles_by_player = extract_head_angles_batch(phase_df, player_keys)

    results = []
    for phase_info in phases:
        section_dfl = {v: k for k, v in {
            "firstHalf": "1st half", "secondHalf": "2nd half",
            "firstExtraTime": "1st extra time", "secondExtraTime": "2nd extra time",
        }.get("", {})}.get(phase_info["label"], None)

        for key, head_df in angles_by_player.items():
            if head_df.empty:
                continue

            scan_df = detect_scans(head_df)

            # Filter passes: this player, this section
            player_id = player_meta[key].get("player_id", "")
            section_label = phase_info["label"]
            dfl_section = next(
                (k for k, v in _SECTION_LABEL_MAP.items() if v == section_label), None
            )
            player_passes = pass_df[
                (pass_df["player_id"] == player_id) &
                (pass_df["game_section"] == dfl_section)
            ]["frame_number"] if dfl_section else pd.Series(dtype="Int64")

            awi_info = compute_pre_pass_awi(scan_df, player_passes, framerate=framerate)
            results.append({
                "jersey":             key[0],
                "team":               key[1],
                "match_id":           match_id,
                "phase_label":        section_label,
                **awi_info,
            })

    return pd.DataFrame(results)
