"""
event_parser.py

Parses DFL MatchInformation XML, match metadata, and Events XML for the AWI pipeline.

Key data sources:
  - Parquet TF15 metadata header: phase frame boundaries (primary per TF15 1.1 spec)
  - MatchInformation XML (Sportec Solutions format): player roster with jersey numbers
  - Events XML (DFL-03-EventData-Match-Raw): pass events for context enrichment

DFL MatchInformation XML structure (Sportec Solutions):
  <MatchInformation>
    <General MatchId="DFL-MAT-..." HomeTeamId="DFL-CLU-..." GuestTeamId="DFL-CLU-..."/>
    <Teams>
      <HomeTeam TeamId="DFL-CLU-...">
        <Players>
          <Player PlayerId="DFL-OBJ-..." ShirtNumber="6" Name="Kimmich, Joshua"
                  PlayingPosition="MZ" Status="Active" StartEleven="1"/>
        </Players>
      </HomeTeam>
      <GuestTeam TeamId="DFL-CLU-...">
        <Players>
          <Player ShirtNumber="9" Name="..." PlayingPosition="STZ" .../>
        </Players>
      </GuestTeam>
    </Teams>
  </MatchInformation>
"""

import json
import xml.etree.ElementTree as ET
import pandas as pd


# ── Phase labels (TF15 section numbers) ─────────────────────────────────────

_PHASE_LABELS = {
    "1": "1st half",
    "2": "2nd half",
    "3": "1st extra time",
    "4": "2nd extra time",
    "5": "penalty shootout",
}


# ── Phase extraction ─────────────────────────────────────────────────────────

def extract_phases_from_metadata(metadata: dict) -> list[dict]:
    """Extract match phase frame boundaries from metadata.

    Primary source (per TF15 1.1 spec): Parquet key-value metadata header,
    accessed via ``pq.ParquetFile(...).metadata.metadata`` — a dict with
    bytes keys. Pass that dict directly to this function.

    Also accepts str-keyed dicts (e.g. from a parsed metadata JSON) and tries
    several common JSON key conventions as fallbacks.

    Args:
        metadata: Raw metadata dict. Keys may be bytes (Parquet KV) or str.

    Returns:
        List of phase dicts ordered by section number, each containing:
        {section (str), label (str), start_frame (int), end_frame (int)}.
        Phases with zero start/end (unused slots) are omitted.

    Raises:
        ValueError: If no known structure is found, with diagnostic key list.
    """
    # Normalise bytes keys/values to str (Parquet KV gives bytes)
    norm: dict[str, str] = {}
    for k, v in metadata.items():
        sk = k.decode() if isinstance(k, bytes) else str(k)
        sv = v.decode() if isinstance(v, bytes) else str(v)
        norm[sk] = sv

    # ── Strategy A: TF15 Parquet KV keys (phase_N_start / phase_N_end) ──────
    phases = []
    for n in ("1", "2", "3", "4", "5"):
        s_key, e_key = f"phase_{n}_start", f"phase_{n}_end"
        if s_key in norm and e_key in norm:
            try:
                s, e = int(norm[s_key]), int(norm[e_key])
            except ValueError:
                continue
            if s > 0 and e > 0:
                phases.append({
                    "section": n,
                    "label": _PHASE_LABELS.get(n, f"Phase {n}"),
                    "start_frame": s,
                    "end_frame": e,
                })
    if phases:
        return phases

    # ── Strategy B: camelCase flat JSON (phase1Start / phase1End) ───────────
    phases = []
    for n, label in _PHASE_LABELS.items():
        s_key = f"phase{n}Start"
        e_key = f"phase{n}End"
        if s_key in norm and e_key in norm:
            try:
                s, e = int(norm[s_key]), int(norm[e_key])
            except ValueError:
                continue
            if s > 0 and e > 0:
                phases.append({"section": n, "label": label,
                                "start_frame": s, "end_frame": e})
    if phases:
        return phases

    # ── Strategy C: nested list under "sections" / "periods" / "phases" ─────
    for container_key in ("sections", "periods", "phases", "halves", "matchParts"):
        raw_val = norm.get(container_key)
        if raw_val is None:
            continue
        try:
            items = json.loads(raw_val)
        except (ValueError, TypeError):
            continue
        if not isinstance(items, list):
            continue
        phases = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            # Try multiple key name conventions for start/end
            s = None
            for sk in ("start_frame", "startFrame", "StartFrameCount", "start", "kickoffFrame"):
                if sk in item:
                    try:
                        s = int(item[sk])
                    except (ValueError, TypeError):
                        pass
                    break
            e = None
            for ek in ("end_frame", "endFrame", "EndFrameCount", "end", "finalWhistleFrame"):
                if ek in item:
                    try:
                        e = int(item[ek])
                    except (ValueError, TypeError):
                        pass
                    break
            if s is None or e is None or s == 0 or e == 0:
                continue
            sec = str(item.get("section", item.get("period", item.get("half", str(i + 1)))))
            phases.append({"section": sec,
                           "label": _PHASE_LABELS.get(sec, f"Phase {sec}"),
                           "start_frame": s, "end_frame": e})
        if phases:
            return phases

    # ── Strategy D: top-level verbose keys ──────────────────────────────────
    mapping = [
        ("firstHalfStart",  "firstHalfEnd",  "1"),
        ("secondHalfStart", "secondHalfEnd", "2"),
        ("extraTime1Start", "extraTime1End", "3"),
        ("extraTime2Start", "extraTime2End", "4"),
    ]
    phases = []
    for sk, ek, n in mapping:
        if sk in norm and ek in norm:
            try:
                s, e = int(norm[sk]), int(norm[ek])
            except ValueError:
                continue
            if s > 0 and e > 0:
                phases.append({"section": n,
                               "label": _PHASE_LABELS.get(n, f"Phase {n}"),
                               "start_frame": s, "end_frame": e})
    if phases:
        return phases

    raise ValueError(
        f"Unrecognized metadata format. Top-level keys (first 10): "
        f"{list(norm.keys())[:10]}. "
        "Add a new strategy to extract_phases_from_metadata() in event_parser.py."
    )


# ── Player roster ────────────────────────────────────────────────────────────

def extract_players_from_match_info(root: ET.Element) -> list[dict]:
    """Parse DFL MatchInformation XML (Sportec format) to extract player roster.

    The actual DFL feed structure is:
        PutDataRequest > MatchInformation > Teams > Team (repeated)
    Teams are listed home-first by DFL convention; team code is assigned by
    position (first Team = 1/home, second Team = 0/away).

    Args:
        root: Parsed root ET.Element (may be PutDataRequest wrapper or
              MatchInformation directly).

    Returns:
        List of player dicts:
        {jersey (int), team (int: 1=home, 0=away), name (str),
         position (str), player_id (str)}.
        Returns empty list if XML structure is unexpected.
    """
    players = []
    # Unwrap PutDataRequest envelope if present
    if root.tag != "MatchInformation":
        root = root.find("MatchInformation") or root

    teams_el = root.find("Teams")
    if teams_el is None:
        return players

    # Support both named tags (HomeTeam/GuestTeam/AwayTeam) and generic <Team> ordering.
    # Named tags take precedence; fall back to positional <Team> children (home first).
    home_el = teams_el.find("HomeTeam")
    guest_el = teams_el.find("GuestTeam") or teams_el.find("AwayTeam")

    if home_el is not None or guest_el is not None:
        named_pairs = []
        if home_el is not None:
            named_pairs.append((home_el, 1))
        if guest_el is not None:
            named_pairs.append((guest_el, 0))
        team_pairs = named_pairs
    else:
        # Generic <Team> children: first = home, second = away
        team_pairs = list(zip(teams_el.findall("Team"), [1, 0]))

    def _parse_players(team_el, team_code):
        players_el = team_el.find("Players")
        if players_el is None:
            return
        for p in players_el.findall("Player"):
            shirt = p.get("ShirtNumber", "-1")
            name = f"{p.get('FirstName', '')} {p.get('LastName', p.get('Name', ''))}".strip()
            try:
                jersey = int(shirt)
            except ValueError:
                jersey = -1
            start_raw = p.get("StartEleven", "")
            players.append({
                "jersey":       jersey,
                "team":         team_code,
                "name":         name,
                "position":     p.get("PlayingPosition", ""),
                "player_id":    p.get("PersonId", p.get("PlayerId", "")),
                "start_eleven": start_raw == "1",
            })

    for team_el, team_code in team_pairs:
        _parse_players(team_el, team_code)

    return players


# ── Pass events ──────────────────────────────────────────────────────────────

_PASS_COLUMNS = [
    "event_id", "type", "game_section", "team",
    "player_id", "recipient_id", "evaluation",
    "pos_x", "pos_y", "real_time",
]


def extract_pass_events(events_root: ET.Element) -> pd.DataFrame:
    """Extract pass events from DFL Events XML (DFL-03-EventData format).

    DFL Events XML structure::

        <EventData>   <!-- or <PutDataRequest> -->
          <Event EventId="1" MatchId="..." EventTime="2025-09-13T15:31:00.000+02:00">
            <Play Team="DFL-CLU-..." Player="DFL-OBJ-..." Recipient="DFL-OBJ-..."
                  Evaluation="successfulDribbling" Height="..." Distance="...">
              <Pass Direction="..." FreeKickLayup="..."/>
            </Play>
          </Event>
          <Event ...>
            <FinalWhistle GameSection="firstHalf"/>
          </Event>
          <Event ...>
            <KickOff GameSection="secondHalf" .../>
          </Event>
        </EventData>

    Notes:
        - ``pos_x`` / ``pos_y`` are **not** in the EventData feed; always ``None``.
        - ``game_section`` is tracked by scanning ``FinalWhistle`` and ``KickOff``
          events in document order.
        - Root tag is flexible: works regardless of ``<PutDataRequest>``,
          ``<EventData>``, or any other wrapper.

    Args:
        events_root: Parsed root ET.Element of Events XML.

    Returns:
        DataFrame with columns: event_id, type, game_section, team, player_id,
        recipient_id, evaluation, pos_x, pos_y, real_time.
        Empty DataFrame (correct columns) if no pass events found.
    """
    rows = []
    current_section = "firstHalf"

    for event_el in events_root.iter("Event"):
        event_id = event_el.get("EventId", "")
        event_time = event_el.get("EventTime", "")

        # Track game section from section-marker events (in document order)
        fw = event_el.find("FinalWhistle")
        if fw is not None:
            gs = fw.get("GameSection")
            if gs:
                current_section = gs

        for ko_tag in ("KickOff", "Kickoff", "kickOff"):
            ko = event_el.find(ko_tag)
            if ko is not None:
                gs = ko.get("GameSection")
                if gs:
                    current_section = gs
                break

        play_el = event_el.find("Play")
        if play_el is None:
            continue
        if play_el.find("Pass") is None:
            continue  # Cross, Shot attempt, etc. — skip non-pass plays

        rows.append({
            "event_id": event_id,
            "type": "pass",
            "game_section": current_section,
            "team": play_el.get("Team", ""),
            "player_id": play_el.get("Player", ""),
            "recipient_id": play_el.get("Recipient", ""),
            "evaluation": play_el.get("Evaluation", ""),
            "pos_x": None,   # not present in DFL EventData feed
            "pos_y": None,
            "real_time": event_time,
        })

    if not rows:
        return pd.DataFrame(columns=_PASS_COLUMNS)
    return pd.DataFrame(rows)[_PASS_COLUMNS]
