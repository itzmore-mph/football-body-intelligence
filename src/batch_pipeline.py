"""
batch_pipeline.py

Orchestrates multi-player, multi-phase, multi-match AWI computation.

Pipeline per match:
  1. Read phase boundaries from Parquet TF15 metadata header (pf.metadata.metadata)
  2. Read player roster from MatchInformation XML
  3. For each phase:
     a. Load parquet with frame filter + column pruning (frame_number, skeletons)
     b. extract_head_angles_batch() - single pass for all players
     c. detect_scans() + compute_awi() per player
     d. del phase_df to free ~200-400 MB before loading the next phase
  4. Concatenate results into a tidy DataFrame

Memory note:
  A 50-min phase at 50 fps = ~150k rows. With just frame_number + skeletons
  and ~25 players per frame, this decompresses to roughly 200-400 MB in pandas.
  Never hold two phase DataFrames in memory simultaneously.

S3 path conventions (challenge_prefix already excludes bucket):
  parquet  : {bucket}/{challenge_prefix}/{parquet_key}
  XML keys : {challenge_prefix}/{key}

MATCH_CONFIGS defines all 5 matches. Folder names for BVB-VFB, SGE-FCB, SGE-FCU,
FCU-FCB are placeholders - verify with list_bucket() on first run.
"""

import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq  # noqa: F401 (kept for tests that patch src.batch_pipeline.pq.read_table)
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pandas as pd

from src.skeleton_parser import extract_head_angles_batch, PART, _compute_yaw
from src.awi_calculator import detect_scans, compute_awi
from src.angle_utils import circular_diff
from src.event_parser import extract_players_from_match_info
from src.eda_helpers import load_xml
from src.pipeline_io import (
    load_phases_from_parquet as _load_phases_from_parquet,
    load_phase_df,
    read_phase_table_with_retry,
)

# Public re-exports so downstream modules and tests can patch them via
# ``src.batch_pipeline.<name>`` (patching through the alias updates this
# module's global binding, which is what the runners resolve at call time).
__all__ = [
    "MATCH_CONFIGS",
    "load_phase_df",
    "stream_player_angles",
    "compute_phase_awi_all_players",
    "run_match_awi",
    "run_all_matches",
    "run_match_unified",
    "run_all_matches_unified",
]


# ── Match configuration ──────────────────────────────────────────────────────
# parquet_key and match_info_key are relative to challenge_prefix.
# Verify folder names for the 4 unknown matches with list_bucket() output from
# the EDA notebook (Cell 1) before running the full pipeline.

MATCH_CONFIGS = [
    {
        "match_id":       "FCB-HSV",
        "label":          "FC Bayern Muenchen vs Hamburger SV",
        "parquet_key":    "Bayern_Hamburg/FCB-HSV.parquet",
        "match_info_key": "Bayern_Hamburg/MatchInformations_Bayern_Hamburg.xml",
    },
    {
        "match_id":       "BVB-VFB",
        "label":          "Borussia Dortmund vs VfB Stuttgart",
        "parquet_key":    "Dortmund_Stuttgart/BVB-VFB.parquet",
        "match_info_key": "Dortmund_Stuttgart/MatchInformations_Dortmund_Sttutgart.xml",
    },
    {
        "match_id":       "SGE-FCB",
        "label":          "Eintracht Frankfurt vs FC Bayern Muenchen",
        "parquet_key":    "Frankfurt_Bayern/SGE-FCB.parquet",
        "match_info_key": "Frankfurt_Bayern/MatchInformations_Frankfurt_Bayern.xml",
    },
    {
        "match_id":       "SGE-FCU",
        "label":          "Eintracht Frankfurt vs Union Berlin",
        "parquet_key":    "Frankfurt_Union/SGE-FCU.parquet",
        "match_info_key": "Frankfurt_Union/MatchInformations_Frankfurt_Union.xml",
    },
    {
        "match_id":       "FCU-FCB",
        "label":          "Union Berlin vs FC Bayern Muenchen",
        "parquet_key":    "Union_Bayern/FCU-FCB.parquet",
        "match_info_key": "Union_Bayern/MatchInformations_Union_Bayern.xml",
    },
]


# ── Low-level helpers ────────────────────────────────────────────────────────
# Phase-boundary and parquet-read helpers live in src.pipeline_io (imported
# above as _load_phases_from_parquet, load_phase_df, read_phase_table_with_retry).

# Part IDs needed for vectorized yaw extraction
_NOSE, _NECK = PART["nose"], PART["neck"]
_LEAR, _REAR  = PART["left_ear"], PART["right_ear"]
_LSHO, _RSHO  = PART["left_shoulder"], PART["right_shoulder"]
_LHIP, _RHIP  = PART["left_hip"], PART["right_hip"]


def _extract_angles_vectorized(
    table: pa.Table,
    player_keys: list[tuple[int, int]],
) -> dict[tuple[int, int], pd.DataFrame]:
    """Extract head and body yaw using PyArrow + numpy - no Python loops per frame.

    Flattens the nested skeletons ListArray into numpy arrays, filters to target
    players with a boolean mask, then computes all atan2 values vectorized.
    Roughly 20-50x faster than itertuples + Python dict iteration.

    Args:
        table:       PyArrow Table with columns [frame_number, skeletons].
        player_keys: (jersey_number, team) pairs to extract.

    Returns:
        Dict mapping (jersey, team) -> DataFrame[frame_number, head_yaw_deg, body_yaw_deg].
        body_yaw_deg is NaN where shoulder/hip joints are unavailable.
    """
    empty = {k: pd.DataFrame(columns=["frame_number", "head_yaw_deg", "body_yaw_deg"])
             for k in player_keys}
    player_set = set(player_keys)

    # Flatten skeletons: ListArray<struct<...>> -> flat StructArray (one row per frame-player)
    skel_col = table.column("skeletons")
    if isinstance(skel_col, pa.ChunkedArray):
        skel_col = skel_col.combine_chunks()

    skel_lengths = pc.list_value_length(skel_col).fill_null(0)
    skel_flat = pc.list_flatten(skel_col)
    if len(skel_flat) == 0:
        return empty

    # Vectorized jersey/team extraction
    jersey_np = skel_flat.field("jersey_number").to_numpy(zero_copy_only=False)
    team_np   = skel_flat.field("team").to_numpy(zero_copy_only=False)

    # Repeat frame_number to align with flattened skeletons
    frame_np = table.column("frame_number").to_numpy(zero_copy_only=False)
    frame_repeated = np.repeat(frame_np, skel_lengths.to_numpy(zero_copy_only=False))

    # Boolean mask for target players (vectorized, O(n_players * n_skeletons))
    player_mask = np.zeros(len(jersey_np), dtype=bool)
    for j, t in player_set:
        player_mask |= (jersey_np == j) & (team_np == t)

    filtered_idx = np.where(player_mask)[0]
    if len(filtered_idx) == 0:
        return empty

    jersey_f = jersey_np[filtered_idx]
    team_f   = team_np[filtered_idx]
    frame_f  = frame_repeated[filtered_idx]
    n = len(filtered_idx)

    # Extract parts only for target-player skeletons
    parts_list = skel_flat.field("parts").take(pa.array(filtered_idx, type=pa.int64()))
    if isinstance(parts_list, pa.ChunkedArray):
        parts_list = parts_list.combine_chunks()

    part_lengths = pc.list_value_length(parts_list).fill_null(0).to_numpy(zero_copy_only=False)
    parts_flat = pc.list_flatten(parts_list)
    if len(parts_flat) == 0:
        return empty

    # Which skeleton does each part belong to?
    skel_idx_for_part = np.repeat(np.arange(n), part_lengths)

    part_name_np = parts_flat.field("name").to_numpy(zero_copy_only=False)
    part_x_np    = parts_flat.field("position_x").to_numpy(zero_copy_only=False).astype(np.float64)
    part_y_np    = parts_flat.field("position_y").to_numpy(zero_copy_only=False).astype(np.float64)

    # Allocate keypoint position arrays (NaN = joint not present in this frame)
    def _fill(part_id: int) -> tuple[np.ndarray, np.ndarray]:
        px = np.full(n, np.nan)
        py = np.full(n, np.nan)
        m = (part_name_np == part_id)
        if m.any():
            px[skel_idx_for_part[m]] = part_x_np[m]
            py[skel_idx_for_part[m]] = part_y_np[m]
        return px, py

    nose_x, nose_y = _fill(_NOSE)
    neck_x, neck_y = _fill(_NECK)
    lear_x, lear_y = _fill(_LEAR)
    rear_x, rear_y = _fill(_REAR)
    lsho_x, lsho_y = _fill(_LSHO)
    rsho_x, rsho_y = _fill(_RSHO)
    lhip_x, lhip_y = _fill(_LHIP)
    rhip_x, rhip_y = _fill(_RHIP)

    # Head yaw: primary = nose -> neck vector, fallback = ear-based
    dx_nn = nose_x - neck_x
    dy_nn = nose_y - neck_y
    valid_nn = ~(np.isnan(dx_nn) | np.isnan(dy_nn)) & ~((dx_nn == 0.0) & (dy_nn == 0.0))

    head_yaw = np.full(n, np.nan)
    head_yaw[valid_nn] = np.degrees(np.arctan2(dy_nn[valid_nn], dx_nn[valid_nn]))

    ear_dx = lear_x - rear_x
    ear_dy = lear_y - rear_y
    needs_ear = np.isnan(head_yaw)
    if needs_ear.any():
        valid_ear = needs_ear & ~(np.isnan(ear_dx) | np.isnan(ear_dy)) & ~((ear_dx == 0.0) & (ear_dy == 0.0))
        if valid_ear.any():
            # clockwise 90° rotation of left->right ear vector
            head_yaw[valid_ear] = np.degrees(np.arctan2(-ear_dx[valid_ear], ear_dy[valid_ear]))

    # Body yaw: primary = shoulder vector, fallback = hip vector (same formula: atan2(-dx, dy))
    dx_sh = lsho_x - rsho_x
    dy_sh = lsho_y - rsho_y
    valid_sh = ~(np.isnan(dx_sh) | np.isnan(dy_sh)) & ~((dx_sh == 0.0) & (dy_sh == 0.0))

    body_yaw = np.full(n, np.nan)
    body_yaw[valid_sh] = np.degrees(np.arctan2(-dx_sh[valid_sh], dy_sh[valid_sh]))

    dx_hip = lhip_x - rhip_x
    dy_hip = lhip_y - rhip_y
    needs_hip = np.isnan(body_yaw)
    if needs_hip.any():
        valid_hip = needs_hip & ~(np.isnan(dx_hip) | np.isnan(dy_hip)) & ~((dx_hip == 0.0) & (dy_hip == 0.0))
        if valid_hip.any():
            body_yaw[valid_hip] = np.degrees(np.arctan2(-dx_hip[valid_hip], dy_hip[valid_hip]))

    # Build result DataFrame for each player
    valid_head = ~np.isnan(head_yaw)
    if not valid_head.any():
        return empty

    combined = pd.DataFrame({
        "jersey":       jersey_f[valid_head].astype(int),
        "team":         team_f[valid_head].astype(int),
        "frame_number": frame_f[valid_head].astype(int),
        "head_yaw_deg": head_yaw[valid_head],
        "body_yaw_deg": body_yaw[valid_head],
    })

    result: dict[tuple[int, int], pd.DataFrame] = {}
    for j, t in player_set:
        sub = combined.loc[
            (combined["jersey"] == j) & (combined["team"] == t),
            ["frame_number", "head_yaw_deg", "body_yaw_deg"],
        ]
        result[(j, t)] = sub.reset_index(drop=True) if not sub.empty else \
            pd.DataFrame(columns=["frame_number", "head_yaw_deg", "body_yaw_deg"])
    return result


def stream_player_angles(
    s3fs: pafs.S3FileSystem,
    parquet_path: str,
    phase_start: int,
    phase_end: int,
    player_keys: list[tuple[int, int]],
    batch_size: int = 50_000,  # unused; kept for API compatibility
) -> dict[tuple[int, int], pd.DataFrame]:
    """Read a phase from S3 then extract head + body angles via vectorized PyArrow ops.

    Uses :func:`pipeline_io.read_phase_table_with_retry` for the parquet read
    (frame-number filter, column pruning, token-aware retry/backoff), then
    :func:`_extract_angles_vectorized` to compute yaw with numpy instead of
    Python loops. Typical speedup: 20-50x over the previous itertuples approach.

    Returns:
        Dict mapping (jersey, team) -> DataFrame[frame_number, head_yaw_deg, body_yaw_deg].
    """
    table = read_phase_table_with_retry(
        s3fs,
        parquet_path,
        phase_start,
        phase_end,
        columns=["frame_number", "skeletons"],
    )
    result = _extract_angles_vectorized(table, player_keys)
    del table
    return result


def _compute_phase_awi_from_angles(
    angles_by_player: dict[tuple[int, int], pd.DataFrame],
    player_meta: dict[tuple[int, int], dict],
    phase_info: dict,
    match_id: str,
) -> list[dict]:
    """Compute AWI for all players from pre-extracted angle DataFrames.

    Replaces the compute_phase_awi_all_players + extract_head_angles_batch
    (slow Python itertuples) path when angles are already available from
    the vectorized _extract_angles_vectorized extraction.

    Args:
        angles_by_player: (jersey, team) -> DataFrame[frame_number, head_yaw_deg, ...]
        player_meta:      (jersey, team) -> player info dict (name, position, ...)
        phase_info:       {section, label, start_frame, end_frame}
        match_id:         Match identifier string

    Returns:
        List of AWI result dicts, one per player.
    """
    phase_start = phase_info["start_frame"]
    phase_end   = phase_info["end_frame"]
    total_phase_frames = phase_end - phase_start + 1

    results = []
    for key, p in player_meta.items():
        angles_df = angles_by_player.get(
            key, pd.DataFrame(columns=["frame_number", "head_yaw_deg"])
        )
        detected = len(angles_df)
        coverage_pct = round(detected / total_phase_frames, 4) if total_phase_frames > 0 else 0.0

        if angles_df.empty:
            results.append({
                "jersey":        key[0],
                "team":          key[1],
                "name":          p["name"],
                "position":      p["position"],
                "match_id":      match_id,
                "phase_label":   phase_info["label"],
                "phase_start":   phase_start,
                "phase_end":     phase_end,
                "scan_count":    0,
                "total_minutes": 0.0,
                "awi_per_minute": 0.0,
                "coverage_pct":  0.0,
            })
            continue

        scan_df = detect_scans(angles_df[["frame_number", "head_yaw_deg"]])
        awi_result = compute_awi(scan_df, phase_start, phase_end)

        results.append({
            "jersey":        key[0],
            "team":          key[1],
            "name":          p["name"],
            "position":      p["position"],
            "match_id":      match_id,
            "phase_label":   phase_info["label"],
            "phase_start":   phase_start,
            "phase_end":     phase_end,
            "scan_count":    awi_result["scan_count"],
            "total_minutes": awi_result["total_minutes"],
            "awi_per_minute": awi_result["awi_per_minute"],
            "coverage_pct":  coverage_pct,
        })

    return results


def compute_scan_direction_profile(
    scan_df: pd.DataFrame,
    angles_df: pd.DataFrame,
) -> dict:
    """Classify scan events by direction relative to body orientation.

    At each scan leading edge, computes the angular difference between
    head yaw and body yaw to classify scan type:
      - forward:    |head - body| < 60°   (looking where body faces)
      - lateral:    60° <= diff < 120°    (sideways glance)
      - blind_side: diff >= 120°          (looking behind/opposite - highest value)

    Blind-side scans are tactically most valuable: the player gathers information
    from directions their body cannot see, disrupting opponents' model of their
    attention. Defenders who blind-side scan backward anticipate counter-attacks;
    midfielders who scan blind-side laterally pick up third-man runs.

    Args:
        scan_df:   DataFrame from detect_scans() with [frame_number, is_scan].
        angles_df: DataFrame with [frame_number, head_yaw_deg, body_yaw_deg].

    Returns:
        Dict with scan_forward_pct, scan_lateral_pct, scan_blindside_pct,
        n_classified_scans. Percentages are None when no scans can be classified.
    """
    empty = {
        "scan_forward_pct":   None,
        "scan_lateral_pct":   None,
        "scan_blindside_pct": None,
        "n_classified_scans": 0,
    }
    is_scan = scan_df["is_scan"]
    prev = is_scan.shift(1, fill_value=False)
    scan_frames = scan_df.loc[is_scan & ~prev, "frame_number"].reset_index(drop=True)
    if scan_frames.empty:
        return empty

    event_df = pd.merge(
        scan_frames.rename("frame_number").to_frame(),
        angles_df[["frame_number", "head_yaw_deg", "body_yaw_deg"]].dropna(
            subset=["head_yaw_deg", "body_yaw_deg"]
        ),
        on="frame_number",
        how="inner",
    )
    if event_df.empty:
        return empty

    diff = circular_diff(
        event_df["head_yaw_deg"].to_numpy(),
        event_df["body_yaw_deg"].to_numpy(),
    )
    n = len(diff)
    return {
        "scan_forward_pct":   round(int((diff < 60).sum()) / n, 4),
        "scan_lateral_pct":   round(int(((diff >= 60) & (diff < 120)).sum()) / n, 4),
        "scan_blindside_pct": round(int((diff >= 120).sum()) / n, 4),
        "n_classified_scans": n,
    }


def _empty_enrich() -> dict:
    """Return zero/None enrichment dict when event data is unavailable."""
    return {
        "pre_pass_awi":       None,
        "pre_pass_scan_pct":  None,
        "n_passes_tracked":   0,
        "hbd_mean_deg":       None,
        "hbd_n_frames":       0,
        "scan_forward_pct":   None,
        "scan_lateral_pct":   None,
        "scan_blindside_pct": None,
        "n_classified_scans": 0,
    }


def compute_hbd(angles_df: pd.DataFrame) -> dict:
    """Compute Head-Body Decoupling (HBD) from a player's angle DataFrame.

    HBD measures how often a player's head faces a different direction than
    their body - the anatomical signal for a blind-side scan. A high HBD score
    means the player frequently looks in directions their body is not facing.

    Args:
        angles_df: DataFrame[frame_number, head_yaw_deg, body_yaw_deg].

    Returns:
        Dict with hbd_mean_deg (mean angular diff, NaN-excluded) and
        hbd_n_frames (frames where both head and body yaw were available).
    """
    df = angles_df.dropna(subset=["head_yaw_deg", "body_yaw_deg"])
    if df.empty:
        return {"hbd_mean_deg": None, "hbd_n_frames": 0}
    diffs = circular_diff(
        df["head_yaw_deg"].to_numpy(),
        df["body_yaw_deg"].to_numpy(),
    )
    return {
        "hbd_mean_deg": round(float(diffs.mean()), 2),
        "hbd_n_frames": len(df),
    }


# load_phase_df (pandas-returning) is imported from src.pipeline_io at the top
# of the module. stream_player_angles now delegates its parquet read to
# read_phase_table_with_retry, so no separate retry wrapper is required.


# ── Core computation ─────────────────────────────────────────────────────────

def compute_phase_awi_all_players(
    phase_df: pd.DataFrame,
    players: list[dict],
    phase_info: dict,
    match_id: str,
) -> list[dict]:
    """Compute AWI for all players in one phase via a single parquet pass.

    Calls extract_head_angles_batch() once for all players, then runs
    detect_scans() + compute_awi() per player.

    coverage_pct = frames with valid head yaw / total phase frames.
    This reflects data quality: a full-match player should be > 0.85;
    a substitute with limited playing time will be much lower.

    Args:
        phase_df:   DataFrame[frame_number, skeletons] for this phase.
        players:    List of player dicts from extract_players_from_match_info().
        phase_info: {section, label, start_frame, end_frame}.
        match_id:   Match identifier string (e.g. "FCB-HSV").

    Returns:
        List of result dicts, one per player × phase:
        jersey, team, name, position, match_id, phase_label,
        phase_start, phase_end, scan_count, total_minutes,
        awi_per_minute, coverage_pct.
    """
    phase_start = phase_info["start_frame"]
    phase_end   = phase_info["end_frame"]
    total_phase_frames = phase_end - phase_start + 1

    player_keys = [(p["jersey"], p["team"]) for p in players]
    angles_by_player = extract_head_angles_batch(phase_df, player_keys)
    player_meta = {(p["jersey"], p["team"]): p for p in players}

    results = []
    for key, head_df in angles_by_player.items():
        p = player_meta[key]
        detected = len(head_df)
        coverage_pct = round(detected / total_phase_frames, 4) if total_phase_frames > 0 else 0.0

        if head_df.empty:
            results.append({
                "jersey":        key[0],
                "team":          key[1],
                "name":          p["name"],
                "position":      p["position"],
                "match_id":      match_id,
                "phase_label":   phase_info["label"],
                "phase_start":   phase_start,
                "phase_end":     phase_end,
                "scan_count":    0,
                "total_minutes": 0.0,
                "awi_per_minute": 0.0,
                "coverage_pct":  0.0,
            })
            continue

        scan_df = detect_scans(head_df)
        awi = compute_awi(scan_df, phase_start, phase_end)

        results.append({
            "jersey":        key[0],
            "team":          key[1],
            "name":          p["name"],
            "position":      p["position"],
            "match_id":      match_id,
            "phase_label":   phase_info["label"],
            "phase_start":   phase_start,
            "phase_end":     phase_end,
            "scan_count":    awi["scan_count"],
            "total_minutes": awi["total_minutes"],
            "awi_per_minute": awi["awi_per_minute"],
            "coverage_pct":  coverage_pct,
        })

    return results


# ── Match-level orchestration ────────────────────────────────────────────────

def run_match_awi(
    s3_client,
    s3fs: pafs.S3FileSystem,
    bucket: str,
    challenge_prefix: str,
    match_config: dict,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """Full AWI pipeline for one match: all players × all phases.

    Args:
        s3_client:        boto3 S3 client.
        s3fs:             pyarrow S3FileSystem.
        bucket:           S3 bucket name.
        challenge_prefix: S3 prefix for the challenge data (no trailing slash).
        match_config:     One entry from MATCH_CONFIGS.
        checkpoint_path:  Optional path to CSV for phase-level checkpointing.
                          Completed phases are saved after each phase and skipped
                          on re-run, so token expiry mid-match loses at most one
                          phase of work.

    Returns:
        DataFrame with AWI results for all players × all phases.

    Raises:
        RuntimeError: If MatchInformation XML cannot be loaded or phase data
                      is missing.
    """
    import os

    match_id     = match_config["match_id"]
    parquet_path = f"{bucket}/{challenge_prefix}/{match_config['parquet_key']}"
    xml_key      = f"{challenge_prefix}/{match_config['match_info_key']}"

    # Load completed phases from checkpoint
    completed_phases: set[str] = set()
    frames: list[pd.DataFrame] = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        match_rows = existing[existing["match_id"] == match_id] if "match_id" in existing.columns else pd.DataFrame()
        if not match_rows.empty and "phase_label" in match_rows.columns:
            completed_phases = set(match_rows["phase_label"].unique())
            frames.append(match_rows)
            print(f"  [{match_id}] Checkpoint: skipping phases {sorted(completed_phases)}")

    # Load player roster
    match_info_root = load_xml(s3_client, bucket, xml_key)
    if match_info_root is None:
        raise RuntimeError(
            f"[{match_id}] Failed to load MatchInformation XML: s3://{bucket}/{xml_key}"
        )
    players = extract_players_from_match_info(match_info_root)
    # Exclude referees (team=3) and unassigned jersey (-1)
    players = [p for p in players if p["team"] in (0, 1) and p["jersey"] >= 0]

    if not players:
        raise RuntimeError(f"[{match_id}] No valid players found in MatchInformation XML.")

    player_keys = [(p["jersey"], p["team"]) for p in players]
    player_meta = {(p["jersey"], p["team"]): p for p in players}

    # Load phase boundaries from Parquet TF15 metadata
    phases = _load_phases_from_parquet(s3fs, parquet_path)
    print(f"  [{match_id}] {len(players)} players, {len(phases)} phases")

    for phase_info in phases:
        label = phase_info["label"]
        if label in completed_phases:
            print(f"  [{match_id}] Phase {phase_info['section']} ({label}): skipped (checkpoint)")
            continue

        t_phase = time.time()
        print(
            f"  [{match_id}] Phase {phase_info['section']} ({label}): "
            f"frames {phase_info['start_frame']:,} – {phase_info['end_frame']:,} ... ",
            end="", flush=True,
        )
        angles_by_player = stream_player_angles(
            s3fs, parquet_path,
            phase_info["start_frame"], phase_info["end_frame"],
            player_keys,
        )
        results = _compute_phase_awi_from_angles(angles_by_player, player_meta, phase_info, match_id)
        del angles_by_player
        print(f"{time.time() - t_phase:.0f}s")

        phase_frame = pd.DataFrame(results)
        frames.append(phase_frame)

        # Save checkpoint after each phase
        if checkpoint_path:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(checkpoint_path, index=False)
            print(f"  [{match_id}] Phase {label} done - {len(phase_frame)} rows saved to checkpoint.")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Multi-match pipeline ─────────────────────────────────────────────────────

def run_all_matches(
    s3_client,
    s3fs: pafs.S3FileSystem,
    bucket: str,
    challenge_prefix: str,
    match_configs: list[dict] | None = None,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """Compute AWI for all matches and return a combined DataFrame.

    Skips any match with "TODO" in its parquet_key (unverified path).
    Errors in individual matches are caught and logged; other matches continue.

    If checkpoint_path points to an existing CSV, already-completed match_ids
    are skipped and their rows are loaded from disk. New results are appended
    to the checkpoint file after each match so partial progress survives
    token expiry.

    Args:
        s3_client:        boto3 S3 client.
        s3fs:             pyarrow S3FileSystem.
        bucket:           S3 bucket name.
        challenge_prefix: Challenge data prefix in S3 (no trailing slash).
        match_configs:    Defaults to module-level MATCH_CONFIGS.
        checkpoint_path:  Path to CSV used for incremental saves (optional).

    Returns:
        DataFrame of all AWI results, or empty DataFrame if all matches fail.
    """
    import os

    if match_configs is None:
        match_configs = MATCH_CONFIGS

    # Load already-completed matches from checkpoint
    completed_ids: set[str] = set()
    frames: list[pd.DataFrame] = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        if not existing.empty and "match_id" in existing.columns:
            completed_ids = set(existing["match_id"].unique())
            frames.append(existing)
            print(f"[checkpoint] Loaded {len(existing)} rows for: {sorted(completed_ids)}")

    for mc in match_configs:
        if "TODO" in mc.get("parquet_key", ""):
            print(f"\n[SKIP] {mc['match_id']}: path not configured yet.")
            continue

        if mc["match_id"] in completed_ids:
            print(f"\n[SKIP] {mc['match_id']}: already in checkpoint, skipping.")
            continue

        print(f"\n=== {mc['match_id']}: {mc.get('label', '')} ===")
        try:
            df = run_match_awi(s3_client, s3fs, bucket, challenge_prefix, mc,
                               checkpoint_path=checkpoint_path)
            frames.append(df)
            print(f"  Done: {len(df)} player-phase rows")
            if checkpoint_path:
                combined_so_far = pd.concat(frames, ignore_index=True)
                combined_so_far.to_csv(checkpoint_path, index=False)
                print(f"  [checkpoint] Saved {len(combined_so_far)} rows to {checkpoint_path}")
        except Exception as e:
            print(f"  [ERROR] {mc['match_id']}: {e}")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ── Unified pipeline (AWI + enrichment in one S3 read per phase) ─────────────

def run_match_unified(
    s3_client,
    s3fs: pafs.S3FileSystem,
    bucket: str,
    challenge_prefix: str,
    match_config: dict,
    events_xml_key: str | None = None,
    checkpoint_path: str | None = None,
    moments_path: str | None = None,
    framerate: int = 50,
) -> pd.DataFrame:
    """Full AWI + enrichment pipeline for one match in a single S3 read per phase.

    Each phase's Parquet data is read exactly once. In the same pass:
      - AWI (scan_count, awi_per_minute, coverage_pct)
      - HBD (head-body decoupling: hbd_mean_deg, hbd_n_frames)
      - Pre-pass AWI (pre_pass_awi, pre_pass_scan_pct, n_passes_tracked)
      - Scan direction profile (scan_forward_pct, scan_lateral_pct, scan_blindside_pct)

    Pre-pass AWI and scan direction require events_xml_key (Events XML relative to
    challenge_prefix). If omitted or if the XML fails to load, those columns are
    written as None so the AWI + HBD columns are never blocked.

    Per-pass scan data (for match moments) is written to moments_path if provided.

    Args:
        s3_client:        boto3 S3 client.
        s3fs:             pyarrow S3FileSystem.
        bucket:           S3 bucket name.
        challenge_prefix: S3 prefix for the challenge data (no trailing slash).
        match_config:     One entry from MATCH_CONFIGS.
        events_xml_key:   Events XML key relative to challenge_prefix (optional).
        checkpoint_path:  Path to CSV for phase-level checkpointing (optional).
        moments_path:     Path to CSV for per-pass match-moment data (optional).
        framerate:        Frames per second (default 50).

    Returns:
        Wide DataFrame: one row per player × phase, all metrics in one table.
    """
    import os
    from src.pre_pass_awi import (
        extract_kickoff_times, build_phase_frame_map,
        pass_events_to_frames, compute_pre_pass_awi,
        compute_pre_pass_awi_per_pass, _SECTION_LABEL_MAP,
    )
    from src.event_parser import extract_pass_events

    match_id     = match_config["match_id"]
    parquet_path = f"{bucket}/{challenge_prefix}/{match_config['parquet_key']}"
    xml_key      = f"{challenge_prefix}/{match_config['match_info_key']}"

    # Load completed phases from checkpoint (this match only).
    # NOTE: checkpoint file writing is handled by run_all_matches_unified.
    # This function reads the checkpoint to skip already-completed phases
    # and includes those rows in its return value for a complete match DataFrame.
    completed_phases: set[str] = set()
    frames: list[pd.DataFrame] = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        match_rows = (
            existing[existing["match_id"] == match_id]
            if "match_id" in existing.columns else pd.DataFrame()
        )
        if not match_rows.empty and "phase_label" in match_rows.columns:
            completed_phases = set(match_rows["phase_label"].unique())
            frames.append(match_rows)
            print(f"  [{match_id}] Checkpoint: skipping phases {sorted(completed_phases)}")

    # Player roster
    match_info_root = load_xml(s3_client, bucket, xml_key)
    if match_info_root is None:
        raise RuntimeError(
            f"[{match_id}] Failed to load MatchInformation XML: s3://{bucket}/{xml_key}"
        )
    players = [
        p for p in extract_players_from_match_info(match_info_root)
        if p["team"] in (0, 1) and p["jersey"] >= 0
    ]
    if not players:
        raise RuntimeError(f"[{match_id}] No valid players in MatchInformation XML.")

    player_keys = [(p["jersey"], p["team"]) for p in players]
    player_meta = {(p["jersey"], p["team"]): p for p in players}

    phases = _load_phases_from_parquet(s3fs, parquet_path)
    print(f"  [{match_id}] {len(players)} players, {len(phases)} phases")

    # Load event data (optional - enables pre-pass AWI + scan direction + moments)
    has_events = False
    pass_df = pd.DataFrame()
    if events_xml_key:
        events_full_key = f"{challenge_prefix}/{events_xml_key}"
        events_root = load_xml(s3_client, bucket, events_full_key)
        if events_root is not None:
            pass_df_raw = extract_pass_events(events_root)
            kickoff_times = extract_kickoff_times(events_root)
            phase_frame_map = build_phase_frame_map(phases)
            pass_df = pass_events_to_frames(pass_df_raw, kickoff_times, phase_frame_map, framerate)
            has_events = True
            print(f"  [{match_id}] Events XML loaded: {len(pass_df)} passes")
        else:
            print(f"  [{match_id}] Events XML not found - skipping pre-pass AWI / moments.")

    all_moments: list[dict] = []

    for phase_info in phases:
        label = phase_info["label"]
        if label in completed_phases:
            print(f"  [{match_id}] {label}: skipped (checkpoint)")
            continue

        t_phase = time.time()
        print(
            f"  [{match_id}] {label}: frames "
            f"{phase_info['start_frame']:,} – {phase_info['end_frame']:,} ...",
            end=" ", flush=True,
        )

        angles_by_player = stream_player_angles(
            s3fs, parquet_path,
            phase_info["start_frame"], phase_info["end_frame"],
            player_keys,
        )

        # AWI (head only)
        awi_rows = _compute_phase_awi_from_angles(angles_by_player, player_meta, phase_info, match_id)

        # Per-player enrichment (head + body yaw)
        dfl_section = next(
            (k for k, v in _SECTION_LABEL_MAP.items() if v == label), None
        )
        enrich_map: dict[tuple[int, int], dict] = {}

        for key, angles_df in angles_by_player.items():
            if angles_df.empty:
                continue

            scan_df = detect_scans(angles_df[["frame_number", "head_yaw_deg"]])
            hbd = compute_hbd(angles_df)
            direction = compute_scan_direction_profile(scan_df, angles_df)

            pre_pass: dict = {
                "pre_pass_awi": None, "pre_pass_scan_pct": None, "n_passes_tracked": 0
            }
            if has_events and dfl_section:
                player_id = player_meta[key].get("player_id", "")
                player_passes = pass_df.loc[
                    (pass_df["player_id"] == player_id) &
                    (pass_df["game_section"] == dfl_section),
                    "frame_number",
                ]
                pre_pass = compute_pre_pass_awi(scan_df, player_passes, framerate=framerate)

                moment_rows = compute_pre_pass_awi_per_pass(
                    scan_df, player_passes,
                    jersey=key[0], team=key[1],
                    name=player_meta[key]["name"],
                    match_id=match_id,
                    phase_label=label,
                    phase_start=phase_info["start_frame"],
                    framerate=framerate,
                )
                all_moments.extend(moment_rows)

            enrich_map[key] = {**hbd, **direction, **pre_pass}

        # Merge AWI + enrichment into wide rows
        wide_rows = [
            {**row, **enrich_map.get((row["jersey"], row["team"]), _empty_enrich())}
            for row in awi_rows
        ]
        phase_frame = pd.DataFrame(wide_rows)
        frames.append(phase_frame)

        del angles_by_player
        print(f"{time.time() - t_phase:.0f}s")

    # Save top-5 moments per match to moments CSV
    if all_moments and moments_path:
        match_moments = (
            pd.DataFrame(all_moments)
            .nlargest(5, "pre_pass_scan_count")
            .assign(match_id=match_id)
        )
        # Remove existing rows for this match before appending (idempotent on re-run)
        if os.path.exists(moments_path):
            existing_moments = pd.read_csv(moments_path)
            existing_moments = existing_moments[existing_moments["match_id"] != match_id]
            existing_moments.to_csv(moments_path, index=False)
            match_moments.to_csv(moments_path, mode="a", index=False, header=False)
        else:
            match_moments.to_csv(moments_path, index=False)
        print(f"  [{match_id}] Top-5 moments saved to {moments_path}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_all_matches_unified(
    s3_client,
    s3fs: pafs.S3FileSystem,
    bucket: str,
    challenge_prefix: str,
    match_configs: list[dict] | None = None,
    events_xml_keys: dict[str, str] | None = None,
    checkpoint_path: str | None = None,
    moments_path: str | None = None,
) -> pd.DataFrame:
    """Unified AWI + enrichment pipeline for all matches. Each phase is read once.

    Drop-in replacement for running run_all_matches() + the enrichment loop
    separately. Produces a single wide CSV with AWI, HBD, pre-pass AWI, and
    scan direction columns per player × phase.

    Args:
        s3_client:        boto3 S3 client.
        s3fs:             pyarrow S3FileSystem.
        bucket:           S3 bucket name.
        challenge_prefix: Challenge data prefix in S3 (no trailing slash).
        match_configs:    Defaults to module-level MATCH_CONFIGS.
        events_xml_keys:  Dict mapping match_id -> Events XML key relative to
                          challenge_prefix. Matches not in this dict skip
                          pre-pass AWI and scan direction (AWI + HBD still run).
        checkpoint_path:  CSV path for incremental saves (optional).
        moments_path:     CSV path for per-pass match-moment data (optional).

    Returns:
        Wide DataFrame with all metrics, or empty DataFrame if all matches fail.
    """
    import os

    if match_configs is None:
        match_configs = MATCH_CONFIGS
    if events_xml_keys is None:
        events_xml_keys = {}

    # Enrichment columns produced by the unified pipeline (run_match_unified).
    # If the checkpoint was written by an older AWI-only run, these will be
    # missing and the checkpoint must be treated as stale so that matches are
    # re-processed with the full column set.
    _ENRICHMENT_COLS = {"pre_pass_awi", "hbd_mean_deg", "scan_blindside_pct"}
    _EXPECTED_PHASES_PER_MATCH = 2  # 1st half + 2nd half

    completed_ids: set[str] = set()
    partial_ids: set[str] = set()
    frames: list[pd.DataFrame] = []
    if checkpoint_path and os.path.exists(checkpoint_path):
        existing = pd.read_csv(checkpoint_path)
        if not existing.empty and "match_id" in existing.columns:
            if _ENRICHMENT_COLS.issubset(existing.columns):
                phase_counts = existing.groupby("match_id")["phase_label"].nunique()
                for mid, n_phases in phase_counts.items():
                    if n_phases >= _EXPECTED_PHASES_PER_MATCH:
                        completed_ids.add(mid)
                    else:
                        partial_ids.add(mid)
                # Only keep rows from fully completed matches in frames.
                # Partial matches will be re-processed by run_match_unified
                # which reads the checkpoint itself to skip completed phases.
                complete_rows = existing[existing["match_id"].isin(completed_ids)]
                if not complete_rows.empty:
                    frames.append(complete_rows)
                print(f"[checkpoint] Loaded {len(complete_rows)} rows for: {sorted(completed_ids)}")
                if partial_ids:
                    print(f"[checkpoint] Incomplete matches (will resume): {sorted(partial_ids)}")
            else:
                missing = sorted(_ENRICHMENT_COLS - set(existing.columns))
                print(
                    f"[checkpoint] Stale checkpoint detected (missing columns: {missing}). "
                    f"Deleting {checkpoint_path} and re-running all matches."
                )
                os.remove(checkpoint_path)

    for mc in match_configs:
        if "TODO" in mc.get("parquet_key", ""):
            print(f"\n[SKIP] {mc['match_id']}: path not configured.")
            continue
        if mc["match_id"] in completed_ids:
            print(f"\n[SKIP] {mc['match_id']}: already in checkpoint.")
            continue

        print(f"\n=== {mc['match_id']}: {mc.get('label', '')} ===")
        try:
            df = run_match_unified(
                s3_client, s3fs, bucket, challenge_prefix, mc,
                events_xml_key=events_xml_keys.get(mc["match_id"]),
                checkpoint_path=checkpoint_path,  # read-only: skips completed phases
                moments_path=moments_path,
            )
            if not df.empty:
                frames.append(df)
                completed_ids.add(mc["match_id"])
                partial_ids.discard(mc["match_id"])
                print(f"  Done: {len(df)} player-phase rows")
                if checkpoint_path:
                    pd.concat(frames, ignore_index=True).to_csv(checkpoint_path, index=False)
            else:
                print(f"  [WARN] {mc['match_id']}: returned 0 rows")
        except Exception as e:
            print(f"  [ERROR] {mc['match_id']}: {e}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
