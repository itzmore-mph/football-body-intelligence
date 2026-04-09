"""
pressure_pipeline.py
Orchestrates PQI computation across all matches and phases.
Mirrors batch_pipeline.py structure.
"""

import time
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.fs as pafs
import pandas as pd

from src.batch_pipeline import MATCH_CONFIGS, _load_phases_from_parquet  # noqa: F401
from src.pqi_calculator import (
    compute_orientation_score,
    compute_stance_score,
    compute_proximity_score,
    compute_pqi,
    compute_knee_flexion,
    aggregate_pqi_for_player,
)
from src.event_parser import extract_players_from_match_info
from src.eda_helpers import load_xml

# Schema for results/pqi_full.csv — mirrors awi_full.csv structure
PQI_SCHEMA = {
    # Identity (join keys — same as awi_full.csv)
    "jersey":           int,
    "team":             int,
    "name":             str,
    "position":         str,
    "match_id":         str,
    "phase_label":      str,
    "phase_start":      int,
    "phase_end":        int,
    # PQI aggregate scores
    "mean_pqi":         float,
    "median_pqi":       float,
    "std_pqi":          float,
    "n_press_frames":   int,
    "press_minutes":    float,
    # Sub-score means
    "orientation_mean": float,
    "stance_mean":      float,
    "proximity_mean":   float,
    # Data quality
    "coverage_pct":     float,
}

# Part IDs (TF15 Parquet v1.1)
PART_LEFT_SHOULDER = 4
PART_RIGHT_SHOULDER = 6
PART_LEFT_HIP = 11
PART_PELVIS = 12
PART_RIGHT_HIP = 13
PART_LEFT_KNEE = 14
PART_RIGHT_KNEE = 15
PART_LEFT_ANKLE = 16
PART_RIGHT_ANKLE = 17

_PRESS_PART_IDS = {
    PART_LEFT_SHOULDER,
    PART_RIGHT_SHOULDER,
    PART_LEFT_HIP,
    PART_PELVIS,
    PART_RIGHT_HIP,
    PART_LEFT_KNEE,
    PART_RIGHT_KNEE,
    PART_LEFT_ANKLE,
    PART_RIGHT_ANKLE,
}

_PRESS_OUTPUT_COLUMNS = [
    "frame_number",
    "body_yaw_deg",
    "pelvis_x",
    "pelvis_y",
    "lknee_x",
    "lknee_y",
    "rknee_x",
    "rknee_y",
    "lhip_x",
    "lhip_y",
    "rhip_x",
    "rhip_y",
    "lankle_x",
    "lankle_y",
    "rankle_x",
    "rankle_y",
    "ball_x_m",
    "ball_y_m",
]


def extract_press_angles_vectorized(
    table: pa.Table,
    player_keys: list[tuple[int, int]],
) -> dict[tuple[int, int], pd.DataFrame]:
    """Extract body yaw and joint positions for pressing analysis.

    Extends the _extract_angles_vectorized pattern from batch_pipeline.py to
    also extract pelvis, knee, hip, and ankle positions, plus ball position.

    All joint position_x/y values are returned in raw parquet units (cm).
    Ball positions are converted to metres (cm / 100).

    Args:
        table:       PyArrow Table with columns [frame_number, skeletons, ball, ball_exists].
        player_keys: (jersey_number, team) pairs to extract.

    Returns:
        Dict mapping (jersey, team) -> DataFrame with columns:
        [frame_number, body_yaw_deg, pelvis_x, pelvis_y,
         lknee_x, lknee_y, rknee_x, rknee_y,
         lhip_x, lhip_y, rhip_x, rhip_y,
         lankle_x, lankle_y, rankle_x, rankle_y,
         ball_x_m, ball_y_m]
        Missing joints produce NaN. No Python loops over frames.
    """
    empty = {
        k: pd.DataFrame(columns=_PRESS_OUTPUT_COLUMNS)
        for k in player_keys
    }
    player_set = set(player_keys)

    # ── Extract ball position (per-frame, not per-skeleton) ──────────────────
    ball_col = table.column("ball")
    # combine_chunks() required: table.column() always returns ChunkedArray,
    # but .field() only exists on StructArray (the underlying type after combining).
    if isinstance(ball_col, pa.ChunkedArray):
        ball_col = ball_col.combine_chunks()
    ball_exists = table.column("ball_exists").to_numpy(zero_copy_only=False).astype(bool)
    ball_x_cm = ball_col.field("position_x").to_numpy(zero_copy_only=False)
    ball_y_cm = ball_col.field("position_y").to_numpy(zero_copy_only=False)
    ball_x_m = np.where(ball_exists, ball_x_cm / 100.0, np.nan)
    ball_y_m = np.where(ball_exists, ball_y_cm / 100.0, np.nan)

    # ── Flatten skeletons ────────────────────────────────────────────────────
    skel_col = table.column("skeletons")
    if isinstance(skel_col, pa.ChunkedArray):
        skel_col = skel_col.combine_chunks()

    skel_lengths = pc.list_value_length(skel_col).fill_null(0)
    skel_flat = pc.list_flatten(skel_col)
    if len(skel_flat) == 0:
        return empty

    # Vectorized jersey/team extraction
    jersey_np = skel_flat.field("jersey_number").to_numpy(zero_copy_only=False)
    team_np = skel_flat.field("team").to_numpy(zero_copy_only=False)

    # Repeat frame_number and ball coords to align with flattened skeletons
    frame_np = table.column("frame_number").to_numpy(zero_copy_only=False)
    skel_len_np = skel_lengths.to_numpy(zero_copy_only=False)
    frame_repeated = np.repeat(frame_np, skel_len_np)
    ball_x_repeated = np.repeat(ball_x_m, skel_len_np)
    ball_y_repeated = np.repeat(ball_y_m, skel_len_np)

    # Boolean mask for target players
    player_mask = np.zeros(len(jersey_np), dtype=bool)
    for j, t in player_set:
        player_mask |= (jersey_np == j) & (team_np == t)

    filtered_idx = np.where(player_mask)[0]
    if len(filtered_idx) == 0:
        return empty

    jersey_f = jersey_np[filtered_idx]
    team_f = team_np[filtered_idx]
    frame_f = frame_repeated[filtered_idx]
    ball_x_f = ball_x_repeated[filtered_idx]
    ball_y_f = ball_y_repeated[filtered_idx]
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
    part_x_np = parts_flat.field("position_x").to_numpy(zero_copy_only=False).astype(np.float64)
    part_y_np = parts_flat.field("position_y").to_numpy(zero_copy_only=False).astype(np.float64)

    # Helper: fill position arrays for a given part ID
    def _fill(part_id: int) -> tuple[np.ndarray, np.ndarray]:
        px = np.full(n, np.nan)
        py = np.full(n, np.nan)
        m = part_name_np == part_id
        if m.any():
            px[skel_idx_for_part[m]] = part_x_np[m]
            py[skel_idx_for_part[m]] = part_y_np[m]
        return px, py

    lsho_x, lsho_y = _fill(PART_LEFT_SHOULDER)
    rsho_x, rsho_y = _fill(PART_RIGHT_SHOULDER)
    lhip_x, lhip_y = _fill(PART_LEFT_HIP)
    rhip_x, rhip_y = _fill(PART_RIGHT_HIP)
    pelvis_x, pelvis_y = _fill(PART_PELVIS)
    lknee_x, lknee_y = _fill(PART_LEFT_KNEE)
    rknee_x, rknee_y = _fill(PART_RIGHT_KNEE)
    lankle_x, lankle_y = _fill(PART_LEFT_ANKLE)
    rankle_x, rankle_y = _fill(PART_RIGHT_ANKLE)

    # ── Body yaw: shoulder vector primary, hip fallback ──────────────────────
    dx_sh = lsho_x - rsho_x
    dy_sh = lsho_y - rsho_y
    valid_sh = ~(np.isnan(dx_sh) | np.isnan(dy_sh)) & ~((dx_sh == 0.0) & (dy_sh == 0.0))

    body_yaw = np.full(n, np.nan)
    body_yaw[valid_sh] = np.degrees(np.arctan2(-dx_sh[valid_sh], dy_sh[valid_sh]))

    dx_hip = lhip_x - rhip_x
    dy_hip = lhip_y - rhip_y
    needs_hip = np.isnan(body_yaw)
    if needs_hip.any():
        valid_hip = (
            needs_hip
            & ~(np.isnan(dx_hip) | np.isnan(dy_hip))
            & ~((dx_hip == 0.0) & (dy_hip == 0.0))
        )
        if valid_hip.any():
            body_yaw[valid_hip] = np.degrees(np.arctan2(-dx_hip[valid_hip], dy_hip[valid_hip]))

    # ── Build per-player DataFrames ──────────────────────────────────────────
    combined = pd.DataFrame(
        {
            "jersey": jersey_f.astype(int),
            "team": team_f.astype(int),
            "frame_number": frame_f.astype(int),
            "body_yaw_deg": body_yaw,
            "pelvis_x": pelvis_x,
            "pelvis_y": pelvis_y,
            "lknee_x": lknee_x,
            "lknee_y": lknee_y,
            "rknee_x": rknee_x,
            "rknee_y": rknee_y,
            "lhip_x": lhip_x,
            "lhip_y": lhip_y,
            "rhip_x": rhip_x,
            "rhip_y": rhip_y,
            "lankle_x": lankle_x,
            "lankle_y": lankle_y,
            "rankle_x": rankle_x,
            "rankle_y": rankle_y,
            "ball_x_m": ball_x_f,
            "ball_y_m": ball_y_f,
        }
    )

    result: dict[tuple[int, int], pd.DataFrame] = {}
    for j, t in player_set:
        sub = combined.loc[
            (combined["jersey"] == j) & (combined["team"] == t),
            _PRESS_OUTPUT_COLUMNS,
        ]
        result[(j, t)] = sub.reset_index(drop=True) if not sub.empty else pd.DataFrame(
            columns=_PRESS_OUTPUT_COLUMNS
        )
    return result


def identify_press_frames(
    presser_df: pd.DataFrame,
    carrier_df: pd.DataFrame,
    min_run_length: int = 10,
) -> pd.Series:
    """Identify frames where the presser is within 5 m of the ball carrier
    for at least `min_run_length` consecutive frames.

    Args:
        presser_df:     DataFrame with columns [frame_number, pelvis_x, pelvis_y]
        carrier_df:     DataFrame with columns [frame_number, pelvis_x, pelvis_y]
        min_run_length: minimum consecutive frames to count as a press (default 10)

    Returns:
        Boolean Series indexed by frame_number.
    """
    merged = presser_df[["frame_number", "pelvis_x", "pelvis_y"]].merge(
        carrier_df[["frame_number", "pelvis_x", "pelvis_y"]],
        on="frame_number",
        suffixes=("_presser", "_carrier"),
    )

    # Pelvis positions are in cm — divide by 100 to get metres
    dx = merged["pelvis_x_presser"] - merged["pelvis_x_carrier"]
    dy = merged["pelvis_y_presser"] - merged["pelvis_y_carrier"]
    distance_m = np.sqrt(dx ** 2 + dy ** 2) / 100.0
    merged["is_close"] = distance_m <= 5.0

    # Minimum duration filter using cumsum grouping trick (no Python loops)
    is_close = merged["is_close"]
    run_id = (is_close != is_close.shift()).cumsum()
    run_lengths = is_close.groupby(run_id).transform("sum")
    merged["is_press"] = is_close & (run_lengths >= min_run_length)

    return merged.set_index("frame_number")["is_press"]


def compute_phase_pqi_all_players(
    angles_by_player: dict[tuple[int, int], pd.DataFrame],
    player_meta: dict[tuple[int, int], dict],
    phase_info: dict,
    match_id: str,
) -> list[dict]:
    """Compute PQI for all players in one phase from pre-extracted angle DataFrames.

    For each player: identifies press frames (based on proximity to ball),
    computes PQI sub-scores, aggregates to a phase-level summary dict.

    Args:
        angles_by_player: (jersey, team) -> DataFrame with columns from
                          extract_press_angles_vectorized.
        player_meta:      (jersey, team) -> dict with keys: name, position.
        phase_info:       dict with keys: phase_label, phase_start, phase_end.
        match_id:         Match identifier string (e.g. "FCB-HSV").

    Returns:
        List of result dicts (one per player), each matching PQI_SCHEMA.
    """
    phase_label = phase_info["phase_label"]
    phase_start = phase_info["phase_start"]
    phase_end = phase_info["phase_end"]
    total_phase_frames = max(1, phase_end - phase_start + 1)

    results = []

    for (jersey, team), player_df in angles_by_player.items():
        meta = player_meta.get((jersey, team), {})
        name = meta.get("name", "")
        position = meta.get("position", "")

        coverage_pct = len(player_df) / total_phase_frames

        # Build a synthetic "ball carrier" DataFrame using ball position
        # (pelvis positions are in cm; convert back to cm for identify_press_frames)
        if player_df.empty or player_df["ball_x_m"].isna().all():
            # No valid data — return zero-press result
            agg = aggregate_pqi_for_player(
                pd.Series([], dtype=float),
                pd.Series([], dtype=float),
                pd.Series([], dtype=float),
                pd.Series([], dtype=float),
                phase_start,
                phase_end,
            )
            results.append({
                "jersey": jersey,
                "team": team,
                "name": name,
                "position": position,
                "match_id": match_id,
                "phase_label": phase_label,
                "phase_start": phase_start,
                "phase_end": phase_end,
                **agg,
                "press_minutes": 0.0,
                "coverage_pct": coverage_pct,
            })
            continue

        # Synthetic ball-carrier DataFrame: ball position in cm (multiply m back by 100)
        ball_carrier_df = pd.DataFrame({
            "frame_number": player_df["frame_number"],
            "pelvis_x": player_df["ball_x_m"] * 100.0,
            "pelvis_y": player_df["ball_y_m"] * 100.0,
        })

        # Identify press frames (proximity to ball >= 10 consecutive frames)
        press_mask = identify_press_frames(player_df, ball_carrier_df)

        # Filter player_df to press frames only
        press_frames = press_mask[press_mask].index
        press_df = player_df[player_df["frame_number"].isin(press_frames)].copy()

        if press_df.empty:
            agg = aggregate_pqi_for_player(
                pd.Series([], dtype=float),
                pd.Series([], dtype=float),
                pd.Series([], dtype=float),
                pd.Series([], dtype=float),
                phase_start,
                phase_end,
            )
            results.append({
                "jersey": jersey,
                "team": team,
                "name": name,
                "position": position,
                "match_id": match_id,
                "phase_label": phase_label,
                "phase_start": phase_start,
                "phase_end": phase_end,
                **agg,
                "press_minutes": 0.0,
                "coverage_pct": coverage_pct,
            })
            continue

        # Compute knee flexion (left and right), average per frame
        left_flex = compute_knee_flexion(
            press_df["lknee_x"].values, press_df["lknee_y"].values,
            press_df["lhip_x"].values,  press_df["lhip_y"].values,
            press_df["lankle_x"].values, press_df["lankle_y"].values,
        )
        right_flex = compute_knee_flexion(
            press_df["rknee_x"].values, press_df["rknee_y"].values,
            press_df["rhip_x"].values,  press_df["rhip_y"].values,
            press_df["rankle_x"].values, press_df["rankle_y"].values,
        )
        # Average left and right per frame (nanmean semantics)
        knee_flex = np.nanmean(np.stack([left_flex, right_flex], axis=1), axis=1)

        # Ball direction from player pelvis (pelvis in cm → divide by 100 for metres)
        ball_dir = np.degrees(np.arctan2(
            press_df["ball_y_m"].values - press_df["pelvis_y"].values / 100.0,
            press_df["ball_x_m"].values - press_df["pelvis_x"].values / 100.0,
        ))

        # Sub-scores
        orientation_score = compute_orientation_score(
            press_df["body_yaw_deg"].values, ball_dir
        )
        stance_score = compute_stance_score(knee_flex)

        distance_m = np.sqrt(
            (press_df["pelvis_x"].values / 100.0 - press_df["ball_x_m"].values) ** 2
            + (press_df["pelvis_y"].values / 100.0 - press_df["ball_y_m"].values) ** 2
        )
        proximity_score = compute_proximity_score(distance_m)

        pqi = compute_pqi(orientation_score, stance_score, proximity_score)

        # Aggregate
        agg = aggregate_pqi_for_player(
            pd.Series(pqi),
            pd.Series(orientation_score),
            pd.Series(stance_score),
            pd.Series(proximity_score),
            phase_start,
            phase_end,
        )

        results.append({
            "jersey": jersey,
            "team": team,
            "name": name,
            "position": position,
            "match_id": match_id,
            "phase_label": phase_label,
            "phase_start": phase_start,
            "phase_end": phase_end,
            **agg,
            "press_minutes": agg["n_press_frames"] / 50 / 60,
            "coverage_pct": coverage_pct,
        })

    return results


# ── Match-level orchestration ────────────────────────────────────────────────

def run_match_pqi(
    s3_client,
    s3fs,
    bucket: str,
    challenge_prefix: str,
    match_config: dict,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """Full PQI pipeline for one match: all players × all phases.

    Args:
        s3_client:        boto3 S3 client.
        s3fs:             pyarrow S3FileSystem.
        bucket:           S3 bucket name.
        challenge_prefix: S3 prefix for the challenge data (no trailing slash).
        match_config:     One entry from MATCH_CONFIGS.
        checkpoint_path:  Optional path to CSV for phase-level checkpointing.

    Returns:
        DataFrame with PQI results for all players × all phases.
    """
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

    # Load player roster from MatchInformation XML
    match_info_root = load_xml(s3_client, bucket, xml_key)
    if match_info_root is None:
        raise RuntimeError(
            f"[{match_id}] Failed to load MatchInformation XML: s3://{bucket}/{xml_key}"
        )
    players = extract_players_from_match_info(match_info_root)
    # Filter: team in (0, 1) and jersey >= 0
    players = [p for p in players if p["team"] in (0, 1) and p["jersey"] >= 0]

    if not players:
        raise RuntimeError(f"[{match_id}] No valid players found in MatchInformation XML.")

    player_keys = [(p["jersey"], p["team"]) for p in players]
    player_meta = {(p["jersey"], p["team"]): p for p in players}

    # Load phase boundaries from Parquet TF15 metadata
    phases = _load_phases_from_parquet(s3fs, parquet_path)
    print(f"  [{match_id}] {len(players)} players, {len(phases)} phases")

    for phase_info in phases:
        label       = phase_info["label"]
        phase_start = phase_info["start_frame"]
        phase_end   = phase_info["end_frame"]

        if label in completed_phases:
            print(f"  [{match_id}] Phase {label}: skipped (checkpoint)")
            continue

        t_phase = time.time()
        print(
            f"  [{match_id}] Phase {label}: "
            f"frames {phase_start:,} – {phase_end:,} ... ",
            end="", flush=True,
        )

        # Read parquet with retry logic
        last_exc: Exception | None = None
        table = None
        for attempt in range(3):
            try:
                table = pq.read_table(
                    parquet_path,
                    filesystem=s3fs,
                    columns=["frame_number", "skeletons", "ball", "ball_exists"],
                    filters=[
                        ("frame_number", ">=", phase_start),
                        ("frame_number", "<=", phase_end),
                    ],
                )
                break
            except Exception as e:
                err_str = str(e)
                if "ExpiredToken" in err_str:
                    print(f"\n    [token expired] {e}. Refresh token and re-run.")
                    raise
                last_exc = e
                wait = 15.0 * (2 ** attempt)
                print(f"\n    [retry {attempt + 1}/3] {e}. Waiting {wait:.0f}s...")
                time.sleep(wait)

        if table is None:
            raise last_exc  # type: ignore[misc]

        # Map phase_info keys to the dict expected by compute_phase_pqi_all_players
        phase_info_dict = {
            "phase_label": label,
            "phase_start": phase_start,
            "phase_end":   phase_end,
        }

        angles_by_player = extract_press_angles_vectorized(table, player_keys)
        del table

        results = compute_phase_pqi_all_players(
            angles_by_player, player_meta, phase_info_dict, match_id
        )
        print(f"{time.time() - t_phase:.0f}s")

        phase_frame = pd.DataFrame(results)
        frames.append(phase_frame)

        # Save checkpoint after each phase
        if checkpoint_path:
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(checkpoint_path, index=False)
            print(f"  [{match_id}] Phase {label} done — {len(phase_frame)} rows saved to checkpoint.")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Multi-match pipeline ─────────────────────────────────────────────────────

def run_all_matches_pqi(
    s3_client,
    s3fs,
    bucket: str,
    challenge_prefix: str,
    match_configs: list[dict] | None = None,
    checkpoint_path: str | None = None,
) -> pd.DataFrame:
    """Compute PQI for all matches and return a combined DataFrame.

    Skips any match with "TODO" in its parquet_key (unverified path).
    Errors in individual matches are caught and logged; other matches continue.

    Args:
        s3_client:        boto3 S3 client.
        s3fs:             pyarrow S3FileSystem.
        bucket:           S3 bucket name.
        challenge_prefix: Challenge data prefix in S3 (no trailing slash).
        match_configs:    Defaults to module-level MATCH_CONFIGS.
        checkpoint_path:  Path to CSV used for incremental saves (optional).

    Returns:
        DataFrame of all PQI results, or empty DataFrame if all matches fail.
    """
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
            df = run_match_pqi(s3_client, s3fs, bucket, challenge_prefix, mc,
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

    result = pd.concat(frames, ignore_index=True)

    # Save final output
    os.makedirs("results", exist_ok=True)
    result.to_csv("results/pqi_full.csv", index=False)
    print(f"\n[done] Saved {len(result)} rows to results/pqi_full.csv")

    return result
