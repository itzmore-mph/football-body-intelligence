"""
scripts/aggregate_results.py
SageMaker Processing entry point — aggregation step.

Runs after all AWI and PQI match jobs complete (depends_on in Pipeline).
Reads the per-match CSVs from the input directory, concatenates them,
validates row counts, and writes awi_full.csv + pqi_full.csv to the output.

SageMaker mounts the outputs of the upstream Processing steps as inputs here:
    /opt/ml/processing/input/awi/   -> all awi_<MATCH_ID>.csv files
    /opt/ml/processing/input/pqi/   -> all pqi_<MATCH_ID>.csv files
    /opt/ml/processing/output/      -> final awi_full.csv + pqi_full.csv

These mount paths are defined in sagemaker_pipeline.py (ProcessingInput sources).
"""

import os
import sys
import glob

import pandas as pd


# ── I/O paths set by SageMaker Processing ───────────────────────────────────
AWI_INPUT_DIR  = os.environ.get("AWI_INPUT_DIR",  "/opt/ml/processing/input/awi")
PQI_INPUT_DIR  = os.environ.get("PQI_INPUT_DIR",  "/opt/ml/processing/input/pqi")
OUTPUT_DIR     = os.environ.get("SM_OUTPUT_DIR",   "/opt/ml/processing/output")

# Expected matches — used for completeness validation
EXPECTED_MATCH_IDS = {"FCB-HSV", "BVB-VFB", "SGE-FCB", "SGE-FCU", "FCU-FCB"}


def _read_csvs(directory: str, prefix: str) -> pd.DataFrame:
    """Read all CSVs matching prefix_*.csv in directory and concatenate."""
    pattern = os.path.join(directory, f"{prefix}_*.csv")
    files   = sorted(glob.glob(pattern))

    if not files:
        print(f"[AGG] WARNING: No files found matching {pattern}")
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f)
            print(f"  {os.path.basename(f)}: {len(df)} rows")
            frames.append(df)
        except Exception as e:
            print(f"  ERROR reading {f}: {e}")

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _validate(df: pd.DataFrame, name: str, expected_ids: set[str]) -> bool:
    """Basic sanity checks on the aggregated DataFrame. Returns True if clean."""
    ok = True

    if df.empty:
        print(f"[AGG] FAIL {name}: DataFrame is empty.")
        return False

    if "match_id" not in df.columns:
        print(f"[AGG] FAIL {name}: column 'match_id' missing.")
        return False

    found_ids = set(df["match_id"].unique())
    missing   = expected_ids - found_ids
    if missing:
        print(f"[AGG] WARNING {name}: missing matches: {sorted(missing)}")
        ok = False  # warning only — partial results are still saved

    # Check for NaN in key columns
    key_cols = {"jersey", "team", "match_id", "phase_label"} & set(df.columns)
    for col in key_cols:
        n_null = df[col].isna().sum()
        if n_null > 0:
            print(f"[AGG] WARNING {name}: {n_null} null values in '{col}'")

    print(f"[AGG] {name}: {len(df)} total rows, {len(found_ids)} matches")
    return ok


def main() -> None:
    print("[AGG] Starting aggregation step")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── AWI ──────────────────────────────────────────────────────────────────
    print("\n[AGG] Reading AWI results:")
    awi_df = _read_csvs(AWI_INPUT_DIR, "awi")
    _validate(awi_df, "AWI", EXPECTED_MATCH_IDS)

    if not awi_df.empty:
        awi_out = os.path.join(OUTPUT_DIR, "awi_full.csv")
        awi_df.to_csv(awi_out, index=False)
        print(f"[AGG] Saved {len(awi_df)} rows → {awi_out}")
    else:
        print("[AGG] ERROR: AWI aggregation produced no output.")
        sys.exit(1)

    # ── PQI ──────────────────────────────────────────────────────────────────
    print("\n[AGG] Reading PQI results:")
    pqi_df = _read_csvs(PQI_INPUT_DIR, "pqi")
    _validate(pqi_df, "PQI", EXPECTED_MATCH_IDS)

    if not pqi_df.empty:
        pqi_out = os.path.join(OUTPUT_DIR, "pqi_full.csv")
        pqi_df.to_csv(pqi_out, index=False)
        print(f"[AGG] Saved {len(pqi_df)} rows → {pqi_out}")
    else:
        print("[AGG] ERROR: PQI aggregation produced no output.")
        sys.exit(1)

    print("\n[AGG] Aggregation complete.")


if __name__ == "__main__":
    main()
